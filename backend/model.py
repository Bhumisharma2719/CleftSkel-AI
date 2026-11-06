"""Aariz dataset loader, multi-task MobileNetV2 training, and inference.

Aariz supplies CVM stages rather than skeletal ages. Age regression uses direct
age labels when complete labels are supplied, otherwise the checked-in stage
map is used as a clearly marked synthetic proxy target.
"""

from __future__ import annotations

import csv
import copy
import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_DATASET_DIR = Path(os.getenv("CLEFTSKEL_DATASET_DIR", r"C:\Users\Bhumi\Desktop\Aariz"))
MODEL_PATH = Path(os.getenv("CLEFTSKEL_MODEL_PATH", Path(__file__).parent / "artifacts" / "mobilenetv2.keras"))
LABELS_PATH = MODEL_PATH.with_suffix(".labels.json")
DEFAULT_STAGE_AGE_MAP_PATH = Path(__file__).resolve().parent.parent / "stage_age_map.json"
DEFAULT_CVA_MODEL_PATH = Path(os.getenv("CLEFTSKEL_CVA_MODEL_PATH", Path(__file__).parent / "artifacts" / "mobilenetv2_cva.keras"))
CVA_LABELS_PATH = DEFAULT_CVA_MODEL_PATH.with_suffix(".labels.json")
ROI_MODEL_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_roi.keras")
IMAGE_SIZE = (224, 224)
PUBLISHED_BENCHMARKS = {
    "citation": "Kamei et al., DOI 10.1177/10556656241299890",
    "average_accuracy_percent": 74.5,
    "skeletal_maturity_accuracy_max_percent": 88.0,
    "cva_detection_accuracy_percent": 83.0,
    "cvmi_6_accuracy_percent": None,
    "note": "The abstract reports up to 88% for skeletal maturity, not specifically CVMI-6. These are separate-study reference figures and are not targets achieved by this checkpoint.",
}
_loaded_model = None
_loaded_metadata: dict[str, Any] | None = None
_loaded_roi_model = None
_loaded_cva_model = None
_loaded_cva_metadata: dict[str, Any] | None = None
_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_cache_lock = threading.Lock()
_prediction_cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
_model_generation = 0
_PREDICTION_CACHE_LIMIT = int(os.getenv("CLEFTSKEL_PREDICTION_CACHE_SIZE", "64"))
logger = logging.getLogger("cleftskel.model")


def _generate_gradcam_overlay(model, pixels: np.ndarray, source_image: Image.Image, class_index: int) -> dict[str, Any]:
    """Create a Grad-CAM overlay for the selected CVM class on MobileNetV2's last conv map."""
    tf = _tensorflow()
    backbone_index = next(
        (i for i, layer in enumerate(model.layers)
         if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower()),
        None,
    )
    if backbone_index is None:
        raise ValueError("Could not locate the nested MobileNetV2 backbone in the saved model.")
    backbone = model.layers[backbone_index]
    conv_layer = next(
        (layer for layer in reversed(backbone.layers)
         if getattr(getattr(layer, "output", None), "shape", None) is not None
         and len(layer.output.shape) == 4),
        None,
    )
    if conv_layer is None:
        raise ValueError("Could not locate a spatial convolutional feature layer for Grad-CAM.")

    feature_extractor = tf.keras.Model(backbone.inputs, conv_layer.output)
    image_tensor = tf.convert_to_tensor(pixels, dtype=tf.float32)
    with tf.GradientTape() as tape:
        normalized = tf.keras.applications.mobilenet_v2.preprocess_input(image_tensor)
        conv_features = feature_extractor(normalized, training=False)
        head_output = conv_features
        class_scores = None
        for layer in model.layers[backbone_index + 1:]:
            if layer.name == "skeletal_age_years":
                continue
            head_output = layer(head_output, training=False)
            if layer.name == "cvm_stage":
                class_scores = head_output
                break
        if class_scores is None:
            raise ValueError("Could not connect Grad-CAM to the CVM classification head.")
        target_score = class_scores[:, int(class_index)]

    gradients = tape.gradient(target_score, conv_features)
    if gradients is None:
        raise ValueError("Gradients for the CVM target class were unavailable.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights * conv_features, axis=-1))[0]
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap_array = np.nan_to_num(heatmap.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap_gray = Image.fromarray(np.uint8(np.clip(heatmap_array * 255, 0, 255)), mode="L")
    heatmap_gray = heatmap_gray.resize(source_image.size, Image.Resampling.BILINEAR)
    intensity = np.asarray(heatmap_gray, dtype=np.float32) / 255.0
    # Compact jet-like color map generated with NumPy; no OpenCV dependency.
    red = np.clip(1.5 - np.abs(4 * intensity - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * intensity - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * intensity - 1), 0, 1)
    colored = np.uint8(np.stack((red, green, blue), axis=-1) * 255)
    heatmap_rgb = Image.fromarray(colored, mode="RGB")
    source_rgb = source_image.convert("RGB")
    overlay = Image.blend(source_rgb, heatmap_rgb, alpha=0.40)
    output = io.BytesIO()
    overlay.save(output, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "status": "generated",
        "target": "predicted_cvm_stage",
        "layer": conv_layer.name,
        "model_input_size": list(source_image.size),
        "overlay_data_url": f"data:image/jpeg;base64,{encoded}",
        "interpretation_note": "Coarse gradient-based activation map for the predicted CVM class. Highlighted pixels are not proof of causal reasoning, lesion location, or anatomically faithful attention.",
    }


def _experimental_morphology_review(image: Image.Image) -> dict[str, Any]:
    """Return transparent generic image measurements, not a CVA probability or diagnosis."""
    gray = np.asarray(image.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    threshold = float(np.percentile(np.concatenate((grad_x.ravel(), grad_y.ravel())), 75))
    horizontal_edge_density = float(np.mean(grad_y >= threshold))
    vertical_edge_density = float(np.mean(grad_x >= threshold))
    edge_total = horizontal_edge_density + vertical_edge_density
    return {
        "status": "research_measurements_only",
        "heuristic_risk_score": None,
        "measurements": {
            "analysis_crop_width_height_ratio": round(image.width / max(image.height, 1), 4),
            "horizontal_edge_density_percent": round(horizontal_edge_density * 100, 2),
            "vertical_edge_density_percent": round(vertical_edge_density * 100, 2),
            "edge_orientation_ratio": round(horizontal_edge_density / edge_total, 4) if edge_total else None,
            "grayscale_contrast_std": round(float(np.std(gray)), 5),
        },
        "finding": "No lesion-specific CVA finding generated.",
        "limitation": (
            "These global image statistics cannot identify C2-C3 fusion, atlas arch deficiency, spina bifida, "
            "or other CVA. Anatomically localized vertebral landmarks/segmentation and expert-labeled CVA data are required."
        ),
        "disclaimer": "Non-diagnostic research measurements only; not a CVA risk estimate or substitute for clinician image review.",
    }


def _read_age_labels(age_labels_csv: str | Path | None) -> dict[str, float]:
    """Read optional cephalogram_id,skeletal_age_years labels."""
    if not age_labels_csv:
        return {}
    labels: dict[str, float] = {}
    with Path(age_labels_csv).open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_id = row.get("cephalogram_id") or row.get("image_id") or row.get("ceph_id")
            age = row.get("skeletal_age_years") or row.get("age_years")
            if image_id and age:
                labels[image_id.strip()] = float(age)
    return labels


def _read_roi_boxes(roi_boxes_csv: str | Path | None) -> dict[str, tuple[float, float, float, float]]:
    """Read normalized ROI boxes: id,x_min,y_min,x_max,y_max (each 0..1)."""
    if not roi_boxes_csv:
        return {}
    boxes = {}
    with Path(roi_boxes_csv).open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_id = row.get("cephalogram_id") or row.get("image_id") or row.get("ceph_id")
            try:
                box = tuple(float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("ROI CSV must contain x_min,y_min,x_max,y_max normalized columns.") from exc
            if not image_id or len(box) != 4 or any(not np.isfinite(v) or v < 0 or v > 1 for v in box):
                raise ValueError(f"Invalid normalized ROI box for image {image_id!r}.")
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"ROI box must have positive width and height for image {image_id}.")
            boxes[image_id.strip()] = box
    return boxes


def load_aariz_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    age_labels_csv: str | Path | None = None,
    roi_boxes_csv: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load matching image/CVM annotation pairs from Aariz train/valid/test.

    Returns a dict keyed by split. Each record includes `image_path`, numeric
    `cvm_stage` (1..6), its title, and optional `skeletal_age_years`.
    """
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Aariz dataset directory does not exist: {root}")
    age_by_id = _read_age_labels(age_labels_csv)
    roi_by_id = _read_roi_boxes(roi_boxes_csv)
    result: dict[str, list[dict[str, Any]]] = {}

    for split in ("train", "valid", "test"):
        split_dir = root / split
        image_dir = split_dir / "Cephalograms"
        annotation_dir = split_dir / "Annotations" / "CVM Stages"
        if not split_dir.exists():
            continue
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(f"Expected Cephalograms and Annotations/CVM Stages under {split_dir}")

        records = []
        for annotation_path in sorted(annotation_dir.glob("*.json")):
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                stage_info = annotation["cvm_stage"]
                stage_value = int(stage_info["value"])
                stage_title = str(stage_info.get("title", f"CVM-S{stage_value}"))
                image_id = str(annotation.get("ceph_id", annotation_path.stem))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CVM annotation {annotation_path}: {exc}") from exc

            image_path = next((p for p in image_dir.glob(f"{image_id}.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}), None)
            if image_path is None:
                continue
            age_value = annotation.get("skeletal_age_years", annotation.get("age_years"))
            age = float(age_value) if age_value is not None else age_by_id.get(image_id)
            records.append({
                "image_id": image_id,
                "image_path": image_path,
                "cvm_stage": stage_value,
                "cvm_stage_title": stage_title,
                "skeletal_age_years": age,
                "roi_box": roi_by_id.get(image_id),
            })
        result[split] = records

    if not result.get("train"):
        raise ValueError(f"No labeled training images found under {root / 'train'}")
    return result


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install backend/requirements.txt before training or inference.") from exc
    return tf


def _load_stage_age_map(path: str | Path | None = None) -> dict[int, float]:
    """Load and validate explicit CVM-stage age proxies from JSON."""
    mapping_path = Path(path).expanduser() if path else DEFAULT_STAGE_AGE_MAP_PATH
    if not mapping_path.is_file():
        return {}
    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = {int(stage): float(age) for stage, age in raw.items()}
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid stage-to-age JSON mapping at {mapping_path}: {exc}") from exc
    if set(mapping) != {1, 2, 3, 4, 5, 6} or any(not np.isfinite(age) or age <= 0 for age in mapping.values()):
        raise ValueError("Stage-age mapping must include positive finite ages for stages 1 through 6.")
    return mapping


def _model_output_dict(model, raw_output: Any) -> dict[str, np.ndarray]:
    """Normalize Keras dict/list/single outputs using the saved output names."""
    def as_array(value: Any) -> np.ndarray:
        return np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    if isinstance(raw_output, dict):
        return {str(name): as_array(value) for name, value in raw_output.items()}
    names = list(getattr(model, "output_names", []) or [])
    if isinstance(raw_output, (tuple, list)):
        if len(names) != len(raw_output):
            raise ValueError(f"Model returned {len(raw_output)} outputs but declares {len(names)} output names.")
        return {name: as_array(value) for name, value in zip(names, raw_output)}
    name = names[0] if names else "cvm_stage"
    return {name: as_array(raw_output)}


def build_mobilenetv2_model(
    stage_values: list[int], train_age_head: bool = True,
    imagenet_weights: str | None = "imagenet", fine_tune_layers: int = 35,
    focal_alpha: list[float] | None = None, focal_gamma: float = 2.0,
):
    """Build MobileNetV2 with CVM and optional age heads, fine-tuning its tail."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=imagenet_weights
    )
    backbone.trainable = True
    # Keep most ImageNet features fixed and fine-tune only the final feature
    # blocks. Batch-normalization statistics remain frozen for small datasets.
    freeze_before = max(0, len(backbone.layers) - max(0, int(fine_tune_layers)))
    for layer in backbone.layers[:freeze_before]:
        layer.trainable = False
    for layer in backbone.layers[freeze_before:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs: dict[str, Any] = {
        "cvm_stage": tf.keras.layers.Dense(len(stage_values), activation="softmax", name="cvm_stage")(x)
    }
    if train_age_head:
        outputs["skeletal_age_years"] = tf.keras.layers.Dense(1, name="skeletal_age_years")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    alpha_values = focal_alpha or [1.0] * len(stage_values)
    if len(alpha_values) != len(stage_values):
        raise ValueError("focal_alpha must contain one weight for each CVM stage.")
    alpha_tensor = tf.constant(alpha_values, dtype=tf.float32)

    # Return one loss per example so tf.data sample weights can be applied by
    # Keras. Alpha corrects class prevalence; the focal term emphasizes hard
    # examples. All behavior is captured in the trained checkpoint's graph.
    def balanced_sparse_focal_loss(y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        probs = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        rows = tf.range(tf.shape(labels)[0], dtype=tf.int32)
        true_class_prob = tf.gather_nd(probs, tf.stack([rows, labels], axis=1))
        alpha = tf.gather(alpha_tensor, labels)
        return -alpha * tf.pow(1.0 - true_class_prob, focal_gamma) * tf.math.log(true_class_prob)

    balanced_sparse_focal_loss.__name__ = "balanced_sparse_focal_loss"
    losses: dict[str, Any] = {"cvm_stage": balanced_sparse_focal_loss}
    metrics: dict[str, Any] = {"cvm_stage": ["accuracy"]}
    if train_age_head:
        losses["skeletal_age_years"] = "mean_squared_error"
        metrics["skeletal_age_years"] = ["mean_absolute_error"]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, metrics=metrics)
    return model


def build_roi_localizer(imagenet_weights: str | None = "imagenet"):
    """Build a lightweight normalized C2-C4 bounding-box regression model."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=imagenet_weights)
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    boxes = tf.keras.layers.Dense(4, activation="sigmoid", name="roi_box")(x)
    model = tf.keras.Model(inputs, boxes, name="cervical_roi_localizer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="huber", metrics=["mae"])
    return model


def build_cva_classifier(imagenet_weights: str | None = "imagenet"):
    """Build the separate MobileNetV2 binary head (0=normal, 1=CVA present)."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=imagenet_weights)
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="cva_present")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_cva_binary")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def _training_image_augmenter():
    """Construct the requested clinical-image training augmentation pipeline."""
    tf = _tensorflow()
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def _checkpoint_callbacks(tf, destination: Path, validation_available: bool):
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    monitor = "val_loss" if validation_available else "loss"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(destination), monitor=monitor, mode="min",
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="min", patience=20, restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="min", factor=0.2, patience=5,
            min_lr=1e-7, cooldown=1, verbose=1,
        ),
    ]


def _balanced_class_weights(labels: list[int]) -> dict[int, float]:
    """Compute sklearn's balanced weights for zero-based CVM class indices."""
    try:
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for balanced CVM class weights. "
            "Install backend/requirements.txt."
        ) from exc
    classes = np.unique(np.asarray(labels, dtype=np.int32))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels, dtype=np.int32))
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def _focal_and_sample_weights(class_weights: dict[int, float], labels: list[int]) -> tuple[list[float], dict[int, float]]:
    """Split balanced emphasis between focal alpha and mild per-example weights."""
    class_ids = sorted(class_weights)
    root_weights = {class_id: float(np.sqrt(class_weights[class_id])) for class_id in class_ids}
    alpha_mean = float(np.mean(list(root_weights.values()))) or 1.0
    focal_alpha = [root_weights[class_id] / alpha_mean for class_id in class_ids]
    population_mean = float(np.mean([root_weights[int(label)] for label in labels])) or 1.0
    sample_weights = {class_id: root_weights[class_id] / population_mean for class_id in class_ids}
    return focal_alpha, sample_weights


def _coerce_cva_label(value: Any) -> int | None:
    """Convert only explicit annotation values to the binary CVA target."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "abnormal", "present", "positive", "cva"}:
            return 1
        if normalized in {"0", "false", "normal", "absent", "negative", "no_cva"}:
            return 0
    return None


def _find_explicit_cva_annotation(value: Any) -> int | None:
    """Search JSON only for CVA-specific fields; ignore unrelated stage labels."""
    cva_keys = {
        "cva", "cva_present", "cervical_vertebral_anomaly",
        "cervical_vertebral_anomalies", "anomaly_present", "has_cva",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized_key in cva_keys:
                label = _coerce_cva_label(item)
                if label is not None:
                    return label
                if isinstance(item, dict):
                    for nested_key in ("present", "label", "value", "status"):
                        if nested_key in item:
                            label = _coerce_cva_label(item[nested_key])
                            if label is not None:
                                return label
            label = _find_explicit_cva_annotation(item)
            if label is not None:
                return label
    elif isinstance(value, list):
        for item in value:
            label = _find_explicit_cva_annotation(item)
            if label is not None:
                return label
    return None


def _discover_explicit_cva_labels(
    dataset_dir: str | Path, splits: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    """Use dataset-embedded CVA labels only when explicitly annotated.

    Supports CVA-specific fields in annotation JSON and filenames containing
    explicit forms such as ``cva_positive`` or ``cva_normal``. CVM stages,
    landmarks, and arbitrary filenames are never converted into CVA targets.
    """
    records = [record for rows in splits.values() for record in rows]
    record_ids = {record["image_id"] for record in records}
    labels: dict[str, int] = {}
    root = Path(dataset_dir).expanduser()

    for split in ("train", "valid", "test"):
        annotation_root = root / split / "Annotations"
        if not annotation_root.is_dir():
            continue
        for annotation_path in annotation_root.rglob("*.json"):
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring unreadable annotation while searching for explicit CVA labels: %s", annotation_path)
                continue
            image_id = str(
                annotation.get("ceph_id") or annotation.get("cephalogram_id")
                or annotation.get("image_id") or ""
            ).strip()
            if image_id in record_ids:
                label = _find_explicit_cva_annotation(annotation)
                if label is not None:
                    labels[image_id] = label

    label_tokens = {"positive", "negative", "normal", "abnormal", "present", "absent"}
    cva_tokens = {"cva", "anomaly", "anomalies"}
    for record in records:
        if record["image_id"] in labels:
            continue
        tokens = set(re.split(r"[^a-z0-9]+", Path(record["image_path"]).stem.lower()))
        if not (tokens & cva_tokens and tokens & label_tokens):
            continue
        if tokens & {"abnormal", "positive", "present"}:
            labels[record["image_id"]] = 1
        elif tokens & {"normal", "negative", "absent"}:
            labels[record["image_id"]] = 0
    return labels


def train_cva_model(
    dataset_dir: str | Path,
    cva_labels_csv: str | Path | None = None,
    output_path: str | Path = DEFAULT_CVA_MODEL_PATH,
    epochs: int = 120,
    batch_size: int = 16,
    roi_boxes_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Train the independent binary CVA model from a cva_present CSV.

    Required columns: cephalogram_id,cva_present (0=normal, 1=anomaly present).
    The source image files must follow the Aariz train/valid/test directory layout.
    """
    labels_path = Path(cva_labels_csv).expanduser() if cva_labels_csv and str(cva_labels_csv).strip() else None
    if labels_path is not None and labels_path.exists() and not labels_path.is_file():
        raise ValueError(f"CVA labels path is not a file: {labels_path}")

    splits = load_aariz_dataset(dataset_dir, roi_boxes_csv=roi_boxes_csv)
    labels: dict[str, int] = {}
    label_source = "cva_labels_csv"
    if labels_path is not None and labels_path.is_file():
        with labels_path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                image_id = (row.get("cephalogram_id") or row.get("image_id") or row.get("ceph_id") or "").strip()
                raw_label = (row.get("cva_present") or row.get("label") or "").strip().lower()
                label = _coerce_cva_label(raw_label)
                if label is None:
                    raise ValueError(f"Invalid CVA label {raw_label!r} for image {image_id!r}; use 0/1 or normal/abnormal.")
                if not image_id:
                    raise ValueError("Each CVA label row requires cephalogram_id.")
                labels[image_id] = label
    else:
        message = "CVA labels CSV missing; searching dataset annotations and filenames for explicit CVA labels."
        logger.warning(message)
        labels = _discover_explicit_cva_labels(dataset_dir, splits)
        label_source = "explicit_dataset_cva_annotations"

    if not labels:
        message = (
            "CVA training skipped: no explicit CVA labels were found in the CSV, dataset annotation JSON, "
            "or filenames. CVM stages and arbitrary filenames cannot safely serve as CVA labels."
        )
        logger.warning(message)
        return {
            "status": "skipped",
            "skipped": True,
            "reason": "no_explicit_cva_labels",
            "message": message,
            "labels_path": str(labels_path) if labels_path else None,
            "label_source": label_source,
            "cva_model_trained": False,
            "model_path": str(output_path),
        }

    tf = _tensorflow()
    records_by_split = {
        split: [dict(row, cva_present=labels[row["image_id"]]) for row in rows if row["image_id"] in labels]
        for split, rows in splits.items()
    }
    train_records = records_by_split.get("train", [])
    all_labeled_records = [record for rows in records_by_split.values() for record in rows]
    if roi_boxes_csv and any(record.get("roi_box") is None for record in all_labeled_records):
        raise ValueError("CVA ROI training requires a normalized ROI box for every CVA-labeled image.")
    if not train_records or len({row["cva_present"] for row in train_records}) != 2:
        raise ValueError("CVA training requires labeled train images from both classes: normal (0) and abnormal (1).")

    augmenter = _training_image_augmenter()

    def dataset(records: list[dict[str, Any]], shuffle: bool):
        def generator():
            for record in records:
                with Image.open(record["image_path"]) as source:
                    rgb = source.convert("RGB")
                    if record.get("roi_box"):
                        rgb = _crop_normalized(rgb, record["roi_box"])
                    pixels = np.ascontiguousarray(rgb.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR), dtype=np.float32)
                    if shuffle:
                        pixels = augmenter.random_transform(pixels)
                    yield pixels, np.asarray([record["cva_present"]], dtype=np.float32)
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32), tf.TensorSpec(shape=(1,), dtype=tf.float32)),
        )
        if shuffle:
            ds = ds.shuffle(min(len(records), 1000), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    valid_records = records_by_split.get("valid", [])
    test_records = records_by_split.get("test", [])
    model = build_cva_classifier()
    destination = Path(output_path).expanduser().resolve()
    callbacks = _checkpoint_callbacks(tf, destination, bool(valid_records))
    model.fit(
        dataset(train_records, True), validation_data=dataset(valid_records, False) if valid_records else None,
        epochs=epochs, callbacks=callbacks,
    )
    # Always evaluate and publish the best checkpoint, not just the final epoch.
    model = tf.keras.models.load_model(destination)
    validation_metrics = _evaluate_cva_records(model, valid_records, bool(roi_boxes_csv))
    test_metrics = _evaluate_cva_records(model, test_records, bool(roi_boxes_csv))
    logger.info("CVA validation metrics: %s", json.dumps({"validation": validation_metrics, "test": test_metrics}))
    metadata = {
        "image_size": list(IMAGE_SIZE), "class_names": {"0": "Normal", "1": "Potential CVA"},
        "label_column": "cva_present", "label_semantics": "binary_any_cva", "threshold": 0.5,
        "label_source": label_source,
        "roi_training_enabled": bool(roi_boxes_csv),
        "anomaly_type_examples": ["C2-C3 fusion", "block fusion", "atlas spina bifida", "arch deficiency"],
        "subtype_detection_supported": False,
        "evaluation_metrics": {"validation": validation_metrics, "test": test_metrics},
    }
    destination.with_suffix(".labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    clear_model_cache()
    return {"status": "trained", "model_path": str(destination), "label_source": label_source, "train_images": len(train_records), "valid_images": len(valid_records), "test_images": len(test_records), "cva_model_trained": True, "evaluation_metrics": metadata["evaluation_metrics"]}


def _crop_normalized(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crop a normalized x0,y0,x1,y1 box, expanding slightly for context."""
    width, height = image.size
    x0, y0, x1, y1 = box
    pad_x, pad_y = (x1 - x0) * 0.08, (y1 - y0) * 0.08
    left = max(0, int((x0 - pad_x) * width))
    top = max(0, int((y0 - pad_y) * height))
    right = min(width, int((x1 + pad_x) * width))
    bottom = min(height, int((y1 + pad_y) * height))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _cohen_kappa(y_true: list[int], y_pred: list[int], labels: list[int]) -> float | None:
    if not y_true:
        return None
    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    total = len(y_true)
    observed = sum(matrix[label][label] for label in labels) / total
    expected = sum(
        (sum(matrix[label].values()) / total)
        * (sum(matrix[row][label] for row in labels) / total)
        for label in labels
    )
    return None if abs(1 - expected) < 1e-12 else float((observed - expected) / (1 - expected))


def _classification_summary(y_true: list[int], probabilities: np.ndarray, labels: list[int], class_names: dict[str, str]) -> dict[str, Any]:
    if not y_true:
        return {"sample_count": 0, "overall_accuracy_percent": None, "per_stage_accuracy": {}, "cohen_kappa": None, "macro_f1_percent": None, "multiclass_brier_score": None, "expected_calibration_error_percent": None}
    predictions = np.argmax(probabilities, axis=1).astype(int).tolist()
    class_indices = list(range(len(labels)))
    correct = sum(actual == predicted for actual, predicted in zip(y_true, predictions))
    per_stage = {}
    f1_values = []
    for index, label in zip(class_indices, labels):
        support = sum(actual == index for actual in y_true)
        tp = sum(actual == index and predicted == index for actual, predicted in zip(y_true, predictions))
        fp = sum(actual != index and predicted == index for actual, predicted in zip(y_true, predictions))
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_stage[class_names.get(str(label), f"CVMI-{label}")] = {
            "accuracy_percent": round(recall * 100, 2) if recall is not None else None,
            "support": support,
        }
    one_hot = np.eye(len(labels), dtype=np.float64)[np.asarray(y_true, dtype=np.int64)]
    brier = float(np.mean(np.sum((probabilities.astype(np.float64) - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    is_correct = (np.asarray(y_true) == np.asarray(predictions)).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 if lower < 0.9 else confidence <= 1.0)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(is_correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "sample_count": len(y_true),
        "overall_accuracy_percent": round(correct / len(y_true) * 100, 2),
        "per_stage_accuracy": per_stage,
        "cohen_kappa": round(_cohen_kappa(y_true, predictions, class_indices), 4) if len(class_indices) > 1 else None,
        "macro_f1_percent": round(float(np.mean(f1_values)) * 100, 2) if f1_values else None,
        "multiclass_brier_score": round(brier, 5),
        "expected_calibration_error_percent": round(ece * 100, 2),
        "reliability_note": "Kappa is model/reference-label agreement; ECE and Brier score are uncalibrated probability diagnostics, not clinical reliability certification.",
    }


def _evaluate_cvm_records(
    model, records: list[dict[str, Any]], stage_values: list[int], stage_titles: dict[str, str],
    image_size: tuple[int, int], use_roi: bool, age_target_source: str | None = None,
    stage_age_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    y_true: list[int] = []
    probabilities: list[np.ndarray] = []
    true_ages: list[float] = []
    predicted_ages: list[float] = []
    age_prediction_methods: set[str] = set()
    index_by_stage = {stage: index for index, stage in enumerate(stage_values)}
    for record in records:
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
            if use_roi:
                if record.get("roi_box") is None:
                    raise ValueError(f"Held-out evaluation requires an annotated ROI for {record['image_id']}.")
                image = _crop_normalized(image, record["roi_box"])
            pixels = np.ascontiguousarray(image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
        raw = model(pixels, training=False)
        outputs = _model_output_dict(model, raw)
        y_true.append(index_by_stage[record["cvm_stage"]])
        stage_probabilities = np.asarray(outputs["cvm_stage"])[0]
        probabilities.append(stage_probabilities)
        predicted_stage = int(stage_values[int(np.argmax(stage_probabilities))])
        target = record.get("skeletal_age_years")
        if target is None and stage_age_map:
            target = stage_age_map.get(str(record["cvm_stage"]))
        age_output = outputs.get("skeletal_age_years")
        if age_output is not None:
            predicted_age = float(np.asarray(age_output).reshape(-1)[0])
            age_prediction_methods.add("regression_head")
        elif stage_age_map and stage_age_map.get(str(predicted_stage)) is not None:
            # Legacy checkpoints without the regression branch remain
            # evaluable, but this is a CVM-to-age proxy, not a regressor.
            predicted_age = float(stage_age_map[str(predicted_stage)])
            age_prediction_methods.add("stage_mapping_fallback_from_predicted_cvm")
        else:
            predicted_age = None
        if target is not None and predicted_age is not None:
            true_ages.append(float(target))
            predicted_ages.append(predicted_age)
    class_metrics = _classification_summary(y_true, np.asarray(probabilities), stage_values, stage_titles) if records else _classification_summary([], np.empty((0, len(stage_values))), stage_values, stage_titles)
    age_metrics: dict[str, Any] = {
        "status": "not_available",
        "sample_count": 0,
        "mae_years": None,
        "rmse_years": None,
        "prediction_method": None,
        "target_source": age_target_source,
        "interpretation": "Train the regression head with observed age labels or an explicit stage-age mapping to report age metrics.",
    }
    if true_ages:
        errors = np.asarray(predicted_ages) - np.asarray(true_ages)
        age_metrics = {
            "status": "evaluated",
            "sample_count": len(true_ages), "mae_years": round(float(np.mean(np.abs(errors))), 3),
            "rmse_years": round(float(np.sqrt(np.mean(errors ** 2))), 3),
            "prediction_method": sorted(age_prediction_methods),
            "target_source": age_target_source or ("synthetic_stage_mapping" if stage_age_map else "observed_labels"),
            "interpretation": (
                "Stage-mapping fallback metrics measure agreement with a synthetic stage proxy; they do not evaluate a regression head or clinical skeletal age."
                if "stage_mapping_fallback_from_predicted_cvm" in age_prediction_methods
                else "Proxy-age metrics are synthetic mapping agreement, not skeletal-age clinical accuracy."
                if age_target_source == "synthetic_stage_mapping"
                else "Age-head error against supplied age labels; clinical validity requires external validation."
            ),
        }
    class_metrics["skeletal_age_regression"] = age_metrics
    return class_metrics


def _evaluate_cva_records(model, records: list[dict[str, Any]], use_roi: bool) -> dict[str, Any]:
    y_true: list[int] = []
    scores: list[float] = []
    for record in records:
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
            if use_roi:
                if record.get("roi_box") is None:
                    raise ValueError(f"Held-out CVA evaluation requires an annotated ROI for {record['image_id']}.")
                image = _crop_normalized(image, record["roi_box"])
            pixels = np.ascontiguousarray(image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
        score = model(pixels, training=False)
        scores.append(float(np.asarray(score.numpy() if hasattr(score, "numpy") else score).reshape(-1)[0]))
        y_true.append(int(record["cva_present"]))
    if not records:
        return {"sample_count": 0, "overall_accuracy_percent": None, "sensitivity_percent": None, "specificity_percent": None, "cohen_kappa": None, "macro_f1_percent": None, "auc": None, "brier_score": None}
    predictions = [int(score >= 0.5) for score in scores]
    tp = sum(a == 1 and p == 1 for a, p in zip(y_true, predictions))
    tn = sum(a == 0 and p == 0 for a, p in zip(y_true, predictions))
    positives, negatives = sum(y_true), len(y_true) - sum(y_true)
    auc = None
    if positives and negatives:
        positive_scores = [score for score, label in zip(scores, y_true) if label == 1]
        negative_scores = [score for score, label in zip(scores, y_true) if label == 0]
        auc = sum((pos > neg) + 0.5 * (pos == neg) for pos in positive_scores for neg in negative_scores) / (positives * negatives)
    f1_values = []
    for label in (0, 1):
        tp_class = sum(actual == label and predicted == label for actual, predicted in zip(y_true, predictions))
        fp_class = sum(actual != label and predicted == label for actual, predicted in zip(y_true, predictions))
        fn_class = sum(actual == label and predicted != label for actual, predicted in zip(y_true, predictions))
        denom = 2 * tp_class + fp_class + fn_class
        f1_values.append(2 * tp_class / denom if denom else 0.0)
    return {
        "sample_count": len(y_true), "overall_accuracy_percent": round((tp + tn) / len(y_true) * 100, 2),
        "sensitivity_percent": round(tp / positives * 100, 2) if positives else None,
        "specificity_percent": round(tn / negatives * 100, 2) if negatives else None,
        "cohen_kappa": round(_cohen_kappa(y_true, predictions, [0, 1]), 4),
        "macro_f1_percent": round(float(np.mean(f1_values)) * 100, 2),
        "auc": round(float(auc), 4) if auc is not None else None,
        "brier_score": round(float(np.mean((np.asarray(scores) - np.asarray(y_true)) ** 2)), 5),
        "reliability_note": "Kappa is model/reference-label agreement. AUC and Brier score are uncalibrated screening diagnostics, not clinical reliability certification.",
    }


def evaluate_saved_models(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    model_path: str | Path = MODEL_PATH,
    age_labels_csv: str | Path | None = None,
    roi_boxes_csv: str | Path | None = None,
    cva_labels_csv: str | Path | None = None,
    cva_model_path: str | Path = DEFAULT_CVA_MODEL_PATH,
    metrics_output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved checkpoints on validation and held-out test partitions."""
    cvm_path = Path(model_path)
    cvm_labels = cvm_path.with_suffix(".labels.json")
    if not cvm_path.is_file() or not cvm_labels.is_file():
        raise FileNotFoundError(f"CVM checkpoint and sidecar are required: {cvm_path} / {cvm_labels}")
    splits = load_aariz_dataset(dataset_dir, age_labels_csv, roi_boxes_csv)
    metadata = json.loads(cvm_labels.read_text(encoding="utf-8"))
    cvm_model = _tensorflow().keras.models.load_model(cvm_path, compile=False)
    has_age_output = "skeletal_age_years" in (getattr(cvm_model, "output_names", []) or [])
    stages = [int(value) for value in metadata["stage_values"]]
    class_titles = metadata.get("stage_titles", {})
    use_roi = bool(metadata.get("roi_training_source"))
    evaluation_age_map = metadata.get("stage_age_map")
    if not has_age_output and not evaluation_age_map:
        fallback_age_map = _load_stage_age_map()
        evaluation_age_map = {str(key): value for key, value in fallback_age_map.items()} or None
    evaluation_age_source = metadata.get("age_target_source")
    if not has_age_output and evaluation_age_map:
        evaluation_age_source = "legacy_checkpoint_stage_mapping_fallback"
    metadata["has_skeletal_age_head"] = has_age_output
    metadata["checkpoint_output_names"] = list(getattr(cvm_model, "output_names", []) or [])
    evaluation = {
        "validation": _evaluate_cvm_records(cvm_model, splits.get("valid", []), stages, class_titles, tuple(metadata.get("image_size", IMAGE_SIZE)), use_roi, evaluation_age_source, evaluation_age_map),
        "test": _evaluate_cvm_records(cvm_model, splits.get("test", []), stages, class_titles, tuple(metadata.get("image_size", IMAGE_SIZE)), use_roi, evaluation_age_source, evaluation_age_map),
    }
    metadata["evaluation_age_proxy_map_used"] = bool(evaluation_age_map and not has_age_output)
    metadata["evaluation_metrics"] = evaluation
    metadata["published_benchmarks"] = PUBLISHED_BENCHMARKS
    cvm_labels.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result: dict[str, Any] = {"cvm": evaluation, "published_benchmarks": PUBLISHED_BENCHMARKS}

    if cva_labels_csv:
        cva_path = Path(cva_model_path)
        cva_sidecar = cva_path.with_suffix(".labels.json")
        if not cva_path.is_file() or not cva_sidecar.is_file():
            raise FileNotFoundError(f"CVA checkpoint and sidecar are required: {cva_path} / {cva_sidecar}")
        cva_meta = json.loads(cva_sidecar.read_text(encoding="utf-8"))
        labels: dict[str, int] = {}
        with Path(cva_labels_csv).open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                image_id = (row.get("cephalogram_id") or row.get("image_id") or row.get("ceph_id") or "").strip()
                raw = (row.get("cva_present") or row.get("label") or "").strip().lower()
                if raw in {"1", "true", "abnormal", "present", "cva"}:
                    labels[image_id] = 1
                elif raw in {"0", "false", "normal", "absent", "no_cva"}:
                    labels[image_id] = 0
                elif image_id:
                    raise ValueError(f"Invalid CVA label {raw!r} for {image_id}.")
        labeled_splits = {
            split: [dict(record, cva_present=labels[record["image_id"]]) for record in rows if record["image_id"] in labels]
            for split, rows in splits.items()
        }
        cva_model = _tensorflow().keras.models.load_model(cva_path)
        cva_evaluation = {
            "validation": _evaluate_cva_records(cva_model, labeled_splits.get("valid", []), bool(cva_meta.get("roi_training_enabled"))),
            "test": _evaluate_cva_records(cva_model, labeled_splits.get("test", []), bool(cva_meta.get("roi_training_enabled"))),
        }
        cva_meta["evaluation_metrics"] = cva_evaluation
        cva_meta["published_benchmarks"] = PUBLISHED_BENCHMARKS
        cva_sidecar.write_text(json.dumps(cva_meta, indent=2), encoding="utf-8")
        result["cva"] = cva_evaluation
    logger.info("Saved-checkpoint evaluation metrics: %s", json.dumps(result))
    report_path = Path(metrics_output) if metrics_output else Path(__file__).parent / "artifacts" / "evaluation_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["metrics_json_path"] = str(report_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    clear_model_cache()
    return result


def train_cvmi_model(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_path: str | Path = MODEL_PATH,
    age_labels_csv: str | Path | None = None,
    stage_age_map: dict[int, float] | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    image_size: tuple[int, int] = IMAGE_SIZE,
    roi_boxes_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Train MobileNetV2 from Aariz and save the model plus stage metadata."""
    tf = _tensorflow()
    splits = load_aariz_dataset(dataset_dir, age_labels_csv, roi_boxes_csv)
    stage_values = sorted({r["cvm_stage"] for r in splits["train"]})
    if stage_values != [1, 2, 3, 4, 5, 6]:
        raise ValueError(f"CVM classifier requires labeled CVMI stages 1 through 6; found {stage_values}.")
    age_training_records = splits["train"] + splits.get("valid", [])
    has_direct_age = bool(age_training_records) and all(r["skeletal_age_years"] is not None for r in age_training_records)
    mapped_age_map = {int(key): float(value) for key, value in (stage_age_map or {}).items()}
    if not has_direct_age and not mapped_age_map:
        # Aariz supplies CVM stages only. Use the checked-in proxy map so a
        # standard training run still builds the requested regression branch.
        mapped_age_map = _load_stage_age_map()
        if mapped_age_map:
            logger.warning(
                "No complete observed-age labels found; training skeletal_age_years with the project stage-age proxy map at %s.",
                DEFAULT_STAGE_AGE_MAP_PATH,
            )
    has_mapped_age = bool(mapped_age_map) and all(r["cvm_stage"] in mapped_age_map for r in age_training_records)
    has_age = has_direct_age or has_mapped_age
    if not has_age:
        raise ValueError(
            "CVM training requires a skeletal-age target source to build the regression head. "
            f"Provide complete age labels or a complete stage-age map (default: {DEFAULT_STAGE_AGE_MAP_PATH})."
        )
    age_target_source = "observed_labels" if has_direct_age else ("synthetic_stage_mapping" if has_mapped_age else None)
    if has_age:
        age_values = [
            float(r["skeletal_age_years"]) if has_direct_age else mapped_age_map[r["cvm_stage"]]
            for r in age_training_records
        ]
        if any(not np.isfinite(value) or value <= 0 for value in age_values):
            raise ValueError("Skeletal age targets must be finite positive numbers.")
    elif stage_age_map:
        missing_stages = sorted({r["cvm_stage"] for r in age_training_records} - mapped_age_map.keys())
        raise ValueError(f"Stage-to-age mapping is missing CVM stages: {missing_stages}")
    stage_to_index = {value: index for index, value in enumerate(stage_values)}
    train_stage_indices = [stage_to_index[record["cvm_stage"]] for record in splits["train"]]
    class_weights = _balanced_class_weights(train_stage_indices)
    focal_alpha, balanced_sample_weights = _focal_and_sample_weights(class_weights, train_stage_indices)
    roi_training_records = splits["train"] + splits.get("valid", [])
    has_roi_labels = bool(roi_training_records) and all(record["roi_box"] is not None for record in roi_training_records)

    roi_model = None
    roi_destination = Path(output_path).with_name(f"{Path(output_path).stem}_roi.keras")
    if has_roi_labels:
        def roi_dataset(records, shuffle: bool):
            def generator():
                for record in records:
                    with Image.open(record["image_path"]) as source:
                        image = source.convert("RGB").resize((image_size[1], image_size[0]))
                        yield np.asarray(image, dtype=np.float32), np.asarray(record["roi_box"], dtype=np.float32)
            ds = tf.data.Dataset.from_generator(
                generator,
                output_signature=(tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32), tf.TensorSpec(shape=(4,), dtype=tf.float32)),
            )
            if shuffle:
                ds = ds.shuffle(min(len(records), 1000), reshuffle_each_iteration=True)
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        roi_model = build_roi_localizer()
        roi_valid_ds = roi_dataset(splits.get("valid", []), False) if splits.get("valid") else None
        roi_model.fit(
            roi_dataset(splits["train"], True), validation_data=roi_valid_ds,
            epochs=epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss" if roi_valid_ds else "loss", patience=4, restore_best_weights=True)],
        )
        roi_destination.parent.mkdir(parents=True, exist_ok=True)
        roi_model.save(roi_destination)

    augmenter = _training_image_augmenter()

    def make_dataset(records, shuffle: bool):
        if not records:
            return None

        def generator():
            for record in records:
                try:
                    with Image.open(record["image_path"]) as image:
                        rgb = image.convert("RGB")
                        if has_roi_labels:
                            rgb = _crop_normalized(rgb, record["roi_box"])
                        rgb = rgb.resize((image_size[1], image_size[0]))
                        pixels = np.asarray(rgb, dtype=np.float32)
                        if shuffle:
                            pixels = augmenter.random_transform(pixels)
                except (OSError, UnidentifiedImageError) as exc:
                    raise ValueError(f"Could not read image {record['image_path']}: {exc}") from exc
                stage_index = stage_to_index[record["cvm_stage"]]
                label = {"cvm_stage": np.int32(stage_index)}
                if has_age:
                    target = record["skeletal_age_years"] if has_direct_age else mapped_age_map[record["cvm_stage"]]
                    label["skeletal_age_years"] = np.asarray([target], dtype=np.float32)
                if shuffle:
                    # tf.data streams batches and does not reliably support
                    # Keras class_weight across multi-output models. Per-example
                    # sample weights apply the identical balanced weighting.
                    sample_weight = {"cvm_stage": np.float32(balanced_sample_weights[stage_index])}
                    if has_age:
                        sample_weight["skeletal_age_years"] = np.float32(1.0)
                    yield pixels, label, sample_weight
                else:
                    yield pixels, label

        output_signature = (
            tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
            {"cvm_stage": tf.TensorSpec(shape=(), dtype=tf.int32), **({"skeletal_age_years": tf.TensorSpec(shape=(1,), dtype=tf.float32)} if has_age else {})},
        )
        if shuffle:
            output_signature = (*output_signature, {
                "cvm_stage": tf.TensorSpec(shape=(), dtype=tf.float32),
                **({"skeletal_age_years": tf.TensorSpec(shape=(), dtype=tf.float32)} if has_age else {}),
            })
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        if shuffle:
            dataset = dataset.shuffle(min(len(records), 1000), reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(splits["train"], shuffle=True)
    valid_ds = make_dataset(splits.get("valid", []), shuffle=False)
    model = build_mobilenetv2_model(
        stage_values, train_age_head=has_age, focal_alpha=focal_alpha, focal_gamma=2.0
    )
    destination = Path(output_path).expanduser().resolve()
    callbacks = _checkpoint_callbacks(tf, destination, valid_ds is not None)
    model.fit(
        train_ds, validation_data=valid_ds, epochs=epochs, callbacks=callbacks,
        # class_weights are encoded as per-example sample weights in the
        # streaming dataset above, which supports both CVM and age outputs.
    )
    # Reload the best ModelCheckpoint artifact so metrics and sidecar describe it.
    model = tf.keras.models.load_model(destination, compile=False)

    stage_titles = {str(stage): f"CVMI-{stage}" for stage in stage_values}
    age_map_for_eval = {str(key): value for key, value in mapped_age_map.items()} if has_mapped_age else None
    validation_metrics = _evaluate_cvm_records(model, splits.get("valid", []), stage_values, stage_titles, image_size, has_roi_labels, age_target_source, age_map_for_eval)
    test_metrics = _evaluate_cvm_records(model, splits.get("test", []), stage_values, stage_titles, image_size, has_roi_labels, age_target_source, age_map_for_eval)
    logger.info("CVM validation metrics: %s", json.dumps({"validation": validation_metrics, "test": test_metrics}))

    metadata = {
        "stage_values": stage_values,
        "stage_titles": stage_titles,
        "image_size": list(image_size),
        "has_skeletal_age_head": has_age,
        "skeletal_age_head_architecture": "parallel_dense_regression" if has_age else None,
        "age_target_source": age_target_source,
        "stage_age_map": {str(key): value for key, value in mapped_age_map.items()} if has_mapped_age else None,
        "cvm_class_weights": {str(key): value for key, value in class_weights.items()},
        "cvm_balancing": {
            "classification_loss": "balanced_sparse_focal_loss",
            "focal_gamma": 2.0,
            "focal_alpha": focal_alpha,
            "sample_weight_strategy": "normalized_sqrt_balanced_class_weight",
            "sample_weights": {str(key): value for key, value in balanced_sample_weights.items()},
        },
        "fine_tuning": {"unfrozen_backbone_layers": 35, "optimizer": "Adam", "learning_rate": 1e-4},
        "roi_model_path": roi_destination.name if has_roi_labels else None,
        "roi_training_source": "c2_c3_c4_box_annotations" if has_roi_labels else None,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "evaluation_metrics": {"validation": validation_metrics, "test": test_metrics},
        "published_benchmarks": PUBLISHED_BENCHMARKS,
    }
    destination.with_suffix(".labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    clear_model_cache()
    return {"model_path": str(destination), "train_images": len(splits["train"]), "valid_images": len(splits.get("valid", [])), "test_images": len(splits.get("test", [])), "cvm_stages": stage_values, "skeletal_age_head_trained": has_age, "age_target_source": age_target_source, "roi_localizer_trained": has_roi_labels, "roi_model_path": str(roi_destination) if has_roi_labels else None, "evaluation_metrics": metadata["evaluation_metrics"], "published_benchmarks": PUBLISHED_BENCHMARKS}


# Backward-compatible function name used by earlier project scripts.
train_model = train_cvmi_model


def clear_model_cache() -> None:
    global _loaded_model, _loaded_metadata, _loaded_roi_model, _loaded_cva_model, _loaded_cva_metadata, _model_generation
    _loaded_model = None
    _loaded_metadata = None
    _loaded_roi_model = None
    _loaded_cva_model = None
    _loaded_cva_metadata = None
    with _cache_lock:
        _model_generation += 1
        _prediction_cache.clear()


def _get_model():
    global _loaded_model, _loaded_metadata
    if _loaded_model is not None:
        return _loaded_model, _loaded_metadata
    with _load_lock:
        if _loaded_model is not None:
            return _loaded_model, _loaded_metadata
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train it with `python backend/model.py --train`.")
        tf = _tensorflow()
        # Inference does not need a compiled optimizer/loss. compile=False also
        # supports checkpoints saved with the training-time focal-loss closure.
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        if not LABELS_PATH.is_file():
            raise FileNotFoundError(f"Model metadata not found at {LABELS_PATH}.")
        loaded_metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        _loaded_model = loaded_model
        _loaded_metadata = loaded_metadata
    return _loaded_model, _loaded_metadata


def _get_roi_model(metadata: dict[str, Any]):
    global _loaded_roi_model
    roi_file = metadata.get("roi_model_path")
    if not roi_file:
        return None
    roi_path = Path(roi_file)
    if not roi_path.is_absolute():
        roi_path = MODEL_PATH.parent / roi_path
    if not roi_path.is_file():
        return None
    if _loaded_roi_model is None:
        with _load_lock:
            if _loaded_roi_model is None:
                _loaded_roi_model = _tensorflow().keras.models.load_model(roi_path)
    return _loaded_roi_model


def _get_cva_model():
    """Return the independent CVA detector, or (None, None) if not trained."""
    global _loaded_cva_model, _loaded_cva_metadata
    if _loaded_cva_model is not None:
        return _loaded_cva_model, _loaded_cva_metadata
    if not DEFAULT_CVA_MODEL_PATH.is_file() or not CVA_LABELS_PATH.is_file():
        return None, None
    with _load_lock:
        if _loaded_cva_model is None:
            _loaded_cva_model = _tensorflow().keras.models.load_model(DEFAULT_CVA_MODEL_PATH)
            _loaded_cva_metadata = json.loads(CVA_LABELS_PATH.read_text(encoding="utf-8"))
    return _loaded_cva_model, _loaded_cva_metadata


def get_model_status() -> dict[str, Any]:
    """Inspect checkpoint and sidecar availability without loading TensorFlow models."""
    warnings: list[str] = []
    status: dict[str, Any] = {"cvm": {}, "cva": {}, "roi": {}, "warnings": warnings}
    cvm_sidecar: dict[str, Any] = {}
    cva_sidecar: dict[str, Any] = {}
    status["cvm"] = {
        "weights_present": MODEL_PATH.is_file(), "metadata_present": LABELS_PATH.is_file(),
        "state": "ready" if MODEL_PATH.is_file() and LABELS_PATH.is_file() else "missing_artifact",
        "weights_path": str(MODEL_PATH),
    }
    if not MODEL_PATH.is_file():
        warnings.append(f"CVM model weights missing: {MODEL_PATH}. Train the CVMI model before inference.")
    elif not LABELS_PATH.is_file():
        status["cvm"]["state"] = "metadata_missing"
        warnings.append(f"CVM model metadata missing: {LABELS_PATH}.")
    else:
        try:
            cvm_sidecar = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            status["cvm"]["validation_metrics"] = cvm_sidecar.get("evaluation_metrics")
        except (OSError, json.JSONDecodeError) as exc:
            status["cvm"]["state"] = "metadata_invalid"
            warnings.append(f"CVM model metadata could not be read: {exc}")
    roi_relative = cvm_sidecar.get("roi_model_path")
    roi_path = Path(roi_relative) if roi_relative else None
    if roi_path is not None and not roi_path.is_absolute():
        roi_path = MODEL_PATH.parent / roi_path
    roi_present = roi_path.is_file() if roi_path else False
    status["roi"] = {"required_by_checkpoint": bool(roi_relative), "weights_present": roi_present, "state": "ready" if not roi_relative or roi_present else "missing", "weights_path": str(roi_path) if roi_path else None}
    if roi_relative and not roi_present:
        warnings.append(f"ROI localizer weights missing: {roi_path}; inference may use its full-image fallback.")

    status["cva"] = {
        "weights_present": DEFAULT_CVA_MODEL_PATH.is_file(), "metadata_present": CVA_LABELS_PATH.is_file(),
        "state": "ready" if DEFAULT_CVA_MODEL_PATH.is_file() and CVA_LABELS_PATH.is_file() else "unavailable",
        "weights_path": str(DEFAULT_CVA_MODEL_PATH),
    }
    if not DEFAULT_CVA_MODEL_PATH.is_file():
        warnings.append(f"CVA model weights missing: {DEFAULT_CVA_MODEL_PATH}; CVA predictions will be reported as not evaluated.")
    elif not CVA_LABELS_PATH.is_file():
        status["cva"]["state"] = "metadata_missing"
        warnings.append(f"CVA model metadata missing: {CVA_LABELS_PATH}; CVA predictions will be reported as not evaluated.")
    else:
        try:
            cva_sidecar = json.loads(CVA_LABELS_PATH.read_text(encoding="utf-8"))
            status["cva"]["validation_metrics"] = cva_sidecar.get("evaluation_metrics")
        except (OSError, json.JSONDecodeError) as exc:
            status["cva"]["state"] = "metadata_invalid"
            warnings.append(f"CVA model metadata could not be read: {exc}")
    status["published_benchmarks"] = PUBLISHED_BENCHMARKS
    status["ready"] = status["cvm"]["state"] == "ready"
    status["training_status"] = (
        "training_required" if not status["ready"]
        else "ready" if status["cva"]["state"] == "ready"
        else "partial"
    )
    status["validation_summary"] = {
        "cvm": cvm_sidecar.get("evaluation_metrics"),
        "cva": cva_sidecar.get("evaluation_metrics"),
        "published_reference": PUBLISHED_BENCHMARKS,
        "comparability_note": "Published reference metrics came from a different cohort/model task and cannot certify this checkpoint. Local values are reported only when produced on labeled validation/test splits.",
    }
    return status


def _cached_prediction(image_hash: str) -> dict[str, Any] | None:
    with _cache_lock:
        value = _prediction_cache.get((_model_generation, image_hash))
        if value is None:
            return None
        _prediction_cache.move_to_end((_model_generation, image_hash))
        return copy.deepcopy(value)


def _store_prediction(image_hash: str, result: dict[str, Any]) -> None:
    with _cache_lock:
        key = (_model_generation, image_hash)
        _prediction_cache[key] = copy.deepcopy(result)
        _prediction_cache.move_to_end(key)
        while len(_prediction_cache) > max(0, _PREDICTION_CACHE_LIMIT):
            _prediction_cache.popitem(last=False)


def warmup_model() -> bool:
    """Warm each available model independently; return whether CVM is inference-ready."""
    cvm_ready = False
    if MODEL_PATH.is_file() and LABELS_PATH.is_file():
        model, metadata = _get_model()
        image_size = metadata["image_size"]
        dummy = np.zeros((1, image_size[0], image_size[1], 3), dtype=np.float32)
        with _inference_lock:
            model(dummy, training=False)
            roi_model = _get_roi_model(metadata)
            if roi_model is not None:
                roi_model(dummy, training=False)
        cvm_ready = True
    if DEFAULT_CVA_MODEL_PATH.is_file() and CVA_LABELS_PATH.is_file():
        cva_model, cva_metadata = _get_cva_model()
        if cva_model is not None:
            image_size = (cva_metadata or {}).get("image_size", IMAGE_SIZE)
            with _inference_lock:
                cva_model(np.zeros((1, image_size[0], image_size[1], 3), dtype=np.float32), training=False)
    return cvm_ready


def _cva_result_for_pixels(pixels: np.ndarray, source_image: Image.Image | None = None) -> dict[str, Any]:
    cva_model, cva_metadata = _get_cva_model()
    if cva_model is None:
        result = {
            "cva_status": "not_evaluated", "cva_model_available": False,
            "cva_probability": None,
            "cva_label": "Trained CVA model unavailable; research measurements only",
            "cva_message": "No trained CVA detector or CVA-annotated training set is configured.",
        }
        result["cva_morphology_review"] = (
            _experimental_morphology_review(source_image) if source_image is not None else None
        )
        return result
    raw_score = cva_model(pixels, training=False)
    score = float(np.asarray(raw_score.numpy() if hasattr(raw_score, "numpy") else raw_score).reshape(-1)[0])
    threshold = float((cva_metadata or {}).get("threshold", 0.5))
    flagged = score >= threshold
    return {
        "cva_status": "potential_cva_flagged" if flagged else "normal_pattern_predicted",
        "cva_model_available": True,
        "cva_probability": round(score, 6),
        "cva_confidence_percent": round(score * 100 if flagged else (1 - score) * 100, 2),
        "cva_threshold": threshold,
        "cva_label": "Potential Cervical Vertebral Anomaly Flagged" if flagged else "Normal pattern predicted",
        "cva_score_calibration": "uncalibrated_sigmoid_score",
        "cva_message": (
            "Binary image-level screening only; this model does not localize or distinguish CVA subtypes. "
            "Review C2-C4 and the atlas manually."
        ),
        "cva_anomaly_type_examples_not_individually_detected": (cva_metadata or {}).get("anomaly_type_examples", []),
        "cva_subtype_detection_supported": False,
    }


def _apply_cva_roi_compatibility(result: dict[str, Any], roi_status: str) -> dict[str, Any]:
    """Do not emit a CVA prediction if its trained crop input is unavailable."""
    _, metadata = _get_cva_model()
    result["cva_roi_status"] = roi_status
    morphology = result.get("cva_morphology_review")
    if isinstance(morphology, dict):
        morphology["roi_status"] = roi_status
    model_expects_roi = bool(metadata and metadata.get("roi_training_enabled"))
    inference_has_roi = roi_status == "automatic_c2_c3_c4_roi"
    if metadata is not None and model_expects_roi != inference_has_roi:
        result.update({
            "cva_status": "not_evaluated_roi_mode_mismatch",
            "cva_label": "Not evaluated: CVA model / ROI mode mismatch",
            "cva_probability": None,
            "cva_confidence_percent": None,
            "cva_message": "CVA training and inference do not use the same C2-C4 crop configuration. Retrain with matching ROI boxes/localizer settings before using this result.",
        })
    return result


def predict_cva_from_image(image_bytes: bytes) -> dict[str, Any]:
    """Standalone endpoint hook for CVA-only screening."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            rgb = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc
    model, metadata = _get_cva_model()
    roi_status = "full_image_no_localizer"
    try:
        _, cvm_metadata = _get_model()
        roi_model = _get_roi_model(cvm_metadata or {})
    except (FileNotFoundError, RuntimeError):
        roi_model = None
    if roi_model is not None:
        roi_pixels = np.ascontiguousarray(rgb.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
        with _inference_lock:
            box_raw = roi_model(roi_pixels, training=False)
        box = np.asarray(box_raw.numpy() if hasattr(box_raw, "numpy") else box_raw).reshape(-1)
        if box.size == 4:
            x0, x1 = sorted((float(box[0]), float(box[2])))
            y0, y1 = sorted((float(box[1]), float(box[3])))
            if x1 - x0 >= 0.03 and y1 - y0 >= 0.03:
                rgb = _crop_normalized(rgb, (x0, y0, x1, y1))
                roi_status = "automatic_c2_c3_c4_roi"
            else:
                roi_status = "full_image_fallback_invalid_roi"
    image_size = (metadata or {}).get("image_size", list(IMAGE_SIZE))
    pixels = np.ascontiguousarray(rgb.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
    with _inference_lock:
        result = _cva_result_for_pixels(pixels, rgb)
    return _apply_cva_roi_compatibility(result, roi_status)


def predict_from_image(image_bytes: bytes) -> dict[str, Any]:
    """Run an uploaded image through the trained model and map outputs."""
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    cached = _cached_prediction(image_hash)
    if cached is not None:
        return cached

    model, metadata = _get_model()
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc

    roi_model = _get_roi_model(metadata)
    roi_status = "full_image_fallback_no_roi_model"
    width, height = metadata["image_size"][1], metadata["image_size"][0]
    with _inference_lock:
        # Another worker may have completed this image while this one decoded it.
        cached = _cached_prediction(image_hash)
        if cached is not None:
            return cached
        if roi_model is not None:
            localizer_pixels = np.ascontiguousarray(
                image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
            )[None, ...]
            box_tensor = roi_model(localizer_pixels, training=False)
            box = np.asarray(box_tensor.numpy() if hasattr(box_tensor, "numpy") else box_tensor).reshape(-1)
            if box.size == 4:
                x0, x1 = sorted((float(box[0]), float(box[2])))
                y0, y1 = sorted((float(box[1]), float(box[3])))
                if x1 - x0 >= 0.03 and y1 - y0 >= 0.03:
                    image = _crop_normalized(image, (x0, y0, x1, y1))
                    roi_status = "automatic_c2_c3_c4_roi"
                else:
                    roi_status = "full_image_fallback_invalid_roi"

        classifier_image = image.resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.ascontiguousarray(classifier_image, dtype=np.float32)[None, ...]
        raw = model(pixels, training=False)
        outputs = _model_output_dict(model, raw)
        cva_result = _cva_result_for_pixels(pixels, classifier_image)
        cva_result = _apply_cva_roi_compatibility(cva_result, roi_status)
    probabilities = np.asarray(outputs["cvm_stage"])[0]
    index = int(np.argmax(probabilities))
    stage_value = int(metadata["stage_values"][index])
    top_indices = np.argsort(probabilities)[::-1][:min(2, len(probabilities))]
    top2 = [
        {
            "stage": int(metadata["stage_values"][candidate_index]),
            "label": metadata.get("stage_titles", {}).get(
                str(metadata["stage_values"][candidate_index]),
                f"CVM-S{metadata['stage_values'][candidate_index]}",
            ),
            "softmax_score": round(float(probabilities[candidate_index]), 6),
            "softmax_score_percent": round(float(probabilities[candidate_index]) * 100, 2),
        }
        for candidate_index in top_indices
    ]
    age_output = outputs.get("skeletal_age_years")
    age_prediction_method = "regression_head" if age_output is not None else None
    predicted_age = float(np.asarray(age_output).reshape(-1)[0]) if age_output is not None else None
    if predicted_age is not None and not np.isfinite(predicted_age):
        logger.warning("Age regression head returned a non-finite value; using stage-map fallback if available.")
        predicted_age = None
        age_prediction_method = None
    age_map = metadata.get("stage_age_map")
    if not age_map:
        try:
            default_map = _load_stage_age_map()
            age_map = {str(key): value for key, value in default_map.items()}
        except ValueError as exc:
            logger.warning("Could not load default stage-age proxy map: %s", exc)
            age_map = None
    if predicted_age is None and age_map:
        proxy_age = age_map.get(str(stage_value))
        if proxy_age is not None and np.isfinite(float(proxy_age)):
            predicted_age = float(proxy_age)
            age_prediction_method = "stage_mapping_fallback_from_predicted_cvm"
    effective_age_target_source = metadata.get("age_target_source")
    if age_prediction_method == "stage_mapping_fallback_from_predicted_cvm":
        effective_age_target_source = "synthetic_stage_mapping"
    try:
        with _inference_lock:
            explainability = _generate_gradcam_overlay(model, pixels, classifier_image, index)
        explainability["predicted_stage_label"] = metadata.get("stage_titles", {}).get(
            str(stage_value), f"CVM-S{stage_value}"
        )
        explainability["roi_status"] = roi_status
    except Exception as exc:
        logger.warning("Grad-CAM generation failed; returning prediction without overlay: %s", exc)
        explainability = {
            "status": "unavailable",
            "target": "predicted_cvm_stage",
            "layer": None,
            "overlay_data_url": None,
            "interpretation_note": "Grad-CAM could not be generated for this checkpoint.",
        }
    result = {
        "predicted_skeletal_age_years": round(predicted_age, 1) if predicted_age is not None else None,
        "cvm_stage": stage_value,
        "cvm_stage_label": metadata.get("stage_titles", {}).get(str(stage_value), f"CVM-S{stage_value}"),
        **cva_result,
        "cvm_stage_confidence": round(float(probabilities[index]), 4),
        "cvm_stage_top2": top2,
        "confidence_calibration": "uncalibrated_softmax_scores",
        "roi_status": roi_status,
        "xai_gradcam": explainability,
        "age_target_source": effective_age_target_source,
        "skeletal_age_prediction_method": age_prediction_method,
        "skeletal_age_is_proxy": bool(
            effective_age_target_source == "synthetic_stage_mapping"
            or age_prediction_method == "stage_mapping_fallback_from_predicted_cvm"
        ),
        "status": "complete" if predicted_age is not None else "age_labels_required",
        "message": (
            "This age was predicted by a regression head trained on synthetic stage-mapped proxy targets; it is not an observed or clinically validated age."
            if effective_age_target_source == "synthetic_stage_mapping" and age_prediction_method == "regression_head"
            else "Legacy checkpoint has no regression head; this value is the configured age proxy for its predicted CVM stage, not a learned age estimate."
            if age_prediction_method == "stage_mapping_fallback_from_predicted_cvm"
            else "Age head output is based on observed age labels; independent clinical validation is still required."
            if age_prediction_method == "regression_head"
            else "CVM stage predicted. No age head or valid stage-age mapping is available."
        ),
    }
    _store_prediction(image_hash, result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CLEFTSKEL-AI MobileNetV2 model on Aariz.")
    parser.add_argument("--train", "--train-cvmi", dest="train", action="store_true", help="Train and save the six-stage CVMI model.")
    parser.add_argument("--train-cva", action="store_true", help="Train the separate binary CVA detector.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved checkpoints on validation and held-out test splits.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Output .keras model path.")
    parser.add_argument("--age-labels", help="Optional CSV with cephalogram_id,skeletal_age_years columns.")
    parser.add_argument("--stage-age-map", help="JSON mapping stages to synthetic age targets; defaults to project-root stage_age_map.json when observed age labels are unavailable.")
    parser.add_argument("--roi-boxes", help="Optional CSV with normalized C2-C4 ROI boxes: cephalogram_id,x_min,y_min,x_max,y_max.")
    parser.add_argument("--cva-labels", help="CSV with cephalogram_id,cva_present (0=normal, 1=abnormal) columns.")
    parser.add_argument("--cva-roi-boxes", help="Optional normalized C2-C4 ROI boxes CSV for CVA training.")
    parser.add_argument("--cva-output", default=str(DEFAULT_CVA_MODEL_PATH), help="Output path for the separate CVA model.")
    parser.add_argument("--metrics-output", help="Evaluation JSON destination (default: backend/artifacts/evaluation_metrics.json).")
    parser.add_argument("--epochs", type=int, help="Optional override applied to whichever model is being trained.")
    parser.add_argument("--cvmi-epochs", type=int, default=100)
    parser.add_argument("--cva-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate_saved_models(args.dataset, args.output, args.age_labels, args.roi_boxes, args.cva_labels, args.cva_output, args.metrics_output), indent=2))
    elif args.train_cva:
        print(json.dumps(train_cva_model(args.dataset, args.cva_labels, args.cva_output, args.epochs or args.cva_epochs, args.batch_size, args.cva_roi_boxes), indent=2))
    elif args.train:
        age_map = json.loads(Path(args.stage_age_map).read_text(encoding="utf-8")) if args.stage_age_map else None
        print(json.dumps(train_cvmi_model(args.dataset, args.output, args.age_labels, age_map, args.epochs or args.cvmi_epochs, args.batch_size, roi_boxes_csv=args.roi_boxes), indent=2))
    else:
        parser.error("Pass --train, --train-cva, or --evaluate.")
