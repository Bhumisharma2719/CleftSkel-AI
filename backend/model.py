"""Aariz dataset loader, MobileNetV2 training, and inference for CLEFTSKEL-AI.

The Aariz CVM annotation files contain CVM stages, not skeletal age in years.
Age regression can use direct age labels, or explicitly opted-in mapped stage
targets. Mapped targets are synthetic proxies and are marked as such in output.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_DATASET_DIR = Path(os.getenv("CLEFTSKEL_DATASET_DIR", r"C:\Users\Bhumi\Desktop\Aariz"))
MODEL_PATH = Path(os.getenv("CLEFTSKEL_MODEL_PATH", Path(__file__).parent / "artifacts" / "mobilenetv2.keras"))
LABELS_PATH = MODEL_PATH.with_suffix(".labels.json")
ROI_MODEL_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_roi.keras")
IMAGE_SIZE = (224, 224)
_loaded_model = None
_loaded_metadata: dict[str, Any] | None = None
_loaded_roi_model = None
_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_cache_lock = threading.Lock()
_prediction_cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
_model_generation = 0
_PREDICTION_CACHE_LIMIT = int(os.getenv("CLEFTSKEL_PREDICTION_CACHE_SIZE", "64"))


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


def build_mobilenetv2_model(stage_values: list[int], train_age_head: bool = False, imagenet_weights: str | None = "imagenet"):
    """Build MobileNetV2 with a CVM classifier and optional age regressor."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=imagenet_weights
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs: dict[str, Any] = {
        "cvm_stage": tf.keras.layers.Dense(len(stage_values), activation="softmax", name="cvm_stage")(x)
    }
    if train_age_head:
        outputs["skeletal_age_years"] = tf.keras.layers.Dense(1, name="skeletal_age_years")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    losses: dict[str, Any] = {"cvm_stage": "sparse_categorical_crossentropy"}
    metrics: dict[str, Any] = {"cvm_stage": ["accuracy"]}
    if train_age_head:
        losses["skeletal_age_years"] = "mean_squared_error"
        metrics["skeletal_age_years"] = ["mean_absolute_error"]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses, metrics=metrics)
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


def train_model(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_path: str | Path = MODEL_PATH,
    age_labels_csv: str | Path | None = None,
    stage_age_map: dict[int, float] | None = None,
    epochs: int = 15,
    batch_size: int = 16,
    image_size: tuple[int, int] = IMAGE_SIZE,
    roi_boxes_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Train MobileNetV2 from Aariz and save the model plus stage metadata."""
    tf = _tensorflow()
    splits = load_aariz_dataset(dataset_dir, age_labels_csv, roi_boxes_csv)
    stage_values = sorted({r["cvm_stage"] for r in splits["train"]})
    if len(stage_values) < 2:
        raise ValueError("At least two CVM stages are required to train a classifier.")
    age_training_records = splits["train"] + splits.get("valid", [])
    has_direct_age = bool(age_training_records) and all(r["skeletal_age_years"] is not None for r in age_training_records)
    mapped_age_map = {int(key): float(value) for key, value in (stage_age_map or {}).items()}
    has_mapped_age = bool(mapped_age_map) and all(r["cvm_stage"] in mapped_age_map for r in age_training_records)
    has_age = has_direct_age or has_mapped_age
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
                except (OSError, UnidentifiedImageError) as exc:
                    raise ValueError(f"Could not read image {record['image_path']}: {exc}") from exc
                label = {"cvm_stage": np.int32(stage_to_index[record["cvm_stage"]])}
                if has_age:
                    target = record["skeletal_age_years"] if has_direct_age else mapped_age_map[record["cvm_stage"]]
                    label["skeletal_age_years"] = np.float32(target)
                yield pixels, label

        output_signature = (
            tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
            {"cvm_stage": tf.TensorSpec(shape=(), dtype=tf.int32), **({"skeletal_age_years": tf.TensorSpec(shape=(), dtype=tf.float32)} if has_age else {})},
        )
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        if shuffle:
            dataset = dataset.shuffle(min(len(records), 1000), reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(splits["train"], shuffle=True)
    valid_ds = make_dataset(splits.get("valid", []), shuffle=False)
    model = build_mobilenetv2_model(stage_values, train_age_head=has_age)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss" if valid_ds else "loss", patience=4, restore_best_weights=True)]
    model.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=callbacks)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    metadata = {
        "stage_values": stage_values,
        "stage_titles": {str(r["cvm_stage"]): r["cvm_stage_title"] for r in splits["train"]},
        "image_size": list(image_size),
        "has_skeletal_age_head": has_age,
        "age_target_source": age_target_source,
        "stage_age_map": {str(key): value for key, value in mapped_age_map.items()} if has_mapped_age else None,
        "roi_model_path": roi_destination.name if has_roi_labels else None,
        "roi_training_source": "c2_c3_c4_box_annotations" if has_roi_labels else None,
        "dataset_dir": str(Path(dataset_dir).resolve()),
    }
    destination.with_suffix(".labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    clear_model_cache()
    return {"model_path": str(destination), "train_images": len(splits["train"]), "valid_images": len(splits.get("valid", [])), "cvm_stages": stage_values, "skeletal_age_head_trained": has_age, "age_target_source": age_target_source, "roi_localizer_trained": has_roi_labels, "roi_model_path": str(roi_destination) if has_roi_labels else None}


def clear_model_cache() -> None:
    global _loaded_model, _loaded_metadata, _loaded_roi_model, _model_generation
    _loaded_model = None
    _loaded_metadata = None
    _loaded_roi_model = None
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
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
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
    """Load model artifacts and execute a dummy call at application startup."""
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        return False
    model, metadata = _get_model()
    image_size = metadata["image_size"]
    dummy = np.zeros((1, image_size[0], image_size[1], 3), dtype=np.float32)
    with _inference_lock:
        model(dummy, training=False)
        roi_model = _get_roi_model(metadata)
        if roi_model is not None:
            roi_model(dummy, training=False)
    return True


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
        outputs = {
            name: np.asarray(value.numpy() if hasattr(value, "numpy") else value)
            for name, value in raw.items()
        } if isinstance(raw, dict) else {model.output_names[0]: np.asarray(raw.numpy() if hasattr(raw, "numpy") else raw)}
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
    predicted_age = float(np.asarray(age_output).reshape(-1)[0]) if age_output is not None else None
    result = {
        "predicted_skeletal_age_years": round(predicted_age, 1) if predicted_age is not None else None,
        "cvm_stage": stage_value,
        "cvm_stage_label": metadata.get("stage_titles", {}).get(str(stage_value), f"CVM-S{stage_value}"),
        "cva_status": "Not evaluated",
        "cvm_stage_confidence": round(float(probabilities[index]), 4),
        "cvm_stage_top2": top2,
        "confidence_calibration": "uncalibrated_softmax_scores",
        "roi_status": roi_status,
        "age_target_source": metadata.get("age_target_source"),
        "status": "complete" if predicted_age is not None else "age_labels_required",
        "message": (
            "Age head trained on synthetic ages mapped from CVM stages; this is not an observed or clinically validated age prediction."
            if metadata.get("age_target_source") == "synthetic_stage_mapping"
            else ("" if predicted_age is not None else "CVM stage predicted. Age-in-years labels or an explicit stage-age mapping are required to train the regression head.")
        ),
    }
    _store_prediction(image_hash, result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CLEFTSKEL-AI MobileNetV2 model on Aariz.")
    parser.add_argument("--train", action="store_true", help="Train and save the model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Output .keras model path.")
    parser.add_argument("--age-labels", help="Optional CSV with cephalogram_id,skeletal_age_years columns.")
    parser.add_argument("--stage-age-map", help="Optional JSON mapping CVM stage numbers to synthetic age targets, e.g. {\"1\": 9.0}.")
    parser.add_argument("--roi-boxes", help="Optional CSV with normalized C2-C4 ROI boxes: cephalogram_id,x_min,y_min,x_max,y_max.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.train:
        parser.error("Pass --train to begin model training.")
    age_map = json.loads(Path(args.stage_age_map).read_text(encoding="utf-8")) if args.stage_age_map else None
    print(json.dumps(train_model(args.dataset, args.output, args.age_labels, age_map, args.epochs, args.batch_size, roi_boxes_csv=args.roi_boxes), indent=2))
