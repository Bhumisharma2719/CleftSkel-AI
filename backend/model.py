"""Aariz dataset loader, MobileNetV2 training, and inference for CLEFTSKEL-AI.

The Aariz CVM annotation files contain CVM stages, not skeletal age in years.
Age regression can use direct age labels, or explicitly opted-in mapped stage
targets. Mapped targets are synthetic proxies and are marked as such in output.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_DATASET_DIR = Path(os.getenv("CLEFTSKEL_DATASET_DIR", r"C:\Users\Bhumi\Desktop\Aariz"))
MODEL_PATH = Path(os.getenv("CLEFTSKEL_MODEL_PATH", Path(__file__).parent / "artifacts" / "mobilenetv2.keras"))
LABELS_PATH = MODEL_PATH.with_suffix(".labels.json")
IMAGE_SIZE = (224, 224)
_loaded_model = None
_loaded_metadata: dict[str, Any] | None = None


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


def load_aariz_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    age_labels_csv: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load matching image/CVM annotation pairs from Aariz train/valid/test.

    Returns a dict keyed by split. Each record includes `image_path`, numeric
    `cvm_stage` (1..6), its title, and optional `skeletal_age_years`.
    """
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Aariz dataset directory does not exist: {root}")
    age_by_id = _read_age_labels(age_labels_csv)
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


def train_model(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_path: str | Path = MODEL_PATH,
    age_labels_csv: str | Path | None = None,
    stage_age_map: dict[int, float] | None = None,
    epochs: int = 15,
    batch_size: int = 16,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, Any]:
    """Train MobileNetV2 from Aariz and save the model plus stage metadata."""
    tf = _tensorflow()
    splits = load_aariz_dataset(dataset_dir, age_labels_csv)
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

    def make_dataset(records, shuffle: bool):
        if not records:
            return None

        def generator():
            for record in records:
                try:
                    with Image.open(record["image_path"]) as image:
                        rgb = image.convert("RGB").resize((image_size[1], image_size[0]))
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
        "dataset_dir": str(Path(dataset_dir).resolve()),
    }
    destination.with_suffix(".labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    clear_model_cache()
    return {"model_path": str(destination), "train_images": len(splits["train"]), "valid_images": len(splits.get("valid", [])), "cvm_stages": stage_values, "skeletal_age_head_trained": has_age, "age_target_source": age_target_source}


def clear_model_cache() -> None:
    global _loaded_model, _loaded_metadata
    _loaded_model = None
    _loaded_metadata = None


def _get_model():
    global _loaded_model, _loaded_metadata
    if _loaded_model is not None:
        return _loaded_model, _loaded_metadata
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train it with `python backend/model.py --train`.")
    tf = _tensorflow()
    _loaded_model = tf.keras.models.load_model(MODEL_PATH)
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Model metadata not found at {LABELS_PATH}.")
    _loaded_metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return _loaded_model, _loaded_metadata


def predict_from_image(image_bytes: bytes) -> dict[str, Any]:
    """Run an uploaded image through the trained model and map outputs."""
    model, metadata = _get_model()
    tf = _tensorflow()
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB").resize((metadata["image_size"][1], metadata["image_size"][0]))
            pixels = np.asarray(image, dtype=np.float32)[None, ...]
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc

    raw = model.predict(pixels, verbose=0)
    outputs = raw if isinstance(raw, dict) else {model.output_names[0]: raw}
    probabilities = np.asarray(outputs["cvm_stage"])[0]
    index = int(np.argmax(probabilities))
    stage_value = int(metadata["stage_values"][index])
    age_output = outputs.get("skeletal_age_years")
    predicted_age = float(np.asarray(age_output).reshape(-1)[0]) if age_output is not None else None
    return {
        "predicted_skeletal_age_years": round(predicted_age, 1) if predicted_age is not None else None,
        "cvm_stage": stage_value,
        "cvm_stage_label": metadata.get("stage_titles", {}).get(str(stage_value), f"CVM-S{stage_value}"),
        "cva_status": "Not evaluated",
        "cvm_stage_confidence": round(float(probabilities[index]), 4),
        "age_target_source": metadata.get("age_target_source"),
        "status": "complete" if predicted_age is not None else "age_labels_required",
        "message": (
            "Age head trained on synthetic ages mapped from CVM stages; this is not an observed or clinically validated age prediction."
            if metadata.get("age_target_source") == "synthetic_stage_mapping"
            else ("" if predicted_age is not None else "CVM stage predicted. Age-in-years labels or an explicit stage-age mapping are required to train the regression head.")
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CLEFTSKEL-AI MobileNetV2 model on Aariz.")
    parser.add_argument("--train", action="store_true", help="Train and save the model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Output .keras model path.")
    parser.add_argument("--age-labels", help="Optional CSV with cephalogram_id,skeletal_age_years columns.")
    parser.add_argument("--stage-age-map", help="Optional JSON mapping CVM stage numbers to synthetic age targets, e.g. {\"1\": 9.0}.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.train:
        parser.error("Pass --train to begin model training.")
    age_map = json.loads(Path(args.stage_age_map).read_text(encoding="utf-8")) if args.stage_age_map else None
    print(json.dumps(train_model(args.dataset, args.output, args.age_labels, age_map, args.epochs, args.batch_size), indent=2))
