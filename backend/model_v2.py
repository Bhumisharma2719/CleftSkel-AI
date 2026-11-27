"""Experimental reduced-stage CVM/age trainer, isolated from the six-stage baseline.

Default grouping:
  output 0, Early / Pre-Peak: CVMI-1 and CVMI-2
  output 1, Peak / Transitional: CVMI-3 and CVMI-4
  output 2, Post-Peak / Maturation Completion: CVMI-5 and CVMI-6

The project stage-age map is averaged within each group to create proxy targets.
Original Aariz train/valid/test partitions are retained without re-splitting.
Default outputs are mobilenetv2_v2.keras and evaluation_metrics_v2.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

try:  # Supports `python backend/model_v2.py` and `python -m backend.model_v2`.
    from . import model as baseline
except ImportError:
    import model as baseline


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_V2_MODEL_PATH = Path(
    os.getenv("CLEFTSKEL_MODEL_V2_PATH", ARTIFACT_DIR / "mobilenetv2_v2.keras")
)
DEFAULT_V2_METRICS_PATH = Path(
    os.getenv("CLEFTSKEL_METRICS_V2_PATH", ARTIFACT_DIR / "evaluation_metrics_v2.json")
)
DEFAULT_STAGE_GROUPS: dict[str, dict[str, Any]] = {
    "0": {"name": "Early / Pre-Peak", "stages": [1, 2]},
    "1": {"name": "Peak / Transitional", "stages": [3, 4]},
    "2": {"name": "Post-Peak / Maturation Completion", "stages": [5, 6]},
}
TRAINING_SEED = 42
IMAGE_SIZE = baseline.IMAGE_SIZE


def _validate_stage_groups(raw: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Normalize and validate a complete, non-overlapping 1..6 stage partition."""
    if not isinstance(raw, dict) or len(raw) < 2 or len(raw) > 6:
        raise ValueError("Stage groups must be a JSON object containing between 2 and 6 groups.")
    try:
        groups = {
            int(group_id): {
                "name": str(spec["name"]).strip(),
                "stages": [int(stage) for stage in spec["stages"]],
            }
            for group_id, spec in raw.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Each group must have a "name" and a "stages" list of CVMI values.') from exc
    expected_ids = list(range(len(groups)))
    if sorted(groups) != expected_ids:
        raise ValueError(f"Group IDs must be contiguous and start at 0; found {sorted(groups)}.")
    assigned = [stage for group in groups.values() for stage in group["stages"]]
    if any(not group["name"] or not group["stages"] for group in groups.values()):
        raise ValueError("Each stage group must have a non-empty name and at least one CVMI stage.")
    if sorted(assigned) != [1, 2, 3, 4, 5, 6] or len(set(assigned)) != 6:
        raise ValueError("Stage groups must assign CVMI-1 through CVMI-6 exactly once, with no overlap.")
    return groups


def _read_stage_groups(path: str | Path | None = None) -> dict[int, dict[str, Any]]:
    if path is None:
        return _validate_stage_groups(DEFAULT_STAGE_GROUPS)
    group_path = Path(path).expanduser()
    try:
        raw = json.loads(group_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read stage grouping JSON from {group_path}: {exc}") from exc
    return _validate_stage_groups(raw)


def _stage_to_group(groups: dict[int, dict[str, Any]]) -> dict[int, int]:
    return {
        stage: group_id
        for group_id, spec in groups.items()
        for stage in spec["stages"]
    }


def _group_age_targets(
    groups: dict[int, dict[str, Any]], stage_age_map: dict[int, float]
) -> dict[int, float]:
    """Assign each reduced class the arithmetic mean of its source-stage proxies."""
    return {
        group_id: float(np.mean([stage_age_map[stage] for stage in spec["stages"]]))
        for group_id, spec in groups.items()
    }


def _assert_no_split_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Fail closed if an image ID or resolved image path appears in multiple splits."""
    seen_ids: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    for split_name in ("train", "valid", "test"):
        for record in splits.get(split_name, []):
            image_id = str(record.get("image_id", "")).strip()
            image_path = str(Path(record["image_path"]).resolve()).casefold()
            if image_id:
                previous = seen_ids.get(image_id)
                if previous is not None and previous != split_name:
                    raise ValueError(f"Data leakage check failed: image ID {image_id!r} occurs in {previous} and {split_name}.")
                seen_ids[image_id] = split_name
            previous_path_split = seen_paths.get(image_path)
            if previous_path_split is not None and previous_path_split != split_name:
                raise ValueError(f"Data leakage check failed: image path is shared between {previous_path_split} and {split_name}: {image_path}")
            seen_paths[image_path] = split_name


def _build_v2_model(num_groups: int):
    """Build regularized MobileNetV2 with reduced-stage and proxy-age heads."""
    tf = baseline._tensorflow()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    # Fine-tune the final feature blocks only; frozen BatchNorm statistics are
    # less noisy for this small radiograph dataset.
    backbone.trainable = True
    unfreeze_tail = 35
    freeze_before = max(0, len(backbone.layers) - unfreeze_tail)
    for layer in backbone.layers[:freeze_before]:
        layer.trainable = False
    for layer in backbone.layers[freeze_before:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="task_features",
    )(x)
    x = tf.keras.layers.Dropout(0.40, name="task_dropout_v2")(x)
    outputs = {
        "cvm_stage": tf.keras.layers.Dense(num_groups, activation="softmax", name="cvm_stage")(x),
        "skeletal_age_years": tf.keras.layers.Dense(1, name="skeletal_age_years")(x),
    }
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_reduced_cvm_age_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss={
            "cvm_stage": "sparse_categorical_crossentropy",
            "skeletal_age_years": "mean_squared_error",
        },
        loss_weights={"cvm_stage": 1.0, "skeletal_age_years": 0.01},
        metrics={
            "cvm_stage": ["accuracy"],
            "skeletal_age_years": ["mean_absolute_error"],
        },
    )
    return model


def _smoothed_group_weights(labels: list[int], num_groups: int) -> dict[int, float]:
    """Use square-root inverse-frequency weights with conservative clipping."""
    counts = Counter(int(label) for label in labels)
    if set(counts) != set(range(num_groups)):
        missing = sorted(set(range(num_groups)) - set(counts))
        raise ValueError(f"Training split is missing reduced stage groups: {missing}.")
    sample_count = len(labels)
    return {
        group: float(np.clip(np.sqrt(sample_count / (num_groups * counts[group])), 0.5, 2.5))
        for group in range(num_groups)
    }


def _make_dataset(
    records: list[dict[str, Any]],
    stage_to_group: dict[int, int],
    group_age_map: dict[int, float],
    stage_age_map: dict[int, float],
    class_weights: dict[int, float],
    *,
    batch_size: int,
    training: bool,
):
    tf = baseline._tensorflow()
    augmenter = None
    if training:
        augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.08,
            horizontal_flip=False,
            fill_mode="nearest",
        )

    def generator():
        for record in records:
            try:
                with Image.open(record["image_path"]) as source:
                    rgb = source.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
                    pixels = np.asarray(rgb, dtype=np.float32)
                    if augmenter is not None:
                        pixels = augmenter.random_transform(pixels)
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Could not read image {record['image_path']}: {exc}") from exc

            group = stage_to_group[int(record["cvm_stage"])]
            # Group means are used for both training and evaluation to keep age
            # targets consistent with the reduced classification task.
            age = group_age_map[group]
            labels = {
                "cvm_stage": np.int32(group),
                "skeletal_age_years": np.asarray([age], dtype=np.float32),
            }
            if training:
                weights = {
                    "cvm_stage": np.float32(class_weights[group]),
                    "skeletal_age_years": np.float32(1.0),
                }
                yield pixels, labels, weights
            else:
                yield pixels, labels

    output_signature = (
        tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),
        {
            "cvm_stage": tf.TensorSpec(shape=(), dtype=tf.int32),
            "skeletal_age_years": tf.TensorSpec(shape=(1,), dtype=tf.float32),
        },
    )
    if training:
        output_signature = (*output_signature, {
            "cvm_stage": tf.TensorSpec(shape=(), dtype=tf.float32),
            "skeletal_age_years": tf.TensorSpec(shape=(), dtype=tf.float32),
        })
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if training:
        dataset = dataset.shuffle(
            min(len(records), 1000), seed=TRAINING_SEED, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _evaluate_split(
    model,
    records: list[dict[str, Any]],
    stage_to_group: dict[int, int],
    groups: dict[int, dict[str, Any]],
    group_age_map: dict[int, float],
) -> dict[str, Any]:
    num_groups = len(groups)
    confusion = np.zeros((num_groups, num_groups), dtype=np.int64)
    age_targets: list[float] = []
    age_predictions: list[float] = []
    for record in records:
        with Image.open(record["image_path"]) as source:
            rgb = source.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            pixels = np.asarray(rgb, dtype=np.float32)[None, ...]
        outputs = baseline._model_output_dict(model, model(pixels, training=False))
        probabilities = outputs["cvm_stage"][0]
        actual_group = stage_to_group[int(record["cvm_stage"])]
        predicted_group = int(np.argmax(probabilities))
        confusion[actual_group, predicted_group] += 1
        age_targets.append(float(group_age_map[actual_group]))
        age_predictions.append(float(np.asarray(outputs["skeletal_age_years"]).reshape(-1)[0]))

    supports = confusion.sum(axis=1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_supports = confusion.sum(axis=0)
    precision = np.divide(true_positive, predicted_supports, out=np.zeros(num_groups), where=predicted_supports > 0)
    recall = np.divide(true_positive, supports, out=np.zeros(num_groups), where=supports > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_groups), where=(precision + recall) > 0)
    total = int(confusion.sum())
    correct = int(true_positive.sum())
    expected_agreement = float(np.dot(supports, predicted_supports) / total**2) if total else 0.0
    observed_agreement = correct / total if total else 0.0
    kappa = None if not total or abs(1 - expected_agreement) < 1e-12 else (observed_agreement - expected_agreement) / (1 - expected_agreement)
    errors = np.asarray(age_predictions, dtype=np.float64) - np.asarray(age_targets, dtype=np.float64)
    per_group = {
        groups[group_id]["name"]: {
            "group_id": group_id,
            "source_cvmi_stages": groups[group_id]["stages"],
            "accuracy_percent": round(float(recall[group_id] * 100), 2) if supports[group_id] else None,
            "support": int(supports[group_id]),
        }
        for group_id in range(num_groups)
    }
    return {
        "sample_count": total,
        "overall_accuracy_percent": round(observed_agreement * 100, 2) if total else None,
        "balanced_accuracy_percent": round(float(np.mean(recall[supports > 0])) * 100, 2) if total else None,
        "macro_f1_percent": round(float(np.mean(f1)) * 100, 2) if total else None,
        "per_stage_accuracy": per_group,
        "cohen_kappa": round(float(kappa), 4) if kappa is not None else None,
        "skeletal_age_regression": {
            "status": "evaluated" if total else "not_available",
            "sample_count": total,
            "mae_years": round(float(np.mean(np.abs(errors))), 3) if total else None,
            "rmse_years": round(float(np.sqrt(np.mean(errors**2))), 3) if total else None,
            "target_source": "synthetic_grouped_stage_mapping",
            "interpretation": "Age errors compare against group-mean stage_age_map proxy targets, not observed clinical age.",
        },
    }


def _evaluate_v2(
    dataset_dir: str | Path,
    model_path: str | Path,
    metrics_output: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(model_path).expanduser().resolve()
    sidecar = checkpoint.with_suffix(".labels.json")
    if not checkpoint.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"V2 checkpoint and metadata are required: {checkpoint} / {sidecar}")
    splits = baseline.load_aariz_dataset(dataset_dir)
    _assert_no_split_overlap(splits)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    groups = _validate_stage_groups(metadata["stage_groups"])
    stage_to_group = _stage_to_group(groups)
    group_age_map = {int(key): float(value) for key, value in metadata["group_age_map"].items()}
    model = baseline._tensorflow().keras.models.load_model(checkpoint, compile=False)
    results = {
        "validation": _evaluate_split(model, splits.get("valid", []), stage_to_group, groups, group_age_map),
        "test": _evaluate_split(model, splits.get("test", []), stage_to_group, groups, group_age_map),
    }
    split_counts = {name: len(records) for name, records in splits.items()}
    result = {
        "model_path": str(checkpoint),
        "experiment": "reduced_cvm_groups_v2",
        "training_seed": metadata.get("training_seed", TRAINING_SEED),
        "stage_groups": {str(key): value for key, value in groups.items()},
        "stage_age_map": metadata.get("source_stage_age_map"),
        "group_age_map": {str(key): value for key, value in group_age_map.items()},
        "dataset_split_counts": split_counts,
        "split_integrity": "preserved original train/valid/test partitions; duplicate IDs and paths checked",
        "cvm": results,
        "age_metric_note": "MAE/RMSE use group-mean synthetic stage-age proxies, not observed skeletal ages.",
    }
    output = Path(metrics_output).expanduser().resolve()
    _assert_v2_paths(checkpoint, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metadata["evaluation_metrics"] = results
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return result


def _assert_v2_paths(checkpoint: Path, metrics: Path) -> None:
    """Protect both baseline files even when custom CLI paths are supplied."""
    baseline_checkpoint = baseline.MODEL_PATH.expanduser().resolve()
    baseline_metrics = (ARTIFACT_DIR / "evaluation_metrics.json").resolve()
    if checkpoint.resolve() == baseline_checkpoint:
        raise ValueError(f"V2 cannot overwrite the six-stage baseline checkpoint: {baseline_checkpoint}")
    if metrics.resolve() == baseline_metrics:
        raise ValueError(f"V2 cannot overwrite the baseline metrics JSON: {baseline_metrics}")


def train_model_v2(
    dataset_dir: str | Path = baseline.DEFAULT_DATASET_DIR,
    output_path: str | Path = DEFAULT_V2_MODEL_PATH,
    metrics_output: str | Path = DEFAULT_V2_METRICS_PATH,
    stage_groups: dict[int, dict[str, Any]] | None = None,
    stage_age_map_path: str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Train and evaluate the reduced-stage experiment without touching v1."""
    checkpoint = Path(output_path).expanduser().resolve()
    metrics_path = Path(metrics_output).expanduser().resolve()
    _assert_v2_paths(checkpoint, metrics_path)
    tf = baseline._set_global_training_seed(TRAINING_SEED)
    splits = baseline.load_aariz_dataset(dataset_dir)
    _assert_no_split_overlap(splits)
    for split_name in ("train", "valid"):
        if not splits.get(split_name):
            raise ValueError(f"V2 training requires a non-empty {split_name} split.")

    groups = _validate_stage_groups(stage_groups or DEFAULT_STAGE_GROUPS)
    stage_to_group = _stage_to_group(groups)
    source_age_map = baseline._load_stage_age_map(stage_age_map_path)
    if any(stage not in source_age_map for stage in stage_to_group):
        raise ValueError("The source stage-age map must contain proxy ages for CVMI-1 through CVMI-6.")
    group_age_map = _group_age_targets(groups, source_age_map)
    train_group_labels = [stage_to_group[int(record["cvm_stage"])] for record in splits["train"]]
    class_weights = _smoothed_group_weights(train_group_labels, len(groups))

    train_ds = _make_dataset(
        splits["train"], stage_to_group, group_age_map, source_age_map, class_weights,
        batch_size=batch_size, training=True,
    )
    valid_ds = _make_dataset(
        splits["valid"], stage_to_group, group_age_map, source_age_map, class_weights,
        batch_size=batch_size, training=False,
    )
    model = _build_v2_model(num_groups=len(groups))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    callbacks = _validation_callbacks(
        tf, checkpoint, valid_ds, num_groups=len(groups)
    )
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metadata = {
        "stage_values": list(range(len(groups))),
        "stage_titles": {str(key): value["name"] for key, value in groups.items()},
        "source_stage_groups": {str(key): value for key, value in groups.items()},
        "stage_groups": {str(key): value for key, value in groups.items()},
        "source_stage_age_map": {str(key): value for key, value in source_age_map.items()},
        "stage_age_map": {str(key): value for key, value in group_age_map.items()},
        "group_age_map": {str(key): value for key, value in group_age_map.items()},
        "has_skeletal_age_head": True,
        "skeletal_age_head_architecture": "parallel_dense_regression",
        "age_target_source": "synthetic_grouped_stage_mapping",
        "image_size": list(IMAGE_SIZE),
        "training_seed": TRAINING_SEED,
        "experiment": "reduced_cvm_groups_v2",
        "split_integrity": "original Aariz train/valid/test partitions preserved; overlap check passed",
        "class_weights": {str(key): value for key, value in class_weights.items()},
        "class_weight_formula": "sqrt(N / (K * group_count)), clipped to [0.5, 2.5]",
        "task_head": {"dense_units": 128, "dropout": 0.40, "l2_kernel_regularization": 1e-4},
        "losses": {"cvm_stage": "sparse_categorical_crossentropy", "skeletal_age_years": "mean_squared_error"},
        "loss_weights": {"cvm_stage": 1.0, "skeletal_age_years": 0.01},
        "fine_tuning": {
            "backbone": "MobileNetV2 ImageNet weights",
            "backbone_top_layers_selected": 35,
            "batch_normalization_trainable": False,
            "optimizer": "Adam",
            "learning_rate": 5e-5,
            "checkpoint_monitor": "val_cvm_balanced_accuracy",
            "reduce_lr_monitor": "val_cvm_stage_loss",
        },
        "augmentation": {
            "rotation_degrees": 8,
            "width_shift": 0.05,
            "height_shift": 0.05,
            "zoom": 0.08,
            "horizontal_flip": False,
        },
        "roi_training_source": None,
        "dataset_dir": str(Path(dataset_dir).resolve()),
    }
    checkpoint.with_suffix(".labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    evaluation = _evaluate_v2(dataset_dir, checkpoint, metrics_path)
    return {
        "status": "trained_and_evaluated",
        "model_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "dataset_split_counts": {key: len(value) for key, value in splits.items()},
        "stage_groups": metadata["stage_groups"],
        "group_age_map": metadata["group_age_map"],
        "evaluation": evaluation["cvm"],
    }


def _validation_callbacks(tf, checkpoint: Path, valid_ds, num_groups: int):
    """Select on balanced validation recall; also log accuracy and macro-F1."""
    class ReducedStageValidationMetrics(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            confusion = np.zeros((num_groups, num_groups), dtype=np.int64)
            for batch in valid_ds:
                images, labels = batch[0], batch[1]
                outputs = baseline._model_output_dict(self.model, self.model(images, training=False))
                predicted = np.argmax(outputs["cvm_stage"], axis=1).astype(np.int64)
                actual = np.asarray(labels["cvm_stage"].numpy(), dtype=np.int64).reshape(-1)
                np.add.at(confusion, (actual, predicted), 1)

            supports = confusion.sum(axis=1)
            tp = np.diag(confusion).astype(np.float64)
            pred_support = confusion.sum(axis=0)
            precision = np.divide(tp, pred_support, out=np.zeros(num_groups), where=pred_support > 0)
            recall = np.divide(tp, supports, out=np.zeros(num_groups), where=supports > 0)
            f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_groups), where=(precision + recall) > 0)
            accuracy = float(tp.sum() / supports.sum()) if supports.sum() else 0.0
            balanced_accuracy = float(np.mean(recall[supports > 0])) if np.any(supports > 0) else 0.0
            macro_f1 = float(np.mean(f1))
            logs["val_cvm_balanced_accuracy"] = balanced_accuracy
            logs["val_cvm_macro_f1"] = macro_f1
            print(
                f"Epoch {epoch + 1}: val_accuracy={accuracy:.4f}, "
                f"val_balanced_accuracy={balanced_accuracy:.4f}, val_macro_f1={macro_f1:.4f}",
                flush=True,
            )

    return [
        ReducedStageValidationMetrics(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint), monitor="val_cvm_balanced_accuracy", mode="max",
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_cvm_balanced_accuracy", mode="max", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_cvm_stage_loss", mode="min", factor=0.2, patience=5,
            min_lr=1e-7, cooldown=1, verbose=1,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated reduced-stage CVM MobileNetV2 experiment (V2).")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--train", action="store_true", help="Train V2 and evaluate its validation/test splits.")
    action.add_argument("--evaluate", action="store_true", help="Evaluate the selected V2 checkpoint.")
    parser.add_argument("--dataset", default=str(baseline.DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument("--output", default=str(DEFAULT_V2_MODEL_PATH), help="V2 .keras checkpoint path.")
    parser.add_argument("--metrics-output", default=str(DEFAULT_V2_METRICS_PATH), help="V2 metrics JSON path.")
    parser.add_argument("--stage-groups", help="Optional JSON mapping group IDs to names and CVMI member stages.")
    parser.add_argument("--stage-age-map", help="Optional source CVMI-to-age JSON; defaults to project stage_age_map.json.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-stage-groups", action="store_true", help="Print and exit with the default grouping.")
    args = parser.parse_args()

    if args.show_stage_groups:
        print(json.dumps({str(key): value for key, value in _read_stage_groups(args.stage_groups).items()}, indent=2))
        return
    if not args.train and not args.evaluate:
        parser.error("Pass --train or --evaluate (or use --show-stage-groups).")

    checkpoint = Path(args.output).expanduser().resolve()
    metrics_path = Path(args.metrics_output).expanduser().resolve()
    try:
        _assert_v2_paths(checkpoint, metrics_path)
    except ValueError as exc:
        parser.error(str(exc))

    groups = _read_stage_groups(args.stage_groups) if args.train else None
    if args.train:
        age_path = Path(args.stage_age_map).expanduser() if args.stage_age_map else None
        result = train_model_v2(
            dataset_dir=args.dataset,
            output_path=checkpoint,
            metrics_output=metrics_path,
            stage_groups=groups,
            stage_age_map_path=age_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        result = _evaluate_v2(args.dataset, checkpoint, metrics_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
