"""Experimental CVM/age trainer isolated from the production baseline.

This iteration uses deterministic, class-balanced sampling with replacement to
give CVMI-1/2/3 more optimizer updates, while retaining the same MobileNetV2
multi-task outputs and mapped-age supervision. It writes only V2 artifacts by
default. It does not promise an accuracy improvement; compare its held-out
metrics with the unchanged baseline before considering any promotion.

Run from the repository root:
    python backend/model_v2.py --train --dataset "C:\\Users\\Bhumi\\Desktop\\Aariz"
    python backend/model_v2.py --evaluate --dataset "C:\\Users\\Bhumi\\Desktop\\Aariz"
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

try:  # Supports both `python backend/model_v2.py` and package imports.
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
TRAINING_SEED = 42


def _build_v2_model(stage_values: list[int], train_age_head: bool = True):
    """Build the multi-task MobileNetV2 experiment with regularized task heads."""
    tf = baseline._tensorflow()
    image_size = baseline.IMAGE_SIZE
    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = True
    freeze_before = max(0, len(backbone.layers) - 35)
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
    x = tf.keras.layers.Dropout(0.35, name="task_dropout")(x)
    outputs: dict[str, Any] = {
        "cvm_stage": tf.keras.layers.Dense(
            len(stage_values), activation="softmax", name="cvm_stage"
        )(x)
    }
    losses: dict[str, Any] = {"cvm_stage": "sparse_categorical_crossentropy"}
    metrics: dict[str, Any] = {"cvm_stage": ["accuracy"]}
    loss_weights: dict[str, float] = {"cvm_stage": 1.0}
    if train_age_head:
        outputs["skeletal_age_years"] = tf.keras.layers.Dense(
            1, name="skeletal_age_years"
        )(x)
        losses["skeletal_age_years"] = "mean_squared_error"
        metrics["skeletal_age_years"] = ["mean_absolute_error"]
        # Age targets are expressed in years; this keeps their MSE scale from
        # dominating the categorical objective while preserving age learning.
        loss_weights["skeletal_age_years"] = 0.01

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_cvm_age_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


def _make_v2_dataset(
    records: list[dict[str, Any]],
    stage_to_index: dict[int, int],
    age_map: dict[int, float],
    has_direct_age: bool,
    *,
    batch_size: int,
    training: bool,
    target_per_stage: int | None = None,
):
    """Create deterministic balanced train draws or a natural validation set."""
    tf = baseline._tensorflow()
    if not records:
        return None

    image_size = baseline.IMAGE_SIZE
    augmenter = baseline._training_image_augmenter() if training else None
    by_stage = {
        stage_index: [record for record in records if stage_to_index[record["cvm_stage"]] == stage_index]
        for stage_index in range(len(stage_to_index))
    }
    if training and any(not stage_records for stage_records in by_stage.values()):
        empty = [stage_index + 1 for stage_index, stage_records in by_stage.items() if not stage_records]
        raise ValueError(f"Balanced V2 sampling requires examples for each CVMI class; missing indices {empty}.")

    def read_record(record: dict[str, Any]):
        try:
            with Image.open(record["image_path"]) as source:
                rgb = source.convert("RGB").resize((image_size[1], image_size[0]))
                pixels = np.asarray(rgb, dtype=np.float32)
                if training and augmenter is not None:
                    pixels = augmenter.random_transform(pixels)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read image {record['image_path']}: {exc}") from exc
        stage_index = stage_to_index[record["cvm_stage"]]
        age = record["skeletal_age_years"] if has_direct_age else age_map[record["cvm_stage"]]
        labels = {
            "cvm_stage": np.int32(stage_index),
            "skeletal_age_years": np.asarray([age], dtype=np.float32),
        }
        return pixels, labels

    output_signature = (
        tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
        {
            "cvm_stage": tf.TensorSpec(shape=(), dtype=tf.int32),
            "skeletal_age_years": tf.TensorSpec(shape=(1,), dtype=tf.float32),
        },
    )

    if training:
        if not target_per_stage:
            raise ValueError("target_per_stage must be set for balanced V2 training.")
        # Separate RNG ensures repeatable class-balanced draws across runs.
        rng = np.random.default_rng(TRAINING_SEED)

        def balanced_generator():
            while True:
                epoch_records: list[dict[str, Any]] = []
                for stage_index in range(len(stage_to_index)):
                    examples = by_stage[stage_index]
                    sampled = rng.integers(0, len(examples), size=target_per_stage)
                    epoch_records.extend(examples[int(index)] for index in sampled)
                order = rng.permutation(len(epoch_records))
                for index in order:
                    yield read_record(epoch_records[int(index)])

        dataset = tf.data.Dataset.from_generator(balanced_generator, output_signature=output_signature)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def validation_generator():
        for record in records:
            yield read_record(record)

    dataset = tf.data.Dataset.from_generator(validation_generator, output_signature=output_signature)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_model_v2(
    dataset_dir: str | Path = baseline.DEFAULT_DATASET_DIR,
    output_path: str | Path = DEFAULT_V2_MODEL_PATH,
    metrics_output: str | Path = DEFAULT_V2_METRICS_PATH,
    age_labels_csv: str | Path | None = None,
    stage_age_map: dict[int, float] | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    allow_baseline_overwrite: bool = False,
) -> dict[str, Any]:
    """Train the isolated V2 experiment and evaluate its own checkpoint."""
    destination = Path(output_path).expanduser().resolve()
    if destination == baseline.MODEL_PATH.expanduser().resolve() and not allow_baseline_overwrite:
        raise ValueError(
            "Refusing to overwrite the stable baseline checkpoint; pass "
            "allow_baseline_overwrite=True only when that is intentional."
        )
    tf = baseline._set_global_training_seed(TRAINING_SEED)
    splits = baseline.load_aariz_dataset(dataset_dir, age_labels_csv)
    for split_name in ("train", "valid"):
        if not splits.get(split_name):
            raise ValueError(f"V2 training requires a non-empty {split_name} split.")
    stage_values = sorted({int(record["cvm_stage"]) for record in splits["train"]})
    if stage_values != [1, 2, 3, 4, 5, 6]:
        raise ValueError(f"V2 requires CVMI-1 through CVMI-6 in train; found {stage_values}.")

    training_records = splits["train"] + splits["valid"]
    has_direct_age = bool(training_records) and all(
        record.get("skeletal_age_years") is not None for record in training_records
    )
    age_map = {int(key): float(value) for key, value in (stage_age_map or {}).items()}
    if not has_direct_age and not age_map:
        age_map = baseline._load_stage_age_map()
    if not has_direct_age and any(record["cvm_stage"] not in age_map for record in training_records):
        raise ValueError(
            "V2 age regression needs complete observed age labels or a stage_age_map.json "
            "covering every stage in train and validation."
        )
    age_target_source = "observed_labels" if has_direct_age else "synthetic_stage_mapping"

    stage_to_index = {stage: index for index, stage in enumerate(stage_values)}
    class_records = {
        stage: [record for record in splits["train"] if record["cvm_stage"] == stage]
        for stage in stage_values
    }
    target_per_stage = int(math.ceil(max(map(len, class_records.values())) / batch_size) * batch_size)
    draws_per_epoch = len(stage_values) * target_per_stage
    train_ds = _make_v2_dataset(
        splits["train"], stage_to_index, age_map, has_direct_age,
        batch_size=batch_size, training=True, target_per_stage=target_per_stage,
    )
    valid_ds = _make_v2_dataset(
        splits["valid"], stage_to_index, age_map, has_direct_age,
        batch_size=batch_size, training=False,
    )

    model = _build_v2_model(stage_values, train_age_head=True)
    callbacks = baseline._checkpoint_callbacks(
        tf,
        destination,
        validation_available=True,
        checkpoint_monitor="val_cvm_macro_f1",
        classification_loss_monitor="val_cvm_stage_loss",
        macro_f1_dataset=valid_ds,
        num_classes=len(stage_values),
    )
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        steps_per_epoch=draws_per_epoch // batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    metadata = {
        "stage_values": stage_values,
        "stage_titles": {str(stage): f"CVMI-{stage}" for stage in stage_values},
        "image_size": list(baseline.IMAGE_SIZE),
        "has_skeletal_age_head": True,
        "skeletal_age_head_architecture": "parallel_dense_regression",
        "age_target_source": age_target_source,
        "stage_age_map": {str(key): value for key, value in age_map.items()} if not has_direct_age else None,
        "training_seed": TRAINING_SEED,
        "experiment": "deterministic_class_balanced_oversampling_v2",
        "balanced_sampling": {
            "method": "uniform_stage_sampling_with_replacement",
            "samples_per_stage_per_epoch": target_per_stage,
            "draws_per_epoch": draws_per_epoch,
            "age_head_receives_same_balanced_stage_distribution": True,
        },
        "task_head": {"dense_units": 128, "dropout": 0.35, "l2_kernel_regularization": 1e-4},
        "losses": {
            "cvm_stage": "sparse_categorical_crossentropy",
            "skeletal_age_years": "mean_squared_error",
        },
        "loss_weights": {"cvm_stage": 1.0, "skeletal_age_years": 0.01},
        "fine_tuning": {
            "backbone_top_layers_selected": 35,
            "batch_normalization_trainable": False,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "best_checkpoint_monitor": "val_cvm_macro_f1",
        },
        "roi_model_path": None,
        "roi_training_source": None,
        "dataset_dir": str(Path(dataset_dir).resolve()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evaluation = baseline.evaluate_saved_models(
        dataset_dir=dataset_dir,
        model_path=destination,
        age_labels_csv=age_labels_csv,
        metrics_output=metrics_output,
    )
    return {
        "status": "trained_and_evaluated",
        "model_path": str(destination),
        "metrics_path": str(Path(metrics_output).expanduser().resolve()),
        "train_images": len(splits["train"]),
        "valid_images": len(splits["valid"]),
        "test_images": len(splits.get("test", [])),
        "age_target_source": age_target_source,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated experimental CleftSkel-AI CVM model v2.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train V2 and evaluate validation/test splits.")
    action.add_argument("--evaluate", action="store_true", help="Evaluate the selected V2 checkpoint.")
    parser.add_argument("--dataset", default=str(baseline.DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument("--output", default=str(DEFAULT_V2_MODEL_PATH), help="V2 checkpoint destination (.keras).")
    parser.add_argument("--metrics-output", default=str(DEFAULT_V2_METRICS_PATH), help="V2 evaluation JSON destination.")
    parser.add_argument("--age-labels", help="Optional CSV containing complete observed skeletal ages.")
    parser.add_argument("--stage-age-map", help="Optional JSON stage-to-age proxy map; defaults to project stage_age_map.json.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-baseline-overwrite",
        action="store_true",
        help="Explicitly permit --output to target the stable model.py checkpoint path.",
    )
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    baseline_path = baseline.MODEL_PATH.expanduser().resolve()
    if output_path == baseline_path and not args.allow_baseline_overwrite:
        parser.error(
            "Refusing to target the stable baseline checkpoint. Use the default V2 path, "
            "or pass --allow-baseline-overwrite explicitly."
        )

    stage_age_map = None
    if args.stage_age_map:
        stage_age_map = json.loads(Path(args.stage_age_map).expanduser().read_text(encoding="utf-8"))
    if args.train:
        result = train_model_v2(
            dataset_dir=args.dataset,
            output_path=output_path,
            metrics_output=args.metrics_output,
            age_labels_csv=args.age_labels,
            stage_age_map=stage_age_map,
            epochs=args.epochs,
            batch_size=args.batch_size,
            allow_baseline_overwrite=args.allow_baseline_overwrite,
        )
    else:
        result = baseline.evaluate_saved_models(
            dataset_dir=args.dataset,
            model_path=output_path,
            age_labels_csv=args.age_labels,
            metrics_output=args.metrics_output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
