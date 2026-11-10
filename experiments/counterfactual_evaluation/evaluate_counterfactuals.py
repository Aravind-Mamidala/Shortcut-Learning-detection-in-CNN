#!/usr/bin/env python3
"""Evaluate the frozen baseline on counterfactual test composites only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_baseline import CLASS_NAMES, CLASS_TO_LABEL, IMAGE_SIZE, SmallCNN


BACKGROUND_CATEGORIES = ("grass", "sand", "snow")
PREFERRED_BACKGROUND = {"cow": "grass", "horse": "sand"}
EXPECTED_TEST_IMAGES = 48
EXPECTED_SOURCES_PER_CLASS = 8
EXPECTED_VARIANTS_PER_GROUP = 3


class TestCompositeDataset(Dataset):
    """Load only test metadata rows using the baseline's image preprocessing."""

    def __init__(self, dataset_dir: Path, rows: list[dict[str, str]]) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        row = self.rows[index]
        image_path = self.dataset_dir / row["path"]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                if image.size != IMAGE_SIZE:
                    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot load test image {image_path}: {exc}") from exc
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hashes(dataset_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(dataset_dir).as_posix(): sha256_file(path)
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file()
    }


def load_test_rows(dataset_dir: Path) -> list[dict[str, str]]:
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    required = {
        "path",
        "split",
        "class_name",
        "label",
        "source_image_id",
        "source_annotation_id",
        "counterfactual_group_id",
        "background_category",
    }
    test_rows: list[dict[str, str]] = []
    with metadata_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"Metadata is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            if row["split"] != "test":
                continue
            class_name = row["class_name"]
            if class_name not in CLASS_TO_LABEL:
                raise RuntimeError(f"Unexpected test class: {class_name!r}")
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid test label in row: {row}") from exc
            if label != CLASS_TO_LABEL[class_name]:
                raise RuntimeError(
                    f"Test label mismatch for {class_name}: {label}"
                )
            if row["background_category"] not in BACKGROUND_CATEGORIES:
                raise RuntimeError(
                    f"Unexpected test background: {row['background_category']!r}"
                )
            image_path = dataset_dir / row["path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Test image is missing: {image_path}")
            test_rows.append(row)

    if len(test_rows) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TEST_IMAGES} test composites, "
            f"found {len(test_rows)}."
        )
    validate_test_groups(test_rows)
    return test_rows


def validate_test_groups(rows: list[dict[str, str]]) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["counterfactual_group_id"]].append(row)
        source_splits[row["source_image_id"]].add(row["split"])
    if any(splits != {"test"} for splits in source_splits.values()):
        raise RuntimeError("A source image has variants outside the test split.")

    source_counts = Counter(
        group[0]["class_name"] for group in groups.values() if group
    )
    if source_counts != Counter(
        {name: EXPECTED_SOURCES_PER_CLASS for name in CLASS_NAMES}
    ):
        raise RuntimeError(
            f"Unexpected test source groups by class: {dict(source_counts)}"
        )
    for group_id, variants in groups.items():
        if len(variants) != EXPECTED_VARIANTS_PER_GROUP:
            raise RuntimeError(
                f"Test group {group_id} has {len(variants)} variants; "
                f"expected {EXPECTED_VARIANTS_PER_GROUP}."
            )
        if {row["background_category"] for row in variants} != set(
            BACKGROUND_CATEGORIES
        ):
            raise RuntimeError(
                f"Test group {group_id} does not contain grass, sand, and snow."
            )
        for field in (
            "source_image_id",
            "source_annotation_id",
            "class_name",
            "label",
            "split",
        ):
            if len({row[field] for row in variants}) != 1:
                raise RuntimeError(
                    f"Test group {group_id} has inconsistent {field} values."
                )


def run_inference(
    model: SmallCNN, loader: DataLoader
) -> list[dict[str, Any]]:
    model.eval()
    output: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, rows in loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            for index in range(len(rows["path"])):
                true_label = int(rows["label"][index])
                predicted_label = int(predicted[index])
                output.append(
                    {
                        "path": rows["path"][index],
                        "split": rows["split"][index],
                        "source_image_id": rows["source_image_id"][index],
                        "source_annotation_id": rows["source_annotation_id"][index],
                        "counterfactual_group_id": rows[
                            "counterfactual_group_id"
                        ][index],
                        "class_name": rows["class_name"][index],
                        "true_label": true_label,
                        "background": rows["background_category"][index],
                        "predicted_label": predicted_label,
                        "predicted_class": CLASS_NAMES[predicted_label],
                        "confidence": float(confidence[index]),
                        "correct": predicted_label == true_label,
                    }
                )
    return output


def calculate_metrics(
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    y_true = [item["true_label"] for item in predictions]
    y_pred = [item["predicted_label"] for item in predictions]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    overall = {
        "split": "test",
        "sample_count": len(predictions),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
        ),
        "per_class": {},
    }
    for label, class_name in enumerate(CLASS_NAMES):
        indexes = [index for index, value in enumerate(y_true) if value == label]
        overall["per_class"][class_name] = {
            "sample_count": len(indexes),
            "accuracy": sum(y_pred[index] == label for index in indexes)
            / len(indexes),
            "precision": precision_score(
                y_true,
                y_pred,
                labels=[label],
                average="micro",
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                labels=[label],
                average="micro",
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                labels=[label],
                average="micro",
                zero_division=0,
            ),
        }

    background_rows: list[dict[str, Any]] = []
    for category in BACKGROUND_CATEGORIES:
        subset = [item for item in predictions if item["background"] == category]
        background_rows.append(
            {
                "animal_class": "all",
                "background": category,
                "sample_count": len(subset),
                "accuracy": accuracy_score(
                    [item["true_label"] for item in subset],
                    [item["predicted_label"] for item in subset],
                ),
            }
        )
        for class_name in CLASS_NAMES:
            class_subset = [
                item for item in subset if item["class_name"] == class_name
            ]
            background_rows.append(
                {
                    "animal_class": class_name,
                    "background": category,
                    "sample_count": len(class_subset),
                    "accuracy": accuracy_score(
                        [item["true_label"] for item in class_subset],
                        [item["predicted_label"] for item in class_subset],
                    ),
                }
            )
    by_background = {
        category: next(
            row["accuracy"]
            for row in background_rows
            if row["animal_class"] == "all" and row["background"] == category
        )
        for category in BACKGROUND_CATEGORIES
    }
    return overall, by_background, matrix, background_rows


def summarize_counterfactuals(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        groups[item["counterfactual_group_id"]].append(item)

    unchanged = 0
    changed = 0
    all_correct = 0
    all_incorrect = 0
    pairwise_comparisons = 0
    pairwise_prediction_flips = 0
    per_class: dict[str, Any] = {}
    preferred_association: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        class_groups = [
            variants
            for variants in groups.values()
            if variants[0]["class_name"] == class_name
        ]
        preferred = PREFERRED_BACKGROUND[class_name]
        preferred_rows = [
            item for item in predictions
            if item["class_name"] == class_name and item["background"] == preferred
        ]
        alternative_rows = [
            item for item in predictions
            if item["class_name"] == class_name and item["background"] != preferred
        ]
        preferred_association[class_name] = {
            "preferred_training_background": preferred,
            "preferred_variant_accuracy": accuracy_score(
                [item["true_label"] for item in preferred_rows],
                [item["predicted_label"] for item in preferred_rows],
            ),
            "alternative_variant_accuracy": accuracy_score(
                [item["true_label"] for item in alternative_rows],
                [item["predicted_label"] for item in alternative_rows],
            ),
            "preferred_variant_predicted_as_true_class": sum(
                item["predicted_label"] == item["true_label"]
                for item in preferred_rows
            ),
            "preferred_variant_count": len(preferred_rows),
            "preferred_variant_true_class_prediction_rate": sum(
                item["predicted_label"] == item["true_label"]
                for item in preferred_rows
            ) / len(preferred_rows),
            "alternative_variant_predicted_as_true_class": sum(
                item["predicted_label"] == item["true_label"]
                for item in alternative_rows
            ),
            "alternative_variant_count": len(alternative_rows),
            "alternative_variant_true_class_prediction_rate": sum(
                item["predicted_label"] == item["true_label"]
                for item in alternative_rows
            ) / len(alternative_rows),
        }
        class_changed = 0
        class_unchanged = 0
        class_all_correct = 0
        class_all_incorrect = 0
        for variants in class_groups:
            variants.sort(key=lambda item: BACKGROUND_CATEGORIES.index(item["background"]))
            outputs = [item["predicted_label"] for item in variants]
            if len(set(outputs)) == 1:
                unchanged += 1
                class_unchanged += 1
            else:
                changed += 1
                class_changed += 1
            if all(item["correct"] for item in variants):
                all_correct += 1
                class_all_correct += 1
            if all(not item["correct"] for item in variants):
                all_incorrect += 1
                class_all_incorrect += 1
            for first in range(len(outputs)):
                for second in range(first + 1, len(outputs)):
                    pairwise_comparisons += 1
                    pairwise_prediction_flips += outputs[first] != outputs[second]
        per_class[class_name] = {
            "source_groups": len(class_groups),
            "groups_with_identical_predictions": class_unchanged,
            "groups_with_prediction_changes": class_changed,
            "groups_correct_on_all_variants": class_all_correct,
            "groups_incorrect_on_all_variants": class_all_incorrect,
        }

    group_count = len(groups)
    summary = {
        "test_source_group_count": group_count,
        "groups_with_identical_predictions": unchanged,
        "groups_with_at_least_one_prediction_change": changed,
        "group_prediction_change_rate": changed / group_count,
        "pairwise_variant_comparisons": pairwise_comparisons,
        "pairwise_prediction_flip_count": pairwise_prediction_flips,
        "pairwise_prediction_flip_rate": (
            pairwise_prediction_flips / pairwise_comparisons
        ),
        "groups_correct_across_all_three_variants": all_correct,
        "groups_incorrect_across_all_three_variants": all_incorrect,
        "groups_with_mixed_correctness": group_count - all_correct - all_incorrect,
        "per_class": per_class,
        "preferred_training_background_association": preferred_association,
        "interpretation_note": (
            "These are objective prediction and accuracy comparisons across "
            "background variants; background sensitivity alone is not proof "
            "of shortcut learning."
        ),
    }
    return summary


def save_outputs(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    overall: dict[str, Any],
    by_background: dict[str, Any],
    matrix: np.ndarray,
    background_rows: list[dict[str, Any]],
    counterfactual_summary: dict[str, Any],
    checkpoint_hash: str,
    dataset_hashes_before: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    script_copy = output_dir / "evaluate_counterfactuals.py"
    shutil.copy2(Path(__file__).resolve(), script_copy)
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fields = [
            "path",
            "split",
            "source_image_id",
            "source_annotation_id",
            "counterfactual_group_id",
            "class_name",
            "true_label",
            "background",
            "predicted_label",
            "predicted_class",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    with (output_dir / "background_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fields = ["animal_class", "background", "sample_count", "accuracy"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(background_rows)
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])

    overall["accuracy_by_background"] = by_background
    overall["evaluated_splits"] = ["test"]
    overall["checkpoint_sha256"] = checkpoint_hash
    overall["dataset_file_count"] = len(dataset_hashes_before)
    overall["test_composite_count"] = len(predictions)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    (output_dir / "counterfactual_summary.json").write_text(
        json.dumps(counterfactual_summary, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    values = [by_background[name] for name in BACKGROUND_CATEGORIES]
    bars = axis.bar(BACKGROUND_CATEGORIES, values, color="#548235")
    axis.set(
        ylim=(0, 1),
        ylabel="Accuracy",
        xlabel="Background category",
        title="Test accuracy by background",
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "background_accuracy.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Test confusion matrix",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_hash_before = sha256_file(checkpoint_path)
    dataset_hashes_before = dataset_hashes(dataset_dir)
    test_rows = load_test_rows(dataset_dir)
    loader = DataLoader(
        TestCompositeDataset(dataset_dir, test_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("class_to_label") != CLASS_TO_LABEL:
        raise RuntimeError("Checkpoint class labels do not match test metadata.")
    if tuple(checkpoint.get("image_size", ())) != IMAGE_SIZE:
        raise RuntimeError("Checkpoint input dimensions do not match the baseline.")
    model = SmallCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device("cpu"))
    model.eval()
    predictions = run_inference(model, loader)
    if len(predictions) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Evaluated {len(predictions)} images, expected {EXPECTED_TEST_IMAGES}."
        )

    overall, by_background, matrix, background_rows = calculate_metrics(predictions)
    counterfactual_summary = summarize_counterfactuals(predictions)
    if (
        checkpoint_hash_before != sha256_file(checkpoint_path)
        or dataset_hashes_before != dataset_hashes(dataset_dir)
    ):
        raise RuntimeError("Frozen checkpoint or dataset files changed during evaluation.")

    save_outputs(
        output_dir,
        predictions,
        overall,
        by_background,
        matrix,
        background_rows,
        counterfactual_summary,
        checkpoint_hash_before,
        dataset_hashes_before,
    )
    return {
        "metrics": overall,
        "counterfactual_summary": counterfactual_summary,
        "confusion_matrix": matrix.tolist(),
        "output_dir": output_dir,
        "checkpoint_hash": checkpoint_hash_before,
        "dataset_file_count": len(dataset_hashes_before),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data/pilot/dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "experiments/baseline_cnn/best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/counterfactual_evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["counterfactual_summary"], indent=2))
    print(f"Counterfactual evaluation outputs: {result['output_dir']}")
