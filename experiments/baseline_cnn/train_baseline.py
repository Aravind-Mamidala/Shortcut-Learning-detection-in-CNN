#!/usr/bin/env python3
"""Train and evaluate a small CNN baseline on train and validation splits only."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_TO_LABEL = {"cow": 0, "horse": 1}
CLASS_NAMES = ("cow", "horse")
IMAGE_SIZE = (224, 224)
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 30
MAX_EPOCHS = 100


class CompositeDataset(Dataset):
    """Load image paths and class labels directly from the dataset metadata."""

    def __init__(self, dataset_dir: Path, rows: list[dict[str, str]]) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.dataset_dir / row["path"]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                if image.size != IMAGE_SIZE:
                    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot load dataset image {image_path}: {exc}") from exc
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, int(row["label"])


class SmallCNN(nn.Module):
    """Compact convolutional baseline suitable for CPU training."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(96, len(CLASS_NAMES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, start_dim=1))


def load_metadata(dataset_dir: Path) -> dict[str, list[dict[str, str]]]:
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    with metadata_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required_columns = {"path", "split", "class_name", "label"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise RuntimeError(
                f"Metadata is missing required columns: {sorted(missing_columns)}"
            )
        for row in reader:
            split = row["split"]
            if split not in rows_by_split:
                continue
            class_name = row["class_name"]
            if class_name not in CLASS_TO_LABEL:
                raise RuntimeError(
                    f"Unexpected class {class_name!r} in {metadata_path}"
                )
            expected_label = CLASS_TO_LABEL[class_name]
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid label for {class_name!r} at {row['path']!r}"
                ) from exc
            if label != expected_label:
                raise RuntimeError(
                    f"Metadata label mismatch for {class_name!r}: "
                    f"expected {expected_label}, got {label}"
                )
            image_path = dataset_dir / row["path"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{split} image listed in metadata is missing: {image_path}"
                )
            rows_by_split[split].append(row)

    for split, rows in rows_by_split.items():
        if not rows:
            raise RuntimeError(f"No {split} samples found in {metadata_path}")
        counts = Counter(row["class_name"] for row in rows)
        if set(counts) != set(CLASS_NAMES):
            raise RuntimeError(
                f"{split} must contain both classes; found {dict(counts)}"
            )
    return rows_by_split


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Cannot calculate epoch metrics for an empty split.")
    return total_loss / sample_count, total_correct / sample_count


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ),
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(
        epochs, [item["train_accuracy"] for item in history], label="Train"
    )
    axes[1].plot(
        epochs, [item["val_accuracy"] for item in history], label="Validation"
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels_device)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    metrics: dict[str, Any] = {
        "split": "val",
        "sample_count": total,
        "loss": total_loss / total,
        "accuracy": accuracy_score(targets, predictions),
        "precision_macro": precision_score(
            targets, predictions, labels=[0, 1], average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            targets, predictions, labels=[0, 1], average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            targets, predictions, labels=[0, 1], average="macro", zero_division=0
        ),
        "per_class": classification_report(
            targets,
            predictions,
            labels=[0, 1],
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, matrix


def save_evaluation(
    metrics: dict[str, Any], matrix: np.ndarray, output_dir: Path
) -> None:
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Validation confusion matrix",
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    saved_script = output_dir / "train_baseline.py"
    if script_path != saved_script:
        shutil.copy2(script_path, saved_script)

    set_reproducible_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")
    rows = load_metadata(dataset_dir)

    train_dataset = CompositeDataset(dataset_dir, rows["train"])
    val_dataset = CompositeDataset(dataset_dir, rows["val"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_label": CLASS_TO_LABEL,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break

    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("Training did not produce a best model checkpoint.")
    save_history(history, output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, matrix = evaluate_validation(model, val_loader, criterion, device)
    metrics["best_epoch"] = best_epoch
    metrics["seed"] = args.seed
    save_evaluation(metrics, matrix, output_dir)
    return {
        "history": history,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/pilot/dataset")
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.epochs <= MAX_EPOCHS:
        parser.error(f"--epochs must be between 1 and {MAX_EPOCHS}.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.patience < 1:
        parser.error("--patience must be positive.")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive.")
    return args


if __name__ == "__main__":
    result = train(parse_args())
    metrics = result["metrics"]
    print(
        "Best validation checkpoint: "
        f"epoch={metrics['best_epoch']} accuracy={metrics['accuracy']:.3f} "
        f"precision={metrics['precision_macro']:.3f} "
        f"recall={metrics['recall_macro']:.3f} f1={metrics['f1_macro']:.3f}"
    )
    print(f"Experiment outputs: {result['output_dir']}")
