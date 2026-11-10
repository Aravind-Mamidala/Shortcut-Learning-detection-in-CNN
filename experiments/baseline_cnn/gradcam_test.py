#!/usr/bin/env python3
"""Create Grad-CAM visualizations for test counterfactuals using a frozen CNN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preparation import decode_mask
from train_baseline import CLASS_NAMES, CLASS_TO_LABEL, IMAGE_SIZE, SmallCNN


BACKGROUND_CATEGORIES = ("grass", "sand", "snow")
EXPECTED_TEST_IMAGES = 48
IMAGE_SIDE = IMAGE_SIZE[0]
ANIMAL_MAX_SIDE = int(IMAGE_SIDE * 0.78)
ANIMAL_BOTTOM_MARGIN = int(IMAGE_SIDE * 0.06)


class TestImages(Dataset):
    def __init__(self, dataset_dir: Path, rows: list[dict[str, str]]) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        row = self.rows[index]
        path = self.dataset_dir / row["path"]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                if image.size != IMAGE_SIZE:
                    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Unable to read test composite {path}: {exc}") from exc
        return torch.from_numpy(array).permute(2, 0, 1).contiguous(), row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataset(dataset_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(dataset_dir).as_posix(): sha256_file(path)
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file()
    }


def load_test_rows(dataset_dir: Path) -> list[dict[str, str]]:
    metadata = dataset_dir / "metadata.csv"
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
    rows: list[dict[str, str]] = []
    with metadata.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise RuntimeError(
                f"Dataset metadata is missing columns: {sorted(missing_columns)}"
            )
        for row in reader:
            if row["split"] != "test":
                continue
            if row["class_name"] not in CLASS_TO_LABEL:
                raise RuntimeError(f"Unexpected test class: {row['class_name']}")
            if int(row["label"]) != CLASS_TO_LABEL[row["class_name"]]:
                raise RuntimeError(f"Incorrect test label in row: {row}")
            if row["background_category"] not in BACKGROUND_CATEGORIES:
                raise RuntimeError(f"Unexpected test background: {row}")
            if not (dataset_dir / row["path"]).is_file():
                raise FileNotFoundError(f"Test composite is missing: {row['path']}")
            rows.append(row)
    if len(rows) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TEST_IMAGES} test images; found {len(rows)}."
        )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["counterfactual_group_id"]].append(row)
    if len(grouped) != 16:
        raise RuntimeError(f"Expected 16 counterfactual source groups; found {len(grouped)}.")
    for group_id, variants in grouped.items():
        if len(variants) != 3 or {
            row["background_category"] for row in variants
        } != set(BACKGROUND_CATEGORIES):
            raise RuntimeError(
                f"Test group {group_id} must have exactly grass, sand, and snow."
            )
        for key in ("source_image_id", "source_annotation_id", "class_name", "label"):
            if len({row[key] for row in variants}) != 1:
                raise RuntimeError(f"Test group {group_id} has inconsistent {key}.")
    return rows


def load_test_annotations(
    annotation_zip: Path, rows: list[dict[str, str]]
) -> dict[tuple[int, int], dict[str, Any]]:
    wanted = {
        (int(row["source_image_id"]), int(row["source_annotation_id"]))
        for row in rows
    }
    found: dict[tuple[int, int], dict[str, Any]] = {}
    with zipfile.ZipFile(annotation_zip) as archive:
        for split_name in ("train2017", "val2017"):
            member = f"annotations/instances_{split_name}.json"
            if member not in archive.namelist():
                raise RuntimeError(f"COCO archive is missing {member}")
            with archive.open(member) as source:
                document = json.load(source)
            categories = {
                category["id"]: category["name"]
                for category in document["categories"]
            }
            images = {image["id"]: image for image in document["images"]}
            for annotation in document["annotations"]:
                key = (annotation["image_id"], annotation["id"])
                if key not in wanted:
                    continue
                class_name = categories.get(annotation["category_id"])
                if class_name != next(
                    row["class_name"]
                    for row in rows
                    if (int(row["source_image_id"]), int(row["source_annotation_id"]))
                    == key
                ):
                    raise RuntimeError(
                        f"COCO annotation {key} has a class inconsistent with metadata."
                    )
                found[key] = {
                    **annotation,
                    "image_info": images[annotation["image_id"]],
                }
            del document
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(
            f"COCO segmentation annotations are missing for {len(missing)} "
            f"test source(s): {sorted(missing)[:5]}"
        )
    return found


def composite_animal_mask(
    annotation: dict[str, Any], source_image_path: Path
) -> np.ndarray:
    """Map the COCO segmentation mask using the dataset compositing transform."""
    with Image.open(source_image_path) as source:
        width, height = source.size
    if (
        width != annotation["image_info"]["width"]
        or height != annotation["image_info"]["height"]
    ):
        raise RuntimeError(
            f"COCO source dimensions mismatch for {source_image_path}: "
            f"actual {(width, height)}, metadata "
            f"{(annotation['image_info']['width'], annotation['image_info']['height'])}"
        )
    decoded = decode_mask(annotation, height, width)
    ys, xs = np.nonzero(decoded)
    if not len(xs):
        raise RuntimeError(f"COCO annotation {annotation['id']} has an empty mask.")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    alpha = Image.fromarray(decoded).crop((left, top, right, bottom))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))
    alpha.thumbnail(
        (ANIMAL_MAX_SIDE, ANIMAL_MAX_SIDE), Image.Resampling.LANCZOS
    )
    x = (IMAGE_SIDE - alpha.width) // 2
    y = max(0, IMAGE_SIDE - alpha.height - ANIMAL_BOTTOM_MARGIN)
    composite_mask = Image.new("L", IMAGE_SIZE, 0)
    composite_mask.paste(alpha, (x, y))
    return np.asarray(composite_mask, dtype=np.uint8) >= 128


def create_gradcam(model: SmallCNN, image: torch.Tensor) -> tuple[np.ndarray, int, float]:
    layer = model.features[9]
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        captured["activation"] = output

    handle = layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image.unsqueeze(0))
        activation = captured.get("activation")
        if activation is None:
            raise RuntimeError("Grad-CAM target layer did not produce activations.")
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())
        gradient = torch.autograd.grad(logits[0, predicted_class], activation)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        minimum = cam.min()
        maximum = cam.max()
        if float((maximum - minimum).item()) > 0:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)
        return cam.detach().cpu().numpy(), predicted_class, confidence
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def save_overlay(
    image_path: Path, cam: np.ndarray, destination: Path
) -> Image.Image:
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    if original.size != IMAGE_SIZE:
        original = original.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    heatmap = (plt.get_cmap("jet")(cam)[..., :3] * 255).astype(np.uint8)
    color = Image.fromarray(heatmap, mode="RGB")
    overlay = Image.blend(original, color, alpha=0.42)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(destination, format="PNG")
    return overlay


def save_comparison_sheet(
    group_id: str,
    variants: list[dict[str, Any]],
    output_path: Path,
) -> None:
    variants.sort(key=lambda item: BACKGROUND_CATEGORIES.index(item["background"]))
    panel_width, panel_height, header_height = IMAGE_SIDE, IMAGE_SIDE, 54
    sheet = Image.new(
        "RGB",
        (panel_width * len(variants), panel_height + header_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, variant in enumerate(variants):
        with Image.open(variant["overlay_path"]) as image:
            sheet.paste(image.convert("RGB"), (index * panel_width, header_height))
        title = (
            f"{variant['background']} | pred {variant['predicted_class']} "
            f"({variant['confidence']:.3f})"
        )
        draw.text((index * panel_width + 5, 5), title, fill="black")
    draw.text((5, 31), group_id, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def localization_summary(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_background: dict[str, Any] = {}
    by_class_background: dict[str, Any] = {}
    for category in BACKGROUND_CATEGORIES:
        rows = [item for item in predictions if item["background"] == category]
        inside_values = [item["animal_activation_percent"] for item in rows]
        by_background[category] = {
            "sample_count": len(rows),
            "mean_activation_inside_animal_percent": float(np.mean(inside_values)),
            "mean_activation_outside_animal_percent": float(
                np.mean([item["background_activation_percent"] for item in rows])
            ),
        }
        by_class_background[category] = {}
        for class_name in CLASS_NAMES:
            subset = [
                item
                for item in rows
                if item["true_class"] == class_name
            ]
            by_class_background[category][class_name] = {
                "sample_count": len(subset),
                "mean_activation_inside_animal_percent": float(
                    np.mean([item["animal_activation_percent"] for item in subset])
                ),
                "mean_activation_outside_animal_percent": float(
                    np.mean(
                        [item["background_activation_percent"] for item in subset]
                    )
                ),
            }
    return {
        "method": (
            "COCO segmentation polygons/RLE were decoded and mapped to each "
            "composite using the same animal bounding-box crop, thumbnail scale, "
            "centering, and bottom offset as dataset preparation. The binary "
            "animal region thresholds the transformed, blurred alpha mask at 128."
        ),
        "activation_measure": (
            "Sum of positive, min-max-normalized predicted-class Grad-CAM values "
            "within or outside the binary animal region, expressed as a percentage."
        ),
        "by_background": by_background,
        "by_class_and_background": by_class_background,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    annotation_zip = args.annotations.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Grad-CAM output directory already exists: {output_dir}")
    for path in (checkpoint_path, annotation_zip):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input is missing: {path}")

    checkpoint_hash_before = sha256_file(checkpoint_path)
    dataset_hashes_before = sha256_dataset(dataset_dir)
    rows = load_test_rows(dataset_dir)
    annotations = load_test_annotations(annotation_zip, rows)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_to_label") != CLASS_TO_LABEL:
        raise RuntimeError("Checkpoint label mapping does not match the baseline.")
    if tuple(checkpoint.get("image_size", ())) != IMAGE_SIZE:
        raise RuntimeError("Checkpoint image size does not match the baseline.")
    model = SmallCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device("cpu"))
    model.eval()

    loader = DataLoader(
        TestImages(dataset_dir, rows),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[dict[str, Any]] = []
    comparison_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_root = output_dir / "overlays"
    comparison_root = output_dir / "counterfactual_comparisons"
    copied_script = output_dir / "gradcam_test.py"
    copied_script.write_bytes(Path(__file__).read_bytes())

    for image, row_batch in loader:
        row = {key: values[0] for key, values in row_batch.items()}
        cam, predicted_label, confidence = create_gradcam(model, image[0])
        image_path = dataset_dir / row["path"]
        annotation_key = (
            int(row["source_image_id"]),
            int(row["source_annotation_id"]),
        )
        annotation = annotations[annotation_key]
        source_info = annotation["image_info"]
        # COCO stores separate train2017/val2017 files with the same basename.
        source_candidates = [
            args.raw_dir.resolve() / "coco" / "images" / split / source_info["file_name"]
            for split in ("train2017", "val2017")
        ]
        source_path = next((path for path in source_candidates if path.is_file()), None)
        if source_path is None:
            raise FileNotFoundError(
                f"COCO source image for annotation {annotation_key} was not found."
            )
        animal_mask = composite_animal_mask(annotation, source_path)
        activation_total = float(cam.sum())
        if activation_total <= 0:
            animal_percent = 0.0
            background_percent = 0.0
        else:
            animal_percent = float(cam[animal_mask].sum() / activation_total * 100)
            background_percent = float(cam[~animal_mask].sum() / activation_total * 100)

        group_id = row["counterfactual_group_id"]
        overlay_path = overlay_root / group_id / f"{row['background_category']}.png"
        save_overlay(image_path, cam, overlay_path)
        prediction = {
            "source_image_id": row["source_image_id"],
            "source_annotation_id": row["source_annotation_id"],
            "counterfactual_group_id": group_id,
            "background": row["background_category"],
            "true_class": row["class_name"],
            "true_label": int(row["label"]),
            "predicted_label": predicted_label,
            "predicted_class": CLASS_NAMES[predicted_label],
            "confidence": confidence,
            "correct": predicted_label == int(row["label"]),
            "animal_activation_percent": animal_percent,
            "background_activation_percent": background_percent,
            "image_path": row["path"],
            "overlay_path": overlay_path.relative_to(output_dir).as_posix(),
        }
        predictions.append(prediction)
        comparison_groups[group_id].append(
            {**prediction, "overlay_path": overlay_path}
        )

    if len(predictions) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Generated {len(predictions)} Grad-CAMs; expected {EXPECTED_TEST_IMAGES}."
        )
    for group_id, variants in comparison_groups.items():
        if len(variants) != 3 or {
            item["background"] for item in variants
        } != set(BACKGROUND_CATEGORIES):
            raise RuntimeError(f"Incomplete Grad-CAM comparison group {group_id}.")
        save_comparison_sheet(
            group_id,
            variants,
            comparison_root / f"{group_id}.png",
        )

    with (output_dir / "gradcam_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fields = [
            "source_image_id",
            "source_annotation_id",
            "counterfactual_group_id",
            "background",
            "true_class",
            "true_label",
            "predicted_class",
            "predicted_label",
            "confidence",
            "correct",
            "animal_activation_percent",
            "background_activation_percent",
            "image_path",
            "overlay_path",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    localization = localization_summary(predictions)
    summary = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": checkpoint_hash_before,
        "model_mode": "eval",
        "gradient_scope": "Gradients computed only for the target activation; weights are not updated.",
        "target_layer": "SmallCNN.features[9] (Conv2d 64 -> 96 channels)",
        "target_class": "predicted class for each test image",
        "test_image_count": len(predictions),
        "counterfactual_group_count": len(comparison_groups),
        "all_groups_have_three_background_variants": True,
        "class_to_label": CLASS_TO_LABEL,
        "localization_analysis": localization,
        "observations": (
            "Grad-CAM heatmaps and mask-aligned activation percentages describe "
            "where the frozen model's predicted-class evidence is concentrated. "
            "Differences across backgrounds are descriptive and do not by "
            "themselves establish shortcut learning or causality."
        ),
    }
    (output_dir / "gradcam_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "observations.txt").write_text(
        summary["observations"] + "\n", encoding="utf-8"
    )

    if (
        checkpoint_hash_before != sha256_file(checkpoint_path)
        or dataset_hashes_before != sha256_dataset(dataset_dir)
    ):
        raise RuntimeError("Checkpoint or dataset changed during Grad-CAM generation.")
    return {
        "output_dir": output_dir,
        "prediction_count": len(predictions),
        "group_count": len(comparison_groups),
        "localization": localization,
        "checkpoint_hash": checkpoint_hash_before,
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
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data/raw/coco/annotations_trainval2017.zip",
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "experiments/gradcam_test"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    result = run_evaluation(parse_args())
    print(
        f"Generated {result['prediction_count']} Grad-CAM overlays across "
        f"{result['group_count']} counterfactual groups at {result['output_dir']}"
    )
    print(json.dumps(result["localization"], indent=2))
