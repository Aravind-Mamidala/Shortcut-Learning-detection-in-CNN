#!/usr/bin/env python3
"""Build a small, reproducible cow/horse background-bias pilot dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
from pycocotools import mask as mask_utils

from background_review import APPROVALS_NAME, prepare_backgrounds, read_approved_backgrounds


COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_ANNOTATIONS_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
COCO_IMAGE_BASE_URL = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org"
CLASS_LABELS = {"cow": 0, "horse": 1}
PREFERRED_BACKGROUND = {"cow": "grass", "horse": "sand"}
BACKGROUND_CATEGORIES = ("grass", "sand", "snow")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
ANIMALS_PER_CLASS = 50
IMAGE_SIZE = 224


def _download(
    url: str,
    destination: Path,
    *,
    expected_md5: str | None = None,
    image: bool = False,
    retries: int = 3,
) -> None:
    """Download to a temporary file, validate it, and never replace existing data."""
    if destination.exists():
        if image:
            _validate_image(destination)
        if expected_md5 and _md5(destination) != expected_md5:
            raise RuntimeError(f"Existing file checksum mismatch: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, headers={"User-Agent": "ShortcutLearningPilot/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                expected_length = response.headers.get("Content-Length")
                with temporary_path.open("wb") as target:
                    shutil.copyfileobj(response, target)
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Empty download from {url}")
            if expected_length and temporary_path.stat().st_size != int(expected_length):
                raise RuntimeError(f"Incomplete download from {url}")
            if expected_md5 and _md5(temporary_path) != expected_md5:
                raise RuntimeError(f"Checksum mismatch for {url}")
            if image:
                _validate_image(temporary_path)
            temporary_path.replace(destination)
            return
        except (
            OSError,
            urllib.error.URLError,
            RuntimeError,
            ValueError,
        ) as exc:
            error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {error}") from error


def _md5(path: Path) -> str:
    import hashlib

    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise RuntimeError(f"Missing, invalid, or corrupt image {path}: {exc}") from exc


def load_coco_candidates(
    raw_dir: Path, per_class: int = ANIMALS_PER_CLASS, seed: int = SEED
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Read COCO annotations and select up to N original images per class."""
    annotation_zip = raw_dir / "coco" / "annotations_trainval2017.zip"
    _download(
        COCO_ANNOTATIONS_URL,
        annotation_zip,
        expected_md5=COCO_ANNOTATIONS_MD5,
    )

    images: dict[int, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with zipfile.ZipFile(annotation_zip) as archive:
        for split_name in ("train2017", "val2017"):
            member = f"annotations/instances_{split_name}.json"
            if member not in archive.namelist():
                raise RuntimeError(f"COCO annotation archive is missing {member}")
            with archive.open(member) as source:
                data = json.load(source)
            categories = {
                item["id"]: item["name"]
                for item in data["categories"]
                if item["name"] in CLASS_LABELS
            }
            license_by_id = {item["id"]: item for item in data.get("licenses", [])}
            image_info = {item["id"]: item for item in data["images"]}
            for annotation in data["annotations"]:
                class_name = categories.get(annotation["category_id"])
                if not class_name:
                    continue
                image_id = annotation["image_id"]
                image = image_info[image_id]
                license_info = license_by_id.get(image.get("license"), {})
                images.setdefault(
                    image_id,
                    {
                        "file_name": image["file_name"],
                        "source_split": split_name,
                        "source_url": image.get("coco_url", ""),
                        "license_id": image.get("license", ""),
                        "license_name": license_info.get("name", ""),
                        "license_url": license_info.get("url", ""),
                        "width": image["width"],
                        "height": image["height"],
                    },
                )
                annotations_by_image[image_id].append(
                    {**annotation, "class_name": class_name}
                )
            del data

    rng = random.Random(seed)
    selected_ids: set[int] = set()
    for class_name in CLASS_LABELS:
        matching = [
            image_id
            for image_id, annotations in annotations_by_image.items()
            if any(item["class_name"] == class_name for item in annotations)
        ]
        rng.shuffle(matching)
        selected_ids.update(matching[:per_class])
        if len(matching) < per_class:
            print(
                f"Only {len(matching)} COCO images found for {class_name}; "
                f"using all available.",
                flush=True,
            )

    return (
        {image_id: images[image_id] for image_id in selected_ids},
        {image_id: annotations_by_image[image_id] for image_id in selected_ids},
    )


def split_source_images(
    annotations_by_image: dict[int, list[dict[str, Any]]], seed: int = SEED
) -> dict[int, str]:
    """Assign each original source image to exactly one deterministic split."""
    labels = {
        image_id: {annotation["class_name"] for annotation in annotations}
        for image_id, annotations in annotations_by_image.items()
    }
    targets = {
        split: {
            class_name: sum(class_name in item for item in labels.values()) * ratio
            for class_name in CLASS_LABELS
        }
        for split, ratio in SPLIT_RATIOS.items()
    }
    counts = {
        split: Counter({class_name: 0 for class_name in CLASS_LABELS})
        for split in SPLIT_RATIOS
    }
    rng = random.Random(seed)
    image_ids = list(labels)
    rng.shuffle(image_ids)
    image_ids.sort(key=lambda image_id: -len(labels[image_id]))
    result: dict[int, str] = {}
    for image_id in image_ids:
        split = min(
            SPLIT_RATIOS,
            key=lambda name: (
                sum(
                    counts[name][label] / max(targets[name][label], 1)
                    for label in sorted(labels[image_id])
                ),
                rng.random(),
            ),
        )
        result[image_id] = split
        counts[split].update(labels[image_id])
    if any(counts[split][label] == 0 for split in SPLIT_RATIOS for label in CLASS_LABELS):
        raise RuntimeError("Unable to place both classes in every source-image split.")
    return result


def choose_instance(
    annotations: list[dict[str, Any]], class_name: str
) -> dict[str, Any] | None:
    matching = [item for item in annotations if item["class_name"] == class_name]
    return max(matching, key=lambda item: item.get("area", 0), default=None)


def decode_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if not segmentation:
        raise RuntimeError(f"COCO annotation {annotation.get('id')} has no mask.")
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(
        segmentation.get("counts"), list
    ):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width) or not np.any(decoded):
        raise RuntimeError(
            f"COCO annotation {annotation.get('id')} has an invalid or empty mask."
        )
    return decoded.astype(np.uint8) * 255


def composite_animal(
    animal_path: Path,
    annotation: dict[str, Any],
    background_path: Path,
    image_size: int = IMAGE_SIZE,
) -> Image.Image:
    """Composite the annotated animal mask over a resized background."""
    _validate_image(animal_path)
    _validate_image(background_path)
    with Image.open(animal_path) as source:
        animal = source.convert("RGB")
    mask = decode_mask(annotation, animal.height, animal.width)
    ys, xs = np.nonzero(mask)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1
    foreground = animal.crop((left, top, right, bottom)).convert("RGBA")
    alpha = Image.fromarray(mask).crop((left, top, right, bottom))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))
    foreground.putalpha(alpha)
    foreground.thumbnail((int(image_size * 0.78), int(image_size * 0.78)), Image.Resampling.LANCZOS)

    with Image.open(background_path) as source:
        background = ImageOps.fit(
            source.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
        ).convert("RGBA")
    x = (image_size - foreground.width) // 2
    y = max(0, image_size - foreground.height - int(image_size * 0.06))
    background.alpha_composite(foreground, (x, y))
    return background.convert("RGB")


def background_for(
    backgrounds: dict[str, list[dict[str, Any]]],
    category: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    choices = [item for item in backgrounds[category] if item["split"] == split]
    if not choices:
        raise RuntimeError(f"No approved {category} backgrounds for split {split}.")
    return choices[index % len(choices)]


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty metadata file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _generate_variant(
    source_id: int,
    annotation: dict[str, Any],
    class_name: str,
    split: str,
    background: dict[str, Any],
    source_info: dict[str, Any],
    source_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    filename = (
        f"{class_name}_{source_id}_{annotation['id']}_"
        f"{background['category']}_{background['background_id']}.jpg"
    )
    relative_path = Path("images") / split / filename
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    composite_animal(source_path, annotation, Path(background["path"])).save(
        destination, quality=92
    )
    width, height = _validate_image(destination)
    if (width, height) != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Unexpected composite dimensions at {destination}")
    rows.append(
        {
            "path": relative_path.as_posix(),
            "split": split,
            "class_name": class_name,
            "label": CLASS_LABELS[class_name],
            "source_image_id": source_id,
            "source_annotation_id": annotation["id"],
            "counterfactual_group_id": f"coco-{source_id}-{class_name}",
            "variant": "biased_train" if split == "train" else "counterfactual",
            "background_id": background["background_id"],
            "background_category": background["category"],
            "background_source_url": background["source_url"],
            "background_license": background["license"],
            "background_attribution": background.get("attribution", ""),
            "background_width": background["width"],
            "background_height": background["height"],
            "animal_source_url": source_info["source_url"],
            "animal_license_id": source_info["license_id"],
            "animal_license_name": source_info["license_name"],
            "animal_license_url": source_info["license_url"],
            "animal_source_width": source_info["width"],
            "animal_source_height": source_info["height"],
            "image_width": width,
            "image_height": height,
        }
    )
    return destination


def _validate_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_splits: dict[int, set[str]] = defaultdict(set)
    background_splits: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_splits[int(row["source_image_id"])].add(row["split"])
        background_splits[str(row["background_id"])].add(row["split"])
        if row["split"] in ("val", "test"):
            groups[row["counterfactual_group_id"]].append(row)
        if int(row["label"]) != CLASS_LABELS[row["class_name"]]:
            raise RuntimeError(f"Incorrect class label in metadata: {row}")
    for group_id, variants in groups.items():
        if len(variants) > len(BACKGROUND_CATEGORIES):
            raise RuntimeError(
                f"Counterfactual group {group_id} has more than three variants."
            )
        if len(variants) < len(BACKGROUND_CATEGORIES):
            raise RuntimeError(
                f"Counterfactual group {group_id} is missing background variants."
            )
        categories = [row["background_category"] for row in variants]
        if len(set(categories)) != len(categories):
            raise RuntimeError(
                f"Counterfactual group {group_id} has duplicate background variants."
            )
        if set(categories) != set(BACKGROUND_CATEGORIES):
            raise RuntimeError(
                f"Counterfactual group {group_id} has incorrect background categories."
            )
        if len({row["split"] for row in variants}) != 1:
            raise RuntimeError(
                f"Counterfactual group {group_id} spans multiple splits."
            )
        if len({row["source_image_id"] for row in variants}) != 1:
            raise RuntimeError(f"Counterfactual group {group_id} mixes source images.")
        if len({row["class_name"] for row in variants}) != 1 or len(
            {row["label"] for row in variants}
        ) != 1:
            raise RuntimeError(
                f"Counterfactual group {group_id} mixes classes or labels."
            )
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise RuntimeError("Source image leakage found across data splits.")
    if any(len(splits) != 1 for splits in background_splits.values()):
        raise RuntimeError("Background image leakage found across data splits.")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLIT_RATIOS:
        counts[split] = {}
        for class_name in CLASS_LABELS:
            distribution = Counter(
                row["background_category"]
                for row in rows
                if row["split"] == split and row["class_name"] == class_name
            )
            if split == "train":
                preferred = PREFERRED_BACKGROUND[class_name]
                sample_count = sum(distribution.values())
                if not sample_count or distribution[preferred] != round(sample_count * 0.90):
                    raise RuntimeError(
                        f"Unexpected {class_name} training distribution: {dict(distribution)}"
                    )
            else:
                values = [distribution[category] for category in BACKGROUND_CATEGORIES]
                if not values[0] or len(set(values)) != 1:
                    raise RuntimeError(
                        f"Unbalanced {class_name} {split} backgrounds: {dict(distribution)}"
                    )
            counts[split][class_name] = {
                category: distribution[category] for category in BACKGROUND_CATEGORIES
            }
    return counts


def _contact_sheet(rows: list[dict[str, Any]], output_dir: Path) -> None:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['split']}_{row['class_name']}_{row['background_category']}"
        if len(samples[key]) < 4:
            samples[key].append(row)
    for key, items in samples.items():
        columns, thumb_size = 4, 180
        cell_height = thumb_size + 30
        sheet = Image.new("RGB", (columns * thumb_size, cell_height), "white")
        draw_index = 0
        draw = ImageDraw.Draw(sheet)
        for row in items:
            image_path = output_dir / row["path"]
            with Image.open(image_path) as source:
                thumb = ImageOps.fit(source.convert("RGB"), (thumb_size, thumb_size))
            x = draw_index * thumb_size
            sheet.paste(thumb, (x, 0))
            draw.text(
                (x + 2, thumb_size + 3),
                f"{row['class_name']} {row['background_category']} "
                f"src {row['source_image_id']} bg {row['background_id']}"[:34],
                fill="black",
            )
            draw_index += 1
        visualizations = output_dir / "visualizations"
        visualizations.mkdir(parents=True, exist_ok=True)
        sheet.save(visualizations / f"composites_{key}.jpg", quality=90)


def build_dataset(args: argparse.Namespace) -> None:
    raw_dir = args.raw.resolve()
    output_dir = args.output.resolve()
    if args.prepare_backgrounds:
        prepare_backgrounds(raw_dir, args.backgrounds_per_category, args.seed)
        print(
            "Background candidates prepared. Review the contact sheets and mark "
            f"approved rows in {raw_dir / 'backgrounds' / APPROVALS_NAME}."
        )
        return
    if output_dir.exists():
        raise RuntimeError(f"Output already exists; choose a new path: {output_dir}")

    approvals_path = args.approve_backgrounds or (
        raw_dir / "backgrounds" / APPROVALS_NAME
    )
    backgrounds = read_approved_backgrounds(
        raw_dir, approvals_path, args.backgrounds_per_category, args.seed
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        rows = _build_pilot(args, raw_dir, staging, backgrounds)
        distributions = _validate_metadata(rows)
        _save_csv(staging / "metadata.csv", rows)
        _save_csv(staging / "background_sources.csv", [
            {key: value for key, value in item.items() if key != "path"}
            for category in BACKGROUND_CATEGORIES
            for item in backgrounds[category]
        ])
        summary = {
            "seed": args.seed,
            "animals_per_class_limit": args.animals_per_class,
            "source_image_counts": {
                split: {
                    class_name: len({
                        int(row["source_image_id"])
                        for row in rows
                        if row["split"] == split and row["class_name"] == class_name
                    })
                    for class_name in CLASS_LABELS
                }
                for split in SPLIT_RATIOS
            },
            "generated_sample_counts": dict(Counter(row["split"] for row in rows)),
            "background_distribution": distributions,
            "class_to_label": CLASS_LABELS,
        }
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _validate_staged_images(staging, rows)
        _contact_sheet(rows, staging)
        staging.replace(output_dir)
    print(f"Pilot dataset generated: {output_dir}")
    print(f"Generated sample counts: {summary['generated_sample_counts']}")


def _build_pilot(
    args: argparse.Namespace,
    raw_dir: Path,
    output_dir: Path,
    backgrounds: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    images, annotations = load_coco_candidates(
        raw_dir, args.animals_per_class, args.seed
    )
    assignments = split_source_images(annotations, args.seed)
    rows: list[dict[str, Any]] = []
    image_cache: dict[int, Path] = {}
    class_sources: dict[str, list[int]] = {}
    for class_name in CLASS_LABELS:
        image_ids = [
            image_id
            for image_id, items in annotations.items()
            if choose_instance(items, class_name) is not None
        ]
        selection_rng = random.Random(f"{args.seed}:class-sources:{class_name}")
        selection_rng.shuffle(image_ids)
        class_sources[class_name] = sorted(image_ids[: args.animals_per_class])
    for class_name, image_ids in class_sources.items():
        training_ids = [image_id for image_id in image_ids if assignments[image_id] == "train"]
        rng = random.Random(f"{args.seed}:{class_name}")
        rng.shuffle(training_ids)
        preferred_count = round(len(training_ids) * 0.90)
        categories = [PREFERRED_BACKGROUND[class_name]] * preferred_count
        alternatives = [
            category
            for category in BACKGROUND_CATEGORIES
            if category != PREFERRED_BACKGROUND[class_name]
        ]
        categories.extend(alternatives[index % len(alternatives)] for index in range(len(training_ids) - preferred_count))
        rng.shuffle(categories)
        for index, (image_id, category) in enumerate(zip(training_ids, categories)):
            source_path = _animal_image(images[image_id], raw_dir)
            image_cache[image_id] = source_path
            annotation = choose_instance(annotations[image_id], class_name)
            background = background_for(backgrounds, category, "train", index)
            _generate_variant(
                image_id, annotation, class_name, "train", background,
                images[image_id], source_path, output_dir, rows,
            )

    for split in ("val", "test"):
        for class_name, image_ids in class_sources.items():
            selected = [
                image_id for image_id in image_ids if assignments[image_id] == split
            ]
            for index, image_id in enumerate(selected):
                source_path = image_cache.get(image_id) or _animal_image(images[image_id], raw_dir)
                annotation = choose_instance(annotations[image_id], class_name)
                for category in BACKGROUND_CATEGORIES:
                    background = background_for(backgrounds, category, split, index)
                    _generate_variant(
                        image_id, annotation, class_name, split, background,
                        images[image_id], source_path, output_dir, rows,
                    )
    return rows


def _animal_image(image_info: dict[str, Any], raw_dir: Path) -> Path:
    source_split = image_info["source_split"]
    filename = image_info["file_name"]
    destination = raw_dir / "coco" / "images" / source_split / filename
    if not destination.exists():
        url = f"{COCO_IMAGE_BASE_URL}/{source_split}/{urllib.parse.quote(filename)}"
        _download(url, destination, image=True)
    else:
        _validate_image(destination)
    return destination


def _validate_staged_images(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        width, height = _validate_image(output_dir / row["path"])
        if width != IMAGE_SIZE or height != IMAGE_SIZE:
            raise RuntimeError(f"Invalid generated dimensions for {row['path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/pilot/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/pilot/dataset"))
    parser.add_argument("--animals-per-class", type=int, default=ANIMALS_PER_CLASS)
    parser.add_argument("--backgrounds-per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prepare-backgrounds", action="store_true")
    parser.add_argument("--approve-backgrounds", type=Path)
    args = parser.parse_args()
    if not 10 <= args.animals_per_class <= ANIMALS_PER_CLASS:
        parser.error(
            f"--animals-per-class must be between 10 and {ANIMALS_PER_CLASS} "
            "for the pilot."
        )
    if args.backgrounds_per_category < 6:
        parser.error("--backgrounds-per-category must be at least 6.")
    if args.prepare_backgrounds and args.approve_backgrounds:
        parser.error("Do not combine --prepare-backgrounds and --approve-backgrounds.")
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
