import csv
import io
import json
import random
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
from PIL import Image

from background_review import read_approved_backgrounds
from data_preparation import (
    BACKGROUND_CATEGORIES,
    CLASS_LABELS,
    IMAGE_SIZE,
    _download,
    _animal_image,
    _validate_image,
    _validate_metadata,
    build_dataset,
    composite_animal,
    decode_mask,
    split_source_images,
)


def _polygon_annotation(image_id: int, category: str, annotation_id: int) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "class_name": category,
        "area": 100 * 100,
        "segmentation": [[50, 50, 150, 50, 150, 150, 50, 150]],
    }


class DatasetPreparationTests(unittest.TestCase):
    def test_source_image_split_is_seeded_and_leakage_free(self):
        annotations = {
            index: [{"class_name": "cow" if index < 30 else "horse"}]
            for index in range(60)
        }

        first = split_source_images(annotations, seed=19)
        second = split_source_images(annotations, seed=19)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(annotations))
        for split in ("train", "val", "test"):
            self.assertTrue(any(value == split for value in first.values()))

    def test_mask_decoding_rejects_missing_masks_and_returns_image_shape(self):
        annotation = _polygon_annotation(1, "cow", 101)
        mask = decode_mask(annotation, 200, 200)

        self.assertEqual(mask.shape, (200, 200))
        self.assertGreater(np.count_nonzero(mask), 0)
        with self.assertRaisesRegex(RuntimeError, "has no mask"):
            decode_mask({"id": 102}, 200, 200)

    def test_mask_decoding_converts_uncompressed_coco_rle(self):
        height, width = 8, 7
        pixels = np.zeros((height, width), dtype=np.uint8)
        pixels[2:6, 1:5] = 1
        flat = pixels.flatten(order="F")
        counts = []
        current = 0
        run_length = 0
        for value in flat:
            if value == current:
                run_length += 1
            else:
                counts.append(run_length)
                run_length = 1
                current = int(value)
        counts.append(run_length)
        annotation = {
            "id": 103,
            "segmentation": {"size": [height, width], "counts": counts},
        }

        decoded = decode_mask(annotation, height, width)

        np.testing.assert_array_equal(decoded, pixels * 255)

    def test_composite_changes_background_and_keeps_requested_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            animal_path = root / "animal.jpg"
            background_path = root / "background.jpg"
            Image.new("RGB", (200, 200), (200, 30, 30)).save(animal_path)
            Image.new("RGB", (300, 240), (30, 180, 30)).save(background_path)

            composite = composite_animal(
                animal_path,
                _polygon_annotation(1, "cow", 101),
                background_path,
                image_size=IMAGE_SIZE,
            )

        self.assertEqual(composite.size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertNotEqual(composite.getpixel((0, 0)), (200, 30, 30))

    def test_animal_download_uses_https_retries_and_leaves_no_bad_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "animal.jpg"
            with patch(
                "data_preparation.urllib.request.urlopen",
                side_effect=URLError("offline"),
            ) as urlopen, patch("data_preparation.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                    _download(
                        "https://images.cocodataset.org/train2017/animal.jpg",
                        destination,
                        image=True,
                    )

            self.assertEqual(urlopen.call_count, 3)
            self.assertTrue(urlopen.call_args.args[0].full_url.startswith("https://"))
            self.assertFalse(destination.exists())

    def test_animal_image_uses_certificate_valid_coco_s3_endpoint(self):
        image_info = {
            "source_split": "val2017",
            "file_name": "000000244411.jpg",
        }
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            with patch("data_preparation._download") as download:
                path = _animal_image(image_info, raw_dir)

        self.assertEqual(path, raw_dir / "coco" / "images" / "val2017" / "000000244411.jpg")
        download.assert_called_once_with(
            "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/"
            "val2017/000000244411.jpg",
            path,
            image=True,
        )

    def test_corrupt_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.jpg"
            path.write_bytes(b"not an image")

            with self.assertRaisesRegex(RuntimeError, "corrupt image"):
                _validate_image(path)

    def test_manual_approval_uses_only_approved_valid_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            approval_csv = root / "approvals.csv"
            rows = []
            for category in BACKGROUND_CATEGORIES:
                for index in range(8):
                    image_path = root / f"{category}-{index}.jpg"
                    Image.new(
                        "RGB",
                        (256, 256),
                        (index * 15, index * 10, index * 5),
                    ).save(image_path)
                    rows.append(
                        {
                            "background_id": f"{category}-{index}",
                            "category": category,
                            "approved": "yes" if index < 6 else "no",
                            "path": str(image_path),
                            "source_url": f"https://example.test/{category}/{index}",
                            "license": "CC BY 4.0",
                        }
                    )
            with approval_csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            selected = read_approved_backgrounds(
                root, approval_csv, per_category=6, seed=42
            )

            for category in BACKGROUND_CATEGORIES:
                self.assertEqual(len(selected[category]), 6)
                self.assertEqual(
                    {item["split"] for item in selected[category]},
                    {"train", "val", "test"},
                )

    def test_missing_approved_background_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            approval_csv = root / "approvals.csv"
            with approval_csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "background_id", "category", "approved", "path", "license"
                    ],
                )
                writer.writeheader()
            with self.assertRaisesRegex(RuntimeError, "Approve at least 6"):
                read_approved_backgrounds(root, approval_csv, 6, 42)

    def test_prepare_backgrounds_mode_does_not_load_coco(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = type(
                "Args",
                (),
                {
                    "prepare_backgrounds": True,
                    "raw": root,
                    "output": root / "dataset",
                    "backgrounds_per_category": 6,
                    "seed": 42,
                    "approve_backgrounds": None,
                },
            )()
            with (
                patch("data_preparation.prepare_backgrounds") as prepare,
                patch("data_preparation.load_coco_candidates") as load_coco,
            ):
                build_dataset(args)
            prepare.assert_called_once()
            load_coco.assert_not_called()
            self.assertFalse((root / "dataset").exists())

    def test_pilot_distributions_counterfactuals_and_source_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "coco" / "images" / "train2017"
            source_dir.mkdir(parents=True)
            images = {}
            annotations = {}
            image_id = 1
            annotation_id = 1001
            for class_name in CLASS_LABELS:
                for _ in range(55):
                    filename = f"{image_id}.jpg"
                    Image.new("RGB", (200, 200), (220, 30, 30)).save(
                        source_dir / filename
                    )
                    images[image_id] = {
                        "file_name": filename,
                        "source_split": "train2017",
                        "source_url": f"https://example.test/coco/{image_id}",
                        "license_id": 1,
                        "license_name": "CC BY 4.0",
                        "license_url": "https://example.test/license",
                        "width": 200,
                        "height": 200,
                    }
                    annotations[image_id] = [
                        _polygon_annotation(image_id, class_name, annotation_id)
                    ]
                    image_id += 1
                    annotation_id += 1

            backgrounds = {category: [] for category in BACKGROUND_CATEGORIES}
            colors = {
                "grass": (20, 180, 20),
                "sand": (200, 170, 40),
                "snow": (230, 230, 230),
            }
            for category in BACKGROUND_CATEGORIES:
                for index, split in enumerate(
                    ("train", "train", "train", "train", "val", "test")
                ):
                    background_path = root / f"{category}-{index}.jpg"
                    Image.new("RGB", (256, 256), colors[category]).save(
                        background_path
                    )
                    backgrounds[category].append(
                        {
                            "background_id": f"{category}-{index}",
                            "category": category,
                            "path": str(background_path),
                            "source_url": f"https://example.test/{category}/{index}",
                            "license": "CC BY 4.0",
                            "attribution": "Test Artist",
                            "width": 256,
                            "height": 256,
                            "split": split,
                        }
                    )

            args = type(
                "Args",
                (),
                {
                    "prepare_backgrounds": False,
                    "raw": root,
                    "output": root / "dataset",
                    "animals_per_class": 50,
                    "backgrounds_per_category": 6,
                    "seed": 42,
                    "approve_backgrounds": None,
                },
            )()
            with patch(
                "data_preparation.load_coco_candidates",
                return_value=(images, annotations),
            ), patch(
                "data_preparation.read_approved_backgrounds",
                return_value=backgrounds,
            ):
                build_dataset(args)

            output_dir = root / "dataset"
            with (output_dir / "metadata.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))
            summary = json.loads(
                (output_dir / "dataset_summary.json").read_text(encoding="utf-8")
            )
            assignments = split_source_images(annotations, seed=args.seed)
            self.assertEqual(
                assignments, split_source_images(annotations, seed=args.seed)
            )
            for class_name in CLASS_LABELS:
                available = [
                    source_id
                    for source_id, source_annotations in annotations.items()
                    if any(
                        item["class_name"] == class_name
                        for item in source_annotations
                    )
                ]
                selection_rng = random.Random(
                    f"{args.seed}:class-sources:{class_name}"
                )
                selection_rng.shuffle(available)
                expected_sources = set(available[: args.animals_per_class])
                actual_sources = {
                    int(row["source_image_id"])
                    for row in rows
                    if row["class_name"] == class_name
                }
                self.assertEqual(actual_sources, expected_sources)
                self.assertLessEqual(len(actual_sources), 50)

            for source_id in {int(row["source_image_id"]) for row in rows}:
                source_rows = [
                    row for row in rows if int(row["source_image_id"]) == source_id
                ]
                self.assertEqual({row["split"] for row in source_rows}, {
                    assignments[source_id]
                })

            expected_source_counts = {
                split: {
                    class_name: len(
                        {
                            int(row["source_image_id"])
                            for row in rows
                            if row["split"] == split
                            and row["class_name"] == class_name
                        }
                    )
                    for class_name in CLASS_LABELS
                }
                for split in ("train", "val", "test")
            }
            self.assertEqual(summary["source_image_counts"], expected_source_counts)
            self.assertEqual(
                summary["generated_sample_counts"],
                dict(Counter(row["split"] for row in rows)),
            )

            for class_name, preferred in (("cow", "grass"), ("horse", "sand")):
                training = [
                    row for row in rows
                    if row["split"] == "train" and row["class_name"] == class_name
                ]
                # Use Python's round() for the 90% target.
                expected_preferred_count = round(len(training) * 0.90)
                self.assertEqual(
                    sum(row["background_category"] == preferred for row in training),
                    expected_preferred_count,
                )
                recorded_train_distribution = summary["background_distribution"][
                    "train"
                ][class_name]
                for category in BACKGROUND_CATEGORIES:
                    self.assertEqual(
                        recorded_train_distribution[category],
                        sum(
                            row["background_category"] == category
                            for row in training
                        ),
                    )
                for split in ("val", "test"):
                    counts = [
                        sum(
                            row["split"] == split
                            and row["class_name"] == class_name
                            and row["background_category"] == category
                            for row in rows
                        )
                        for category in BACKGROUND_CATEGORIES
                    ]
                    self.assertEqual(len(set(counts)), 1)
                    self.assertGreater(counts[0], 0)

            groups = {}
            for row in rows:
                if row["split"] != "train":
                    groups.setdefault(row["counterfactual_group_id"], []).append(row)
            self.assertTrue(groups)
            for variants in groups.values():
                self.assertEqual(len(variants), 3)
                self.assertEqual(
                    {row["background_category"] for row in variants},
                    set(BACKGROUND_CATEGORIES),
                )
                self.assertEqual(len({row["source_image_id"] for row in variants}), 1)
                self.assertEqual(len({row["class_name"] for row in variants}), 1)
                self.assertEqual(len({row["label"] for row in variants}), 1)
                self.assertEqual(len({row["split"] for row in variants}), 1)

            valid_group_id = next(iter(groups))
            malformed_cases = (
                ("missing", "missing background variants"),
                ("duplicate", "duplicate background variants"),
                ("too_many", "more than three variants"),
                ("incorrect_category", "incorrect background categories"),
                ("mixed_split", "spans multiple splits"),
            )
            for case, expected_error in malformed_cases:
                with self.subTest(counterfactual_validation=case):
                    candidate = deepcopy(rows)
                    group_rows = [
                        row
                        for row in candidate
                        if row["counterfactual_group_id"]
                        == valid_group_id
                    ]
                    if case == "missing":
                        candidate.remove(group_rows[0])
                    elif case == "duplicate":
                        group_rows[2]["background_category"] = group_rows[0][
                            "background_category"
                        ]
                    elif case == "too_many":
                        candidate.append(deepcopy(group_rows[0]))
                    elif case == "incorrect_category":
                        group_rows[2]["background_category"] = "desert"
                    elif case == "mixed_split":
                        group_rows[2]["split"] = (
                            "test" if group_rows[0]["split"] == "val" else "val"
                        )
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        _validate_metadata(candidate)

    def test_failed_build_cleans_staging_and_keeps_final_path_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            approvals_path = root / "approvals.csv"
            approvals_path.write_text("background_id,category,approved\n", encoding="utf-8")
            output_dir = root / "dataset"
            args = type(
                "Args",
                (),
                {
                    "prepare_backgrounds": False,
                    "raw": root,
                    "output": output_dir,
                    "approve_backgrounds": approvals_path,
                    "backgrounds_per_category": 6,
                    "seed": 42,
                    "animals_per_class": 50,
                },
            )()
            with (
                patch("data_preparation.read_approved_backgrounds", return_value={}),
                patch(
                    "data_preparation._build_pilot",
                    side_effect=RuntimeError("synthetic build failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic build failure"):
                    build_dataset(args)

            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob(".dataset-*")), [])


if __name__ == "__main__":
    unittest.main()
