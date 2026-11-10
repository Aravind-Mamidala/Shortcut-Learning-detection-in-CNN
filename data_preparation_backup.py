#!/usr/bin/env python3
"""Build a reproducible cow/horse background-bias dataset from COCO and Commons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import http.client
import json
import math
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
from pycocotools import mask as mask_utils


COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_ANNOTATIONS_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
COCO_IMAGE_BASE_URL = "https://images.cocodataset.org"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_USER_AGENT = (
    "ShortcutLearningDatasetBuilder/1.0 "
    "(educational research; contact the project maintainer)"
)
CLASSES = {"cow": 0, "horse": 1}
BACKGROUNDS = {
    "grass": "grass meadow field landscape",
    "sand": "sand desert dunes landscape",
    "snow": "snow field winter landscape",
}
PREFERRED_BACKGROUND = {"cow": "grass", "horse": "sand"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
ALLOWED_LICENSES = ("cc0", "public domain", "cc by", "cc-by")
RELEVANCE_TERMS = {
    "grass": ("grass", "meadow", "lawn", "field", "pasture", "prairie", "turf"),
    "sand": ("sand", "sandy", "desert", "dune", "beach", "seashore"),
    "snow": ("snow", "snowy", "winter", "ice", "glacier", "frost", "blizzard"),
}
IRRELEVANT_TERMS = (
    "drawing",
    "illustration",
    "map",
    "logo",
    "icon",
    "screenshot",
    "diagram",
    "painting",
    "clipart",
    "coat of arms",
)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    user_agent: str,
    expected_md5: str | None = None,
    validate_image: bool = False,
    retries: int = 3,
    timeout: int = 60,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(
                f"Existing download destination is not a valid non-empty file: "
                f"{destination}; preserving it rather than overwriting."
            )
        if expected_md5 and _md5_file(destination) != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for existing file {destination}; "
                "preserving it rather than overwriting."
            )
        if validate_image:
            _validate_image(destination)
        return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded an empty file from {url}")
            if content_length and temporary_path.stat().st_size != int(content_length):
                raise RuntimeError(f"Incomplete download from {url}")
            if expected_md5 and _md5_file(temporary_path) != expected_md5:
                raise RuntimeError(f"Checksum mismatch for {url}")
            if validate_image:
                _validate_image(temporary_path)
            temporary_path.replace(destination)
            return
        except (
            OSError,
            urllib.error.URLError,
            http.client.HTTPException,
            RuntimeError,
            ValueError,
        ) as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))

    raise RuntimeError(
        f"Could not download a valid file from {url} after {retries} attempts: "
        f"{last_error}"
    ) from last_error


def _validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width < 1 or height < 1:
                raise ValueError("image has invalid dimensions")
            return width, height
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise RuntimeError(f"Invalid or corrupted image {path}: {exc}") from exc


def _image_fingerprints(path: Path) -> tuple[str, int, tuple[int, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        dimensions = image.size
        pixels = image.tobytes()
        thumbnail = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
    difference_hash = 0
    values = list(thumbnail.getdata())
    for row in range(8):
        for column in range(8):
            difference_hash = (difference_hash << 1) | int(
                values[row * 9 + column] > values[row * 9 + column + 1]
            )
    return hashlib.sha256(pixels).hexdigest(), difference_hash, dimensions


def _hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def load_coco_candidates(
    raw_root: Path, per_class: int, seed: int
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    annotation_zip = raw_root / "coco" / "annotations_trainval2017.zip"
    download_file(
        COCO_ANNOTATIONS_URL,
        annotation_zip,
        "ShortcutLearningDatasetBuilder/1.0",
        expected_md5=COCO_ANNOTATIONS_MD5,
    )

    candidates: dict[int, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with zipfile.ZipFile(annotation_zip) as archive:
        for split_name in ("train2017", "val2017"):
            member = f"annotations/instances_{split_name}.json"
            if member not in archive.namelist():
                raise RuntimeError(f"Missing {member} in {annotation_zip}")
            with archive.open(member) as annotation_file:
                data = json.load(annotation_file)

            category_ids = {
                category["id"]: category["name"]
                for category in data["categories"]
                if category["name"] in CLASSES
            }
            license_by_id = {
                license_info["id"]: license_info
                for license_info in data.get("licenses", [])
            }
            image_info = {image["id"]: image for image in data["images"]}
            for annotation in data["annotations"]:
                class_name = category_ids.get(annotation["category_id"])
                if class_name is None:
                    continue
                image_id = annotation["image_id"]
                image = image_info[image_id]
                candidates.setdefault(
                    image_id,
                    {
                        "file_name": image["file_name"],
                        "coco_url": image.get("coco_url", ""),
                        "source_split": split_name,
                        "license_id": image.get("license", ""),
                        "license_name": license_by_id.get(
                            image.get("license"), {}
                        ).get("name", ""),
                        "license_url": license_by_id.get(
                            image.get("license"), {}
                        ).get("url", ""),
                    },
                )
                annotations_by_image[image_id].append(
                    {
                        **annotation,
                        "class_name": class_name,
                        "animal_license_id": image.get("license", ""),
                        "animal_license_name": license_by_id.get(
                            image.get("license"), {}
                        ).get("name", ""),
                        "animal_license_url": license_by_id.get(
                            image.get("license"), {}
                        ).get("url", ""),
                        "animal_source_url": image.get("coco_url", ""),
                        "animal_source_width": image.get("width", ""),
                        "animal_source_height": image.get("height", ""),
                    }
                )
            del data

    rng = random.Random(seed)
    selected_ids: set[int] = set()
    for class_name in CLASSES:
        class_ids = [
            image_id
            for image_id, annotations in annotations_by_image.items()
            if any(item["class_name"] == class_name for item in annotations)
        ]
        rng.shuffle(class_ids)
        selected_ids.update(class_ids[:per_class])
        if len(class_ids) < per_class:
            print(
                f"Warning: requested {per_class} {class_name} source images, "
                f"but COCO has only {len(class_ids)} matching images.",
                file=sys.stderr,
            )

    selected_candidates = {
        image_id: candidates[image_id] for image_id in selected_ids
    }
    selected_annotations = {
        image_id: annotations_by_image[image_id] for image_id in selected_ids
    }
    return selected_candidates, selected_annotations


def split_source_images(
    annotations_by_image: dict[int, list[dict[str, Any]]], seed: int
) -> dict[int, str]:
    labels = {
        image_id: {
            annotation["class_name"] for annotation in annotations
        }
        for image_id, annotations in annotations_by_image.items()
    }
    totals = {
        class_name: sum(class_name in image_labels for image_labels in labels.values())
        for class_name in CLASSES
    }
    targets = {
        split: {
            class_name: totals[class_name] * ratio
            for class_name in CLASSES
        }
        for split, ratio in SPLIT_RATIOS.items()
    }
    counts = {
        split: {class_name: 0 for class_name in CLASSES}
        for split in SPLIT_RATIOS
    }

    rng = random.Random(seed)
    image_ids = list(labels)
    rng.shuffle(image_ids)
    image_ids.sort(key=lambda image_id: -len(labels[image_id]))
    assignments: dict[int, str] = {}
    for image_id in image_ids:
        image_labels = labels[image_id]
        split = min(
            SPLIT_RATIOS,
            key=lambda candidate: (
                sum(
                    counts[candidate][label]
                    / max(targets[candidate][label], 1)
                    for label in sorted(image_labels)
                ),
                rng.random(),
            ),
        )
        assignments[image_id] = split
        for label in image_labels:
            counts[split][label] += 1

    for split in SPLIT_RATIOS:
        for class_name in CLASSES:
            if counts[split][class_name] == 0:
                raise RuntimeError(
                    f"Image-level split produced no {class_name} samples for {split}."
                )
    return assignments


def _license_text(metadata: dict[str, Any]) -> str:
    short_name = metadata.get("LicenseShortName", {}).get("value", "")
    return html.unescape(short_name).strip()


def _license_is_redistributable(license_name: str) -> bool:
    normalized = license_name.lower()
    if any(
        restriction in normalized
        for restriction in ("noncommercial", "no derivatives", "-nc", "-nd")
    ):
        return False
    return any(allowed in normalized for allowed in ALLOWED_LICENSES)


def _commons_search(query: str, offset: int) -> dict[str, Any]:
    parameters = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": "50",
        "gsroffset": str(offset),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "1200",
    }
    request_url = f"{COMMONS_API_URL}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(
        request_url, headers={"User-Agent": COMMONS_USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Wikimedia Commons search failed for {query!r}: {exc}") from exc


def _record_rejection(
    rejected: list[dict[str, Any]],
    category: str,
    reason: str,
    background_id: str = "",
    title: str = "",
    source_url: str = "",
    license_name: str = "",
) -> None:
    rejected.append(
        {
            "background_id": background_id,
            "category": category,
            "title": title,
            "source_url": source_url,
            "license": license_name,
            "reason": reason,
        }
    )


def _background_relevance_rejection(
    category: str, title: str, description: str = ""
) -> str:
    words = re.findall(r"[a-z0-9]+", f"{title} {description}".lower())
    if any(term in words for term in IRRELEVANT_TERMS):
        return "title indicates a drawing, map, logo, or other non-photo"
    category_terms = set(RELEVANCE_TERMS[category])
    other_terms = {
        term
        for other_category, terms in RELEVANCE_TERMS.items()
        if other_category != category
        for term in terms
    }
    if any(term in words for term in other_terms) and not any(
        term in words for term in category_terms
    ):
        return "title or description indicates a different background category"
    return ""


def _write_review_csv(
    path: Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_paged_contact_sheets(
    samples: list[tuple[Path, str]], directory: Path, prefix: str
) -> None:
    for page, start in enumerate(range(0, len(samples), 24), start=1):
        _make_contact_sheet(
            samples[start : start + 24],
            directory / f"{prefix}_{page:02d}.jpg",
        )


def _prepare_background_review(
    raw_root: Path,
    per_category: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    background_root = raw_root / "commons_backgrounds"
    background_root.mkdir(parents=True, exist_ok=True)
    manifest_path = background_root / "source_manifest.json"
    rejection_path = background_root / "background_rejections.csv"
    approvals_path = background_root / "background_approvals.csv"
    backgrounds: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    seen_pixel_hashes: dict[str, str] = {}
    seen_difference_hashes: list[tuple[int, str]] = []

    if manifest_path.exists():
        cached_items = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in cached_items:
            category = item["category"]
            if category not in BACKGROUNDS:
                raise RuntimeError(
                    f"Unexpected category {category!r} in {manifest_path}"
                )
            if not _license_is_redistributable(item.get("license", "")):
                _record_rejection(
                    rejected,
                    category,
                    "cached background license not in allowed set",
                    item.get("background_id", ""),
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            item["path"] = str(raw_root / item["relative_path"])
            if not Path(item["path"]).is_file():
                _record_rejection(
                    rejected,
                    category,
                    "cached background file is missing",
                    item.get("background_id", ""),
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            try:
                width, height = _validate_image(Path(item["path"]))
                item["width"], item["height"] = width, height
                pixel_hash, difference_hash, _ = _image_fingerprints(
                    Path(item["path"])
                )
            except (OSError, RuntimeError, UnidentifiedImageError) as exc:
                _record_rejection(
                    rejected,
                    category,
                    f"invalid cached image: {exc}",
                    item.get("background_id", ""),
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            if item["background_id"] in seen_page_ids:
                _record_rejection(
                    rejected,
                    category,
                    "duplicate Commons page ID",
                    item["background_id"],
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            if pixel_hash in seen_pixel_hashes or any(
                _hash_distance(difference_hash, prior_hash) <= 2
                for prior_hash, _ in seen_difference_hashes
            ):
                _record_rejection(
                    rejected,
                    category,
                    "duplicate or near-duplicate image content",
                    item["background_id"],
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            if min(width, height) < 256:
                _record_rejection(
                    rejected,
                    category,
                    "image is smaller than 256 pixels on one side",
                    item["background_id"],
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            relevance_reason = _background_relevance_rejection(
                category,
                item.get("title", ""),
                item.get("description", ""),
            )
            if relevance_reason:
                _record_rejection(
                    rejected,
                    category,
                    relevance_reason,
                    item["background_id"],
                    item.get("title", ""),
                    item.get("source_url", ""),
                    item.get("license", ""),
                )
                continue
            seen_page_ids.add(item["background_id"])
            seen_pixel_hashes[pixel_hash] = item["background_id"]
            seen_difference_hashes.append((difference_hash, item["background_id"]))
            item["pixel_sha256"] = pixel_hash
            item["difference_hash"] = f"{difference_hash:016x}"
            backgrounds.setdefault(category, []).append(item)
    else:
        all_metadata: list[dict[str, Any]] = []
        for category, query in BACKGROUNDS.items():
            category_root = background_root / category
            category_root.mkdir(parents=True, exist_ok=True)
            accepted: list[dict[str, Any]] = []
            offset = 0
            while len(accepted) < per_category and offset < 1000:
                result = _commons_search(query, offset)
                pages = result.get("query", {}).get("pages", {})
                if not pages:
                    break
                for page in pages.values():
                    page_id = str(page.get("pageid", ""))
                    title = page.get("title", "")
                    image_info = (page.get("imageinfo") or [{}])[0]
                    metadata = image_info.get("extmetadata", {})
                    license_name = _license_text(metadata)
                    source_url = image_info.get("descriptionurl", "")
                    image_url = image_info.get("thumburl") or image_info.get("url")
                    mime = image_info.get("mime", "")
                    if not page_id or page_id in seen_page_ids:
                        _record_rejection(
                            rejected, category, "missing or duplicate Commons page ID",
                            page_id, title, source_url, license_name,
                        )
                        continue
                    if not image_url or not mime.startswith("image/") or mime == "image/svg+xml":
                        _record_rejection(
                            rejected, category, "missing or unsupported image format",
                            page_id, title, source_url, license_name,
                        )
                        continue
                    if not _license_is_redistributable(license_name):
                        _record_rejection(
                            rejected, category, "license not in allowed set",
                            page_id, title, source_url, license_name,
                        )
                        continue
                    description = html.unescape(
                        metadata.get("ImageDescription", {}).get("value", "")
                    )
                    relevance_reason = _background_relevance_rejection(
                        category, title, description
                    )
                    if relevance_reason:
                        _record_rejection(
                            rejected, category, relevance_reason, page_id, title,
                            source_url, license_name,
                        )
                        continue

                    destination = category_root / f"{page_id}.jpg"
                    try:
                        download_file(
                            image_url,
                            destination,
                            COMMONS_USER_AGENT,
                            validate_image=True,
                            retries=3,
                            timeout=45,
                        )
                        width, height = _validate_image(destination)
                        if min(width, height) < 256:
                            raise RuntimeError(
                                f"image dimensions are too small: {width}x{height}"
                            )
                        pixel_hash, difference_hash, _ = _image_fingerprints(
                            destination
                        )
                    except (OSError, RuntimeError, UnidentifiedImageError) as exc:
                        _record_rejection(
                            rejected, category, f"invalid download: {exc}",
                            page_id, title, source_url, license_name,
                        )
                        continue

                    if pixel_hash in seen_pixel_hashes or any(
                        _hash_distance(difference_hash, prior_hash) <= 2
                        for prior_hash, _ in seen_difference_hashes
                    ):
                        _record_rejection(
                            rejected, category, "duplicate or near-duplicate image content",
                            page_id, title, source_url, license_name,
                        )
                        continue

                    item = {
                        "background_id": page_id,
                        "category": category,
                        "title": title,
                        "description": description,
                        "path": str(destination),
                        "relative_path": destination.relative_to(raw_root).as_posix(),
                        "source_url": source_url,
                        "download_url": image_url,
                        "license": license_name,
                        "artist": html.unescape(
                            metadata.get("Artist", {}).get("value", "")
                        ),
                        "credit": html.unescape(
                            metadata.get("Credit", {}).get("value", "")
                        ),
                        "attribution": html.unescape(
                            metadata.get("Attribution", {}).get("value", "")
                        ),
                        "width": width,
                        "height": height,
                        "pixel_sha256": pixel_hash,
                        "difference_hash": f"{difference_hash:016x}",
                    }
                    accepted.append(item)
                    all_metadata.append(item)
                    seen_page_ids.add(page_id)
                    seen_pixel_hashes[pixel_hash] = page_id
                    seen_difference_hashes.append((difference_hash, page_id))
                    if len(accepted) >= per_category:
                        break
                if "continue" not in result:
                    break
                offset += 50
            backgrounds[category] = accepted

        manifest_items = [
            {key: value for key, value in item.items() if key != "path"}
            for item in all_metadata
        ]
        manifest_path.write_text(
            json.dumps(manifest_items, indent=2), encoding="utf-8"
        )

    candidates: list[dict[str, Any]] = []
    for category in BACKGROUNDS:
        category_samples: list[tuple[Path, str]] = []
        for item in backgrounds.get(category, []):
            item["approved"] = ""
            item["approval_reason"] = ""
            candidates.append(item)
            category_samples.append(
                (
                    Path(item["path"]),
                    f"{category} | ID {item['background_id']} | "
                    f"{item['width']}x{item['height']}",
                )
            )
        _make_paged_contact_sheets(
            category_samples,
            background_root / "contact_sheets",
            f"background_candidates_{category}",
        )

    if not approvals_path.exists():
        _write_review_csv(
            approvals_path,
            [
                "background_id",
                "category",
                "approved",
                "approval_reason",
                "title",
                "source_url",
                "download_url",
                "license",
                "artist",
                "credit",
                "attribution",
                "width",
                "height",
                "relative_path",
            ],
            candidates,
        )
    _write_review_csv(
        rejection_path,
        ["background_id", "category", "title", "source_url", "license", "reason"],
        rejected,
    )
    return backgrounds, candidates, rejected


def download_backgrounds(
    raw_root: Path,
    per_category: int,
    seed: int,
    approvals_path: Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    backgrounds, candidates, auto_rejections = _prepare_background_review(
        raw_root, per_category, seed
    )
    if approvals_path is None or not approvals_path.is_file():
        raise RuntimeError(
            "Backgrounds are prepared for manual review. Inspect the contact sheets, "
            "set approved=yes/no in "
            f"{raw_root / 'commons_backgrounds' / 'background_approvals.csv'}, "
            "then rerun with --approve-backgrounds <that CSV path>."
        )

    approvals: dict[str, dict[str, str]] = {}
    with approvals_path.open(newline="", encoding="utf-8") as approval_file:
        for row in csv.DictReader(approval_file):
            background_id = row.get("background_id", "").strip()
            if not background_id or background_id in approvals:
                raise RuntimeError(
                    f"Missing or duplicate background_id in {approvals_path}"
                )
            approvals[background_id] = row

    selected: dict[str, list[dict[str, Any]]] = {category: [] for category in BACKGROUNDS}
    rejected = list(auto_rejections)
    candidate_by_id = {
        item["background_id"]: item for item in candidates
    }
    for background_id, approval in approvals.items():
        candidate = candidate_by_id.get(background_id)
        if candidate is None:
            raise RuntimeError(
                f"Approval references unknown background ID {background_id!r}."
            )
        if approval.get("category", candidate["category"]) != candidate["category"]:
            raise RuntimeError(
                f"Approval category does not match candidate {background_id!r}."
            )
        decision = approval.get("approved", "").strip().lower()
        reason = approval.get("approval_reason", "").strip()
        if decision in {"yes", "y", "true", "1"}:
            candidate["approved"] = "yes"
            selected[candidate["category"]].append(candidate)
        elif decision in {"no", "n", "false", "0"}:
            if not reason:
                reason = "manually rejected; no reason supplied"
            _record_rejection(
                rejected,
                candidate["category"],
                reason,
                background_id,
                candidate.get("title", ""),
                candidate.get("source_url", ""),
                candidate.get("license", ""),
            )
        elif decision:
            raise RuntimeError(
                f"Invalid approved value {decision!r} for {background_id}; "
                "use yes, no, or leave pending."
            )

    for category, items in selected.items():
        if len(items) < 6:
            raise RuntimeError(
                f"Only {len(items)} {category} backgrounds are approved; "
                "at least 6 are required (2 per split)."
            )

    rng = random.Random(seed)
    for items in selected.values():
        rng.shuffle(items)
        count = len(items)
        train_end = max(1, math.floor(count * SPLIT_RATIOS["train"]))
        val_end = max(train_end + 1, train_end + math.floor(count * SPLIT_RATIOS["val"]))
        if val_end >= count:
            train_end, val_end = count - 2, count - 1
        for item in items[:train_end]:
            item["split"] = "train"
        for item in items[train_end:val_end]:
            item["split"] = "val"
        for item in items[val_end:]:
            item["split"] = "test"

    _write_review_csv(
        raw_root / "commons_backgrounds" / "background_rejections.csv",
        ["background_id", "category", "title", "source_url", "license", "reason"],
        rejected,
    )
    selected_metadata = [
        item for category in BACKGROUNDS for item in selected[category]
    ]
    return selected, selected_metadata


def _decode_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8) * 255


def _composite(
    animal_path: Path,
    annotation: dict[str, Any],
    background_path: Path,
    image_size: int,
) -> Image.Image:
    with Image.open(animal_path) as source:
        animal = source.convert("RGB")
    mask_array = _decode_mask(annotation, animal.height, animal.width)
    ys, xs = np.nonzero(mask_array)
    if not len(xs):
        raise ValueError(f"Empty mask for COCO annotation {annotation['id']}")

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    foreground = animal.crop((left, top, right, bottom)).convert("RGBA")
    alpha = Image.fromarray(mask_array).crop((left, top, right, bottom))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))
    foreground.putalpha(alpha)

    max_width = int(image_size * 0.78)
    max_height = int(image_size * 0.78)
    scale = min(max_width / foreground.width, max_height / foreground.height)
    new_size = (
        max(1, round(foreground.width * scale)),
        max(1, round(foreground.height * scale)),
    )
    foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)

    with Image.open(background_path) as source:
        background = ImageOps.fit(
            source.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
        ).convert("RGBA")

    x = (image_size - foreground.width) // 2
    y = image_size - foreground.height - int(image_size * 0.06)
    background.alpha_composite(foreground, (x, max(0, y)))
    return background.convert("RGB")


def _choose_instance(
    annotations: list[dict[str, Any]], class_name: str
) -> dict[str, Any] | None:
    matching = [
        annotation
        for annotation in annotations
        if annotation["class_name"] == class_name
    ]
    return max(matching, key=lambda item: item.get("area", 0), default=None)


def _background_for(
    backgrounds: dict[str, list[dict[str, Any]]],
    category: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    choices = [
        item
        for item in backgrounds[category]
        if item.get("split") == split
    ]
    if not choices:
        raise RuntimeError(
            f"No {category} background images are assigned to split {split}."
        )
    return choices[index % len(choices)]


def _make_contact_sheet(
    samples: list[tuple[Path, str]], destination: Path, thumbnail_size: int = 180
) -> None:
    if not samples:
        return
    columns = 4
    cell_height = thumbnail_size + 24
    rows = math.ceil(min(len(samples), 24) / columns)
    sheet = Image.new("RGB", (columns * thumbnail_size, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image_path, caption) in enumerate(samples[:24]):
        with Image.open(image_path) as source:
            thumbnail = ImageOps.fit(
                source.convert("RGB"),
                (thumbnail_size, thumbnail_size),
                method=Image.Resampling.LANCZOS,
            )
        x = (index % columns) * thumbnail_size
        y = (index // columns) * cell_height
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 4, y + thumbnail_size + 4), caption[:42], fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def build_dataset(args: argparse.Namespace) -> None:
    if args.prepare_backgrounds:
        _, candidates, rejected = _prepare_background_review(
            args.raw.resolve(), args.backgrounds_per_category, args.seed
        )
        print(
            f"Prepared {len(candidates)} background candidates; "
            f"rejected {len(rejected)} automatically."
        )
        print(
            f"Review category contact sheets and edit "
            f"{args.raw.resolve() / 'commons_backgrounds' / 'background_approvals.csv'}."
        )
        return

    approval_path = args.approve_backgrounds or (
        args.raw.resolve()
        / "commons_backgrounds"
        / "background_approvals.csv"
    )
    if not approval_path.is_file():
        raise RuntimeError(
            "Backgrounds have not been manually approved. First run "
            "`python3 data_preparation.py --prepare-backgrounds`, inspect the "
            "candidate sheets, and mark approved=yes/no in "
            f"{approval_path}."
        )
    args.approve_backgrounds = approval_path

    output_root = args.output.resolve()
    if output_root.exists():
        raise RuntimeError(
            f"Output path already exists: {output_root}. "
            "Choose a new path; existing output will not be overwritten."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.staging-",
        dir=output_root.parent,
    ) as temporary_directory:
        staging_root = Path(temporary_directory) / "dataset"
        staging_root.mkdir()
        _build_dataset_in_directory(args, staging_root)
        _validate_staged_dataset(staging_root, args.image_size)
        output_root.mkdir()
        try:
            staging_root.replace(output_root)
        except OSError:
            output_root.rmdir()
            raise
    print(f"Dataset created at: {output_root}")


def _build_dataset_in_directory(
    args: argparse.Namespace, output_root: Path
) -> None:
    raw_root = args.raw.resolve()

    backgrounds, background_metadata = download_backgrounds(
        raw_root,
        args.backgrounds_per_category,
        args.seed,
        args.approve_backgrounds,
    )
    candidates, annotations_by_image = load_coco_candidates(
        raw_root, args.animals_per_class, args.seed
    )
    counts_by_class = {
        class_name: sum(
            any(item["class_name"] == class_name for item in annotations)
            for annotations in annotations_by_image.values()
        )
        for class_name in CLASSES
    }
    if min(counts_by_class.values()) < 6:
        raise RuntimeError(
            f"Too few animal source images after filtering: {counts_by_class}"
        )

    image_splits = split_source_images(annotations_by_image, args.seed)
    image_root = output_root / "images"
    for split in SPLIT_RATIOS:
        (image_root / split).mkdir(parents=True, exist_ok=True)

    coco_root = raw_root / "coco" / "images"
    rows: list[dict[str, Any]] = []
    contact_sheets: dict[
        str, dict[str, list[tuple[Path, str]]]
    ] = defaultdict(dict)
    train_instances_by_class: dict[str, list[tuple[int, dict[str, Any]]]] = {
        class_name: [] for class_name in CLASSES
    }
    for image_id, annotations in annotations_by_image.items():
        for class_name in CLASSES:
            instance = _choose_instance(annotations, class_name)
            if instance is not None:
                train_instances_by_class[class_name].append((image_id, instance))

    for class_name, items in train_instances_by_class.items():
        items = [
            item for item in items if image_splits[item[0]] == "train"
        ]
        rng = random.Random(f"{args.seed}:{class_name}")
        rng.shuffle(items)
        preferred_count = round(len(items) * 0.90)
        preferred_category = PREFERRED_BACKGROUND[class_name]
        alternative_categories = [
            category for category in BACKGROUNDS if category != preferred_category
        ]
        category_assignments = [preferred_category] * preferred_count
        category_assignments.extend(
            alternative_categories[index % len(alternative_categories)]
            for index in range(len(items) - preferred_count)
        )
        rng.shuffle(category_assignments)

        for index, ((image_id, annotation), background_category) in enumerate(
            zip(items, category_assignments)
        ):
            split = "train"
            image_info = candidates[image_id]
            source_path = _animal_source_path(image_info, coco_root)
            background = _background_for(
                backgrounds, background_category, "train", index
            )
            _write_variant(
                image_id,
                annotation,
                class_name,
                split,
                background,
                source_path,
                image_root,
                args.image_size,
                rows,
                contact_sheets,
                counterfactual_group=f"coco-{image_id}-{annotation['id']}",
                variant="biased_train",
            )

    for split in ("val", "test"):
        for class_name in CLASSES:
            items = [
                (image_id, _choose_instance(annotations, class_name))
                for image_id, annotations in annotations_by_image.items()
                if image_splits[image_id] == split
            ]
            items = [(image_id, item) for image_id, item in items if item is not None]
            items.sort(key=lambda item: item[0])
            for index, (image_id, annotation) in enumerate(items):
                image_info = candidates[image_id]
                source_path = _animal_source_path(image_info, coco_root)
                for background_category in BACKGROUNDS:
                    background = _background_for(
                        backgrounds, background_category, split, index
                    )
                    _write_variant(
                        image_id,
                        annotation,
                        class_name,
                        split,
                        background,
                        source_path,
                        image_root,
                        args.image_size,
                        rows,
                        contact_sheets,
                        counterfactual_group=(
                            f"coco-{image_id}-{annotation['id']}"
                        ),
                        variant="counterfactual",
                    )

    _validate_generated_rows(rows)
    _write_csv(output_root / "metadata.csv", rows)
    _write_csv(output_root / "background_sources.csv", background_metadata)
    config = {
        "seed": args.seed,
        "image_size": args.image_size,
        "requested_animals_per_class": args.animals_per_class,
        "source_images_per_class": counts_by_class,
        "class_to_label": CLASSES,
        "training_preferred_background": PREFERRED_BACKGROUND,
        "training_preferred_fraction": 0.90,
        "splits": SPLIT_RATIOS,
        "animal_source": "COCO 2017 instance annotations",
        "animal_license_note": (
            "Image-specific license fields are in metadata.csv; verify terms "
            "before redistributing generated images."
        ),
        "background_source": "Wikimedia Commons API; source licenses in background_sources.csv",
        "background_images_are_split_disjoint": True,
    }
    (output_root / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    for split, cells in contact_sheets.items():
        samples = [
            sample
            for cell in sorted(cells)
            for sample in cells[cell]
        ]
        _make_paged_contact_sheets(
            samples,
            output_root / "visualizations",
            f"composites_{split}",
        )

    print(f"Generated image variants: {len(rows)}")
    print(f"Metadata: {output_root / 'metadata.csv'}")
    print(f"Sample contact sheets: {output_root / 'visualizations'}")


def _animal_source_path(image_info: dict[str, Any], coco_root: Path) -> Path:
    filename = image_info["file_name"]
    coco_url = image_info.get("coco_url", "")
    match = re.search(r"/(train2017|val2017)/", coco_url)
    source_split = match.group(1) if match else image_info["source_split"]
    destination = coco_root / source_split / filename
    if not destination.exists():
        url = f"{COCO_IMAGE_BASE_URL}/{source_split}/{urllib.parse.quote(filename)}"
        download_file(
            url,
            destination,
            "ShortcutLearningDatasetBuilder/1.0",
            validate_image=True,
            retries=3,
            timeout=60,
        )
    else:
        _validate_image(destination)
    return destination


def _write_variant(
    image_id: int,
    annotation: dict[str, Any],
    class_name: str,
    split: str,
    background: dict[str, Any],
    source_path: Path,
    image_root: Path,
    image_size: int,
    rows: list[dict[str, Any]],
    contact_sheets: dict[str, dict[str, list[tuple[Path, str]]]],
    counterfactual_group: str,
    variant: str,
) -> None:
    label = CLASSES[class_name]
    filename = (
        f"{class_name}_{image_id}_{annotation['id']}_"
        f"{background['category']}_{background['background_id']}.jpg"
    )
    destination = image_root / split / filename
    if not destination.exists():
        composite = _composite(
            source_path, annotation, Path(background["path"]), image_size
        )
        composite.save(destination, quality=92, optimize=True)
    with Image.open(source_path) as animal_image:
        animal_width, animal_height = animal_image.size
    with Image.open(background["path"]) as background_image:
        background_width, background_height = background_image.size
    rows.append(
        {
            "path": destination.relative_to(image_root.parent).as_posix(),
            "split": split,
            "label": label,
            "class_name": class_name,
            "source_image_id": image_id,
            "source_annotation_id": annotation["id"],
            "animal_license_id": annotation["animal_license_id"],
            "animal_license_name": annotation["animal_license_name"],
            "animal_license_url": annotation["animal_license_url"],
            "animal_source_url": annotation["animal_source_url"],
            "animal_source_width": annotation["animal_source_width"],
            "animal_source_height": annotation["animal_source_height"],
            "animal_download_width": animal_width,
            "animal_download_height": animal_height,
            "background_id": background["background_id"],
            "background_category": background["category"],
            "background_title": background.get("title", ""),
            "background_width": background_width,
            "background_height": background_height,
            "background_license": background["license"],
            "background_source_url": background["source_url"],
            "background_download_url": background.get("download_url", ""),
            "background_artist": background.get("artist", ""),
            "background_credit": background.get("credit", ""),
            "background_attribution": background.get("attribution", ""),
            "image_width": image_size,
            "image_height": image_size,
            "counterfactual_group": counterfactual_group,
            "variant": variant,
        }
    )
    cell = f"{class_name}_{background['category']}"
    cell_samples = contact_sheets[split].setdefault(cell, [])
    if len(cell_samples) < 4:
        cell_samples.append(
            (
                destination,
                f"{class_name} / {background['category']} | "
                f"src {image_id} / bg {background['background_id']}",
            )
        )


def _validate_generated_rows(rows: list[dict[str, Any]]) -> None:
    source_splits: dict[int, set[str]] = defaultdict(set)
    background_splits: dict[str, set[str]] = defaultdict(set)
    counterfactuals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_splits[row["source_image_id"]].add(row["split"])
        background_splits[row["background_id"]].add(row["split"])
        if row["split"] in ("val", "test") and row.get("counterfactual_group"):
            counterfactuals[row["counterfactual_group"]].append(row)

    if any(len(splits) != 1 for splits in source_splits.values()):
        raise RuntimeError("A source animal image appears in more than one split.")
    if any(len(splits) != 1 for splits in background_splits.values()):
        raise RuntimeError("A background source image appears in more than one split.")
    for group_id, variants in counterfactuals.items():
        if {row["background_category"] for row in variants} != set(BACKGROUNDS):
            raise RuntimeError(
                f"Counterfactual group {group_id} does not contain all backgrounds."
            )
        if len({row["source_image_id"] for row in variants}) != 1:
            raise RuntimeError(
                f"Counterfactual group {group_id} mixes animal source images."
            )
        if len({row["class_name"] for row in variants}) != 1:
            raise RuntimeError(f"Counterfactual group {group_id} mixes classes.")

    for class_name, preferred_category in PREFERRED_BACKGROUND.items():
        training_rows = [
            row
            for row in rows
            if row["split"] == "train" and row["class_name"] == class_name
        ]
        if not training_rows:
            raise RuntimeError(f"No training rows were generated for {class_name}.")
        preferred_fraction = sum(
            row["background_category"] == preferred_category
            for row in training_rows
        ) / len(training_rows)
        if abs(preferred_fraction - 0.90) > 0.05:
            raise RuntimeError(
                f"{class_name} training correlation is {preferred_fraction:.1%}, "
                "outside the expected 90% +/- 5% range."
            )

    for split in ("val", "test"):
        cell_counts = {
            (class_name, category): sum(
                row["split"] == split
                and row["class_name"] == class_name
                and row["background_category"] == category
                for row in rows
            )
            for class_name in CLASSES
            for category in BACKGROUNDS
        }
        for class_name in CLASSES:
            counts = [
                cell_counts[(class_name, category)] for category in BACKGROUNDS
            ]
            if not counts[0] or len(set(counts)) != 1:
                raise RuntimeError(
                    f"{split} is not balanced across backgrounds for {class_name}: "
                    f"{counts}"
                )


def _validate_staged_dataset(output_root: Path, image_size: int) -> None:
    metadata_path = output_root / "metadata.csv"
    if not metadata_path.is_file():
        raise RuntimeError("Staged dataset is missing metadata.csv.")
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    if not rows:
        raise RuntimeError("Staged dataset metadata contains no samples.")
    _validate_generated_rows(rows)

    for row in rows:
        image_path = output_root / row["path"]
        width, height = _validate_image(image_path)
        if (width, height) != (image_size, image_size):
            raise RuntimeError(
                f"Generated image has unexpected dimensions {width}x{height}: "
                f"{image_path}"
            )
        if int(row["label"]) != CLASSES[row["class_name"]]:
            raise RuntimeError(
                f"Label mismatch in staged metadata for {image_path}."
            )
    for split in SPLIT_RATIOS:
        if not any(row["split"] == split for row in rows):
            raise RuntimeError(f"Staged dataset is missing {split} samples.")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and create a reproducible cow/horse dataset with "
            "biased training and balanced counterfactual evaluation variants."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded source data (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shortcut_dataset"),
        help="New output directory (must not already exist).",
    )
    parser.add_argument(
        "--animals-per-class",
        type=int,
        default=500,
        help="Maximum unique COCO source images per class (default: 500).",
    )
    parser.add_argument(
        "--backgrounds-per-category",
        type=int,
        default=30,
        help="Maximum Commons backgrounds per category (default: 30).",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare-backgrounds",
        action="store_true",
        help=(
            "Download and filter only background candidates, create review sheets, "
            "and stop before downloading COCO animal data."
        ),
    )
    parser.add_argument(
        "--approve-backgrounds",
        type=Path,
        help=(
            "CSV edited after --prepare-backgrounds; only rows marked approved=yes "
            "are used to create the dataset."
        ),
    )
    args = parser.parse_args()
    if args.animals_per_class < 10:
        parser.error(
            "--animals-per-class must be at least 10 to support all splits "
            "and an approximately 90% biased training ratio."
        )
    if args.backgrounds_per_category < 6:
        parser.error("--backgrounds-per-category must be at least 6.")
    if args.image_size < 64:
        parser.error("--image-size must be at least 64.")
    if args.prepare_backgrounds and args.approve_backgrounds:
        parser.error(
            "--prepare-backgrounds and --approve-backgrounds cannot be used together."
        )
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
