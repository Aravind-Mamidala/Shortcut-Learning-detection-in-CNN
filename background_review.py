"""Download a small set of Commons backgrounds for manual approval."""

from __future__ import annotations

import csv
import email.utils
import html
import json
import math
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_REQUEST_DELAY_SECONDS = 1.0
COMMONS_MAX_RETRIES = 3
COMMONS_BACKOFF_MAX_SECONDS = 8.0
COMMONS_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
APPROVALS_NAME = "background_approvals.csv"
SEARCH_TERMS = {
    "grass": "grass meadow field landscape",
    "sand": "sand desert dunes landscape",
    "snow": "snow field winter landscape",
}
ALLOWED_LICENSES = ("cc0", "public domain", "cc by", "cc-by")
APPROVAL_FIELDS = (
    "background_id",
    "category",
    "approved",
    "title",
    "source_url",
    "download_url",
    "license",
    "artist",
    "credit",
    "attribution",
    "width",
    "height",
    "path",
)


def _license_allowed(name: str) -> bool:
    normalized = name.lower()
    if "noncommercial" in normalized or "no derivatives" in normalized:
        return False
    return any(item in normalized for item in ALLOWED_LICENSES)


def _search(query: str, offset: int) -> dict[str, Any]:
    parameters = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": "50",
        "gsroffset": str(offset),
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "1200",
    }
    request = urllib.request.Request(
        f"{COMMONS_API_URL}?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": "ShortcutLearningPilot/1.0"},
    )
    time.sleep(COMMONS_REQUEST_DELAY_SECONDS)
    for attempt in range(COMMONS_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in COMMONS_RETRYABLE_STATUS_CODES:
                raise RuntimeError(
                    f"Commons search failed for {query!r}: {exc}"
                ) from exc
            if attempt == COMMONS_MAX_RETRIES:
                raise RuntimeError(
                    f"Commons search failed for {query!r} after "
                    f"{COMMONS_MAX_RETRIES} retries: {exc}"
                ) from exc
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            delay = (
                retry_after
                if retry_after is not None
                else min(2**attempt, COMMONS_BACKOFF_MAX_SECONDS)
            )
            time.sleep(delay)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Commons search failed for {query!r}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Commons search returned malformed response for {query!r}: "
            "expected a JSON object."
        )
    if "error" in result:
        raise RuntimeError(
            f"Commons API error for {query!r}: "
            f"{json.dumps(result['error'], sort_keys=True)}"
        )
    query_result = result.get("query")
    if not isinstance(query_result, dict):
        raise RuntimeError(
            f"Commons search returned malformed response for {query!r}: "
            "expected an object at 'query'."
        )
    pages = query_result.get("pages")
    if not isinstance(pages, dict) or any(
        not isinstance(page, dict) for page in pages.values()
    ):
        raise RuntimeError(
            f"Commons search returned malformed response for {query!r}: "
            "expected 'query.pages' to be an object of page objects."
        )
    return result


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _download_image(url: str, destination: Path) -> tuple[int, int]:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing raw background: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "ShortcutLearningPilot/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            expected_length = response.headers.get("Content-Length")
            with temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Empty background download from {url}")
        if expected_length and temporary.stat().st_size != int(expected_length):
            raise RuntimeError(f"Incomplete background download from {url}")
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            image.load()
            dimensions = image.size
        if min(dimensions) < 256:
            raise RuntimeError(f"Background is too small: {dimensions}")
        temporary.replace(destination)
        return dimensions
    except (
        OSError,
        urllib.error.URLError,
        RuntimeError,
        ValueError,
        UnidentifiedImageError,
    ) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid background {url}: {exc}") from exc


def _candidate_image_path(item: dict[str, Any], raw_dir: Path) -> Path:
    path = Path(item["path"])
    return path if path.is_absolute() else raw_dir / path


def _contact_sheets(
    candidates: list[dict[str, Any]], output_dir: Path, raw_dir: Path | None = None
) -> None:
    image_root = raw_dir or output_dir.parent
    for category in SEARCH_TERMS:
        items = [item for item in candidates if item["category"] == category]
        for page, start in enumerate(range(0, len(items), 20), start=1):
            page_items = items[start : start + 20]
            columns, thumb_size = 4, 180
            cell_height = thumb_size + 38
            rows = math.ceil(len(page_items) / columns)
            sheet = Image.new("RGB", (columns * thumb_size, rows * cell_height), "white")
            draw = ImageDraw.Draw(sheet)
            for index, item in enumerate(page_items):
                image_path = _candidate_image_path(item, image_root)
                try:
                    with Image.open(image_path) as source:
                        source.load()
                        thumb = ImageOps.fit(
                            source.convert("RGB"),
                            (thumb_size, thumb_size),
                            method=Image.Resampling.LANCZOS,
                        )
                except (OSError, UnidentifiedImageError) as exc:
                    raise RuntimeError(
                        f"Cannot create contact sheet for background "
                        f"{item['background_id']} at {image_path}: {exc}"
                    ) from exc
                x = (index % columns) * thumb_size
                y = (index // columns) * cell_height
                sheet.paste(thumb, (x, y))
                draw.text(
                    (x + 3, y + thumb_size + 3),
                    f"{category} | ID {item['background_id']}"[:30],
                    fill="black",
                )
            sheet.save(
                output_dir / f"backgrounds_{category}_{page:02d}.jpg",
                quality=90,
            )


def prepare_backgrounds(
    raw_dir: Path, per_category: int, seed: int
) -> list[dict[str, Any]]:
    """Fetch candidates, record rejections, and make sheets plus approval CSV."""
    root = raw_dir / "backgrounds"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "candidates.json"
    rejection_path = root / "background_rejections.csv"
    approvals_path = root / APPROVALS_NAME
    rejections: list[dict[str, str]] = []

    cached_candidates = None
    if manifest_path.exists():
        cached_candidates = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cached_candidates:
        candidates = []
        for item in cached_candidates:
            path = _candidate_image_path(item, raw_dir)
            if not path.is_file():
                rejections.append(
                    {
                        "background_id": item["background_id"],
                        "category": item["category"],
                        "source_url": item["source_url"],
                        "reason": f"missing cached image: {path}",
                    }
                )
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as exc:
                rejections.append(
                    {
                        "background_id": item["background_id"],
                        "category": item["category"],
                        "source_url": item["source_url"],
                        "reason": f"corrupt cached image: {exc}",
                    }
                )
                continue
            if not _license_allowed(item["license"]):
                rejections.append(
                    {
                        "background_id": item["background_id"],
                        "category": item["category"],
                        "source_url": item["source_url"],
                        "reason": "license is not in the allowed set",
                    }
                )
                continue
            if min(width, height) < 256:
                rejections.append(
                    {
                        "background_id": item["background_id"],
                        "category": item["category"],
                        "source_url": item["source_url"],
                        "reason": f"cached image is too small: {(width, height)}",
                    }
                )
                continue
            item["width"], item["height"] = width, height
            candidates.append(item)
    else:
        candidates = []
    existing_approvals: dict[str, dict[str, str]] = {}
    if approvals_path.is_file():
        with approvals_path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                background_id = row.get("background_id", "").strip()
                if background_id:
                    existing_approvals[background_id] = row

    eligible_counts = {
        category: sum(
            item["category"] == category
            and existing_approvals.get(item["background_id"], {}).get(
                "approved", ""
            ).strip().lower()
            not in {"no", "n", "false", "0"}
            for item in candidates
        )
        for category in SEARCH_TERMS
    }
    seen_ids = {
        str(item.get("background_id", ""))
        for item in (cached_candidates or [])
    }
    seen_ids.update(item["background_id"] for item in candidates)
    for category, terms in SEARCH_TERMS.items():
        needed = max(0, per_category - eligible_counts[category])
        found = 0
        offset = 0
        while found < needed and offset < 1000:
            result = _search(terms, offset)
            pages = result["query"]["pages"]
            if not pages:
                break
            for page in pages.values():
                background_id = str(page.get("pageid", ""))
                title = page.get("title", "")
                info = (page.get("imageinfo") or [{}])[0]
                metadata = info.get("extmetadata", {})
                license_name = html.unescape(
                    metadata.get("LicenseShortName", {}).get("value", "")
                )
                source_url = info.get("descriptionurl", "")
                image_url = info.get("thumburl") or info.get("url")
                mime = info.get("mime", "")
                reason = ""
                if not background_id or background_id in seen_ids:
                    reason = "missing or duplicate Commons image ID"
                elif not image_url or not mime.startswith("image/") or mime == "image/svg+xml":
                    reason = "missing or unsupported image format"
                elif not _license_allowed(license_name):
                    reason = "license is not in the allowed set"
                if reason:
                    rejections.append(
                        {
                            "background_id": background_id,
                            "category": category,
                            "title": title,
                            "source_url": source_url,
                            "reason": reason,
                        }
                    )
                    continue

                path = raw_dir / "backgrounds" / category / f"{background_id}.jpg"
                if path.is_file():
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            image.load()
                            width, height = image.size
                        if min(width, height) < 256:
                            raise RuntimeError(
                                f"Existing image is too small: {(width, height)}"
                            )
                    except (OSError, UnidentifiedImageError, RuntimeError) as exc:
                        rejections.append(
                            {
                                "background_id": background_id,
                                "category": category,
                                "title": title,
                                "source_url": source_url,
                                "reason": f"invalid existing image {path}: {exc}",
                            }
                        )
                        continue
                else:
                    try:
                        width, height = _download_image(image_url, path)
                    except RuntimeError as exc:
                        rejections.append(
                            {
                                "background_id": background_id,
                                "category": category,
                                "title": title,
                                "source_url": source_url,
                                "reason": str(exc),
                            }
                        )
                        continue
                item = {
                    "background_id": background_id,
                    "category": category,
                    "title": title,
                    "path": path.relative_to(raw_dir).as_posix(),
                    "source_url": source_url,
                    "download_url": image_url,
                    "license": license_name,
                    "artist": html.unescape(metadata.get("Artist", {}).get("value", "")),
                    "credit": html.unescape(metadata.get("Credit", {}).get("value", "")),
                    "attribution": html.unescape(
                        metadata.get("Attribution", {}).get("value", "")
                    ),
                    "width": width,
                    "height": height,
                }
                candidates.append(item)
                seen_ids.add(background_id)
                found += 1
                if found >= needed:
                    break
            if "continue" not in result:
                break
            offset += 50
        if found < needed:
            print(
                f"Found {found} additional {category} candidates "
                f"(needed {needed})."
            )

    random.Random(seed).shuffle(candidates)
    with rejection_path.open("w", newline="", encoding="utf-8") as output:
        fields = ["background_id", "category", "title", "source_url", "reason"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejections)
    if not candidates:
        raise RuntimeError(
            "Background preparation found zero usable images across all "
            "categories; no approval CSV was generated."
        )
    counts = {category: 0 for category in SEARCH_TERMS}
    for item in candidates:
        counts[item["category"]] += 1
    insufficient = {
        category: count for category, count in counts.items() if count < per_category
    }
    if insufficient:
        details = ", ".join(
            f"{category}={count}/{per_category}"
            for category, count in insufficient.items()
        )
        raise RuntimeError(
            f"Insufficient valid background candidates ({details}); see "
            f"{rejection_path} for rejected or missing images."
        )
    manifest_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    approval_rows = dict(existing_approvals)
    for item in candidates:
        if item["background_id"] not in approval_rows:
            approval_rows[item["background_id"]] = {
                **item,
                "approved": "",
                "path": str(_candidate_image_path(item, raw_dir)),
            }
    with approvals_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=APPROVAL_FIELDS)
        writer.writeheader()
        writer.writerows(approval_rows.values())
    _contact_sheets(candidates, root, raw_dir)
    return candidates


def read_approved_backgrounds(
    raw_dir: Path,
    approvals_path: Path,
    per_category: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not approvals_path.is_file():
        raise RuntimeError(
            f"Missing background approval CSV: {approvals_path}. "
            "Run with --prepare-backgrounds first."
        )
    selected: dict[str, list[dict[str, Any]]] = {
        category: [] for category in SEARCH_TERMS
    }
    seen_ids: set[str] = set()
    with approvals_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            background_id = row.get("background_id", "").strip()
            category = row.get("category", "").strip()
            if row.get("approved", "").strip().lower() not in {"yes", "y", "true", "1"}:
                continue
            if category not in selected or not background_id:
                raise RuntimeError(f"Invalid approved background row: {row}")
            if background_id in seen_ids:
                raise RuntimeError(f"Duplicate approved background ID: {background_id}")
            if not _license_allowed(row.get("license", "")):
                raise RuntimeError(f"Approved background has an unsupported license: {row}")
            path = Path(row["path"])
            if not path.is_absolute():
                path = raw_dir / path
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            row.update(
                {
                    "background_id": background_id,
                    "category": category,
                    "path": str(path),
                    "source_url": row.get("source_url", ""),
                    "license": row.get("license", ""),
                    "width": width,
                    "height": height,
                }
            )
            selected[category].append(row)
            seen_ids.add(background_id)

    for category, items in selected.items():
        if len(items) < 6:
            raise RuntimeError(
                f"Approve at least 6 {category} backgrounds so all splits have "
                "held-out background images."
            )
        random.Random(f"{seed}:{category}").shuffle(items)
        items = items[:per_category]
        train_end = max(1, math.floor(len(items) * 0.70))
        val_end = max(train_end + 1, train_end + math.floor(len(items) * 0.15))
        if val_end >= len(items):
            train_end, val_end = len(items) - 2, len(items) - 1
        for item in items[:train_end]:
            item["split"] = "train"
        for item in items[train_end:val_end]:
            item["split"] = "val"
        for item in items[val_end:]:
            item["split"] = "test"
        selected[category] = items
    return selected
