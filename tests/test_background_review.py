import csv
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, call, patch
from urllib.error import HTTPError

from PIL import Image

from background_review import _search, prepare_backgrounds


def _http_error(status: int, retry_after: str | None = None) -> HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(
        "https://commons.wikimedia.org/w/api.php",
        status,
        "temporary failure",
        headers,
        io.BytesIO(),
    )


def _success_response(payload: bytes = b'{"query": {"pages": {}}}') -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = payload
    return response


class CommonsSearchRetryTests(unittest.TestCase):
    def test_empty_manifest_triggers_fresh_search(self):
        def result_for_category(query: str, offset: int) -> dict:
            category = next(
                category for category in ("grass", "sand", "snow")
                if category in query
            )
            page_id = {"grass": 101, "sand": 102, "snow": 103}[category]
            return {
                "query": {
                    "pages": {
                        str(page_id): {
                            "pageid": page_id,
                            "title": f"File:{category}.jpg",
                            "imageinfo": [
                                {
                                    "url": f"https://example.test/{category}.jpg",
                                    "thumburl": f"https://example.test/{category}-thumb.jpg",
                                    "descriptionurl": f"https://example.test/wiki/{category}",
                                    "mime": "image/jpeg",
                                    "extmetadata": {
                                        "LicenseShortName": {"value": "CC BY 4.0"},
                                    },
                                }
                            ],
                        }
                    }
                }
            }

        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            backgrounds_dir = raw_dir / "backgrounds"
            backgrounds_dir.mkdir()
            (backgrounds_dir / "candidates.json").write_text("[]", encoding="utf-8")
            with patch(
                "background_review._search",
                side_effect=result_for_category,
            ) as search, patch(
                "background_review._download_image",
                return_value=(1200, 800),
            ), patch("background_review._contact_sheets"), patch(
                "background_review.time.sleep"
            ):
                candidates = prepare_backgrounds(raw_dir, 1, 42)

            self.assertEqual(search.call_count, 3)
            self.assertEqual(
                {candidate["category"] for candidate in candidates},
                {"grass", "sand", "snow"},
            )
            self.assertTrue((backgrounds_dir / "background_approvals.csv").is_file())

    def test_nonempty_manifest_is_reused_without_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            items = []
            for index, category in enumerate(("grass", "sand", "snow"), start=201):
                image_path = raw_dir / "backgrounds" / category / f"{index}.jpg"
                image_path.parent.mkdir(parents=True)
                Image.new("RGB", (300, 300), "green").save(image_path)
                items.append(
                    {
                        "background_id": str(index),
                        "category": category,
                        "title": f"File:{category}.jpg",
                        "path": image_path.relative_to(raw_dir).as_posix(),
                        "source_url": f"https://example.test/wiki/{category}",
                        "download_url": f"https://example.test/{category}.jpg",
                        "license": "CC BY 4.0",
                        "width": 300,
                        "height": 300,
                    }
                )
            manifest = raw_dir / "backgrounds" / "candidates.json"
            manifest.write_text(json.dumps(items), encoding="utf-8")
            approvals_path = raw_dir / "backgrounds" / "background_approvals.csv"
            approvals_path.write_text(
                "background_id,category,approved,title,source_url,download_url,"
                "license,artist,credit,attribution,width,height,path\n",
                encoding="utf-8",
            )

            with patch("background_review._search") as search:
                candidates = prepare_backgrounds(raw_dir, 1, 42)

            search.assert_not_called()
            self.assertEqual(len(candidates), 3)
            for category in ("grass", "sand", "snow"):
                self.assertTrue(
                    (raw_dir / "backgrounds" / f"backgrounds_{category}_01.jpg").is_file()
                )
            with approvals_path.open(newline="", encoding="utf-8") as source:
                approval_rows = list(csv.DictReader(source))
            self.assertEqual(len(approval_rows), 3)

    def test_contact_sheet_resolves_relative_and_absolute_candidate_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            output_dir = raw_dir / "backgrounds"
            relative_path = raw_dir / "backgrounds" / "grass" / "301.jpg"
            absolute_path = raw_dir / "backgrounds" / "sand" / "302.jpg"
            for image_path in (relative_path, absolute_path):
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (300, 300), "green").save(image_path)
            candidates = [
                {
                    "background_id": "301",
                    "category": "grass",
                    "path": "backgrounds/grass/301.jpg",
                },
                {
                    "background_id": "302",
                    "category": "sand",
                    "path": str(absolute_path),
                },
            ]

            from background_review import _contact_sheets

            _contact_sheets(candidates, output_dir, raw_dir)

            self.assertTrue((output_dir / "backgrounds_grass_01.jpg").is_file())
            self.assertTrue((output_dir / "backgrounds_sand_01.jpg").is_file())

    def test_missing_cached_image_fails_with_candidate_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            backgrounds_dir = raw_dir / "backgrounds"
            backgrounds_dir.mkdir()
            items = []
            for index, category in enumerate(("grass", "sand", "snow"), start=401):
                image_path = raw_dir / "backgrounds" / category / f"{index}.jpg"
                image_path.parent.mkdir()
                Image.new("RGB", (300, 300), "green").save(image_path)
                items.append(
                    {
                        "background_id": str(index),
                        "category": category,
                        "title": f"File:{category}.jpg",
                        "path": image_path.relative_to(raw_dir).as_posix(),
                        "source_url": f"https://example.test/wiki/{category}",
                        "license": "CC BY 4.0",
                    }
                )
            missing_id = items[0]["background_id"]
            (raw_dir / items[0]["path"]).unlink()
            (backgrounds_dir / "candidates.json").write_text(
                json.dumps(items), encoding="utf-8"
            )

            with patch(
                "background_review._search",
                return_value={"query": {"pages": {}}},
            ), self.assertRaisesRegex(RuntimeError, r"grass=0/1"):
                prepare_backgrounds(raw_dir, 1, 42)

            rejection_text = (
                backgrounds_dir / "background_rejections.csv"
            ).read_text(encoding="utf-8")
            self.assertIn(missing_id, rejection_text)
            self.assertIn("missing cached image", rejection_text)

    def test_approved_cache_searches_only_for_category_deficits(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            backgrounds_dir = raw_dir / "backgrounds"
            candidates = []
            approvals = []
            background_id = 601
            approved_ids = {
                "grass": {"601", "602", "603", "604"},
                "sand": {"607", "608", "609", "610", "611", "612"},
                "snow": {"613", "614", "615", "616", "617"},
            }
            for category in ("grass", "sand", "snow"):
                category_dir = backgrounds_dir / category
                category_dir.mkdir(parents=True)
                for _ in range(6):
                    image_path = category_dir / f"{background_id}.jpg"
                    Image.new("RGB", (300, 300), "green").save(image_path)
                    item = {
                        "background_id": str(background_id),
                        "category": category,
                        "title": f"File:{category}-{background_id}.jpg",
                        "path": image_path.relative_to(raw_dir).as_posix(),
                        "source_url": f"https://example.test/wiki/{background_id}",
                        "download_url": f"https://example.test/{background_id}.jpg",
                        "license": "CC BY 4.0",
                        "width": 300,
                        "height": 300,
                    }
                    candidates.append(item)
                    approvals.append(
                        {
                            **item,
                            "approved": (
                                "yes" if str(background_id) in approved_ids[category]
                                else "no"
                            ),
                            "path": str(image_path),
                        }
                    )
                    background_id += 1
            manifest = backgrounds_dir / "candidates.json"
            manifest.write_text(json.dumps(candidates), encoding="utf-8")
            approvals_path = backgrounds_dir / "background_approvals.csv"
            with approvals_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=approvals[0].keys())
                writer.writeheader()
                writer.writerows(approvals)

            def search_result(query: str, offset: int) -> dict:
                category = next(name for name in approved_ids if name in query)
                page_id = {
                    ("grass", 0): 701,
                    ("grass", 50): 702,
                    ("snow", 0): 703,
                }[(category, offset)]
                page = {
                    "pageid": page_id,
                    "title": f"File:{category}-extra-{page_id}.jpg",
                    "imageinfo": [
                        {
                            "url": f"https://example.test/{page_id}.jpg",
                            "thumburl": f"https://example.test/{page_id}-thumb.jpg",
                            "descriptionurl": f"https://example.test/wiki/{page_id}",
                            "mime": "image/jpeg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY 4.0"},
                            },
                        }
                    ],
                }
                result = {"query": {"pages": {str(page_id): page}}}
                if category == "grass" and offset == 0:
                    result["continue"] = {"gsroffset": 50}
                return result

            def save_image(url: str, destination: Path) -> tuple[int, int]:
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (300, 300), "blue").save(destination)
                return 300, 300

            with patch(
                "background_review._search", side_effect=search_result
            ) as search, patch(
                "background_review._download_image", side_effect=save_image
            ):
                result = prepare_backgrounds(raw_dir, 6, 42)

            self.assertEqual(
                search.call_args_list,
                [
                    call("grass meadow field landscape", 0),
                    call("grass meadow field landscape", 50),
                    call("snow field winter landscape", 0),
                ],
            )
            self.assertEqual(len(result), 21)
            self.assertEqual(
                len({
                    item["background_id"]
                    for item in result
                    if item["category"] == "grass"
                }),
                8,
            )
            with approvals_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            preserved = {row["background_id"]: row["approved"] for row in rows}
            for old_id, status in (
                ("601", "yes"),
                ("605", "no"),
                ("607", "yes"),
                ("612", "yes"),
                ("613", "yes"),
                ("618", "no"),
            ):
                self.assertEqual(preserved[old_id], status)
            self.assertEqual(
                sum(row["category"] == "grass" for row in rows), 8
            )
            self.assertEqual(
                sum(row["category"] == "sand" for row in rows), 6
            )
            self.assertEqual(
                sum(row["category"] == "snow" for row in rows), 7
            )

    def test_api_error_response_fails_with_details(self):
        response = _success_response(
            b'{"error": {"code": "ratelimited", "info": "Too many requests"}}'
        )
        with patch(
            "background_review.urllib.request.urlopen", return_value=response
        ), patch("background_review.time.sleep"):
            with self.assertRaisesRegex(
                RuntimeError, 'ratelimited.*Too many requests'
            ):
                _search("grass", 0)

    def test_missing_or_malformed_search_structure_fails_clearly(self):
        invalid_responses = (
            (b'{"batchcomplete": ""}', "expected an object at 'query'"),
            (b'{"query": {}}', "expected 'query.pages'"),
            (b'{"query": {"pages": []}}', "expected 'query.pages'"),
            (b'{"query": {"pages": {"1": "invalid"}}}', "page objects"),
        )
        for payload, message in invalid_responses:
            with self.subTest(payload=payload):
                with patch(
                    "background_review.urllib.request.urlopen",
                    return_value=_success_response(payload),
                ), patch("background_review.time.sleep"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        _search("grass", 0)

    def test_valid_empty_search_response_is_accepted(self):
        payload = b'{"query": {"pages": {}}}'
        with patch(
            "background_review.urllib.request.urlopen",
            return_value=_success_response(payload),
        ), patch("background_review.time.sleep"):
            self.assertEqual(_search("grass", 0), {"query": {"pages": {}}})

    def test_zero_accepted_images_fails_without_approval_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            with patch(
                "background_review._search",
                return_value={"query": {"pages": {}}},
            ) as search, patch("background_review._contact_sheets"), patch(
                "background_review.time.sleep"
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "zero usable images.*no approval CSV"
                ):
                    prepare_backgrounds(raw_dir, 1, 42)

            self.assertEqual(search.call_count, 3)
            self.assertFalse(
                (raw_dir / "backgrounds" / "background_approvals.csv").exists()
            )
            self.assertFalse(
                (raw_dir / "backgrounds" / "candidates.json").exists()
            )
            self.assertTrue(
                (raw_dir / "backgrounds" / "background_rejections.csv").is_file()
            )

    def test_jpeg_mime_is_requested_and_accepted_as_candidate(self):
        category_ids = {"grass": 123, "sand": 124, "snow": 125}

        def api_response(request, timeout):
            query = parse_qs(urlparse(request.full_url).query)["gsrsearch"][0]
            category = next(name for name in category_ids if name in query)
            image_page = {
                "pageid": category_ids[category],
                "title": f"File:{category}.jpg",
                "imageinfo": [
                    {
                        "url": f"https://example.test/{category}.jpg",
                        "thumburl": f"https://example.test/{category}-thumb.jpg",
                        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{category}.jpg",
                        "mime": "image/jpeg",
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY 4.0"},
                        },
                    }
                ],
            }
            return _success_response(
                json.dumps(
                    {"query": {"pages": {str(category_ids[category]): image_page}}}
                ).encode()
            )

        def save_image(url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (300, 300), "green").save(destination)
            return (300, 300)

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "background_review.urllib.request.urlopen",
                side_effect=api_response,
            ) as urlopen, patch(
                "background_review._download_image", side_effect=save_image
            ), patch(
                "background_review.time.sleep"
            ):
                candidates = prepare_backgrounds(Path(temporary), 1, 42)

            for call in urlopen.call_args_list:
                request_url = call.args[0].full_url
                self.assertEqual(
                    parse_qs(urlparse(request_url).query)["iiprop"],
                    ["url|mime|extmetadata"],
                )
            self.assertEqual(len(candidates), 3)
            self.assertEqual(
                {candidate["category"] for candidate in candidates},
                {"grass", "sand", "snow"},
            )
            self.assertIn(
                "https://example.test/grass-thumb.jpg",
                {candidate["download_url"] for candidate in candidates},
            )
            self.assertTrue(
                (
                    Path(temporary)
                    / "backgrounds"
                    / "backgrounds_grass_01.jpg"
                ).is_file()
            )

    def test_429_respects_retry_after_and_retries(self):
        response = _success_response()
        with patch(
            "background_review.urllib.request.urlopen",
            side_effect=[_http_error(429, "3.5"), response],
        ) as urlopen, patch("background_review.time.sleep") as sleep:
            result = _search("grass", 0)

        self.assertIn("query", result)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 3.5])
        sleep.assert_any_call(3.5)

    def test_temporary_server_error_uses_exponential_backoff(self):
        response = _success_response()
        with patch(
            "background_review.urllib.request.urlopen",
            side_effect=[_http_error(503), response],
        ) as urlopen, patch("background_review.time.sleep") as sleep:
            result = _search("snow", 50)

        self.assertIn("query", result)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 1])

    def test_retry_limit_is_bounded(self):
        with patch(
            "background_review.urllib.request.urlopen",
            side_effect=_http_error(502),
        ) as urlopen, patch("background_review.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 retries"):
                _search("sand", 100)

        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1.0, 1, 2, 4],
        )


if __name__ == "__main__":
    unittest.main()
