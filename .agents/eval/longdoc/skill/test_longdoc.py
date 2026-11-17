# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Frozen inputs and source-compatible metric checks, without GPU dependencies."""

import base64
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import longdoc
from longdoc_scoring import eval_docqa_score, parse_output


class FrozenLongdocTests(unittest.TestCase):
    def test_ground_truth_and_evidence_do_not_change_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["document/abcdefg/one.jpg", "document/abcdefg/two.jpg"]
            hashes = {}
            for index, name in enumerate(names):
                path = root / "images" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b"\xff\xd8" + bytes([index])
                path.write_bytes(raw)
                hashes[name] = longdoc.digest(raw)
            row = {
                "doc_name": "abcdefg",
                "question": "Compare the figures.",
                "page_list": names,
                "answer": "SECRET_GROUND_TRUTH",
                "ans_page_list": [1],
            }
            before = longdoc.messages(row, root, hashes)
            changed = deepcopy(row)
            changed.update(answer="DIFFERENT", ans_page_list=[999])
            self.assertEqual(before, longdoc.messages(changed, root, hashes))
            self.assertNotIn("SECRET_GROUND_TRUTH", json.dumps(before))
            images = [
                x["image_url"]["url"]
                for x in before[0]["content"]
                if x["type"] == "image_url"
            ]
            self.assertEqual(
                [base64.b64decode(x.split(",", 1)[1]) for x in images],
                [b"\xff\xd8\x00", b"\xff\xd8\x01"],
            )

    def test_modified_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images/a.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed image"):
                longdoc.messages(
                    {"doc_name": "a", "question": "Q", "page_list": ["a.jpg"]},
                    root,
                    {"a.jpg": "incorrect"},
                )

    def test_frozen_50_contains_unique_multimage_ids(self):
        manifest = json.loads(longdoc.DEFAULT_MANIFEST.read_text())
        ids = [row["id"] for row in manifest["samples"]]
        self.assertEqual(len(ids), 50)
        self.assertEqual(len(set(ids)), 50)
        self.assertTrue(all(row["image_count"] > 1 for row in manifest["samples"]))

    def test_official_numeric_tolerance_and_list_partial_credit(self):
        self.assertEqual(eval_docqa_score("100", "Answer: 1%", "Float"), 1.0)
        self.assertEqual(eval_docqa_score("100", "101", "Integer"), 0.0)
        self.assertEqual(eval_docqa_score("[1, 2]", "1", "List"), 0.5)
        self.assertEqual(parse_output("Answer:Answer: 42"), "42")


if __name__ == "__main__":
    unittest.main()
