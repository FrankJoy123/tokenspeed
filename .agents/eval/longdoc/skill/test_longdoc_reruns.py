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

"""Checks for fixed-question DocQA rerun statistics and protocol validation."""

import unittest
from copy import deepcopy

from summarize_longdoc_reruns import (
    PROTOCOL_FIELDS,
    score_statistics,
    validate_protocols,
)


class LongdocRerunTests(unittest.TestCase):
    def test_frozen_200_question_runs_and_incomplete_or_mixed_runs(self):
        protocol = {key: "same" for key in PROTOCOL_FIELDS}
        protocol.update(sample_ids=[str(i) for i in range(200)], max_tokens=4096)
        runs = [deepcopy(protocol) for _ in range(3)]
        validate_protocols(runs)
        runs[2]["sample_ids"].pop()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_protocols(runs)
        scores = [
            dict(
                expected=200,
                completed=200,
                errors=0,
                skipped=0,
                doc_qa_score=value,
                strict_doc_qa_score=value,
                output_truncated=0,
                wall_seconds=10,
            )
            for value in [0.4, 0.5, 0.6]
        ]
        result = score_statistics(scores, 200)
        self.assertEqual(result["questions_per_run"], 200)
        self.assertEqual(result["doc_qa_mean_percent"], 50)
        scores[2]["expected"] = scores[2]["completed"] = 50
        with self.assertRaises(ValueError):
            score_statistics(scores, 200)
        for score in scores:
            score["expected"] = score["completed"] = 50
        with self.assertRaises(ValueError):
            score_statistics(scores, 200)

    def test_partial_credit_statistics(self):
        scores = [
            dict(
                expected=50,
                completed=50,
                errors=0,
                skipped=0,
                doc_qa_score=value,
                strict_doc_qa_score=value,
                output_truncated=0,
                wall_seconds=10,
            )
            for value in [0.4, 0.5, 0.6]
        ]
        summary = score_statistics(scores, 50)
        self.assertEqual(summary["doc_qa_mean_percent"], 50)
        self.assertEqual(summary["doc_qa_sample_std_percentage_points"], 10)
        scores[1]["completed"] = 49
        with self.assertRaises(ValueError):
            score_statistics(scores, 50)

    def test_protocol_budget_mismatch_and_duplicate_ids_are_rejected(self):
        protocol = {key: "same" for key in PROTOCOL_FIELDS}
        protocol.update(sample_ids=[str(i) for i in range(50)], max_tokens=4096)
        runs = [deepcopy(protocol) for _ in range(3)]
        validate_protocols(runs)
        runs[1]["max_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            validate_protocols(runs)
        runs[1] = deepcopy(protocol)
        runs[2]["sample_ids"][1] = runs[2]["sample_ids"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_protocols(runs)


if __name__ == "__main__":
    unittest.main()
