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

"""Frozen image inputs and selection must be independent of ground-truth answers."""

import gzip
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import mmmu_pro
import pytest


def test_selection_keeps_multi_image_coverage_and_subject_diversity():
    rows = [
        {
            "id": str(i),
            "subject": subject,
            "answer": "A",
            "image_1": {},
            "image_2": {} if multi else None,
        }
        for i, (subject, multi) in enumerate(
            [
                ("Art", True),
                ("Art", False),
                ("Biology", True),
                ("Chemistry", False),
                ("Math", False),
            ]
        )
    ]
    selected = mmmu_pro.select_rows(rows, count=4, multi_min=2)
    assert len({r["id"] for r in selected}) == 4
    assert sum(r["image_2"] is not None for r in selected) == 2
    assert len({r["subject"] for r in selected}) == 4
    changed = deepcopy(rows)
    for r in changed:
        r["answer"] = "J"
    assert [r["id"] for r in selected] == [
        r["id"] for r in mmmu_pro.select_rows(changed, count=4, multi_min=2)
    ]


def _sample(tmp_path):
    images = {}
    for index, raw in [("1", b"first-image"), ("2", b"second-image")]:
        path = tmp_path / f"{index}.png"
        path.write_bytes(raw)
        images[index] = dict(
            path=path.name, sha256=hashlib.sha256(raw).hexdigest(), mime="image/png"
        )
    return dict(
        question="Compare <image 2> with <image 1>.",
        images=images,
        options=["yes", "no"],
        answer="B",
    )


def test_payload_preserves_interleaved_images_and_does_not_send_ground_truth(tmp_path):
    sample = _sample(tmp_path)
    actual = mmmu_pro.messages(sample, tmp_path, prompt_style="direct")
    content = actual[0]["content"]
    assert [p["type"] for p in content[:5]] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert content[1]["image_url"]["url"].endswith("c2Vjb25kLWltYWdl")
    changed = deepcopy(sample)
    changed.update(answer="A", explanation="Never send this reference explanation")
    assert actual == mmmu_pro.messages(changed, tmp_path, prompt_style="direct")


def test_image_checksum_changes_are_refused(tmp_path):
    sample = _sample(tmp_path)
    (tmp_path / "1.png").write_bytes(b"changed-image")
    with pytest.raises(ValueError, match="checksum changed"):
        mmmu_pro.messages(sample, tmp_path, prompt_style="direct")


def _official(tmp_path):
    source = Path(__file__).resolve().parents[1] / "ref" / "official-evaluate.py.gz"
    raw = gzip.decompress(source.read_bytes())
    (tmp_path / "evaluate.py").write_bytes(raw)
    manifest = {"scorer_files": {"evaluate.py": hashlib.sha256(raw).hexdigest()}}
    return mmmu_pro.official_scorer(tmp_path, manifest), manifest


def test_official_score_counts_truncation_zero_and_preserves_random_state(tmp_path):
    scorer, _ = _official(tmp_path)
    sample = dict(id="one", options=["yes", "no"], answer="A")
    result = dict(ok=True, content="A", finish_reason="stop")
    state = random.getstate()
    assert mmmu_pro.score_result(result, sample, scorer, seed=42)["correct"]
    result["finish_reason"] = "length"
    assert not mmmu_pro.score_result(result, sample, scorer, seed=42)["correct"]
    result.update(finish_reason="stop", content="unparseable")
    first = mmmu_pro.score_result(result, sample, scorer, seed=42)
    assert first == mmmu_pro.score_result(result, sample, scorer, seed=42)
    assert first["parse_random_fallback"]
    assert random.getstate() == state


def test_public_rerun_audit_rebuilds_request_and_rejects_changed_answer(tmp_path):
    from summarize_mmmu_pro_reruns import audit

    scorer, manifest = _official(tmp_path)
    sample = _sample(tmp_path)
    sample.update(id="one", answer="A")
    manifest["samples"] = [sample]
    protocol = dict(
        repeats=1,
        model="test-model",
        prompt_style="direct",
        temperature=0,
        max_tokens=4096,
        seed=42,
        reasoning_effort="none",
    )
    payload = dict(
        model=protocol["model"],
        messages=mmmu_pro.messages(sample, tmp_path, prompt_style="direct"),
        temperature=0,
        max_tokens=4096,
        stream=False,
        seed=42,
        reasoning_effort="none",
    )
    result = dict(
        id="one",
        repeat=0,
        ok=True,
        content="A",
        finish_reason="stop",
        response={"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]},
        request_sha256=hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode()
        ).hexdigest(),
    )
    result.update(mmmu_pro.score_result(result, sample, scorer, seed=42))
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "responses.jsonl").write_text(json.dumps(result) + "\n")
    (directory / "summary.json").write_text(
        json.dumps(dict(correct=1, output_truncated=0, errors=0))
    )
    assert audit(directory, protocol, tmp_path, manifest) == dict(
        correct=1, total=1, truncated=0
    )
    result["response"]["choices"][0]["message"]["content"] = "B"
    (directory / "responses.jsonl").write_text(json.dumps(result) + "\n")
    with pytest.raises(ValueError, match="Saved answer differs"):
        audit(directory, protocol, tmp_path, manifest)
