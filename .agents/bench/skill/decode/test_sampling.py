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

"""Behavioral checks for CUSTOM95 input identity and accepted timing windows."""

import json
import os
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path

from audit import collect_node, strict_window, validate_fixed_width
from inputs import common_prompt, digest, request_tokens


class SamplingTests(unittest.TestCase):
    def samples(self, tag, *, batch=1, rank=0, offset=0):
        return [
            {
                "rank": rank,
                "step": i + offset,
                "all_decode": True,
                "num_extends": 0,
                "global_decodes": [batch] * 16,
                "request_ids": [f"{tag}-lane{rank}-r{j}" for j in range(batch)],
                "input_rows": 6 * batch,
                "output_lengths": [6] * batch,
                "compiles": 4,
                "completed_ns": (i + offset) * 10_000_000,
            }
            for i in range(191)
        ]

    def window(self, samples, *, prefix="b1-r1-lane", batch=1):
        return strict_window(
            samples, prefix, batch, warmup=110, steps=60, world_size=16
        )

    def test_common_prefix_and_independent_cohort_suffixes(self):
        vocab = list(range(512))
        prompt = common_prompt(vocab, context=7000, seed=37)
        seen = set()
        for batch in [1, 2, 4, 8, 16, 32]:
            for rank in range(16):
                for index in range(batch):
                    for cohort in [0, 1, 2]:
                        values = request_tokens(
                            prompt, vocab, batch, rank, index, cohort=cohort
                        )
                        self.assertEqual(values[:-6656], prompt[:-6656])
                        h = digest(values[-6656:])
                        self.assertNotIn(h, seen)
                        seen.add(h)
        self.assertEqual(len(seen), 3024)
        values = request_tokens(prompt, vocab, 1, 0, 0, cohort=1)
        self.assertEqual(values, request_tokens(prompt, vocab, 1, 0, 0, cohort=1))
        full = common_prompt(vocab, context=131072, seed=37)
        self.assertEqual(len(request_tokens(full, vocab, 1, 0, 0, cohort=1)), 131072)

    def test_formal_window_excludes110_full_points(self):
        samples = self.samples("b1-r1")
        result = self.window(samples)
        self.assertEqual(result["first_ns"], 1_100_000_000)
        self.assertEqual(result["last_ns"], 1_700_000_000)
        self.assertEqual(result["step_ms"], 10)
        self.assertEqual(result["actual_kept_tokens_per_request_step"], 6)
        validate_fixed_width(samples, result, 1)

    def test_warmup_and_independent_rounds_do_not_mix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source-sha256.json").write_text("{}")
            rows = (
                self.samples("b1-r0")
                + self.samples("b1-r1", offset=200)
                + self.samples("b1-r2", offset=400)
            )
            (root / "steps-1.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            windows = collect_node(root, 16)["windows"]
            self.assertEqual(set(windows), {"b1-r1", "b1-r2"})
            self.assertLess(
                windows["b1-r1"][0]["last_ns"], windows["b1-r2"][0]["first_ns"]
            )

    def test_incomplete_work_and_compile_changes_are_rejected(self):
        for field, value in [
            ("output_lengths", [5]),
            ("input_rows", 5),
            ("global_decodes", [1] * 15 + [0]),
            ("compiles", None),
        ]:
            samples = self.samples("b1-r1")
            samples[120][field] = value
            with self.assertRaises(RuntimeError):
                self.window(samples)
        samples = self.samples("b1-r1")
        samples[130]["compiles"] = 5
        with self.assertRaises(RuntimeError):
            self.window(samples)
        samples = self.samples("b1-r1")
        samples[120]["step"] += 1
        with self.assertRaises(RuntimeError):
            self.window(samples)

    def test_shell_child_receives_fixed_sm120_transport_without_stale_profiling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            envdir = root / "venv"
            venv.EnvBuilder(with_pip=False).create(envdir)
            inherited = dict(
                os.environ,
                RUN_DIR=str(root / "run"),
                MODEL_PATH=str(root / "model"),
                CC="/usr/bin/true",
                CXX="/usr/bin/true",
                TOKENSPEED_EXPERIMENT_SM120_DEEPEP="0",
                TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER="1",
                TOKENSPEED_MOE_ROUTING_SIMULATION="uniform",
                PREFILL_BENCH_OBSERVE="1",
                TOKENSPEED_PREFILL_BENCHMARK="1",
                TOKENSPEED_TRACE_HOST_STAGES="1",
                TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING="1",
                TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK="stale",
            )
            command = [
                "bash",
                str(Path(__file__).with_name("engine.sh")),
                "--print-env",
                "w4a8",
                "0",
                "test",
                "256",
                "fixed",
                "16",
                "2",
                "127.0.0.1",
                "32051",
                "128",
                str(envdir),
                "7200",
            ]
            actual = json.loads(
                subprocess.check_output(command, env=inherited, text=True, timeout=15)
            )
            rejected = subprocess.run(
                command,
                env=dict(inherited, TOKENSPEED_HOST_BINARY="/old/ts-host"),
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("TOKENSPEED_HOST_BINARY", rejected.stderr)
            self.assertEqual(actual["TOKENSPEED_EXPERIMENT_SM120_DEEPEP"], "1")
            self.assertEqual(actual["TOKENSPEED_SM120_DEEPEP_WIRE"], "elastic_fp8_bf16")
            self.assertEqual(actual["TOKENSPEED_SM120_DEEPEP_CAPACITY"], "128")
            self.assertEqual(actual["TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN"], "6")
            self.assertEqual(actual["TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER"], "0")
            self.assertEqual(actual["TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP"], "1")
            for name in [
                "TOKENSPEED_MOE_ROUTING_SIMULATION",
                "PREFILL_BENCH_OBSERVE",
                "TOKENSPEED_PREFILL_BENCHMARK",
                "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING",
                "TOKENSPEED_TRACE_HOST_STAGES",
                "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK",
            ]:
                self.assertIsNone(actual[name])


if __name__ == "__main__":
    unittest.main()
