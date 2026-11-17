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

"""Reject incomplete, discontinuous, or compiling Prefill timing windows."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from prefill_steady import steady_window
from run_point import benchmark_environment


class PrefillWindowTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "request_ids": [f"cohort-request-{i}"],
                "step": i,
                "all_extend": True,
                "num_extends": 1,
                "prefill_rows": 4224,
                "global_rows": [4224],
                "replay_rows": 128,
                "new_rows": 4096,
                "serving_compiles": 3,
                "completed_ns": i * 100_000_000,
            }
            for i in range(16)
        ]

    def select(self, rows):
        return steady_window(
            rows,
            request_prefix="cohort-request-",
            rows=4224,
            dp_size=1,
            warmup_steps=5,
            measure_steps=10,
        )

    def test_replay_is_excluded_from_new_token_throughput(self):
        result = self.select(self.rows())
        self.assertEqual(result["window_steps"], 10)
        self.assertEqual(result["window_seconds"], 1)
        self.assertEqual(result["step_latency_ms"], 100)
        self.assertEqual(result["computed_rows"], 42240)
        self.assertEqual(result["replay_rows"], 1280)
        self.assertEqual(result["new_tokens_s_gpu"], 40960)

    def test_short_discontinuous_or_partial_work_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select(self.rows()[:-1])
        for field, value in [
            ("global_rows", [4223]),
            ("prefill_rows", 4223),
            ("all_extend", False),
        ]:
            rows = self.rows()
            rows[8][field] = value
            with self.assertRaises(ValueError):
                self.select(rows)
        rows = self.rows()
        rows[8]["step"] += 1
        with self.assertRaises(ValueError):
            self.select(rows)

    def test_compile_and_row_accounting_failures_are_rejected(self):
        rows = self.rows()
        rows[-1]["serving_compiles"] = 4
        with self.assertRaises(ValueError):
            self.select(rows)
        rows = self.rows()
        rows[8]["serving_compiles"] = None
        with self.assertRaises(ValueError):
            self.select(rows)
        rows = self.rows()
        rows[8]["new_rows"] = 4095
        with self.assertRaises(ValueError):
            self.select(rows)

    def test_stale_host_override_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark_environment(
                {"TOKENSPEED_HOST_BINARY": "/old/ts-host"},
                script_dir=Path(__file__).parent,
                expert_routing="natural",
                indexer="bf16",
            )

    def test_child_process_receives_explicit_transport_and_clean_observations(self):
        inherited = dict(
            os.environ,
            TOKENSPEED_EXPERIMENT_SM120_DEEPEP="0",
            TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="6",
            DECODE_BENCH_OBSERVE="1",
            TOKENSPEED_PREFILL_BENCHMARK="1",
            TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING="1",
            TOKENSPEED_MOE_ROUTING_SIMULATION="uniform",
            TOKENSPEED_TRACE_HOST_STAGES="1",
            TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK="stale",
        )
        env = benchmark_environment(
            inherited,
            script_dir=Path(__file__).parent,
            expert_routing="natural",
            indexer="bf16",
        )
        raw = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json,os; print(json.dumps(dict(os.environ)))",
            ],
            env=env,
            text=True,
            timeout=10,
        )
        actual = json.loads(raw)
        self.assertEqual(actual["TOKENSPEED_EXPERIMENT_SM120_DEEPEP"], "1")
        self.assertEqual(actual["TOKENSPEED_SM120_DEEPEP_WIRE"], "elastic_fp8_bf16")
        self.assertEqual(actual["TOKENSPEED_SM120_DEEPEP_CAPACITY"], "2048")
        self.assertEqual(actual["TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP"], "0")
        self.assertEqual(actual["TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER"], "0")
        for name in [
            "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN",
            "DECODE_BENCH_OBSERVE",
            "TOKENSPEED_PREFILL_BENCHMARK",
            "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING",
            "TOKENSPEED_MOE_ROUTING_SIMULATION",
            "TOKENSPEED_TRACE_HOST_STAGES",
            "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK",
        ]:
            self.assertNotIn(name, actual)
        fp4 = benchmark_environment(
            inherited,
            script_dir=Path(__file__).parent,
            expert_routing="uniform",
            indexer="native-fp4",
        )
        self.assertEqual(fp4["TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER"], "1")
        self.assertEqual(fp4["TOKENSPEED_MOE_ROUTING_SIMULATION"], "uniform")


if __name__ == "__main__":
    unittest.main()
