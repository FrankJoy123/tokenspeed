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
import tempfile
import unittest
from pathlib import Path

from prefill_steady import steady_window
from run_point import benchmark_cache_environment, benchmark_environment


class PrefillWindowTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "request_ids": [f"cohort-request-{i}"],
                "step": i,
                "global_decodes": [0],
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
            active_lanes=1,
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
            ("global_decodes", [1]),
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

    def test_intermediate_nonmonotonic_completion_is_rejected(self):
        rows = self.rows()
        rows[9]["completed_ns"] = rows[8]["completed_ns"] - 1
        # The final timestamp still exceeds the first: total elapsed alone
        # must not accept an invalid individual completion interval.
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.select(rows)

    def test_slow_interval_is_retained_in_raw_values_and_distribution(self):
        rows = self.rows()
        for row in rows[10:]:
            row["completed_ns"] += 900_000_000
        result = self.select(rows)
        self.assertEqual(result["step_ids"], list(range(5, 16)))
        self.assertEqual(len(result["endpoint_completed_ns"]), 11)
        self.assertEqual(result["interval_ns"].count(100_000_000), 9)
        self.assertEqual(result["interval_ns"].count(1_000_000_000), 1)
        self.assertEqual(result["window_seconds"], 1.9)
        self.assertEqual(result["step_latency_ms"], 190)
        distribution = result["interval_ms_distribution"]
        self.assertEqual(distribution["min"], 100)
        self.assertEqual(distribution["p50"], 100)
        self.assertAlmostEqual(distribution["p95"], 595)
        self.assertEqual(distribution["max"], 1000)

    def test_stale_host_override_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark_environment(
                {"TOKENSPEED_HOST_BINARY": "/old/ts-host"},
                script_dir=Path(__file__).parent,
                expert_routing="natural",
                indexer="bf16",
            )

    def test_simulated_routing_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark_environment(
                {},
                script_dir=Path(__file__).parent,
                expert_routing="uniform",
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
            expert_routing="natural",
            indexer="native-fp4",
        )
        self.assertEqual(fp4["TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER"], "1")
        self.assertNotIn("TOKENSPEED_MOE_ROUTING_SIMULATION", fp4)

    def test_two_points_give_real_children_separate_cache_workspaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            inherited = dict(
                os.environ,
                FLASHINFER_WORKSPACE_BASE="/stale/flashinfer",
                XDG_CACHE_HOME="/stale/xdg",
            )
            children = []
            for name in ("point-a", "point-b"):
                point = Path(tmp) / name
                env = benchmark_cache_environment(inherited, out=point)
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
                self.assertEqual(
                    actual["FLASHINFER_WORKSPACE_BASE"],
                    str(point / "cache/flashinfer-workspace"),
                )
                self.assertEqual(actual["XDG_CACHE_HOME"], str(point / "cache/xdg"))
                for key in (
                    "DG_JIT_CACHE_DIR",
                    "DG_SM120_JIT_CACHE_DIR",
                    "EP_JIT_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "CUTE_DSL_CACHE_DIR",
                    "TOKENSPEED_KERNEL_CACHE_ROOT",
                    "FLASHINFER_WORKSPACE_BASE",
                    "XDG_CACHE_HOME",
                ):
                    self.assertTrue(Path(actual[key]).is_relative_to(point / "cache"))
                children.append(actual)
            self.assertNotEqual(
                children[0]["FLASHINFER_WORKSPACE_BASE"],
                children[1]["FLASHINFER_WORKSPACE_BASE"],
            )
            self.assertEqual(
                inherited["FLASHINFER_WORKSPACE_BASE"], "/stale/flashinfer"
            )


class PrefillAuditTests(unittest.TestCase):
    def point(self, root, *, layout="cp8"):
        samples = root / "samples"
        samples.mkdir()
        dp = 1 if layout == "cp8" else 8
        config = {
            "routing": "natural",
            "redundant_experts": 0,
            "eplb": False,
            "dp": dp,
            "active_attention_lanes": dp,
            "cp": 8 // dp,
            "requests_per_round": 64 * dp,
            "prefix_sha256": "fixture-prefix",
            "benchmark": {
                "rounds": 1,
                "warmup_steps": 5,
                "measure_steps": 10,
                "prefix_tokens": 126976,
                "new_tokens": 4096,
                "computed_rows_per_request": 4224,
                "prefill_rows_per_rank": 4224,
            },
            "sampling": {
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": -1,
                "max_new_tokens": 1,
            },
        }
        (samples / "configuration.json").write_text(json.dumps(config))
        (root / "terminal.json").write_text('{"returncode": 0}')
        (samples / "state.json").write_text('{"stage": "complete"}')
        logs = []
        for iteration in (0, 1):
            prime = {
                "prompt_tokens": 126976,
                "cached_tokens": 0,
                "completion_tokens": 1,
                "finish_reason": "length",
            }
            (samples / f"r{iteration}-primes.json").write_text(json.dumps([prime] * dp))
            requests = {}
            for index in range(64 * dp):
                rank, i = divmod(index, 64)
                rid = f"steady-s4224-r{iteration}-request-{index}"
                requests[rid] = {
                    "prompt_tokens": 131072,
                    "cached_tokens": 126976,
                    "completion_tokens": 1,
                    "finish_reason": "length",
                }
                step = iteration * 64 + i
                row = {
                    "rank": rank,
                    "request_ids": [rid],
                    "step": step,
                    "global_decodes": [0] * dp,
                    "num_extends": 1,
                    "prefill_rows": 4224,
                    "global_rows": [4224] * dp,
                    "replay_rows": 128,
                    "new_rows": 4096,
                    "serving_compiles": 3,
                    "completed_ns": (step + 1) * 100_000_000,
                }
                logs.append(
                    "[2026-01-01 00:00:00] PREFILL_BENCHMARK " + json.dumps(row)
                )
            (samples / f"r{iteration}-requests.json").write_text(json.dumps(requests))
        (root / "engine.log").write_text("\n".join(logs) + "\n")
        return config

    def audit(self, root):
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("audit_point.py")),
                "--point",
                str(root),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_one_warm_queue_and_one_formal_queue_are_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.point(root)
            result = self.audit(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "samples/accepted-summary.json").read_text())
            self.assertEqual(summary["formal_cohorts"], 1)
            self.assertEqual(summary["step_ms"], 100)
            self.assertEqual(summary["new_tokens_s_gpu"], 5120)
            self.assertEqual(len(summary["trials"]), 1)
            self.assertFalse((root / "samples/r2-requests.json").exists())

    def test_dp8_audits_all_lanes_without_qcp_division(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.point(root, layout="dp8")
            result = self.audit(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "samples/accepted-summary.json").read_text())
            self.assertEqual(summary["new_tokens_s_gpu"], 40960)
            self.assertEqual(len(summary["trials"][0]["ranks"]), 8)
            (root / "samples/accepted-summary.json").unlink()
            path = root / "engine.log"
            path.write_text(
                "\n".join(
                    line
                    for line in path.read_text().splitlines()
                    if '"rank": 7' not in line
                )
                + "\n"
            )
            self.assertNotEqual(self.audit(root).returncode, 0)
            self.assertFalse((root / "samples/accepted-summary.json").exists())

    def test_wrong_protocol_or_unflushed_prime_is_rejected(self):
        for corruption in ("two-formal", "sampling", "prime-cache"):
            with self.subTest(corruption=corruption):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    config = self.point(root)
                    if corruption == "two-formal":
                        config["benchmark"]["rounds"] = 2
                    elif corruption == "sampling":
                        config["sampling"]["temperature"] = 0.0
                    else:
                        path = root / "samples/r1-primes.json"
                        rows = json.loads(path.read_text())
                        rows[0]["cached_tokens"] = 256
                        path.write_text(json.dumps(rows))
                    (root / "samples/configuration.json").write_text(json.dumps(config))
                    result = self.audit(root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((root / "samples/accepted-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
