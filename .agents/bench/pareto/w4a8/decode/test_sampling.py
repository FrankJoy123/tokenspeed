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

"""Raw-step protocol fixtures: no GPU or elapsed-time filtering is simulated."""

import copy
import json
import os
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path

from analysis import (
    FORMAL_INTERVALS,
    SAMPLING,
    TAU,
    WARM_STEPS,
    align_global_dispatches,
    global_full_prefix,
    output_budget,
    warmup,
    window,
)
from audit import (
    audit_runs,
    collect_node,
    combine_nodes,
    sha,
    strict_window,
    validate_fixed_width,
    validate_run_identity,
)
from inputs import common_prompt, digest, request_tokens


class SamplingTests(unittest.TestCase):
    def samples(self, tag, count=154, *, batch=1, rank=0, offset=0):
        return [
            {
                "rank": rank,
                "step": i + offset,
                "all_decode": True,
                "num_extends": 0,
                "global_decodes": [batch] * 16,
                "global_batch": [batch] * 16,
                "request_ids": [f"{tag}-lane{rank}-r{j}" for j in range(batch)],
                "input_rows": 6 * batch,
                "output_lengths": [6] * batch,
                "prefill_lengths": [131072] * batch,
                "compiles": 4,
                "completed_ns": 1_000_000 + (i + offset) * 10_000_000,
                "context_min": 131072 + i * 6,
                "context_max": 131072 + i * 6,
                "ep_nic_name": f"nic{rank}",
                "nvshmem_hca_list": f"nic{rank}:1",
            }
            for i in range(count)
        ]

    def test_token_budget_accounts_for_prefill_and_is_not_evidence(self):
        for batch in (1, 2, 4, 8, 16, 32):
            for cohort, target in ((0, 200), (1, 151), (2, 151)):
                budget = output_budget(batch, cohort)
                self.assertEqual(budget["required_full_points"], target)
                self.assertEqual(budget["max_new_tokens"], 1 + 6 * (target + batch + 2))
        # 1200 output tokens = one Prefill token + 199 full keeps + a 5-token tail.
        rows = self.samples("b1-r0", count=199)
        rows.append(
            self.samples("b1-r0", count=1, offset=199)[0] | {"output_lengths": [5]}
        )
        self.assertEqual(1 + sum(sum(row["output_lengths"]) for row in rows), 1200)
        with self.assertRaisesRegex(RuntimeError, "200.*observed 199"):
            warmup(rows, "b1-r0-lane", 1, world_size=16)
        with self.assertRaises(RuntimeError):
            warmup(self.samples("b1-r0", count=199), "b1-r0-lane", 1, world_size=16)
        result = warmup(
            self.samples("b1-r0", count=203), "b1-r0-lane", 1, world_size=16
        )
        self.assertEqual(len(result["checked_steps"]), 200)

    def test_first_100_discarded_then_51_endpoints(self):
        rows = self.samples("b1-r1")
        result = strict_window(rows, "b1-r1-lane", 1)
        self.assertEqual(result["endpoint_steps"], list(range(100, 151)))
        self.assertEqual(len(result["interval_ns"]), 50)
        self.assertEqual(result["step_ms"], 10)
        self.assertEqual(result["actual_kept_tokens_per_request_step"], 6)
        self.assertEqual(result["deepinfer_tokens_s_user"], TAU / 0.01)
        self.assertEqual(result["actual_tokens_s_gpu"], 6 / 0.01)
        validate_fixed_width(rows, result, 1)

    def test_global_window_starts_after_last_ranks_first_actual_width6(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, _ = self.fixture(Path(temp))
            path = roots[1] / "steps-15.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            first = next(
                row
                for row in rows
                if row["request_ids"] == ["b1-r1-lane15-r0"] and row["all_decode"]
            )
            first["output_lengths"] = [4]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = audit_runs(roots)
            expected = list(range(first["step"] + 101, first["step"] + 152))
            for row in result["windows"]["b1-r1"].values():
                self.assertEqual(row["endpoint_steps"], expected)
                self.assertEqual(row["discarded_full_points"], 100)
            preview = collect_node(roots[0], 16, live=True)
            self.assertEqual(
                preview["windows"]["b1-r1"][0]["endpoint_steps"][0], expected[0] - 1
            )
            self.assertIn("pending", preview["acceptance_scope"])

    def test_global_counter_alignment_requires_every_rank_and_dispatch(self):
        raw = {rank: self.samples("b1-r1", rank=rank) for rank in range(16)}
        for mutation in ("rank", "counter", "metadata"):
            changed = copy.deepcopy(raw)
            if mutation == "rank":
                del changed[15]
            elif mutation == "counter":
                del changed[15][0]
            else:
                changed[15][10]["global_decodes"] = [0] * 16
            with self.assertRaises(RuntimeError):
                align_global_dispatches(changed)

    def test_global_rule_rejects_gap_and_never_selects_later_faster_segment(self):
        for key, value in (
            ("output_lengths", [4]),
            ("input_rows", 5),
            ("prefill_lengths", [131071]),
        ):
            raw = {
                rank: self.samples("b1-r1", count=400, rank=rank) for rank in range(16)
            }
            raw[15][120][key] = value
            # A later segment being long and faster cannot repair the first one.
            for rank in raw:
                for index, row in enumerate(raw[rank][121:], 121):
                    row["completed_ns"] = 1_300_000_000 + (index - 121) * 1_000_000
            with self.assertRaisesRegex(RuntimeError, "resumed"):
                global_full_prefix(align_global_dispatches(raw), "b1-r1-lane", 1)

    def test_global_warm_count_and_compile_checks_cover_last_rank(self):
        raw = {rank: self.samples("b1-r0", count=200, rank=rank) for rank in range(16)}
        raw[15][0]["output_lengths"] = [4]
        common = global_full_prefix(align_global_dispatches(raw), "b1-r0-lane", 1)
        with self.assertRaisesRegex(RuntimeError, "200.*observed 199"):
            warmup(common[0], "b1-r0-lane", 1, world_size=16)
        raw = {rank: self.samples("b1-r1", rank=rank) for rank in range(16)}
        raw[15][0]["output_lengths"] = [4]
        for row in raw[15][151:]:
            row["compiles"] = 5
        common = global_full_prefix(align_global_dispatches(raw), "b1-r1-lane", 1)
        with self.assertRaisesRegex(RuntimeError, "compilation inside"):
            strict_window(common[15], "b1-r1-lane", 1)

    def test_no_search_past_broken_or_underloaded_prefix(self):
        for key, value in (
            ("output_lengths", [5]),
            ("input_rows", 5),
            ("global_decodes", [1] * 15 + [0]),
            ("global_batch", [1] * 15 + [0]),
        ):
            rows = self.samples("b1-r1", count=400)
            rows[120][key] = value
            with self.assertRaises(RuntimeError):
                strict_window(rows, "b1-r1-lane", 1)
        for key, value in (("step", 999), ("completed_ns", 1)):
            rows = self.samples("b1-r1")
            rows[120][key] = value
            with self.assertRaises(RuntimeError):
                strict_window(rows, "b1-r1-lane", 1)

    def test_slow_intervals_are_retained(self):
        rows = self.samples("b1-r1")
        for row in rows[120:]:
            row["completed_ns"] += 990_000_000
        result = strict_window(rows, "b1-r1-lane", 1)
        self.assertIn(1_000_000_000, result["interval_ns"])
        self.assertEqual(result["interval_ms_distribution"]["max"], 1000)
        self.assertAlmostEqual(result["step_ms"], (49 * 10 + 1000) / 50)

    def test_compile_monitor_and_measured_compilation(self):
        for value in (None, -1):
            rows = self.samples("b1-r1")
            rows[120]["compiles"] = value
            with self.assertRaises(RuntimeError):
                strict_window(rows, "b1-r1-lane", 1)
        rows = self.samples("b1-r1")
        for row in rows[120:]:
            row["compiles"] = 5
        with self.assertRaisesRegex(RuntimeError, "compilation inside"):
            strict_window(rows, "b1-r1-lane", 1)
        # Discarded points remain recorded; an early compile is not moved or hidden.
        rows = self.samples("b1-r1")
        for row in rows[20:]:
            row["compiles"] = 5
        result = strict_window(rows, "b1-r1-lane", 1)
        self.assertEqual(result["cohort_compile_delta"], 1)
        self.assertEqual(result["measured_compile_delta"], 0)

    def fixture(self, base):
        roots = [base / "node0", base / "node1"]
        model = {
            "profile": "w4a8",
            "metadata_sha256": {"config.json": "a" * 64},
            "shards": {"part.safetensors": {"bytes": 80, "header_sha256": "b" * 64}},
            "indexed_tensors": 1,
            "scope": "metadata and tensor headers, not payload",
        }
        modules = {
            name: {"sha256": "d" * 64, "native_sha256": {}}
            for name in (
                "tokenspeed",
                "tokenspeed_kernel",
                "tokenspeed_engine",
                "tokenspeed_scheduler",
                "deep_gemm",
                "deep_gemm_sm120",
                "deep_ep",
                "flashinfer",
            )
        }
        fixed = {
            "source": {
                "commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "archive_sha256": "c" * 64,
            },
            "dependencies": {
                "packages": {"torch": "fixture"},
                "modules": modules,
                "host_binary_sha256": "e" * 64,
                "worker_source_sha256": {
                    "runtime_model": "f" * 64,
                    "indexer": "1" * 64,
                },
            },
            "model": {
                "checkpoint": "fixture-checkpoint",
                "headers_manifest_sha256": "",
            },
            "indexer": {
                "TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER": "1",
                "TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT": "1",
                "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER": "0",
                "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER": "0",
            },
            "network": {
                "wire": "elastic_fp8_bf16",
                "capacity": 128,
                "shared_overlap": 1,
                "nodes": [
                    {
                        "ranks": list(range(n * 8, n * 8 + 8)),
                        "nics": [f"nic{rank}" for rank in range(n * 8, n * 8 + 8)],
                        "gid_index": 7,
                        "socket_ifname": "eth0",
                        "nccl_ib_hca": ",".join(
                            f"nic{rank}" for rank in range(n * 8, n * 8 + 8)
                        ),
                    }
                    for n in range(2)
                ],
            },
        }
        for n, root in enumerate(roots):
            root.mkdir()
            (root / "model-identity.json").write_text(json.dumps(model))
            fixed["model"]["headers_manifest_sha256"] = sha(
                root / "model-identity.json"
            )
            (root / "source-sha256.json").write_text(
                json.dumps({"observer.py": "2" * 64})
            )
            (root / "runtime-identity.json").write_text(
                json.dumps(
                    {
                        "source_is_clean": True,
                        "source_commit": "a" * 40,
                        "source_tree_sha256": "b" * 64,
                        "packages": {"torch": "fixture"},
                        "modules": modules,
                        "host_binary": {"sha256": "e" * 64, "override": None},
                    }
                )
            )
            topology = fixed["network"]["nodes"][n]
            for rank in topology["ranks"]:
                records = []
                for cohort in (0, 1, 2):
                    budget = output_budget(1, cohort)
                    count = (budget["max_new_tokens"] - 1) // 6
                    rows = self.samples(
                        f"b1-r{cohort}", count=count + 1, rank=rank, offset=len(records)
                    )
                    rows[0].update(
                        all_decode=False,
                        num_extends=1,
                        global_decodes=[0] * 16,
                        input_rows=6656,
                        output_lengths=[1],
                    )
                    records += rows
                (root / f"steps-{rank}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in records)
                )
                env = {
                    "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN": "6",
                    "TOKENSPEED_MOE_ROUTING_SIMULATION": None,
                    "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING": None,
                    "TOKENSPEED_SM120_DEEPEP_WIRE": "elastic_fp8_bf16",
                    "TOKENSPEED_SM120_DEEPEP_CAPACITY": "128",
                    "TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP": "1",
                    "TOKENSPEED_SM120_DEEPEP_NICS": ",".join(topology["nics"]),
                    "NCCL_IB_GID_INDEX": "7",
                    "NVSHMEM_IB_GID_INDEX": "7",
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": "eth0",
                    "NCCL_IB_HCA": topology["nccl_ib_hca"],
                }
                env.update(
                    {
                        k: v
                        for k, v in fixed["indexer"].items()
                        if k.startswith("TOKENSPEED_")
                    }
                )
                (root / f"process-identity-{rank}.json").write_text(
                    json.dumps(
                        {
                            "selected_environment": env,
                            "source_sha256": fixed["dependencies"][
                                "worker_source_sha256"
                            ],
                        }
                    )
                )
        for root in roots:
            (root / "run-identity.json").write_text(json.dumps(fixed))
        config = {
            "protocol": "decode",
            "protocol_version": 1,
            "batches": [1],
            "world_size": 16,
            "rounds": 2,
            "sampling": SAMPLING,
            "warm_steps": 200,
            "formal_skip": 100,
            "formal_intervals": 50,
            "run_identity_sha256": sha(roots[0] / "run-identity.json"),
        }
        (roots[0] / "config.json").write_text(json.dumps(config))
        vocab = list(range(16))
        prompt = common_prompt(vocab)
        (roots[0] / "input-identity.json").write_text(
            json.dumps(
                {
                    "excluded_token_ids": [],
                    "vocab_size": 16,
                    "seed": 37,
                    "prefix_sha256": digest(prompt[:-6656]),
                    "prefix_tokens": 124416,
                    "prime_prompt_tokens": 124672,
                    "prime_prompt_sha256": digest(prompt[:-6656] + prompt[-6656:-6400]),
                }
            )
        )
        for cohort in (0, 1, 2):
            tag = f"b1-r{cohort}"
            requests = {}
            suffixes = {}
            fulls = {}
            for rank in range(16):
                rid = f"{tag}-lane{rank}-r0"
                values = request_tokens(prompt, vocab, 1, rank, 0, cohort=cohort)
                requests[rid] = {
                    "prompt_tokens": 131072,
                    "cached_tokens": 124416,
                    "completion_tokens": output_budget(1, cohort)["max_new_tokens"],
                    "finish_reason": "length",
                }
                suffixes[rid] = digest(values[-6656:])
                fulls[rid] = digest(values)
            (roots[0] / f"{tag}-requests.json").write_text(json.dumps(requests))
            (roots[0] / f"{tag}-input-hashes.json").write_text(
                json.dumps(
                    {
                        "suffix_sha256_by_request": suffixes,
                        "input_sha256_by_request": fulls,
                    }
                )
            )
        (roots[0] / "common-prime-requests.json").write_text(
            json.dumps(
                {
                    f"common-prime-lane{rank}-r0": {
                        "prompt_tokens": 124672,
                        "cached_tokens": 0,
                        "completion_tokens": 1,
                        "finish_reason": "length",
                    }
                    for rank in range(16)
                }
            )
        )
        (roots[0] / "common-prime-timing.json").write_text(
            json.dumps({"flush_count": 1, "real_prime_count_per_rank": 1})
        )
        return roots, config

    def test_warmup_after_formal_requests_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config = self.fixture(Path(temp))
            nodes = [collect_node(root, 16) for root in roots]
            for node in nodes:
                for row in node["warmups"]["b1-r0"].values():
                    row["cohort_first_step"] += 10000
                    row["cohort_last_step"] += 10000
                    row["cohort_first_ns"] += 100_000_000_000
                    row["cohort_last_ns"] += 100_000_000_000
            with self.assertRaisesRegex(RuntimeError, "chronology"):
                combine_nodes(nodes, config)

    def test_real_prime_cannot_be_replaced_with_a_cached_request(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, _ = self.fixture(Path(temp))
            path = roots[0] / "common-prime-requests.json"
            values = json.loads(path.read_text())
            values["common-prime-lane15-r0"]["cached_tokens"] = 124416
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(RuntimeError, "uncached prefix"):
                audit_runs(roots)

    def test_live_preview_does_not_accept_a_truncated_final_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, _ = self.fixture(Path(temp))
            path = roots[0] / "steps-0.jsonl"
            with path.open("a") as stream:
                stream.write('{"unfinished":')
            preview = collect_node(roots[0], 16, live=True)
            self.assertEqual(preview["windows"]["b1-r1"][0]["steps"], 50)
            with self.assertRaisesRegex(RuntimeError, "incomplete raw record"):
                audit_runs(roots)

    def test_all16_warm_and_two_independent_round_distributions(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config = self.fixture(Path(temp))
            result = audit_runs(roots)
            self.assertEqual(result["formal_intervals_per_rank_per_batch"], 100)
            self.assertEqual(
                result["batches"][1]["rank_interval_distribution_ms"]["count"], 1600
            )
            self.assertEqual(
                result["rounds"]["b1-r1"]["rank_interval_distribution_ms"]["count"], 800
            )
            self.assertEqual(result["input_audit"]["unique_suffixes"], 48)
            self.assertEqual(result["input_audit"]["cache_fraction"], 0.94921875)
            self.assertEqual(len(result["warmups"]["b1-r0"]), 16)

    def test_every_rank_must_supply_200_warm_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config = self.fixture(Path(temp))
            path = roots[1] / "steps-15.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = [
                row
                for row in rows
                if not (
                    row["request_ids"][0].startswith("b1-r0") and row["step"] >= 200
                )
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(RuntimeError):
                combine_nodes([collect_node(root, 16) for root in roots], config)

    def test_missing_rank_and_missing_round_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config = self.fixture(Path(temp))
            nodes = [collect_node(root, 16) for root in roots]
            for mutation in ("rank", "round"):
                changed = copy.deepcopy(nodes)
                if mutation == "rank":
                    changed[1]["ranks"].remove(15)
                else:
                    del changed[1]["windows"]["b1-r2"]
                with self.assertRaises(RuntimeError):
                    combine_nodes(changed, config)

    def test_actual_identity_and_partial_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config = self.fixture(Path(temp))
            path = roots[1] / "process-identity-15.json"
            row = json.loads(path.read_text())
            row["selected_environment"]["NCCL_IB_GID_INDEX"] = "8"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, "GID_INDEX"):
                audit_runs(roots)
            path = roots[0] / "steps-0.jsonl"
            path.write_text(path.read_text() + '{"rank":0')
            with self.assertRaises(RuntimeError):
                collect_node(roots[0], 16)

    def test_empty_fixed_identity_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_run_identity(
                {
                    name: {}
                    for name in (
                        "source",
                        "dependencies",
                        "model",
                        "indexer",
                        "network",
                    )
                }
            )

    def test_common_prefix_and_unique_cohort_suffixes(self):
        vocab = list(range(32))
        prompt = common_prompt(vocab, context=7000)
        tails = set()
        for cohort in (0, 1, 2):
            for rank in range(16):
                ids = request_tokens(prompt, vocab, 1, rank, 0, cohort=cohort)
                self.assertEqual(ids[:-6656], prompt[:-6656])
                self.assertNotIn(digest(ids[-6656:]), tails)
                tails.add(digest(ids[-6656:]))

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
            self.assertEqual(actual["TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER"], "1")
            self.assertEqual(actual["TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT"], "1")
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
