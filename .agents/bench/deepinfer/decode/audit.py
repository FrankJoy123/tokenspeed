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

"""Strict raw-completion and fixed-identity acceptance for the Decode protocol."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from analysis import (
    BATCHES,
    CACHED_PREFIX,
    CONTEXT,
    FORMAL_INTERVALS,
    FORMAL_SKIP,
    SAMPLING,
    TAU,
    WARM_STEPS,
    WIDTH,
    WORLD_SIZE,
    distribution,
    output_budget,
)
from analysis import warmup as warmup_window
from analysis import (
    window,
)


def strict_window(
    samples,
    prefix,
    batch,
    warmup=FORMAL_SKIP,
    steps=FORMAL_INTERVALS,
    world_size=WORLD_SIZE,
):
    if warmup != FORMAL_SKIP or steps != FORMAL_INTERVALS:
        raise RuntimeError(
            "this protocol requires 100 discarded points and 50 intervals"
        )
    return window(samples, prefix, batch, warmup, steps, world_size=world_size)


def validate_fixed_width(samples, result, batch):
    selected = [s for s in samples if s["step"] in result["endpoint_steps"]]
    if len(selected) != FORMAL_INTERVALS + 1:
        raise RuntimeError("expected 51 raw endpoints for 50 intervals")
    if any(
        s["input_rows"] != WIDTH * batch or s["output_lengths"] != [WIDTH] * batch
        for s in selected
    ):
        raise RuntimeError("every request must verify and keep exactly 6 tokens")


def read_samples(root, *, live=False):
    by_rank, writers, seen = {}, {}, set()
    for path in sorted(Path(root).glob("steps-*.jsonl")):
        text = path.read_text()
        if text and not text.endswith("\n"):
            if not live:
                raise RuntimeError(f"incomplete raw record in {path.name}")
            text = text.rsplit("\n", 1)[0] + "\n" if "\n" in text else ""
        for line in text.splitlines():
            s = json.loads(line)
            rank, step = s["rank"], s["step"]
            if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
                raise RuntimeError("invalid rank")
            if type(step) is not int or step < 0 or (rank, step) in seen:
                raise RuntimeError("invalid or duplicate rank/step")
            seen.add((rank, step))
            if rank in writers and writers[rank] != path.name:
                raise RuntimeError("multiple worker processes for one rank")
            writers[rank] = path.name
            by_rank.setdefault(rank, []).append(s)
    return by_rank, writers


def collect_node(root, world_size, *, live=False):
    root = Path(root)
    by_rank, writers = read_samples(root, live=live)
    tags = set()
    for samples in by_rank.values():
        for sample in samples:
            for rid in sample["request_ids"]:
                tag = rid.split("-lane", 1)[0]
                if re.fullmatch(r"b(?:1|2|4|8|16|32)-r[012]", tag):
                    tags.add(tag)
    windows, warmups = {}, {}
    for tag in sorted(tags):
        batch = int(tag.split("-")[0][1:])
        rows = {}
        for rank, samples in by_rank.items():
            try:
                if tag.endswith("-r0"):
                    result = warmup_window(
                        samples, tag + "-lane", batch, world_size=world_size
                    )
                else:
                    result = strict_window(
                        samples, tag + "-lane", batch, world_size=world_size
                    )
                    validate_fixed_width(samples, result, batch)
                cohort = [
                    sample
                    for sample in samples
                    if any(
                        rid.startswith(tag + "-lane") for rid in sample["request_ids"]
                    )
                ]
                result.update(
                    cohort_first_step=min(sample["step"] for sample in cohort),
                    cohort_last_step=max(sample["step"] for sample in cohort),
                    cohort_first_ns=min(sample["completed_ns"] for sample in cohort),
                    cohort_last_ns=max(sample["completed_ns"] for sample in cohort),
                )
            except (RuntimeError, KeyError, ValueError) as error:
                result = {"error": str(error)}
            rows[rank] = result
        (warmups if tag.endswith("-r0") else windows)[tag] = rows
    return {
        "windows": windows,
        "warmups": warmups,
        "ranks": sorted(by_rank),
        "writers": writers,
        "source_sha256": json.loads((root / "source-sha256.json").read_text()),
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run_identity(value):
    for name in ("source", "dependencies", "model", "indexer", "network"):
        require(
            isinstance(value.get(name), dict) and value[name],
            f"missing fixed {name} identity",
        )
    source = value["source"]
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", source.get("commit", ""))),
        "invalid source commit",
    )
    for key in ("source_tree_sha256", "archive_sha256"):
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", source.get(key, ""))),
            f"invalid source {key}",
        )
    deps = value["dependencies"]
    require(
        bool(deps.get("packages"))
        and bool(deps.get("modules"))
        and bool(deps.get("worker_source_sha256")),
        "missing resolved package/native/worker-source identity",
    )
    modules = {
        "tokenspeed",
        "tokenspeed_kernel",
        "tokenspeed_engine",
        "tokenspeed_scheduler",
        "deep_gemm",
        "deep_gemm_sm120",
        "deep_ep",
        "flashinfer",
    }
    require(modules.issubset(deps["modules"]), "incomplete loaded module identity")
    require(
        set(deps["worker_source_sha256"]) == {"runtime_model", "indexer"},
        "incomplete worker source identity",
    )
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", deps.get("host_binary_sha256", ""))),
        "missing actual host binary SHA",
    )
    require(bool(value["model"].get("checkpoint")), "missing checkpoint identity")
    require(
        bool(
            re.fullmatch(
                r"[0-9a-f]{64}", value["model"].get("headers_manifest_sha256", "")
            )
        ),
        "missing model metadata/header manifest SHA",
    )
    require(value["indexer"].get("precision") == "bf16", "requires BF16 Indexer")
    for key in (
        "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER",
        "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER",
    ):
        require(value["indexer"].get(key) == "0", f"invalid {key}")
    network = value["network"]
    require(
        network.get("wire") == "elastic_fp8_bf16"
        and network.get("capacity") == 128
        and network.get("shared_overlap") == 1,
        "wrong fixed communication settings",
    )
    ranks = []
    for node in network.get("nodes", []):
        require(
            len(node.get("ranks", [])) == len(node.get("nics", [])) == 8,
            "need explicit eight-GPU NIC mapping per node",
        )
        require(
            all(node["nics"])
            and type(node.get("gid_index")) is int
            and bool(node.get("socket_ifname"))
            and bool(node.get("nccl_ib_hca")),
            "incomplete network topology identity",
        )
        ranks += node["ranks"]
    require(
        sorted(ranks) == list(range(WORLD_SIZE)),
        "network identity must cover all 16 ranks once",
    )
    return value


def audit_identity(root, expected, node):
    root = Path(root)
    actual = json.loads((root / "runtime-identity.json").read_text())
    require(
        actual["source_is_clean"]
        and actual["source_commit"] == expected["source"]["commit"]
        and actual["source_tree_sha256"] == expected["source"]["source_tree_sha256"],
        "runtime source identity changed",
    )
    require(
        actual["host_binary"]["override"] is None
        and actual["host_binary"]["sha256"]
        == expected["dependencies"]["host_binary_sha256"],
        "actual host binary changed",
    )
    for name, version in expected["dependencies"]["packages"].items():
        require(
            actual["packages"].get(name) == version,
            f"dependency version changed: {name}",
        )
    for name, fields in expected["dependencies"]["modules"].items():
        for key in ("sha256", "native_sha256"):
            require(
                actual["modules"].get(name, {}).get(key) == fields[key],
                f"loaded module/native changed: {name}",
            )
    require(
        sha(root / "model-identity.json")
        == expected["model"]["headers_manifest_sha256"],
        "model metadata/header identity changed",
    )
    model = json.loads((root / "model-identity.json").read_text())
    require(
        model.get("profile") == "w4a8"
        and bool(model.get("metadata_sha256"))
        and bool(model.get("shards"))
        and model.get("indexed_tensors", 0) > 0,
        "invalid checkpoint metadata/header manifest",
    )
    samples, writers = read_samples(root)
    for rank, name in writers.items():
        pid = name.removeprefix("steps-").removesuffix(".jsonl")
        process = json.loads((root / f"process-identity-{pid}.json").read_text())
        env = process["selected_environment"]
        topology = next(n for n in expected["network"]["nodes"] if rank in n["ranks"])
        required = {
            "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN": "6",
            "TOKENSPEED_MOE_ROUTING_SIMULATION": None,
            "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING": None,
            "TOKENSPEED_SM120_DEEPEP_WIRE": expected["network"]["wire"],
            "TOKENSPEED_SM120_DEEPEP_CAPACITY": str(expected["network"]["capacity"]),
            "TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP": "1",
            "TOKENSPEED_SM120_DEEPEP_NICS": ",".join(topology["nics"]),
            "NCCL_IB_GID_INDEX": str(topology["gid_index"]),
            "NVSHMEM_IB_GID_INDEX": str(topology["gid_index"]),
            "NCCL_SOCKET_IFNAME": topology["socket_ifname"],
            "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": topology["socket_ifname"],
            "NCCL_IB_HCA": topology["nccl_ib_hca"],
        }
        required.update(
            {
                key: value
                for key, value in expected["indexer"].items()
                if key.startswith("TOKENSPEED_")
            }
        )
        for key, value in required.items():
            require(env.get(key) == value, f"rank {rank} actual {key} changed")
        nic = topology["nics"][topology["ranks"].index(rank)]
        for sample in samples[rank]:
            if any(rid.startswith("b") for rid in sample["request_ids"]):
                require(
                    sample["ep_nic_name"] == nic
                    and sample["nvshmem_hca_list"] == nic + ":1",
                    f"rank {rank} actual NIC changed",
                )
        require(
            process.get("source_sha256")
            == expected["dependencies"]["worker_source_sha256"],
            "actual worker source changed",
        )
    return {"ranks": node["ranks"], "source_commit": actual["source_commit"]}


def audit_inputs(root, config):
    from inputs import common_prompt, digest, request_tokens

    root = Path(root)
    identity = json.loads((root / "input-identity.json").read_text())
    excluded = set(identity["excluded_token_ids"])
    vocab = [i for i in range(identity["vocab_size"]) if i not in excluded]
    prompt = common_prompt(vocab, context=CONTEXT, seed=identity["seed"])
    require(
        digest(prompt[:-6656]) == identity["prefix_sha256"], "common prefix changed"
    )
    seen = set()
    for batch in config["batches"]:
        for cohort in (0, 1, 2):
            tag = f"b{batch}-r{cohort}"
            requests = json.loads((root / f"{tag}-requests.json").read_text())
            hashes = json.loads((root / f"{tag}-input-hashes.json").read_text())
            require(len(requests) == batch * WORLD_SIZE, "incomplete request cohort")
            for rank in range(WORLD_SIZE):
                for index in range(batch):
                    rid = f"{tag}-lane{rank}-r{index}"
                    row = requests[rid]
                    require(
                        row["prompt_tokens"] == CONTEXT
                        and row["cached_tokens"] == CACHED_PREFIX,
                        "actual cache/prompt length changed",
                    )
                    require(
                        row["completion_tokens"]
                        == output_budget(batch, cohort)["max_new_tokens"]
                        and row["finish_reason"] != "abort",
                        "request completion/budget failed",
                    )
                    values = request_tokens(
                        prompt, vocab, batch, rank, index, cohort=cohort
                    )
                    suffix, full = digest(values[-6656:]), digest(values)
                    require(
                        suffix not in seen, "suffix reused across requests or cohorts"
                    )
                    seen.add(suffix)
                    require(
                        hashes["suffix_sha256_by_request"][rid] == suffix
                        and hashes["input_sha256_by_request"][rid] == full,
                        "actual request hash changed",
                    )
    prime_requests = json.loads((root / "common-prime-requests.json").read_text())
    require(
        set(prime_requests)
        == {f"common-prime-lane{rank}-r0" for rank in range(WORLD_SIZE)},
        "incomplete real prime cohort",
    )
    require(
        identity["prime_prompt_tokens"] == 124672
        and identity["prefix_tokens"] == CACHED_PREFIX
        and identity["prime_prompt_sha256"]
        == digest(prompt[:-6656] + prompt[-6656:-6400]),
        "prime/prefix token identity changed",
    )
    for row in prime_requests.values():
        require(
            row["prompt_tokens"] == 124672
            and row["cached_tokens"] == 0
            and row["completion_tokens"] == 1
            and row["finish_reason"] != "abort",
            "prime did not really compute the full uncached prefix",
        )
    prime = json.loads((root / "common-prime-timing.json").read_text())
    require(
        prime["flush_count"] == prime["real_prime_count_per_rank"] == 1,
        "expected one real prime/cache flush",
    )
    return {
        "complete_requests": len(seen),
        "unique_suffixes": len(seen),
        "every_cached_tokens": CACHED_PREFIX,
        "cache_fraction": CACHED_PREFIX / CONTEXT,
    }


def combine_nodes(nodes, config):
    expected_ranks = set(range(WORLD_SIZE))
    ranks = [rank for node in nodes for rank in node["ranks"]]
    require(sorted(ranks) == list(range(WORLD_SIZE)), "need all 16 ranks exactly once")
    source = nodes[0]["source_sha256"]
    require(
        bool(source) and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in source.values()),
        "invalid frozen observer source identity",
    )
    require(
        all(node["source_sha256"] == source for node in nodes),
        "observer source differs across nodes",
    )
    windows, warmups = {}, {}
    for key, target in (("windows", windows), ("warmups", warmups)):
        for node in nodes:
            for tag, rows in node[key].items():
                bucket = target.setdefault(tag, {})
                require(
                    not (bucket.keys() & rows.keys()), "duplicate rank across nodes"
                )
                bucket.update(rows)
    require(
        set(warmups) == {f"b{b}-r0" for b in config["batches"]}, "missing warm cohort"
    )
    require(
        set(windows)
        == {f"b{b}-r{round_}" for b in config["batches"] for round_ in (1, 2)},
        "missing independent formal cohort",
    )
    for rows in [*warmups.values(), *windows.values()]:
        require(
            set(rows) == expected_ranks
            and all("error" not in row for row in rows.values()),
            f"incomplete/invalid raw completion evidence: {rows}",
        )
    # Prove execution order independently from controller summaries. A later
    # warm cohort or overlapping formal cohorts cannot qualify an earlier point.
    for rank in expected_ranks:
        previous = None
        for batch in config["batches"]:
            for cohort in (0, 1, 2):
                row = (warmups if cohort == 0 else windows)[f"b{batch}-r{cohort}"][rank]
                if previous is not None:
                    require(
                        previous["cohort_last_step"] < row["cohort_first_step"]
                        and previous["cohort_last_ns"] < row["cohort_first_ns"],
                        f"rank {rank} warm/formal/batch chronology violated",
                    )
                previous = row
    rounds = {}
    for tag, rows in windows.items():
        all_intervals = [v / 1e6 for row in rows.values() for v in row["interval_ns"]]
        slowest = max(row["step_ms"] for row in rows.values())
        batch = int(tag.split("-")[0][1:])
        rank0 = rows[0]["step_ms"]
        rounds[tag] = {
            "rank0_step_ms": rank0,
            "rank0_interval_ms": [v / 1e6 for v in rows[0]["interval_ns"]],
            "raw_rank_interval_ms": all_intervals,
            "rank_interval_distribution_ms": distribution(all_intervals),
            "slowest_rank_step_ms": slowest,
            "deepinfer_tokens_s_user": TAU * 1000 / rank0,
            "deepinfer_tokens_s_gpu": batch * TAU * 1000 / rank0,
        }
    batches = {}
    for batch in config["batches"]:
        tags = [f"b{batch}-r1", f"b{batch}-r2"]
        raw = [value for tag in tags for value in rounds[tag]["raw_rank_interval_ms"]]
        slowest = max(
            sum(windows[tag][rank]["seconds"] for tag in tags)
            / (2 * FORMAL_INTERVALS)
            * 1000
            for rank in expected_ranks
        )
        rank0 = (
            sum(windows[tag][0]["seconds"] for tag in tags)
            / (2 * FORMAL_INTERVALS)
            * 1000
        )
        batches[batch] = {
            "round_tags": tags,
            "formal_intervals_per_rank": 100,
            "rank0_mean_step_ms": rank0,
            "raw_rank_interval_ms": raw,
            "rank_interval_distribution_ms": distribution(raw),
            "slowest_rank_mean_step_ms": slowest,
            "deepinfer_tokens_s_user": TAU * 1000 / rank0,
            "deepinfer_tokens_s_gpu": batch * TAU * 1000 / rank0,
        }
    return {
        "windows": windows,
        "warmups": warmups,
        "rounds": rounds,
        "batches": batches,
        "formal_intervals_per_rank_per_batch": 2 * FORMAL_INTERVALS,
        "assumed_tau": TAU,
        "actual_kept_width": WIDTH,
    }


def audit_runs(roots, world_size=WORLD_SIZE):
    require(world_size == WORLD_SIZE, "requires 16 ranks")
    roots = [Path(root) for root in roots]
    config = json.loads((roots[0] / "config.json").read_text())
    require(
        config.get("protocol") == "decode" and config.get("protocol_version") == 1,
        "unsupported protocol record",
    )
    require(
        config["sampling"] == SAMPLING
        and config["warm_steps"] == WARM_STEPS
        and config["formal_skip"] == FORMAL_SKIP
        and config["formal_intervals"] == FORMAL_INTERVALS
        and config["world_size"] == WORLD_SIZE
        and config["rounds"] == 2,
        "wrong protocol configuration",
    )
    require(
        config["batches"]
        and len(set(config["batches"])) == len(config["batches"])
        and all(b in BATCHES for b in config["batches"]),
        "invalid batch sweep",
    )
    fixed = validate_run_identity(
        json.loads((roots[0] / "run-identity.json").read_text())
    )
    identity_sha = sha(roots[0] / "run-identity.json")
    require(
        config["run_identity_sha256"] == identity_sha, "controller identity changed"
    )
    nodes = [collect_node(root, world_size) for root in roots]
    for root, node in zip(roots, nodes):
        require(
            sha(root / "run-identity.json") == identity_sha,
            "node fixed identities differ",
        )
        audit_identity(root, fixed, node)
    result = combine_nodes(nodes, config)
    result["input_audit"] = audit_inputs(roots[0], config)
    result["run_identity_sha256"] = identity_sha
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    args = parser.parse_args()
    result = audit_runs(args.run_dir, args.world_size)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_ranks": WORLD_SIZE,
                "formal_intervals_per_batch": 100,
                "warm_steps_per_batch": WARM_STEPS,
                "assumed_tau": TAU,
            }
        )
    )


if __name__ == "__main__":
    main()
