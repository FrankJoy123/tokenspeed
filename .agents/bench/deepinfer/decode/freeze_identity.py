"""Freeze a public checkout, installed components, model headers and EP16 NIC map."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from audit import validate_run_identity

PACKAGES = (
    "torch",
    "tokenspeed",
    "tokenspeed-engine",
    "tokenspeed-scheduler",
    "tokenspeed-kernel",
    "flashinfer-python",
    "tokenspeed-deepgemm",
    "tokenspeed-deepgemm-sm120",
    "tokenspeed-deepep",
    "tokenspeed-mooncake",
    "tokenspeed-triton",
    "nvidia-cutlass-dsl",
    "nvidia-nvshmem-cu13",
    "nvidia-nccl-cu13",
    "transformers",
    "xgrammar",
    "msgspec",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--node0-nics", required=True)
    parser.add_argument("--node1-nics", required=True)
    parser.add_argument("--gid-index", required=True, type=int)
    parser.add_argument("--socket-ifname", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[4]
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        parser.error("Choose a new identity output; do not overwrite a frozen run")
    if not args.model.is_absolute():
        parser.error("Use the same absolute model path on both nodes")
    if args.gid_index < 0:
        parser.error("GID index must be nonnegative")
    nics = [value.split(",") for value in (args.node0_nics, args.node1_nics)]
    if any(len(row) != 8 or len(set(row)) != 8 or not all(row) for row in nics):
        parser.error("Each node needs eight distinct NIC names in visible GPU order")
    artifact = out.parent / (out.stem + "-artifacts")
    artifact.mkdir(exist_ok=False)
    runtime = artifact / "runtime-identity.json"
    model = artifact / "model-identity.json"
    subprocess.run(
        [
            sys.executable,
            str(source / ".agents/deploy/setup/identity.py"),
            "--source",
            str(source),
            "--output",
            str(runtime),
        ],
        check=True,
        timeout=240,
    )
    loaded = json.loads(runtime.read_text())
    if not loaded["source_is_clean"] or loaded["host_binary"]["override"]:
        parser.error("Use a clean committed checkout and its actual built host")
    subprocess.run(
        [
            sys.executable,
            str(source / ".agents/deploy/launch/preflight_model.py"),
            "--model",
            str(args.model),
            "--profile",
            "w4a8",
            "--output",
            str(model),
        ],
        check=True,
        timeout=180,
    )
    archive = artifact / "source.tar.gz"
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "archive",
            "--format=tar.gz",
            "--output=" + str(archive),
            "HEAD",
        ],
        check=True,
        timeout=90,
    )
    nodes = [
        {
            "ranks": list(range(index * 8, (index + 1) * 8)),
            "nics": row,
            "gid_index": args.gid_index,
            "socket_ifname": args.socket_ifname,
            "nccl_ib_hca": ",".join(row),
        }
        for index, row in enumerate(nics)
    ]
    value = {
        "source": {
            "commit": loaded["source_commit"],
            "source_tree_sha256": loaded["source_tree_sha256"],
            "archive_sha256": digest(archive),
        },
        "dependencies": {
            "packages": {name: loaded["packages"][name] for name in PACKAGES},
            "modules": loaded["modules"],
            "host_binary_sha256": loaded["host_binary"]["sha256"],
            "worker_source_sha256": {
                "runtime_model": digest(
                    source / "python/tokenspeed/runtime/models/deepseek_v41.py"
                ),
                "indexer": digest(
                    source
                    / "tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py"
                ),
            },
        },
        "model": {
            "checkpoint": str(args.model),
            "headers_manifest_sha256": digest(model),
        },
        "indexer": {
            "precision": "bf16",
            "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER": "0",
            "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER": "0",
        },
        "network": {
            "wire": "elastic_fp8_bf16",
            "capacity": 128,
            "shared_overlap": 1,
            "nodes": nodes,
        },
    }
    validate_run_identity(value)
    out.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {"identity_sha256": digest(out), "source_commit": loaded["source_commit"]}
        )
    )


if __name__ == "__main__":
    main()
