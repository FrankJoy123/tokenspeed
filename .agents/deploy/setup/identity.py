"""Record loaded code, native artifacts and resolved versions for a run."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_versions():
    names = {
        canonicalize_name(dist.metadata["Name"])
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    return {name: importlib.metadata.version(name) for name in sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    source_sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=15
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary", "HEAD"], timeout=30
    )
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], timeout=30
    )
    paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        timeout=30,
    ).split(b"\0")
    tree = hashlib.sha256()
    for raw in sorted(set(p for p in paths if p)):
        path = source / raw.decode()
        if not path.exists() and not path.is_symlink():
            continue
        tree.update(raw + b"\0")
        if path.is_symlink():
            tree.update(path.readlink().as_posix().encode())
        elif path.is_file():
            tree.update(digest(path).encode())
        tree.update(b"\0")
    modules = {}
    for name in (
        "tokenspeed",
        "tokenspeed_kernel",
        "tokenspeed_scheduler",
        "tokenspeed_engine",
        "deep_gemm",
        "deep_gemm_sm120",
        "deep_ep",
        "flashinfer",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        native = {
            str(p.relative_to(path.parent)): digest(p)
            for p in path.parent.rglob("*.so")
            if p.is_file()
        }
        modules[name] = {
            "path": str(path),
            "sha256": digest(path),
            "native_sha256": native,
        }
    from tokenspeed.runtime.engine.rust_host import host_binary

    host = Path(host_binary()).resolve(strict=True)
    result = {
        "host_binary": {
            "path": str(host),
            "sha256": digest(host),
            "override": os.environ.get("TOKENSPEED_HOST_BINARY"),
        },
        "python": sys.version,
        "source_commit": source_sha,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_tree_sha256": tree.hexdigest(),
        "source_is_clean": not bool(status),
        "modules": modules,
        "packages": package_versions(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
