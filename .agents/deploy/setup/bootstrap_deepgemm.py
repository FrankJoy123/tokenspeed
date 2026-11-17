#!/usr/bin/env python3
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

"""Build the fixed public SM120 DeepGEMM dependency in its own namespace."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def run(*args, cwd=None, env=None, limit=600, capture=False):
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        timeout=limit,
        text=True,
        capture_output=capture,
    )


def checkout(path, url, revision):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Dependency revisions must be full commit hashes")
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise RuntimeError(f"Use an empty dependency directory: {path}")
        run(
            "git",
            "init",
            "--quiet",
            "--initial-branch=feature/sm120-dependency",
            str(path),
        )
        run("git", "-C", str(path), "remote", "add", "origin", url)
    origin = run(
        "git", "-C", str(path), "remote", "get-url", "origin", capture=True
    ).stdout.strip()
    if origin != url:
        raise RuntimeError(f"Dependency origin mismatch at {path}")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if head.returncode:
        if any(item.name != ".git" for item in path.iterdir()):
            raise RuntimeError(f"Incomplete checkout has local files: {path}")
        run(
            "git",
            "-C",
            str(path),
            "fetch",
            "--no-tags",
            "--depth=1",
            "origin",
            revision,
            limit=1800,
        )
        run("git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD")
        actual = run(
            "git", "-C", str(path), "rev-parse", "HEAD", capture=True
        ).stdout.strip()
    else:
        actual = head.stdout.strip()
    if actual != revision:
        raise RuntimeError(f"Dependency revision mismatch at {path}: {actual}")
    return actual


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the dedicated build venv first")
    os.environ["CUDA_HOME"]  # CUDA source builds require an explicit toolkit.
    profile = Path(__file__).resolve().parent.parent / "w4a8" / "patches"
    manifest = json.loads((profile / "sources.json").read_text())
    patch = profile / manifest["patch"]
    if sha(patch) != manifest["patch_sha256"]:
        raise RuntimeError("SM120 source patch checksum mismatch")
    source = args.work_dir.resolve()
    checkout(
        source,
        **dict(url=manifest["source"]["url"], revision=manifest["source"]["commit"]),
    )
    marker = source / ".git" / "tokenspeed-sm120-patch.sha256"
    if marker.exists():
        if marker.read_text().strip() != manifest["patch_sha256"]:
            raise RuntimeError("Dependency patch changed; use a fresh work directory")
        run("git", "-C", str(source), "apply", "--reverse", "--check", str(patch))
    else:
        if run("git", "-C", str(source), "status", "--porcelain", capture=True).stdout:
            raise RuntimeError("Refusing to patch a dirty dependency checkout")
        run("git", "-C", str(source), "apply", "--check", str(patch))
        run("git", "-C", str(source), "apply", str(patch))
        marker.write_text(manifest["patch_sha256"] + "\n")
    headers = {}
    for item in manifest["headers"]:
        path = source / item["path"]
        headers[item["path"]] = checkout(path, item["url"], item["commit"])
    wheels = args.wheel_dir.resolve()
    wheels.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DG_FORCE_BUILD="1", DG_SKIP_CUDA_BUILD="0")
    # Runtime cache locations belong to the consumer, not to a portable wheel.
    env.pop("DG_SM120_JIT_CACHE_DIR", None)
    run(
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        str(source),
        "--wheel-dir",
        str(wheels),
        env=env,
        limit=3600,
    )
    candidates = list(
        wheels.glob(f"tokenspeed_deepgemm_sm120-{manifest['version']}-*.whl")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one SM120 dependency wheel, got {candidates}")
    wheel = candidates[0]
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        str(wheel),
    )
    # A subprocess observes the installed extension rather than a build-tree import.
    verify = """import deep_gemm, deep_gemm_sm120, importlib.metadata as m, json
from pathlib import Path
assert Path(deep_gemm.__file__).parent != Path(deep_gemm_sm120.__file__).parent
assert m.version('tokenspeed-deepgemm-sm120') == EXPECTED
info = json.loads((Path(deep_gemm_sm120.__file__).parent/'SM120_SOURCE.json').read_text())
assert info['upstream_commit'] == COMMIT
print(json.dumps({'ordinary':m.version('tokenspeed-deepgemm'),'sm120':m.version('tokenspeed-deepgemm-sm120'),'extension':deep_gemm_sm120._C.__file__}))
""".replace("EXPECTED", repr(manifest["version"])).replace(
        "COMMIT", repr(manifest["source"]["commit"])
    )
    installed = json.loads(run(sys.executable, "-c", verify, capture=True).stdout)
    installed.update(
        source=manifest["source"],
        headers=headers,
        patch_sha256=manifest["patch_sha256"],
        wheel=wheel.name,
        wheel_sha256=sha(wheel),
        extension_sha256=sha(Path(installed["extension"])),
    )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(installed, indent=2) + "\n")
    print(f"Built {manifest['distribution']}=={manifest['version']}")


if __name__ == "__main__":
    main()
