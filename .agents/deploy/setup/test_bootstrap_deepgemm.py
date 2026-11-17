"""A failed initial fetch can be retried without discarding a pinned checkout."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


def test_retry_initial_fetch_and_reuse_exact_checkout(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "bootstrap", Path(__file__).with_name("bootstrap_deepgemm.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=feature/seed", str(seed)],
        check=True,
        timeout=10,
    )
    (seed / "payload.txt").write_text("fixed public dependency source\n")
    subprocess.run(
        ["git", "-C", str(seed), "add", "payload.txt"], check=True, timeout=10
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(seed),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "seed",
        ],
        check=True,
        timeout=10,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(seed), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    remote = tmp_path / "remote.git"
    target = tmp_path / "checkout"
    with pytest.raises(subprocess.CalledProcessError):
        module.checkout(target, remote.as_uri(), revision)
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(seed), str(remote)],
        check=True,
        timeout=10,
    )
    assert module.checkout(target, remote.as_uri(), revision) == revision
    assert (target / "payload.txt").read_bytes() == (seed / "payload.txt").read_bytes()
    shutil.rmtree(remote)
    assert module.checkout(target, remote.as_uri(), revision) == revision
