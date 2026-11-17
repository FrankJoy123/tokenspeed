"""Dependency identity follows the packages resolved by the active interpreter."""

import importlib.util
import sys
import tempfile
from pathlib import Path


def test_venv_version_wins_over_inherited_system_metadata():
    spec = importlib.util.spec_from_file_location(
        "deploy_identity", Path(__file__).with_name("identity.py")
    )
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for location, version in (("venv", "2.0"), ("system", "1.0")):
            metadata = (
                root / location / f"tokenspeed_test_dependency-{version}.dist-info"
            )
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: TokenSpeed_Test_Dependency\nVersion: {version}\n"
            )
        original = sys.path.copy()
        try:
            sys.path[:0] = [str(root / "venv"), str(root / "system")]
            assert identity.package_versions()["tokenspeed-test-dependency"] == "2.0"
        finally:
            sys.path[:] = original
