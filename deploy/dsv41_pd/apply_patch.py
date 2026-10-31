"""Apply a dependency patch once; reject a conflicting source instead of guessing."""
import argparse
import subprocess
from pathlib import Path


def apply_patch(root, patch):
    command = ["patch", "--batch", "-p0", "-d", str(root), "-i", str(patch.resolve())]
    def accepts(direction):
        return subprocess.run(command + ["--dry-run", direction], capture_output=True, text=True).returncode == 0
    if accepts("--forward"):
        subprocess.run(command + ["--forward"], check=True)
        return "applied"
    if accepts("--reverse"):
        return "already_applied"
    raise RuntimeError(f"Patch does not match dependency source: {patch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("patch", type=Path)
    args = parser.parse_args()
    print(apply_patch(args.root, args.patch))
