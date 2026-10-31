"""Verify portable source/data artifacts and unpack into the deployment directory."""
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "artifacts.json").read_text())
for name, checksum in manifest.items():
    file = root / "artifacts" / name
    assert file.is_file(), f"Missing migration artifact: {file}"
    with file.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert digest == checksum, f"Artifact hash mismatch: {file}"
source = Path(os.environ['SOURCE_DIR'])
for name in ['p-deepep', 'p-flashinfer', 'p-nvshmem']:
    destination = source / name
    if destination.exists():
        raise FileExistsError(f"Preserve existing source; choose a fresh SOURCE_DIR: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(root / 'artifacts' / f'{name}.tar.gz') as archive:
        archive.extractall(destination, filter='data')
data = Path(os.environ['SESSIONS_FILE'])
if data.exists():
    raise FileExistsError(f"Preserve existing data: {data}")
data.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(root / 'artifacts/sessions.json.gz', 'rb') as stream:
    content = stream.read()
assert hashlib.sha256(content).hexdigest() == os.environ['SESSIONS_SHA256']
data.write_bytes(content)
print('Source snapshots and benchmark token data verified and unpacked.')
