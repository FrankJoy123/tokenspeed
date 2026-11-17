#!/usr/bin/env bash
set -Eeuo pipefail
: "${VENV:?set VENV}" "${STATE_DIR:?set STATE_DIR}" "${CUDA_HOME:?set CUDA_HOME}"
: "${NVSHMEM_DIR:?set NVSHMEM_DIR}" "${NCCL_ROOT_DIR:?set NCCL_ROOT_DIR}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
revision=01dc3aaac82068020353dce2c302e38153c0bfaa
source_dir="$STATE_DIR/sources/deepep-wire1"
patch="$repo/.agents/deploy/fp4/patches/deepep-nvfp4-fp8-wire1.patch"
source "$VENV/bin/activate"
python -m pip install --no-deps nvidia-nvshmem-cu13==3.4.5
[[ -d "$NVSHMEM_DIR/include" && -d "$NCCL_ROOT_DIR/include" ]] || { echo 'NVSHMEM/NCCL development headers are required' >&2; exit 2; }
mkdir -p "$STATE_DIR/sources" "$STATE_DIR/setup"
if [[ ! -d "$source_dir/.git" ]]; then
 git clone --no-checkout https://github.com/deepseek-ai/DeepEP.git "$source_dir"
 git -C "$source_dir" checkout --detach "$revision"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'DeepEP revision mismatch' >&2; exit 2; }
patch_sha=$(sha256sum "$patch" | cut -d ' ' -f 1)
marker="$source_dir/.git/tokenspeed-wire1-patch.sha256"
if [[ -f "$marker" ]]; then
 [[ "$(cat "$marker")" == "$patch_sha" ]] || { echo 'Patch changed; use a fresh source directory' >&2; exit 2; }
 git -C "$source_dir" apply --reverse --check "$patch"
else
 [[ -z "$(git -C "$source_dir" status --porcelain)" ]] || { echo 'DeepEP source is dirty' >&2; exit 2; }
 git -C "$source_dir" apply --check "$patch"
 git -C "$source_dir" apply "$patch"
 printf '%s\n' "$patch_sha" > "$marker"
fi
export NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NCCL_ROOT_DIR/lib:$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$NCCL_ROOT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=12.0a EP_LL_COMBINE_MAX_SEND_UNROLLS=2 DISABLE_AGGRESSIVE_PTX_INSTRS=1
(cd "$source_dir" && python setup.py build_ext --inplace)
PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}" python - <<'PYAPI'
import deep_ep
if deep_ep.Buffer.wire_format_version < 1:
    raise RuntimeError("DeepEP wire format 1 was not built")
print(deep_ep.__file__)
PYAPI
printf 'revision=%s\npatch_sha256=%s\n' "$revision" "$patch_sha" > "$STATE_DIR/setup/deepep-wire1.txt"
printf 'Build complete. Multi-rank GPU validation is still required. Source: %s\n' "$source_dir"
