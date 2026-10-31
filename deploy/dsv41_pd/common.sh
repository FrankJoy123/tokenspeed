#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
cd "$REPO_ROOT"
set -a
source "${PD_CONFIG:-$DEPLOY_DIR/env.local.sh}"
set +a
require() { local key; for key in "$@"; do [[ -n "${!key:-}" && "${!key}" != CHANGE_ME ]] || { echo "Configure $key in PD_CONFIG" >&2; exit 2; }; done; }
require ROLE VENV CUDA_HOME STATE_DIR
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/tokenspeed-kernel/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
unset LD_PRELOAD CUDA_LAUNCH_BLOCKING FLASHINFER_DISABLE_JIT TOKENSPEED_HOST_BINARY
export NO_PROXY="localhost,127.0.0.1,${P_HOST:-},${D_HOST:-},${GATEWAY_HOST:-}" no_proxy="localhost,127.0.0.1,${P_HOST:-},${D_HOST:-},${GATEWAY_HOST:-}"
export TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300
export MAX_JOBS="${MAX_JOBS:-8}" FLASHINFER_NVCC_THREADS=2
export TOKENSPEED_CUDA_ARCH=120a FLASHINFER_CUDA_ARCH_LIST=12.0a
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export TOKENSPEED_KERNEL_CACHE_ROOT="$STATE_DIR/cache/kernel"
export TRITON_CACHE_DIR="${PD_TRITON_CACHE_DIR:-$STATE_DIR/cache/triton}" CUDA_CACHE_PATH="$STATE_DIR/cache/cuda"
export FLASHINFER_CACHE_DIR="$STATE_DIR/cache/flashinfer" FLASHINFER_WORKSPACE_BASE="$STATE_DIR/cache/flashinfer-workspace"
export CUTE_DSL_CACHE_DIR="$STATE_DIR/cache/cute" TORCH_EXTENSIONS_DIR="$STATE_DIR/cache/torch"
export XDG_CACHE_HOME="$STATE_DIR/cache/xdg" TMPDIR="$STATE_DIR/tmp"
export DG_JIT_CACHE_DIR="$STATE_DIR/cache/deepgemm" EP_JIT_CACHE_DIR="$STATE_DIR/cache/deepep"
export TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR="$STATE_DIR/cache/autotune"
export DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc" EP_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
mkdir -p "$STATE_DIR/logs" "$TMPDIR" "$STATE_DIR/cache"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/torch/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib${EXTRA_LIBRARY_PATH:+:$EXTRA_LIBRARY_PATH}"
export BENCH_URL="http://${GATEWAY_HOST:-localhost}:18500"
run() {
 if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else exec "$@"; fi
}
