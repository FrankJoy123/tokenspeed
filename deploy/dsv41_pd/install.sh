#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require ROLE
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT
export PIP_CONSTRAINT="$DEPLOY_DIR/requirements.txt"
export TOKENSPEED_KERNEL_BACKEND=cuda
# Fresh venv only: never modify the running service's environment implicitly.
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --resume ) ]] || { echo 'Usage: install.sh [--resume]' >&2; exit 2; }
if [[ -e "$VENV" ]]; then
 [[ "${1:-}" == --resume && -x "$VENV/bin/python" ]] || { echo 'VENV exists; use --resume only for this dedicated deployment venv' >&2; exit 2; }
fi
command -v uv >/dev/null
command -v cargo >/dev/null
require CUDA_TOOLKIT_MODE
case "$CUDA_TOOLKIT_MODE" in
 system) [[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'CUDA toolkit with nvcc is required' >&2; exit 2; } ;;
 python) [[ "$CUDA_HOME" == "$VENV/lib/python3.12/site-packages/nvidia/cu13" ]] || { echo 'Python CUDA mode requires CUDA_HOME inside the new venv' >&2; exit 2; } ;;
 *) echo 'CUDA_TOOLKIT_MODE must be system or python' >&2; exit 2 ;;
esac
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$STATE_DIR/uv-python}"
[[ -x "$VENV/bin/python" ]] || uv venv "$VENV" --python 3.12 --seed
if [[ "$CUDA_TOOLKIT_MODE" == python ]]; then
 "$VENV/bin/python" -m pip install 'cuda-toolkit[nvcc,nvrtc,cccl,cudart]==13.0.3.0'
 bash "$DEPLOY_DIR/prepare_cuda_tools.sh"
fi
"$VENV/bin/python" -m pip install -r "$DEPLOY_DIR/requirements.txt"
export CARGO_TARGET_DIR="$STATE_DIR/cargo-target" CARGO_BUILD_JOBS="$MAX_JOBS"
# Build host and scheduler from THIS checkout, including the PD/control fixes.
"$VENV/bin/python" -m pip install "$REPO_ROOT/tokenspeed-scheduler" "$REPO_ROOT/tokenspeed-engine"
"$VENV/bin/python" -m pip install --no-build-isolation -e "$REPO_ROOT/tokenspeed-kernel/python"
"$VENV/bin/python" -m pip install -c "$DEPLOY_DIR/requirements.txt" -e "$REPO_ROOT/python"
# Restore explicit runtime pins after resolving TS dependencies.
"$VENV/bin/python" -m pip install -r "$DEPLOY_DIR/requirements.txt"
"$VENV/bin/python" -m pip check
printf 'Base environment ready. Next: install_dependencies.sh, check.sh, then launch scripts.\n'
