#!/usr/bin/env bash
set -Eeuo pipefail
: "${VENV:?set VENV}" "${STATE_DIR:?set STATE_DIR}" "${CUDA_HOME:?set CUDA_HOME}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
test -x "$CUDA_HOME/bin/nvcc"
command -v cargo >/dev/null
command -v rustc >/dev/null
if [[ ! -x "$VENV/bin/python" ]]; then python3.12 -m venv "$VENV"; fi
source "$VENV/bin/activate"
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENSPEED_KERNEL_BACKEND=cuda TOKENSPEED_CUDA_ARCH=120a FLASHINFER_CUDA_ARCH_LIST=12.0a
export TOKENSPEED_KERNEL_SKIP_BUILD_DEPS=1 MAX_JOBS=8
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export CARGO_TARGET_DIR="$STATE_DIR/build/cargo"
export TOKENSPEED_KERNEL_GIT_SHA=$(git -C "$repo" rev-parse HEAD)
export TOKENSPEED_KERNEL_GIT_BRANCH=$(git -C "$repo" branch --show-current)
export TMPDIR="$STATE_DIR/tmp/build" XDG_CACHE_HOME="$STATE_DIR/cache/xdg"
export LD_LIBRARY_PATH="${TOOLCHAIN_LIBRARY_DIR:+$TOOLCHAIN_LIBRARY_DIR:}$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$STATE_DIR/setup" "$STATE_DIR/wheels" "$TMPDIR" "$XDG_CACHE_HOME"
python -m pip install -r "$repo/.agents/deploy/setup/build-requirements.txt"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
 -r "$repo/tokenspeed-kernel/python/requirements/cuda.txt" \
 -r "$repo/tokenspeed-kernel/python/requirements/cuda-thirdparty.txt"
python "$repo/.agents/deploy/setup/bootstrap_deepgemm.py" \
 --work-dir "$STATE_DIR/sources/deepgemm-sm120" \
 --wheel-dir "$STATE_DIR/wheels/dependencies" \
 --record "$STATE_DIR/setup/deepgemm-sm120.json"
for component in tokenspeed-scheduler tokenspeed-engine tokenspeed-kernel/python python; do
 python -m pip wheel --no-build-isolation --no-deps "$repo/$component" --wheel-dir "$STATE_DIR/wheels"
done
python -m pip install --no-deps --force-reinstall "$STATE_DIR"/wheels/*.whl
# Resolve runtime requirements while pinning the native components just built.
python - <<'PYNATIVE' > "$STATE_DIR/setup/native-constraints.txt"
import importlib.metadata as m
for name in ("tokenspeed", "tokenspeed-engine", "tokenspeed-scheduler", "tokenspeed-kernel"):
    print(f"{name}=={m.version(name)}")
PYNATIVE
python -m pip install --constraint "$STATE_DIR/setup/native-constraints.txt" "$STATE_DIR"/wheels/tokenspeed-*.whl
python -m pip freeze --all > "$STATE_DIR/setup/requirements-resolved.txt"
python "$repo/.agents/deploy/setup/identity.py" --source "$repo" --output "$STATE_DIR/setup/identity.json"
python -m pip check
