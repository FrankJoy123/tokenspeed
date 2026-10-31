#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
case "$ROLE" in
 prefill)
  require P_DEEPEP_SOURCE P_FLASHINFER_SOURCE P_NVSHMEM_LIB_DIR
  [[ -f "$P_NVSHMEM_LIB_DIR/libnvshmem_host.so.3" ]] || { echo 'Build the validated ahfix NVSHMEM source first; see dependencies.tsv' >&2; exit 2; }
  source "$DEPLOY_DIR/prepare_rdma_headers.sh"
  export TORCH_CUDA_ARCH_LIST=12.0
  export NVSHMEM_DIR="$(dirname "$P_NVSHMEM_LIB_DIR")"
  export LD_LIBRARY_PATH="$P_NVSHMEM_LIB_DIR:$LD_LIBRARY_PATH"
  # Keep successful artifacts when another dependency fails.
  mkdir -p "$STATE_DIR/dependency-wheels"
  wheel_dir=$(mktemp -d "$STATE_DIR/dependency-wheels/deepep.XXXXXX")
  "$VENV/bin/python" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$wheel_dir" "$P_DEEPEP_SOURCE"
  "$VENV/bin/python" -m pip install --no-deps "$wheel_dir"/*.whl
  # The validated source tree is retained as part of this deployment.
  "$VENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$P_FLASHINFER_SOURCE"
  ;;
 decode)
  require D_DEEPGEMM_SOURCE
  [[ "$(git -C "$D_DEEPGEMM_SOURCE" rev-parse HEAD)" == 14073b4e1e706506e193231209738c848d092a1f ]] || { echo 'Unexpected DeepGEMM revision' >&2; exit 2; }
  if git -C "$D_DEEPGEMM_SOURCE" apply --check "$DEPLOY_DIR/patches/deepgemm.patch"; then
   git -C "$D_DEEPGEMM_SOURCE" apply "$DEPLOY_DIR/patches/deepgemm.patch"
  else
   git -C "$D_DEEPGEMM_SOURCE" apply --reverse --check "$DEPLOY_DIR/patches/deepgemm.patch"
  fi
  [[ -z "$(git -C "$D_DEEPGEMM_SOURCE/third-party/cutlass" status --porcelain)" ]] || { echo 'CUTLASS has local changes; refusing checkout' >&2; exit 2; }
  git -C "$D_DEEPGEMM_SOURCE/third-party/cutlass" fetch origin f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
  git -C "$D_DEEPGEMM_SOURCE/third-party/cutlass" checkout --detach f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
  # A cached upstream wheel must not bypass the SM120 source patch.
  export DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=1
  "$VENV/bin/python" -m pip install --force-reinstall --no-cache-dir --no-build-isolation --no-deps "$D_DEEPGEMM_SOURCE"
  dependency_root=$("$VENV/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("deep_ep").origin).parent.parent)')
  "$VENV/bin/python" "$DEPLOY_DIR/apply_patch.py" "$dependency_root" "$DEPLOY_DIR/patches/elastic-rdma-tuner.patch"
  ;;
 gateway)
  require GATEWAY_SOURCE
  [[ "$(git -C "$GATEWAY_SOURCE" rev-parse HEAD)" == c52056d4c00a1bccd69836b466e5565ba26c13f0 ]] || { echo 'Unexpected gateway revision' >&2; exit 2; }
  # Resolve Rust from the gateway checkout, not the caller's TS directory.
  gateway_rust=$("$VENV/bin/python" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["toolchain"]["channel"])' "$GATEWAY_SOURCE/rust-toolchain.toml")
  rustup toolchain install "$gateway_rust" --profile minimal
  export RUSTUP_TOOLCHAIN="$gateway_rust"
  export CARGO_BUILD_JOBS="$MAX_JOBS" CARGO_TARGET_DIR="$STATE_DIR/gateway-target"
  "$VENV/bin/maturin" build --locked --release --manifest-path "$GATEWAY_SOURCE/bindings/python/Cargo.toml" --interpreter "$VENV/bin/python" --out "$STATE_DIR/gateway-wheels"
  "$VENV/bin/python" -m pip install "$STATE_DIR"/gateway-wheels/*.whl
  ;;
 *) echo 'ROLE must be prefill, decode or gateway' >&2; exit 2 ;;
esac
