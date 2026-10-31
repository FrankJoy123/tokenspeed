#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
[[ "$CUDA_TOOLKIT_MODE" == python ]] || exit 0
[[ "$CUDA_HOME" == "$VENV/lib/python3.12/site-packages/nvidia/cu13" ]] || exit 2
mkdir -p "$STATE_DIR/downloads" "$STATE_DIR/cuda-tools"
archive="$STATE_DIR/downloads/cuobjdump-13.0.85.tar.xz"
if [[ ! -f "$archive" ]]; then
 curl -fL --retry 2 --max-time 180 -o "$archive" https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-13.0.85-archive.tar.xz
fi
printf '%s  %s\n' bd624dcb2089842add8f293efab1d21aa076c98b36d8dcb64347fe08fb03315d "$archive" | sha256sum -c -
tar -xf "$archive" -C "$STATE_DIR/cuda-tools" --strip-components=1
install -m 755 "$STATE_DIR/cuda-tools/bin/cuobjdump" "$CUDA_HOME/bin/cuobjdump"
if [[ ! -e "$CUDA_HOME/lib64" ]]; then ln -s lib "$CUDA_HOME/lib64"; fi
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
 [[ -f "$CUDA_HOME/lib/libcudart.so.13" ]]
 ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
"$CUDA_HOME/bin/nvcc" --version
"$CUDA_HOME/bin/cuobjdump" --version
