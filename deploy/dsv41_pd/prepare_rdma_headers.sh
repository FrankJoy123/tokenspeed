#!/usr/bin/env bash
# Source after common.sh. Expose only RDMA headers, not the system libc headers.
rdma_root="${RDMA_INCLUDE_DIR:-/usr/include}"
rdma_overlay="$STATE_DIR/rdma-include"
mkdir -p "$rdma_overlay"
for name in infiniband rdma; do
 [[ -d "$rdma_root/$name" ]] || { echo "Missing RDMA development headers: $rdma_root/$name" >&2; exit 2; }
 if [[ -L "$rdma_overlay/$name" ]]; then
  [[ "$(readlink "$rdma_overlay/$name")" == "$rdma_root/$name" ]] || { echo "RDMA header link differs: $rdma_overlay/$name" >&2; exit 2; }
 elif [[ -e "$rdma_overlay/$name" ]]; then
  echo "Unexpected RDMA overlay entry: $rdma_overlay/$name" >&2; exit 2
 else
  ln -s "$rdma_root/$name" "$rdma_overlay/$name"
 fi
done
[[ -f "$CUDA_HOME/include/cccl/cuda/std/tuple" ]] || { echo "Missing CUDA CCCL headers under $CUDA_HOME/include/cccl" >&2; exit 2; }
export CPATH="$rdma_overlay:$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
