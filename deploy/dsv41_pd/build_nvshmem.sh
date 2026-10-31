#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
source_dir="$SOURCE_DIR/p-nvshmem"
build_dir="${NVSHMEM_BUILD_DIR:-$STATE_DIR/build-nvshmem}"
export CC="${NVSHMEM_CC:-${CC:-cc}}"
export CXX="${NVSHMEM_CXX:-${CXX:-c++}}"
require P_NVSHMEM_LIB_DIR
prefix="$(dirname "$P_NVSHMEM_LIB_DIR")"
cmake -S "$source_dir" -B "$build_dir" \
 -DCMAKE_BUILD_TYPE=release -DCMAKE_CUDA_ARCHITECTURES=120 -DCUDA_HOME="$CUDA_HOME" \
 -DCMAKE_INSTALL_PREFIX="$prefix" -DNVSHMEM_PREFIX="$prefix" \
 -DNVSHMEM_BUILD_TESTS=OFF -DNVSHMEM_BUILD_EXAMPLES=OFF -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
 -DNVSHMEM_MPI_SUPPORT=OFF -DNVSHMEM_UCX_SUPPORT=OFF -DNVSHMEM_USE_GDRCOPY=OFF \
 -DNVSHMEM_USE_NCCL=OFF -DNVSHMEM_IBRC_SUPPORT=ON -DNVSHMEM_IBGDA_SUPPORT=ON \
 -DNVSHMEM_USE_MLX5DV=ON
cmake --build "$build_dir" -j "$MAX_JOBS"
cmake --install "$build_dir"
