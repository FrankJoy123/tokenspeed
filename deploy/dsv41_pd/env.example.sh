# Copy to env.local.sh. Paths can be relative to this repository root.
# This file is sourced by bash; it must be trusted.
ROLE=prefill                     # prefill | decode | gateway
VENV="$REPO_ROOT/.venv-pd"
CUDA_TOOLKIT_MODE=system         # system | python (install CUDA into the new venv)
CUDA_HOME=/usr/local/cuda-13.0
# For python mode: CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
MODEL_PATH="$REPO_ROOT/models/DeepSeek-V4.1-Flash-1017"
P_HOST=CHANGE_ME
D_HOST=CHANGE_ME
GATEWAY_HOST=CHANGE_ME
EP_NIC_NAME=CHANGE_ME
NCCL_IB_HCA=CHANGE_ME
NCCL_IB_SL=1
# Explicit topology choices; these match the measured PCIe/RDMA setup.
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
GPU_MEMORY_UTILIZATION=0.85
P_CHUNK_SIZE=16384
D_CHUNK_SIZE=8192
MAX_NUM_SEQS=32
MAX_MODEL_LEN=1048576
SIMULATED_ACCEPT_LEN=             # empty: real acceptance; 4: performance-only simulation
STATE_DIR="$DEPLOY_DIR/.runtime/$ROLE"
SOURCE_DIR="$DEPLOY_DIR/.runtime/sources"
# Optional package mirror for private/nightly wheels; never store credentials here.
# PIP_INDEX_URL=https://your-package-mirror/simple
# P needs the validated PCIe DeepEP / FlashInfer / NVSHMEM variants.
# Fill Git URLs + full commit SHAs in sources.local.tsv, or supply checked-out sources.
P_DEEPEP_SOURCE="$SOURCE_DIR/p-deepep"
P_FLASHINFER_SOURCE="$SOURCE_DIR/p-flashinfer"
P_NVSHMEM_LIB_DIR="$SOURCE_DIR/p-nvshmem/install/lib"
D_DEEPGEMM_SOURCE="$SOURCE_DIR/deepgemm"
GATEWAY_SOURCE="$SOURCE_DIR/gateway"
# Optional extra shared-library directories for compiled dependencies.
EXTRA_LIBRARY_PATH=
# Benchmark uses original token IDs, not retokenized text. Supply the 32-session dataset.
SESSIONS_FILE="$DEPLOY_DIR/.runtime/data/sessions.json"
SESSIONS_SHA256=6abc53b0e713025136cb25cbf6cd9e5b4a339c74ce0ed6fec5fd0dda6384b0a5
OUTPUT_TOKENS=4096
ROUNDS=10
PRIME_TOKENS=102400
CHUNK_TOKENS=10240
WARMUP_PASSES=2
CONCURRENCIES="1 2 4 8 16 32"

# Avoid a read-only globally inherited /opt/uv-python.
UV_PYTHON_INSTALL_DIR="$STATE_DIR/uv-python"

# Optional node-local Triton cache; isolate by deployment and P/D role.
# PD_TRITON_CACHE_DIR=<writable-node-local-cache-directory>
