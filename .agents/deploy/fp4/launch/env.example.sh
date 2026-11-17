# Source a local copy after filling absolute paths and reachable node addresses.
# W4A8 requires original MXFP4 MoE / MXFP8 Engram; FP4 requires
# an independently converted NVFP4 MoE / MXFP4 Engram checkpoint.
# Both launchers currently use Elastic FP8 dispatch / BF16 combine.
export VENV=/path/to/venv
export CUDA_HOME=/usr/local/cuda-13.0
export STATE_DIR=/path/to/run/_local/state
export MODEL_PATH=/path/to/complete/checkpoint
export GATEWAY_HOST=GATEWAY_HOSTNAME
export NODE_HOST=THIS_NODE_HOSTNAME
export P_HANDSHAKE_PORT=18510
export D_HANDSHAKE_PORT=18511
export P_DIST_INIT_ADDR=PREFILL_HOSTNAME:19400
export D_DIST_INIT_ADDR=DECODE_LEADER_HOSTNAME:19500
# Decode leader: 0; follower: 1. Prefill ignores NODE_RANK.
export NODE_RANK=0
# If the host C++ runtime is older than the selected compiler, set its library directory.
# export TOOLCHAIN_LIBRARY_DIR=/path/to/compiler/lib
# Optional requested NIC environment map, in CUDA_VISIBLE_DEVICES order:
# export TOKENSPEED_SM120_DEEPEP_NICS=HCA_GPU0,HCA_GPU1,HCA_GPU2,HCA_GPU3,HCA_GPU4,HCA_GPU5,HCA_GPU6,HCA_GPU7
# Elastic uses NCCL's communicator; the map above does not prove actual HCA binding.
# Restrict the NCCL HCA allowlist before initialization using actual local names:
# export NCCL_IB_HCA='=HCA_GPU0,=HCA_GPU1,=HCA_GPU2,=HCA_GPU3,=HCA_GPU4,=HCA_GPU5,=HCA_GPU6,=HCA_GPU7'
# Set the actual RoCE GID index/socket interface when required by the local fabric:
# export NCCL_IB_GID_INDEX=GID_INDEX NVSHMEM_IB_GID_INDEX=GID_INDEX
# export NCCL_SOCKET_IFNAME=INTERFACE NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=INTERFACE
