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
# Optional per-visible-GPU NIC map, in CUDA_VISIBLE_DEVICES order (use actual HCA names):
# export TOKENSPEED_SM120_DEEPEP_NICS=HCA_GPU0,HCA_GPU1,HCA_GPU2,HCA_GPU3,HCA_GPU4,HCA_GPU5,HCA_GPU6,HCA_GPU7
