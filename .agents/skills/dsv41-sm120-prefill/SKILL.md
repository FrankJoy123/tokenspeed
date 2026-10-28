---
name: dsv41-sm120-prefill
description: Locate and run DeepSeek V4.1 Flash SM120 encoder-only Prefill and TokenSpeed DeepEP, Marlin, FlashInfer local W4A8 and Cake MoE paths. Use for this model and hardware.
---

# DeepSeek V4.1 Flash on SM120

This branch contains a local FlashInfer CUTLASS W4A8 candidate, Marlin baseline and Cake candidate. Record the source commit, compiled kernels, backend options, checkpoint and input. Local W4A8 and Cake passed bounded functional and Prefill performance checks; full model task quality and general deployment are not established.

## Code navigation

- Model, compressor and indexer: python/tokenspeed/runtime/models/deepseek_v41.py and deepseek_v41_moe.py.
- Attention: python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py.
- Backend selection and validation: python/tokenspeed/runtime/layers/moe/expert.py, utils.py, and python/tokenspeed/runtime/utils/server_args.py.
- Local W4A8: tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer_cutlass_sm120_deepep.py; Marlin: ops/moe/marlin.py; Cake: ops/moe/flashinfer_cake_sm120.py.
- Tests: test/runtime/test_server_args_context_parallel.py and tokenspeed-kernel/test/nvidia/ops/moe/. Use the matching checkout and compiled runtime.

## Environment and launch

Use a Python virtual environment, compatible CUDA 13 and PyTorch CUDA, a FlashInfer build with SM120 MXFP4/MXFP8 support, and a DeepEP normal backend built for the actual transport. Build the TokenSpeed kernel and scheduler from the selected source; verify imports and compiled binary identity. Obtain the complete checkpoint and static expert-placement map separately. Check GPU ownership and port availability; do not stop unrelated processes.

For ordinary service installation and entry points read docs/guides/getting-started.md and docs/guides/launching.md. Check tokenspeed serve --help for options in this revision. The measured encoder-only Prefill used the Engine API and a separate HTTP benchmark harness; it is not a full-model serving recipe. Reproduce it only with the matching harness, placement map, workload, runtime and environment. Prefill completion is not TTFT or full-model quality.

The measured CP8 / dual EP4 setup used eight GPUs, query_context_parallel_size=8, attn_tp_size=8, attn_head_tp_size=1, dense_tp_size=1, ep_size=4, moe_tp_size=1, pipeline_parallel_size=1, data_parallel_size=1, prefill_only_mode="encoder-cache", eager execution, static expert placement and chunk size 16640. Both local options use BF16 input, all2all_backend="deepep", deepep_mode="normal" and moe_combine_order="rank":

~~~python
# Apply as Engine keyword arguments with the actual model, topology and map.
marlin = dict(moe_backend="marlin", moe_mxfp4_fp8_activation=False)
cutlass = dict(
    moe_backend="flashinfer_cutlass_sm120_deepep",
    moe_mxfp4_fp8_activation=True,
)
~~~

Use ep_dispatch_algorithm="static" with a verified init_expert_location map. Compare with the same checkpoint, inputs, routing, backend and capacity. Match Engine/launcher arguments and library builds; consult Engine and server_args.py. Do not silently convert this single-machine Prefill recipe into a distributed P/D deployment.

Cake requires all2all_backend="flashinfer_moe_ep", moe_backend="flashinfer_cake_sm120_moe_ep", BF16 interface tensors, moe_mxfp4_fp8_activation=True and moe_combine_order="slot". It needs a compatible FlashInfer/NCCL device runtime and is a separate communication-and-compute candidate.

## Precision and verification

Marlin uses MXFP4 weights with BF16 activation (W4A16). Local CUTLASS uses MXFP4 weights with MXFP8 activation (W4A8); DeepEP transports BF16, and FC1, SwiGLU/quantization, FC2 and combine have distinct rounding boundaries. Cake also uses W4A8 but applies routing weights at a different boundary. Compare first against an independent matching-format reference, then against W4A16 outputs. Test empty/ragged ranks and graph replay where applicable. Record warmup, input/cache identity, per-rank timing, ownership and JIT; task quality requires separate evaluation.
