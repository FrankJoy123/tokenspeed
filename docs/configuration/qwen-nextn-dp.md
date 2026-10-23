# Qwen NextN with attention data parallelism

The Qwen3.5 NextN draft shares the target LM head. Both its head and its logits
processor follow mapping.lm_head, independently of the attention TP group.
With attention DP and the default LM-head width, each rank holds a complete
head and emits its own full-vocabulary logits. Attention TP2/DP2 therefore
does not reduce the draft argmax across attention peers.

An explicit --lm-head-tp-size uses the existing DP row-gather / vocabulary
transpose route on both target and draft. Its mapping restrictions still
apply. A tied draft embedding/head requires identical embedding and LM-head
TP groups; distinct layouts are refused before sharing weights.

Attention-DP idle ranks still participate in the caller's required layer
communication. The local FlashInfer CUTLASS BF16/FP16 expert kernel returns
an empty [0, hidden] tensor for an empty local batch, without entering
the vendor kernel that rejects a null input pointer. This return does not
skip the caller's dispatch/combine or post-MLP collectives.

Regression coverage:

- test/runtime/test_qwen35_nextn_dp.py: real target/draft construction on
  every rank, TP1/DP4 and TP2/DP2 full-vocabulary logits against a dense
  reference, tied-layout rejection, explicit head sharding and TP4 layout.
- tokenspeed-kernel/test/nvidia/ops/moe/test_flashinfer_empty.py: empty
  BF16/FP16 local expert batches with EP1/EP2 and either routing form.

Runtime serving and performance validation must also cover uneven active
DP groups, MTP, CUDA Graph replay and all workers' profiler exports.

The profiler scope scheduler.dp_metadata_all_gather includes the blocking
Python Work.wait() as well as C10d submission. Analyze other control-plane
collectives separately; a short metadata gather does not rule out a long
flush or weight-operation rendezvous.
