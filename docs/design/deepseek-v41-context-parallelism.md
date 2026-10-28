# DeepSeek V4.1 context parallelism

V4.1 has three independent responsibilities: QCP partitions prefill queries,
KVP partitions persistent global KV and indexer K pages, and DCP computes
decode attention over those pages. All groups are subgroups of attention TP;
these modes do not add devices.

## Query computation

The model-specific indexer QCP path interleaves query rows across attention TP
before indexer query/weight projection. Each rank scores its local queries
against the complete logical history. Logical Top-k IDs and lengths are
gathered back in original row order. The candidate-producing Full indexer and
CED-tail Reindex remain on their existing paths.

Attention QCP exchanges local-head queries for a contiguous subset of query
rows with all heads. Selected prefill attention computes that subset, then an
inverse exchange restores the TP head layout. SWA-only and global-attention
layers have separate minimum query counts to amortize communication.

These are operator-local partitions. Hidden states, Engram, MoE, compressor
work, and DSpark auxiliary states retain their existing layouts. Generic
`--query-context-parallel-size` remains the separate DSA model-wide layout.

## Cache ownership

The global ratio-1 and ratio-2 cache groups use existing page-cyclic KVP
ownership. Indexer K shares the ownership of its global KV group. Writes
translate virtual slots to local slots and mask nonowners; logical selections
remain valid across all owners. Prefix matching, allocation, transfer and
eviction continue to use LCM cache groups.

SWA, compressor tails and DSpark state remain replicated. Prefill materializes
one request's packed history at a time. Communication copies complete packed
pages, preserving quantized values and scale offsets without dequantizing
and requantizing them.

## Decode computation

Decode and DSpark verification share the metadata-refresh path for eager
execution and CUDA Graph replay. Query heads are gathered across the KVP
group; each rank attends its owned global pages. Replicated SWA contributes
on one rank, and the attention sink is applied once when merging the partial
outputs by log-sum-exp.

On supported Hopper indexer paths, ordinary Full and Reindex compute local
Top-k candidates and merge their scores and logical IDs across ranks.
The candidate-producing Full source retains complete-history materialization
because its block maxima feed later Reindex layers. Unsupported native paths
also materialize history. This is exact distributed Top-k selection under the
kernel's score/tie rules, not a reduced candidate budget.

Graph decisions use static capacity bounds, never the current verify width as
a substitute for history length. Pointer-keyed selection and geometry caches
are scoped to one forward and invalidated by metadata refresh.

## Configuration

For TP8, enable either or both operator-level QCP paths with:

```text
--deepseek-v41-indexer-query-parallel-size 8
--deepseek-v41-attention-query-parallel-size 8
--disable-prefill-graph
```

Each query size is either 1 (disabled) or the attention TP size. Indexer QCP
requires at least one query per rank and history larger than the exact
all-keys selection budget. There is no additional minimum local query or
history size: communication break-even is measured per workload. Mixed,
candidate-producing, Reindex, and decode eligibility stays unchanged.
Attention thresholds are set by `--deepseek-v41-attention-query-min-tokens` and
`--deepseek-v41-attention-query-swa-min-tokens`.

For V4.1 aggregated KVP+DCP, select the backend and matching sizes:

```text
--attention-backend deepseek_v41
--kv-parallel-size 8
--decode-context-parallel-size 8
```

KVP may be any supported divisor of attention TP. Shared launch validation
requires matching KVP/DCP sizes for aggregated and PD decode engines, and
separate prefill/decode rounds for aggregated DCP. QCP can be enabled with
either replicated or sharded KV. V4.1-specific restrictions are listed below.

PD prefill can use KVP with DCP size 1 and
`--disaggregation-layerwise-interval 0`. V4.1 PD decode with KVP is not
supported. KVP requires eager prefill, no mixed batches, and either no
speculation or DSpark. CUDA Graph decode is supported.

## Validation and tradeoffs

Runtime tests cover query ordering, head exchange, packed-page transport,
cache ownership, metadata refresh and selection reuse. Kernel tests compare
partial outputs/LSE against a reference, local Top-k against replicated
scoring, and repeated graph replay with changing lengths.

QCP targets long-prefill latency. KVP trades communication for persistent KV
capacity; V4.1's small selected-attention workload does not imply a decode
speedup. Performance and model accuracy need paired workload measurements.
Changing collective/reduction order can change greedy outputs, so equal task
accuracy is not a claim of token-for-token equivalence.

## Encoder-cache attention DP

A PP1 encoder-cache engine can assign requests to attention-DP owners with
ordinary expert parallelism. Each owner retains its own request and cache
pages; the local prefill completion emits no sampled token and performs no
PD transfer. The engine remains eager, without mixed batches or overlapping
scheduler rounds. PP with attention DP remains unsupported.

An idle expert peer enters every encoder layer's ordinary EP collective.
It skips the final cache producer: that boundary has no expert collective,
and an idle owner has no rows to write. This is ordinary request-level DP,
not splitting a single request's queries across replicas. Full-model query
context parallelism still requires a separate V4.1 layout/backend contract.


## Experimental encoder-cache model-wide query shards

The encoder-cache-only layout can use attention TP as the query group, with
head TP1, KVP1, expert TP1 and DeepEP over that group. It is currently PP1,
eager, pure prefill, without speculation or the operator-local query exchanges.
Every scheduled forward must have at least one query per rank; metadata rejects
empty shards on all ranks before any collective. This is an experimental
contract, not a performance or general serving claim.

The executor slices token IDs, positions, prepared global Engram hashes and
masks together before model entry; the model must not slice them again.
Vocabulary weights are replicated over query owners so different local token
IDs never enter a vocabulary-TP all-reduce. Prepared hashes keep their global
ngram history at shard boundaries;
Engram table/head projections follow each query owner with replicated heads.
Attention projections and mHC run only local rows. Expert communication consumes
those rows directly, without applying replicated-input TP ownership a second time.

Each source gathers its normalized hidden rows in canonical request order
before the existing ratio-2 compressor and global/index-K writers. A temporary
canonical metadata view preserves pairing, tails, replay floors and LCM page
addresses; it contains only the current forward, not request history.
Every rank therefore publishes the same complete compressed state.

SWA gathers only the new 512-wide key rows. The existing prefill path protects
old prefixes before publishing current keys, and retains the full current-key
span while slicing only query rows/indices and outputs. Publishing a ring
before those old-prefix reads would corrupt shard-boundary attention.
All persistent cache state stays in the existing LCM groups.

### Expert replicas within the query group

For eager encoder-cache V4.1, the DeepEP group can be a divisor of QCP
(EP2, EP4 or EP8 under QCP8), with expert TP1. Each contiguous expert group
contains every global expert once, and different groups replicate weights.
All query owners retain different rows; shared-expert weights are complete
on each owner, and dispatch/combine returns those same owned rows.

The generic communication manager admits such narrower expert groups only
when the model explicitly declares layer-owned row communication. Other
models retain the replicated-forward guard. The V4.1 encoder-cache admission
excludes the replicated decode arm, and expert group token counts are selected
from the canonical query shard table, rather than the full query-group sum.

This trades smaller communication domains and lower fanout against more
expert weights per GPU and fewer rows per expert per replica. It does not
reduce the global expert arithmetic and needs paired measurements.


### Experimental SM120 Cake expert path

The eager encoder-cache QCP8/EP4 layout admits the explicit
`flashinfer_moe_ep` transport, `flashinfer_cake_sm120_moe_ep` provider,
`moe_mxfp4_fp8_activation=True` and `moe_combine_order=slot`.
It consumes the same owned query rows and agreed expert-group row counts.
The complete shared expert remains a separate computation on each owner.

This provider qualifies FlashInfer commit `399b6c1d`, its Cake device-source
manifest, and the compatible NCCL Device ABI. All remote peers use GIN in
this qualified configuration. Cake owns dispatch, both expert GEMMs,
activation, result return and slot-order combination; no DeepEP buffer or
outer all-reduce is allocated. Its NCCL-based environment is isolated from
the production DeepEP/NVSHMEM environment.

The library transforms the existing MXFP4 checkpoint once. Parameters alias
the transformed weight planes so raw weight copies are released.
Compatible layers share one serial workspace. The logical per-owner bound
is the host prefill bound divided by QCP8 with upward rounding; physical
workspace capacity is rounded to 128 rows. The provider prepares a 512-row
bucket and the logical capacity bucket before serving. Live ragged sources,
including an empty source, use the same host-agreed bucket on all EP peers.
This bounds compilation rather than specializing every request length.

A same-stream device assertion checks each layer's protocol error flag
before the next layer clears the shared workspace. The returned tensor is
cloned so consumers do not retain the next layer's shared output slot.

Input and intermediate activations are quantized to K32 MXFP8 and final
rows are BF16. The route weight is applied before intermediate quantization,
and contributions are combined in top-k slot order. This differs from the
original Marlin W4A16 / DeepEP rank-order path. Independent same-format
operator tests, full-model task quality and serving performance are
separate acceptance gates; passing the first does not imply the others.


### Local W4A8 expert alternative

The same encoder-cache CP8 / dual EP4 row ownership can select
`flashinfer_cutlass_sm120_deepep`. DeepEP normal continues to transport BF16
rows and return rank-combined contributions. Only the local expert path changes:
original MXFP4 weights, MXFP8 input quantization, CUTLASS grouped GEMM,
clamped SwiGLU with MXFP8 output, and CUTLASS grouped GEMM/finalize.
Static expert placement and the replicated shared expert retain their
existing meanings. Explicit activation precision and rank combination are
required; Cake's slot-order combination is a separate configuration.


For the explicit local CUTLASS W4A8 provider, DeepEP still transfers BF16 rows.
After dispatch, each receiver quantizes its own rows to K32 MXFP8 and runs both
MXFP4 expert projections with a fused activation/quantization step between them.
The local expert call skips process-wide tactic profiling because received row
counts need not agree across ranks. This restriction leaves dense-op profiling
enabled and does not alter query ownership, cache visibility, or EP transport.
