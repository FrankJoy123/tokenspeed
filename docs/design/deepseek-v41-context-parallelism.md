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


### Experimental SM120 FlashInfer expert path

The eager encoder-cache QCP8/EP4 layout also admits the explicit
flashinfer_moe_ep all-to-all backend, with the
flashinfer_cake_sm120_moe_ep MoE provider and moe_mxfp4_fp8_activation enabled.
It uses the same owned query rows, expert group row counts and replicated
shared expert. It does not allocate a DeepEP buffer.

The optional FlashInfer0.7.2 SM120 library transforms the existing packed
MXFP4 checkpoint once. Parameters alias transformed weights so no raw weight
copy survives preprocessing. EP4 layers share the library workspace at the
configured per-owner capacity, derived from the host prefill bound divided
by QCP8. The host supplies the agreed maximum EP source count on each call.

The verified prefill provider prepares1024-row and capacity buckets before
cache profiling and serving. Smaller live row spans are padded inside the
prepared1024 bucket; they never compile a new tiny decode kernel. Tiny
Green Context buckets are not part of this provider's contract. This trades
extra small-input work for a bounded prefill compilation set.

This path quantizes input and intermediate activations to MXFP8 and returns
BF16 finalized rows. It is not numerically equivalent to Marlin W4A16.
Operator-reference correctness, model quality and serving performance are
separate gates. The deployment must supply the library's compatible Python
and host NVSHMEM pair without injecting the production DeepEP preload.


Cake candidate: FlashInfer399b6c1d, SM120 MXFP4 weights / K32 MXFP8 activations, all-GIN transport, explicit EP4 encoder-cache CP8 only. Source manifest and NCCL Device ABI are checked. Transformed parameters alias the retained weight planes; one compatible workspace is shared by serial layers. A same-stream device assertion checks each protocol flag before reuse. Numerical-contract tests do not replace full-model task quality.

The Cake provider requires --moe-combine-order slot: it returns complete token outputs in top-k slot order, so no outer all-reduce is permitted. The original Marlin/DeepEP rank-order configuration remains a separately frozen baseline.
