# V4.1 encoder cache pipeline

The explicit `--prefill-only-mode encoder-cache` runs a local zero-output
cache producer using the ordinary scheduler, input preparation and model execution
path. Requests must set `max_new_tokens=0`. There is no PD peer or transfer.

The model loads its encoder layers and the candidate source's cache projections.
The final stage writes that global cache, without running candidate attention,
the remaining decoder, the LM head or sampling. Completion waits for the final
prompt result. Unlike generation, the computed cache frontier includes every
input token; the last aligned prefix page is published for later requests.

Each pipeline boundary carries the complete HC residual and input mixes, the live
shared owner's packed pages and the current logical sparse selections. Packed
values and scales travel together. Request-row/position descriptors resolve
receiver-local pages, so physical page IDs never cross stages. V4.1 FlatKV is an
indivisible plane: every stage keeps full field views while weight loading and
execution remain stage-local. Prepared input spans derive the wire geometry;
no activation count is downloaded to size a receive. Every stage completes its
deferred HC post before sending the boundary bundle.

Immutable Engram prompt windows come from the scheduler's request state for each
forward. The chunk pipeline may therefore have multiple encoder chunks in flight;
autoregressive decode feedback is excluded. Attention DP and KVP are excluded
from this recipe.

MoE communication is explicit: `none` uses the ordinary EP partial-output
reduction; `nccl` chooses dynamic destination-deduplicated dispatch/combine.
Replicated attention rows first receive unique token owners. Each owner routes
its tokens, dispatches rows/TopK metadata to the ranks owning selected experts,
combines returned partials, adds its local shared expert, then restores attention's
replicated row order. Routed experts use TP1 and EP partitions. EP1 reduces to
local expert execution. This eager NCCL implementation exchanges dynamic counts
through the host; it is a correctness baseline rather than a graph-safe backend.

Prefill-only benchmarks measure the zero-output cache completion response.
Profile requests run separately after priming, with distinct suffixes to avoid
accidental full-cache hits. Compare equal new-token counts and actual stage
layer work; normal pipeline filling and draining are not redundant computation.
