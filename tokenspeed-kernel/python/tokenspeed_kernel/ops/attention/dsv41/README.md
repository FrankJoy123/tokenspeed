# V4.1 selected prefill

The portable path expands a request's protected SWA rows and compressed history
into a BF16 workspace. Its indices address that workspace. Native implementations
must retain the same causal selection and count the attention sink once.

SM120/SM121 can use try_paged_prefill. It receives the already codec-rounded
SWA prefix and current rows separately from the original FP4 global pages.
It repacks only SWA into temporary 64-row planar pages; that round trip preserves
the decoded values (a different valid scale may change packed bytes). It never writes the persistent SWA ring, so a long chunk cannot
overwrite old prefix rows before their queries read them. Compressed history
stays in its original E2M1/E4M3 format and is selected by physical slot, avoiding
a full-history BF16 gather.

The optional FlashInfer 0.7.2 reader uses SparseMLASm120Wrapper with explicit
compute_precision="bf16": BF16 QK and PV with FP32 accumulation and softmax.
No Q/P FP8 route is selected. Outputs can differ from the vector FP32-reduction
fallback due to reduction order and BF16 PV operands; numerical tests cover the
joint softmax, empty rows, sinks and refreshed selections. Runners own scratch
and are cached separately by device and stream; eager warmup must precede graph
capture. Extends of 64 rows or fewer are masked-padded to 65 rows to retain the
same BF16 prefill route across chunk sizes instead of switching to split decode. This first integration serves eager prefill with KVP1 and no attention
head exchange. Other layouts keep their existing path.

FlashInfer is imported only by the kernel package's third-party adapter.
No implementation is advertised outside SM120/SM121. A missing registered reader
returns None from the family helper; errors from a selected reader are surfaced.
The CUDA environment must provide a compatible optional FlashInfer installation.

## SM120 indexer

The SM120/SM121 registered Triton launcher keeps the portable scoring and
selection math. Full MXFP4 scans with 32 index heads, at least 256 queries,
Top512 and a 256-key score tile use eight warps instead of four to reduce
register spilling. Short queries, candidate production and Reindex retain four
warps. Other architectures retain the portable implementation. TP head gathering
and rank-ordered rounding are unchanged; the optimization changes launch geometry,
not cache format or selection semantics.


The optional SM120 FlashInfer Full Top512 route preserves the same quantized
query/key decode, BF16 dot and rank-ordered head rounding. It materializes at
most 256 query rows by 65536 history columns as FP32 scores, then uses public
graph-safe TopK with smaller-ID boundary ties. The existing finalizer restores
ascending logical IDs and -1 padding, so the library need not return a fixed
ordering. Candidate production, Reindex and short/unmeasured shapes retain the
streaming selector. Scratch lives only for one query tile; history/cache bytes
and the model's selection contract are unchanged.
