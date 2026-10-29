# DeepSeek V4.1 QCP prefill with disaggregated decode

The normal prefill role can shard encoder query rows with replicated KV and
expert-owned DeepEP dispatch. Before the CED decoder, gather the row state and
restore canonical attention metadata. The CED tail then uses replicated rows;
expert dispatch must report the full row count on each participating rank.
Odd-width byte masks use an exact int16 wire representation for this gather.

The supported layout uses KVP1, head TP1, no pipeline parallelism or mixed
prefill/decode, and no prefill graph. DSpark is supported; the prefill engine
prepares its context while the decode engine performs speculative generation.
CUTLASS SM120 local experts accept EP4 or EP8. A normal-only DeepEP buffer may
serve target and draft expert counts with the same hidden width; buffers with
low-latency legs retain strict expert-count checks.

The READY handshake advertises the prefill bootstrap host and port so a remote
gateway can rendezvous with the prefill worker. Missing fields remain valid for
legacy peers. CPU metadata and row-count tests cover the CED transition, and
wire tests cover the bootstrap fields and legacy decoding.
