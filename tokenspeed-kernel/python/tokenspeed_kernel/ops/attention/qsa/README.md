# QSA compilation and startup warmup

The Blackwell CuTe sparse-attention kernel takes query/output row counts,
selected-slot row counts and strides, prefix flags, cache values and scales
at runtime. Q/O are contiguous: their strides follow the model's fixed head
geometry. Selected slots may use padded rows or non-unit column strides.
The selection width remains 2051; Q/KV head counts, head dimension and cache
slot capacity also remain static.

Compilation specializes on a finite execution configuration: query grouping,
head tile, KV splits, rows per CTA, pipeline stages, PDL and prefix detection.
A single additional small/large-grid flag controls prefix bucketing for
two-query groups with twelve heads per KV head. Query groups must divide each
uniform request's query width. Prefix equality is checked again on every call
or graph replay; it never enters the compilation key.

During the existing startup `autotune(tune_mode=True, ...)` window, ordinary
QSA calls warm one representative of each reachable CuTe execution configuration
within the token budget. This includes decode configurations encountered later
than the initial prefill. Scratch queries and masked selections leave model
cache contents untouched.

Streaming top-k already uses dynamic row counts and strides. Startup warms its
small, split, grouped, dense and overlapping algorithms across capacity buckets.
For shortened prefill tables, the allocation's row stride supplies the maximum
capacity. Zero block frontiers avoid reading request-owned keys during warmup.
Warmup eligibility checks the shared tensor layout independently of the initial
batch's tactic, so a single-row startup does not exclude grouped configurations
when the scratch budget is zero. The original budget filters each configuration;
with zero scratch, the prefill, dense and overlap kernels are still warmed.

Warmup is cached per model/cache geometry and PDL setting. Explicit choices of
other solutions skip CuTe warmup. Disabling startup autotuning skips this
comprehensive warmup; a new execution configuration or cache capacity can still
compile when first used. CUDA graph ladders continue to warm their capture
streams normally.
