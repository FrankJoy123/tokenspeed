# V4.1 static expert placement
V4.1 accepts an explicit initial placement through --init-expert-location and
--ep-dispatch-algorithm static (or static_with_zero_expert) with Marlin W4A16, DeepEP normal, and MoE TP1.
The map is a per-layer permutation: every logical expert has exactly one
physical slot, and the EP group and per-rank slot count stay unchanged.

The gate selects logical experts in the original checkpoint order, including
the fused V4.1 gate path. Only those selected IDs are mapped onto physical
slots; scores, slot order, route weights and output scaling are preserved.
The checkpoint coverage list, remote-expert filtering and common MoE loader
consume the same placement, so weight and scale constituents follow their
logical expert together. Placement is applied before weight preprocessing. Host-side static-map
planning allocates its temporary tables on CPU even when model construction
uses a CUDA default device; only the completed table follows the placement device.

Trivial placement keeps the original routing object. This admission does not
enable online EPLB, redundant experts, NCCL dispatch or MegaMoE placement.
Those paths retain their own validation and need separate integration.

Reordering can change the grouping of floating-point contributions in
distributed combine. It preserves logical model work, not bitwise output
across placements. Performance must include dispatch through combine:
spreading expert compute may increase distinct destination ranks and bytes.
