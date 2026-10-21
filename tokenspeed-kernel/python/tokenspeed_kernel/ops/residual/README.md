# Gated residual mixer warmup

The Blackwell fused CuTe mixer uses runtime row counts and dynamic launch grids.
Compiled plans specialize on the CTA tactic, full/tail tiles, resident workers,
scheduling rounds, model geometry, projection scale and PDL setting. The
cooperative workspace and occupancy rules remain in the same launcher used by
serving.

During startup autotuning, `gated_residual_mix` warms all eligible fused mixer
configurations for 1–256 rows, even if the model's initial forward uses the
larger GEMM-based mixer. Exact fused overrides also warm the diagnostic range
through 1024 rows. Enumerating rows reaches occupancy-dependent round counts
without a separate scheduler. Equal configurations reuse their compiled plans.

Warmup uses scratch activations and the loaded projection weights, and is
cached by model geometry, dtype, scale, PDL and stream. Explicit choices of
other solutions skip it. Each graph capture stream still needs its ordinary
eager warmup for that stream's occupancy result. Disabling startup autotuning
skips the comprehensive warmup.
