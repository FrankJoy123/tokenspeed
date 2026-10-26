# Residual kernels

## SM120 mHC coefficients

The SM120/SM121 mHC coefficient implementation uses compensated TF32x3 dot
products with FP32 mixing weights and accumulation. It keeps the portable
split geometry, FP32 square sums and Sinkhorn mapping. Row tiles stay below
64 to retain the compensated lowering. This changes floating-point reduction
rounding, not the stored weight dtype; FP64 coefficient and graph-replay tests
cover its numerical contract. Other architectures retain their implementations.
