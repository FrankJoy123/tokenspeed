# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Large-M MXFP8 projection launcher for gfx1250 batches above 16 rows."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import triton
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.cluster_mm import (
    cluster_mm_config,
    launch_cluster_mm,
)


def launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project E4M3 matrices with row-major ``[1,32]`` raw UE8M0 scales.

    Args:
        A: Row-major E4M3 activation matrix ``[M,K]``; K divisible by 32.
        B: Row-major E4M3 weight matrix ``[N,K]``; N divisible by 16.
        A_scales: Raw UE8M0 bytes ``[M,K/32]`` with unit column stride.
        B_scales: Raw UE8M0 bytes ``[N,K/32]`` with unit column stride.
        out_dtype: Explicit output dtype; only BF16 is supported.
        alpha: Unsupported global output scale.
        block_size: Explicit scale block, which must be ``[1,32]``.
        out: Optional BF16 output buffer ``[M,N]`` with unit column stride.

    Returns:
        The caller-owned ``out`` tensor or a newly allocated BF16 result.
    """
    if alpha is not None:
        raise ValueError("gfx1250 MXFP8 GEMM does not support alpha")
    if block_size is None or tuple(block_size) != (1, 32):
        raise ValueError("gfx1250 UE8M0 MXFP8 GEMM requires block_size=[1, 32]")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    m, k = A.shape
    n = B.shape[0]
    if B.shape[1] != k:
        raise ValueError(f"B must be [N, {k}], got {tuple(B.shape)}")
    for name, tensor in (("A", A), ("B", B)):
        if (
            tensor.dtype != torch.float8_e4m3fn
            or not tensor.is_cuda
            or tensor.device != A.device
            or tensor.stride(-1) != 1
        ):
            raise ValueError(f"{name} must be row-major GPU E4M3 colocated with A")
    if A_scales is None or B_scales is None:
        raise ValueError("A_scales and B_scales are required")
    for name, scales, shape in (
        ("A_scales", A_scales, (m, triton.cdiv(k, 32))),
        ("B_scales", B_scales, (n, triton.cdiv(k, 32))),
    ):
        if (
            scales.dtype != torch.uint8
            or tuple(scales.shape) != shape
            or scales.device != A.device
            or scales.stride(-1) != 1
        ):
            raise ValueError(
                f"{name} must be row-major GPU uint8 {shape} colocated with A"
            )
    if out_dtype != torch.bfloat16:
        raise ValueError(f"gfx1250 MXFP8 GEMM requires BF16 output, got {out_dtype}")
    if m < 1 or k < 32 or k % 32 or n % 16:
        raise ValueError(
            "gfx1250 MXFP8 GEMM requires M >= 1, K divisible by 32 and N "
            f"divisible by 16, got M={m}, N={n}, K={k}"
        )
    if out is None:
        out = A.new_empty((m, n), dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or out.stride(-1) != 1
    ):
        raise ValueError(
            f"out must be row-major GPU {out_dtype} ({m}, {n}) colocated with A"
        )

    tile, cluster = cluster_mm_config(m, n, k)
    launch_cluster_mm(A, B, A_scales, B_scales, out, tile=tile, cluster=cluster)
    return out


__all__ = ["launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250"]
