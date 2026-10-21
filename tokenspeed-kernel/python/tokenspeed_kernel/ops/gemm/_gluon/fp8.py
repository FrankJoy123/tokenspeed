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

"""Canonical gfx950 FP32 block-scale GEMM dispatch and workspace ownership."""

from __future__ import annotations

import torch
from tokenspeed_kernel.workspace import zeroed_state
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
    gluon_mm_mxfp8_decode_num_counters_gfx950 as _mm_mxfp8_decode_counters,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
    launch_gluon_mm_mxfp8_decode_gfx950 as _mm_mxfp8_decode_impl,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
    supports_gluon_mm_mxfp8_decode_gfx950 as _supports_mxfp8_decode_problem,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
    launch_gluon_mm_mxfp8_gfx950 as _mm_mxfp8_impl,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
    supports_mxfp8_gemm_shape as _supports_mxfp8_gemm_shape,
)


def _prefers_mxfp8_large_tiles(m: int, n: int, k: int) -> bool:
    # Up to 256 rows the weight-streaming decode kernel wins, except on
    # weights of 64 Mi elements or more (V4.1 25600x6144 and 5120x15360):
    # there the two tie near 48-64 rows and the large-tile kernel is up to
    # 1.6x faster from 96 rows on.
    return m > 256 or (m >= 64 and n * k >= 1 << 26)


def _is_mxfp8_prefill_problem(m: int, n: int, k: int) -> bool:
    return _supports_mxfp8_gemm_shape(m, n, k) and _prefers_mxfp8_large_tiles(m, n, k)


def _is_mxfp8_decode_problem(m: int, n: int, k: int) -> bool:
    return _supports_mxfp8_decode_problem(m, n, k) and not _prefers_mxfp8_large_tiles(
        m, n, k
    )


def _is_fp32_blockscale_decode_problem(m: int, n: int, k: int) -> bool:
    # Measured FP32 decode wins cover short/moderate reductions; longer
    # reductions retain existing GEMM selection, not a decoder limitation.
    return k <= 3072 and _is_mxfp8_decode_problem(m, n, k)


def _is_fp32_blockscale_problem(m: int, n: int, k: int) -> bool:
    return _supports_mxfp8_gemm_shape(m, n, k) or _is_fp32_blockscale_decode_problem(
        m, n, k
    )


def gluon_mm_fp8_blockscale_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None,
    block_size: list[int] | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiply canonical E4M3 matrices with exact FP32 K128 block scales.

    Activation scales have shape ``[M, ceil(K/128)]`` and weight scales
    ``[ceil(N/128), ceil(K/128)]``. Scale strides are unrestricted; values
    and the optional BF16 output require unit inner stride. Shape-based
    dispatch selects streaming decode or the general matrix launcher;
    native E8M0 registration remains separate.
    """
    if (
        A_scales is None
        or B_scales is None
        or block_size is None
        or list(block_size) != [128, 128]
    ):
        raise ValueError(
            "gfx950 FP32 block-scale GEMM requires both scales and block_size=[128, 128]"
        )
    if _is_fp32_blockscale_decode_problem(A.shape[0], B.shape[0], A.shape[1]):
        counters = zeroed_state(
            _mm_mxfp8_decode_counters(A.shape[0], B.shape[0], A.shape[1]),
            dtype=torch.int32,
            device=A.device,
        )
        return _mm_mxfp8_decode_impl(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            counters=counters,
            out=out,
        )
    return _mm_mxfp8_impl(
        A,
        B,
        A_scales,
        B_scales,
        out_dtype,
        alpha=alpha,
        block_size=block_size,
        out=out,
    )
