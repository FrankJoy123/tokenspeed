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

"""Official SM120 CUTLASS MXFP8 entry point, separate from the SM100 adapter."""

from __future__ import annotations

import torch


def has_cutlass_mxfp8_sm120() -> bool:
    """Return whether the optional FlashInfer build exposes the SM120 generator."""
    try:
        from flashinfer import mm_mxfp8  # noqa: F401
        from flashinfer.jit.gemm import (  # noqa: F401
            gen_gemm_sm120_module_cutlass_mxfp8,
        )
    except ImportError:
        return False
    return True


def cutlass_mxfp8_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Multiply E4M3 A[M,K] by column-major B[K,N] with swizzled UE8M0 scales.

    The caller supplies F8_128x4 scale storage. The official CUTLASS backend
    consumes the existing FP8 codes directly and returns an M-by-N output.
    """
    from flashinfer import mm_mxfp8

    return mm_mxfp8(a, b, a_scale, b_scale, out_dtype=out_dtype, backend="cutlass")
