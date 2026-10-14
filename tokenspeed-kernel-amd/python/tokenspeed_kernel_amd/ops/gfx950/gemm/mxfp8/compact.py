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

"""Compact canonical FP8 tiles with ordered FP32 K128 scale arithmetic."""

from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton


def _compact_launch_metadata(grid, kernel, args):
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        "bytes": (m + n) * k
        + m * n * args["SPLITS"] * args["c_ptr"].element_size()
        + 4 * (m + triton.cdiv(n, 128)) * triton.cdiv(k, 128),
    }


@gluon.jit(
    launch_metadata=_compact_launch_metadata,
    do_not_specialize=("M", "SPLITS", "stride_cs"),
)
def gluon_mm_fp8_blockscale_compact_gfx950(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    SPLITS,
    stride_cs,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WAVES: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
):
    """Accumulate each unscaled K128 partial before applying its FP32 scales."""
    gl.static_assert(WARPS_M * WARPS_N == WAVES)
    pid = gl.program_id(0)
    split = gl.program_id(1)
    pid_m = pid // gl.cdiv(N, BLOCK_N)
    pid_n = pid % gl.cdiv(N, BLOCK_N)
    loads: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [WAVES, 1], [1, 0])
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 32)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 32)
    am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, loads))
    bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, loads))
    kk = gl.arange(0, 128, gl.SliceLayout(0, loads))
    cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    groups: gl.constexpr = gl.cdiv(K, 128)
    lo = split * groups // SPLITS
    hi = (split + 1) * groups // SPLITS
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    for group in tl.range(lo, hi, num_stages=1):
        ak = group * 128 + kk
        a = gl.load(
            a_ptr + am[:, None] * stride_am + ak[None, :] * stride_ak,
            mask=(am[:, None] < M) & (ak[None, :] < K),
            other=0.0,
        )
        b = gl.load(
            b_ptr + bn[:, None] * stride_bn + ak[None, :] * stride_bk,
            mask=(bn[:, None] < N) & (ak[None, :] < K),
            other=0.0,
        )
        a = gl.convert_layout(a, dot_a)
        b = gl.convert_layout(b.trans(), dot_b)
        partial = gl.amd.cdna4.mfma_scaled(
            a,
            None,
            "e4m3",
            b,
            None,
            "e4m3",
            gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma),
        )
        a_scale = gl.load(
            as_ptr + cm * stride_asm + group * stride_ask, mask=cm < M, other=1
        )
        b_scale = gl.load(
            bs_ptr + (cn // 128) * stride_bsn + group * stride_bsk, mask=cn < N, other=1
        )
        # Keep the two scale operations ordered; their product may overflow or
        # underflow even when this scaled contribution is representable.
        accumulator = gl.fma(partial * a_scale[:, None], b_scale[None, :], accumulator)
    output = (
        c_ptr + split * stride_cs + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    )
    gl.store(
        output,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(cm[:, None] < M) & (cn[None, :] < N),
    )
