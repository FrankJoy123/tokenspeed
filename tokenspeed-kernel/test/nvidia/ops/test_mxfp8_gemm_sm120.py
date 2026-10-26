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

"""Identical-code SM120 MXFP8 numerics, dispatch and graph replay."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _online_quantize_mxfp8, mm
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.flashinfer.mxfp8_sm120 import has_cutlass_mxfp8_sm120

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not has_cutlass_mxfp8_sm120(),
    reason="SM120 and FlashInfer CUTLASS MXFP8 are required",
)


@pytest.fixture(autouse=True)
def disable_pdl():
    previous = pdl_enabled()
    pdl_enabled(False)
    yield
    pdl_enabled(previous)


def _operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    a, sa = _online_quantize_mxfp8(
        x, [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
    )
    b = (torch.randn(n, k, device="cuda", generator=gen) * 64).to(torch.float8_e4m3fn)
    # The V4 checkpoint has shared 32x32 exponents, expanded losslessly per row.
    sb = torch.randint(
        115,
        120,
        ((n + 31) // 32, k // 32),
        device="cuda",
        dtype=torch.uint8,
        generator=gen,
    )
    sb = sb.repeat_interleave(32, dim=0)[:n].contiguous()
    return a, b, sa, sb


def _run(args, *, override=None, **kwargs):
    a, b, sa, sb = args
    return mm(
        a,
        b,
        A_scales=sa,
        B_scales=sb,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
        override=override,
        **kwargs,
    )


def _reference(args):
    a, b, sa, sb = args
    # Independent FP64 dot, all output entries, no TF32 or shared GEMM kernel.
    da = a.double() * torch.exp2(sa.double() - 127).repeat_interleave(32, dim=1)
    db = b.double() * torch.exp2(sb.double() - 127).repeat_interleave(32, dim=1)
    return da @ db.T


@pytest.mark.parametrize(
    "m,n,k", [(1024, 1024, 1024), (1025, 1056, 1280), (1024, 1856, 5120)]
)
def test_identical_codes_match_fp64(m, n, k):
    args = _operands(m, n, k, m + n)
    ref = _reference(args)
    for override in (None, "flashinfer_mm_mxfp8_sm120", "triton_mm_fp8_blockscale"):
        got = _run(args, override=override)
        # BF16 rounding plus FP32 accumulation; the absolute floor covers cancellation.
        torch.testing.assert_close(got.double(), ref, rtol=2**-7, atol=2e-4)


@pytest.mark.parametrize("transpose", [False, True])
def test_square_weight_orientation_and_out_bias(transpose):
    args = list(_operands(1024, 1024, 1024, 87))
    ref = _reference(args)
    if transpose:
        args[1] = args[1].T
    unbiased = _run(args, override="flashinfer_mm_mxfp8_sm120")
    torch.testing.assert_close(unbiased.double(), ref, rtol=2**-7, atol=2e-4)
    backing = torch.empty(1024, 1040, device="cuda", dtype=torch.bfloat16)
    out = backing[:, :1024]
    bias = torch.linspace(-1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    got = _run(args, override="flashinfer_mm_mxfp8_sm120", out=out, bias=bias)
    assert got.data_ptr() == out.data_ptr()
    # Test bias/out semantics exactly, separately from accumulation error:
    # cancellation after adding bias can magnify a one-ULP pre-bias difference.
    expected = unbiased + bias
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_graph_replay_refreshes_codes_and_scales():
    args = _operands(1024, 1056, 1024, 42)
    expected = _run(args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _run(args)
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for seed in (43, 44):
        for dst, src in zip(args, _operands(1024, 1056, 1024, seed), strict=True):
            dst.copy_(src)
        graph.replay()
        torch.testing.assert_close(
            output.double(), _reference(args), rtol=2**-7, atol=2e-4
        )


@pytest.mark.parametrize("m,expected", [(128, False), (1024, True)])
def test_auto_dispatch_uses_measured_large_m_range(m, expected, tmp_path):
    import json

    from tokenspeed_kernel.profiling import ShapeCapture

    args = _operands(m, 1856, 5120, 11)
    capture = ShapeCapture.get()
    previous = capture.enabled
    capture.clear()
    capture.enabled = True
    try:
        got = _run(args)
        capture.dump(tmp_path / "shapes.json")
    finally:
        capture.enabled = previous
        capture.clear()
    names = {
        r["kernel_name"]
        for r in json.loads((tmp_path / "shapes.json").read_text())
        if r["family"] == "gemm" and r["mode"] == "mm"
    }
    assert ("flashinfer_mm_mxfp8_sm120" in names) == expected
    torch.testing.assert_close(got.double(), _reference(args), rtol=2**-7, atol=2e-4)


def test_online_quantization_preserves_codes():
    args = list(_operands(1024, 1056, 1024, 93))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    args[0], args[2] = _online_quantize_mxfp8(
        x, [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
    )
    got = mm(
        x,
        args[1],
        B_scales=args[3],
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
    )
    torch.testing.assert_close(got, _run(args), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["scale_dtype", "scale_shape", "alpha"])
def test_forced_kernel_rejects_unsupported_contract(invalid):
    args = list(_operands(1024, 1024, 1024, 23))
    extra = {}
    if invalid == "scale_dtype":
        args[3] = args[3].float()
    elif invalid == "scale_shape":
        args[3] = args[3][:-1]
    else:
        extra["alpha"] = torch.tensor(1.0, device="cuda")
    with pytest.raises(ValueError):
        _run(args, override="flashinfer_mm_mxfp8_sm120", **extra)
