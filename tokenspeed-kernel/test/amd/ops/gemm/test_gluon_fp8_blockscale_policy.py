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

from __future__ import annotations

"""Behavioral CPU checks for the shape-derived FP32 launch policy."""
import importlib

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _gemm_format_signature
from tokenspeed_kernel.selection import select_kernel

mm = importlib.import_module("tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm")


@pytest.mark.parametrize(
    "shape,expected,hint",
    [
        ((1, 1024, 4096), (True, 64, 64, 4, 8), 0),
        ((64, 1024, 4096), (True, 64, 64, 4, 8), 2),
        ((65, 1024, 4096), (True, 64, 64, 4, 6), 2),
        ((128, 1024, 4096), (True, 64, 64, 4, 8), 2),
        ((129, 4096, 512), (False, 32, 32, 4, 1), 2),
        ((16, 4096, 1536), (False, 16, 64, 4, 4), 2),
        ((896, 6144, 4096), (True, 128, 128, 8, 1), 2),
        ((1616, 1024, 4096), (True, 128, 128, 8, 2), 2),
        ((1616, 2048, 4096), (True, 128, 128, 8, 1), 2),
        ((3536, 4096, 512), (True, 256, 256, 8, 1), 2),
    ],
)
def test_fp32_shape_policy_boundaries(shape, expected, hint):
    config = mm._fp32_launch_config(*shape)
    assert tuple(config) == expected
    assert mm._fp32_pipeline_waves_per_eu(*shape, config) == hint


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 16, 288),
        (37, 208, 800),
        (129, 144, 1056),
        (641, 1152, 576),
        (641, 1152, 5120),
    ],
)
def test_fp32_ragged_partitions_have_work_and_legal_wave_tiles(m, n, k):
    config = mm._fp32_launch_config(m, n, k)
    if config.pipeline:
        assert ((k + 255) // 256) // config.splits >= 2
    else:
        assert ((k + 127) // 128) // config.splits >= 1
        wm, wn = mm._compact_warp_shape(config.block_m, config.block_n, config.waves)
        assert wm * wn == config.waves
        assert 16 * wm <= config.block_m and 16 * wn <= config.block_n


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((128, 1024, 4096), (4, 64)),
        ((129, 1024, 4096), (16, 256)),
        ((128, 256, 4096), (16, 256)),
        ((128, 1024, 576), (16, 256)),
    ],
)
def test_reducer_family_boundaries(shape, expected):
    assert mm._fp32_reducer_tile(*shape) == expected


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize(
    "shape,output_dtype,expected",
    [
        ((1, 1024, 4096), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((17, 130, 3072), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((17, 128, 3104), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((129, 4096, 512), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((896, 6144, 4096), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((37, 208, 800), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((4, 128, 256), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((4, 130, 256), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((256, 130, 256), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((257, 4096, 512), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 15, 128), torch.bfloat16, "triton_mm_fp8_blockscale"),
        ((1, 16, 128), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 16, 96), torch.bfloat16, "triton_mm_fp8_blockscale"),
        ((63, 32768, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((64, 32768, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 1048560, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 1048576, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((4, 128, 512), torch.float16, "triton_mm_fp8_blockscale"),
    ],
)
def test_registry_fp32_contract_and_fallback(shape, output_dtype, expected, online):
    m, n, k = shape
    a = torch.empty(
        (m, k), device="meta", dtype=torch.bfloat16 if online else torch.float8_e4m3fn
    )
    b = torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn)
    sa = None if online else torch.empty((m, (k + 127) // 128), device="meta")
    sb = torch.empty(((n + 127) // 128, (k + 127) // 128), device="meta")
    signature = _gemm_format_signature(a, b, sa, sb, output_dtype, "mxfp8", [128, 128])
    selected = select_kernel(
        "gemm",
        "mm",
        signature,
        traits={
            "m": m,
            "n": n,
            "k": k,
            "b_layout": "NK",
            "a_inner_stride_one": True,
            "a_scales_inner_stride_one": True,
            "b_inner_stride_one": True,
            "b_scales_inner_stride_one": True,
            "block_scale_layout": "canonical",
            "out_dtype": output_dtype,
            "out_inner_stride_one": True,
            "pdl_enabled": False,
        },
    )
    assert selected.name == expected


@pytest.mark.parametrize(
    "trait,expected",
    [
        ("a_inner_stride_one", "triton_mm_fp8_blockscale"),
        ("b_inner_stride_one", "triton_mm_fp8_blockscale"),
        ("out_inner_stride_one", "triton_mm_fp8_blockscale"),
        ("a_scales_inner_stride_one", "gluon_mm_fp8_blockscale_gfx950"),
        ("b_scales_inner_stride_one", "gluon_mm_fp8_blockscale_gfx950"),
    ],
)
def test_registry_value_and_scale_stride_boundaries(trait, expected):
    a = torch.empty((4, 512), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((128, 512), device="meta", dtype=torch.float8_e4m3fn)
    sa = torch.empty((4, 4), device="meta")
    sb = torch.empty((1, 4), device="meta")
    signature = _gemm_format_signature(
        a, b, sa, sb, torch.bfloat16, "mxfp8", [128, 128]
    )
    traits = {
        "m": 4,
        "n": 128,
        "k": 512,
        "b_layout": "NK",
        "a_inner_stride_one": True,
        "b_inner_stride_one": True,
        "a_scales_inner_stride_one": True,
        "b_scales_inner_stride_one": True,
        "block_scale_layout": "canonical",
        "out_dtype": torch.bfloat16,
        "out_inner_stride_one": True,
        "pdl_enabled": False,
    }
    traits[trait] = False
    assert select_kernel("gemm", "mm", signature, traits=traits).name == expected
