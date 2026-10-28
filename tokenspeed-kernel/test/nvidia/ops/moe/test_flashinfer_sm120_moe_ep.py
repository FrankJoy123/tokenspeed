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

"""Four-rank correctness and replay checks for the optional SM120 library adapter."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.flashinfer.sm120_cake_moe_ep import Sm120CakeMoeEp


def _quant_dequant(x: torch.Tensor) -> torch.Tensor:
    blocks = x.float().reshape(x.shape[0], x.shape[1] // 32, 32)
    scale = 2.0 ** torch.ceil(
        torch.log2(blocks.abs().amax(-1).clamp_min(1e-20) / 448.0)
    )
    q = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.float() * scale[..., None]).reshape_as(x)


def _gather_bytes(x: torch.Tensor) -> torch.Tensor:
    raw = x.view(torch.uint8)
    parts = [torch.empty_like(raw) for _ in range(4)]
    dist.all_gather(parts, raw)
    return torch.cat(parts).view(x.dtype)


def _unpack(x: torch.Tensor) -> torch.Tensor:
    raw = x.view(torch.uint8)
    return torch.stack(
        ((raw & 15).float() / 2, (raw >> 4).float() / 2), dim=-1
    ).flatten(-2)


def test_sm120_ep4_numerics_ragged_rows_and_replay():
    """An empty source peer participates; replay uses the exact eager result."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("launch four ranks with torchrun")
    # Exercise registration imports as well as the isolated library adapter.
    from tokenspeed_kernel.ops.moe.flashinfer_cake_sm120 import (
        flashinfer_cake_sm120_moe_ep_apply,
    )

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    g = torch.Generator(device="cuda").manual_seed(4721 + rank)
    h = i = 1024
    e = 8
    k = 4

    def packed(shape):
        lo = torch.randint(0, 2, shape, generator=g, device="cuda", dtype=torch.uint8)
        hi = torch.randint(0, 2, shape, generator=g, device="cuda", dtype=torch.uint8)
        return (lo | (hi << 4)).view(torch.float4_e2m1fn_x2)

    w13 = packed((2, 2 * i, h // 2))
    w2 = packed((2, h, i // 2))
    s13 = torch.full((2, 2 * i, h // 32), 127, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    s2 = torch.full((2, h, i // 32), 127, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    state = Sm120CakeMoeEp(
        w13=w13,
        w2=w2,
        w13_scale=s13,
        w2_scale=s2,
        group=dist.group.WORLD,
        capacity=67,
        compile_buckets=(67,),
        hidden=h,
        intermediate=i,
        experts=e,
        top_k=k,
        clamp=0,
        fast_math=True,
    )
    try:
        state.prepare()
        rows = (67, 64, 19, 0)[rank]
        x = (torch.randn((rows, h), generator=g, device="cuda") * 0.125).to(
            torch.bfloat16
        )
        ids = (
            torch.arange(rows, device="cuda")[:, None]
            + torch.arange(k, device="cuda")[None, :]
        ) % e
        weights = torch.full((rows, k), 1.0 / k, device="cuda", dtype=torch.float32)
        actual = flashinfer_cake_sm120_moe_ep_apply(
            {"flashinfer_cake_sm120_moe_ep": state},
            x,
            None,
            None,
            weights,
            ids,
            150,
            67,
            True,
            False,
            False,
            None,
        ).clone()
        all13 = _unpack(_gather_bytes(w13))
        all2 = _unpack(_gather_bytes(w2))
        activation = _quant_dequant(x)
        terms = torch.zeros((rows, k, h), device="cuda", dtype=torch.float32)
        for expert in range(e):
            row, slot = torch.where(ids == expert)
            if row.numel() == 0:
                continue
            gate, up = (
                (activation[row] @ all13[expert].T)
                .to(torch.bfloat16)
                .float()
                .chunk(2, dim=-1)
            )
            middle = torch.nn.functional.silu(gate) * up * weights[row, slot, None]
            terms[row, slot] = (
                (_quant_dequant(middle) @ all2[expert].T).to(torch.bfloat16).float()
            )
        reference = terms.sum(1).to(torch.bfloat16).float()
        rel = (actual.float() - reference).norm() / reference.norm().clamp_min(1e-6)
        assert rel.item() < 0.01
        for invalid in (0, 68):
            with pytest.raises(ValueError, match="capacity"):
                state.run(x, ids, weights, maximum=invalid)
        state.prepare()
        torch.testing.assert_close(
            actual, state.run(x, ids, weights, maximum=67), atol=0, rtol=0
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = state.run(x, ids, weights, maximum=67)
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, output, atol=0, rtol=0)
        dist.barrier()
    finally:
        state.close()
        dist.destroy_process_group()
