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

"""The SM120 local adapter keeps masked routing and buffer ownership correct."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.thirdparty.flashinfer.sm120_cutlass_local import (
    Sm120CutlassLocalMoe,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Qualified SM120 expert adapter",
)


def _decode(w, s):
    codes = torch.stack((w & 15, w >> 4), dim=-1).flatten(-2)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    value = lut[(codes & 7).long()]
    value = torch.where((codes & 8) != 0, -value, value)
    scale = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.int() - 127)
    return value * scale.repeat_interleave(32, dim=-1)


def _qdq(x):
    v = x.float().reshape(len(x), -1, 32)
    sf = torch.pow(2, torch.ceil(torch.log2(v.abs().amax(-1).clamp_min(1.0e-30) / 448)))
    return (
        (v / sf[..., None]).to(torch.float8_e4m3fn).float() * sf[..., None]
    ).reshape_as(x)


def test_sm120_local_masked_routes_and_retained_output():
    pytest.importorskip("flashinfer")
    torch.manual_seed(71)
    device = "cuda"
    w13 = torch.randint(0, 256, (2, 256, 64), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (2, 128, 64), device=device, dtype=torch.uint8)
    s13 = torch.full((2, 256, 4), 122, device=device, dtype=torch.uint8)
    s2 = torch.full((2, 128, 4), 122, device=device, dtype=torch.uint8)
    state = Sm120CutlassLocalMoe(
        w13=w13, w2=w2, s13=s13, s2=s2, clamp=10, hidden=128, intermediate=128, top_k=2
    )
    x = torch.randn(17, 128, device=device, dtype=torch.bfloat16) * 0.25
    ids = torch.tensor([[0, 1], [0, -1], [-1, 1], [-1, -1]], device=device).repeat(
        5, 1
    )[:17]
    weights = torch.full((17, 2), 0.5, device=device)
    # Runtime warmup nests local experts in a process-wide tuning context.
    # This must still use the same numerical path as normal inference.
    from flashinfer.autotuner import autotune

    with autotune(True):
        y = state.run(x, ids, weights, enable_pdl=False)
    retained = y.clone()
    reference = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(2):
        rows, routes = torch.where(ids == slot)
        z = (_qdq(x[rows]) @ _decode(w13[slot], s13[slot]).T).to(torch.bfloat16).float()
        gate = z[:, :128].clamp(max=10)
        up = z[:, 128:].clamp(-10, 10)
        h = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        z = (_qdq(h) @ _decode(w2[slot], s2[slot]).T) * weights[rows, routes, None]
        reference.index_add_(0, rows, z)
    delta = y.float() - reference
    assert float(delta.norm() / reference.norm()) < 0.03
    assert torch.count_nonzero(y[3]) == 0
    zero = state.run(torch.zeros_like(x), ids, weights, enable_pdl=False)
    assert torch.count_nonzero(zero) == 0
    torch.testing.assert_close(y, retained, rtol=0, atol=0)
    empty = state.run(x[:0], ids[:0], weights[:0], enable_pdl=False)
    assert empty.shape == (0, 128) and empty.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="routing shape"):
        state.run(x, ids[:, :1], weights, enable_pdl=False)


def _ragged_autotune_worker(rank, rendezvous):
    from flashinfer.autotuner import autotune, set_autotune_process_group

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=25),
    )
    try:
        torch.manual_seed(71)
        state = Sm120CutlassLocalMoe(
            w13=torch.randint(0, 256, (2, 256, 64), device="cuda", dtype=torch.uint8),
            w2=torch.randint(0, 256, (2, 128, 64), device="cuda", dtype=torch.uint8),
            s13=torch.full((2, 256, 4), 122, device="cuda", dtype=torch.uint8),
            s2=torch.full((2, 128, 4), 122, device="cuda", dtype=torch.uint8),
            clamp=10,
            hidden=128,
            intermediate=128,
            top_k=2,
        )
        rows = 0 if rank == 0 else 17
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) * 0.25
        ids = torch.zeros((rows, 2), device="cuda", dtype=torch.int64)
        ids[:, 1] = 1
        weights = torch.full((rows, 2), 0.5, device="cuda")
        dist.barrier()
        set_autotune_process_group(dist.group.WORLD)
        with autotune(True):
            output = state.run(x, ids, weights, enable_pdl=False)
            torch.cuda.synchronize()
            marker = torch.tensor([rank], dtype=torch.int32)
            dist.all_reduce(marker)
        assert marker.item() == 6 and output.shape == x.shape
        assert torch.isfinite(output).all()
        restored = state.run(x, ids, weights, enable_pdl=False)
        torch.testing.assert_close(output, restored, rtol=0, atol=0)
    finally:
        set_autotune_process_group(None)
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Four SM120 GPUs required")
def test_sm120_local_empty_peer_in_distributed_autotune(tmp_path):
    pytest.importorskip("flashinfer")
    mp.spawn(
        _ragged_autotune_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=4,
        join=True,
    )
