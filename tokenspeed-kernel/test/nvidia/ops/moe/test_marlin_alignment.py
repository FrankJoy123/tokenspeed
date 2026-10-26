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

"""Marlin alignment preserves route ownership across scan warps."""

import pytest
import torch
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    is_marlin_moe_available,
    moe_align_block_size,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_marlin_moe_available(),
    reason="CUDA Marlin library required",
)


@pytest.mark.parametrize("experts", [48, 192, 384])
@pytest.mark.parametrize("rows,block", [(128, 32), (1024, 64)])
def test_alignment_keeps_each_route_in_its_expert_block(experts, rows, block):
    """Include EP-masked routes and scan tables spanning several warps.

    Run this test under compute-sanitizer --tool racecheck as well: ordinary
    execution need not expose a missing cross-warp shared-memory dependency.
    """
    generator = torch.Generator().manual_seed(4491)
    for _ in range(8):
        global_ids = torch.rand(rows, 384, generator=generator).topk(6, dim=1).indices
        ids = torch.where(global_ids < experts, global_ids, -1).to(torch.int32)
        perm, expert_ids, padded = moe_align_block_size(ids.cuda(), block, experts)
        total = padded.item()
        counts = torch.bincount((ids.flatten() + 1).long(), minlength=experts + 1)
        expected = ((counts + block - 1) // block * block).sum().item()
        assert total == expected
        assigned = expert_ids[: total // block].cpu()
        assert bool(((assigned >= -1) & (assigned < experts)).all())
        routes = perm[:total].cpu()
        assert bool((routes >= 0).all())
        live = routes[routes < ids.numel()]
        torch.testing.assert_close(
            live.sort().values,
            torch.arange(ids.numel(), dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        for i, expert in enumerate(assigned.tolist()):
            chunk = routes[i * block : (i + 1) * block]
            indices = chunk[chunk < ids.numel()].long()
            assert bool((ids.flatten()[indices] == expert).all())
