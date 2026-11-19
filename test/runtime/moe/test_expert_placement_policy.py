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


import pytest
import torch

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.placement_maps import compute_placement_maps


@pytest.mark.parametrize("gpus,extra_per_gpu", [(4, 1), (4, 2), (8, 1), (8, 0)])
def test_hot_local_keeps_every_home_and_adds_replica_to_each_gpu(gpus, extra_per_gpu):
    experts = 24
    counts = torch.arange(experts, dtype=torch.int64).repeat(2, 1)
    counts[1] = counts[1].flip(0)
    capacity = experts + gpus * extra_per_gpu
    common = dict(ep_size=gpus, num_groups=None, num_nodes=1)
    home, _ = compute_placement_maps(
        counts, num_physical_experts=experts, algorithm=EplbAlgorithm.deepseek, **common
    )
    physical, inverse = compute_placement_maps(
        counts,
        num_physical_experts=capacity,
        algorithm=EplbAlgorithm.hot_local,
        **common
    )
    by_gpu = physical.reshape(2, gpus, -1)
    base = experts // gpus
    assert torch.equal(by_gpu[..., :base].flatten(1), home)
    expected_hot = counts.argsort(dim=-1, descending=True)[:, :extra_per_gpu]
    assert torch.equal(
        by_gpu[..., base:].long(), expected_hot[:, None].expand(-1, gpus, -1)
    )
    for layer in range(2):
        for expert in range(experts):
            slot_ids = inverse[layer, expert]
            assert torch.equal(
                slot_ids[slot_ids >= 0].long().sort().values,
                (physical[layer] == expert).nonzero().flatten(),
            )
