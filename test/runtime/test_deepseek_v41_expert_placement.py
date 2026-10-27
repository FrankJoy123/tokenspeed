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

import pytest
import torch

from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.models.deepseek_v41_moe import (
    _map_precomputed_expert_routes,
    _precomputed_expert_dispatch_info,
)
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata


def _placement(rows, rank=0):
    return ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor(rows), 4, ep_size=2, ep_rank=rank, ep_rank_nodes=(0, 0)
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_static_precomputed_routes_keep_the_same_weighted_experts(dtype):
    maps = [[3, 1, 0, 2], [2, 0, 3, 1]]
    placement = _placement(maps)
    ids = torch.tensor([[0, 3], [2, 1]], dtype=dtype)
    weights = torch.tensor([[0.3, 1.2], [0.7, 0.8]])
    x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    logical_weights = torch.arange(36, dtype=torch.float32).reshape(4, 3, 3) / 16
    logits = torch.zeros((2, 4))
    output = StandardTopKOutput(weights, ids, logits, 1.5)
    for layer, row in enumerate(maps):
        info = _precomputed_expert_dispatch_info(
            placement,
            layer_id=layer,
            num_experts=4,
            ep_size=2,
            backend="deepep",
            algorithm="static",
        )
        mapped = _map_precomputed_expert_routes(output, info)
        physical_weights = logical_weights[torch.tensor(row)]
        expected = sum(
            torch.bmm(logical_weights[ids[:, k].long()], x[..., None]).squeeze(-1)
            * weights[:, k, None]
            for k in range(2)
        )
        actual = sum(
            torch.bmm(
                physical_weights[mapped.topk_ids[:, k].long()], x[..., None]
            ).squeeze(-1)
            * mapped.topk_weights[:, k, None]
            for k in range(2)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert mapped.topk_weights is weights and mapped.router_logits is logits
        assert mapped.output_scale == 1.5 and mapped.topk_ids.dtype == dtype
        torch.testing.assert_close(ids, torch.tensor([[0, 3], [2, 1]], dtype=dtype))


def test_static_route_mapping_handles_idle_rows_and_masked_routes():
    placement = _placement([[3, 1, 0, 2]])
    info = _precomputed_expert_dispatch_info(
        placement,
        layer_id=0,
        num_experts=4,
        ep_size=2,
        backend="deepep",
        algorithm="static_with_zero_expert",
    )
    for ids in (
        torch.empty((0, 2), dtype=torch.int32),
        torch.tensor([[-1, 0], [4, 2]], dtype=torch.int32),
    ):
        weights = torch.ones(ids.shape)
        output = _map_precomputed_expert_routes(
            StandardTopKOutput(weights, ids, None), info
        )
        assert output.topk_ids.shape == ids.shape
        if ids.numel():
            torch.testing.assert_close(
                output.topk_ids, torch.tensor([[-1, 2], [4, 3]], dtype=torch.int32)
            )


def test_no_placement_keeps_the_original_route_object():
    output = StandardTopKOutput(
        torch.ones((2, 1)), torch.zeros((2, 1), dtype=torch.int64), None
    )
    assert _map_precomputed_expert_routes(output, None) is output


@pytest.mark.parametrize(
    "backend,algorithm", [("nccl", "static"), ("deepep", "dynamic")]
)
def test_unverified_dispatch_modes_are_rejected(backend, algorithm):
    with pytest.raises(ValueError, match="requires"):
        _precomputed_expert_dispatch_info(
            _placement([[3, 1, 0, 2]]),
            layer_id=0,
            num_experts=4,
            ep_size=2,
            backend=backend,
            algorithm=algorithm,
        )


def test_static_v41_admission_does_not_silently_enable_replicas():
    placement = _placement([[0, 1, 2, 3, 0, 1]])
    with pytest.raises(ValueError, match="one slot"):
        _precomputed_expert_dispatch_info(
            placement,
            layer_id=0,
            num_experts=4,
            ep_size=2,
            backend="deepep",
            algorithm="static",
        )
