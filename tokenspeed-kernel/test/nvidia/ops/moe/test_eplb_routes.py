# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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
from tokenspeed_kernel.ops.moe._triton.eplb_routes import map_and_record_static_routes


@pytest.mark.parametrize("rows", [0, 24, 48, 96, 192, 2048, 4224])
def test_static_eplb_routes_and_counts(rows):
    torch.manual_seed(42)
    rank_map = torch.randperm(416, dtype=torch.int32, device="cuda")[:384]
    logical = torch.randint(0, 384, (6, rows), dtype=torch.int32, device="cuda").t()
    live = torch.arange(rows, device="cuda") % 3 != 1
    physical_count = torch.full((416,), 17, dtype=torch.int64, device="cuda")
    actual = map_and_record_static_routes(logical, rank_map, live, physical_count)
    expected = rank_map[logical].long()
    assert torch.equal(actual, expected)
    expected_count = torch.bincount(expected[live].reshape(-1), minlength=416)
    assert torch.equal(physical_count, expected_count + 17)


def test_static_eplb_uses_updated_maps_in_graph_replay():
    ids = torch.tensor([[0, 2], [1, -1]], dtype=torch.int32, device="cuda")
    rank_map = torch.tensor([2, 3, 1], dtype=torch.int32, device="cuda")
    live = torch.tensor([True, False], device="cuda")
    counts = torch.zeros(4, dtype=torch.int64, device="cuda")
    map_and_record_static_routes(ids, rank_map, live, counts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = map_and_record_static_routes(ids, rank_map, live, counts)
    rank_map.copy_(torch.tensor([3, 2, 0], device="cuda"))
    live.logical_not_()
    counts.zero_()
    graph.replay()
    assert actual.tolist() == [[3, 0], [2, -1]]
    assert counts.tolist() == [0, 0, 1, 0]
