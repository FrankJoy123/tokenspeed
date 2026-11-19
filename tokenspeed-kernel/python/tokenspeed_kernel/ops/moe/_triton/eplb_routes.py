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

"""Map static EPLB routes and count the real rows in one GPU launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _map_and_record(
    Logical,
    RankMap,
    Live,
    Counts,
    Physical,
    rows,
    TOPK: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // TOPK
    slot = offsets % TOPK
    in_bounds = row < rows
    logical = tl.load(Logical + row * ROW_STRIDE + slot * SLOT_STRIDE, in_bounds, 0)
    is_expert = in_bounds & (logical >= 0) & (logical < EXPERTS)
    mapped = tl.load(RankMap + logical, is_expert, -1)
    physical = tl.where(is_expert, mapped, logical)
    tl.store(Physical + offsets, physical, in_bounds)
    live = tl.load(Live + row, in_bounds, 0)
    tl.atomic_add(
        Counts + physical, 1, is_expert & live & (physical >= 0), sem="relaxed"
    )


def map_and_record_static_routes(
    logical: torch.Tensor,
    rank_map: torch.Tensor,
    live: torch.Tensor,
    physical_count: torch.Tensor,
) -> torch.Tensor:
    """Map logical Top-K to int64 physical IDs and count live routes in place."""
    rows, top_k = logical.shape
    physical = torch.empty(logical.shape, dtype=torch.int64, device=logical.device)
    if rows:
        block = 256
        _map_and_record[(triton.cdiv(rows * top_k, block),)](
            logical,
            rank_map,
            live,
            physical_count,
            physical,
            rows,
            top_k,
            logical.stride(0),
            logical.stride(1),
            rank_map.numel(),
            block,
            num_warps=4,
        )
    return physical
