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

"""Real GPU equivalence for SM120 indexer launch geometry."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 indexer",
)


@pytest.mark.parametrize(
    "tokens,history,weight_dtype,candidate_topk",
    [
        (257, 40960, torch.bfloat16, 0),
        (256, 8192, torch.float32, 0),
        (128, 1024, torch.bfloat16, 17),
    ],
)
def test_full_scan_preserves_selected_ids_and_lengths(
    tokens, history, weight_dtype, candidate_topk
):
    torch.manual_seed(271)
    device = "cuda"
    q = torch.randn(tokens, 32, 128, device=device, dtype=torch.bfloat16)
    w = torch.randn(tokens, 32, device=device, dtype=weight_dtype)
    keys = torch.randn(history, 128, device=device, dtype=torch.bfloat16)
    pages = history // 64 + 1
    # Reverse physical pages and leave the sentinel page untouched.
    cache = torch.zeros(pages, 64, 68, device=device, dtype=torch.uint8)
    table = torch.arange(pages - 1, 0, -1, device=device, dtype=torch.int32)
    slots = table.long().repeat_interleave(64) * 64
    slots += torch.arange(history, device=device) % 64
    dsv41.cache_scatter(keys, cache, slots, "index")
    table = table[None, :].expand(tokens, -1)
    visible = torch.full((tokens,), history, device=device, dtype=torch.int32)
    visible -= torch.arange(tokens - 1, -1, -1, device=device, dtype=torch.int32)
    visible[0] = 0
    args = (
        q,
        w,
        cache,
        table,
        visible,
        None,
        512,
        candidate_topk,
        8,
        256,
        256,
        None,
        None,
    )
    reference = dsv41.index_topk(*args, solution="triton")
    for solution in ("triton_sm120", None):
        actual = dsv41.index_topk(*args, solution=solution)
        for got, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    if candidate_topk:
        reindex_args = (
            q,
            w,
            cache,
            table,
            visible,
            reference[2],
            512,
            0,
            8,
            256,
            256,
            None,
            None,
        )
        expected = dsv41.index_topk(*reindex_args, solution="triton")
        actual = dsv41.index_topk(*reindex_args, solution="triton_sm120")
        for got, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
