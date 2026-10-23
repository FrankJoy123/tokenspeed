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

"""Empty local expert batches from attention-DP idle forwards."""

import pytest
import torch
from tokenspeed_kernel.ops.moe.flashinfer import (
    flashinfer_cutlass_unquant_moe_apply,
)
from torch import nn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ep_size,ep_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("precomputed", [False, True])
def test_cutlass_empty_local_batch(dtype, ep_size, ep_rank, precomputed):
    weights = nn.Module()
    weights.register_parameter(
        "w13_weight", nn.Parameter(torch.empty(4 // ep_size, 256, 128, dtype=dtype))
    )
    weights.register_parameter(
        "w2_weight", nn.Parameter(torch.empty(4 // ep_size, 128, 128, dtype=dtype))
    )
    weights.ep_size, weights.ep_rank = ep_size, ep_rank
    weights.tp_size, weights.tp_rank = 1, 0
    weights.top_k = 2
    x = torch.empty(0, 128, dtype=dtype)
    kwargs = {}
    if precomputed:
        kwargs = {
            "topk_ids": torch.empty(0, 2, dtype=torch.int32),
            "topk_weights": torch.empty(0, 2, dtype=dtype),
        }
    result = flashinfer_cutlass_unquant_moe_apply(
        {}, x, weights, torch.empty(0, 4, dtype=dtype), **kwargs
    )
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == x.device
