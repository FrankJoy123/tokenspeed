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

"""Actual two-rank NCCL dispatch/combine versus replicated-input expert sums."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.layers.moe.nccl_dispatch import (
    dispatch_combine,
    replicated_input_moe,
)
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput


def _worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    device = torch.device("cuda", rank)
    for n in [0, 1, 5, 129, 1024]:
        for concentrated in [False, True]:
            g = torch.Generator().manual_seed(20261026 + n)
            hidden = torch.randn(n, 32, generator=g).to(device)
            ids = torch.arange(n, device=device)
            routes = torch.rand(n, 384, generator=g).topk(6, dim=1).indices.to(device)
            if concentrated:
                routes = (
                    routes % 192
                )  # all routes target rank 0, including empty rank 1
            weights = torch.rand(n, 6, generator=g).to(device)
            weights /= weights.sum(dim=1, keepdim=True)
            # Distinct diagonal experts make duplicate dispatch, wrong row order,
            # missing destination sums, or a repeated shared contribution observable.
            expert_scale = torch.linspace(0.1, 1.1, 384, device=device)

            def local_experts(rows, topk):
                mask = topk.topk_ids // 192 == rank
                scale = (expert_scale[topk.topk_ids] * topk.topk_weights * mask).sum(1)
                return rows * scale[:, None]

            def owned(rows, row_ids):
                topk = StandardTopKOutput(weights[row_ids], routes[row_ids], None)
                return (
                    dispatch_combine(
                        rows,
                        topk,
                        num_experts=384,
                        group=None,
                        local_experts=local_experts,
                    )
                    + rows.square() * 0.125
                )

            actual = replicated_input_moe(hidden, ids, group=None, local_forward=owned)
            topk = StandardTopKOutput(weights, routes, None)
            reference = local_experts(hidden, topk)
            dist.all_reduce(reference)
            reference += hidden.square() * 0.125
            torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_replicated_tp_rows_dispatch_once_and_restore():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
