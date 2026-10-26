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


"""Check V4.1 query/head exchange against real selected attention on CUDA."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from tokenspeed_kernel.ops.attention import dsv41

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.prefill_head_exchange import (
        selected_prefill_with_head_exchange,
    )

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=False)
        set_global_backend(backend)
        heads = 64 // degree
        for tokens in (degree, degree + 1, 128, 1024, 1025):
            generator = torch.Generator().manual_seed(1700 + tokens)
            full_q = torch.randn(
                tokens, 64, 512, generator=generator, dtype=torch.bfloat16
            )
            q = full_q[:, rank * heads : (rank + 1) * heads].contiguous().to(device)
            workspace = torch.randn(
                2048, 1, 512, generator=generator, dtype=torch.bfloat16
            ).to(device)
            indices = torch.randint(
                0, 2048, (tokens, 128), generator=generator, dtype=torch.int32
            )
            indices[:, -3:] = -1
            indices = indices.to(device)
            sink = torch.full((64,), -torch.inf, device=device)
            sink[:heads] = torch.linspace(-1, 1, 64, device=device)[
                rank * heads : (rank + 1) * heads
            ]
            expected = dsv41.selected_attention(
                q,
                None,
                None,
                None,
                None,
                None,
                None,
                sink,
                512**-0.5,
                None,
                256,
                None,
                workspace,
                indices,
                lse=None,
            )
            actual = selected_prefill_with_head_exchange(
                q,
                workspace,
                indices,
                sink,
                softmax_scale=512**-0.5,
                query_chunk_size=256,
                group=group,
                rank=rank,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 8])
def test_selected_attention_after_real_head_exchange(tmp_path, degree):
    if not torch.cuda.is_available() or torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    mp.spawn(
        _worker,
        args=(degree, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )
