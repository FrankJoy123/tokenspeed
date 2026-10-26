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

"""Dynamic token/destination-rank dispatch for eager text prefill.

This is a correctness baseline over existing NCCL all-to-all, not an optimized
DeepEP implementation. Split counts are derived from actual GPU routing on each
forward and synchronized to the host because torch's variable-split API takes
Python lists. No request-keyed state or scheduler lives here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.utils.host_sync import allow_host_sync


def dispatch_combine(
    hidden: torch.Tensor,
    topk: StandardTopKOutput,
    *,
    num_experts: int,
    group,
    local_experts,
) -> torch.Tensor:
    """Send one hidden row per token/destination; combine local expert sums."""
    size = dist.get_world_size(group)
    if num_experts % size:
        raise ValueError("NCCL dispatch requires contiguous equal expert ownership")
    if hidden.ndim != 2 or topk.topk_ids.shape != topk.topk_weights.shape:
        raise ValueError("NCCL dispatch input/routing shape mismatch")
    if hidden.shape[0] != topk.topk_ids.shape[0]:
        raise ValueError("NCCL dispatch routing rows differ from input rows")
    owners = topk.topk_ids // (num_experts // size)
    destinations = torch.arange(size, device=hidden.device)
    incidence = (owners[:, :, None] == destinations).any(dim=1)
    send_counts = incidence.sum(dim=0, dtype=torch.int64)
    recv_counts = torch.empty_like(send_counts)
    # Every rank enters, including a rank with zero local/CED rows.
    with torch.profiler.record_function("tokenspeed.moe.dispatch-counts"):
        dist.all_to_all_single(recv_counts, send_counts, group=group)
    # Current torch NCCL variable-split API needs host lists. This is an
    # intentional per-layer synchronization cost of the baseline.
    with allow_host_sync("dynamic eager NCCL dispatch split lists and compact indices"):
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        indices = incidence.T.nonzero(as_tuple=True)[1]
    send_hidden = hidden.index_select(0, indices).contiguous()
    send_ids = topk.topk_ids.index_select(0, indices).to(torch.int32).contiguous()
    send_weights = topk.topk_weights.index_select(0, indices).contiguous()
    n = sum(recv_splits)
    recv_hidden = hidden.new_empty((n, hidden.shape[1]))
    recv_ids = send_ids.new_empty((n, send_ids.shape[1]))
    recv_weights = send_weights.new_empty((n, send_weights.shape[1]))
    for incoming, outgoing in (
        (recv_hidden, send_hidden),
        (recv_ids, send_ids),
        (recv_weights, send_weights),
    ):
        with torch.profiler.record_function("tokenspeed.moe.dispatch-payload"):
            dist.all_to_all_single(
                incoming, outgoing, recv_splits, send_splits, group=group
            )
    with torch.profiler.record_function(
        f"tokenspeed.moe.local-experts:owned={hidden.shape[0]},received={n}"
    ):
        partial = local_experts(
            recv_hidden, StandardTopKOutput(recv_weights, recv_ids, None)
        )
    if partial.shape != recv_hidden.shape or partial.dtype != hidden.dtype:
        raise ValueError("NCCL local expert output must match received hidden rows")
    returned = hidden.new_empty((sum(send_splits), hidden.shape[1]))
    with torch.profiler.record_function("tokenspeed.moe.combine-payload"):
        dist.all_to_all_single(
            returned, partial.contiguous(), send_splits, recv_splits, group=group
        )
    result = torch.zeros_like(hidden)
    result.index_add_(0, indices, returned)
    return result


def replicated_input_moe(
    hidden: torch.Tensor,
    input_ids: torch.Tensor | None,
    *,
    group,
    local_forward,
) -> torch.Tensor:
    """Assign each replicated TP row once, then restore the replicated output.

    Contiguous balanced ownership is independent of routing. Ragged shards are
    padded only for the final equal-size all-gather, never sent to the experts.
    The owner computes both its routed sum and its complete shared expert.
    """
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    rows = hidden.shape[0]
    counts = [rows // size + (i < rows % size) for i in range(size)]
    start = sum(counts[:rank])
    with torch.profiler.record_function("tokenspeed.moe.owner-slice"):
        local_hidden = hidden.narrow(0, start, counts[rank]).contiguous()
        local_ids = (
            None if input_ids is None else input_ids.narrow(0, start, counts[rank])
        )
    output = local_forward(local_hidden, local_ids)
    if output.shape != local_hidden.shape or output.dtype != hidden.dtype:
        raise ValueError("Owned MoE output must match local input shape and dtype")
    if rows == 0:
        return output
    with torch.profiler.record_function("tokenspeed.moe.restore-tp-rows"):
        padded = hidden.new_zeros((max(counts), hidden.shape[1]))
        padded[: counts[rank]].copy_(output)
        gathered = [torch.empty_like(padded) for _ in range(size)]
        dist.all_gather(gathered, padded, group=group)
        return torch.cat([part[:n] for part, n in zip(gathered, counts)], dim=0)
