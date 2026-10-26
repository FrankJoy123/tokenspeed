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

"""Packed-row views for V4.1 model-wide query parallel prefill."""

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.query_shard import QueryShardPlan


def projection_mapping(mapping: Mapping) -> Mapping:
    """Query owners hold every attention head; expert ownership stays global."""
    if not mapping.attn.has_qcp:
        return mapping
    if mapping.attn.head_tp_size != 1 or mapping.has_pp:
        raise ValueError("V4.1 query shards currently require PP1 and replicated heads")
    return Mapping(
        rank=mapping.rank,
        world_size=mapping.world_size,
        attn_tp_size=1,
        attn_dp_size=mapping.attn.world_size,
        dense_tp_size=1,
        moe_tp_size=mapping.moe.tp_size,
        moe_ep_size=mapping.moe.ep_size,
        moe_dp_size=mapping.moe.dp_size,
        pp_size=1,
    )


def request_query_rows(
    plan: QueryShardPlan, source_rows: slice
) -> tuple[slice, slice] | None:
    """Local output rows and their indices into one request's full SWA plan."""
    start = max(plan.local_start, source_rows.start)
    end = min(plan.local_end, source_rows.stop)
    if start >= end:
        return None
    return (
        slice(start - plan.local_start, end - plan.local_start),
        slice(start - source_rows.start, end - source_rows.start),
    )
