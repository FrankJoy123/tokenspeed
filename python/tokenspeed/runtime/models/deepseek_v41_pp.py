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

"""Lossless shared-cache and selection transport at V4.1 pipeline boundaries.

LCM owns every persistent page. The boundary contains only touched packed
pages and per-forward selection scratch; no request-keyed cache lives here.
Page-planar values/scales travel together and are never decoded/requantized.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.distributed.pp_stage import PPStageState

# The FP8 row layout is the widest supported representation: 584 KV bytes
# and 132 index bytes. FP4 uses its native narrower 288+68 row.
V41_PP_PAGE_BYTES = 584 + 132


def page_capacity(num_tokens: int, max_requests: int) -> int:
    """Bound touched 64-row pages, including each request's two edge pages."""
    return (num_tokens + 63) // 64 + 2 * max_requests


def logical_page_descriptors(spans, *, ratio: int) -> list[tuple[int, int]]:
    """Derive touched pages from prepared host spans, without a GPU readback."""
    descriptors = []
    stride = 64 * ratio
    for span in spans:
        if span.count:
            first = span.prefix // stride
            last = (span.prefix + span.count - 1) // stride
            descriptors.extend(
                (span.request, page * stride) for page in range(first, last + 1)
            )
    return descriptors


def copy_packed_pages(
    global_cache: torch.Tensor,
    index_cache: torch.Tensor,
    page_ids: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Snapshot complete strided pages into fixed-capacity wire storage."""
    if page_ids.numel() > capacity:
        raise ValueError("V4.1 PP page count exceeds its wire capacity")
    g, k = global_cache.shape[-1], index_cache.shape[-1]
    if g + k > V41_PP_PAGE_BYTES:
        raise ValueError("V4.1 PP cache format exceeds its wire width")
    out = global_cache.new_zeros((capacity, 64, g + k))
    n = page_ids.numel()
    out[:n, :, :g].copy_(global_cache.index_select(0, page_ids))
    out[:n, :, g : g + k].copy_(index_cache.index_select(0, page_ids))
    return out


def land_packed_pages(
    payload: torch.Tensor,
    global_cache: torch.Tensor,
    index_cache: torch.Tensor,
    page_ids: torch.Tensor,
    payload_rows: torch.Tensor,
) -> None:
    """Land bytes into receiver-local pages; physical IDs never cross stages."""
    g, k = global_cache.shape[-1], index_cache.shape[-1]
    incoming = payload.index_select(0, payload_rows)
    global_cache.index_copy_(0, page_ids, incoming[:, :, :g])
    index_cache.index_copy_(0, page_ids, incoming[:, :, g : g + k])


def snapshot(
    backend, *, num_tokens: int, max_requests: int, layer_end: int, mode
) -> dict[str, torch.Tensor]:
    """Snapshot touched owner pages and the latest source's logical selection."""
    meta = backend.query_metadata(mode)
    positions, requests = meta.positions, meta.request_indices
    known = [
        i
        for i, owner in enumerate(backend.spec.kv_owners)
        if i == owner and i < layer_end
    ]
    owners = tuple(known[-1:])
    if len(backend._prefill_spans) > max_requests:
        raise ValueError("V4.1 PP request count exceeds its declared budget")
    logical = (
        logical_page_descriptors(
            backend._prefill_spans, ratio=backend.spec.compress_ratios[owners[0]]
        )
        if owners
        else []
    )
    capacity = len(logical)
    if capacity > page_capacity(num_tokens, max_requests):
        raise ValueError("V4.1 PP page count exceeds the prepared input span")
    row_layout = backend.spec.row_layout()
    wire_bytes = row_layout.global_row_bytes + row_layout.index_row_bytes
    descriptors = torch.full(
        (len(owners), capacity, 2), -1, dtype=torch.int64, device=positions.device
    )
    pages = torch.zeros(
        (len(owners), capacity, 64, wire_bytes),
        dtype=torch.uint8,
        device=positions.device,
    )
    for index, owner in enumerate(owners):
        if owner >= layer_end:
            continue
        ratio = backend.spec.compress_ratios[owner]
        # Only the currently live owner crosses this boundary. A later source
        # supersedes it before any still later layer may consume the old one.
        logical = logical_page_descriptors(backend._prefill_spans, ratio=ratio)
        if len(logical) > capacity:
            raise ValueError("V4.1 PP logical pages exceed the fixed wire capacity")
        n = len(logical)
        descriptors[index, :n].copy_(
            torch.tensor(logical, dtype=torch.int64, device=positions.device).reshape(
                -1, 2
            )
        )
        req, raw = descriptors[index, :n].unbind(1)
        gid = backend._owner_group(owner)
        slots = backend.cache_slots(gid, raw, req, mode)
        torch._assert_async((slots >= 64).all(), "V4.1 PP source page is not allocated")
        pages[index].copy_(
            copy_packed_pages(
                backend.cache_pool.global_kv(owner),
                backend.cache_pool.index_k(owner),
                slots // 64,
                capacity,
            )
        )
    query_meta = (
        backend.decoder_view().metadata
        if layer_end > backend.spec.candidate_source
        else meta
    )
    positions, requests = query_meta.positions, query_meta.request_indices
    num_queries = positions.numel()
    topk = torch.full(
        (num_queries, backend.spec.index_topk),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    lengths = torch.zeros(num_queries, dtype=torch.int32, device=positions.device)
    narrowed = layer_end > backend.spec.candidate_source
    candidates = (
        torch.full(
            (num_queries, backend.spec.candidate_topk),
            -1,
            dtype=torch.int32,
            device=positions.device,
        )
        if narrowed
        else None
    )
    candidate_lengths = torch.zeros_like(lengths) if narrowed else None
    record = backend.sparse_topk.prefill
    if record is not None:
        # Place saved logical selections into this boundary's prepared query
        # view. Post-CED boundaries never expand back to the full prompt.
        rows = backend._lookup_rows(
            positions, requests, record.positions, record.request_indices
        )
        valid = rows >= 0
        at = rows[valid]
        topk[at] = record.logical_rows[valid].to(torch.int32)
        lengths[at] = record.lengths[valid].to(torch.int32)
        if record.candidates is not None:
            c = record.candidates
            rows = backend._lookup_rows(
                positions, requests, c.positions, c.request_indices
            )
            valid = rows >= 0
            width = c.block_ids.shape[1]
            candidates[rows[valid], :width] = c.block_ids[valid].to(torch.int32)
            candidate_lengths[rows[valid]] = c.lengths[valid].to(torch.int32)
    return dict(
        v41_cache_positions=descriptors,
        v41_cache_pages=pages,
        v41_topk_rows=topk,
        v41_topk_lengths=lengths,
        v41_candidate_blocks=(
            candidates if layer_end > backend.spec.candidate_source else None
        ),
        v41_candidate_lengths=(
            candidate_lengths if layer_end > backend.spec.candidate_source else None
        ),
    )


def restore(backend, inbound: PPStageState, *, layer_start: int, mode) -> None:
    """Restore current pages and logical selections before local layer execution."""
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        V41Candidates,
        V41Selection,
    )

    known = [
        i
        for i, owner in enumerate(backend.spec.kv_owners)
        if i == owner and i < layer_start
    ]
    owners = tuple(known[-1:])
    for index, owner in enumerate(owners):
        if owner >= layer_start:
            continue
        descriptor = inbound.v41_cache_positions[index]
        count = len(
            logical_page_descriptors(
                backend._prefill_spans, ratio=backend.spec.compress_ratios[owner]
            )
        )
        req, raw = descriptor[:count].unbind(1)
        wire_rows = torch.arange(count, device=descriptor.device)
        gid = backend._owner_group(owner)
        slots = backend.cache_slots(gid, raw, req, mode)
        torch._assert_async(
            (slots >= 64).all(), "V4.1 PP destination page is not allocated"
        )
        land_packed_pages(
            inbound.v41_cache_pages[index],
            backend.cache_pool.global_kv(owner),
            backend.cache_pool.index_k(owner),
            slots // 64,
            wire_rows,
        )
    source_layers = [i for i in range(layer_start) if backend.spec.kv_owners[i] >= 0]
    if not source_layers:
        return
    layer = source_layers[-1]
    owner, source = backend.spec.kv_owners[layer], backend.spec.index_sources[layer]
    meta = backend.query_metadata(mode)
    if layer_start > backend.spec.candidate_source:
        meta = backend.decoder_view().metadata
    logical, lengths = inbound.v41_topk_rows, inbound.v41_topk_lengths
    candidates = None
    if layer >= backend.spec.candidate_source:
        candidates = V41Candidates(
            meta.positions,
            meta.request_indices,
            inbound.v41_candidate_blocks,
            inbound.v41_candidate_lengths,
        )
    physical = backend.global_read_slots(
        owner, logical, meta.positions, meta.request_indices, mode
    )
    backend.sparse_topk.prefill = V41Selection(
        owner,
        source,
        meta.positions,
        meta.request_indices,
        logical,
        lengths,
        candidates,
        physical,
    )
