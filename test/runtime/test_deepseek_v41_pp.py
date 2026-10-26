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

"""Lossless packed-page transport across different stage-local allocators."""

import pytest
import torch

from tokenspeed.runtime.models.deepseek_v41_pp import (
    copy_packed_pages,
    land_packed_pages,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("global_bytes,index_bytes", [(288, 68), (584, 132)])
def test_transport_preserves_page_planar_codes_scales_and_padding(
    global_bytes, index_bytes, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Fields are strided slices of an LCM plane, not contiguous token rows.
    source = torch.randint(
        0,
        256,
        (7, 64, global_bytes + index_bytes + 31),
        dtype=torch.uint8,
        device=device,
    )
    source_global = source[:, :, :global_bytes]
    source_index = source[:, :, global_bytes : global_bytes + index_bytes]
    original = source.clone()
    page_ids = torch.tensor([5, 2, 6], device=device)
    payload = copy_packed_pages(source_global, source_index, page_ids, capacity=6)
    # Source writes after enqueue/snapshot cannot mutate in-flight wire bytes.
    source[page_ids] = 0
    destination = torch.full_like(source, 173)
    target_global = destination[:, :, :global_bytes]
    target_index = destination[:, :, global_bytes : global_bytes + index_bytes]
    land_packed_pages(
        payload,
        target_global,
        target_index,
        torch.tensor([1, 6, 3], device=device),
        torch.arange(3, device=device),
    )
    for incoming, outgoing in zip(page_ids.tolist(), [1, 6, 3]):
        assert torch.equal(
            target_global[outgoing], original[incoming, :, :global_bytes]
        )
        assert torch.equal(
            target_index[outgoing],
            original[incoming, :, global_bytes : global_bytes + index_bytes],
        )
    assert (destination[0] == 173).all()  # null page is never overwritten
    assert (destination[:, :, -31:] == 173).all()  # unrelated arena fields
    assert (payload[3:] == 0).all()


def test_transport_rejects_page_overflow_without_truncating():
    cache = torch.zeros(3, 64, 288, dtype=torch.uint8)
    index = torch.zeros(3, 64, 68, dtype=torch.uint8)
    with pytest.raises(ValueError, match="page count"):
        copy_packed_pages(cache, index, torch.tensor([1, 2]), capacity=1)


def test_page_descriptors_cover_odd_chunks_and_distinct_requests():
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        V41PrefillSpan,
    )
    from tokenspeed.runtime.models.deepseek_v41_pp import logical_page_descriptors

    spans = (
        V41PrefillSpan(0, 0, 127, 65, 0),
        V41PrefillSpan(1, 65, 256, 1, 0),
        V41PrefillSpan(2, 66, 5, 0, 0),
    )
    assert logical_page_descriptors(spans, ratio=2) == [(0, 0), (0, 128), (1, 256)]
    assert logical_page_descriptors(spans, ratio=1) == [(0, 64), (0, 128), (1, 256)]


@pytest.mark.parametrize("rank", range(8))
def test_host_file_writer_is_local_tp_owner_not_world_rank_zero(rank):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.deepseek_v41_engram import _is_host_table_writer

    mapping = Mapping(world_size=8, pp_size=4, attn_tp_size=2)
    mapping.rank = rank
    assert _is_host_table_writer(mapping) == (rank % 2 == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_backend_relocates_live_owner_and_compact_selection():
    from test.runtime.test_deepseek_v41_cache import _backend, _extend, _tables

    from tokenspeed.runtime.distributed.pp_stage import PPStageState
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        V41Candidates,
        V41Selection,
    )
    from tokenspeed.runtime.models.deepseek_v41_pp import (
        logical_page_descriptors,
        restore,
        snapshot,
    )

    source, destination = _backend("cuda", 2), _backend("cuda", 2)
    source_tables = _tables("cuda")
    destination_tables = {}
    for gid, table in source_tables.items():
        plan = destination.cache_pool.arena.plan
        packing = next(
            group.cache_blocks_per_lcm_block
            for group in plan.groups
            if group.group_id == gid
        )
        # LCM groups are mutually exclusive parent tenants. Apply the same
        # parent permutation across all groups, preserving each child offset.
        parent = (table - 1) // packing
        child = (table - 1) % packing
        relocated = ((parent + 3) % plan.num_lcm_blocks) * packing + child + 1
        destination_tables[gid] = torch.where(table > 0, relocated, table)
    for backend, tables in [(source, source_tables), (destination, destination_tables)]:
        _extend(backend, tables, [129, 65], [1, 127], [0, 0], [130, 192])
    for owner in [2, 8, 14, 20]:
        for cache in [
            source.cache_pool.global_kv(owner),
            source.cache_pool.index_k(owner),
        ]:
            cache.copy_(torch.randint_like(cache, 0, 256))
        destination.cache_pool.global_kv(owner).fill_(173)
        destination.cache_pool.index_k(owner).fill_(173)
    meta = source.decoder_view().metadata
    n = meta.positions.numel()
    rows = torch.arange(4, device="cuda", dtype=torch.int32).expand(n, -1).clone()
    rows.masked_fill_(rows > meta.positions[:, None], -1)
    lengths = (rows >= 0).sum(1).to(torch.int32)
    candidates = V41Candidates(
        meta.positions,
        meta.request_indices,
        torch.zeros(n, 2, device="cuda", dtype=torch.int32),
        torch.ones(n, device="cuda", dtype=torch.int32),
    )
    slots = source.global_read_slots(
        20, rows, meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    source.sparse_topk.prefill = V41Selection(
        20,
        24,
        meta.positions,
        meta.request_indices,
        rows,
        lengths,
        candidates,
        slots,
    )
    payload = snapshot(
        source, num_tokens=194, max_requests=2, layer_end=25, mode=ForwardMode.EXTEND
    )
    assert payload["v41_cache_pages"].shape[0] == 1
    assert payload["v41_topk_rows"].shape[0] == n < 194
    state = PPStageState(hidden_states=torch.empty(n, 4, 1, device="cuda"), **payload)
    restore(destination, state, layer_start=25, mode=ForwardMode.EXTEND)
    result = destination.sparse_topk.prefill
    assert torch.equal(result.logical_rows, rows)
    assert torch.equal(result.lengths, lengths)
    assert torch.equal(result.candidates.block_ids, candidates.block_ids)
    old_gid = destination._owner_group(2)
    live_old_pages = destination_tables[old_gid].long().flatten().unique()
    live_old_pages = live_old_pages[live_old_pages > 0]
    assert (destination.cache_pool.global_kv(2)[live_old_pages] == 173).all()
    descriptors = logical_page_descriptors(source._prefill_spans, ratio=1)
    req, pos = torch.tensor(descriptors, device="cuda", dtype=torch.int64).unbind(1)
    gid = source._owner_group(20)
    source_slots = source.cache_slots(gid, pos, req, ForwardMode.EXTEND)
    target_slots = destination.cache_slots(gid, pos, req, ForwardMode.EXTEND)
    assert not torch.equal(source_slots, target_slots)
    for source_cache, target_cache in [
        (source.cache_pool.global_kv(20), destination.cache_pool.global_kv(20)),
        (source.cache_pool.index_k(20), destination.cache_pool.index_k(20)),
    ]:
        assert torch.equal(
            source_cache[source_slots // 64], target_cache[target_slots // 64]
        )


@pytest.mark.parametrize("pp_size", [4, 8])
def test_prefill_engram_snapshots_allow_pipeline_depth_without_decode_feedback(pp_size):
    from tokenspeed.runtime.engine.scheduler_utils import validate_engram_inflight

    validate_engram_inflight(
        enabled=True, pp_size=pp_size, in_flight_depth=pp_size, prefill_only=True
    )
    with pytest.raises(NotImplementedError, match="only for prefill"):
        validate_engram_inflight(
            enabled=True, pp_size=pp_size, in_flight_depth=1, prefill_only=False
        )


def test_single_stage_engram_retains_decode_feedback_depth_limit():
    from tokenspeed.runtime.engine.scheduler_utils import validate_engram_inflight

    validate_engram_inflight(
        enabled=True, pp_size=1, in_flight_depth=1, prefill_only=False
    )
    with pytest.raises(NotImplementedError, match="depth"):
        validate_engram_inflight(
            enabled=True, pp_size=1, in_flight_depth=2, prefill_only=False
        )


@pytest.mark.parametrize("pp_size", [4, 8])
def test_flatkv_shared_owner_views_survive_pipeline_narrowing(pp_size):
    from test.runtime.test_deepseek_v41_cache import _layout, _recipe

    from tokenspeed.runtime.distributed.pp_stage import pp_stage_windows
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
    from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
        DeepseekV41CachePool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
        cache_field_placement,
        pipeline_cache_ownership,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec
    from tokenspeed.runtime.layers.attention.registry import _narrow_spec_for_pp

    recipe = _recipe("cpu")
    plan = _layout(recipe).bind(2)
    groups = tuple(spec for spec, _ in recipe.groups())
    spec = CachePoolSpec(
        family="deepseek_v41",
        memory_plan=plan,
        layer_types=recipe.layer_types,
        cache_group_specs=groups,
        token_capacity=256,
    )
    owners = pipeline_cache_ownership(40, 0, pp_stage_windows(40, pp_size))
    fields, _ = cache_field_placement(plan, owners)
    assert sum("layer.2.global_kv" in stage for stage in fields) == 1
    for owner in owners:
        stage = _narrow_spec_for_pp(spec, owner)
        assert stage.memory_plan.arena_bytes == plan.arena_bytes
        arena = CacheArena(
            stage.memory_plan,
            "cpu",
            cache_group_specs=groups,
            token_capacity=256,
            enable_memory_saver=False,
        )
        pool = DeepseekV41CachePool(arena, layer_num=40, rank=0, field_layer_offset=0)
        # A later stage must be able to land/read a source outside its window.
        gid = plan.field("layer.2.global_kv").group_id
        page = plan.group(gid).cache_blocks_per_lcm_block + 1
        pool.global_kv(2)[page].fill_(73)
        assert (pool.global_kv(2)[page] == 73).all()
        pool.swa(owner.target_cache_window[0])[1].fill_(21)
        assert (pool.global_kv(2)[page] == 73).all()
