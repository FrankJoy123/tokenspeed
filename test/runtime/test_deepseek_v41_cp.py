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

"""V4.1 query rows, canonical compression, and protected full SWA inputs."""

from test.runtime.test_deepseek_v41_cache import _backend, _extend, _final, _tables

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.deepseek_v41_cp import (
    projection_mapping,
    request_query_rows,
)
from tokenspeed.runtime.utils.server_args import validate_context_parallel_roles


def _cp_extend(backend, lengths, prefixes, replays, rank, size):
    counts = torch.tensor(lengths, dtype=torch.int32)
    prefix = torch.tensor(prefixes, dtype=torch.int32)
    plan = QueryShardPlan.from_forward(
        total_tokens=sum(lengths), input_lengths=lengths, size=size, rank=rank
    )
    backend.init_forward_metadata(
        len(lengths),
        len(lengths),
        torch.arange(len(lengths), device=backend.device),
        (counts + prefix).to(backend.device),
        ForwardMode.EXTEND,
        block_tables=_tables(backend.device),
        extend_seq_lens=counts.to(backend.device),
        extend_seq_lens_cpu=counts,
        extend_prefix_lens=prefix.to(backend.device),
        extend_prefix_lens_cpu=prefix,
        extend_replay_lens_cpu=torch.tensor(replays, dtype=torch.int32),
        extend_prompt_lens_cpu=counts + prefix,
        extend_with_prefix=bool(prefixes),
        query_shard=plan,
    )
    return plan, backend.query_metadata(ForwardMode.EXTEND)


@pytest.mark.parametrize("rank", range(8))
def test_query_projection_rows_hold_all_heads_and_preserve_global_ep(rank):
    mapping = Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        attn_qcp_size=8,
        attn_kvp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    projected = projection_mapping(mapping)
    assert projected.attn.tp_size == 1 and projected.attn.tp_group == (rank,)
    assert projected.moe.ep_group == mapping.moe.ep_group == tuple(range(8))
    assert mapping.attn.tp_size == 8 and mapping.attn.qcp_size == 8


def test_off_layout_retains_identical_projection_mapping():
    mapping = Mapping(
        rank=0,
        world_size=2,
        attn_tp_size=2,
        attn_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=2,
    )
    assert projection_mapping(mapping) is mapping


@pytest.mark.parametrize("rank", range(8))
def test_canonical_compressor_and_swa_plans_survive_local_query_view(rank):
    lengths, prefixes, replays = [19, 46], [129, 255], [1, 3]
    reference = _backend("cpu", 2)
    full = _extend(
        reference, _tables("cpu"), lengths, prefixes, replays, _final(lengths, prefixes)
    )
    backend = _backend("cpu", 2)
    plan, local = _cp_extend(backend, lengths, prefixes, replays, rank, 8)
    assert torch.equal(local.positions, full.positions[plan.local_slice])
    assert torch.equal(local.request_indices, full.request_indices[plan.local_slice])
    assert torch.equal(local.global_write_floor, full.global_write_floor)
    local_pointer = local.positions.data_ptr()
    with backend.full_query_metadata():
        current = backend.query_metadata(ForwardMode.EXTEND)
        assert torch.equal(current.positions, full.positions)
        for actual, expected in zip(current.compressor, full.compressor, strict=True):
            assert torch.equal(actual, expected)
        actual_plan = backend._swa_query_plan(
            current.positions, current.request_indices, ForwardMode.EXTEND
        )
        expected_plan = reference._swa_query_plan(
            full.positions, full.request_indices, ForwardMode.EXTEND
        )
        for actual, expected in zip(
            actual_plan.requests, expected_plan.requests, strict=True
        ):
            assert actual.rows == expected.rows
            assert torch.equal(actual.swa_indices, expected.swa_indices)
            assert torch.equal(actual.prefix_slots, expected.prefix_slots)
    assert (
        backend.query_metadata(ForwardMode.EXTEND).positions.data_ptr() == local_pointer
    )


def test_request_query_rows_cover_every_packed_row_once_without_reordering():
    plans = [
        QueryShardPlan.from_forward(
            total_tokens=65, input_lengths=[19, 46], size=8, rank=r
        )
        for r in range(8)
    ]
    covered = []
    for plan in plans:
        for source in [slice(0, 19), slice(19, 65)]:
            bound = request_query_rows(plan, source)
            if bound is None:
                continue
            query_rows, source_queries = bound
            packed = list(range(plan.local_start, plan.local_end))[query_rows]
            assert packed == list(range(source.start, source.stop))[source_queries]
            covered += packed
    assert covered == list(range(65))


def test_full_source_scope_restores_local_metadata_on_failure(monkeypatch):
    backend = _backend("cpu", 2)
    plan, local = _cp_extend(backend, [65], [129], [1], 3, 8)
    global_hidden = torch.arange(65 * 16, dtype=torch.float32).reshape(65, 16)
    local_hidden = global_hidden[plan.local_slice].contiguous()
    # Only the unavailable multi-process network is replaced; metadata is real.
    monkeypatch.setattr(backend, "gather_query_rows", lambda values: global_hidden)
    with pytest.raises(RuntimeError, match="consumer"):
        with backend.global_source_inputs(
            local_hidden, local.positions, local.request_indices, ForwardMode.EXTEND
        ) as source:
            values, positions, requests = source
            assert torch.equal(values, global_hidden)
            assert positions.numel() == 65 and requests.numel() == 65
            raise RuntimeError("consumer")
    assert backend.query_metadata(ForwardMode.EXTEND) is local


@pytest.mark.parametrize("rank", range(8))
def test_empty_shards_are_rejected_consistently_before_collectives(rank):
    backend = _backend("cpu", 2)
    with pytest.raises(ValueError, match="nonempty shard"):
        _cp_extend(backend, [7], [0], [0], rank, 8)


def test_encoder_cache_cp_role_is_explicit_and_keeps_kv_replicated():
    kwargs = dict(
        kvp_size=1,
        dcp_size=1,
        qcp_size=8,
        disaggregation_mode="null",
        enable_mixed_batch=False,
        prefill_only_mode="encoder-cache",
    )
    validate_context_parallel_roles(**kwargs)
    with pytest.raises(ValueError, match="replicated KV"):
        validate_context_parallel_roles(**(kwargs | dict(kvp_size=8)))
    with pytest.raises(ValueError, match="pure prefill"):
        validate_context_parallel_roles(**(kwargs | dict(enable_mixed_batch=True)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_gpu_query_slices_read_the_same_full_current_swa_and_protect_prefix(
    monkeypatch,
):
    torch.manual_seed(91)
    reference = _backend("cuda", 2)
    tables = _tables("cuda")
    count = 140
    prefix = 131
    pre_q = torch.randn(prefix, 2, 512, device="cuda", dtype=torch.bfloat16)
    pre_swa = torch.randn(prefix, 512, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(count, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)
    reference.cache_pool.arena.buffer.zero_()

    def attend(backend, query, keys, meta):
        return backend.forward_v41(
            query,
            keys,
            layer_id=0,
            positions=meta.positions,
            request_indices=meta.request_indices,
            forward_mode=ForwardMode.EXTEND,
            index_q=None,
            index_weights=None,
            attn_sink=sink,
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=None,
            swa_norm_weight=None,
            swa_norm_eps=0.0,
        )

    initial = _extend(reference, tables, [prefix], [0], [0], [prefix + count])
    attend(reference, pre_q, pre_swa, initial)
    meta = _extend(reference, tables, [count], [prefix], [0], [prefix + count])
    expected = attend(reference, q, swa, meta)
    expected_cache = reference.cache_pool.arena.buffer.clone()
    outputs = []
    for rank in range(8):
        backend = _backend("cuda", 2)
        backend.cache_pool.arena.buffer.zero_()
        initial = _extend(
            backend, _tables("cuda"), [prefix], [0], [0], [prefix + count]
        )
        attend(backend, pre_q, pre_swa, initial)
        plan, local = _cp_extend(backend, [count], [prefix], [0], rank, 8)
        monkeypatch.setattr(backend, "gather_query_rows", lambda values: swa)
        outputs.append(
            attend(
                backend,
                q[plan.local_slice].contiguous(),
                swa[plan.local_slice].contiguous(),
                local,
            )
        )
        assert torch.equal(backend.cache_pool.arena.buffer, expected_cache)
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_gpu_query_views_keep_global_selection_compressor_tail_and_replay_floor(
    monkeypatch,
):
    # Odd prefixes and ragged requests put both query boundaries and pairs
    # across shard boundaries. Only the absent network is replaced.
    torch.manual_seed(92)
    lengths, prefixes, replays = [19, 46], [3, 0], [1, 0]
    count = sum(lengths)
    content = torch.randn(count, 512, device="cuda", dtype=torch.float32)
    scores = torch.randn_like(content)
    fused = torch.cat((content, scores), dim=1)
    q = torch.randn(count, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
    iq = torch.randn(count, 2, 128, device="cuda", dtype=torch.bfloat16)
    iw = torch.rand(count, 2, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)

    def publish(backend, values):
        pooled, positions, requests = backend.compress(
            2,
            values[:, :512],
            values[:, 512:],
            ForwardMode.EXTEND,
            norm_weight=None,
            norm_eps=0.0,
        )
        latent = torch.nn.functional.rms_norm(
            pooled.to(torch.bfloat16), (512,), weight=None, eps=1e-6
        )
        backend.write_global(
            2,
            latent,
            latent[:, :128].contiguous(),
            positions,
            requests,
            ForwardMode.EXTEND,
            main_rope_cache=None,
        )
        return pooled

    def attend(backend, rows, meta):
        return backend.forward_v41(
            q[rows].contiguous(),
            swa[rows].contiguous(),
            layer_id=2,
            positions=meta.positions,
            request_indices=meta.request_indices,
            forward_mode=ForwardMode.EXTEND,
            index_q=iq[rows].contiguous(),
            index_weights=iw[rows].contiguous(),
            attn_sink=sink,
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=None,
            swa_norm_weight=None,
            swa_norm_eps=0.0,
        )

    reference = _backend("cuda", 2)
    reference.cache_pool.arena.buffer.zero_()
    full = _extend(
        reference,
        _tables("cuda"),
        lengths,
        prefixes,
        replays,
        _final(lengths, prefixes),
    )
    expected_pooled = publish(reference, fused)
    expected = attend(reference, slice(None), full)
    selection = reference.sparse_topk.prefill
    expected_rows, expected_lens = (
        selection.logical_rows.clone(),
        selection.lengths.clone(),
    )
    expected_cache = reference.cache_pool.arena.buffer.clone()
    outputs = []
    for rank in range(8):
        backend = _backend("cuda", 2)
        backend.cache_pool.arena.buffer.zero_()
        plan, meta = _cp_extend(backend, lengths, prefixes, replays, rank, 8)
        monkeypatch.setattr(backend, "gather_query_rows", lambda values: fused)
        with backend.global_source_inputs(
            fused[plan.local_slice].contiguous(),
            meta.positions,
            meta.request_indices,
            ForwardMode.EXTEND,
        ) as source:
            pooled = publish(backend, source[0])
            torch.testing.assert_close(pooled, expected_pooled, rtol=0, atol=0)
        monkeypatch.setattr(backend, "gather_query_rows", lambda values: swa)
        outputs.append(attend(backend, plan.local_slice, meta))
        actual = backend.sparse_topk.prefill
        assert torch.equal(actual.logical_rows, expected_rows[plan.local_slice])
        assert torch.equal(actual.lengths, expected_lens[plan.local_slice])
        assert torch.equal(backend.cache_pool.arena.buffer, expected_cache)
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("qcp_size", [1, 8])
@pytest.mark.parametrize("encoding", ["bf16", "fp8-weight-first", "fp8-scale-first"])
def test_output_projection_checkpoint_uses_its_actual_weight_shard(
    rank, qcp_size, encoding
):
    from types import SimpleNamespace

    from torch import nn

    from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41ForCausalLM, _column

    mapping = Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        attn_qcp_size=qcp_size,
        attn_kvp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    linear = _column(32, 64, torch.bfloat16, None, "wo_a", projection_mapping(mapping))
    adapter = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(adapter)
    adapter.mapping = mapping
    adapter.model = SimpleNamespace(
        layers=[SimpleNamespace(attn=SimpleNamespace(wo_a=linear))]
    )
    pending = {}
    raw = ((torch.arange(64 * 32).reshape(64, 32) % 11) - 5).float()
    if encoding == "bf16":
        adapter._load_wo_a("layers.0.attn.wo_a.weight", raw.to(torch.bfloat16), pending)
        expected = raw.to(torch.bfloat16)
    else:
        codes = raw.to(torch.float8_e4m3fn)
        scales = torch.tensor([[127], [128]], dtype=torch.uint8)
        values = {"weight": codes, "scale": scales}
        order = (
            ("weight", "scale")
            if encoding == "fp8-weight-first"
            else ("scale", "weight")
        )
        for field in order:
            adapter._load_wo_a("layers.0.attn.wo_a." + field, values[field], pending)
        expected = (
            codes.float()
            * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 0)
        ).to(torch.bfloat16)
        assert not pending
    if qcp_size == 1:
        expected = expected.chunk(8, dim=0)[rank]
    torch.testing.assert_close(linear.weight, expected, rtol=0, atol=0)


def test_replicated_kv_query_owners_do_not_allocate_sharded_history_workspace():
    from test.runtime.test_deepseek_v41_cache import _recipe
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.registry import _prepare_fixed_workspaces

    backend = _backend("cpu", 2)
    # V4.1 already budgets and owns its ordinary cache scratch. Its base
    # history-gather hook raises, so using it here reproduces engine startup.
    _prepare_fixed_workspaces(
        server_args=SimpleNamespace(speculative_num_draft_tokens=None),
        config=SimpleNamespace(qcp_size=8, kvp_size=1, max_bs=2, context_len=1024),
        backend=backend,
        draft_backend=None,
        uses_paged_state_verify=False,
        is_inkling=False,
        expected_bytes=_recipe("cpu").workspace_bytes(),
    )
    assert backend.history_gather_workspace() is None
