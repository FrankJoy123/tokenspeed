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

"""Real Qwen NextN construction and full-vocabulary logits under attention DP."""

import copy

import pytest
import torch

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5Config, Qwen3_5TextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.factory import _check_shared_head_layout
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from tokenspeed.runtime.models.qwen3_5_nextn import (
    Qwen3_5ForConditionalGenerationNextN,
)


def config(tied):
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        full_attention_interval=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=128,
        tie_word_embeddings=tied,
    )


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("tied", [False, True])
def test_attention_dp_draft_matches_target_full_vocab_logits(tp, rank, tied):
    mapping = Mapping(
        rank=rank,
        world_size=4,
        attn_tp_size=tp,
        attn_dp_size=4 // tp,
        dense_tp_size=tp,
        moe_ep_size=tp,
        moe_tp_size=1,
    )
    cfg = config(tied)
    if tied and tp > 1:
        with pytest.raises(
            ValueError, match="matching embedding and LM-head TP groups"
        ):
            Qwen3_5ForConditionalGenerationNextN(copy.deepcopy(cfg), mapping)
        return
    target = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(text_config=cfg.to_dict()), mapping, is_multimodal_active=False
    )
    draft = Qwen3_5ForConditionalGenerationNextN(copy.deepcopy(cfg), mapping)
    # Sharing must accept the actual target/draft modules on every DP rank.
    _check_shared_head_layout(target, draft)
    draft.lm_head = target.lm_head
    draft.logits_processor.require_full_vocab_logits()
    hidden = torch.arange(3 * cfg.hidden_size, dtype=torch.float32).reshape(3, -1) / 200
    weight = torch.arange(cfg.vocab_size * cfg.hidden_size, dtype=torch.float32)
    weight = torch.sin(weight.reshape(cfg.vocab_size, cfg.hidden_size) / 31)
    with torch.no_grad():
        target.lm_head.weight.copy_(weight)
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    actual = draft.logits_processor._get_logits(
        hidden, draft.lm_head, metadata, require_full_vocab=True
    )
    torch.testing.assert_close(actual, hidden @ weight.T)


@pytest.mark.parametrize("rank", range(4))
def test_explicit_dp_vocab_sharding_matches_real_target(rank):
    mapping = Mapping(
        rank=rank,
        world_size=4,
        attn_tp_size=1,
        attn_dp_size=4,
        dense_tp_size=1,
        moe_ep_size=1,
        moe_tp_size=1,
        lm_head_tp_size=2,
    )
    cfg = config(False)
    target = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(text_config=cfg.to_dict()), mapping, is_multimodal_active=False
    )
    draft = Qwen3_5ForConditionalGenerationNextN(copy.deepcopy(cfg), mapping)
    _check_shared_head_layout(target, draft)
    assert target.lm_head.weight.shape == draft.lm_head.weight.shape == (64, 64)


def test_tied_draft_refuses_vocab_sharding_over_attention_dp():
    mapping = Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=1,
        attn_dp_size=4,
        dense_tp_size=1,
        moe_ep_size=1,
        moe_tp_size=1,
        lm_head_tp_size=2,
    )
    with pytest.raises(ValueError, match="matching embedding and LM-head TP groups"):
        Qwen3_5ForConditionalGenerationNextN(config(True), mapping)


@pytest.mark.parametrize("rank", range(4))
def test_tp4_draft_keeps_target_vocab_layout(rank):
    mapping = Mapping(
        rank=rank,
        world_size=4,
        attn_tp_size=4,
        dense_tp_size=4,
        moe_ep_size=4,
        moe_tp_size=1,
    )
    cfg = config(False)
    target = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(text_config=cfg.to_dict()), mapping, is_multimodal_active=False
    )
    draft = Qwen3_5ForConditionalGenerationNextN(copy.deepcopy(cfg), mapping)
    _check_shared_head_layout(target, draft)
    assert draft.lm_head.weight.shape == target.lm_head.weight.shape
