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

"""Reviewed attention-TP/EP layouts on the current control plane."""

import json

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.mark.parametrize("pp,tp", [(1, 8), (2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize("backend", ["none", "nccl", "deepep"])
def test_local_encoder_cache_layout(tmp_path, pp, tp, backend):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV41ForCausalLM"]})
    )
    args = ServerArgs(
        model=str(tmp_path),
        world_size=8,
        attn_tp_size=tp,
        data_parallel_size=1,
        pipeline_parallel_size=pp,
        ep_size=tp,
        moe_tp_size=1,
        enable_expert_parallel=True,
        moe_backend="marlin",
        all2all_backend=backend,
        prefill_only_mode="encoder-cache",
        disable_kvstore=True,
        language_model_only=True,
    )
    assert args.mapping.pp_size == pp
    assert args.mapping.attn.tp_size == tp
    assert args.mapping.attn.dp_size == 1
    assert args.mapping.moe.ep_size == tp
    assert args.mapping.moe.tp_size == 1
    assert args.disable_prefill_graph and args.enforce_eager
    assert args.disaggregation_mode == "null"


@pytest.mark.parametrize(
    "changes",
    [
        {"disaggregation_mode": "decode"},
        {"speculative_algorithm": "DSPARK"},
        {"prefill_only_mode": "other"},
        {"enable_mixed_batch": True},
    ],
)
def test_local_cache_refuses_incompatible_execution(tmp_path, changes):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV41ForCausalLM"]})
    )
    options = dict(
        model=str(tmp_path),
        world_size=8,
        attn_tp_size=8,
        prefill_only_mode="encoder-cache",
        disable_kvstore=True,
    )
    options.update(changes)
    with pytest.raises(ValueError):
        ServerArgs(**options)
