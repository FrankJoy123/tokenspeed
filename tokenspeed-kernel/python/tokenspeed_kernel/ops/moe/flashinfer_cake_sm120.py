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

"""Explicit FlashInfer SM120 W4A8 expert-parallel backend."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.sm120_cake_moe_ep import Sm120CakeMoeEp


def flashinfer_cake_sm120_moe_ep_weights(plan: dict, w: torch.nn.Module) -> None:
    """Transform loaded MXFP4 weights and prepare the shared workspace."""
    if w.tp_size != 1 or w.ep_size != 4:
        raise ValueError("FlashInfer SM120 MoE EP requires expert TP1/EP4")
    if w.w13_input_layout != "concatenated":
        raise ValueError(
            "FlashInfer SM120 MoE EP requires concatenated gate/up weights"
        )
    if w.swiglu_beta is not None or (
        w.swiglu_arg is not None and w.swiglu_arg.alpha not in (None, 1.0)
    ):
        raise ValueError("FlashInfer SM120 MoE EP requires standard SwiGLU")
    capacity = plan["persistent_max_num_tokens_per_gpu"]
    if capacity is None:
        raise ValueError(
            "FlashInfer SM120 MoE EP requires explicit persistent capacity"
        )
    state = Sm120CakeMoeEp(
        w13=w.w13_weight.data,
        w2=w.w2_weight.data,
        w13_scale=w.w13_weight_scale.data,
        w2_scale=w.w2_weight_scale.data,
        group=plan["process_group"],
        capacity=capacity,
        compile_buckets=tuple(sorted({min(capacity, 512), capacity})),
        hidden=w.hidden_size,
        intermediate=w.intermediate_size,
        experts=w.num_experts,
        top_k=w.top_k,
        clamp=None if w.swiglu_arg is None else w.swiglu_arg.limit,
        fast_math=plan["fast_math"],
    )
    transformed = state.transformed_weights
    fc1, sf1 = transformed.w1_fp4, transformed.w1_sfb
    fc2, sf2 = transformed.w2_fp4, transformed.w2_sfb
    # Alias transformed storage; retaining source parameters would double weights.
    w.w13_weight = torch.nn.Parameter(fc1, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(sf1, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(fc2, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(sf2, requires_grad=False)
    plan["flashinfer_cake_sm120_moe_ep"] = state
    # Initialize per-layer descriptors and the shared serial workspace before forward.
    state.prepare()


def flashinfer_cake_sm120_moe_ep_warmup(plan: dict, w: torch.nn.Module) -> None:
    """Preserve the common warmup lifecycle with idempotent preparation."""
    plan["flashinfer_cake_sm120_moe_ep"].prepare()


@register_kernel(
    "moe",
    "apply",
    name="flashinfer_cake_sm120_moe_ep_apply",
    solution="flashinfer_cake_sm120_moe_ep",
    weight_preprocessor=flashinfer_cake_sm120_moe_ep_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "ep_size": frozenset({4}),
        "supports_all_to_all_ep": frozenset({True}),
        "a2a_backend": frozenset({"flashinfer_moe_ep"}),
        "internal_activation_dtype": frozenset({"fp8"}),
        "persistent_workspace": frozenset({True}),
        "supports_bias": frozenset({False}),
        "combine_order": frozenset({"slot"}),
    },
    priority=Priority.PORTABLE,
)
def flashinfer_cake_sm120_moe_ep_apply(
    plan,
    x,
    w,
    router_logits,
    topk_weights,
    topk_ids,
    num_tokens_global,
    max_num_tokens_per_gpu,
    do_finalize,
    enable_pdl,
    low_latency,
    overlap_fn,
):
    """Return local finalized BF16 rows, including dispatch and combine.

    Inputs are quantized to MXFP8. The library owns its internal launch
    schedule; enable_pdl is a generic launch hint. The host supplies an
    agreed source-row maximum rather than choosing from local x.shape[0].
    """
    if not do_finalize or topk_weights is None or topk_ids is None:
        raise ValueError(
            "FlashInfer SM120 MoE EP requires finalized precomputed routing"
        )
    if low_latency not in (None, False) or overlap_fn is not None:
        raise ValueError(
            "FlashInfer SM120 MoE EP does not accept DeepEP legs/callbacks"
        )
    if max_num_tokens_per_gpu is None:
        raise ValueError("FlashInfer SM120 MoE EP needs an agreed host row bound")
    if num_tokens_global is not None and not (x.shape[0] <= num_tokens_global):
        raise ValueError("MoE global token count is smaller than local rows")
    return plan["flashinfer_cake_sm120_moe_ep"].run(
        x,
        topk_ids,
        topk_weights,
        maximum=max_num_tokens_per_gpu,
    )


flashinfer_cake_sm120_moe_ep_apply._tokenspeed_warmup = (
    flashinfer_cake_sm120_moe_ep_warmup
)
