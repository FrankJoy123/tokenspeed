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

"""DeepEP normal with a separately selected SM120 CUTLASS W4A8 expert."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe.deep_ep import _marlin_mxfp4_get_dispatcher
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.sm120_cutlass_local import (
    Sm120CutlassLocalMoe,
)

_STATE = "flashinfer_cutlass_sm120_local"


def flashinfer_cutlass_sm120_deepep_weights(plan, w):
    """Re-layout original MXFP4 bytes once and retain only the prepared views."""
    if w.tp_size != 1 or w.ep_size != 4 or w.w13_input_layout != "concatenated":
        raise ValueError(
            "Qualified CUTLASS local experts require TP1/EP4 and concatenated weights"
        )
    if w.swiglu_beta is not None or (
        w.swiglu_arg is not None and w.swiglu_arg.alpha not in (None, 1.0)
    ):
        raise ValueError("Qualified CUTLASS local experts require standard SwiGLU")
    state = Sm120CutlassLocalMoe(
        w13=w.w13_weight.data,
        w2=w.w2_weight.data,
        s13=w.w13_weight_scale.data,
        s2=w.w2_weight_scale.data,
        clamp=None if w.swiglu_arg is None else w.swiglu_arg.limit,
        hidden=w.hidden_size,
        intermediate=w.intermediate_size,
        top_k=w.top_k,
    )
    w.w13_weight = torch.nn.Parameter(state.w13, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(state.s13, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(state.w2, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(state.s2, requires_grad=False)
    plan[_STATE] = state


@register_kernel(
    "moe",
    "apply",
    name="flashinfer_cutlass_sm120_deepep_apply",
    solution="flashinfer_cutlass_sm120_deepep",
    weight_preprocessor=flashinfer_cutlass_sm120_deepep_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 0),
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
        "a2a_backend": frozenset({"deepep"}),
        "deepep_modes": frozenset({"normal"}),
        "internal_activation_dtype": frozenset({"fp8"}),
        "supports_bias": frozenset({False}),
        "combine_order": frozenset({"rank"}),
    },
    priority=Priority.PORTABLE,
)
def flashinfer_cutlass_sm120_deepep_apply(
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
    """Use unchanged BF16 DeepEP legs, with MXFP8 quantization after dispatch."""
    if not do_finalize or topk_ids is None or topk_weights is None:
        raise ValueError("CUTLASS DeepEP requires finalized precomputed routing")
    if low_latency not in (None, False) or plan["deepep_mode"] != "normal":
        raise ValueError("Qualified CUTLASS experts support DeepEP normal only")
    if max_num_tokens_per_gpu is not None and x.shape[0] > max_num_tokens_per_gpu:
        raise ValueError("Local source exceeds the agreed DeepEP row bound")
    if num_tokens_global is not None and x.shape[0] > num_tokens_global:
        raise ValueError("Global row count is smaller than local source")
    # This helper only owns transport configuration; it never accesses the
    # Marlin-specific weight layout or invokes a Marlin GEMM.
    dispatcher = _marlin_mxfp4_get_dispatcher(plan, w, x)
    dispatcher.dispatch_a(
        x, topk_ids.to(torch.int64), topk_weights.to(torch.float32), low_latency=False
    )
    if overlap_fn is not None:
        overlap_fn()
    recv_x, recv_ids, recv_weights, *_ = dispatcher.dispatch_b()
    expert_out = plan[_STATE].run(recv_x, recv_ids, recv_weights, enable_pdl=enable_pdl)
    dispatcher.combine_a(expert_out, recv_ids, recv_weights, low_latency=False)
    return dispatcher.combine_b()
