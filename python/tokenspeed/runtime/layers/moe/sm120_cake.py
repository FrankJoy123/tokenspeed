"""FlashInfer Cake routed experts for the opt-in DSV4.1 SM120 experiment."""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)


def is_aligned_qcp8_ep8(mapping) -> bool:
    attn = mapping.attn
    return (
        attn.dp_size == 1
        and attn.tp_size == attn.qcp_size == 8
        and mapping.moe.ep_size == 8
        and attn.qcp_group == mapping.moe.ep_group
    )


def validate_sm120_cake_mapping(mapping) -> None:
    attn = mapping.attn
    dp8 = (attn.dp_size, attn.tp_size, attn.qcp_size, mapping.moe.ep_size) == (
        8,
        1,
        1,
        8,
    )
    if not (dp8 or is_aligned_qcp8_ep8(mapping)):
        raise ValueError("Prefill Cake requires attention DP8 or QCP8 with EP8")


def use_sm120_cake_for_layer(
    mapping,
    *,
    capacity: int,
    has_kv_sources: bool,
    encoder: bool,
    dp_ced: bool,
    qcp_ced: bool,
) -> bool:
    """Select Cake for Encoder layers and the opt-in CED layers."""
    if qcp_ced and (not capacity or not is_aligned_qcp8_ep8(mapping)):
        raise ValueError("Cake CED requires a nonzero Cake capacity on QCP8/EP8")
    if dp_ced and (capacity == 0 or not has_kv_sources or mapping.attn.has_qcp):
        raise ValueError(
            "DP8 Cake CED requires attention DP8 with EP8, "
            "V4.1 KV sources and nonzero Cake capacity"
        )
    use_cake = capacity > 0 and has_kv_sources and (encoder or dp_ced or qcp_ced)
    if use_cake:
        validate_sm120_cake_mapping(mapping)
    return use_cake


def setup_sm120_cake(experts, mapping, capacity) -> None:
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    if experts.plan["apply_kernel_name"] != "flashinfer_sm12x_mxfp4_moe_apply":
        raise ValueError("SM120 Cake requires the SM120 FlashInfer MXFP4 expert plan")
    if experts.plan.get("a2a_backend") == "deepep":
        raise ValueError("SM120 Cake supplies its own expert transport")

    group = process_group_manager.get_process_group("nccl", mapping.moe.ep_group)
    experts.plan["a2a_backend"] = "flashinfer_moe_ep"
    experts.plan["process_group"] = group
    experts.plan["persistent_max_num_tokens_per_gpu"] = capacity
    experts.plan["weight_preprocessor"] = prepare_sm120_cake_weights
    experts.plan["warmup"] = warmup_sm120_cake


def prepare_sm120_cake_weights(plan, w) -> None:
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
        transform_prequantized_weights,
    )

    limit = None if w.swiglu_arg is None else w.swiglu_arg.limit
    if limit is not None and limit != int(limit):
        raise ValueError("SM120 Cake requires an integer SwiGLU clamp")
    if w.w13_input_layout != "concatenated" or w._swiglu_form() != "standard":
        raise ValueError("SM120 Cake requires canonical weights and standard SwiGLU")

    group = plan["process_group"]
    capacity = plan["persistent_max_num_tokens_per_gpu"]
    transport = os.environ.get("TOKENSPEED_SM120_CAKE_TRANSPORT", "all_gin")
    weights = MoEWeightPack(
        w.w13_weight,
        w.w2_weight,
        w.w13_weight_scale.view(torch.float8_e8m0fnu),
        w.w2_weight_scale.view(torch.float8_e8m0fnu),
    )
    packed = transform_prequantized_weights(
        weights.w13,
        weights.w2,
        weights.w13_scale,
        weights.w2_scale,
        hidden_size=w.hidden_size,
        intermediate_size=w.intermediate_size,
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=group.size(),
            rank=group.rank(),
            process_group=group,
            device=torch.cuda.current_device(),
        ),
        fleet_params=FleetParams(
            num_experts=w.num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=w.hidden_size,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=w.intermediate_size,
                top_k=w.top_k,
                swiglu_limit=0 if limit is None else int(limit),
                transport_table=transport,
            ),
            quantize_input=True,
            preprocess_weights=False,
            transformed_weights=packed,
        ),
    )
    # Keep the module parameters and backend aliased to one expert weight copy.
    w.w13_weight = nn.Parameter(packed.w1_fp4, requires_grad=False)
    w.w13_weight_scale = nn.Parameter(packed.w1_sfb, requires_grad=False)
    w.w2_weight = nn.Parameter(packed.w2_fp4, requires_grad=False)
    w.w2_weight_scale = nn.Parameter(packed.w2_sfb, requires_grad=False)
    w._moe_backend_state = Sm120CakeRoutedExperts(
        layer, capacity, w.hidden_size, w.top_k, w.num_experts, group.rank()
    )


def warmup_sm120_cake(plan, w) -> None:
    adapter = w._moe_backend_state
    adapter.warmup()
    if w.layer_index == 0 and plan["process_group"].rank() == 0:
        table = adapter.layer._workspace.transport_table
        logger.info("SM120 Cake EP transport %s: %s", table.name, table.cells)


class Sm120CakeRoutedExperts:
    """Stage a rank-uniform token bucket without host synchronization."""

    def __init__(self, layer, capacity, hidden, top_k, experts, rank):
        self.layer = layer
        self.capacity = capacity
        self.hidden = hidden
        self.top_k = top_k
        self.experts = experts
        self.rank = rank
        self.ready = False

    def warmup(self) -> None:
        if self.ready:
            return
        rows = min(16, self.capacity)
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.zeros((rows, self.hidden), device=device, dtype=torch.bfloat16)
        ids = (
            (torch.arange(rows * self.top_k, device=device) + self.rank * rows)
            % self.experts
        ).view(rows, self.top_k)
        weights = torch.full(
            (rows, self.top_k), 1.0 / self.top_k, device=device, dtype=torch.float32
        )
        self._forward(x, ids, weights, rows)
        torch.cuda.current_stream().synchronize()
        self.ready = True

    def run(self, x, ids, weights, maximum: int) -> torch.Tensor:
        if not self.ready or not 0 <= x.shape[0] <= maximum <= self.capacity:
            raise RuntimeError("SM120 Cake workspace or rank-uniform bucket invalid")
        return self._forward(x, ids, weights, maximum)

    def _forward(self, x, ids, weights, maximum: int) -> torch.Tensor:
        from flashinfer.moe_ep import MoEEpTensors

        inputs = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
        self.layer.stage_inputs(inputs, compile_tokens_per_rank=maximum)
        output = self.layer.compute_staged(output=None)
        # Check before the next layer resets this pooled workspace.
        protocol_error = self.layer._workspace.storage["protocol_error"].view(
            torch.int32
        )
        torch._assert_async(protocol_error == 0, "SM120 Cake protocol error")
        return output
