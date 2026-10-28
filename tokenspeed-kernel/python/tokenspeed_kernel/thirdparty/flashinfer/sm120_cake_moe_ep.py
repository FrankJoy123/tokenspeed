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

"""Pinned Cake SM120 MXFP4 x MXFP8 expert-parallel adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

EXPECTED_MANIFEST_SHA256 = (
    "8ae21cae7aaec85ab2b0f2c2cb11cb274dc1ec2327ea79eab6023c03ca4d9104"
)


class Sm120CakeMoeEp:
    """Own transformed weights; share the backend's serial workspace across layers."""

    def __init__(
        self,
        *,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        group: Any,
        capacity: int,
        compile_buckets: tuple[int, ...],
        hidden: int,
        intermediate: int,
        experts: int,
        top_k: int,
        clamp: float | None,
        fast_math: bool,
    ) -> None:
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("Cake MoE requires the verified SM120 architecture")
        if group is None or group.size() != 4 or capacity < 1:
            raise ValueError("Cake MoE currently requires positive capacity and EP4")
        if not fast_math:
            raise ValueError("Cake MoE requires fast_math=True")
        if clamp is None or clamp != int(clamp):
            raise ValueError("Cake MoE requires an explicit integral SwiGLU clamp")
        import flashinfer
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEWeightPack,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
            transform_prequantized_weights,
        )

        manifest = Path(flashinfer.__file__).parent / (
            "moe_ep/kernel_src/sm120/cake_mxfp4_megamoe/src/"
            "cake_sm120_megamoe_manifest.json"
        )
        if (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            != EXPECTED_MANIFEST_SHA256
        ):
            raise RuntimeError(
                "Cake MoE requires the qualified FlashInfer399b6c1d source"
            )
        self.capacity = capacity
        self.workspace_capacity = (capacity + 127) // 128 * 128
        self.hidden = hidden
        self.top_k = top_k
        self.buckets = tuple(
            sorted({min(self.capacity, b) for b in compile_buckets} | {self.capacity})
        )
        self.prepared = False
        self.transformed_weights = transform_prequantized_weights(
            w13,
            w2,
            w13_scale,
            w2_scale,
            hidden_size=hidden,
            intermediate_size=intermediate,
        )
        self.layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=group.size(),
                rank=group.rank(),
                process_group=group,
                device=torch.cuda.current_device(),
            ),
            fleet_params=FleetParams(
                num_experts=experts,
                max_tokens_per_rank=self.workspace_capacity,
                token_hidden_size=hidden,
            ),
            weights=MoEWeightPack(
                w13,
                w2,
                w13_scale.view(torch.float8_e8m0fnu),
                w2_scale.view(torch.float8_e8m0fnu),
            ),
            backend=MegaConfig(
                megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                    intermediate_size=intermediate,
                    top_k=top_k,
                    swiglu_limit=int(clamp),
                    codec_overflow_blocks=16,
                    shared_expert=None,
                    transport_table="all_gin",
                ),
                quantize_input=True,
                preprocess_weights=False,
                transformed_weights=self.transformed_weights,
            ),
        )

    def _check_protocol(self) -> None:
        # Same-stream device assertion consumes the flag before the next layer
        # clears the shared workspace. It adds no CPU readback per MoE layer.
        flag = self.layer._workspace.storage["protocol_error"].view(torch.int32)
        torch._assert_async(flag == 0, "Cake protocol error; output/workspace invalid")

    def prepare(self) -> None:
        """Allocate the shared workspace and bind this layer's TMA descriptors."""
        if self.prepared:
            return
        from flashinfer.moe_ep import MoEEpTensors

        rows = 7
        x = torch.full((rows, self.hidden), 0.01, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(self.top_k, device="cuda").expand(rows, -1).contiguous()
        weights = torch.full(
            (rows, self.top_k), 1.0 / self.top_k, device="cuda", dtype=torch.float32
        )
        tensors = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
        self.layer(tensors)
        self._check_protocol()
        for bucket in self.buckets:
            self.layer.stage_inputs(tensors, compile_tokens_per_rank=bucket)
            self.layer.compute_staged(output=None)
            self._check_protocol()
        torch.cuda.synchronize()
        self.prepared = True

    def run(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        *,
        maximum: int,
    ) -> torch.Tensor:
        """Execute local rows using a host-agreed EP-uniform capacity bucket."""
        if not self.prepared:
            raise RuntimeError("Cake MoE must be prepared before forward")
        if not 0 <= x.shape[0] <= maximum <= self.capacity or maximum == 0:
            raise ValueError("Cake MoE rows exceed prepared capacity")
        if x.dtype != torch.bfloat16 or x.shape[1] != self.hidden:
            raise ValueError("Cake MoE expects BF16 hidden rows")
        from flashinfer.moe_ep import MoEEpTensors

        bucket = next(b for b in self.buckets if b >= maximum)
        tensors = MoEEpTensors(
            hidden_states=x,
            topk_ids=ids.to(torch.int64),
            topk_weights=weights.to(torch.float32),
        )
        self.layer.stage_inputs(tensors, compile_tokens_per_rank=bucket)
        output = self.layer.compute_staged(output=None)
        self._check_protocol()
        # Retire consumers independently of the next layer's shared output slot.
        return output.clone()

    def close(self) -> None:
        """Collectively release the layer after device work has retired."""
        self.layer.destroy()
