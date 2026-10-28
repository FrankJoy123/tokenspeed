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

"""Qualified SM120 CUTLASS local MXFP4 x MXFP8 experts; no transport."""

from __future__ import annotations

import torch


class Sm120CutlassLocalMoe:
    def __init__(self, *, w13, w2, s13, s2, clamp, hidden, intermediate, top_k):
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("This CUTLASS expert adapter qualifies SM120 only")
        if clamp is None or hidden % 128 or intermediate % 128:
            raise ValueError(
                "CUTLASS experts require explicit clamp and 128-aligned dimensions"
            )
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
        from flashinfer.tllm_enums import ActivationType

        self.hidden = hidden
        self.top_k = top_k
        self.quantize = mxfp8_quantize
        self.autotune = autotune
        # The qualified fast-build module is explicit; do not mutate the library's
        # global getter or replace the distributed communication implementation.
        self.runner = get_cutlass_fused_moe_module("120", use_fast_build=True)
        self.activation = ActivationType.SwigluBias
        # CUTLASS consumes [up; gate], while the checkpoint stores [gate; up].
        self.w13 = (
            torch.cat((w13[:, intermediate:], w13[:, :intermediate]), dim=1)
            .contiguous()
            .view(torch.int64)
        )
        s13_up_gate = torch.cat(
            (s13[:, intermediate:], s13[:, :intermediate]), dim=1
        ).contiguous()
        self.s13 = (
            block_scale_interleave(s13_up_gate)
            .reshape_as(s13_up_gate)
            .view(torch.int32)
        )
        self.w2 = w2.contiguous().view(torch.int64)
        self.s2 = block_scale_interleave(s2).reshape_as(s2).view(torch.int32)
        self.one = torch.ones(w13.shape[0], dtype=torch.float32, device=w13.device)
        self.zero = torch.zeros_like(self.one)
        self.limit = torch.full_like(self.one, float(clamp))

    def run(self, x, ids, weights, *, enable_pdl):
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != self.hidden:
            raise ValueError("CUTLASS local experts require BF16 [rows, hidden]")
        if ids.shape != (x.shape[0], self.top_k) or weights.shape != ids.shape:
            raise ValueError("CUTLASS local expert routing shape mismatch")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        q, sf = self.quantize(x, is_sf_swizzled_layout=True, alignment=32)
        output = torch.empty_like(x)
        # Received row counts and empty spans differ between EP peers. The
        # process-wide profiling group requires identical operations on every
        # rank, which a local expert cannot promise. Use the qualified heuristic
        # for these two GEMMs while leaving dense-op tuning active.
        with self.autotune(
            False, skip_ops={"trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"}
        ):
            return self.runner.cutlass_fused_moe(
                output=output,
                input=q,
                token_selected_experts=ids.to(torch.int32),
                token_final_scales=weights.to(torch.float32),
                fc1_expert_weights=self.w13,
                fc1_expert_biases=None,
                fc2_expert_weights=self.w2,
                fc2_expert_biases=None,
                output_dtype=torch.bfloat16,
                quant_scales=[self.s13, self.one, self.s2, self.one],
                input_sf=sf,
                swiglu_alpha=self.one,
                swiglu_beta=self.zero,
                swiglu_limit=self.limit,
                activation_type=self.activation,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                enable_alltoall=False,
                use_mxfp8_act_scaling=True,
                use_fused_finalize=True,
                tune_max_num_tokens=16384,
                enable_pdl=enable_pdl,
            )[0]
