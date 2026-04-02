from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from src.debug_log import log_debug
from src.moe_router import topk_router

RUN_ID = os.getenv("DEBUG_RUN_ID", "pre-fix")


class DeepSeekMoE(torch.nn.Module):
    """
    Minimal DeepSeek-style top-k MoE layer:
    - A router picks top-k experts per token
    - Output is weighted sum of selected expert linear projections
    """

    def __init__(self, d_model: int, d_hidden: int, num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k

        self.router_weight = torch.nn.Parameter(torch.randn(num_experts, d_model) * 0.02)
        self.expert_weight = torch.nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        indices, weights = topk_router(x, self.router_weight, self.top_k)
        out = torch.zeros(x.shape[0], self.d_hidden, device=x.device, dtype=x.dtype)

        for token in range(x.shape[0]):
            for slot in range(self.top_k):
                expert_id = int(indices[token, slot])
                gate = weights[token, slot]
                out[token] += gate * (x[token] @ self.expert_weight[expert_id])
        return out

    def forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """
        Grouped GEMM style computation:
        flatten selected expert-token pairs and run batched matmul.
        This is the path that maps well to TK WMMA/TMA kernels.
        """
        logits = x @ self.router_weight.T
        top_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_logits, dim=-1)

        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H1",
            location="src/deepseek_moe.py:forward_grouped",
            message="router outputs",
            data={
                "x_shape": list(x.shape),
                "indices_shape": list(indices.shape),
                "weights_shape": list(weights.shape),
            },
        )
        # endregion

        token_ids = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, self.top_k)
        flat_tokens = token_ids.reshape(-1)
        flat_experts = indices.reshape(-1)
        flat_gates = weights.reshape(-1, 1)

        gathered_x = x.index_select(0, flat_tokens)
        gathered_w = self.expert_weight.index_select(0, flat_experts)
        projected = torch.bmm(gathered_x.unsqueeze(1), gathered_w).squeeze(1)
        weighted = projected * flat_gates

        out = torch.zeros(x.shape[0], self.d_hidden, device=x.device, dtype=x.dtype)
        out.index_add_(0, flat_tokens, weighted)

        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H2",
            location="src/deepseek_moe.py:forward_grouped",
            message="grouped path result stats",
            data={
                "projected_shape": list(projected.shape),
                "out_shape": list(out.shape),
                "out_mean": float(out.mean().item()),
            },
        )
        # endregion

        return out
