import torch

from src.deepseek_moe import DeepSeekMoE


def reference_moe(seed: int = 42):
    torch.manual_seed(seed)
    model = DeepSeekMoE(d_model=8, d_hidden=16, num_experts=4, top_k=2)
    x = torch.randn(4, 8)
    out_ref = model.forward_reference(x)
    out_grouped = model.forward_grouped(x)
    return out_ref, out_grouped
