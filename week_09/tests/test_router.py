import torch
from src.moe_router import topk_router

def test_router():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    W = torch.randn(6, 8)
    k = 2
    idx, w = topk_router(x, W, k)
    assert idx.shape == (4, k)
    assert w.shape == (4, k)
    assert torch.all((idx >= 0) & (idx < 6))
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
