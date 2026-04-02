import torch
from src.tk_backend import tk_forward_or_fallback

def test_expert():
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    W = torch.randn(32, 64)
    ref = x @ W
    out = tk_forward_or_fallback(x, W)
    assert ref.shape == (16, 64)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
