from src.moe_reference import reference_moe
import torch

def test_moe():
    out_ref, out_grouped = reference_moe()
    assert out_ref.shape == out_grouped.shape
    assert torch.allclose(out_ref, out_grouped, atol=1e-5, rtol=1e-5)
