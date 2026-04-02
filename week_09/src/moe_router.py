import torch

def topk_router(x, W, k):
    logits = x @ W.T
    weights, indices = torch.topk(logits, k, dim=-1)
    weights = torch.softmax(weights, dim=-1)
    return indices, weights
