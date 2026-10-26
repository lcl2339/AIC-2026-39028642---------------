import numpy as np
import torch
def final_balance(q, temp, strength, iterations):
    if temp is None:
        return q
    target = q.mean(0).clamp_min(1e-12).pow(1 - strength)
    target /= target.sum()
    p = q.clamp_min(1e-12).pow(1 / temp)
    for _ in range(iterations):
        p *= target[None] / p.mean(0, keepdim=True).clamp_min(1e-30)
        p /= p.sum(-1, keepdim=True).clamp_min(1e-30)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(-1), torch.ones(len(p), device=p.device), atol=1e-05)
    return p

def floor_propagate(seed, index, weight, alpha, iterations, floor):
    assert 0 <= alpha <= 1 and 0 <= floor <= 1
    coefficient = (alpha * (floor + (1 - floor) * (1 - seed.max(-1).values)))[:, None]
    q = seed
    for _ in range(iterations):
        updated = torch.empty_like(q)
        for start in range(0, len(q), 512):
            ids = index[start:start + 512]
            w = weight[start:start + 512]
            total = w.sum(-1, keepdim=True)
            mean = (q[ids] * w[..., None]).sum(1) / total.clamp_min(1e-12)
            updated[start:start + 512] = torch.where(total > 0, mean, seed[start:start + 512])
        q = (1 - coefficient) * seed + coefficient * updated
    assert torch.isfinite(q).all() and torch.allclose(q.sum(-1), torch.ones(len(q), device=q.device), atol=1e-05)
    return q

def balance(probabilities, epsilon):
    k = probabilities.double().clamp_min(1e-12).log()
    k = (k - k.mean()) / (k.std(correction=0).clamp_min(1e-08) * epsilon)
    u = torch.zeros(len(k), device=k.device, dtype=k.dtype)
    v = torch.zeros(k.shape[1], device=k.device, dtype=k.dtype)
    for _ in range(100):
        v = np.log(len(k) / k.shape[1]) - torch.logsumexp(k + u[:, None], 0)
        u = -torch.logsumexp(k + v[None], 1)
    return (k + u[:, None] + v[None]).exp().float()

def affinities(seed, index):
    root = seed.clamp_min(0).sqrt()
    out = torch.empty(index.shape, device=seed.device)
    for start in range(0, len(seed), 512):
        out[start:start + 512] = (root[start:start + 512, None] * root[index[start:start + 512]]).sum(-1)
    return out.clamp(0, 1)
