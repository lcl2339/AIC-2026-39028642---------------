import torch
def neighbors(query, keys, k=32, query_folds=None, key_folds=None, batch=256):
    similarities, indices = ([], [])
    for start in range(0, len(query), batch):
        similarity = query[start:start + batch].half() @ keys.half().T
        if query_folds is not None:
            excluded = query_folds[start:start + batch, None] == key_folds[None]
            similarity.masked_fill_(excluded, -torch.inf)
        value, index = similarity.topk(min(k, len(keys)), dim=-1)
        assert torch.isfinite(value).all(), 'Too few independent support groups'
        similarities.append(value.float().cpu())
        indices.append(index.cpu())
    return (torch.cat(similarities), torch.cat(indices))

def label_distribution(similarity, index, labels, weights, beta, classes=750):
    support_labels = labels[index]
    mass = ((similarity - similarity.max(-1, keepdim=True).values) * beta).exp()
    mass = mass * weights[index]
    probability = torch.zeros(len(index), classes, device=index.device)
    probability.scatter_add_(1, support_labels, mass)
    return probability / probability.sum(-1, keepdim=True).clamp_min(1e-12)
