import torch


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    print(type(idx))
    print(idx.shape)
    print(idx)
    onehot.scatter_(1, idx, 1)
    
    return onehot
