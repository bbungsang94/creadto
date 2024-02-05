import torch


def l2_distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum(2)).mean(1).mean()