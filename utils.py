# utils.py
import torch

def sdr_loss(estimate, target, eps=1e-8):
    """Calculate the SDR (Signal-to-Distortion Ratio) loss.
    
    Arguments:
    estimate -- Estimated source signals
    target -- Target (ground truth) source signals
    
    Returns:
    SDR loss value
    """
    numerator = torch.sum(target ** 2, dim=(-1, -2))
    denominator = torch.sum((target - estimate) ** 2, dim=(-1, -2))
    sdr = -10 * torch.log10((numerator / (denominator + eps)) + eps)
    return torch.mean(sdr)
