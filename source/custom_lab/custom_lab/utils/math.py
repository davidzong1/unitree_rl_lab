from ast import Module
import torch


REWARD_TRACKING_SIGMA = 0.25


def neg_exp(x, scale: float = 1.0):
    """shorthand helper for negative exponential e^(-x/scale)
    scale: range of x
    """
    return torch.exp(-(x / scale) / REWARD_TRACKING_SIGMA)


def negsqrd_exp(x, scale: float = 1.0):
    """shorthand helper for negative squared exponential e^(-(x/scale)^2)
    scale: range of x
    """
    return torch.exp(-torch.square(x / scale) / REWARD_TRACKING_SIGMA)


def ne_sigmodial(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Sigmoidal reward function that maps x to [0, 1] -inf -> 1, 0 --> 1, inf --> 0"""
    return (2.0 / (1.0 + torch.exp(scale * x / REWARD_TRACKING_SIGMA))).clamp(
        min=0.0, max=1.0
    )


def p_clip_sigmodial(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Sigmoidal reward function that maps x to [0, 1]  -inf -> 0, 0 --> 0, inf --> 1"""
    return 2 * (
        (-1 / (1.0 + torch.exp(x * scale / REWARD_TRACKING_SIGMA))) + 0.5
    ).clamp(min=0.0, max=1.0)
