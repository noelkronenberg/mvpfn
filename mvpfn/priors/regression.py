"""
Linear-function prior for PFN training.
"""

from __future__ import annotations

import torch

from mvpfn.train import GetBatch


def regression_prior(
    seq_len: int,
    batch: int,
    num_features: int = 1,
    noise: float = 0.05,
) -> GetBatch:
    """
    Build a sampler of random linear functions.

    Args:
        seq_len (int): Points per dataset.
        batch (int): Datasets per batch.
        num_features (int): Input dimensionality.
        noise (float): Standard deviation of additive observation noise.

    Returns:
        GetBatch: Zero-arg sampler returning ``(x, y)`` 
            of shape ``(seq_len, batch, num_features)`` and ``(seq_len, batch)``.
    """

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(seq_len, batch, num_features) * 2 - 1 # inputs in [-1, 1]
        w = 0.5 * torch.randn(1, batch, num_features) # slope per dataset
        b = 0.5 * torch.randn(1, batch) # intercept per dataset
        y = (x * w).sum(-1) + b + noise * torch.randn(seq_len, batch) # linear + noise
        return x, y

    return get_batch
