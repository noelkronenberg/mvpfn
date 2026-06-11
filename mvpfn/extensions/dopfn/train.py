"""
Prior-data fitting for the Do-PFN training loop.

References:
- Robertson et al., 2025
- https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/train.py
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch
from torch import nn

from mvpfn.extensions.dopfn.model import DoPFN
from mvpfn.train import _batch_loss


# type alias for the causal prior sampler
GetDoBatch = Callable[
    [], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


def train_dopfn(
    model: DoPFN,
    criterion: nn.Module,
    get_batch: GetDoBatch,
    *,
    steps: int = 1000,
    lr: float = 1e-4,
    n_train_sampler: Callable[[int], int] | None = None,
    grad_clip: float = 1.0, # as per original PFN implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/train.py#L95]
    device: str | torch.device = "cpu",
    log_every: int = 100,
) -> DoPFN:
    """
    Train a Do-PFN on datasets drawn from a causal prior.

    Args:
        model (DoPFN): The Do-PFN to train.
        criterion (nn.Module): Loss defining the task (typically ``BarDistribution``).
        get_batch (GetDoBatch): Prior sampler returning one batch ``(x_obs, y_obs, x_int, y_int)`` 
            of shape ``(seq_len, batch, num_features)`` and ``(seq_len, batch)``.
        steps (int): Number of optimisation steps (dataset batches).
        lr (float): Adam learning rate.
        n_train_sampler (Callable[[int], int] | None): Maps ``seq_len`` to ``n_train`` (training points); 
            the rest are query points. Defaults to a uniform draw in ``[1, seq_len - 1]``.
        grad_clip (float): Max gradient norm for clipping.
        device (str | torch.device): Device to train on.
        log_every (int): Steps between average-loss logs (``0`` to disable).

    Returns:
        DoPFN: The trained model (also trained in place).
    """

    if n_train_sampler is None:
        n_train_sampler = lambda seq_len: random.randint(1, seq_len - 1) # uniform draw

    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # as per original implementation [Robertson et al., 2025, p. 20]

    running_loss = 0.0
    for step in range(1, steps + 1):

        # draw a batch of datasets from the prior and move to device
        x_obs, y_obs, x_int, y_int = get_batch()
        x_obs, y_obs = x_obs.to(device), y_obs.to(device) # (seq_len, batch, num_features), (seq_len, batch)
        x_int, y_int = x_int.to(device), y_int.to(device) # (seq_len, batch, num_features), (seq_len, batch)

        n_train = n_train_sampler(x_obs.shape[0])

        # forward pass
        logits = model(
            x_obs[:n_train], y_obs[:n_train], 
            x_int[n_train:] # interventional query [Robertson et al., 2025, Figure 1]
        ) # (n_test, batch, n_out)

        # compute loss
        loss = _batch_loss(
            criterion, 
            logits, 
            y_int[n_train:] # interventional target [Robertson et al., 2025, Figure 1]
        )

        # reset gradients
        optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # update parameters
        optimizer.step()

        running_loss += loss.item()
        if log_every and step % log_every == 0:
            print(f"step {step}/{steps} | loss {running_loss / log_every:.4f}")
            running_loss = 0.0

    return model
