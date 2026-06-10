"""
Prior-data fitting for the PFN training loop.

References:
- Müller et al., 2021
- https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/train.py
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch
from torch import nn

from mvpfn.model import PFN
from mvpfn.bar_distribution import BarDistribution


# type alias for the prior sampler
GetBatch = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def _batch_loss(
    criterion: nn.Module, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Apply the criterion to the query predictions [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/train.py#L14].

    Args:
        criterion (nn.Module): Loss function or output distribution 
            (``BarDistribution`` for regression, ``nn.CrossEntropyLoss`` for classification, ``nn.BCEWithLogitsLoss`` for binary classification).
        logits (torch.Tensor): Query predictions of shape ``(n_test, batch, n_out)``.
        targets (torch.Tensor): Query targets of shape ``(n_test, batch)``.

    Returns:
        torch.Tensor: Scalar mean loss over all query points.
    """

    # CASE: regression
    if isinstance(criterion, BarDistribution):
        losses = criterion(logits.reshape(-1, logits.shape[-1]), targets.flatten()) # (n_test, batch, n_out) -> (n_test * batch, n_out)

    # CASE: classification
    elif isinstance(criterion, nn.CrossEntropyLoss):
        losses = criterion(logits.reshape(-1, logits.shape[-1]), targets.flatten().long()) # (n_test, batch, n_out) -> (n_test * batch, n_out)

    # CASE: binary classification
    elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.MSELoss)):
        losses = criterion(logits.flatten(), targets.flatten()) # (n_test, batch, n_out) -> (n_test * batch)

    else:
        raise ValueError(f"Unsupported criterion: {criterion}")
    
    # mean loss over all query points
    return losses.mean()


def train(
    model: PFN,
    criterion: nn.Module,
    get_batch: GetBatch,
    *,
    steps: int = 1000,
    lr: float = 1e-4,
    n_train_sampler: Callable[[int], int] | None = None,
    grad_clip: float = 1.0, # as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/train.py#L95]
    device: str | torch.device = "cpu",
    log_every: int = 100,
) -> PFN:
    """
    Train a PFN on datasets drawn from a prior.

    Args:
        model (PFN): The PFN to train.
        criterion (nn.Module): Loss defining the task
            (``BarDistribution`` for regression, ``nn.CrossEntropyLoss`` for classification, ``nn.BCEWithLogitsLoss`` for binary classification).
        get_batch (GetBatch): Prior sampler returning one batch ``(x, y)`` of shape ``(seq_len, batch, num_features)`` and ``(seq_len, batch)``, respectively.
        steps (int): Number of optimisation steps (datasets batches).
        lr (float): Adam learning rate.
        n_train_sampler (Callable[[int], int] | None): Maps ``seq_len`` to ``n_train`` (training
            points); the rest are query points. Defaults to a uniform draw in ``[1, seq_len - 1]``.
        grad_clip (float): Max gradient norm for clipping.
        device (str | torch.device): Device to train on.
        log_every (int): Steps between average-loss logs (``0`` to disable).

    Returns:
        PFN: The trained model (also trained in place).
    """

    if n_train_sampler is None:
        n_train_sampler = lambda seq_len: random.randint(1, seq_len - 1) # uniform draw

    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/train.py#L55]

    running_loss = 0.0
    for step in range(1, steps + 1):

        # draw a batch of datasets from the prior and move to device
        x, y = get_batch()
        x, y = x.to(device), y.to(device) # (seq_len, batch, num_features), (seq_len, batch)

        n_train = n_train_sampler(x.shape[0])

        # forward pass
        logits = model(x[:n_train], y[:n_train], x[n_train:]) # (n_test, batch, n_out)
        
        # compute loss
        loss = _batch_loss(criterion, logits, y[n_train:]) 

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
