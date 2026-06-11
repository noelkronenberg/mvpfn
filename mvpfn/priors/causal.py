"""
Confounded linear Structural Causal Model (SCM) prior for Do-PFN training and a fixed SCM for comparison.

References:
- Robertson et al., 2025
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _sample_linear_scm(
    seq_len: int,
    batch: int,
    n_cov: int,
    noise: float,
    ate: torch.Tensor,
    w_t: torch.Tensor,
    w_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample observational and interventional rows from a linear SCM.
    
    Args:
        seq_len (int): Points per dataset.
        batch (int): Datasets per batch.
        n_cov (int): Number of covariates.
        noise (float): Standard deviation of outcome noise.
        ate (torch.Tensor): Average treatment effect.
        w_t (torch.Tensor): Confounding weights.
        w_y (torch.Tensor): Outcome weights.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Observational and interventional inputs and targets.
    """

    x_cov = 0.5 * torch.randn(seq_len, batch, n_cov) # covariates

    # observational data
    logit_t = (x_cov * w_t).sum(-1, keepdim=True) # logit of treatment probability given covariates
    t_obs = (torch.sigmoid(logit_t) > torch.rand(seq_len, batch, 1)).float() # binary treatment
    y_obs = (ate * t_obs).squeeze(-1) + (x_cov * w_y).sum(-1) + noise * torch.randn(seq_len, batch) # outcome as function of treatment and covariates

    # interventional data
    t_int = torch.randint(0, 2, (seq_len, batch, 1)).float() # random binary treatment
    y_int = (ate * t_int).squeeze(-1) + (x_cov * w_y).sum(-1) + noise * torch.randn(seq_len, batch) # outcome as function of treatment and covariates

    # concatenate treatment and covariates
    x_obs = torch.cat([t_obs, x_cov], dim=-1)
    x_int = torch.cat([t_int, x_cov], dim=-1)

    return x_obs, y_obs, x_int, y_int


def scm_prior(
    seq_len: int,
    batch: int,
    num_features: int = 3,
    noise: float = 0.05,
    confound_scale: float = 1.5,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build a training sampler from a confounded linear SCM [Robertson et al., 2025, Figure 1].

    Args:
        seq_len (int): Points per dataset.
        batch (int): Datasets per batch.
        num_features (int): Treatment + covariates.
        noise (float): Standard deviation of outcome noise.
        confound_scale (float): Scale of covariate weights in treatment and outcome.

    Returns:
        Callable: Zero-arg sampler returning ``(x_obs, y_obs, x_int, y_int)`` 
            with shapes ``(seq_len, batch, num_features)`` and ``(seq_len, batch)``.
    """

    if num_features < 2:
        raise ValueError("num_features must be at least 2 (treatment + one covariate).")

    n_cov = num_features - 1 # number of covariates (without treatment)

    def get_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_t = confound_scale * torch.randn(1, batch, n_cov) # weights for treatment
        w_y = confound_scale * torch.randn(1, batch, n_cov) # weights for outcome
        ate = torch.empty(1, batch, 1).uniform_(0, 1.0) # causal effect of t on y
        
        return _sample_linear_scm(seq_len, batch, n_cov, noise, ate, w_t, w_y) # sample from the SCM

    return get_batch


def eval_scm(
    n: int,
    num_features: int = 3,
    noise: float = 0.05,
    ate: float = 0.7,
    confound_scale: float = 1.5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Fixed SCM for evaluation (one dataset, known ATE).

    Args:
        n (int): Number of points in the dataset.
        num_features (int): ``1`` treatment + covariates.
        noise (float): Outcome noise standard deviation.
        ate (float): Known average treatment effect.
        confound_scale (float): Scale of covariate weights in treatment and outcome.
        seed (int): RNG seed for covariates and SCM weights.

    Returns:
        tuple: ``(x_obs, y_obs, x_query, ate)`` where tensors have shapes ``(n, 1, num_features)``, ``(n, 1)``, ``(n, 1, num_features)``,
            and ``ate`` is the scalar ground-truth CATE.
    """

    if num_features < 2:
        raise ValueError("num_features must be at least 2 (treatment + one covariate).")

    torch.manual_seed(seed)

    n_cov = num_features - 1
    
    ate_t = torch.tensor([[[ate]]]) # average treatment effect
    w_t = confound_scale * torch.randn(1, 1, n_cov) # weights for treatment
    w_y = confound_scale * torch.randn(1, 1, n_cov) # weights for outcome

    x_obs, y_obs, _, _ = _sample_linear_scm(n, 1, n_cov, noise, ate_t, w_t, w_y) # sample from the SCM
    x_query = torch.cat(
        [
            torch.zeros(n, 1, 1), # placeholder, treatment column is overwritten at predict time
            x_obs[..., 1:] # covariates
        ], 
        dim=-1 
    )
    
    return x_obs, y_obs, x_query, ate
