"""
Gaussian-process prior (RBF kernel) for PFN training and an exact GP posterior for comparison.

References:
- https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/priors/fast_gp.py
"""

from __future__ import annotations

import gpytorch
import torch

from mvpfn.train import GetBatch


def _rbf_kernel(lengthscale: float) -> gpytorch.kernels.Kernel:
    """
    RBF kernel with a fixed length scale.

    Args:
        lengthscale (float): RBF kernel length scale.

    Returns:
        gpytorch.kernels.Kernel: RBF kernel with the given length scale.
    """

    kernel = gpytorch.kernels.RBFKernel() # function for how similar two points are to each other, as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/priors/fast_gp.py#L15]
    kernel.lengthscale = lengthscale # how far apart two points need to be to be considered different
    
    return kernel


def gp_prior(
    seq_len: int,
    batch: int,
    num_features: int = 1,
    noise: float = 0.05,
    lengthscale: float = 0.5,
) -> GetBatch:
    """
    Build a sampler of functions drawn from a GP prior (RBF kernel).

    Args:
        seq_len (int): Points per dataset.
        batch (int): Datasets per batch.
        num_features (int): Input dimensionality.
        noise (float): Standard deviation of additive observation noise.
        lengthscale (float): RBF kernel length scale.

    Returns:
        GetBatch: Zero-arg sampler returning ``(x, y)`` 
            of shape ``(seq_len, batch, num_features)`` and ``(seq_len, batch)``.
    """

    kernel = _rbf_kernel(lengthscale)

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(seq_len, batch, num_features) * 2 - 1 # inputs in [-1, 1]
        xb = x.transpose(0, 1) # (seq_len, batch, num_features) -> (batch, seq_len, num_features), one dataset per row
        mean = torch.zeros(batch, seq_len) # zero-mean GP
        
        f = gpytorch.distributions.MultivariateNormal(mean, kernel(xb)).rsample() # sample from the GP prior, as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/priors/fast_gp.py#L22]
        y = f.transpose(0, 1) + noise * torch.randn(seq_len, batch) # add observation noise (with seq-first shape the PFN expects)
        
        return x, y

    return get_batch


class _ExactGP(gpytorch.models.ExactGP):
    """
    Minimal zero-mean RBF GP for exact posterior inference.
    
    Args:
        train_x (torch.Tensor): Training inputs of shape ``(n_ctx,)``.
        train_y (torch.Tensor): Training targets of shape ``(n_ctx,)``.
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood function.
        lengthscale (float): RBF kernel length scale.

    Returns:
        gpytorch.distributions.MultivariateNormal: Posterior distribution.
    """

    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ZeroMean() # zero-mean GP
        self.covar_module = _rbf_kernel(lengthscale) # RBF kernel with the given length scale

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x)) # sample from the GP prior, as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/priors/fast_gp.py#L22]


def gp_predict(
    xc: torch.Tensor,
    yc: torch.Tensor,
    xs: torch.Tensor,
    lengthscale: float = 0.5,
    noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact GP posterior (RBF kernel) at query points, for comparison against the PFN.

    Args:
        xc (torch.Tensor): Context inputs of shape ``(n_ctx,)``.
        yc (torch.Tensor): Context targets of shape ``(n_ctx,)``.
        xs (torch.Tensor): Query inputs of shape ``(n_query,)``.
        lengthscale (float): RBF kernel length scale (match the trained prior).
        noise (float): Observation noise standard deviation (match the trained prior).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Posterior ``(mean, std)``, each of shape ``(n_query,)``.
    """

    likelihood = gpytorch.likelihoods.GaussianLikelihood() # add measurement noise model, as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/priors/fast_gp.py#L26]
    likelihood.noise = noise ** 2 # set the noise variance (converted from standard deviation)
    
    model = _ExactGP(
        xc[:, None], # (n_ctx,) -> (n_ctx, 1), add feature dimension
        yc, 
        likelihood, 
        lengthscale
    ).eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model( # conditioned on context (above), evaluate on query
            xs[:, None] # (n_query,) -> (n_query, 1), add feature dimension
        ) 
    
    return post.mean, post.stddev
