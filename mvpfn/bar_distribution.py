"""
Riemann (bar) distribution used as the PFN regression head.

References:
- Müller et al., 2021
- https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/bar_distribution.py
"""

from __future__ import annotations

import torch
from torch import nn


class BarDistribution(nn.Module):
    """
    Discretised distribution over a fixed set of bucket borders (Riemann distribution) [Müller et al., 2021, p. 5].
    """

    def __init__(self, borders: torch.Tensor) -> None:
        """
        Initialise the bar distribution from its bucket borders.

        Args:
            borders (torch.Tensor): Sorted 1D tensor of bucket borders of shape ``(num_bars + 1,)``, 
                starting at the support minimum and ending at the support maximum.

        Raises:
            ValueError: If ``borders`` is not a sorted 1D tensor with at least two entries.
        """

        super().__init__()

        if borders.ndim != 1 or borders.numel() < 2:
            raise ValueError("borders must be a 1D tensor with at least 2 entries.")

        if (borders[1:] - borders[:-1] <= 0).any():
            raise ValueError("borders must be strictly increasing.")

        self.register_buffer( # moves with .to(device) and state_dict (not a model parameter)
            "borders", 
            borders.contiguous() # store in single block of memory for reliable indexing
        )
        
        self.register_buffer(
            "bucket_widths",
            borders[1:] - borders[:-1],
        )

        self.num_bars = len(borders) - 1

    def _bucket_of(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map target values to the index of the bucket that contains them [Müller et al., 2021, p. 18].

        Args:
            y (torch.Tensor): Target values of any shape.

        Returns:
            torch.Tensor: Long tensor of bucket indices in ``[0, num_bars - 1]``.
        """

        idx = torch.searchsorted(self.borders, y) - 1 # index of the bucket that contains the target
        
        idx[y == self.borders[0]] = 0 # y at lower support bound -> first bucket
        idx[y == self.borders[-1]] = self.num_bars - 1 # y at upper support bound -> last bucket
        
        return idx

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-density of targets under the predicted bars [Müller et al., 2021, Equation (2)].

        Args:
            logits (torch.Tensor): Bucket logits of shape ``(*batch, num_bars)``.
            y (torch.Tensor): Target values of shape ``(*batch,)``.

        Returns:
            torch.Tensor: Per-element negative log-density of shape ``(*batch,)``.

        Raises:
            ValueError: If ``logits`` or ``y`` have the wrong shape, or if any
                target lies outside the bucket support.
        """

        if logits.shape[-1] != self.num_bars:
            raise ValueError(
                f"Expected last dim {self.num_bars}, got {logits.shape[-1]}."
            )
        if y.shape != logits.shape[:-1]:
            raise ValueError(
                f"y shape {y.shape} must match logits shape {logits.shape[:-1]}."
            )

        # Riemann density per bucket [Müller et al., 2021, Equation (29)], in log space [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/bar_distribution.py#L31]:
        log_bucket_prob = torch.log_softmax(logits, dim=-1) # bucket probability mass (log p_b)
        log_bucket_widths = torch.log(self.bucket_widths) # bucket width on y-axis (log w(b))
        log_density = log_bucket_prob - log_bucket_widths # Riemann density per bucket (bucket probability mass normalized by bucket width, log p_b - log w(b))

        target_bucket = self._bucket_of(y) # index of the bucket that contains the target

        if (target_bucket < 0).any() or (target_bucket >= self.num_bars).any():
            raise ValueError(
                f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
            )

        nll = -log_density.gather( # gather the log density of the target bucket, (n_test, batch, num_bars) -> (n_test, batch, 1)
            -1, # gather along last dimension
            target_bucket.unsqueeze(-1) # (n_test, batch) -> (n_test, batch, 1), as gather requires rank to match
        ) \
        .squeeze(-1) # (n_test, batch, 1) -> (n_test, batch), squeeze out the last dimension again
        
        return nll

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the expected value of the predicted distribution.

        Args:
            logits (torch.Tensor): Bucket logits of shape ``(*batch, num_bars)``.

        Returns:
            torch.Tensor: Distribution means of shape ``(*batch,)``.
        """

        bucket_centers = (self.borders[:-1] + self.borders[1:]) / 2 # center of each bucket
        mean =  torch.softmax(logits, dim=-1) @ bucket_centers # probability mass * bucket center

        return mean

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the mode (densest bucket center) of the predicted distribution.

        Args:
            logits (torch.Tensor): Bucket logits of shape ``(*batch, num_bars)``.

        Returns:
            torch.Tensor: Distribution modes of shape ``(*batch,)``.
        """

        density = torch.softmax(logits, dim=-1) / self.bucket_widths # density per bucket
        bucket_centers = (self.borders[:-1] + self.borders[1:]) / 2 # center of each bucket
        mode = bucket_centers[density.argmax(dim=-1)] # highest density bucket center

        return mode
