"""
Do-PFN extension: Prior-data Fitted Network (PFN) with observational context and interventional queries.

References:
- Robertson et al., 2025
- https://github.com/jr2021/Do-PFN/blob/90d67433b43c4d52d752dc336070f525ff856e0b/scripts/transformer_prediction_interface/base.py
"""

from __future__ import annotations

import torch

from mvpfn.bar_distribution import BarDistribution
from mvpfn.model import PFN


class DoPFN(PFN):
    """
    Prior-data Fitted Network (PFN) for conditional interventional distributions (CIDs) ``p(y | do(t), x, D)`` [Robertson et al., 2025, Algorithm 1].
    """

    def forward(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        x_int: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict per-query-point logits for the CID ``p(y | do(t), x, D)`` [Robertson et al., 2025, Algorithm 1].

        Args:
            x_obs (torch.Tensor): Observational context inputs of shape ``(n_train, batch, num_features)``.
            y_obs (torch.Tensor): Observational context targets of shape ``(n_train, batch)``.
            x_int (torch.Tensor): Interventional query inputs of shape ``(n_query, batch, num_features)``.

        Returns:
            torch.Tensor: Logits of shape ``(n_query, batch, n_out)``.

        Raises:
            ValueError: If the batch sizes of the three inputs do not match.
        """

        return super().forward(x_obs, y_obs, x_int)

    @torch.no_grad()
    def predict_cid(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        x_query: torch.Tensor,
        t: float,
        criterion: BarDistribution,
    ) -> torch.Tensor:
        """
        Conditional interventional distribution mean at treatment ``t``.

        Args:
            x_obs (torch.Tensor): Observational context inputs of shape ``(n_train, batch, num_features)``.
            y_obs (torch.Tensor): Observational context targets of shape ``(n_train, batch)``.
            x_query (torch.Tensor): Query inputs of shape ``(n_query, batch, num_features)``
                (treatment column is overwritten by ``t``).
            t (float): Interventional treatment value (``0`` or ``1``).
            criterion (BarDistribution): Output distribution head.

        Returns:
            torch.Tensor: Predictive means of shape ``(n_query, batch)``.
        """

        x_int = x_query.clone()
        x_int[..., 0] = t # overwrite treatment column, as per original implementation [https://github.com/jr2021/Do-PFN/blob/90d67433b43c4d52d752dc336070f525ff856e0b/scripts/transformer_prediction_interface/base.py#L1962] 
        logits = self.forward(x_obs, y_obs, x_int)

        return criterion.mean(logits) # collapse to mean, as per original implementation [https://github.com/jr2021/Do-PFN/blob/90d67433b43c4d52d752dc336070f525ff856e0b/scripts/transformer_prediction_interface/base.py#L1882]

    @torch.no_grad()
    def predict_cate(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        x_query: torch.Tensor,
        criterion: BarDistribution,
    ) -> torch.Tensor:
        """
        Conditional average treatment effect (CATE) ``E[y | do(t=1), x, D] - E[y | do(t=0), x, D]`` [Robertson et al., 2025, p. 3].

        Args:
            x_obs (torch.Tensor): Observational context inputs of shape ``(n_train, batch, num_features)``.
            y_obs (torch.Tensor): Observational context targets of shape ``(n_train, batch)``.
            x_query (torch.Tensor): Query inputs of shape ``(n_query, batch, num_features)``
                (treatment column is overwritten).
            criterion (BarDistribution): Output distribution head.

        Returns:
            torch.Tensor: CATE of shape ``(n_query, batch)``.
        """

        # predict CIDs at treatment values 1 and 0, as per original implementation [https://github.com/jr2021/Do-PFN/blob/90d67433b43c4d52d752dc336070f525ff856e0b/scripts/transformer_prediction_interface/base.py#L1969]
        y_1 = self.predict_cid(x_obs, y_obs, x_query, 1.0, criterion)
        y_0 = self.predict_cid(x_obs, y_obs, x_query, 0.0, criterion)

        return y_1 - y_0 # calculate CATE, as per original implementation [https://github.com/jr2021/Do-PFN/blob/90d67433b43c4d52d752dc336070f525ff856e0b/scripts/transformer_prediction_interface/base.py#L1972]
