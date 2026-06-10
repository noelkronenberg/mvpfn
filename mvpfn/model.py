"""
Prior-data Fitted Network (PFN).

References:
- Müller et al., 2021
- https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/transformer.py
"""

from __future__ import annotations

import torch
from torch import nn


def pfn_attention_mask(
    seq_len: int, n_train: int, device: torch.device
) -> torch.Tensor:
    """
    Build the PFN self-attention mask.

    Every position (train and test) may attend to the training points (the first ``n_train`` positions) [Müller et al., 2021, Figure 2a] 
    and to itself [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/transformer.py#L39].

    Args:
        seq_len (int): Total sequence length (train points + test points).
        n_train (int): Number of leading training points.
        device (torch.device): Device on which to allocate the mask.

    Returns:
        torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)``
            where ``True`` marks a (query, key) pair that is *not* allowed to attend.
    """

    not_allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device) # (seq, seq)
    not_allowed[:, :n_train] = False
    not_allowed.fill_diagonal_(False) # as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/transformer.py#L39]

    return not_allowed


class PFN(nn.Module):
    """
    Prior-data Fitted Network (PFN) with linear input encoders, masked encoder-only Transformer, and generic decoder head.
    """

    def __init__(
        self,
        num_features: int,
        n_out: int,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_layers: int = 6,
    ) -> None:
        """
        Initialise the PFN.

        Args:
            num_features (int): Dimensionality of each input ``x``.
            n_out (int): Number of output logits per test point.
            d_model (int): Transformer embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Hidden dimension of the feed-forward sublayers.
            dropout (float): Dropout probability inside the Transformer.
            num_layers (int): Number of Transformer encoder layers.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``nhead``.
        """

        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        self.num_features = num_features
        self.n_out = n_out

        # linear input encoders [Müller et al., 2021, p. 5]
        self.x_encoder = nn.Linear(num_features, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        # Transformer encoder without positional encodings (invariant to permutations) [Müller et al., 2021, p. 5]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu", # as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/transformer.py#L17]
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # generic decoder head, as per original implementation [https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/transformer.py#L23]
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, n_out),
        )

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict per-test-point logits for ``p(y | x, D)`` [Müller et al., 2021, Figure 1].

        Args:
            x_train (torch.Tensor): Training inputs of shape ``(n_train, batch, num_features)``.
            y_train (torch.Tensor): Training targets of shape ``(n_train, batch)``.
            x_test (torch.Tensor): Test inputs of shape ``(n_test, batch, num_features)``.

        Returns:
            torch.Tensor: Logits of shape ``(n_test, batch, n_out)``.

        Raises:
            ValueError: If the batch sizes of the three inputs do not match.
        """

        if not (x_train.shape[1] == y_train.shape[1] == x_test.shape[1]):
            raise ValueError("x_train, y_train and x_test must have the same batch size.")

        n_train = x_train.shape[0]

        x = torch.cat([x_train, x_test], dim=0) # (seq, batch, num_features)

        # encode inputs (train and test) and targets (train only)
        x_embeddings = self.x_encoder(x) # (seq, batch, d_model)
        y_embeddings = self.y_encoder(
            y_train.unsqueeze(-1) # (n_train, batch) -> (n_train, batch, 1), as nn.Linear acts on last dimension (i.e., the scalar target feature)
        ) # (n_train, batch, d_model)

        embeddings = torch.cat(
            [
                x_embeddings[:n_train] + y_embeddings, # train rows (n_train, batch, d_model)
                x_embeddings[n_train:], # test rows (n_test, batch, d_model)
            ],
            dim=0,
        ) # (seq, batch, d_model)

        mask = pfn_attention_mask(x.shape[0], n_train, x.device) # (seq, seq)
        encoded = self.transformer(embeddings, mask=mask) # (seq, batch, d_model)

        return self.decoder(encoded[n_train:]) # (n_test, batch, n_out)