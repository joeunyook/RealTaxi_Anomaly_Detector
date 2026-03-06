"""
Transformer-based anomaly classifier for time-series windows.

Architecture:
  - Positional encoding (sinusoidal)
  - Multi-head self-attention encoder (TransformerEncoder)
  - CLS-token pooling -> classification head

References:
  Vaswani et al. (2017) "Attention Is All You Need"
  Zhou et al. (2025) "Transformer-based anomaly detection for time-series"
  https://arxiv.org/abs/2504.04011
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerAnomalyClassifier(nn.Module):
    """
    Transformer encoder for binary anomaly classification on sliding windows.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (value + optional time features).
    d_model : int
        Internal embedding dimension.
    nhead : int
        Number of attention heads (must divide d_model evenly).
    num_layers : int
        Number of TransformerEncoder layers.
    dim_feedforward : int
        Hidden size of the feedforward sublayer.
    dropout : float
        Dropout probability applied throughout.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, W, input_dim)

        Returns
        -------
        logits : torch.Tensor of shape (B,)
        """
        # Project to d_model
        x = self.input_proj(x)          # (B, W, d_model)
        x = self.pos_enc(x)             # add positional encoding
        x = self.encoder(x)             # (B, W, d_model)
        # Mean pooling over time dimension
        h = x.mean(dim=1)               # (B, d_model)
        logits = self.head(h).squeeze(-1)  # (B,)
        return logits
