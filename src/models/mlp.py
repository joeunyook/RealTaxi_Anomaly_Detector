import torch
import torch.nn as nn


class MLPSeasonalBaseline(nn.Module):
    """
    Pointwise MLP: f(sin_hour, cos_hour, sin_dow, cos_dow) → expected_demand.

    Unlike the GRU, there is no recurrence — each timestep is predicted
    independently. This is the architecturally correct choice when the input
    is purely time features: since sin/cos encodings at t carry no information
    about t-1, a recurrent hidden state adds no value. The MLP learns the
    same seasonal mapping as the GRU without wasted sequential capacity.

    Used as a baseline comparison against GRUSeasonalBaseline.
    """

    def __init__(
        self,
        time_dim: int = 4,
        hidden_1: int = 128,
        hidden_2: int = 64,
        dropout:  float = 0.2,
    ):
        super().__init__()

        # Applied pointwise across the window dimension via PyTorch broadcasting.
        # Input (..., time_dim) → output (..., 1).
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
        )

        self.register_buffer("err_mean", torch.tensor(0.0))
        self.register_buffer("err_std",  torch.tensor(1.0))

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_time : (B, W, time_dim)

        Returns
        -------
        pred_demand : (B, W)
        """
        return self.net(x_time).squeeze(-1)   # (B, W)

    def raw_errors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, D)  full feature window

        Returns
        -------
        err : (B, W)  |pred_demand - actual_demand|
        """
        x_time = x[:, :, 1:]
        x_val  = x[:, :, 0]
        pred   = self.forward(x_time)
        return (pred - x_val).abs()

    def anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep score in (0, 1) via sigmoid normalisation."""
        err = self.raw_errors(x)
        z   = (err - self.err_mean) / (self.err_std + 1e-8)
        return torch.sigmoid(z)
