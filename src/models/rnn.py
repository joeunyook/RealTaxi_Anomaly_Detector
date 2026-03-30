import torch
import torch.nn as nn


class GRUSeasonalBaseline(nn.Module):
    """
    GRU trained to predict demand from time features only.

    Input:  [sin_hour, cos_hour, sin_dow, cos_dow]  — never sees the value channel.
    Target: actual demand value at each timestep.
    Score:  sigmoid-normalized |predicted_demand - actual_demand|.

    Because the model is trained only on normal data, it learns
    "what should demand be at this time of day/week?"
    When actual demand deviates from the seasonal expectation (e.g. Christmas),
    the absolute error is large regardless of how smooth the descent is —
    directly fixing the contextual anomaly blindspot of next-step prediction.

    err_mean / err_std are fitted on training residuals after training and stored
    as model buffers so they are saved/loaded with the weights.
    """

    def __init__(
        self,
        time_dim: int = 4,    # sin_hour, cos_hour, sin_dow, cos_dow
        hidden:   int = 128,
        layers:   int = 2,
        dropout:  float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size    = time_dim,
            hidden_size   = hidden,
            num_layers    = layers,
            dropout       = dropout if layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = False,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        self.register_buffer("err_mean", torch.tensor(0.0))
        self.register_buffer("err_std",  torch.tensor(1.0))

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_time : (B, W, time_dim)  — time features only

        Returns
        -------
        pred_demand : (B, W)  — predicted demand at each timestep
        """
        out, _ = self.gru(x_time)              # (B, W, hidden)
        return self.pred_head(out).squeeze(-1)  # (B, W)

    def raw_errors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Absolute deviation of actual demand from seasonal expectation.

        Parameters
        ----------
        x : (B, W, D)  — full feature window (value channel first, then time features)

        Returns
        -------
        err : (B, W)
        """
        x_time = x[:, :, 1:]          # (B, W, time_dim)
        x_val  = x[:, :, 0]           # (B, W) actual demand
        pred   = self.forward(x_time)  # (B, W)
        return (pred - x_val).abs()    # (B, W)

    def anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-timestep anomaly score in (0, 1) via sigmoid normalization.

        Parameters
        ----------
        x : (B, W, D)

        Returns
        -------
        score : (B, W)  — high means anomalous
        """
        err = self.raw_errors(x)
        z   = (err - self.err_mean) / (self.err_std + 1e-8)
        return torch.sigmoid(z)        # (B, W) in (0, 1)
