import torch
import torch.nn as nn


class MacroGRU(nn.Module):
    """
    Second-stage macro event detector.

    Consumes a MACRO_WINDOW-length sequence of per-slot micro anomaly scores
    — one 3-channel vector [MLP_score, KRN_score, LOF_score] per slot —
    and outputs a raw logit (p(macro_event) after sigmoid) at the last timestep.

    Why the hidden state is now useful
    ------------------------------------
    In the old design the GRU received sin/cos time features whose positional
    information is already fully encoded per slot — no sequential dependency
    existed, so the recurrent state only added inertia.

    Here the input IS the anomaly signal itself (micro scores).  The hidden
    state now answers: "have micro scores been elevated for several consecutive
    slots?"  A single isolated spike → one high input → hidden state recovers
    immediately.  A sustained event (Christmas, blizzard) keeps micro scores
    elevated for hours → hidden state saturates → p(event) → 1.

    The inertia that was a bug in the old design is a feature here: it requires
    sustained micro evidence before firing, which is exactly what distinguishes
    multi-day anomaly events from 30-minute demand spikes.

    Training
    --------
    Trained on validation sequences with BCE (BCEWithLogitsLoss for numerical
    stability).  Val is used because the training split is normal-only — no
    anomaly sequences are available there.
    """

    def __init__(self, input_dim: int = 3, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, 3)  — W timesteps of [MLP_score, KRN_score, LOF_score]

        Returns
        -------
        logit : (B,)  — raw logit; apply sigmoid for probability
        """
        out, _ = self.gru(x)                     # (B, W, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)  # (B,) raw logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid-normalised event probability in (0, 1). Use at inference."""
        return torch.sigmoid(self.forward(x))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"[MACRO] Saved: {path}")

    @staticmethod
    def load(path: str, device,
             input_dim: int = 3, hidden: int = 32, layers: int = 1) -> "MacroGRU":
        m = MacroGRU(input_dim, hidden, layers).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        return m
