import numpy as np
import torch

from src.models.lof import LOFDetector
from src.models.rnn import GRUSeasonalBaseline
from src.models.ensemble import ensemble_mean


def rnn_scores(model, X, device):
    """Max sigmoid-normalised seasonal deviation across the window. Shape: (N,)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i+512], dtype=torch.float32, device=device)
            s  = model.anomaly_scores(xb)
            scores.append(s.max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def run_all_scores(split, cfg, paths):
    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths.TAB_DIR.mkdir(parents=True, exist_ok=True)
    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device   = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    time_dim = split.X_train.shape[-1] - 1

    import pickle
    with open(paths.MODEL_DIR / "lof.pkl", "rb") as f:
        lof = pickle.load(f)

    rnn = GRUSeasonalBaseline(
        time_dim=time_dim, hidden=cfg.RNN_HIDDEN,
        layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT,
    ).to(device)
    rnn.load_state_dict(torch.load(paths.MODEL_DIR / "rnn.pt", map_location=device))

    lof_scores = lof.score(split.X_test)
    rnn_s      = rnn_scores(rnn, split.X_test, device)
    ens_scores = ensemble_mean(lof_scores, rnn_s)

    return {"LOF": lof_scores, "RNN": rnn_s, "ENS": ens_scores}