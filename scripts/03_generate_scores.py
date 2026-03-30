"""
03_generate_scores.py

Produces per-window scores for every model.

Output columns in scores.csv / scores_splits.pkl:
  timestamp    — window-end timestamp
  label        — binary anomaly label
  true_severity
  LOF_score    — LOF window score (normalised [0,1])
  GRU_score    — GRU seasonal deviation, max over window, in (0,1)
  MLP_score    — MLP seasonal deviation, max over window, in (0,1)
  SARIMA_score — SARIMA residual, rolling-max over window, in (0,1)
  ENS_score    — mean(LOF, GRU, MLP, SARIMA)
"""

import pickle
import numpy as np
import pandas as pd
import torch

from src.config import Paths, TrainCfg
from src.models.rnn import GRUSeasonalBaseline
from src.models.mlp import MLPSeasonalBaseline
from src.models.sarima import SARIMADetector
from src.models.ensemble import ensemble_mean


def _neural_scores(model, X, device, batch=512):
    """Max sigmoid-normalised seasonal deviation across the window. Shape: (N,)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            s  = model.anomaly_scores(xb)
            scores.append(s.max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def main():
    paths  = Paths()
    cfg    = TrainCfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split    = obj["split"]
    seq_data = obj["seq"]

    time_dim = split.X_train.shape[-1] - 1
    W        = seq_data["window"]

    # ── load models ───────────────────────────────────────────────────────
    with open(paths.MODEL_DIR / "lof.pkl", "rb") as f:
        lof = pickle.load(f)

    rnn = GRUSeasonalBaseline(
        time_dim=time_dim, hidden=cfg.RNN_HIDDEN,
        layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT,
    ).to(device)
    rnn.load_state_dict(torch.load(paths.MODEL_DIR / "rnn.pt", map_location=device))

    mlp = MLPSeasonalBaseline(
        time_dim=time_dim, hidden_1=cfg.MLP_HIDDEN_1,
        hidden_2=cfg.MLP_HIDDEN_2, dropout=cfg.MLP_DROPOUT,
    ).to(device)
    mlp.load_state_dict(torch.load(paths.MODEL_DIR / "mlp.pt", map_location=device))

    sarima = SARIMADetector.load(str(paths.MODEL_DIR / "sarima.pkl"))

    # ── compute scores ────────────────────────────────────────────────────
    lof_tr = lof.score(split.X_train)
    lof_va = lof.score(split.X_val)
    lof_te = lof.score(split.X_test)

    gru_tr = _neural_scores(rnn, split.X_train, device)
    gru_va = _neural_scores(rnn, split.X_val,   device)
    gru_te = _neural_scores(rnn, split.X_test,  device)

    mlp_tr = _neural_scores(mlp, split.X_train, device)
    mlp_va = _neural_scores(mlp, split.X_val,   device)
    mlp_te = _neural_scores(mlp, split.X_test,  device)

    print("[SARIMA] Scoring val …")
    sar_va = sarima.window_scores(seq_data["val"],  len(split.X_val),  W)
    print("[SARIMA] Scoring test …")
    sar_te = sarima.window_scores(seq_data["test"], len(split.X_test), W)
    # Training scores: use in-sample residuals converted to window scores
    pt_train = 1.0 / (1.0 + np.exp(
        -(np.abs(sarima.result.resid) - sarima.err_mean) / (sarima.err_std + 1e-8)
    ))
    sar_tr = np.array([pt_train[i:i+W].max() for i in range(len(split.X_train))])

    ens_tr = ensemble_mean(lof_tr, gru_tr, mlp_tr, sar_tr)
    ens_va = ensemble_mean(lof_va, gru_va, mlp_va, sar_va)
    ens_te = ensemble_mean(lof_te, gru_te, mlp_te, sar_te)

    sev_tr = split.s_train[:, -1]
    sev_va = split.s_val[:,   -1]
    sev_te = split.s_test[:,  -1]

    # ── save test scores ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":     split.ts_test,
        "label":         split.y_test,
        "true_severity": sev_te,
        "LOF_score":     lof_te,
        "GRU_score":     gru_te,
        "MLP_score":     mlp_te,
        "SARIMA_score":  sar_te,
        "ENS_score":     ens_te,
    })
    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUT_DIR / "scores.csv", index=False)
    print("Saved:", paths.OUT_DIR / "scores.csv")

    # ── save all splits ───────────────────────────────────────────────────
    with open(paths.OUT_DIR / "scores_splits.pkl", "wb") as f:
        pickle.dump({
            "train": {"y": split.y_train, "true_severity": sev_tr,
                      "LOF": lof_tr, "GRU": gru_tr, "MLP": mlp_tr,
                      "SARIMA": sar_tr, "ENS": ens_tr, "ts": split.ts_train},
            "val":   {"y": split.y_val,   "true_severity": sev_va,
                      "LOF": lof_va, "GRU": gru_va, "MLP": mlp_va,
                      "SARIMA": sar_va, "ENS": ens_va, "ts": split.ts_val},
            "test":  {"y": split.y_test,  "true_severity": sev_te,
                      "LOF": lof_te, "GRU": gru_te, "MLP": mlp_te,
                      "SARIMA": sar_te, "ENS": ens_te, "ts": split.ts_test},
        }, f)
    print("Saved:", paths.OUT_DIR / "scores_splits.pkl")


if __name__ == "__main__":
    main()
