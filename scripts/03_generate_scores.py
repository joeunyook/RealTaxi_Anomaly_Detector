"""
03_generate_scores.py

Two-scale scoring pipeline.

MICRO scores  (per slot, window=2):
  MLP_score   — max sigmoid deviation over 2-step window
  KRN_score   — pointwise kernel lookup deviation (last timestep)
  LOF_score   — 5-D local outlier factor (last timestep)
  ENS_micro   — mean(MLP, KRN, LOF)

MACRO score  (per slot, window=6 of micro scores):
  GRU_score — MacroGRU p(sustained event); first 5 positions = 0.0

Output: scores.csv (test) + scores_splits.pkl (train / val / test)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
import pandas as pd
import torch

from src.config import Paths, TrainCfg, MacroCfg
from src.models.mlp import MLPSeasonalBaseline
from src.models.kernel import GaussianKernelDetector
from src.models.macro_gru import MacroGRU


def _mlp_scores(model, X, device, batch=512):
    """Max sigmoid-normalised deviation over micro window. Shape: (N,)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            s  = model.anomaly_scores(xb)           # (B, W)
            scores.append(s.max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def _macro_scores(model, micro_stack, device, W=6, batch=512):
    """
    Run MacroGRU over sliding 6-step windows of micro scores.

    micro_stack : (N, 3)  — [MLP_score, KRN_score, LOF_score] per slot
    Returns     : (N,)    — p(macro_event); first W-1 positions = 0.0
    """
    N = len(micro_stack)
    macro_s = np.zeros(N, dtype=np.float32)
    if N < W:
        return macro_s

    seqs = np.stack([micro_stack[i - W + 1 : i + 1] for i in range(W - 1, N)])

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            xb = torch.tensor(seqs[i:i+batch], dtype=torch.float32, device=device)
            probs.append(model.predict_proba(xb).cpu().numpy())

    macro_s[W - 1:] = np.concatenate(probs)
    return macro_s


def _compute_split_scores(lof, mlp_model, krn, macro_model, X, device, macro_W):
    lof_s  = lof.score(X)
    krn_s  = krn.pointwise_scores(X)
    mlp_s  = _mlp_scores(mlp_model, X, device)

    micro_stack = np.stack([mlp_s, krn_s, lof_s], axis=1)
    mac_s = _macro_scores(macro_model, micro_stack, device, W=macro_W)

    return lof_s, krn_s, mlp_s, mac_s


def main():
    paths     = Paths()
    cfg       = TrainCfg()
    macro_cfg = MacroCfg()
    device    = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]

    time_dim = split.X_train.shape[-1] - 1   # 4

    # ── load models ───────────────────────────────────────────────────────────
    import pickle as _pkl
    with open(paths.MODEL_DIR / "lof.pkl", "rb") as f:
        lof = _pkl.load(f)

    mlp_model = MLPSeasonalBaseline(
        time_dim=time_dim,
        hidden_1=cfg.MLP_HIDDEN_1,
        hidden_2=cfg.MLP_HIDDEN_2,
        dropout=cfg.MLP_DROPOUT,
    ).to(device)
    mlp_model.load_state_dict(
        torch.load(paths.MODEL_DIR / "mlp.pt", map_location=device)
    )

    krn = GaussianKernelDetector.load(str(paths.MODEL_DIR / "kernel.pkl"))

    macro_model = MacroGRU.load(
        str(paths.MODEL_DIR / "macro_gru.pt"), device
    )

    # ── score all splits ──────────────────────────────────────────────────────
    W = macro_cfg.WINDOW

    lof_tr, krn_tr, mlp_tr, mac_tr = _compute_split_scores(
        lof, mlp_model, krn, macro_model, split.X_train, device, W)
    lof_va, krn_va, mlp_va, mac_va = _compute_split_scores(
        lof, mlp_model, krn, macro_model, split.X_val, device, W)
    lof_te, krn_te, mlp_te, mac_te = _compute_split_scores(
        lof, mlp_model, krn, macro_model, split.X_test, device, W)

    sev_tr = split.s_train[:, -1]
    sev_va = split.s_val[:,   -1]
    sev_te = split.s_test[:,  -1]

    # ── save test scores.csv ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":     split.ts_test,
        "label":         split.y_test,
        "true_severity": sev_te,
        "MLP_score":     mlp_te,
        "KRN_score":     krn_te,
        "LOF_score":     lof_te,
        "GRU_score":   mac_te,
    })
    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUT_DIR / "scores.csv", index=False)
    print("Saved:", paths.OUT_DIR / "scores.csv")

    # ── save all splits pickle ────────────────────────────────────────────────
    with open(paths.OUT_DIR / "scores_splits.pkl", "wb") as f:
        pickle.dump({
            "train": {"y": split.y_train, "true_severity": sev_tr, "ts": split.ts_train,
                      "MLP": mlp_tr, "KRN": krn_tr, "LOF": lof_tr, "GRU": mac_tr},
            "val":   {"y": split.y_val,   "true_severity": sev_va, "ts": split.ts_val,
                      "MLP": mlp_va, "KRN": krn_va, "LOF": lof_va, "GRU": mac_va},
            "test":  {"y": split.y_test,  "true_severity": sev_te, "ts": split.ts_test,
                      "MLP": mlp_te, "KRN": krn_te, "LOF": lof_te, "GRU": mac_te},
        }, f)
    print("Saved:", paths.OUT_DIR / "scores_splits.pkl")


if __name__ == "__main__":
    main()
