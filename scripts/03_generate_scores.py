"""
03_generate_scores.py

Produces per-window scores for every model.

Output columns in scores.csv / scores_splits.pkl:
  timestamp   — window-end timestamp
  label       — binary anomaly label
  true_severity
  LOF_score   — LOF window score (normalised [0,1])
  GRU_score   — GRU seasonal deviation, max over window, in (0,1)
  MLP_score   — MLP seasonal deviation, max over window, in (0,1)
  KRN_score   — Gaussian Kernel pointwise deviation, in (0,1)
  ENS_score   — mean(LOF, GRU, MLP, KRN)
  MACRO_score — MacroGRU p(sustained event); 24h context of binary [LOF,MLP,KRN] flags
"""

import json
import pickle
import numpy as np
import pandas as pd
import torch

from src.config import Paths, TrainCfg, MacroCfg
from src.models.rnn import GRUSeasonalBaseline
from src.models.mlp import MLPSeasonalBaseline
from src.models.kernel import GaussianKernelDetector
from src.models.macro_gru import MacroGRU
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


def _macro_scores(model, micro_stack, device, W, smooth_W, batch=512):
    """
    Run MacroGRU over sliding W-step windows of binary micro flags.
    micro_stack : (N, 3) — [LOF_flag, MLP_flag, KRN_flag] each {0,1}
    Returns     : (N,)   — p(macro_event); first W-1 positions = 0.0
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
    out = np.empty_like(macro_s)
    for i in range(N):
        start  = max(0, i - smooth_W + 1)
        out[i] = macro_s[start : i + 1].mean()
    return out


def main():
    paths     = Paths()
    cfg       = TrainCfg()
    macro_cfg = MacroCfg()
    device    = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]

    time_dim = split.X_train.shape[-1] - 1

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

    krn = GaussianKernelDetector.load(str(paths.MODEL_DIR / "kernel.pkl"))

    macro_model = MacroGRU.load(str(paths.MODEL_DIR / "macro_gru.pt"), device,
                                input_dim=macro_cfg.INPUT_DIM)

    with open(paths.MODEL_DIR / "macro_input_taus.json") as _f:
        micro_taus = json.load(_f)
    tau_lof = micro_taus["LOF"]
    tau_mlp = micro_taus["MLP"]
    tau_krn = micro_taus["KRN"]
    print(f"[MACRO] Micro thresholds — LOF τ={tau_lof:.4f}  MLP τ={tau_mlp:.4f}  KRN τ={tau_krn:.4f}")

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

    krn_tr = krn.pointwise_scores(split.X_train)
    krn_va = krn.pointwise_scores(split.X_val)
    krn_te = krn.pointwise_scores(split.X_test)

    ens_tr = ensemble_mean(lof_tr, gru_tr, mlp_tr, krn_tr)
    ens_va = ensemble_mean(lof_va, gru_va, mlp_va, krn_va)
    ens_te = ensemble_mean(lof_te, gru_te, mlp_te, krn_te)

    # ── MacroGRU: binary flags [LOF, MLP, KRN] — 24h momentum context ────
    W  = macro_cfg.WINDOW
    SW = macro_cfg.SMOOTH_WINDOW

    def _binary_stack(lof_s, mlp_s, krn_s):
        return np.stack([
            (lof_s >= tau_lof).astype(np.float32),
            (mlp_s >= tau_mlp).astype(np.float32),
            (krn_s >= tau_krn).astype(np.float32),
        ], axis=1)

    macro_tr = _macro_scores(macro_model, _binary_stack(lof_tr, mlp_tr, krn_tr), device, W, SW)
    macro_va = _macro_scores(macro_model, _binary_stack(lof_va, mlp_va, krn_va), device, W, SW)
    macro_te = _macro_scores(macro_model, _binary_stack(lof_te, mlp_te, krn_te), device, W, SW)

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
        "KRN_score":     krn_te,
        "ENS_score":     ens_te,
        "MACRO_score":   macro_te,
    })
    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUT_DIR / "scores.csv", index=False)
    print("Saved:", paths.OUT_DIR / "scores.csv")

    # ── save all splits ───────────────────────────────────────────────────
    with open(paths.OUT_DIR / "scores_splits.pkl", "wb") as f:
        pickle.dump({
            "train": {"y": split.y_train, "true_severity": sev_tr, "ts": split.ts_train,
                      "LOF": lof_tr, "GRU": gru_tr, "MLP": mlp_tr,
                      "KRN": krn_tr, "ENS": ens_tr, "MACRO": macro_tr},
            "val":   {"y": split.y_val,   "true_severity": sev_va, "ts": split.ts_val,
                      "LOF": lof_va, "GRU": gru_va, "MLP": mlp_va,
                      "KRN": krn_va, "ENS": ens_va, "MACRO": macro_va},
            "test":  {"y": split.y_test,  "true_severity": sev_te, "ts": split.ts_test,
                      "LOF": lof_te, "GRU": gru_te, "MLP": mlp_te,
                      "KRN": krn_te, "ENS": ens_te, "MACRO": macro_te},
        }, f)
    print("Saved:", paths.OUT_DIR / "scores_splits.pkl")


if __name__ == "__main__":
    main()
