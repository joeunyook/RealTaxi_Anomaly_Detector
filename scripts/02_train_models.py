import json
import pickle
import numpy as np
import torch
from src.config import Paths, TrainCfg, LofCfg, KernelCfg, MacroCfg
from src.metrics import best_f1_threshold
from src.models.lof import LOFDetector
from src.models.kernel import GaussianKernelDetector
from src.models.rnn import GRUSeasonalBaseline
from src.models.mlp import MLPSeasonalBaseline
from src.train.train_rnn import train_rnn
from src.train.train_mlp import train_mlp
from src.train.train_macro_gru import train_macro_gru


def _neural_scores(model, X, device, batch=512):
    """Max sigmoid-normalised deviation over window. Shape: (N,)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            s  = model.anomaly_scores(xb)
            scores.append(s.max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def main():
    paths      = Paths()
    cfg        = TrainCfg()
    lof_cfg    = LofCfg()
    kernel_cfg = KernelCfg()
    macro_cfg  = MacroCfg()

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]

    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    # ── LOF ──────────────────────────────────────────────────────────────
    lof = LOFDetector(n_neighbors=lof_cfg.N_NEIGHBORS,
                      contamination=lof_cfg.CONTAMINATION)
    lof.fit(split.X_train)
    with open(paths.MODEL_DIR / "lof.pkl", "wb") as f:
        pickle.dump(lof, f)
    print("Saved:", paths.MODEL_DIR / "lof.pkl")

    # ── GRU seasonal baseline ─────────────────────────────────────────────
    rnn_path = str(paths.MODEL_DIR / "rnn.pt")
    train_rnn(
        split.X_train, split.y_train,
        split.X_val,   split.y_val,
        cfg, rnn_path,
        X_test=split.X_test, y_test=split.y_test,
    )
    print("Saved:", rnn_path)

    # ── MLP seasonal baseline ─────────────────────────────────────────────
    mlp_path = str(paths.MODEL_DIR / "mlp.pt")
    train_mlp(
        split.X_train, split.y_train,
        split.X_val,   split.y_val,
        cfg, mlp_path,
        X_test=split.X_test, y_test=split.y_test,
    )
    print("Saved:", mlp_path)

    # ── Gaussian Kernel detector (replaces SARIMA) ────────────────────────
    # Frozen seasonal expectation from normal training data; no recurrent
    # state, so it never adapts to anomalous demand mid-event.
    krn = GaussianKernelDetector(
        bandwidth_hour=kernel_cfg.BANDWIDTH_HOUR,
        bandwidth_dow=kernel_cfg.BANDWIDTH_DOW,
    )
    krn.fit(split.X_train)
    krn.refit_normalization_pointwise(split.X_val, split.y_val)
    krn.save(str(paths.MODEL_DIR / "kernel.pkl"))

    # ── MacroGRU: train on val micro-score sequences ──────────────────────
    # Micro models are now trained; compute val micro scores so that
    # MacroGRU training sequences match exactly what script 03 will produce.
    print("\n[MACRO] Computing val micro scores for MacroGRU training …")
    time_dim = split.X_train.shape[-1] - 1

    rnn_model = GRUSeasonalBaseline(
        time_dim=time_dim, hidden=cfg.RNN_HIDDEN,
        layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT,
    ).to(device)
    rnn_model.load_state_dict(torch.load(rnn_path, map_location=device))

    mlp_model = MLPSeasonalBaseline(
        time_dim=time_dim, hidden_1=cfg.MLP_HIDDEN_1,
        hidden_2=cfg.MLP_HIDDEN_2, dropout=cfg.MLP_DROPOUT,
    ).to(device)
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))

    lof_va = lof.score(split.X_val)
    mlp_va = _neural_scores(mlp_model, split.X_val, device)
    krn_va = krn.pointwise_scores(split.X_val)

    # Val-F1 thresholds → binary {0,1} flags for MacroGRU input.
    # MacroGRU input: [LOF_flag, MLP_flag, KRN_flag]
    # GRU excluded: it sees the same time-feature signal as MLP but with
    # recurrence, making its flags heavily correlated with MLP's — KRN
    # adds a genuinely independent statistical signal instead.
    tau_lof, _, _, _ = best_f1_threshold(split.y_val, lof_va)
    tau_mlp, _, _, _ = best_f1_threshold(split.y_val, mlp_va)
    tau_krn, _, _, _ = best_f1_threshold(split.y_val, krn_va)
    print(f"[MACRO] Micro thresholds — LOF τ={tau_lof:.4f}  MLP τ={tau_mlp:.4f}  KRN τ={tau_krn:.4f}")

    # Binary flags: (N_val, 3)  channels = [LOF_flag, MLP_flag, KRN_flag]
    micro_va = np.stack([
        (lof_va >= tau_lof).astype(np.float32),
        (mlp_va >= tau_mlp).astype(np.float32),
        (krn_va >= tau_krn).astype(np.float32),
    ], axis=1)

    macro_path = str(paths.MODEL_DIR / "macro_gru.pt")
    with open(paths.MODEL_DIR / "macro_input_taus.json", "w") as _f:
        json.dump({"LOF": tau_lof, "MLP": tau_mlp, "KRN": tau_krn}, _f, indent=2)

    train_macro_gru(micro_va, split.y_val, macro_cfg, macro_path, device)


if __name__ == "__main__":
    main()
