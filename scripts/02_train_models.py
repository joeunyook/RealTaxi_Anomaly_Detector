import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
import torch

from src.config import Paths, TrainCfg, LofCfg, KernelCfg, MacroCfg
from src.models.lof import LOFDetector
from src.models.kernel import GaussianKernelDetector
from src.models.mlp import MLPSeasonalBaseline
from src.train.train_mlp import train_mlp
from src.train.train_macro_gru import train_macro_gru


def _mlp_micro_scores(model, X, device, batch=512):
    """Max sigmoid-normalised deviation over the 2-step micro window. Shape: (N,)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            s  = model.anomaly_scores(xb)           # (B, W)
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

    # ── LOF (micro, window=1 via last-timestep flatten → 5-D density) ────────
    lof = LOFDetector(n_neighbors=lof_cfg.N_NEIGHBORS,
                      contamination=lof_cfg.CONTAMINATION)
    lof.fit(split.X_train)
    with open(paths.MODEL_DIR / "lof.pkl", "wb") as f:
        pickle.dump(lof, f)
    print("Saved:", paths.MODEL_DIR / "lof.pkl")

    # ── MLP seasonal baseline (micro, window=2) ───────────────────────────────
    # Pointwise: f(sin_h, cos_h, sin_d, cos_d) → pred_demand per slot.
    # Max deviation over the 2-step window is the micro score.
    mlp_path = str(paths.MODEL_DIR / "mlp.pt")
    train_mlp(
        split.X_train, split.y_train,
        split.X_val,   split.y_val,
        cfg, mlp_path,
        X_test=split.X_test, y_test=split.y_test,
    )
    print("Saved:", mlp_path)

    # ── Gaussian Kernel detector (micro, pointwise last-timestep lookup) ──────
    # Frozen seasonal expectation per 30-min slot; no observed-value input at
    # inference.  refit_normalization_pointwise anchors the sigmoid to the
    # deployment-period noise floor at slot granularity.
    krn = GaussianKernelDetector(
        bandwidth_hour=kernel_cfg.BANDWIDTH_HOUR,
        bandwidth_dow=kernel_cfg.BANDWIDTH_DOW,
    )
    krn.fit(split.X_train)
    krn.refit_normalization_pointwise(split.X_val, split.y_val)
    krn.save(str(paths.MODEL_DIR / "kernel.pkl"))

    # ── MacroGRU: train on val micro-score sequences ──────────────────────────
    # Micro models are now trained; compute val micro scores inline so that the
    # MacroGRU training sequences reflect exactly what script 03 will produce.
    print("\n[MACRO] Computing val micro scores for MacroGRU training …")

    time_dim  = split.X_train.shape[-1] - 1   # 4
    mlp_model = MLPSeasonalBaseline(
        time_dim=time_dim,
        hidden_1=cfg.MLP_HIDDEN_1,
        hidden_2=cfg.MLP_HIDDEN_2,
        dropout=cfg.MLP_DROPOUT,
    ).to(device)
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))

    lof_va = lof.score(split.X_val)                         # (N_val,)
    krn_va = krn.pointwise_scores(split.X_val)              # (N_val,)
    mlp_va = _mlp_micro_scores(mlp_model, split.X_val, device)  # (N_val,)

    # Stack to (N_val, 3): channels = [MLP, KRN, LOF]
    micro_va = np.stack([mlp_va, krn_va, lof_va], axis=1)

    macro_path = str(paths.MODEL_DIR / "macro_gru.pt")
    train_macro_gru(micro_va, split.y_val, macro_cfg, macro_path, device)


if __name__ == "__main__":
    main()
