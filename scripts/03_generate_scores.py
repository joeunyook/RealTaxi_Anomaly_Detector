"""
Script 03 – Generate anomaly scores for all models on train / val / test splits.

Scores are saved to:
  outputs/scores.csv          (test split, all models)
  outputs/scores_splits.pkl   (all splits, all models)
"""

import pickle
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, TrainCfg
from src.models.rnn import GRUAnomalyClassifier
from src.models.lstm import LSTMAnomalyClassifier
from src.models.cnn import CNNAnomalyClassifier
from src.models.transformer import TransformerAnomalyClassifier
from src.models.vae import WindowVAE
from src.models.ensemble import normalize_minmax, ensemble_mean


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def deep_score(model, X, device):
    """Run a deep classifier and return sigmoid-transformed anomaly scores."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(xb).detach().cpu().numpy()
    return sigmoid(logits)


def vae_raw(model, X, device):
    """Return per-window reconstruction MSE from the VAE."""
    model.eval()
    Xf = X.reshape(X.shape[0], -1)
    with torch.no_grad():
        xb = torch.tensor(Xf, dtype=torch.float32, device=device)
        x_hat, _, _ = model(xb)
        recon = torch.mean((x_hat - xb) ** 2, dim=1).detach().cpu().numpy()
    return recon


def main():
    paths = Paths()
    cfg = TrainCfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]
    input_dim = split.X_train.shape[-1]

    # ------------------------------------------------------------------ #
    # Load traditional / incremental models                               #
    # ------------------------------------------------------------------ #
    with open(paths.MODEL_DIR / "lof.pkl", "rb") as f:
        lof = pickle.load(f)
    with open(paths.MODEL_DIR / "iso_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(paths.MODEL_DIR / "knn.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(paths.MODEL_DIR / "incremental.pkl", "rb") as f:
        incr = pickle.load(f)

    # ------------------------------------------------------------------ #
    # Load deep models                                                    #
    # ------------------------------------------------------------------ #
    gru = GRUAnomalyClassifier(
        input_dim=input_dim, hidden=cfg.RNN_HIDDEN,
        layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT,
    ).to(device)
    gru.load_state_dict(torch.load(paths.MODEL_DIR / "rnn.pt", map_location=device))

    lstm = LSTMAnomalyClassifier(
        input_dim=input_dim, hidden=cfg.LSTM_HIDDEN,
        layers=cfg.LSTM_LAYERS, dropout=cfg.LSTM_DROPOUT,
    ).to(device)
    lstm.load_state_dict(torch.load(paths.MODEL_DIR / "lstm.pt", map_location=device))

    cnn = CNNAnomalyClassifier(
        input_dim=input_dim, num_filters=cfg.CNN_FILTERS,
        kernel_size=cfg.CNN_KERNEL, num_layers=cfg.CNN_LAYERS,
        dropout=cfg.CNN_DROPOUT,
    ).to(device)
    cnn.load_state_dict(torch.load(paths.MODEL_DIR / "cnn.pt", map_location=device))

    tf = TransformerAnomalyClassifier(
        input_dim=input_dim, d_model=cfg.TF_D_MODEL,
        nhead=cfg.TF_NHEAD, num_layers=cfg.TF_LAYERS,
        dim_feedforward=cfg.TF_DIM_FF, dropout=cfg.TF_DROPOUT,
    ).to(device)
    tf.load_state_dict(torch.load(paths.MODEL_DIR / "transformer.pt", map_location=device))

    input_dim_vae = split.X_train.reshape(split.X_train.shape[0], -1).shape[1]
    vae = WindowVAE(input_dim=input_dim_vae, hidden=cfg.VAE_HIDDEN, z_dim=cfg.VAE_Z).to(device)
    vae.load_state_dict(torch.load(paths.MODEL_DIR / "vae.pt", map_location=device))

    # ------------------------------------------------------------------ #
    # Score all splits                                                    #
    # ------------------------------------------------------------------ #
    splits_data = {
        "train": (split.X_train, split.y_train, split.ts_train),
        "val":   (split.X_val,   split.y_val,   split.ts_val),
        "test":  (split.X_test,  split.y_test,  split.ts_test),
    }

    # Pre-compute VAE train raw scores for normalization reference
    vae_tr_raw = vae_raw(vae, split.X_train, device)

    all_scores = {}
    for split_name, (X, y, ts) in splits_data.items():
        lof_s   = lof.score(X)
        iso_s   = iso.score(X)
        knn_s   = knn.score(X)
        incr_s  = incr.score(X)
        gru_s   = deep_score(gru, X, device)
        lstm_s  = deep_score(lstm, X, device)
        cnn_s   = deep_score(cnn, X, device)
        tf_s    = deep_score(tf, X, device)
        vae_raw_s = vae_raw(vae, X, device)
        vae_s   = normalize_minmax(vae_tr_raw, vae_raw_s)

        # Ensemble: mean of all normalized scores
        ens_s = ensemble_mean(lof_s, iso_s, knn_s, incr_s, gru_s, lstm_s, cnn_s, tf_s, vae_s)

        all_scores[split_name] = {
            "y":           y,
            "ts":          ts,
            "LOF":         lof_s,
            "IsoForest":   iso_s,
            "KNN":         knn_s,
            "Incremental": incr_s,
            "GRU":         gru_s,
            "LSTM":        lstm_s,
            "CNN":         cnn_s,
            "Transformer": tf_s,
            "VAE":         vae_s,
            "ENS":         ens_s,
        }

    # ------------------------------------------------------------------ #
    # Save test scores CSV                                                #
    # ------------------------------------------------------------------ #
    test = all_scores["test"]
    model_keys = ["LOF", "IsoForest", "KNN", "Incremental", "GRU", "LSTM", "CNN", "Transformer", "VAE", "ENS"]
    df = pd.DataFrame({"timestamp": test["ts"], "label": test["y"]})
    for k in model_keys:
        df[f"{k}_score"] = test[k]

    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = paths.OUT_DIR / "scores.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Save all splits (needed for threshold selection on val)
    splits_pkl = {
        split_name: {
            "y":  all_scores[split_name]["y"],
            **{k: all_scores[split_name][k] for k in model_keys},
        }
        for split_name in ["train", "val", "test"]
    }
    with open(paths.OUT_DIR / "scores_splits.pkl", "wb") as f:
        pickle.dump(splits_pkl, f)
    print("Saved:", paths.OUT_DIR / "scores_splits.pkl")


if __name__ == "__main__":
    main()
