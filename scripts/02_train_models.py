"""
Script 02 – Train all anomaly detection models.

Models trained
--------------
Traditional (static):
  LOF              LocalOutlierFactor
  IsolationForest  Ensemble of isolation trees
  KNN              k-Nearest Neighbours distance scorer

Incremental (online):
  IncrementalMeanStd  EMA-based z-score detector

Deep temporal (supervised):
  GRU          Gated Recurrent Unit classifier
  LSTM         Long Short-Term Memory classifier
  CNN          1-D Convolutional Neural Network classifier
  Transformer  Multi-head self-attention encoder classifier

Reconstruction-based:
  VAE          Variational Autoencoder (unsupervised)

All models are saved to the ``models/`` directory.
"""

import pickle
import sys
import torch
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, DataCfg, TrainCfg, LofCfg, TradCfg, IncrementalCfg
from src.models.lof import LOFDetector
from src.models.traditional import IsolationForestDetector, KNNDetector
from src.models.incremental import IncrementalMeanStdDetector
from src.models.rnn import GRUAnomalyClassifier
from src.models.lstm import LSTMAnomalyClassifier
from src.models.cnn import CNNAnomalyClassifier
from src.models.transformer import TransformerAnomalyClassifier
from src.train.train_rnn import train_rnn
from src.train.train_vae import train_vae
from src.train.train_deep import train_deep_classifier


def main():
    paths = Paths()
    cfg = TrainCfg()
    lof_cfg = LofCfg()
    trad_cfg = TradCfg()
    incr_cfg = IncrementalCfg()

    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_dim = split.X_train.shape[-1]

    # ------------------------------------------------------------------ #
    # 1. Traditional baselines                                            #
    # ------------------------------------------------------------------ #
    print("\n[1/8] Training LOF ...")
    lof = LOFDetector(n_neighbors=lof_cfg.N_NEIGHBORS, contamination=lof_cfg.CONTAMINATION)
    lof.fit(split.X_train)
    lof_path = str(paths.MODEL_DIR / "lof.pkl")
    with open(lof_path, "wb") as f:
        pickle.dump(lof, f)
    print("Saved:", lof_path)

    print("\n[2/8] Training Isolation Forest ...")
    iso = IsolationForestDetector(
        n_estimators=trad_cfg.IF_N_ESTIMATORS,
        contamination=trad_cfg.IF_CONTAMINATION,
        random_state=trad_cfg.IF_RANDOM_STATE,
    )
    iso.fit(split.X_train)
    iso_path = str(paths.MODEL_DIR / "iso_forest.pkl")
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)
    print("Saved:", iso_path)

    print("\n[3/8] Training KNN ...")
    knn = KNNDetector(n_neighbors=trad_cfg.KNN_N_NEIGHBORS)
    knn.fit(split.X_train)
    knn_path = str(paths.MODEL_DIR / "knn.pkl")
    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    print("Saved:", knn_path)

    # ------------------------------------------------------------------ #
    # 2. Incremental baseline                                             #
    # ------------------------------------------------------------------ #
    print("\n[4/8] Fitting Incremental (EMA z-score) detector ...")
    incr = IncrementalMeanStdDetector(alpha=incr_cfg.ALPHA, warmup=incr_cfg.WARMUP)
    incr.fit(split.X_train)
    incr_path = str(paths.MODEL_DIR / "incremental.pkl")
    with open(incr_path, "wb") as f:
        pickle.dump(incr, f)
    print("Saved:", incr_path)

    # ------------------------------------------------------------------ #
    # 3. Deep temporal classifiers                                        #
    # ------------------------------------------------------------------ #
    print("\n[5/8] Training GRU ...")
    rnn_path = str(paths.MODEL_DIR / "rnn.pt")
    train_rnn(split.X_train, split.y_train, split.X_val, split.y_val, cfg, rnn_path)
    print("Saved:", rnn_path)

    print("\n[6/8] Training LSTM ...")
    lstm_model = LSTMAnomalyClassifier(
        input_dim=input_dim,
        hidden=cfg.LSTM_HIDDEN,
        layers=cfg.LSTM_LAYERS,
        dropout=cfg.LSTM_DROPOUT,
    ).to(device)
    lstm_path = str(paths.MODEL_DIR / "lstm.pt")
    train_deep_classifier(
        model=lstm_model,
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        epochs=cfg.LSTM_EPOCHS,
        lr=cfg.LSTM_LR,
        batch_size=cfg.LSTM_BATCH,
        out_path=lstm_path,
        device=device,
    )
    print("Saved:", lstm_path)

    print("\n[7/8] Training CNN ...")
    cnn_model = CNNAnomalyClassifier(
        input_dim=input_dim,
        num_filters=cfg.CNN_FILTERS,
        kernel_size=cfg.CNN_KERNEL,
        num_layers=cfg.CNN_LAYERS,
        dropout=cfg.CNN_DROPOUT,
    ).to(device)
    cnn_path = str(paths.MODEL_DIR / "cnn.pt")
    train_deep_classifier(
        model=cnn_model,
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        epochs=cfg.CNN_EPOCHS,
        lr=cfg.CNN_LR,
        batch_size=cfg.CNN_BATCH,
        out_path=cnn_path,
        device=device,
    )
    print("Saved:", cnn_path)

    print("\n[8/8] Training Transformer ...")
    tf_model = TransformerAnomalyClassifier(
        input_dim=input_dim,
        d_model=cfg.TF_D_MODEL,
        nhead=cfg.TF_NHEAD,
        num_layers=cfg.TF_LAYERS,
        dim_feedforward=cfg.TF_DIM_FF,
        dropout=cfg.TF_DROPOUT,
    ).to(device)
    tf_path = str(paths.MODEL_DIR / "transformer.pt")
    train_deep_classifier(
        model=tf_model,
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        epochs=cfg.TF_EPOCHS,
        lr=cfg.TF_LR,
        batch_size=cfg.TF_BATCH,
        out_path=tf_path,
        device=device,
    )
    print("Saved:", tf_path)

    # VAE (reconstruction-based, unsupervised)
    print("\nTraining VAE ...")
    vae_path = str(paths.MODEL_DIR / "vae.pt")
    train_vae(split.X_train, split.X_val, cfg, vae_path)
    print("Saved:", vae_path)

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
