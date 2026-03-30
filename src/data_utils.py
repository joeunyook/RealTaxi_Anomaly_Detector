import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    # X_*: (N, W, D)  y_*: (N,) binary label  s_*: (N, W) anomaly_score per timestep
    X_train:  np.ndarray
    y_train:  np.ndarray
    s_train:  np.ndarray   # ground-truth severity target for Beta regression
    X_val:    np.ndarray
    y_val:    np.ndarray
    s_val:    np.ndarray
    X_test:   np.ndarray
    y_test:   np.ndarray
    s_test:   np.ndarray
    ts_train: np.ndarray
    ts_val:   np.ndarray
    ts_test:  np.ndarray


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    dow  = df["timestamp"].dt.dayofweek.astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_dow"]  = np.sin(2 * np.pi * dow  / 7.0)
    df["cos_dow"]  = np.cos(2 * np.pi * dow  / 7.0)
    return df


def load_taxi_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"timestamp", "value", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["value"]         = pd.to_numeric(df["value"],         errors="coerce")
    df["label"]         = pd.to_numeric(df["label"],         errors="coerce").fillna(0).astype(int)
    df["anomaly_score"] = pd.to_numeric(df.get("anomaly_score", 0), errors="coerce").fillna(0.0)

    df = df.dropna(subset=["value"]).sort_values("timestamp").reset_index(drop=True)
    return df


def make_windows(
    df: pd.DataFrame,
    window: int,
    stride: int,
    use_time_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X  : (N, W, D)  — feature windows
    y  : (N,)       — binary label (1 iff the last timestep of the window is anomalous)
    s  : (N, W)     — raw anomaly_score at every timestep in the window
    ts : (N,)       — timestamp of the last step in each window
    """
    df2       = df.copy()
    feat_cols = ["value"]
    if use_time_features:
        df2 = _add_time_features(df2)
        feat_cols += ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]

    values       = df2[feat_cols].to_numpy(dtype=np.float32)
    labels       = df2["label"].to_numpy(dtype=np.int64)
    raw_scores   = df2["anomaly_score"].to_numpy(dtype=np.float32)
    tss          = df2["timestamp"].to_numpy()

    X_list, y_list, s_list, ts_list = [], [], [], []
    for end in range(window - 1, len(df2), stride):
        start = end - window + 1
        X_list.append(values[start:end + 1])
        y_list.append(int(labels[end]))                 # point-level: label of last step
        s_list.append(raw_scores[start:end + 1])        # (W,) severity per timestep
        ts_list.append(tss[end])

    X  = np.stack(X_list,  axis=0)
    y  = np.asarray(y_list, dtype=np.int64)
    s  = np.stack(s_list,  axis=0).astype(np.float32)  # (N, W)
    ts = np.asarray(ts_list)
    return X, y, s, ts


def meter_style_split(
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    ts: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> SplitData:
    """
    METER-style split.

    Train : normal-only windows drawn from the first train_frac of the timeline.
            Anomalous windows in that period are excluded so unsupervised models
            (VAE, LOF) learn a clean normal distribution.
    Val   : ALL windows in the next val_frac of the timeline (used for threshold
            selection; includes anomalous windows so F1/event-recall are meaningful).
    Test  : ALL remaining windows.

    This preserves temporal order and prevents label-horizon inflation from
    contaminating the training set.
    """
    n          = len(X)
    n_tr_end   = int(n * train_frac)
    n_val_end  = int(n * (train_frac + val_frac))

    # Train: keep only normal windows from the first portion of the timeline
    normal_mask = y[:n_tr_end] == 0
    train_idx   = np.where(normal_mask)[0]

    return SplitData(
        X_train  = X[train_idx],
        y_train  = y[train_idx],
        s_train  = s[train_idx],
        ts_train = ts[train_idx],
        X_val    = X[n_tr_end:n_val_end],
        y_val    = y[n_tr_end:n_val_end],
        s_val    = s[n_tr_end:n_val_end],
        ts_val   = ts[n_tr_end:n_val_end],
        X_test   = X[n_val_end:],
        y_test   = y[n_val_end:],
        s_test   = s[n_val_end:],
        ts_test  = ts[n_val_end:],
    )


def fit_standardizer_on_train(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train[:, :, 0].reshape(-1, 1))
    return scaler


def apply_standardizer(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    X2    = X.copy()
    v     = scaler.transform(X2[:, :, 0].reshape(-1, 1)).reshape(X2.shape[0], X2.shape[1])
    X2[:, :, 0] = v
    return X2
