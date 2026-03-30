import numpy as np
import pickle
from src.config import Paths, DataCfg
from src.data_utils import load_taxi_csv, make_windows, meter_style_split
from src.data_utils import fit_standardizer_on_train, apply_standardizer


def main():
    paths    = Paths()
    data_cfg = DataCfg()

    df = load_taxi_csv(str(paths.DATA_CSV))

    # Raw sequential demand values — saved unscaled for SARIMA.
    raw_values = df["value"].to_numpy(dtype=np.float64)

    # make_windows returns (X, y, s, ts)
    # y: point-level label — 1 iff the last timestep of the window is anomalous
    # s: (N, W) — raw anomaly_score per timestep (Beta regression target)
    X, y, s, ts = make_windows(df, data_cfg.WINDOW, data_cfg.STRIDE,
                                data_cfg.USE_TIME_FEATURES)

    # METER-style split: train on normal-only windows; val/test include all windows
    split = meter_style_split(X, y, s, ts, data_cfg.TRAIN_FRAC, data_cfg.VAL_FRAC)

    # Standardize only the value channel (index 0); s is not modified
    scaler        = fit_standardizer_on_train(split.X_train)
    split.X_train = apply_standardizer(split.X_train, scaler)
    split.X_val   = apply_standardizer(split.X_val,   scaler)
    split.X_test  = apply_standardizer(split.X_test,  scaler)

    # ── sequential series for SARIMA ─────────────────────────────────────
    # Window i (stride=1) covers timesteps [i, i+W-1].
    # Split boundary in window-index space → timestep space:
    #   training timeline : timesteps 0          .. n_tr_end  + W - 1
    #   val      timeline : timesteps n_tr_end   .. n_val_end + W - 1
    #   test     timeline : timesteps n_val_end  .. end
    n          = len(X)
    W          = data_cfg.WINDOW
    n_tr_end   = int(n * data_cfg.TRAIN_FRAC)
    n_val_end  = int(n * (data_cfg.TRAIN_FRAC + data_cfg.VAL_FRAC))
    seq_data   = {
        "train":     raw_values[: n_tr_end  + W],   # full training timeline
        "val":       raw_values[n_tr_end  : n_val_end + W],
        "test":      raw_values[n_val_end :],
        "window":    W,
        "n_tr_end":  n_tr_end,
        "n_val_end": n_val_end,
    }

    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.OUT_DIR / "split.pkl", "wb") as f:
        pickle.dump({"split": split, "scaler": scaler, "seq": seq_data}, f)

    print("Saved:", paths.OUT_DIR / "split.pkl")
    print("Shapes X:", split.X_train.shape, split.X_val.shape, split.X_test.shape)
    print("Shapes s:", split.s_train.shape, split.s_val.shape, split.s_test.shape)
    n_tr  = len(split.y_train)
    n_va  = len(split.y_val)
    n_te  = len(split.y_test)
    print(f"train: {split.y_train.sum()} / {n_tr} anomaly windows = {split.y_train.mean()*100:.1f}%")
    print(f"val:   {split.y_val.sum()} / {n_va} anomaly windows = {split.y_val.mean()*100:.1f}%")
    print(f"test:  {split.y_test.sum()} / {n_te} anomaly windows = {split.y_test.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
