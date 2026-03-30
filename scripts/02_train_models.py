import pickle
from src.config import Paths, TrainCfg, LofCfg
from src.models.lof import LOFDetector
from src.models.sarima import SARIMADetector
from src.train.train_rnn import train_rnn
from src.train.train_mlp import train_mlp


def main():
    paths   = Paths()
    cfg     = TrainCfg()
    lof_cfg = LofCfg()

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split    = obj["split"]
    seq_data = obj["seq"]

    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── LOF ──────────────────────────────────────────────────────────────
    lof = LOFDetector(n_neighbors=lof_cfg.N_NEIGHBORS,
                      contamination=lof_cfg.CONTAMINATION)
    lof.fit(split.X_train)
    with open(paths.MODEL_DIR / "lof.pkl", "wb") as f:
        pickle.dump(lof, f)
    print("Saved:", paths.MODEL_DIR / "lof.pkl")

    # ── GRU seasonal baseline ─────────────────────────────────────────────
    # Receives only time features; predicts demand via recurrent state.
    rnn_path = str(paths.MODEL_DIR / "rnn.pt")
    train_rnn(
        split.X_train, split.y_train,
        split.X_val,   split.y_val,
        cfg, rnn_path,
        X_test=split.X_test, y_test=split.y_test,
    )
    print("Saved:", rnn_path)

    # ── MLP seasonal baseline (baseline comparison vs GRU) ────────────────
    # Same objective as GRU but pointwise — no recurrence.
    mlp_path = str(paths.MODEL_DIR / "mlp.pt")
    train_mlp(
        split.X_train, split.y_train,
        split.X_val,   split.y_val,
        cfg, mlp_path,
        X_test=split.X_test, y_test=split.y_test,
    )
    print("Saved:", mlp_path)

    # ── SARIMA seasonal baseline ──────────────────────────────────────────
    # Statistical model; fits on sequential normal training series.
    sarima = SARIMADetector(
        order          = (cfg.SARIMA_P, cfg.SARIMA_D, cfg.SARIMA_Q),
        seasonal_order = (cfg.SARIMA_SP, cfg.SARIMA_SD, cfg.SARIMA_SQ, cfg.SARIMA_S),
    )
    sarima.fit(seq_data["train"])
    sarima.save(str(paths.MODEL_DIR / "sarima.pkl"))


if __name__ == "__main__":
    main()
