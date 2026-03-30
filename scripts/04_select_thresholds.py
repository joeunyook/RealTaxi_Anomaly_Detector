"""
04_select_thresholds.py

For each model, choose τ* = argmax_τ F1_val(τ) on window-level labels.
"""

import json
import pickle
import pandas as pd
from src.config import Paths
from src.metrics import best_f1_threshold, apply_threshold, event_level_stats


def main():
    paths = Paths()

    with open(paths.OUT_DIR / "scores_splits.pkl", "rb") as f:
        obj = pickle.load(f)

    yv   = obj["val"]["y"]
    ts_v = obj["val"]["ts"]

    score_map = {
        "LOF":    obj["val"]["LOF"],
        "GRU":    obj["val"]["GRU"],
        "MLP":    obj["val"]["MLP"],
        "SARIMA": obj["val"]["SARIMA"],
        "ENS":    obj["val"]["ENS"],
    }

    taus  = {}
    stats = {}
    for name, scores in score_map.items():
        tau, f1, p, r = best_f1_threshold(yv, scores)
        ev = event_level_stats(ts_v, yv, scores, tau)
        taus[name] = tau
        stats[name] = {
            "tau":              tau,
            "f1_val":           f1,
            "prec_val":         p,
            "rec_val":          r,
            "event_recall_val": ev["event_recall"],
            "events_detected":  ev["detected"],
            "n_events":         ev["n_events"],
        }
        print(f"[{name}] τ={tau:.4f}  F1={f1:.3f}  "
              f"event_recall={ev['event_recall']:.2f}  "
              f"({ev['detected']}/{ev['n_events']} val events)")

    out_json = paths.OUT_DIR / "taus.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved:", out_json)

    # apply thresholds to test scores → preds.csv
    scores_df = pd.read_csv(paths.OUT_DIR / "scores.csv")
    col_map = {
        "LOF":    "LOF_score",
        "GRU":    "GRU_score",
        "MLP":    "MLP_score",
        "SARIMA": "SARIMA_score",
        "ENS":    "ENS_score",
    }
    for name in col_map:
        scores_df[f"{name}_pred"] = apply_threshold(
            scores_df[col_map[name]].to_numpy(), taus[name]
        )

    scores_df.to_csv(paths.OUT_DIR / "preds.csv", index=False)
    print("Saved:", paths.OUT_DIR / "preds.csv")


if __name__ == "__main__":
    main()
