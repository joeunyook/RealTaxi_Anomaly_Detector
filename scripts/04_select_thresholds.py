"""
04_select_thresholds.py

Two-threshold selection for the two-scale architecture.

τ_micro  — argmax F-beta(β=2) on val per-model micro scores
           Recall weighted 4×: false micro alerts are free — the MacroGRU's
           hidden state recovers immediately from isolated spikes.
           Missing a slot delays macro evidence accumulation.

τ_GRU    — argmax F-beta(β=0.5) on val GRU score
           β=0.5 weights precision 4× over recall: false alarms and missed
           events are treated symmetrically but false alarms are penalised
           more. With 2 val events the metric has enough power to properly
           trade off FA clusters against missed detections.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import pandas as pd
from src.config import Paths, MacroCfg
from src.metrics import best_fbeta_threshold, apply_threshold, event_level_stats


def main():
    paths     = Paths()
    macro_cfg = MacroCfg()

    with open(paths.OUT_DIR / "scores_splits.pkl", "rb") as f:
        obj = pickle.load(f)

    yv   = obj["val"]["y"]
    ts_v = obj["val"]["ts"]

    # Micro models: F-beta(β=2) — recall-heavy, GRU filters the false alarms
    micro_models = {
        "MLP": obj["val"]["MLP"],
        "KRN": obj["val"]["KRN"],
        "LOF": obj["val"]["LOF"],
    }
    # Macro model: argmax F1 — symmetric after GRU temporal filtering
    macro_score = obj["val"]["GRU"]

    taus  = {}
    stats = {}

    print("── Micro thresholds (F-beta β=2, recall 4× precision) ──────────────")
    for name, scores in micro_models.items():
        tau, fb, p, r = best_fbeta_threshold(yv, scores, beta=macro_cfg.MICRO_FBETA)
        ev = event_level_stats(ts_v, yv, scores, tau)
        taus[name]  = tau
        stats[name] = {
            "tau":              tau,
            "fbeta":            fb,
            "beta":             macro_cfg.MICRO_FBETA,
            "prec_val":         p,
            "rec_val":          r,
            "event_recall_val": ev["event_recall"],
            "events_detected":  ev["detected"],
            "n_events":         ev["n_events"],
        }
        print(f"[{name:>5}] τ={tau:.4f}  F{macro_cfg.MICRO_FBETA}={fb:.3f}  "
              f"prec={p:.3f}  rec={r:.3f}  "
              f"event_recall={ev['event_recall']:.2f}  "
              f"({ev['detected']}/{ev['n_events']} val events)")

    print("\n── GRU threshold  (F-beta β=0.5, precision 4× recall) ──────────────")
    tau, fb, p, r = best_fbeta_threshold(yv, macro_score, beta=0.5)
    ev = event_level_stats(ts_v, yv, macro_score, tau)
    taus["GRU"]  = tau
    stats["GRU"] = {
        "tau":              tau,
        "fbeta":            fb,
        "beta":             0.5,
        "prec_val":         p,
        "rec_val":          r,
        "event_recall_val": ev["event_recall"],
        "events_detected":  ev["detected"],
        "n_events":         ev["n_events"],
    }
    print(f"[GRU] τ={tau:.4f}  F0.5={fb:.3f}  "
          f"prec={p:.3f}  rec={r:.3f}  "
          f"event_recall={ev['event_recall']:.2f}  "
          f"({ev['detected']}/{ev['n_events']} val events)")

    out_json = paths.OUT_DIR / "taus.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved:", out_json)

    # ── apply thresholds to test scores → preds.csv ──────────────────────────
    scores_df = pd.read_csv(paths.OUT_DIR / "scores.csv")
    col_map = {
        "MLP":   "MLP_score",
        "KRN":   "KRN_score",
        "LOF":   "LOF_score",
        "GRU": "GRU_score",
    }
    for name, col in col_map.items():
        scores_df[f"{name}_pred"] = apply_threshold(
            scores_df[col].to_numpy(), taus[name]
        )

    scores_df.to_csv(paths.OUT_DIR / "preds.csv", index=False)
    print("Saved:", paths.OUT_DIR / "preds.csv")


if __name__ == "__main__":
    main()
