"""
05_eval_and_plots.py  —  Three-layer evaluation (two-scale architecture)

Layer 1 — Ranking quality  (AUROC / AUPRC / PA-F1)
Layer 2 — Intra-event severity  (Spearman vs true_severity)
Layer 3 — Calibration  (GRU score binned vs true_severity)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd

from src.config import Paths
from src.metrics import (auroc, auprc, point_adjusted_f1,
                          spearman_per_event, calibration_data, event_level_stats)
from src.plotting import (plot_score_distributions, plot_roc_pr,
                          plot_calibration, plot_severity_timeline,
                          plot_anomaly_overlay, plot_full_dataset_overview)


def main():
    paths = Paths()
    paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths.TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(paths.OUT_DIR / "scores.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    y           = df["label"].to_numpy()
    true_sev    = df["true_severity"].to_numpy()
    ts          = df["timestamp"].to_numpy()
    macro_score = df["GRU_score"].to_numpy()

    score_dict = {
        "MLP":   df["MLP_score"].to_numpy(),
        "KRN":   df["KRN_score"].to_numpy(),
        "LOF":   df["LOF_score"].to_numpy(),
        "GRU": macro_score,
    }

    # ── Layer 1: ranking ──────────────────────────────────────────────────────
    plot_score_distributions(paths.FIG_DIR / "fig1_score_dists.png", y, score_dict)
    plot_roc_pr(paths.FIG_DIR / "fig2_roc.png",
                paths.FIG_DIR / "fig2_pr.png", y, score_dict)

    with open(paths.OUT_DIR / "taus.json") as _f:
        taus_data = json.load(_f)

    rows = []
    for name, s in score_dict.items():
        tau = taus_data[name]["tau"]
        rows.append({
            "model": name,
            "AUROC": round(auroc(y, s), 4),
            "AUPRC": round(auprc(y, s), 4),
            "PA-F1": round(point_adjusted_f1(y, s, tau), 4),
        })
    layer1_table = pd.DataFrame(rows)
    layer1_table.to_csv(paths.TAB_DIR / "table1_ranking.csv", index=False)
    print("\n=== Layer 1: Ranking ===")
    print(layer1_table.to_string(index=False))

    # ── Event-level detection ─────────────────────────────────────────────────
    preds_df = pd.read_csv(paths.OUT_DIR / "preds.csv")
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])
    ts_p = preds_df["timestamp"].to_numpy()
    y_p  = preds_df["label"].to_numpy()

    col_map = {
        "MLP":   "MLP_score",
        "KRN":   "KRN_score",
        "LOF":   "LOF_score",
        "GRU": "GRU_score",
    }
    ev_rows = []
    for name, col in col_map.items():
        tau = taus_data[name]["tau"]
        ev  = event_level_stats(ts_p, y_p, preds_df[col].to_numpy(), tau)
        ev_rows.append({
            "model":           name,
            "tau":             round(tau, 4),
            "detected/total":  f"{ev['detected']}/{ev['n_events']}",
            "event_recall":    round(ev["event_recall"], 3),
            "event_precision": round(ev["event_precision"], 3),
            "event_f1":        round(ev["event_f1"], 3),
            "false_alarms":    ev["false_alarm_clusters"],
        })
    ev_table = pd.DataFrame(ev_rows)
    ev_table.to_csv(paths.TAB_DIR / "table2_event_detection.csv", index=False)
    print("\n=== Event-level detection (test) ===")
    print(ev_table.to_string(index=False))

    # ── Layer 2: intra-event Spearman ─────────────────────────────────────────
    print("\n=== Layer 2: Intra-event Spearman (test) ===")
    l2_rows = []
    for name, s in score_dict.items():
        results = spearman_per_event(ts, true_sev, s, y)
        for r in results:
            r["model"] = name
            print(f"  [{name}] {r['start']} → {r['end']}  "
                  f"r={r['spearman_r']:+.3f}  p={r['p_value']:.3f}")
            l2_rows.append(r)
    pd.DataFrame(l2_rows).to_csv(paths.TAB_DIR / "table3_spearman.csv", index=False)

    # ── Layer 3: calibration (GRU score vs true severity) ───────────────────
    bin_centers, mean_pred, mean_true, counts = calibration_data(
        true_sev, macro_score, n_bins=10
    )
    plot_calibration(paths.FIG_DIR / "fig3_calibration.png",
                     bin_centers, mean_pred, mean_true, counts)
    cal_df = pd.DataFrame({
        "bin_center":         bin_centers.round(2),
        "mean_predicted":     np.round(mean_pred, 4),
        "mean_true_severity": np.round(mean_true, 4),
        "count":              counts,
    })
    cal_df.to_csv(paths.TAB_DIR / "table4_calibration.csv", index=False)
    print("\n=== Layer 3: Calibration (GRU) ===")
    print(cal_df.dropna().to_string(index=False))

    # ── Severity timeline plots (GRU score) ─────────────────────────────────
    from src.metrics import find_event_spans
    spans = find_event_spans(ts, y)
    pad   = np.timedelta64(24, "h")

    event_names = {
        0: "Christmas 2014",
        1: "New Year's Eve 2014–15",
        2: "January Blizzard (Juno) 2015",
    }

    for i, (s_ts, e_ts) in enumerate(spans):
        mask = (ts >= s_ts - pad) & (ts <= e_ts + pad)
        if mask.sum() < 2:
            continue
        name  = event_names.get(i, f"Event {i+1}")
        title = f"Event {i+1}: {name}  ({str(s_ts)[:10]} → {str(e_ts)[:10]})"
        plot_severity_timeline(
            out_path   = paths.FIG_DIR / f"fig4_event{i+1}_severity.png",
            ts         = ts[mask],
            true_score = true_sev[mask],
            pred_score = macro_score[mask],
            y_label    = y[mask],
            title      = title,
        )
        print(f"Saved: fig4_event{i+1}_severity.png")

    # ── Diagnostic overlay: demand + GT + model flags ─────────────────────────
    raw_df = pd.read_csv(paths.DATA_CSV, parse_dates=["timestamp"])
    raw_df = raw_df.set_index("timestamp")["value"]

    ts_dt_index = pd.to_datetime(preds_df["timestamp"])
    demand      = raw_df.reindex(ts_dt_index).to_numpy()

    pred_dict = {
        name: preds_df[f"{name}_pred"].to_numpy()
        for name in ["MLP", "KRN", "LOF", "GRU"]
    }

    plot_anomaly_overlay(
        out_path  = paths.FIG_DIR / "fig_overlay.png",
        ts        = ts_dt_index.to_numpy(),
        demand    = demand,
        y_true    = y_p,
        pred_dict = pred_dict,
    )

    # ── Full-dataset overview ─────────────────────────────────────────────────
    import pickle
    with open(paths.OUT_DIR / "split.pkl", "rb") as _f:
        _split_obj = pickle.load(_f)
    _split = _split_obj["split"]

    full_raw = pd.read_csv(paths.DATA_CSV, parse_dates=["timestamp"])
    full_raw.columns = [c.strip().lower() for c in full_raw.columns]
    full_raw = full_raw.sort_values("timestamp")

    val_start  = pd.to_datetime(_split.ts_val[0])
    test_start = pd.to_datetime(_split.ts_test[0])

    plot_full_dataset_overview(
        out_path         = paths.FIG_DIR / "fig_full_dataset.png",
        ts               = full_raw["timestamp"].to_numpy(),
        demand           = full_raw["value"].to_numpy(),
        y_true           = full_raw["label"].to_numpy(),
        split_boundaries = {"val_start": val_start, "test_start": test_start},
    )

    print("\nAll outputs saved to:", paths.FIG_DIR, "and", paths.TAB_DIR)


if __name__ == "__main__":
    main()
