"""
Script 05 – Evaluation and visualisation.

Outputs
-------
outputs/tables/table1.csv       AUROC, AUPRC, Detection Delay for all models
outputs/figures/fig1_score_dists.png
outputs/figures/fig2_roc.png
outputs/figures/fig2_pr.png
outputs/figures/fig3_timeseries.png   Anomaly scores over time (test split)
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths
from src.metrics import auroc, auprc, detection_delay, apply_threshold, best_f1_threshold
from src.plotting import plot_score_distributions, plot_roc_pr

MODEL_NAMES = [
    "LOF", "IsoForest", "KNN", "Incremental",
    "GRU", "LSTM", "CNN", "Transformer",
    "VAE", "ENS",
]

MODEL_FAMILY = {
    "LOF":         "Traditional",
    "IsoForest":   "Traditional",
    "KNN":         "Traditional",
    "Incremental": "Incremental",
    "GRU":         "Deep Temporal",
    "LSTM":        "Deep Temporal",
    "CNN":         "Deep Temporal",
    "Transformer": "Deep Temporal",
    "VAE":         "Reconstruction",
    "ENS":         "Ensemble",
}


def plot_timeseries_scores(out_path, timestamps, y_true, score_dict, models_to_plot=None):
    """Plot anomaly scores over time for selected models on the test split."""
    if models_to_plot is None:
        models_to_plot = list(score_dict.keys())

    fig, axes = plt.subplots(
        len(models_to_plot) + 1, 1,
        figsize=(14, 2.5 * (len(models_to_plot) + 1)),
        sharex=True,
    )

    # Top panel: raw value signal with anomaly shading
    ax0 = axes[0]
    ax0.set_ylabel("Anomaly label", fontsize=9)
    ax0.fill_between(range(len(y_true)), y_true, alpha=0.4, color="red", label="Anomaly")
    ax0.set_ylim(-0.1, 1.5)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.set_title("Ground-truth anomaly labels (test split)", fontsize=10)

    for ax, name in zip(axes[1:], models_to_plot):
        s = score_dict[name]
        ax.plot(s, linewidth=0.7, color="steelblue", label=name)
        ax.fill_between(range(len(y_true)), 0, y_true.astype(float), alpha=0.2, color="red")
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time step (30-min intervals)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    paths = Paths()
    paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths.TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(paths.OUT_DIR / "scores.csv")
    y = df["label"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    score_dict = {name: df[f"{name}_score"].to_numpy() for name in MODEL_NAMES}

    # ------------------------------------------------------------------ #
    # Figures                                                             #
    # ------------------------------------------------------------------ #
    plot_score_distributions(paths.FIG_DIR / "fig1_score_dists.png", y, score_dict)
    plot_roc_pr(
        paths.FIG_DIR / "fig2_roc.png",
        paths.FIG_DIR / "fig2_pr.png",
        y,
        score_dict,
    )
    # Time-series plot: show a representative subset of models
    ts_models = ["LOF", "IsoForest", "GRU", "LSTM", "CNN", "Transformer", "VAE", "ENS"]
    plot_timeseries_scores(
        paths.FIG_DIR / "fig3_timeseries.png",
        timestamps,
        y,
        {k: score_dict[k] for k in ts_models},
        models_to_plot=ts_models,
    )
    print("Saved figures in:", paths.FIG_DIR)

    # ------------------------------------------------------------------ #
    # Table: AUROC, AUPRC, Detection Delay                               #
    # ------------------------------------------------------------------ #
    rows = []
    for name in MODEL_NAMES:
        s = score_dict[name]
        tau, _, _, _ = best_f1_threshold(y, s)
        preds = apply_threshold(s, tau)
        delay = detection_delay(y, preds)
        rows.append({
            "model":            name,
            "family":           MODEL_FAMILY[name],
            "AUROC":            round(auroc(y, s), 4),
            "AUPRC":            round(auprc(y, s), 4),
            "detection_delay":  round(delay, 2) if not np.isnan(delay) else "N/A",
        })

    table = pd.DataFrame(rows)
    out_table = paths.TAB_DIR / "table1.csv"
    table.to_csv(out_table, index=False)
    print("Saved:", out_table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
