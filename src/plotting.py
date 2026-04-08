import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
from src.metrics import roc_points, pr_points


# ── colour palette ────────────────────────────────────────────────────────────
_DEMAND_COLOR   = "#2c3e50"   # dark navy  — demand line
_GT_COLOR       = "#e74c3c"   # red        — ground truth anomaly region
_TP_COLOR       = "#27ae60"   # green      — true positive flag
_FP_COLOR       = "#f39c12"   # orange     — false positive flag
_FN_COLOR       = "#e74c3c"   # red        — false negative (missed)


def plot_score_distributions(out_path, y_true, score_dict):
    plt.figure(figsize=(10, 6))
    for name, s in score_dict.items():
        s0 = s[y_true == 0]
        s1 = s[y_true == 1]
        plt.hist(s0, bins=60, alpha=0.25, density=True, label=f"{name} normal")
        plt.hist(s1, bins=60, alpha=0.25, density=True, label=f"{name} anomaly")
    plt.xlabel("Normalized score [0,1]")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_pr(out_path_roc, out_path_pr, y_true, score_dict):
    _FS = 40      # 300 % increase over matplotlib default (~10 pt)
    _FS_AX = 20  # axis labels and ticks halved

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_points(y_true, s)
        ax.plot(fpr, tpr, label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlabel("FPR", fontsize=_FS_AX)
    ax.set_ylabel("TPR", fontsize=_FS_AX)
    ax.tick_params(axis="both", labelsize=_FS_AX)
    ax.legend(fontsize=_FS // 4)
    fig.tight_layout()
    fig.savefig(out_path_roc, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, s in score_dict.items():
        prec, rec, _ = pr_points(y_true, s)
        ax.plot(rec, prec, label=name, linewidth=2)
    ax.set_xlabel("Recall", fontsize=_FS_AX)
    ax.set_ylabel("Precision", fontsize=_FS_AX)
    ax.tick_params(axis="both", labelsize=_FS_AX)
    ax.legend(fontsize=_FS // 4)
    fig.tight_layout()
    fig.savefig(out_path_pr, dpi=200)
    plt.close(fig)


def plot_calibration(out_path, bin_centers, mean_pred, mean_true, counts):
    valid = ~np.isnan(mean_pred) & ~np.isnan(mean_true)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(mean_pred[valid], mean_true[valid],
                    s=counts[valid] / counts[valid].max() * 300 + 20,
                    c=counts[valid], cmap="Blues", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    plt.colorbar(sc, ax=ax, label="window count in bin")
    ax.set_xlabel("Mean RNN seasonal deviation score")
    ax.set_ylabel("Mean true anomaly_score")
    ax.set_title("Calibration — GRU score vs ground truth")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_severity_timeline(out_path, ts, true_score, pred_score,
                           y_label, title="Severity timeline"):
    ts_dt = pd.to_datetime(ts)
    fig, ax = plt.subplots(figsize=(14, 5))
    in_anom = False
    for i, lab in enumerate(y_label):
        if lab == 1 and not in_anom:
            span_start = ts_dt[i]; in_anom = True
        elif lab == 0 and in_anom:
            ax.axvspan(span_start, ts_dt[i], alpha=0.08, color=_GT_COLOR)
            in_anom = False
    if in_anom:
        ax.axvspan(span_start, ts_dt[-1], alpha=0.08, color=_GT_COLOR)
    ax.plot(ts_dt, true_score, color="black", linewidth=1.5,
            label="True anomaly_score", zorder=3)
    ax.plot(ts_dt, pred_score, color="steelblue", linewidth=1.5,
            label="GRU seasonal deviation score", zorder=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Time");  ax.set_ylabel("Anomaly score")
    ax.set_title(title);    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_anomaly_overlay(out_path, ts, demand, y_true, pred_dict):
    """
    Diagnostic plot: one subplot per model showing the full test demand
    time series with ground truth anomaly regions and model flags overlaid.

    Colour legend (consistent across all subplots):
      Dark navy line  — actual passenger demand
      Red shading     — ground truth anomaly period (label == 1)
      Green triangle  — model correctly flagged (true positive)
      Orange cross    — model incorrectly flagged (false positive / false alarm)

    Parameters
    ----------
    ts        : (N,)  timestamps (test set)
    demand    : (N,)  raw passenger counts
    y_true    : (N,)  binary ground truth labels
    pred_dict : {model_name: binary_predictions (N,)}
    """
    ts_dt   = pd.to_datetime(ts)
    models  = list(pred_dict.keys())
    n       = len(models)

    fig, axes = plt.subplots(n, 1, figsize=(40, 8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # ── shared legend patches ─────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_GT_COLOR, alpha=0.25,  label="Ground truth anomaly"),
        plt.Line2D([0], [0], color=_DEMAND_COLOR, linewidth=1.2, label="Demand"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=_TP_COLOR,
                   markersize=8, label="True positive (correct flag)"),
        plt.Line2D([0], [0], marker="x", color=_FP_COLOR, markersize=8,
                   markeredgewidth=2, label="False positive (false alarm)"),
    ]

    for ax, model_name in zip(axes, models):
        pred = pred_dict[model_name]

        # ── ground truth shading ──────────────────────────────────────────
        in_anom = False
        for i, lab in enumerate(y_true):
            if lab == 1 and not in_anom:
                span_start = ts_dt[i]; in_anom = True
            elif lab == 0 and in_anom:
                ax.axvspan(span_start, ts_dt[i],
                           alpha=0.25, color=_GT_COLOR, zorder=1)
                in_anom = False
        if in_anom:
            ax.axvspan(span_start, ts_dt[-1],
                       alpha=0.25, color=_GT_COLOR, zorder=1)

        # ── demand line ───────────────────────────────────────────────────
        ax.plot(ts_dt, demand, color=_DEMAND_COLOR, linewidth=0.9,
                alpha=0.85, zorder=2)

        # ── true positives (model=1, label=1) ────────────────────────────
        tp_mask = (pred == 1) & (y_true == 1)
        if tp_mask.any():
            ax.scatter(ts_dt[tp_mask], demand[tp_mask],
                       marker="^", color=_TP_COLOR, s=40,
                       zorder=4, label="_tp")

        # ── false positives (model=1, label=0) ───────────────────────────
        fp_mask = (pred == 1) & (y_true == 0)
        if fp_mask.any():
            ax.scatter(ts_dt[fp_mask], demand[fp_mask],
                       marker="x", color=_FP_COLOR, s=40,
                       linewidths=1.8, zorder=4, label="_fp")

        # ── formatting ────────────────────────────────────────────────────
        ax.set_title(model_name, fontsize=72, fontweight="bold", loc="left")
        ax.set_ylabel("Passengers", fontsize=60)
        ax.tick_params(axis="both", labelsize=60)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    # ── x-axis ticks on bottom subplot ───────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=60)
    axes[-1].set_xlabel("Date", fontsize=60)

    # ── shared legend at the top ──────────────────────────────────────────
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=4, fontsize=60, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("Anomaly Detection — Actual Demand vs Model Flags (Test Set)",
                 fontsize=78, fontweight="bold", y=1.03)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
