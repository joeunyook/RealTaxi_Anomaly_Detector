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

# per-model colours used consistently across all plots
_MODEL_COLORS = {
    "MLP":   "#27ae60",   # green  — micro: MLP pointwise seasonal baseline
    "KRN":   "#c0392b",   # red    — micro: Gaussian kernel lookup
    "LOF":   "#2980b9",   # blue   — micro: local outlier factor (5-D)
    "GRU":   "#e67e22",   # orange — GRU aggregating 12h of micro score sequences
}


def plot_score_distributions(out_path, y_true, score_dict):
    """
    One subplot per model showing normal vs anomaly score histograms.
    """
    models = list(score_dict.keys())
    n      = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Anomaly Score Distributions — Normal vs Anomaly Windows (Test Set)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, name in zip(axes, models):
        s     = score_dict[name]
        s0    = s[y_true == 0]
        s1    = s[y_true == 1]
        color = _MODEL_COLORS.get(name, "steelblue")

        # x range: 1st percentile to 99.5th percentile of all scores
        lo = np.percentile(s, 0.5)
        hi = np.percentile(s, 99.5)
        bins = np.linspace(lo, hi, 50)

        ax.hist(s0, bins=bins, alpha=0.55, density=True,
                color=color, label="Normal",  edgecolor="none")
        ax.hist(s1, bins=bins, alpha=0.55, density=True,
                color=_GT_COLOR, label="Anomaly", edgecolor="none")

        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Anomaly score", fontsize=9)
        ax.set_ylabel("Density",       fontsize=9)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_pr(out_path_roc, out_path_pr, y_true, score_dict):
    """ROC and PR curves with AUROC / AUPRC in legend labels."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    # ── ROC ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_points(y_true, s)
        auc_val     = roc_auc_score(y_true, s)
        ax.plot(fpr, tpr,
                color=_MODEL_COLORS.get(name, None),
                linewidth=2,
                label=f"{name}  (AUROC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — Anomaly Detection (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path_roc, dpi=200)
    plt.close()

    # ── PR ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, s in score_dict.items():
        prec, rec, _ = pr_points(y_true, s)
        ap_val       = average_precision_score(y_true, s)
        ax.plot(rec, prec,
                color=_MODEL_COLORS.get(name, None),
                linewidth=2,
                label=f"{name}  (AUPRC={ap_val:.3f})")
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1,
               label=f"Random baseline ({baseline:.2f})")
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Anomaly Detection (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path_pr, dpi=200)
    plt.close()


def plot_calibration(out_path, bin_centers, mean_pred, mean_true, counts):
    valid = ~np.isnan(mean_pred) & ~np.isnan(mean_true)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(mean_pred[valid], mean_true[valid],
                    s=counts[valid] / counts[valid].max() * 300 + 20,
                    c=counts[valid], cmap="Blues", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5,
            label="Perfect calibration (predicted = true)")
    plt.colorbar(sc, ax=ax, label="Number of windows in bin")
    ax.set_xlabel("Mean MACRO anomaly score (predicted)", fontsize=11)
    ax.set_ylabel("Mean true severity (ground truth)",   fontsize=11)
    ax.set_title("Score Calibration — MACRO Predicted Score vs True Severity",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05);  ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_severity_timeline(out_path, ts, true_score, pred_score,
                           y_label, title="Severity timeline"):
    """
    Dual-axis plot: true severity (left y-axis) and GRU score (right y-axis).
    Both curves are readable despite their different scales.
    """
    ts_dt = pd.to_datetime(ts)
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # ── ground truth shading ─────────────────────────────────────────────────
    in_anom = False
    for i, lab in enumerate(y_label):
        if lab == 1 and not in_anom:
            span_start = ts_dt[i]; in_anom = True
        elif lab == 0 and in_anom:
            ax1.axvspan(span_start, ts_dt[i], alpha=0.10, color=_GT_COLOR,
                        zorder=0, label="_gt")
            in_anom = False
    if in_anom:
        ax1.axvspan(span_start, ts_dt[-1], alpha=0.10, color=_GT_COLOR, zorder=0)

    # ── true severity (left axis) ────────────────────────────────────────────
    ax1.plot(ts_dt, true_score, color="black", linewidth=1.8,
             label="True severity score", zorder=3)
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("True severity score", fontsize=11, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # ── GRU score (right axis) ───────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(ts_dt, pred_score, color="steelblue", linewidth=1.8,
             linestyle="--", label="MACRO anomaly score", zorder=4)
    ax2.set_ylabel("MACRO anomaly score", fontsize=11, color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(max(0, pred_score.min() - 0.05), min(1.05, pred_score.max() + 0.05))

    # ── x-axis formatting ────────────────────────────────────────────────────
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── combined legend ──────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # filter out the GT shading dummy handle
    lines1  = [l for l, lb in zip(lines1, labels1) if not lb.startswith("_")]
    labels1 = [lb for lb in labels1 if not lb.startswith("_")]
    gt_patch = mpatches.Patch(color=_GT_COLOR, alpha=0.25, label="Ground truth anomaly")
    ax1.legend(handles=[gt_patch] + lines1 + lines2,
               labels=["Ground truth anomaly"] + labels1 + labels2,
               loc="upper left", fontsize=10, framealpha=0.9)

    ax1.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_anomaly_overlay(out_path, ts, demand, y_true, pred_dict):
    """
    One subplot per model: full test demand time series with ground truth
    anomaly regions and model detection flags overlaid.

    Colour legend (consistent across all subplots):
      Dark navy line  — actual passenger demand
      Red shading     — ground truth anomaly period (label == 1)
      Green triangle  — true positive (model correctly flagged)
      Orange cross    — false positive (false alarm)
    """
    ts_dt   = pd.to_datetime(ts)
    models  = list(pred_dict.keys())
    n       = len(models)

    fig, axes = plt.subplots(n, 1, figsize=(22, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # ── shared legend patches ─────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_GT_COLOR, alpha=0.25, label="Ground truth anomaly"),
        plt.Line2D([0], [0], color=_DEMAND_COLOR, linewidth=1.2, label="Taxi demand"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=_TP_COLOR,
                   markersize=8, label="True positive"),
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

        # ── true positives ────────────────────────────────────────────────
        tp_mask = (pred == 1) & (y_true == 1)
        if tp_mask.any():
            ax.scatter(ts_dt[tp_mask], demand[tp_mask],
                       marker="^", color=_TP_COLOR, s=40, zorder=4)

        # ── false positives ───────────────────────────────────────────────
        fp_mask = (pred == 1) & (y_true == 0)
        if fp_mask.any():
            ax.scatter(ts_dt[fp_mask], demand[fp_mask],
                       marker="x", color=_FP_COLOR, s=40,
                       linewidths=1.8, zorder=4)

        # ── formatting ────────────────────────────────────────────────────
        ax.set_title(model_name, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("Passengers", fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    # ── x-axis ticks on bottom subplot ───────────────────────────────────
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date", fontsize=11)

    # ── shared legend at the top ──────────────────────────────────────────
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=4, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("Model Anomaly Flags vs Ground Truth — Test Set",
                 fontsize=13, fontweight="bold", y=1.03)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_full_dataset_overview(out_path, ts, demand, y_true,
                               split_boundaries=None):
    """
    Single-panel overview of the entire dataset: taxi demand with ground
    truth anomaly regions shaded.  No model flags.

    Parameters
    ----------
    ts               : (N,)  timestamps (full dataset)
    demand           : (N,)  raw passenger counts
    y_true           : (N,)  binary ground truth labels
    split_boundaries : dict with keys 'val_start', 'test_start' (datetime-like)
                       Optional vertical lines marking train / val / test splits.
    """
    ts_dt = pd.to_datetime(ts)
    fig, ax = plt.subplots(figsize=(22, 5))

    # ── ground truth anomaly shading ─────────────────────────────────────
    in_anom   = False
    span_start = None
    first_gt  = True
    for i, lab in enumerate(y_true):
        if lab == 1 and not in_anom:
            span_start = ts_dt[i]; in_anom = True
        elif lab == 0 and in_anom:
            label = "Anomaly event" if first_gt else "_nolegend_"
            ax.axvspan(span_start, ts_dt[i],
                       alpha=0.30, color=_GT_COLOR, zorder=1, label=label)
            in_anom = False; first_gt = False
    if in_anom:
        ax.axvspan(span_start, ts_dt[-1], alpha=0.30, color=_GT_COLOR,
                   zorder=1)

    # ── demand line ──────────────────────────────────────────────────────
    ax.plot(ts_dt, demand, color=_DEMAND_COLOR, linewidth=0.8,
            alpha=0.9, zorder=2, label="Taxi demand")

    # ── split boundary lines ──────────────────────────────────────────────
    if split_boundaries is not None:
        val_start  = pd.to_datetime(split_boundaries.get("val_start"))
        test_start = pd.to_datetime(split_boundaries.get("test_start"))
        ymax = np.nanmax(demand) * 1.02
        ymin = 0
        if val_start is not None:
            ax.axvline(val_start, color="#7f8c8d", linewidth=1.5,
                       linestyle="--", zorder=3)
            ax.text(val_start, ymax * 0.97, "  Val start",
                    color="#7f8c8d", fontsize=9, va="top")
        if test_start is not None:
            ax.axvline(test_start, color="#7f8c8d", linewidth=1.5,
                       linestyle=":", zorder=3)
            ax.text(test_start, ymax * 0.97, "  Test start",
                    color="#7f8c8d", fontsize=9, va="top")

    # ── formatting ───────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Passenger count", fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.set_title("NYC Taxi Demand — Full Dataset with Anomaly Events",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
