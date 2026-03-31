import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve, f1_score)


# ── Layer 1: ranking metrics ──────────────────────────────────────────────────

def auroc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))

def auprc(y_true, y_score):
    return float(average_precision_score(y_true, y_score))

def roc_points(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr

def pr_points(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    return prec, rec, thr


def point_adjusted_f1(y_true: np.ndarray, y_score: np.ndarray, tau: float) -> float:
    """
    Point-adjusted F1 (PA-F1) — standard in time-series anomaly detection papers.

    Rule: if the model predicts ≥1 positive within a contiguous anomaly event,
    ALL points in that event are credited as true positives.

    This matches how METER and most TSAD papers report baseline F1 on chunked
    multi-day anomaly events: detecting the event once is operationally
    sufficient credit (you don't need to fire every 30 minutes for 4 days).

    Parameters
    ----------
    y_true  : (N,) binary ground-truth labels
    y_score : (N,) continuous anomaly scores
    tau     : detection threshold (score >= tau → predicted positive)

    Returns
    -------
    pa_f1 : float
    """
    y_pred = (y_score >= tau).astype(int)
    adj    = y_pred.copy()
    n, i   = len(y_true), 0
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if y_pred[i:j].any():
                adj[i:j] = 1          # credit for the entire event
            i = j
        else:
            i += 1
    return float(f1_score(y_true, adj))


# ── Threshold selection ───────────────────────────────────────────────────────

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """Sweep the PR curve and return (tau*, f1, precision, recall) at max F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    i  = int(np.argmax(f1))
    return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])


def best_combined_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """
    Select τ by maximising Youden's J (from ROC) + F1 (from PR curve).

    Why this beats argmax F1 alone when val has very few positive samples
    -------------------------------------------------------------------
    argmax F1 uses precision in its denominator, which collapses when positives
    are rare — the optimum gets pinned to the lowest τ that catches the single
    val event, ignoring false alarm density entirely.

    Youden's J = TPR - FPR uses FPR, which is computed over *all normal samples*
    (thousands of val-normal slots).  It has full statistical power regardless of
    how few positive events exist.  Adding J to F1 breaks the pin by penalising
    high FPR — the combined argmax finds a τ that balances true-positive rate
    against the false-alarm rate, not just against precision from few positives.

    Returns (tau*, combined_score, precision, recall)
    """
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)
    fpr,  tpr, thr_roc = roc_curve(y_true, y_score)

    # precision_recall_curve returns len(thr_pr) = len(prec) - 1
    # Use thr_pr directly; prec/rec need [:-1] to match
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)

    # Interpolate Youden's J onto the PR threshold grid
    # roc_curve: thr_roc[0] is highest, thr_roc[-1] is lowest — must sort ascending
    sort_idx = np.argsort(thr_roc)
    j_at_pr  = np.interp(thr_pr,
                         thr_roc[sort_idx],
                         (tpr - fpr)[sort_idx])

    combined = j_at_pr + f1
    i        = int(np.argmax(combined))
    return float(thr_pr[i]), float(combined[i]), float(prec[i]), float(rec[i])


def best_fbeta_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float):
    """
    Sweep the PR curve and return (tau*, fbeta, precision, recall) at max F-beta.

    beta > 1 weights recall more than precision:
      beta = 2.0  → recall weighted 4×   (τ_micro: false micro alerts are free)
      beta = 1.5  → recall weighted 2.25× (τ_macro: GRU already filtered spikes)
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    b2 = beta ** 2
    fb = (1 + b2) * prec[:-1] * rec[:-1] / (b2 * prec[:-1] + rec[:-1] + 1e-12)
    i  = int(np.argmax(fb))
    return float(thr[i]), float(fb[i]), float(prec[i]), float(rec[i])

def apply_threshold(y_score: np.ndarray, tau: float):
    return (y_score >= tau).astype(int)


# ── Event utilities ───────────────────────────────────────────────────────────

def find_event_spans(ts: np.ndarray, y: np.ndarray):
    """
    Find contiguous runs of label=1 in the timestamp array.

    Returns list of (start_ts, end_ts) tuples — one per anomaly event.
    """
    spans = []
    in_event = False
    start_ts = None
    for t, label in zip(ts, y):
        if label == 1 and not in_event:
            in_event = True
            start_ts = t
        elif label == 0 and in_event:
            in_event = False
            spans.append((start_ts, t))
    if in_event:
        spans.append((start_ts, ts[-1]))
    return spans

def event_level_stats(ts: np.ndarray, y: np.ndarray,
                      y_score: np.ndarray, tau: float):
    """
    Compute event-level precision, recall, F1.

    An event is detected if at least one timestamp within its span is above tau.
    A false alarm is a contiguous block of predictions above tau outside any event.
    """
    spans    = find_event_spans(ts, y)
    pred     = y_score >= tau
    detected = 0
    for s, e in spans:
        mask = (ts >= s) & (ts <= e)
        if pred[mask].any():
            detected += 1

    event_recall = detected / max(len(spans), 1)

    # false alarm clusters: contiguous runs of pred=1 outside any event span
    in_event_mask = np.zeros(len(ts), dtype=bool)
    for s, e in spans:
        in_event_mask |= (ts >= s) & (ts <= e)
    outside_pred = pred & ~in_event_mask
    fa_clusters  = int(np.diff(np.concatenate([[0], outside_pred.astype(int), [0]])).clip(0).sum())

    event_precision = detected / max(detected + fa_clusters, 1)
    event_f1 = (2 * event_precision * event_recall /
                max(event_precision + event_recall, 1e-12))

    return {
        "n_events":         len(spans),
        "detected":         detected,
        "event_recall":     event_recall,
        "event_precision":  event_precision,
        "event_f1":         event_f1,
        "false_alarm_clusters": fa_clusters,
    }


# ── Layer 2: intra-event severity tracking ────────────────────────────────────

def spearman_per_event(ts: np.ndarray, true_severity: np.ndarray,
                       pred_score: np.ndarray, y: np.ndarray):
    """
    For each anomaly event, compute Spearman rank correlation between
    the model's predicted score and the ground-truth anomaly_score.

    Returns list of dicts: {start, end, n_points, spearman_r, p_value}
    """
    spans   = find_event_spans(ts, y)
    results = []
    for s, e in spans:
        mask = (ts >= s) & (ts <= e)
        if mask.sum() < 3:
            continue
        r, p = spearmanr(pred_score[mask], true_severity[mask])
        results.append({
            "start":      str(s)[:16],
            "end":        str(e)[:16],
            "n_points":   int(mask.sum()),
            "spearman_r": round(float(r), 4),
            "p_value":    round(float(p), 4),
        })
    return results


# ── Layer 3: calibration ──────────────────────────────────────────────────────

def calibration_data(true_severity: np.ndarray, pred_mean: np.ndarray,
                     n_bins: int = 10):
    """
    Bin predictions by predicted mean score, compute actual mean severity in each bin.

    Returns (bin_centers, mean_pred, mean_true, counts) for plotting.
    """
    bins        = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_pred   = np.full(n_bins, np.nan)
    mean_true   = np.full(n_bins, np.nan)
    counts      = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (pred_mean >= bins[i]) & (pred_mean < bins[i + 1])
        if mask.sum() > 0:
            mean_pred[i] = pred_mean[mask].mean()
            mean_true[i] = true_severity[mask].mean()
            counts[i]    = mask.sum()

    return bin_centers, mean_pred, mean_true, counts
