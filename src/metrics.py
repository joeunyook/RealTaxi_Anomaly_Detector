"""
Evaluation metrics for anomaly detection.

Metrics
-------
auroc           Area under the ROC curve
auprc           Area under the precision-recall curve
roc_points      (fpr, tpr, thresholds) for ROC plotting
pr_points       (precision, recall, thresholds) for PR plotting
best_f1_threshold   Optimal threshold by F1 score on the PR curve
apply_threshold     Convert continuous scores to binary predictions
detection_delay     Average number of steps between anomaly onset and first detection
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_points(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr


def pr_points(y_true: np.ndarray, y_score: np.ndarray):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    return prec, rec, thr


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """
    Find the threshold that maximises F1 on the PR curve.

    Returns
    -------
    (tau, f1, precision, recall)
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # thr has length len(prec) - 1
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])


def apply_threshold(y_score: np.ndarray, tau: float) -> np.ndarray:
    return (y_score >= tau).astype(int)


def detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_delay: int = 50,
) -> float:
    """
    Compute the average detection delay over all anomaly events.

    An anomaly event is a contiguous run of y_true == 1.  Detection delay
    for an event is the number of time steps between the start of the event
    and the first y_pred == 1 within the event window.  If no prediction
    fires within the event, the delay is capped at ``max_delay``.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        Binary ground-truth labels.
    y_pred : np.ndarray of shape (N,)
        Binary predictions.
    max_delay : int
        Cap applied when no detection occurs within an event.

    Returns
    -------
    mean_delay : float
        Average detection delay in time steps.  Returns NaN if there are
        no anomaly events.
    """
    delays = []
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            # Find the end of this anomaly run
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            # Look for first detection within the event window
            event_preds = y_pred[i:j]
            hits = np.where(event_preds == 1)[0]
            if len(hits) > 0:
                delays.append(int(hits[0]))
            else:
                delays.append(max_delay)
            i = j
        else:
            i += 1

    if len(delays) == 0:
        return float("nan")
    return float(np.mean(delays))
