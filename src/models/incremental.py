"""
Incremental (online) anomaly detection models.

These models update their internal state continuously as new data arrives,
enabling real-time adaptation to distribution shifts without full retraining.

Models
------
IncrementalMeanStdDetector
    Maintains a running mean and standard deviation of the input signal.
    Anomaly score = z-score of the current observation relative to the
    running statistics.  Parameters are updated with exponential moving
    averages (EMA) to give more weight to recent observations.

    This is a lightweight, interpretable baseline that captures sudden
    deviations from the recent baseline level and variance.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class IncrementalMeanStdDetector:
    """
    Online z-score anomaly detector with exponential moving average statistics.

    The detector processes windows sequentially and updates its running
    mean/variance after each observation using EMA with decay factor alpha.

    Parameters
    ----------
    alpha : float
        EMA decay factor in (0, 1).  Smaller values give more weight to
        historical data; larger values adapt faster to recent changes.
    warmup : int
        Number of initial windows used to initialise statistics before
        scoring begins (scores during warmup are set to 0).
    """

    def __init__(self, alpha: float = 0.05, warmup: int = 48):
        self.alpha = alpha
        self.warmup = warmup
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._mu: float = 0.0
        self._var: float = 1.0
        self._n: int = 0

    def _update(self, v: float) -> None:
        """Update running EMA mean and variance."""
        if self._n == 0:
            self._mu = v
            self._var = 0.0
        else:
            diff = v - self._mu
            self._mu += self.alpha * diff
            self._var = (1 - self.alpha) * (self._var + self.alpha * diff ** 2)
        self._n += 1

    def _zscore(self, v: float) -> float:
        std = max(float(np.sqrt(self._var)), 1e-6)
        return abs(v - self._mu) / std

    def fit(self, X_train: np.ndarray) -> "IncrementalMeanStdDetector":
        """
        Initialise running statistics from training data (first pass).

        Parameters
        ----------
        X_train : np.ndarray of shape (N, W, D)
            Training windows.  Only the value channel (index 0) is used.
        """
        self._mu = 0.0
        self._var = 1.0
        self._n = 0
        # Use the last value of each window as the representative observation
        values = X_train[:, -1, 0].astype(float)
        raw_scores = []
        for v in values:
            z = self._zscore(v) if self._n >= self.warmup else 0.0
            raw_scores.append(z)
            self._update(v)
        raw = np.array(raw_scores, dtype=np.float32).reshape(-1, 1)
        self.scaler.fit(raw)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Score new windows using current running statistics.

        The state is NOT updated during scoring to allow fair evaluation
        on held-out splits.  For true online evaluation, call
        ``score_and_update`` instead.

        Parameters
        ----------
        X : np.ndarray of shape (N, W, D)

        Returns
        -------
        scores : np.ndarray of shape (N,) in [0, 1]
        """
        values = X[:, -1, 0].astype(float)
        raw = np.array([self._zscore(v) for v in values], dtype=np.float32)
        return self.scaler.transform(raw.reshape(-1, 1)).reshape(-1)

    def score_and_update(self, X: np.ndarray) -> np.ndarray:
        """
        Score and update running statistics for each window sequentially.

        Use this method for true online / streaming evaluation.

        Parameters
        ----------
        X : np.ndarray of shape (N, W, D)

        Returns
        -------
        scores : np.ndarray of shape (N,) in [0, 1]
        """
        values = X[:, -1, 0].astype(float)
        raw = []
        for v in values:
            z = self._zscore(v)
            raw.append(z)
            self._update(v)
        raw_arr = np.array(raw, dtype=np.float32)
        return self.scaler.transform(raw_arr.reshape(-1, 1)).reshape(-1)
