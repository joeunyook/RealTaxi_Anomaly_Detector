"""
Traditional (static, pointwise) anomaly detection baselines.

Models
------
IsolationForestDetector
    Ensemble of random isolation trees; anomaly score = average path length.
    Treats each window independently without explicit temporal modeling.

KNNDetector
    k-Nearest Neighbours distance-based anomaly score.
    Anomaly score = mean distance to the k nearest training neighbours.

Both detectors:
  - Accept sliding windows X of shape (N, W, D).
  - Flatten windows to (N, W*D) before fitting / scoring.
  - Return scores normalized to [0, 1] via MinMaxScaler fitted on train data.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    contamination : float
        Expected fraction of anomalies (used internally by sklearn).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        Z = self._flatten(X_train)
        self.model.fit(Z)
        # score_samples returns negative anomaly scores; negate so higher = more anomalous
        raw = -self.model.score_samples(Z)
        self.scaler.fit(raw.reshape(-1, 1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self._flatten(X)
        raw = -self.model.score_samples(Z)
        return self.scaler.transform(raw.reshape(-1, 1)).reshape(-1)


class KNNDetector:
    """
    k-Nearest Neighbours anomaly detector.

    Anomaly score for a test window is the mean Euclidean distance to its
    k nearest neighbours in the training set.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours.
    """

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray) -> "KNNDetector":
        Z = self._flatten(X_train)
        self.nn.fit(Z)
        # Fit score normalizer on train distances
        dists, _ = self.nn.kneighbors(Z)
        raw = dists.mean(axis=1)
        self.scaler.fit(raw.reshape(-1, 1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self._flatten(X)
        dists, _ = self.nn.kneighbors(Z)
        raw = dists.mean(axis=1)
        return self.scaler.transform(raw.reshape(-1, 1)).reshape(-1)
