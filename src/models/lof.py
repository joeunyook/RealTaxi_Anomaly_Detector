import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

class LOFDetector:
    def __init__(self, n_neighbors: int = 35, contamination: float = 0.1):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,  # allows score on new data
        )
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    @staticmethod
    def _flatten_windows(X: np.ndarray) -> np.ndarray:
        # Use the last timestep only: (N, W, D) → (N, D)
        # With D=5: (demand, sin_h, cos_h, sin_d, cos_d) — a 5-D point per slot.
        # LOF is a nearest-neighbour density estimator; the curse of dimensionality
        # makes distances meaningless in the old 240-D space (48×5).  5-D is close
        # to ideal.  The time features already encode position so the full window
        # carries no additional context that LOF could usefully exploit.
        return X[:, -1, :]

    def fit(self, X_train: np.ndarray):
        Z = self._flatten_windows(X_train)
        self.model.fit(Z)
        # fit score normalization on train scores
        raw = -self.model.score_samples(Z)  # higher = more anomalous
        self.scaler.fit(raw.reshape(-1, 1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self._flatten_windows(X)
        raw = -self.model.score_samples(Z)
        s = self.scaler.transform(raw.reshape(-1, 1)).reshape(-1)
        return s  # [0,1]