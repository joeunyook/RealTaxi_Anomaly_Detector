import numpy as np
import pickle
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMADetector:
    """
    SARIMA seasonal baseline anomaly detector.

    Fits SARIMA(p,d,q)(P,D,Q,s) on the sequential normal training series.
    Anomaly score = sigmoid-normalised |actual - 1-step-ahead forecast|.

    Unlike LOF (density in feature space) and the GRU/MLP (learned function),
    SARIMA is a fully statistical model with explicit seasonal parameters.
    It captures the same "wrong demand level for this time of day" signal
    but via a completely different mechanism — making it a useful ensemble
    member whose errors are largely uncorrelated with the neural models.

    Point-level scores are converted to window-level scores via rolling max
    over the window, which gives credit to any timestep within the 24h window
    that shows a large seasonal deviation.
    """

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 48)):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.result         = None
        self.err_mean       = 0.0
        self.err_std        = 1.0

    # ── fitting ───────────────────────────────────────────────────────────

    def fit(self, train_series: np.ndarray) -> "SARIMADetector":
        """
        Fit on the sequential normal training series (raw demand values).

        Parameters
        ----------
        train_series : (T,)  chronological demand values, normal period only
        """
        print(f"[SARIMA] Fitting SARIMA{self.order}×{self.seasonal_order} "
              f"on {len(train_series)} training points …")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train_series,
                order          = self.order,
                seasonal_order = self.seasonal_order,
                enforce_stationarity  = False,
                enforce_invertibility = False,
            )
            self.result = model.fit(disp=False, maxiter=500)

        # Normalisation stats from training residuals.
        # Skip first seasonal period (s steps) to avoid Kalman filter startup bias.
        s     = self.seasonal_order[3]
        resid = np.abs(self.result.resid[s:])
        self.err_mean = float(resid.mean())
        self.err_std  = float(resid.std())
        print(f"[SARIMA] err_mean={self.err_mean:.4f}  err_std={self.err_std:.4f}")
        return self

    # ── scoring ───────────────────────────────────────────────────────────

    def point_scores(self, series: np.ndarray) -> np.ndarray:
        """
        Per-timestep anomaly scores for a sequential series (val or test).

        Uses rolling 1-step-ahead prediction: at each step t the model uses
        the actual observation at t-1 (Kalman filter update), so predictions
        reflect the current demand context without leaking future data.

        Parameters
        ----------
        series : (T,)  sequential demand values

        Returns
        -------
        scores : (T,)  in (0, 1), high = anomalous
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            updated = self.result.append(series, refit=False)

        n_train = len(self.result.fittedvalues)
        pred    = np.asarray(updated.fittedvalues[n_train:])
        err     = np.abs(np.asarray(series) - pred)
        z       = (err - self.err_mean) / (self.err_std + 1e-8)
        return 1.0 / (1.0 + np.exp(-z))    # sigmoid → (0, 1)

    def window_scores(self, series: np.ndarray, n_windows: int, window: int) -> np.ndarray:
        """
        Convert point-level scores to window-level via rolling max.

        Window i covers points series[i : i+window].
        With stride=1 this maps directly to the windowed split indices.

        Parameters
        ----------
        series   : (T,)  sequential demand values (val or test timeline)
        n_windows: number of windows in this split
        window   : window size (W = 48)

        Returns
        -------
        scores : (n_windows,)
        """
        pt = self.point_scores(series)
        return np.array([pt[i : i + window].max() for i in range(n_windows)])

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[SARIMA] Saved: {path}")

    @staticmethod
    def load(path: str) -> "SARIMADetector":
        with open(path, "rb") as f:
            return pickle.load(f)
