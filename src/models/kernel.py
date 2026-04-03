"""
Gaussian Kernel Regression detector (Nadaraya–Watson estimator).

Learns E[demand | sin_hour, cos_hour, sin_dow, cos_dow] from the normal
training windows using a product Gaussian kernel over the cyclic time
features.  At inference the model returns the FROZEN seasonal expectation —
it never updates with observed demand values, which is the key architectural
advantage over SARIMA.

Anomaly score for a window = sigmoid of the mean absolute deviation from the
kernel-predicted expected demand, averaged over all 48 timesteps.  The
threshold τ is chosen by argmax F1 on the validation set (pipeline step 04).
"""

import numpy as np
import pickle
from typing import Optional


class GaussianKernelDetector:
    """
    Non-parametric seasonal baseline: Nadaraya–Watson kernel regression
    over cyclic time-of-day and day-of-week features.

    Why it avoids the SARIMA adaptation problem
    -------------------------------------------
    SARIMA runs a Kalman filter at inference time, updating its state with
    each observed demand value.  After 24 h of anomalous demand it has
    essentially re-learnt the anomaly as normal.

    This model has NO recurrent state and NO observed-value input at
    inference.  The lookup table of expected demand values is fixed at fit
    time from clean training data.  A sustained demand suppression (e.g.
    Christmas, blizzard) remains a large deviation throughout the entire
    event, not just on day one.

    Scoring
    -------
    window_score = sigmoid( (mean_abs_err - μ) / σ )

    where μ, σ are the mean and std of window-level errors computed on
    val-normal windows (refit_normalization), anchoring the sigmoid to the
    actual deployment-period noise floor.
    """

    def __init__(self,
                 bandwidth_hour: float = 0.5,
                 bandwidth_dow:  float = 0.8):
        """
        Parameters
        ----------
        bandwidth_hour : float
            Gaussian bandwidth for the (sin_h, cos_h) part of the kernel.
            In unit-circle feature space: 0.5 ≈ ±3 half-hour slots.
        bandwidth_dow : float
            Gaussian bandwidth for the (sin_d, cos_d) part.
            0.8 ≈ adjacent days retain ~60 % weight.
        """
        self.bw_h = bandwidth_hour
        self.bw_d = bandwidth_dow

        # Populated by fit()
        self._tr_feat:   Optional[np.ndarray] = None  # (N_pts, 4) float32
        self._tr_val:    Optional[np.ndarray] = None  # (N_pts,)   float32
        self.lookup_feat: Optional[np.ndarray] = None  # (336, 4)  — all unique time slots
        self.lookup_val:  Optional[np.ndarray] = None  # (336,)    — expected demand per slot

        self.err_mean: float = 0.0
        self.err_std:  float = 1.0

    # ── kernel helpers ────────────────────────────────────────────────────────

    def _kernel_weights(self,
                        q: np.ndarray,
                        tr: np.ndarray,
                        batch_q: int = 256) -> np.ndarray:
        """
        Product Gaussian kernel  K_h(q, tr) * K_d(q, tr).

        q  : (M, 4)  query features
        tr : (N, 4)  training features
        Returns (M,) kernel-weighted mean of self._tr_val — avoids storing
        the full (M, N) matrix by batching over q.
        """
        M = len(q)
        result = np.empty(M, dtype=np.float32)
        for start in range(0, M, batch_q):
            qb = q[start : start + batch_q].astype(np.float64)  # (B, 4)
            # hour part: features 0-1
            dh = qb[:, None, :2] - tr[None, :, :2]  # (B, N, 2)
            Kh = np.exp(-0.5 * (dh ** 2).sum(-1) / (self.bw_h ** 2 + 1e-12))
            # dow part: features 2-3
            dd = qb[:, None, 2:] - tr[None, :, 2:]  # (B, N, 2)
            Kd = np.exp(-0.5 * (dd ** 2).sum(-1) / (self.bw_d ** 2 + 1e-12))
            W = Kh * Kd                               # (B, N)
            denom = W.sum(axis=1) + 1e-12          # (B,)
            result[start : start + batch_q] = (
                (W @ self._tr_val.astype(np.float64)) / denom
            ).astype(np.float32)
        return result

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray) -> "GaussianKernelDetector":
        """
        Build the frozen seasonal lookup table from normal training windows.

        Parameters
        ----------
        X_train : (N, W, D)
            D=5 — channel 0 is standardized demand; channels 1–4 are
            sin_hour, cos_hour, sin_dow, cos_dow.
        """
        N, W, D = X_train.shape
        flat = X_train.reshape(-1, D).astype(np.float32)
        self._tr_feat = flat[:, 1:]   # (N*W, 4)
        self._tr_val  = flat[:, 0]    # (N*W,)

        # Build one query vector per unique (dow, slot) combination.
        # There are 7 × 48 = 336 such combinations — every possible
        # 30-minute slot of the week.
        queries = []
        for dow in range(7):
            for slot in range(48):
                hour  = slot / 2.0
                queries.append([
                    np.sin(2 * np.pi * hour / 24.0),
                    np.cos(2 * np.pi * hour / 24.0),
                    np.sin(2 * np.pi * dow  / 7.0),
                    np.cos(2 * np.pi * dow  / 7.0),
                ])
        self.lookup_feat = np.array(queries, dtype=np.float32)  # (336, 4)

        print(f"[KRN] Computing kernel estimates for 336 time slots "
              f"from {len(self._tr_feat):,} training timesteps …")
        self.lookup_val = self._kernel_weights(self.lookup_feat, self._tr_feat)

        # Error stats from training residuals (window-mean MAE)
        tr_pred     = self._pointwise_predict(self._tr_feat)
        tr_err      = np.abs(self._tr_val - tr_pred).reshape(N, W).mean(axis=1)
        self.err_mean = float(tr_err.mean())
        self.err_std  = float(tr_err.std())
        print(f"[KRN] Done.  err_mean={self.err_mean:.4f}  err_std={self.err_std:.4f}")
        return self

    def _feat_to_idx(self, feat: np.ndarray) -> np.ndarray:
        """
        Map each (M, 4) feature vector to its nearest lookup entry index.
        Since every observed timestep has an exact (dow, slot) equivalent in
        the 336-entry table, this is essentially an exact lookup.
        """
        # (M, 336) squared distances — stays small: M*336*4 floats
        diff  = feat[:, None, :].astype(np.float32) - self.lookup_feat[None, :, :]
        dists = (diff ** 2).sum(-1)
        return dists.argmin(axis=1)  # (M,)

    def _pointwise_predict(self, feat: np.ndarray) -> np.ndarray:
        """Expected demand for each row of feat using the precomputed lookup."""
        idx = self._feat_to_idx(feat)
        return self.lookup_val[idx]

    # ── scoring ───────────────────────────────────────────────────────────────

    def _raw_window_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Mean absolute error (in standardised demand units) per window.
        Returns (N,) before sigmoid normalisation.
        """
        N, W, D = X.shape
        flat  = X.reshape(-1, D).astype(np.float32)
        pred  = self._pointwise_predict(flat[:, 1:])
        err   = np.abs(flat[:, 0] - pred).reshape(N, W)
        return err.mean(axis=1)   # (N,)  — mean over 48 timesteps

    def pointwise_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample anomaly score using the last timestep of each window.

        This is KRN's natural operating mode: compare the final slot's actual
        demand directly against its frozen seasonal expectation.  No averaging
        over a 48-step window — that diluted the onset signal with 47 normal
        steps.  Here the score reacts the moment a slot's demand is anomalous.

        Returns (N,) scores in (0, 1).
        """
        last = X[:, -1, :].astype(np.float32)       # (N, D)
        pred = self._pointwise_predict(last[:, 1:])  # (N,)
        err  = np.abs(last[:, 0] - pred)             # (N,)
        z    = (err - self.err_mean) / (self.err_std + 1e-8)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def refit_normalization_pointwise(self, X_val: np.ndarray,
                                      y_val: np.ndarray) -> None:
        """
        Refit err_mean / err_std on per-slot errors from val-normal samples.

        Anchors the sigmoid to the deployment-period noise floor — identical
        rationale to the original window refit but at slot granularity.
        """
        last        = X_val[:, -1, :].astype(np.float32)
        pred        = self._pointwise_predict(last[:, 1:])
        err         = np.abs(last[:, 0] - pred)
        normal_errs = err[y_val == 0]
        if len(normal_errs) > 0:
            self.err_mean = float(normal_errs.mean())
            self.err_std  = float(normal_errs.std())
            print(f"[KRN] Pointwise normalization refitted on "
                  f"{(y_val == 0).sum()} val-normal slots: "
                  f"err_mean={self.err_mean:.4f}  err_std={self.err_std:.4f}")

    def window_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Per-window anomaly scores in (0, 1).

        Parameters
        ----------
        X : (N, W, D)

        Returns
        -------
        scores : (N,)  high = anomalous
        """
        win_err = self._raw_window_errors(X)
        z       = (win_err - self.err_mean) / (self.err_std + 1e-8)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def refit_normalization(self, X_val: np.ndarray,
                            y_val:  np.ndarray) -> None:
        """
        Refit err_mean / err_std on val-normal windows.

        Anchors the sigmoid to the deployment-period noise floor rather than
        the training-period reconstruction error — same rationale as the
        SARIMA window-max refit.
        """
        raw = self._raw_window_errors(X_val)
        normal_errs = raw[y_val == 0]
        if len(normal_errs) > 0:
            self.err_mean = float(normal_errs.mean())
            self.err_std  = float(normal_errs.std())
            print(f"[KRN] Normalization refitted on {(y_val==0).sum()} "
                  f"val-normal windows: "
                  f"err_mean={self.err_mean:.4f}  err_std={self.err_std:.4f}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[KRN] Saved: {path}")

    @staticmethod
    def load(path: str) -> "GaussianKernelDetector":
        with open(path, "rb") as f:
            return pickle.load(f)
