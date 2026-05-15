import numpy as np


class Engine:
    """
    Geometry-Aware Precision Agent for PCAM P-04.

    Design (grounded in Theorem F3):
    The scaled Hessian Π^½ R Π^½ is isotropic when π_i ∝ 1/R_ii.
    Setting precision inversely proportional to the diagonal of R
    makes every π_i * R_ii equal to the same constant → spread = 1.

    Combined with class-conditional query trust for retrieval accuracy.
    """

    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        if not isinstance(stored_patterns, np.ndarray) or stored_patterns.ndim != 2:
            raise ValueError("stored_patterns must be a 2D numpy array (K, N)")

        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        # ── Anisotropy-optimal base precision ────────────────────────────────
        # If the R operator (Hessian surrogate) is provided, use its diagonal
        # to set π_i = C / R_ii → π_i × R_ii = C (perfectly isotropic).
        if "R" in model_params:
            R = np.array(model_params["R"])              # (N, N)
            r_diag = np.diag(R) + 1e-8                  # (N,)
            # Optimal precision: π_i = C / R_ii so that π_i × R_ii = C (constant).
            # We pick C = mean(R_ii) so the mean precision ≈ 1.
            C = np.mean(r_diag)
            raw = C / r_diag                             # (N,) — exact inverse curvature
        else:
            # Fallback: proportional to variance (≈ C/H_ii)
            var = np.var(self.X, axis=0) + 1e-8
            raw = np.mean(var) / var

        # Clip to harness bounds WITHOUT linear rescaling — this preserves the
        # ratio-cancellation property: π_i × R_ii ≈ C after clipping.
        self._base_pi = np.clip(raw, self.pi_min, self.pi_max)   # (N,)


        # ── Retrieval: class-conditional setup ───────────────────────────────
        # Per-dim variance for Mahalanobis distance
        self._var = np.var(self.X, axis=0) + 1e-8       # (N,)
        self._sigma = np.std(self.X) + 1e-8

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """
        Returns 64 positive precision values in [pi_min, pi_max].
        """
        q = np.asarray(corrupted_query)
        if q.shape != (self.N,):
            raise ValueError(f"Expected ({self.N},), got {q.shape}")

        # ── Class-conditional trust ──────────────────────────────────────────
        # Find nearest stored attractor via Mahalanobis-style distance
        dists = np.sum((self.X - q) ** 2 / self._var, axis=1)   # (K,)
        nearest = self.X[np.argmin(dists)]                        # (N,)

        # Per-dimension trust: near-zero error → trust ≈ 1; large error → trust → 0
        error = np.abs(q - nearest)
        trust = np.exp(-error / self._sigma)                      # (N,)

        # ── Blend: geometry-optimal base × query trust ───────────────────────
        # High trust on this dimension → use full base precision
        # Low  trust (corrupted dim)  → pull precision toward pi_min to suppress noise
        precision = self._base_pi * trust + self.pi_min * (1.0 - trust)

        return np.clip(precision, self.pi_min, self.pi_max)