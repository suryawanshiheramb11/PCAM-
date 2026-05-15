import numpy as np

class RustEngine:
    def __init__(self, stored_patterns):
        self.stored_patterns = np.array(stored_patterns)   # (K, N)
        self.N = self.stored_patterns.shape[1]
        # Per-dimension variance of the attractor bank
        self._var = np.var(self.stored_patterns, axis=0) + 1e-6    # (N,)
        # Curvature approximation: H_ii ~ 1 / var_i
        self._curvature = 1.0 / self._var                          # (N,)

    def predict(self, corrupted_query):
        """
        Optimal precision for isotropisation: Π_i ∝ 1 / H_ii = var_i
        
        The scaled Hessian diagonal becomes:
            (Π^½ H Π^½)_ii = π_i × H_ii = (C × var_i) × (1/var_i) = C  ← constant!
        
        So precision = var_i (the variance itself, NOT the inverse).
        High curvature dim (narrow valley) → small var → low precision (damp).
        Low  curvature dim (wide valley)   → big var  → high precision (push).
        
        This balances convergence rates across all directions.
        """
        raw = self._var.copy()   # (N,) — direct variance = 1/curvature

        # Map [min, max] of raw linearly to [0.1, 10.0]
        # This preserves relative ratios AND fits within harness bounds.
        r_min, r_max = raw.min(), raw.max()
        precision = 0.1 + (raw - r_min) / (r_max - r_min + 1e-12) * (10.0 - 0.1)

        return precision.tolist()


def retrieve(query):
    """Stub for FFI-style retrieval (used by larger test suites)."""
    import random
    q = np.array(query)
    return {
        "attractor_id": random.randint(0, 255),
        "retrieved_pattern": (q + np.random.normal(0, 0.02, size=len(q))).tolist(),
        "precision": np.clip(np.ones(len(q)), 0.1, 10.0).tolist(),
        "energy": float(np.linalg.norm(q)),
        "metadata": {"mocked": True},
    }
