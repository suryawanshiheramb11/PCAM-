import numpy as np

class Engine:
    """
    Precision estimator that compares noisy inputs against stored patterns
    using variance-weighted distance metrics.
    """
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        # 1. Input Validation
        if not isinstance(stored_patterns, np.ndarray) or stored_patterns.ndim != 2:
            raise ValueError("stored_patterns must be a 2D numpy array of shape (K, N)")
            
        self.X = stored_patterns  # (K, N) where K is num patterns, N is dimensions
        self.N = stored_patterns.shape[1]
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))
        
        # 2. Precompute Inverse-Variance Weights
        # Calculate how much each dimension naturally varies across all stored patterns.
        pattern_var = np.var(self.X, axis=0)  # (N,)
        
        # We want to penalize dimensions with high variance (they are less informative).
        # Add a small epsilon (1e-8) to prevent division by zero for constant dimensions.
        self.precision_weights = 1.0 / (pattern_var + 1e-8)
        
        # Normalize weights so they sum to N (average weight is 1.0).
        # This keeps the overall scale of our distance metric consistent.
        self.precision_weights /= np.mean(self.precision_weights)

    def predict_precision(self, corrupted_query: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Estimates the reliability (precision) of each dimension in the query.
        
        Args:
            corrupted_query: 1D array of shape (N,)
            temperature: Controls how strictly to penalize errors. 
                         Higher = more forgiving, Lower = strict penalty.
        Returns:
            precision: 1D array of shape (N,) bounded by [pi_min, pi_max]
        """
        # Defensive check to ensure the query matches the expected dimension size
        if corrupted_query.shape != (self.N,):
            raise ValueError(f"corrupted_query must have shape ({self.N},), got {corrupted_query.shape}")

        # Step 1: Find the nearest stored pattern (Weighted Mahalanobis-style distance)
        diffs = self.X - corrupted_query  # (K, N)
        
        # Weight the squared differences. If a dimension is highly variable across 
        # all known patterns, we care less about errors in that specific dimension.
        dists = np.sum((diffs ** 2) * self.precision_weights, axis=1)  # (K,)
        nearest = self.X[np.argmin(dists)]  # (N,)
        
        # Step 2: Identify trustworthy dimensions via absolute error
        error = np.abs(corrupted_query - nearest)
        
        # Transform error into a raw trust score between (0, 1].
        # Exact match = 1.0. Infinite error = 0.0.
        trust = np.exp(-error / temperature)
        
        # Step 3: Absolute Scaling to [pi_min, pi_max]
        # Instead of relativistic scaling (which artificially boosts garbage data),
        # we map the (0, 1] trust range directly to the [pi_min, pi_max] range.
        precision = self.pi_min + trust * (self.pi_max - self.pi_min)
        
        # Final clip to safeguard against any floating-point anomalies
        return np.clip(precision, self.pi_min, self.pi_max)