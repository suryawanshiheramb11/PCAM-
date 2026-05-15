import numpy as np
import my_rust_agent

class Engine:
    """
    Precision estimator that compares noisy inputs against stored patterns
    using variance-weighted distance metrics AND the compiled Rust Engine.
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
        pattern_var = np.var(self.X, axis=0)  # (N,)
        self.precision_weights = 1.0 / (pattern_var + 1e-8)
        self.precision_weights /= np.mean(self.precision_weights)

        # 3. Instantiate the high-performance Rust Engine
        R_matrix = model_params.get("R")
        if isinstance(R_matrix, np.ndarray):
            R_list = R_matrix.tolist()
        else:
            R_list = R_matrix

        self.rust_engine = my_rust_agent.RustEngine(
            self.X.tolist(), 
            R_list, 
            self.pi_min, 
            self.pi_max
        )

    def predict_precision(self, corrupted_query: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if corrupted_query.shape != (self.N,):
            raise ValueError(f"corrupted_query must have shape ({self.N},), got {corrupted_query.shape}")

        # --- PATH A: Pure Python Variance-Weighted Estimation ---
        diffs = self.X - corrupted_query  
        dists = np.sum((diffs ** 2) * self.precision_weights, axis=1)  
        nearest = self.X[np.argmin(dists)]  
        
        error = np.abs(corrupted_query - nearest)
        trust = np.exp(-error / temperature)
        precision_python = self.pi_min + trust * (self.pi_max - self.pi_min)
        precision_python = np.clip(precision_python, self.pi_min, self.pi_max)

        # --- PATH B: High-Performance Rust PCAM Engine ---
        precision_rust = np.array(self.rust_engine.predict(corrupted_query.tolist()))

        # --- BLENDING: Working together ---
        # The Rust engine handles structural R-projections and dynamic masking well,
        # while the Python engine handles variance-weighting. Averaging them gives the best of both!
        final_precision = (precision_python + precision_rust) / 2.0
        
        return np.clip(final_precision, self.pi_min, self.pi_max)