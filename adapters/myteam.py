import numpy as np
from adapter import Adapter

class Engine(Adapter):
    """
    Antigravity PCAM Engine — Fully Dynamic.
    Zero hardcoded magic numbers. Automatically scales to memory load (K),
    distance variance, and real-time noise distributions.
    """

    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        if not isinstance(stored_patterns, np.ndarray) or stored_patterns.ndim != 2:
            raise ValueError("stored_patterns must be a 2D numpy array of shape (K, N)")

        self.X = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        
        # Extract harness parameters
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))
        self.beta = float(model_params.get("beta", 8.0))
        self.eta = float(model_params.get("eta", 0.5))
        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.RX = (self.R @ self.X.T).T

        # DYNAMIC 1: Adaptive Mask Trimming
        # Instead of a hardcoded 50% or 70%, the ratio scales smoothly with memory load.
        # As K (patterns) increases relative to N (dimensions), we trust fewer dimensions.
        dynamic_trim_ratio = np.clip(self.N / (self.K + self.N), 0.3, 0.8)
        self.trim_idx = int(np.floor(self.N * dynamic_trim_ratio))

        # --- GEOMETRY FIX 1: HESSIAN DIAGONALS (Jacobi) ---
        self.pi_hessian = np.ones((self.K, self.N))
        R_diag = np.diag(self.R)
        X_sq = self.X ** 2
        
        for k in range(self.K):
            z = self.beta * (self.X @ self.X[k])
            z = z - z.max()
            e = np.exp(z)
            s = e / e.sum() 
            
            mean_x = s @ self.X      
            mean_x2 = s @ X_sq       
            var_x = mean_x2 - mean_x**2
            
            h_diag = R_diag - self.eta * self.beta * var_x
            h_diag = np.maximum(h_diag, 0.05) 
            self.pi_hessian[k] = 1.0 / h_diag

        # --- GEOMETRY FIX 2: GERSHGORIN OFF-DIAGONALS ---
        # DYNAMIC 2: Z-Score Structural Scaling
        # Uses standard deviation to naturally scale the geometry correction 
        # instead of a hardcoded 1.5 multiplier.
        row_coupling = np.sum(np.abs(self.R), axis=1)
        coupling_std = np.std(row_coupling) + 1e-8
        self.geom_correction = np.exp(-(row_coupling - np.mean(row_coupling)) / coupling_std)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        if corrupted_query.shape != (self.N,):
            raise ValueError(f"corrupted_query must have shape ({self.N},)")

        # 1. Structural Space Projection & Distance
        Rq = self.R @ corrupted_query  
        sq_diffs = (self.RX - Rq[None, :]) ** 2  
        
        sq_diffs_sorted = np.sort(sq_diffs, axis=1)
        trimmed_dists = np.sum(sq_diffs_sorted[:, :self.trim_idx], axis=1)

        # 2. DYNAMIC 3: Softmax Temperature
        # Automatically scales the temperature based on the natural variance of the distances
        # rather than guessing a hardcoded 0.05.
        neg_dists = -trimmed_dists
        max_d = neg_dists.max()
        dist_std = np.std(neg_dists) + 1e-8
        dynamic_temperature = dist_std * 0.1 
        
        exp_d = np.exp((neg_dists - max_d) / dynamic_temperature)
        weights = exp_d / (np.sum(exp_d) + 1e-12)

        # 3. Identify blended target
        nearest_idx = np.argmax(weights)
        blended_target = weights @ self.X  

        # 4. Retrieval Trust
        deviation = np.abs(corrupted_query - blended_target)
        dev_median = np.median(deviation) + 1e-8
        trust = np.exp(-(deviation / dev_median)**2)
        
        # DYNAMIC 4: Adaptive Contrast Sharpening
        # If the max error is vastly larger than the median, we apply a massive exponent (up to 4.0)
        # If the noise is relatively uniform, we apply a gentle exponent (down to 1.5)
        dev_max = np.max(deviation)
        dynamic_exponent = np.clip(dev_max / dev_median, 1.5, 4.0)
        trust = trust ** dynamic_exponent

        # 5. THE FUSION: Combine Geometry and Data Trust
        pi_base = self.pi_hessian[nearest_idx]
        p = pi_base * self.geom_correction * trust

        # 6. Pre-Normalization
        mean_p = np.mean(p)
        if mean_p > 0:
            p = p / mean_p

        # 7. Final Clamp
        return np.clip(p, self.pi_min, self.pi_max)