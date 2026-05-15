/// Antigravity — PCAM Hopfield Network Engine
///
/// This library exposes:
/// - `pcam` module: the core LSR kernel, sparsemax, precision optimizer, and dynamics.
/// - `RustEngine` (via PyO3, behind `python` feature): a Python-callable class.
pub mod pcam;

#[cfg(feature = "python")]
mod python_bridge {
    use pyo3::prelude::*;
    use ndarray::{Array1, Array2};
    use nalgebra::DMatrix;

    #[pyclass]
    pub struct RustEngine {
        patterns: Array2<f64>,
        r: Array2<f64>,
        r_inv: Array2<f64>, // Added precomputed inverse
        beta: f64,
        eta: f64,
        pi_min: f64,
        pi_max: f64,
        n_dims: usize,
        n_patterns: usize,
    }

    #[pymethods]
    impl RustEngine {
        /// Create a new engine with stored patterns and dynamic structural properties.
        #[new]
        fn new(
            stored_patterns: Vec<Vec<f64>>, 
            r_matrix: Vec<Vec<f64>>, 
            eta: f64,
            beta: f64,
            pi_min: f64, 
            pi_max: f64
        ) -> PyResult<Self> {
            let n_patterns = stored_patterns.len();
            if n_patterns == 0 {
                return Ok(RustEngine {
                    patterns: Array2::zeros((0, 0)),
                    r: Array2::zeros((0, 0)),
                    r_inv: Array2::zeros((0, 0)),
                    pi_min,
                    pi_max,
                    beta,
                    eta,
                    n_dims: 0,
                    n_patterns: 0,
                });
            }

            let n_dims = stored_patterns[0].len();
            let mut is_mismatched = false;
            for i in 0..n_patterns {
                if stored_patterns[i].len() != n_dims {
                    is_mismatched = true;
                }
            }
            
            if is_mismatched {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "All patterns must have the same dimension",
                ));
            }

            // Parse the seed-dependent structural operator R passed from the harness
            let r_flat: Vec<f64> = r_matrix.into_iter().flatten().collect();
            let r = Array2::from_shape_vec((n_dims, n_dims), r_flat)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            let patterns_flat: Vec<f64> = stored_patterns.into_iter().flatten().collect();
            let patterns = Array2::from_shape_vec((n_patterns, n_dims), patterns_flat)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            // Precompute R inverse for attractor estimation
            let r_na = DMatrix::from_fn(n_dims, n_dims, |i, j| r[[i, j]]);
            let r_inv_na = r_na.try_inverse().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("R matrix is not invertible")
            })?;
            let r_inv = Array2::from_shape_fn((n_dims, n_dims), |(i, j)| r_inv_na[(i, j)]);

            Ok(RustEngine {
                patterns,
                r,
                r_inv,
                beta,
                eta,
                pi_min,
                pi_max,
                n_dims,
                n_patterns,
            })
        }

        /// Compute the optimal per-dimension precision vector for a corrupted query.
        fn predict(&self, corrupted_query: Vec<f64>) -> PyResult<Vec<f64>> {
            if self.n_patterns == 0 {
                return Ok(vec![1.0; corrupted_query.len()]);
            }

            if corrupted_query.len() != self.n_dims {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Query dimension {} does not match pattern dimension {}",
                    corrupted_query.len(),
                    self.n_dims
                )));
            }

            let query = Array1::from_vec(corrupted_query.clone());
            let mut precision = vec![1.0; self.n_dims];

            // 1. Structural Space Projection
            // We use R to project both query and patterns to find the nearest neighbor more reliably
            // in the presence of anisotropy.
            let structural_query = self.r.dot(&query);

            let mut sims_vec = Vec::with_capacity(self.n_patterns);
            for i in 0..self.n_patterns {
                let pattern_row = self.patterns.row(i);
                // Instead of R.dot(pattern), we use the dot product (R @ a) . x_i
                // which is query . (R @ pattern) because R is symmetric.
                // This aligns with the energy term a^T R a.
                let structural_pattern = self.r.dot(&pattern_row);
                
                let mut sq_diffs = Vec::with_capacity(self.n_dims);
                structural_pattern.iter().zip(structural_query.iter()).for_each(|(&p_val, &q_val)| {
                    let diff = p_val - q_val;
                    sq_diffs.push(diff * diff);
                });
                
                sq_diffs.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                // Robust distance: ignore the worst 30% of dimensions (likely masked)
                let anchor_idx = (self.n_dims as f64 * 0.7).floor() as usize;
                let trimmed_dist: f64 = sq_diffs.iter().take(anchor_idx).sum();
                sims_vec.push(-trimmed_dist);
            }
            
            let sims = Array1::from_vec(sims_vec);
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // 2. Softmax weights for Hessian estimation
            let temperature = 0.2; // Sharper to match model's beta=8.0
            let exp_sims = ndarray::Array1::from_shape_fn(self.n_patterns, |i| {
                f64::exp((sims[i] - max_sim) / temperature)
            });
            let sum_exp = exp_sims.sum();
            let weights = if sum_exp > 0.0 { exp_sims / sum_exp } else { Array1::ones(self.n_patterns) / (self.n_patterns as f64) };

            // 3. Compute Blended Target (initial guess for attractor)
            let mut blended_target: Array1<f64> = Array1::zeros(self.n_dims);
            for k in 0..self.n_patterns {
                let w = weights[k];
                if w > 1e-4 {
                    for i in 0..self.n_dims {
                        blended_target[i] += w * self.patterns[[k, i]];
                    }
                }
            }

            // 4. Estimate True Attractor Equilibrium: a* ≈ eta * R^-1 * x_nearest
            let approx_attractor = self.r_inv.dot(&blended_target) * self.eta;

            // 5. Compute Softmax at Attractor for Hessian Estimation
            let mut z = self.patterns.dot(&approx_attractor) * self.beta;
            let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            z.mapv_inplace(|v| f64::exp(v - z_max));
            let z_sum = z.sum();
            let s = z / z_sum;

            // 6. Compute Hessian Diagonal at Attractor
            let mut weighted_target: Array1<f64> = Array1::zeros(self.n_dims);
            let mut weighted_sq_target: Array1<f64> = Array1::zeros(self.n_dims);
            for k in 0..self.n_patterns {
                let sk = s[k];
                if sk > 1e-5 {
                    for i in 0..self.n_dims {
                        let val = self.patterns[[k, i]];
                        weighted_target[i] += sk * val;
                        weighted_sq_target[i] += sk * val * val;
                    }
                }
            }

            // 7. Structural Isotropisation (R-Inverse Diagonal)
            // The diagonal of the inverse is a classic preconditioner for SPD matrices.
            let mut r_inv_diag = Array1::zeros(self.n_dims);
            for i in 0..self.n_dims {
                r_inv_diag[i] = self.r_inv[[i, i]];
            }

            // 8. Retrieval Trust + Masking Detection
            for i in 0..self.n_dims {
                let deviation = f64::abs(query[i] - blended_target[i]);
                // Smoother trust to preserve structural isotropisation
                let trust = f64::exp(-(deviation / 0.5).powi(2));
                
                let mut p = trust * r_inv_diag[i]; 

                // Explicit Masking Detection
                if query[i].abs() < 1e-4 {
                    p = self.pi_min;
                }
                
                precision[i] = p.clamp(self.pi_min, self.pi_max);
            }

            Ok(precision)
        }
    }

    #[pymodule]
    pub fn my_rust_agent(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<RustEngine>()?;
        Ok(())
    }
}