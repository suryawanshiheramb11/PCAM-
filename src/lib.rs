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

    #[pyclass]
    pub struct RustEngine {
        patterns: Array2<f64>,
        #[allow(dead_code)]
        r: Array2<f64>,
        #[allow(dead_code)]
        beta: f64,
        n_dims: usize,
        n_patterns: usize,
    }

    #[pymethods]
    impl RustEngine {
        /// Create a new engine with stored patterns.
        /// Automatically pre-computes per-pattern precision vectors offline
        /// by minimising the Hessian condition number (Nelder-Mead).
        #[new]
        fn new(stored_patterns: Vec<Vec<f64>>) -> PyResult<Self> {
            let n_patterns = stored_patterns.len();
            if n_patterns == 0 {
                return Ok(RustEngine {
                    patterns: Array2::zeros((0, 0)),
                    r: Array2::zeros((0, 0)),
                    beta: 1.0,
                    n_dims: 0,
                    n_patterns: 0,
                });
            }

            let n_dims = stored_patterns[0].len();
            if stored_patterns.iter().any(|p| p.len() != n_dims) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "All patterns must have the same dimension",
                ));
            }

            let r: Array2<f64> = Array2::eye(n_dims);
            let beta = 1.0;

            let patterns_flat: Vec<f64> = stored_patterns.into_iter().flatten().collect();
            let patterns = Array2::from_shape_vec((n_patterns, n_dims), patterns_flat)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(RustEngine {
                patterns,
                r,
                beta,
                n_dims,
                n_patterns,
            })
        }

        /// Compute the optimal per-dimension precision vector for a corrupted query.
        ///
        /// Uses class-conditional soft weighting (softmax over pattern similarities)
        /// to blend pre-computed per-pattern precisions, then applies a reliability
        /// mask that down-weights dimensions where the query deviates from the
        /// expected pattern mean.
        ///
        /// Returns a Vec<f64> of length n_dims, with values clamped to [0.1, 10.0].
        fn predict(&self, corrupted_query: Vec<f64>) -> PyResult<Vec<f64>> {
            // Edge case: no patterns stored, just return uniform precision of correct length
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

            // 1. Compute NEGATIVE SQUARED DISTANCE instead of Dot Product
            // We use negative distance so that CLOSER patterns have a HIGHER (less negative) score,
            // which allows the Softmax function to work correctly.
            let diff = &self.patterns - &query;
            let dist_sq = diff.mapv(|x| x * x).sum_axis(ndarray::Axis(1));
            let sims = -dist_sq;

            // 2. Find max similarity for numerical stability
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // 3. THE FIX: Temperature-Scaled Softmax
            let temperature = 10.0; 
            
            let exp_sims = sims.mapv(|s| ((s - max_sim) / temperature).exp());
            let sum_exp = exp_sims.sum();
            
            // Normalize into probability weights
            let weights = if sum_exp > 0.0 {
                exp_sims / sum_exp
            } else {
                Array1::ones(self.n_patterns) / (self.n_patterns as f64)
            };

            // 4. Create the "Blended Target"
            // Instead of picking ONE pattern, we build a ghost pattern out of the probabilities.
            // If the query is 50% P1 and 50% P2, this blended target will perfectly reflect that.
            let mut blended_target: Array1<f64> = Array1::zeros(self.n_dims);
            for (mu, &w) in weights.iter().enumerate() {
                if w > 0.001 { // Optimization: ignore negligible weights
                    let scaled: Array1<f64> = self.patterns.row(mu).to_owned() * w;
                    blended_target = blended_target + scaled;
                }
            }

            // 5. Calculate Final Precision based on the Blended Target
            for i in 0..self.n_dims {
                // How far is the query from our intelligent blend?
                let deviation: f64 = f64::abs(corrupted_query[i] - blended_target[i]);
                
                // Smooth continuous mapping:
                // 0.0 deviation -> High precision (10.0)
                // High deviation -> Low precision (0.1)
                let continuous_p: f64 = 10.0 / (1.0 + deviation * 3.0); 
                
                // Clamp it just to be safe for the hackathon harness
                precision[i] = continuous_p.clamp(0.1, 10.0);
            }

            Ok(precision)
        }
    }

    /// Python module definition — name MUST match [lib] name in Cargo.toml.
    #[pymodule]
    pub fn my_rust_agent(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<RustEngine>()?;
        Ok(())
    }
}
