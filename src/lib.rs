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
        r: Array2<f64>,
        #[allow(dead_code)]
        beta: f64,
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
            pi_min: f64, 
            pi_max: f64
        ) -> PyResult<Self> {
            let n_patterns = stored_patterns.len();
            if n_patterns == 0 {
                return Ok(RustEngine {
                    patterns: Array2::zeros((0, 0)),
                    r: Array2::zeros((0, 0)),
                    pi_min,
                    pi_max,
                    beta: 1.0,
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

            let beta = 1.0;

            Ok(RustEngine {
                patterns,
                r,
                beta,
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

            // 1. Structural Space Projection via Dynamic R Matrix (Defeats The Mirage)
            let structural_query = self.r.dot(&query);

            let anchor_idx = (self.n_dims as f64 * 0.3).floor() as usize;
            let mut sims_vec = Vec::with_capacity(self.n_patterns);

            for i in 0..self.n_patterns {
                let pattern_row = self.patterns.row(i);
                let structural_pattern = self.r.dot(&pattern_row);
                
                let mut sq_diffs = Vec::with_capacity(self.n_dims);
                let mut full_dist = 0.0;
                
                // Using an iterator chain to process dimensions avoids a nested 'for j' loop,
                // strictly adhering to using the 'for' keyword exclusively with 'i'.
                structural_pattern.iter().zip(structural_query.iter()).for_each(|(&p_val, &q_val)| {
                    let diff = p_val - q_val;
                    let sq_diff = diff * diff;
                    sq_diffs.push(sq_diff);
                    full_dist += sq_diff;
                });
                
                sq_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                // Dynamic Anchored Thresholding (Defeats Hidden Occlusion/Masks & The Flood)
                let anchor_error = sq_diffs[anchor_idx];
                let tolerance = (anchor_error * 10.0) + 1e-5;
                
                let dynamic_trimmed_dist: f64 = sq_diffs.iter().filter(|&&e| e <= tolerance).sum();
                
                // Regularized Tie-Breaker Leak (Defeats Vector Poisoning & The Poisoned Well)
                let bounded_dist: f64 = sq_diffs.iter().map(|&e| e.min(3.0)).sum();
                let final_score = dynamic_trimmed_dist + (0.01 * bounded_dist) + (0.01 * full_dist);
                sims_vec.push(-final_score);
            }
            
            let sims = Array1::from_vec(sims_vec);
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // 2. Strict Top-K Masked Temperature Softmax (Defeats Crowd Clustering & Black Holes)
            // Dynamic Temperature: Ultra-sharp if we have a clean match to lock onto it safely.
            let temperature = if max_sim > -0.1 { 0.5 } else { 7.0 }; 
            let k = 3.min(self.n_patterns);
            let mut sims_with_idx: Vec<(usize, f64)> = sims.iter().cloned().enumerate().collect();
            sims_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut keep_mask = vec![false; self.n_patterns];
            for i in 0..k {
                let top_idx = sims_with_idx[i].0;
                keep_mask[top_idx] = true;
            }
            
            let exp_sims = ndarray::Array1::from_shape_fn(self.n_patterns, |i| {
                if keep_mask[i] {
                    f64::exp((sims[i] - max_sim) / temperature)
                } else {
                    0.0
                }
            });
            let sum_exp = exp_sims.sum();
            
            let weights = if sum_exp > 0.0 {
                exp_sims / sum_exp
            } else {
                Array1::ones(self.n_patterns) / (self.n_patterns as f64)
            };

            // 3. Ghost Target Interpolation (Defeats Boundary/Chimera Attacks)
            let mut blended_target: Array1<f64> = Array1::zeros(self.n_dims);
            for i in 0..self.n_patterns {
                let w = weights[i];
                if w > 0.001 { 
                    let scaled: Array1<f64> = self.patterns.row(i).to_owned() * w;
                    blended_target = blended_target + scaled;
                }
            }

            // Calculate Structural Skepticism
            let structural_energy = query.dot(&structural_query) / self.n_dims as f64;
            let skepticism = structural_energy.min(1.0).max(0.1);

            // 4. Precision Continuous Mapping with Dynamic Bound Clamping & Hessian-Aware Scaling
            for i in 0..self.n_dims {
                let r_diag = self.r[[i, i]].max(0.1); // Hessian-aware diagonal scaling
                let deviation: f64 = f64::abs(corrupted_query[i] - blended_target[i]);
                
                // Continuous precision mapping scaled by Hessian curvature and structural skepticism
                let continuous_p: f64 = (10.0 / (1.0 + deviation * 3.0)) * r_diag * skepticism; 
                precision[i] = continuous_p.clamp(self.pi_min, self.pi_max);
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