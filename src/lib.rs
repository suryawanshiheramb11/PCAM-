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

    use crate::pcam::optimize::optimize_precision;

    #[pyclass]
    pub struct RustEngine {
        patterns: Array2<f64>,
        precisions: Vec<Array1<f64>>,
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
                    precisions: vec![],
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

            // Pre-compute per-pattern precision vectors offline
            let precisions: Vec<Array1<f64>> = (0..n_patterns)
                .map(|mu| {
                    let xi = patterns.row(mu).to_owned();
                    optimize_precision(&xi, &patterns, &r, beta, 200)
                })
                .collect();

            Ok(RustEngine {
                patterns,
                precisions,
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

            let query = Array1::from_vec(corrupted_query);

            // ── Step 1: Compute softmax weights over pattern similarities ──
            let sims = self.patterns.dot(&query);
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let exp_sims = sims.mapv(|s| (s - max_sim).exp());
            let sum_exp = exp_sims.sum();
            let weights = if sum_exp > 0.0 {
                exp_sims / sum_exp
            } else {
                Array1::ones(self.n_patterns) / (self.n_patterns as f64)
            };

            // ── Step 2: Blend per-pattern precisions ──
            let mut blended: Array1<f64> = Array1::zeros(self.n_dims);
            for (mu, &w) in weights.iter().enumerate() {
                if w > 1e-15 {
                    let scaled: Array1<f64> = &self.precisions[mu] * w;
                    blended = blended + scaled;
                }
            }

            // ── Step 3: Reliability mask ──
            // Down-weight dimensions where the query deviates heavily from the
            // weighted pattern mean (likely corrupted).
            let mut weighted_mean: Array1<f64> = Array1::zeros(self.n_dims);
            for (mu, &w) in weights.iter().enumerate() {
                if w > 1e-15 {
                    let pat: Array1<f64> = self.patterns.row(mu).to_owned();
                    let scaled: Array1<f64> = pat * w;
                    weighted_mean = weighted_mean + scaled;
                }
            }
            let deviation: Array1<f64> = (&query - &weighted_mean).mapv(|x| x.abs());
            let reliability = deviation.mapv(|d| (-2.0 * d).exp()); // λ = 2.0
            blended = blended * &reliability;

            // ── Step 4: Clamp to harness range [0.1, 10.0] ──
            let result: Vec<f64> = blended.iter().map(|&v| v.max(0.1).min(10.0)).collect();

            Ok(result)
        }
    }

    /// Python module definition — name MUST match [lib] name in Cargo.toml.
    #[pymodule]
    pub fn my_rust_agent(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<RustEngine>()?;
        Ok(())
    }
}
