use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::thread_rng;
use rand_distr::StandardNormal;

use crate::pcam::math::lsr_kernel;
use crate::pcam::sparse_energy::sparsemax;

/// Unified dynamics with LSR kernel, sparsemax attention, precision, and Langevin noise.
///
/// Energy: E(x) = ½‖x‖² + Ω*(k(x))
/// where k_μ = ReLU(1 − β‖Rx − ξ^μ‖²) and Ω* is the Gini conjugate (sparsemax).
///
/// The pull direction uses R^T * ξ^μ to map patterns back to x-space,
/// and Langevin noise scales as √(2T/Π) so higher precision → less noise (correct
/// fluctuation-dissipation relation).
pub fn unified_dynamics(
    x0: &Array1<f64>,
    patterns: &Array2<f64>,
    r: &Array2<f64>,
    beta: f64,
    eta: f64,
    precision: &Array1<f64>,
    temperature: f64,
    max_iter: usize,
    tol: f64,
) -> Array1<f64> {
    let n = x0.len();
    let mut x = x0.clone();
    let mut rng = thread_rng();
    let rt = r.t().to_owned(); // R^T precomputed once

    // Clip and normalise precision (harness convention)
    let mut prec = precision.mapv(|v| v.max(0.1).min(10.0));
    let mean = prec.sum() / (prec.len() as f64);
    if mean > 1e-12 {
        prec /= mean;
    }

    for _ in 0..max_iter {
        let k_vec = lsr_kernel(&x, patterns, r, beta);
        if k_vec.sum() <= 1e-12 {
            break; // no active memories
        }
        let p = sparsemax(&k_vec); // sparse probability vector

        // Pull direction: R^T * (weighted sum of active patterns) → x-space
        let mut pull: Array1<f64> = Array1::zeros(n);
        for (mu, &w) in p.iter().enumerate() {
            if w > 1e-12 {
                let rtxi: Array1<f64> = rt.dot(&patterns.row(mu).to_owned());
                let scaled: Array1<f64> = rtxi * w;
                pull = pull + scaled;
            }
        }

        let diff: Array1<f64> = &x - &pull;
        let step: Array1<f64> = &prec * &diff * eta;
        let mut x_new: Array1<f64> = &x - &step;

        // Langevin noise for generation mode
        // Noise scales as √(2T·η / Π_i) — correct fluctuation-dissipation:
        //   high precision → small noise (tight dimension)
        //   low precision  → large noise (loose dimension)
        if temperature > 1e-12 {
            let noise: Array1<f64> = Array1::random_using(n, StandardNormal, &mut rng);
            let noise_scaled: Array1<f64> = noise.mapv(|_| 0.0)
                .iter()
                .zip(noise.iter())
                .zip(prec.iter())
                .map(|((_, &ni), &pi)| ni * (2.0 * temperature * eta / pi).sqrt())
                .collect::<Vec<f64>>()
                .into();
            let noise_arr: Array1<f64> = Array1::from_vec(noise_scaled.to_vec());
            x_new = x_new + noise_arr;
        }

        // Convergence check
        let delta: Array1<f64> = &x_new - &x;
        let norm: f64 = delta.dot(&delta).sqrt();
        if norm < tol {
            return x_new;
        }
        x = x_new;
    }
    x
}
