use ndarray::{Array1, Array2};
use nalgebra::DMatrix;
use crate::pcam::math::hessian_at_memory;

/// Compute the condition number of D * H * D where D = diag(d).
/// Uses nalgebra for eigenvalue decomposition (pure Rust, no LAPACK needed).
fn condition_number(h: &Array2<f64>, d: &Array1<f64>, n: usize) -> f64 {
    // Build D * H * D
    let mut dhd_data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            dhd_data[i * n + j] = d[i] * h[[i, j]] * d[j];
        }
    }

    // Convert to nalgebra DMatrix
    let mat = DMatrix::from_fn(n, n, |i, j| dhd_data[i * n + j]);

    // Symmetric eigenvalue decomposition
    let eigen = mat.symmetric_eigen();
    let eigenvalues = &eigen.eigenvalues;

    let mut min_ev = f64::INFINITY;
    let mut max_ev = f64::NEG_INFINITY;
    for &ev in eigenvalues.iter() {
        if ev < min_ev {
            min_ev = ev;
        }
        if ev > max_ev {
            max_ev = ev;
        }
    }

    if min_ev <= 1e-15 {
        return 1e12;
    }
    max_ev / min_ev
}

/// Simple Nelder-Mead optimiser for minimising condition number.
/// Operates in n-dimensional space where each coordinate is sqrt(precision).
fn nelder_mead<F>(f: &F, init_simplex: Vec<Array1<f64>>, max_iters: u64, tol: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = init_simplex[0].len();
    let num_vertices = init_simplex.len(); // should be n + 1

    // Evaluate initial simplex
    let mut simplex: Vec<(Array1<f64>, f64)> = init_simplex
        .into_iter()
        .map(|p| {
            let v = f(&p);
            (p, v)
        })
        .collect();

    let mut stall_count = 0u64;
    let mut prev_best = f64::INFINITY;

    for _ in 0..max_iters {
        // Sort by cost
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check convergence: spread of function values
        let f_best = simplex[0].1;
        let f_worst = simplex[num_vertices - 1].1;
        if (f_worst - f_best).abs() < tol {
            break;
        }

        // Early termination on improvement plateau
        if (prev_best - f_best).abs() < tol * 0.1 {
            stall_count += 1;
            if stall_count > 20 {
                break;
            }
        } else {
            stall_count = 0;
        }
        prev_best = f_best;

        // Centroid of all vertices except worst
        let mut centroid: Array1<f64> = Array1::zeros(n);
        for i in 0..(num_vertices - 1) {
            centroid = centroid + &simplex[i].0;
        }
        centroid /= (num_vertices - 1) as f64;

        let worst_idx = num_vertices - 1;
        let worst_point = simplex[worst_idx].0.clone();

        // Reflection
        let reflected: Array1<f64> = &centroid * 2.0 - &worst_point;
        let f_reflected = f(&reflected);

        if f_reflected < simplex[0].1 {
            // Try expansion
            let expanded: Array1<f64> = &centroid + (&reflected - &centroid) * 2.0;
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst_idx] = (expanded, f_expanded);
            } else {
                simplex[worst_idx] = (reflected, f_reflected);
            }
        } else if f_reflected < simplex[worst_idx - 1].1 {
            simplex[worst_idx] = (reflected, f_reflected);
        } else {
            // Contraction
            let contracted: Array1<f64> = &centroid + (&worst_point - &centroid) * 0.5;
            let f_contracted = f(&contracted);
            if f_contracted < simplex[worst_idx].1 {
                simplex[worst_idx] = (contracted, f_contracted);
            } else {
                // Shrink: move all vertices towards best
                let best = simplex[0].0.clone();
                for i in 1..num_vertices {
                    simplex[i].0 = &best + (&simplex[i].0 - &best) * 0.5;
                    simplex[i].1 = f(&simplex[i].0);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    simplex[0].0.clone()
}

/// Optimize precision for a single stored pattern xi.
/// Returns precision vector Π (diag entries), i.e. d².
/// Minimises the Hessian condition number at the memory location.
///
/// Fast-path: if the Hessian is already well-conditioned (κ < 2.0), returns
/// uniform precision immediately to avoid unnecessary optimisation.
pub fn optimize_precision(
    xi: &Array1<f64>,
    patterns: &Array2<f64>,
    r: &Array2<f64>,
    beta: f64,
    max_iters: u64,
) -> Array1<f64> {
    let n = xi.len();
    let h = hessian_at_memory(xi, patterns, r, beta);

    // Fast path: if Hessian is already well-conditioned, skip optimisation
    let identity_d = Array1::ones(n);
    let base_cond = condition_number(&h, &identity_d, n);
    if base_cond < 2.0 {
        return Array1::ones(n);
    }

    // Cost function: condition number of D * H * D, with bounds penalty
    let cost = |d: &Array1<f64>| -> f64 {
        if d.iter().any(|&v| v < 0.09 || v > 10.1) {
            return 1e12;
        }
        condition_number(&h, d, n)
    };

    // Build initial simplex: identity + perturbations
    let init_d = Array1::ones(n);
    let mut simplex = vec![init_d.clone()];
    for i in 0..n {
        let mut p = init_d.clone();
        p[i] *= 1.1;
        simplex.push(p);
    }

    let best_d = nelder_mead(&cost, simplex, max_iters, 1e-4);
    // Square to get precision (d² = Π)
    best_d.mapv(|v| v.powi(2))
}
