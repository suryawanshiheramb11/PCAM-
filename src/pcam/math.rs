use ndarray::{Array1, Array2, Axis};

/// Compute LSR kernel values k_μ = ReLU(1 - β * ||R*x - ξ^μ||²).
pub fn lsr_kernel(
    x: &Array1<f64>,
    patterns: &Array2<f64>,
    r: &Array2<f64>,
    beta: f64,
) -> Array1<f64> {
    let rx = r.dot(x);
    let k = patterns.nrows();
    let mut out = Array1::zeros(k);
    for (mu, val) in out.iter_mut().enumerate() {
        let xi = patterns.row(mu);
        let diff = &rx - &xi;
        let dist_sq = diff.dot(&diff);
        let arg = 1.0 - beta * dist_sq;
        if arg > 0.0 {
            *val = arg;
        }
    }
    out
}

/// Gradient of the LSR energy (without precision). Not used directly in unified dynamics,
/// but kept for reference and offline optimisation.
///
/// The pull direction uses R^T * ξ^μ to map patterns back into x-space,
/// since the kernel operates in R-space: ||Rx - ξ^μ||².
#[allow(dead_code)]
pub fn gradient_lsr(
    x: &Array1<f64>,
    patterns: &Array2<f64>,
    r: &Array2<f64>,
    beta: f64,
) -> Array1<f64> {
    let k = lsr_kernel(x, patterns, r, beta);
    let sum_k = k.sum();
    if sum_k <= 0.0 {
        return x.clone(); // ∇E = x if no active memory
    }
    // p = sparsemax(k) later; here we use normalised kernel as approximation for gradient
    let p = k / sum_k; // simple normalisation (not sparsemax)
    let rt = r.t();
    let mut pull: Array1<f64> = Array1::zeros(x.len());
    for (mu, &w) in p.iter().enumerate() {
        if w > 0.0 {
            // R^T * ξ^μ maps pattern from R-space back into x-space
            let rtxi: Array1<f64> = rt.dot(&patterns.row(mu).to_owned());
            let scaled: Array1<f64> = rtxi * w;
            pull = pull + scaled;
        }
    }
    x - &pull
}

/// Hessian of the LSR energy at x (using the same kernel but without sparsemax for simplicity).
/// For offline optimisation, we use the Hessian at the stored pattern (where kernel = 1 for that pattern).
pub fn hessian_at_memory(
    xi: &Array1<f64>,
    patterns: &Array2<f64>,
    r: &Array2<f64>,
    beta: f64,
) -> Array2<f64> {
    let n = xi.len();
    let k = lsr_kernel(xi, patterns, r, beta);
    let sum_k = k.sum();
    if sum_k <= 0.0 {
        return Array2::eye(n);
    }
    let p = k.mapv(|v| v / sum_k); // normalised kernel weights

    // Compute R^T * ξ^μ for each pattern (mapped back to x-space): result is K x N
    let rt = r.t();
    let mut rt_xi: Array2<f64> = Array2::zeros((patterns.nrows(), n));
    for mu in 0..patterns.nrows() {
        let row: Array1<f64> = rt.dot(&patterns.row(mu).to_owned());
        rt_xi.row_mut(mu).assign(&row);
    }

    let mut term1: Array2<f64> = Array2::zeros((n, n));
    for (mu, &w) in p.iter().enumerate() {
        if w > 1e-15 {
            let row = rt_xi.row(mu);
            // outer product: row^T * row
            for i in 0..n {
                for j in 0..n {
                    term1[[i, j]] += w * row[i] * row[j];
                }
            }
        }
    }

    let mean_rtxi: Array1<f64> = p
        .iter()
        .zip(rt_xi.axis_iter(Axis(0)))
        .fold(Array1::<f64>::zeros(n), |acc: Array1<f64>, (&w, row)| {
            let scaled: Array1<f64> = row.to_owned() * w;
            acc + scaled
        });

    let mut term2: Array2<f64> = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            term2[[i, j]] = mean_rtxi[i] * mean_rtxi[j];
        }
    }

    let identity: Array2<f64> = Array2::eye(n);
    // Hessian = I - β * ( Σ p_μ (R^Tξ^μ)(R^Tξ^μ)^T - mean * mean^T )
    let cov: Array2<f64> = term1 - term2;
    identity - cov * beta
}

/// Compute squared Euclidean distance between two vectors (utility).
pub fn sq_dist(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let diff: Array1<f64> = a - b;
    diff.dot(&diff)
}
