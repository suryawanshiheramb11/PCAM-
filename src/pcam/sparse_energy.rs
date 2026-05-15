use ndarray::Array1;

/// Exact sparsemax (α=2) using sorting and thresholding.
///
/// Maps arbitrary scores to a sparse probability simplex.
/// Implements the algorithm from Martins & Astudillo (2016).
pub fn sparsemax(scores: &Array1<f64>) -> Array1<f64> {
    let k = scores.len();
    if k == 0 {
        return Array1::zeros(0);
    }
    if k == 1 {
        return Array1::ones(1); // single element always gets weight 1
    }

    let mut indexed: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut threshold = 0.0_f64;
    let mut support_size = k; // fallback

    for (i, &(_, s)) in indexed.iter().enumerate() {
        cumsum += s;
        let t = (cumsum - 1.0) / ((i + 1) as f64);
        if i + 1 == k || indexed[i + 1].1 <= t {
            threshold = t;
            support_size = i + 1;
            break;
        }
    }

    let mut p = Array1::zeros(k);
    for &(idx, s) in &indexed[..support_size] {
        let val = s - threshold;
        if val > 0.0 {
            p[idx] = val;
        }
    }

    // Fallback: if all zero (extremely negative scores), return uniform
    if p.sum() <= 1e-15 {
        return Array1::ones(k) / (k as f64);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sparsemax_basic() {
        let scores = array![2.0, 1.0, 0.0, -1.0];
        let p = sparsemax(&scores);
        // Should produce sparse output
        assert!(p.iter().any(|&v| v == 0.0));
        // Should sum to 1
        assert!((p.sum() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_sparsemax_uniform() {
        let scores = array![1.0, 1.0, 1.0];
        let p = sparsemax(&scores);
        for &v in p.iter() {
            assert!((v - 1.0 / 3.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_sparsemax_single() {
        let scores = array![42.0];
        let p = sparsemax(&scores);
        assert!((p[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_sparsemax_negative() {
        let scores = array![-1.0, -2.0, -3.0];
        let p = sparsemax(&scores);
        assert!((p.sum() - 1.0).abs() < 1e-8);
        assert!(p[0] > 0.0); // highest score gets weight
    }

    #[test]
    fn test_sparsemax_dominated() {
        let scores = array![10.0, -100.0, -100.0];
        let p = sparsemax(&scores);
        assert!((p[0] - 1.0).abs() < 1e-8);
        assert!(p[1].abs() < 1e-8);
        assert!(p[2].abs() < 1e-8);
    }
}
