import numpy as np
import scipy.linalg as la
import antigravity

def test_anisotropy_spread_reduction():
    print("\n--- TEST: Anisotropy Spread Reduction (Geometry Check) ---")

    N = 64
    K = 16
    np.random.seed(42)

    # 1. Generate stored patterns
    stored_patterns = np.random.randn(K, N)
    corrupted_query = np.copy(stored_patterns[0]) + np.random.randn(N) * 0.3

    # 2. Build Hessian directly from pattern geometry.
    #    H_ii = 1 / var_i  (narrow valleys = high curvature).
    var_per_dim = np.var(stored_patterns, axis=0) + 1e-6   # (N,)
    curvature   = 1.0 / var_per_dim                        # (N,)

    # Scale to [1, 150] for a 150× baseline spread
    c_min, c_max = curvature.min(), curvature.max()
    curvature_scaled = 1.0 + (curvature - c_min) / (c_max - c_min + 1e-12) * 149.0
    H = np.diag(curvature_scaled)

    baseline_eigvals = curvature_scaled   # diagonal matrix
    baseline_spread  = baseline_eigvals.max() / baseline_eigvals.min()

    # 3. Compute the ANALYTICALLY OPTIMAL precision for this Hessian.
    #    To make Π^½ H Π^½ perfectly isotropic (spread = 1):
    #    π_i = C / H_ii   for any constant C.
    #    We choose C = mean(H_ii) so the mean precision = 1.
    target     = np.mean(curvature_scaled)
    pi_optimal = target / curvature_scaled           # (N,) — exact inverse of curvature
    pi_clipped = np.clip(pi_optimal, 0.1, 10.0)     # harness bounds

    # Scaled Hessian diagonals after applying optimal precision
    scaled_optimal = pi_optimal  * curvature_scaled
    scaled_clipped = pi_clipped  * curvature_scaled
    spread_optimal = scaled_optimal.max() / scaled_optimal.min()
    spread_clipped = scaled_clipped.max() / scaled_clipped.min()

    print(f"\n  [Analytical optimal — no clip]")
    print(f"  -> Baseline spread:        {baseline_spread:.2f}x")
    print(f"  -> Optimal spread:         {spread_optimal:.4f}x  (perfect = 1.0x)")
    print(f"  -> Reduction factor:       {baseline_spread / spread_optimal:.1f}x")

    print(f"\n  [Optimal with [0.1, 10.0] clip]")
    print(f"  -> Clipped spread:         {spread_clipped:.4f}x")
    print(f"  -> Reduction factor:       {baseline_spread / spread_clipped:.2f}x")

    # 4. Also test what the live engine returns
    engine    = antigravity.RustEngine(stored_patterns.tolist())
    precision = np.array(engine.predict(corrupted_query.tolist()))
    scaled_engine = precision * curvature_scaled
    spread_engine = scaled_engine.max() / scaled_engine.min()
    ratio_engine  = baseline_spread / spread_engine

    print(f"\n  [Live engine output]")
    print(f"  -> Precision range:        [{precision.min():.4f}, {precision.max():.4f}]  mean={precision.mean():.4f}")
    print(f"  -> Engine spread:          {spread_engine:.4f}x")
    print(f"  -> SPREAD REDUCTION FACTOR:{ratio_engine:.2f}x  (Target: > 5.0x)")

    if ratio_engine >= 10.0:
        print("  🏆 FULL MARKS (20 pts)")
    elif ratio_engine >= 5.0:
        print("  ✅ PASSED (5x+ Target Met)")
    elif ratio_engine > 1.0:
        print("  ⚠️  WARNING: Modest reduction.")
    else:
        print("  ❌ FAILED: Precision is making landscape worse.")

    return ratio_engine


if __name__ == "__main__":
    test_anisotropy_spread_reduction()
