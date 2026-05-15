import numpy as np
import my_rust_agent

def test_the_fog():
    print("\n--- TEST: The Fog (Micro-Noise) ---")
    K, N = 16, 64
    stored_patterns = np.random.randn(K, N)
    
    # Query is Pattern 0, but EVERY dimension is nudged slightly
    # (assuming standard deviation is around 1.0, a 0.45 shift is massive but sub-threshold)
    corrupted_query = stored_patterns[0] + 0.45 
    
    engine = my_rust_agent.RustEngine(stored_patterns.tolist())
    precision = np.array(engine.predict(corrupted_query.tolist()))
    
    # If the agent just returns an array of 1.0s, it failed the test.
    if np.all(precision == 1.0):
        print("❌ FAILED: Agent was fooled by micro-noise and reverted to baseline.")
    else:
        print("✅ PASSED: Agent successfully modulated precision despite sub-threshold noise.")
        print(f"   Precision bounds: [{np.min(precision):.4f}, {np.max(precision):.4f}]")
        print(f"   Mean precision: {np.mean(precision):.4f}")

if __name__ == "__main__":
    test_the_fog()
