import numpy as np
import time
import my_rust_agent

def test_the_chimera():
    print("\n--- TEST: The Chimera (Full-Scale Boundary Attack) ---")
    
    # 1. Full Hackathon Scale
    K, N = 200, 64 
    stored_patterns = np.random.randn(K, N)
    
    # 2. Create the Chimera: First 32 dims from Pattern 0, Last 32 dims from Pattern 1
    corrupted_query = np.zeros(N)
    corrupted_query[0:32] = stored_patterns[0, 0:32]
    corrupted_query[32:64] = stored_patterns[1, 32:64]
    
    # 3. Add a Deceptive Anchor: perfectly inject a piece of Pattern 2 to confuse the math
    corrupted_query[15] = stored_patterns[2, 15]
    corrupted_query[45] = stored_patterns[2, 45]
    
    # 4. Measure Execution Speed at Scale
    engine = my_rust_agent.RustEngine(stored_patterns.tolist())
    
    start_time = time.perf_counter()
    precision = np.array(engine.predict(corrupted_query.tolist()))
    exec_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    
    # 5. Output Analysis
    print(f"-> Execution Time (K=200): {exec_time:.3f} ms (Target: < 1.0 ms)")
    
    # Did it output all 1.0s? (Failed to detect the anomaly)
    if np.all(precision == 1.0):
        print("❌ FAILED: Engine completely missed the collision and output baseline 1.0s.")
        return

    print("✅ Engine detected the anomaly.")
    
    # How did it handle the two halves?
    p0_half_mean = np.mean(precision[0:32])
    p1_half_mean = np.mean(precision[32:64])
    
    print(f"-> Mean Precision on Pattern 0's half: {p0_half_mean:.4f}")
    print(f"-> Mean Precision on Pattern 1's half: {p1_half_mean:.4f}")
    
    # A naive engine will have high precision on one half and terrible precision on the other.
    # A smart engine (using Softmax blending of the top 3 neighbors) will balance them.
    difference = abs(p0_half_mean - p1_half_mean)
    if difference > 2.0:
        print("⚠️ WARNING: Engine violently favored one pattern over the other.")
        print("   In a true 50/50 split, it should blend confidence or reduce precision overall.")
    else:
        print("✅ PASSED: Engine maintained symmetry and handled the boundary state gracefully!")

if __name__ == "__main__":
    test_the_chimera()
