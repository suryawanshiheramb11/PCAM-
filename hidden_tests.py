import numpy as np
import my_rust_agent

def run_hidden_test_cases():
    print("=== Antigravity PCAM: Comprehensive Edge Case Testing ===\n")
    
    N = 64
    success_count = 0
    total_tests = 0
    
    def evaluate(test_name, engine, query, expect_failure=False):
        nonlocal success_count, total_tests
        total_tests += 1
        print(f"[{total_tests}] Testing: {test_name}")
        
        try:
            precision = np.array(engine.predict(query))
            if expect_failure:
                print("   ❌ FAILED (Expected an exception, but it succeeded)")
                return
                
            valid_shape = precision.shape == (N,)
            valid_bounds = np.all((precision >= 0.1) & (precision <= 10.0))
            
            if valid_shape and valid_bounds:
                print(f"   ✅ PASSED. Shape: {precision.shape}, Bounds: [{np.min(precision):.4f}, {np.max(precision):.4f}]")
                success_count += 1
            else:
                print("   ❌ FAILED")
                if not valid_shape:
                    print(f"      -> Bad shape: {precision.shape}")
                if not valid_bounds:
                    print(f"      -> Bad bounds: [{np.min(precision):.4f}, {np.max(precision):.4f}]")
                    
        except Exception as e:
            if expect_failure:
                print(f"   ✅ PASSED. Handled expected failure: {e}")
                success_count += 1
            else:
                print(f"   ❌ FAILED with unexpected exception: {e}")

    # --- TEST 1: Extreme Noise/Corruption in Query ---
    K = 16
    patterns = np.random.randn(K, N).tolist()
    engine = my_rust_agent.RustEngine(patterns)
    # Query with massive values
    query_extreme = (np.random.randn(N) * 100000).tolist()
    evaluate("Extreme Noise in Query (values > 100,000)", engine, query_extreme)

    # --- TEST 2: All Zeros Query ---
    query_zeros = np.zeros(N).tolist()
    evaluate("All Zeros Query", engine, query_zeros)

    # --- TEST 3: Identical Patterns ---
    # Store the exact same pattern 16 times
    single_pat = np.random.randn(N).tolist()
    patterns_identical = [single_pat for _ in range(K)]
    engine_id = my_rust_agent.RustEngine(patterns_identical)
    evaluate("Identical Stored Patterns", engine_id, np.random.randn(N).tolist())

    # --- TEST 4: Single Pattern Stored ---
    engine_single = my_rust_agent.RustEngine([single_pat])
    evaluate("Only 1 Pattern Stored", engine_single, np.random.randn(N).tolist())

    # --- TEST 5: Zero Patterns Stored (Empty Memory) ---
    engine_empty = my_rust_agent.RustEngine([])
    evaluate("Zero Patterns Stored", engine_empty, np.random.randn(N).tolist())

    # --- TEST 6: Very Large Stored Patterns ---
    patterns_large = (np.random.randn(K, N) * 1e6).tolist()
    engine_large = my_rust_agent.RustEngine(patterns_large)
    evaluate("Very Large Stored Pattern Values", engine_large, np.random.randn(N).tolist())

    # --- TEST 7: Query is exactly an existing pattern ---
    exact_query = patterns[0]
    evaluate("Query perfectly matches Pattern[0]", engine, exact_query)

    # --- TEST 8: Dimension Mismatch Query (Should Fail Safely) ---
    query_bad_dim = np.random.randn(32).tolist() # N is 64
    evaluate("Query Dimension Mismatch", engine, query_bad_dim, expect_failure=True)

    print(f"\n=== Summary: {success_count}/{total_tests} Tests Passed ===")

if __name__ == "__main__":
    run_hidden_test_cases()
