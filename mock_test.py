import numpy as np
import time
import my_rust_agent

def run_mock_test():
    print("=== Antigravity PCAM Local Mock Test ===\n")
    
    # 1. Setup Dummy Data (Matching hackathon specs: K=16 patterns, N=64 dims)
    K = 16
    N = 64
    print(f"Generating {K} dummy patterns of {N} dimensions...")
    # np.random.randn creates realistic normal-distribution noise
    stored_patterns = np.random.randn(K, N) 
    corrupted_query = np.random.randn(N)
    
    # 2. Initialize the Rust Engine
    print("Initializing RustEngine...")
    start_init = time.perf_counter()
    # We pass the list to Rust just like the real harness will
    engine = my_rust_agent.RustEngine(stored_patterns.tolist())
    init_time = time.perf_counter() - start_init
    print(f"-> Init time: {init_time:.5f} seconds\n")
    
    # 3. Run the Prediction
    print("Running predict()...")
    start_pred = time.perf_counter()
    precision = np.array(engine.predict(corrupted_query.tolist()))
    pred_time = time.perf_counter() - start_pred
    
    # 4. Validate the Outputs against Hackathon Rules
    print(f"-> Predict time: {pred_time:.6f} seconds")
    print(f"-> Output shape: {precision.shape} (Expected: ({N},))")
    
    if precision.shape == (N,):
        print("✅ Shape is correct.")
    else:
        print("❌ Shape is incorrect!")
        
    min_val, max_val = np.min(precision), np.max(precision)
    print(f"-> Output bounds: [{min_val:.4f}, {max_val:.4f}] (Expected: [0.1, 10.0])")
    
    if np.all((precision >= 0.1) & (precision <= 10.0)):
        print("✅ Bounds are correct.")
    else:
        print("❌ Bounds are violated!")

if __name__ == "__main__":
    run_mock_test()
