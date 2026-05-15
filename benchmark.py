import time
import random
import my_rust_agent

def generate_random_vector(dim):
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]

def generate_patterns(n_patterns, dim):
    return [generate_random_vector(dim) for _ in range(n_patterns)]

def run_benchmark():
    print("=== Antigravity PCAM Hackathon Benchmark ===")
    
    # 1. Small test (quick)
    dim = 64
    n_patterns = 10
    n_queries = 100
    
    print(f"\n[Test 1] Small scale: {n_patterns} patterns, {n_queries} queries, {dim} dims")
    patterns = generate_patterns(n_patterns, dim)
    queries = [generate_random_vector(dim) for _ in range(n_queries)]
    
    start_time = time.time()
    engine = my_rust_agent.RustEngine(patterns)
    init_time = time.time() - start_time
    
    start_time = time.time()
    for q in queries:
        engine.predict(q)
    predict_time = time.time() - start_time
    
    print(f"  -> Engine Init (Offline optimization): {init_time:.4f}s")
    print(f"  -> Predict ({n_queries} queries):      {predict_time:.4f}s ({predict_time/n_queries*1000:.2f}ms / query)")

    # 2. Medium test (hackathon scale)
    n_patterns = 100
    n_queries = 1000
    
    print(f"\n[Test 2] Hackathon scale: {n_patterns} patterns, {n_queries} queries, {dim} dims")
    patterns = generate_patterns(n_patterns, dim)
    queries = [generate_random_vector(dim) for _ in range(n_queries)]
    
    start_time = time.time()
    engine = my_rust_agent.RustEngine(patterns)
    init_time = time.time() - start_time
    
    start_time = time.time()
    for q in queries:
        engine.predict(q)
    predict_time = time.time() - start_time
    
    print(f"  -> Engine Init (Offline optimization): {init_time:.4f}s")
    print(f"  -> Predict ({n_queries} queries):      {predict_time:.4f}s ({predict_time/n_queries*1000:.2f}ms / query)")

    # 3. Validation
    print("\n[Test 3] Output Validation")
    # Output must be length 64, and bounded between 0.1 and 10.0
    res = engine.predict(queries[0])
    valid_length = len(res) == dim
    valid_bounds = all(0.1 <= x <= 10.0 for x in res)
    print(f"  -> Valid output length?  {'Yes' if valid_length else 'No'}")
    print(f"  -> Valid output bounds?  {'Yes' if valid_bounds else 'No'} (min: {min(res):.2f}, max: {max(res):.2f})")

if __name__ == "__main__":
    run_benchmark()
