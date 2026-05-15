import numpy as np
import my_rust_agent

def test_the_eclipse():
    print("\n--- TEST: The Eclipse (Massive Partial Masking) ---")
    
    K, N = 200, 64
    stored_patterns = np.random.randn(K, N)
    
    # 1. The True Target is Pattern 0
    corrupted_query = np.copy(stored_patterns[0])
    
    # 2. The Eclipse: We heavily mask/corrupt 40% of the dimensions (25 dims).
    # We leave the first 39 dimensions absolutely pristine.
    mask_indices = np.random.choice(N, 25, replace=False)
    corrupted_query[mask_indices] = 100.0  # Massive corruption
    
    engine = my_rust_agent.RustEngine(stored_patterns.tolist(), np.eye(N).tolist(), 0.1, 10.0)
    precision = np.array(engine.predict(corrupted_query.tolist()))
    
    # 3. Evaluation
    # We want to check the precision of the PRISTINE dimensions. 
    # If the engine correctly kept Pattern 0 in the Top-3, these should be near 10.0.
    # If Pattern 0 was eclipsed and fell out, these will crash.
    pristine_indices = np.setdiff1d(np.arange(N), mask_indices)
    mean_pristine_p = np.mean(precision[pristine_indices])
    mean_corrupted_p = np.mean(precision[mask_indices])
    
    print(f"-> Mean Precision on 39 Pristine Dims: {mean_pristine_p:.4f} (Target: > 8.0)")
    print(f"-> Mean Precision on 25 Masked Dims:   {mean_corrupted_p:.4f} (Target: < 2.0)")
    
    if mean_pristine_p < 5.0:
        print("❌ FAILED: The Eclipse worked. The massive error on the masked dimensions")
        print("   dragged the True Pattern out of the Top-3. The engine hallucinated.")
    else:
        print("✅ PASSED: The engine ignored the mask, found the True Pattern, and surgically isolated the noise!")

def test_delta_accuracy_pull():
    print("\n--- TEST: Retrieval Δ Accuracy (Dynamics Pull) ---")
    
    K, N = 16, 64
    stored_patterns = np.random.randn(K, N)
    
    # 1. True Target
    stored_patterns[0] = np.zeros(N)
    
    # 2. Distractor Target (The Trap)
    stored_patterns[1] = np.ones(N) * 2.0
    
    # 3. The Query: Starts exactly on the True Target
    corrupted_query = np.copy(stored_patterns[0])
    
    # 4. The Trap: 25 dims get hit with a massive +10.0 noise spike.
    # This pushes the Euclidean center of mass closer to the Distractor for the Baseline.
    corrupted_query[0:25] += 10.0 
    
    # --- Execute ---
    engine = my_rust_agent.RustEngine(stored_patterns.tolist(), np.eye(N).tolist(), 0.1, 10.0)
    agent_precision = np.array(engine.predict(corrupted_query.tolist()))
    baseline_precision = np.ones(N)
    
    # --- The Physics Simulation (Energy = Precision * Distance) ---
    # LOWER energy = stronger pull into that valley.
    def calculate_energy(precision, target_pattern, query):
        return np.sum(precision * (target_pattern - query)**2)
    
    base_energy_true = calculate_energy(baseline_precision, stored_patterns[0], corrupted_query)
    base_energy_distractor = calculate_energy(baseline_precision, stored_patterns[1], corrupted_query)
    
    agent_energy_true = calculate_energy(agent_precision, stored_patterns[0], corrupted_query)
    agent_energy_distractor = calculate_energy(agent_precision, stored_patterns[1], corrupted_query)
    
    print(f"-> BASELINE Energy to True Target: {base_energy_true:.2f}")
    print(f"-> BASELINE Energy to Distractor:  {base_energy_distractor:.2f}")
    print(f"   Baseline Winner: {'True Target ✅' if base_energy_true < base_energy_distractor else 'Distractor ❌'}")
    
    print(f"\n-> AGENT Energy to True Target:    {agent_energy_true:.2f}")
    print(f"-> AGENT Energy to Distractor:     {agent_energy_distractor:.2f}")
    print(f"   Agent Winner:    {'True Target ✅' if agent_energy_true < agent_energy_distractor else 'Distractor ❌'}")
    
    # The condition for victory: Baseline must fail, Agent must win.
    if (base_energy_true > base_energy_distractor) and (agent_energy_true < agent_energy_distractor):
        print("\n🏆 MASSIVE SUCCESS: Your agent successfully flipped a guaranteed failure into a correct retrieval!")
        print("   This translates to a massive Δ Accuracy gain.")
    else:
        print("\n⚠️ WARNING: The trap failed or the agent did not overcome it.")

def test_the_flood():
    print("\n--- TEST: The Flood (Majority Masking Exploitation) ---")
    
    K, N = 200, 64
    stored_patterns = np.random.randn(K, N)
    
    # 1. True Target
    stored_patterns[0] = np.zeros(N)
    corrupted_query = np.copy(stored_patterns[0])
    
    # 2. The Flood: We corrupt 35 dimensions (55% of the vector).
    # Your Trimmed MSE can only hide 25. The remaining 10 will leak into the math.
    mask_indices = np.random.choice(N, 35, replace=False)
    corrupted_query[mask_indices] = 100.0  
    
    engine = my_rust_agent.RustEngine(stored_patterns.tolist(), np.eye(N).tolist(), 0.1, 10.0)
    precision = np.array(engine.predict(corrupted_query.tolist()))
    
    # Check the pristine dimensions. If the True Target was dropped, these will crash.
    pristine_indices = np.setdiff1d(np.arange(N), mask_indices)
    mean_pristine_p = np.mean(precision[pristine_indices])
    
    print(f"-> Mean Precision on 29 Pristine Dims: {mean_pristine_p:.4f} (Target: > 8.0)")
    
    if mean_pristine_p < 5.0:
        print("❌ FAILED: The Flood breached the Trimmed MSE hull. The True Target was lost.")
    else:
        print("✅ PASSED: The engine survived a majority-masking attack!")

def test_the_mirage():
    print("\n--- TEST: The Mirage (Top-K Decoy Displacement) ---")
    
    K, N = 200, 64
    stored_patterns = np.random.randn(K, N)
    
    # 1. True Target
    stored_patterns[0] = np.zeros(N)
    
    # 2. Query has uniform sub-threshold noise
    corrupted_query = np.ones(N) * 0.5
    
    # 3. The Mirages: 3 Decoy patterns designed to perfectly mimic the query's noise
    # but they are structurally fake.
    stored_patterns[1] = np.ones(N) * 0.5
    stored_patterns[2] = np.ones(N) * 0.5
    stored_patterns[3] = np.ones(N) * 0.5
    
    engine = my_rust_agent.RustEngine(stored_patterns.tolist(), np.eye(N).tolist(), 0.1, 10.0)
    precision = np.array(engine.predict(corrupted_query.tolist()))
    
    # If the engine locks onto the Mirages, it will think the query is PERFECT (Distance 0)
    # and output 10.0 precision, pulling the system into a fake memory valley.
    mean_p = np.mean(precision)
    print(f"-> Mean Precision: {mean_p:.4f}")
    
    if mean_p > 9.0:
        print("❌ FAILED: The engine locked onto the Mirages and gave 10.0 precision to noise!")
        print("   It pushed the True Target out of the Top 3.")
    else:
        print("✅ PASSED: The engine saw through the Mirage and maintained healthy skepticism.")

def test_the_poisoned_well():
    print("\n--- TEST: The Poisoned Well (Adversarial Blending) ---")
    
    K, N = 200, 64
    stored_patterns = np.random.randn(K, N)
    
    # 1. True Target
    stored_patterns[0] = np.ones(N) * 2.0
    
    # 2. Poisoned Neighbors (98% identical, but completely wrong on dim 10)
    stored_patterns[1] = np.copy(stored_patterns[0])
    stored_patterns[1, 10] = -2.0  # Poison
    
    stored_patterns[2] = np.copy(stored_patterns[0])
    stored_patterns[2, 10] = -2.0  # Poison
    
    # 3. The Query is absolutely perfect. 
    corrupted_query = np.copy(stored_patterns[0])
    
    engine = my_rust_agent.RustEngine(stored_patterns.tolist(), np.eye(N).tolist(), 0.1, 10.0)
    precision = np.array(engine.predict(corrupted_query.tolist()))
    
    # Dimension 10 should have 10.0 precision because the query is perfect.
    dim_10_precision = precision[10]
    
    print(f"-> Precision on Dim 10: {dim_10_precision:.4f} (Target: > 9.0)")
    
    if dim_10_precision < 8.0:
        print("❌ FAILED: The Softmax blending poisoned a perfect query.")
        print("   The near-identical neighbors dragged the blended target down.")
    else:
        print("✅ PASSED: The engine isolated the true target without bleeding weights!")

if __name__ == "__main__":
    test_the_eclipse()
    test_delta_accuracy_pull()
    test_the_flood()
    test_the_mirage()
    test_the_poisoned_well()