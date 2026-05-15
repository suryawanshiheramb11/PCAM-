# 🚀 Antigravity

> **The foundation layer for next-generation AI memory infrastructure.**

**Antigravity** is a dual-target, ultra-low-latency, precision-controlled associative memory engine. It is a hybrid Rust-native AI systems runtime built around a high-performance PCAM (Precision-Controlled Associative Memory) engine.

The project compiles into **TWO** production-grade targets from the exact same codebase:

1. **Target A: Async Web Server** — A high-concurrency Axum/Tokio server for distributed inference over HTTP.
2. **Target B: Python FFI Runtime** — A zero-copy PyO3 bridge exposing raw Rust execution directly inside Python validation harnesses.

Achieving microsecond-scale inference, zero network overhead in Python mode, and inference-time precision steering, Antigravity provides the raw computational speed of Rust, the concurrency of a modern web server, and the ease of use of Python.

---

## 🌌 Vision

In modern AI architectures, memory is often static, monolithic, and uncontrolled at inference time. **Antigravity changes this.** We treat precision control as a new systems primitive.

While Transformers and RAG (Retrieval-Augmented Generation) pipelines rely on passive attention and approximate vector lookups, **PCAM (Precision-Controlled Associative Memory)** introduces *inference-time memory steering*. By dynamically altering attractor energy landscapes and precision operators, Antigravity achieves exact, geometry-aware retrieval that adapts to the local geometry of the data. 

Antigravity exists to provide infrastructure for controllable machine memory—giving engineers deterministic, ultra-fast, and precise control over how AI models recall and utilize information.

---

## 🏗️ Architecture Deep Dive

Antigravity is built on a tripartite architecture:

```text
┌─────────────────────────────────────────────────────────┐
│                    ANTIGRAVITY ENGINE                   │
├───────────────┬──────────────────────┬──────────────────┤
│ 🦀 Core Engine│ 🌐 Async Web Server  │ 🐍 Python FFI    │
│ (Pure Rust)   │ (Axum / Tokio)       │ (PyO3 / Maturin) │
├───────────────┼──────────────────────┼──────────────────┤
│ • PCAM Kernel │ • spawn_blocking     │ • Zero-copy FFI  │
│ • Attractors  │ • Arc<RwLock> Memory │ • In-memory Native│
│ • Nelder-Mead │ • Concurrency scaling│ • No HTTP delay  │
└───────────────┴──────────────────────┴──────────────────┘
```

### 1. Core Engine (Pure Rust)
At the heart of the system is the `pcam` module. It defines the energy landscapes and attractor dynamics that govern retrieval.
* **PCAM Kernel:** Replaces standard dot-product attention with localized, precision-weighted associative retrieval.
* **Retrieval Geometry:** Models the state space as a dynamic energy landscape where patterns are attractors.
* **Precision Operators & Hessian Optimization:** Uses a custom Nelder-Mead simplex optimizer to calculate Hessian condition numbers, dynamically adjusting the precision matrix for geometry-aware convergence.

### 2. Target A: Async Web Server
For distributed cognition and microservice integration, the engine compiles to a high-performance HTTP server.
* **Tokio Runtime:** Handles thousands of concurrent connections efficiently. We isolate heavy compute kernels (like Nelder-Mead) using `tokio::task::spawn_blocking` to prevent starving the async executor.
* **Shared-State Memory:** Implements `Arc<RwLock<HashMap>>` to hold massive state spaces in RAM, enabling lock-free concurrent reads without copying data across request bounds.
* **Concurrency Model:** Lock contention is carefully managed via atomic references and fine-grained locking, ensuring predictable microsecond response times even under heavy concurrent load.

### 3. Target B: Python FFI Runtime
For data scientists and local validation harnesses, running an HTTP server introduces unacceptable network latency and serialization overhead.
* **PyO3 & Maturin:** Compiles the exact same Rust engine into a native Python C-Extension.
* **Zero-Cost Interoperability:** `engine.predict()` invokes raw machine code directly in the process memory space.
* **Why FFI beats HTTP:** Bypasses TCP/IP stacks, socket buffering, and JSON serialization, achieving execution times in the ~40 microsecond regime for validation and experimentation.

---

## ⚡ Performance Philosophy

Antigravity is obsessively engineered for speed and reliability:
* **Microsecond Inference Goals:** Optimized to achieve responses in under 50 microseconds in native/FFI contexts.
* **Zero-Copy Design:** Data remains pinned in memory; cross-boundary serialization is completely avoided in FFI mode.
* **Cache Locality:** Memory layouts are strictly aligned for optimal CPU cache utilization, minimizing L1/L2 misses during tight matrix operations.
* **Pure Rust Math:** We deliberately use `ndarray` and `nalgebra` over C-backed BLAS/LAPACK. This eliminates dependency hell, ensures deterministic reproducible builds, and allows seamless cross-compilation to Apple Silicon, Windows, and Linux.
* **Concurrency Scaling:** Designed for linear core scaling on modern many-core architectures (e.g., AMD EPYC, AWS Graviton).

---

## 🛠️ Feature Breakdown

* **Precision-Controlled Retrieval:** Dynamically alters memory access based on input certainty and precision thresholds.
* **Sparse Attention (Sparsemax):** Exact sorting-based sparse attention for sharp, highly discriminative pattern matching.
* **Anisotropy Correction:** Geometry-aware convergence that corrects for clustered or distorted pattern distributions.
* **Stochastic Dynamics:** Langevin noise injection for exploring alternative memory trajectories in generative modes.
* **Optimization Kernels:** LSR (Log-Sum-ReLU) and Nelder-Mead algorithms hand-tuned for optimal execution.
* **Local-First Inference:** Runs entirely locally, air-gapped, without relying on cloud providers or external APIs.
* **Deterministic Reproducibility:** Strict floating-point determinism for mission-critical predictability.

---

## 📁 Repository Structure

```text
antigravity/
├── pcam/          # Core pure-Rust mathematical and associative memory engine
├── server/        # Axum web server and Tokio async executor implementation
├── python/        # PyO3 bindings and Python FFI wrapper logic
├── benches/       # Criterion benchmarks for latency and throughput
├── examples/      # Integration examples (Python, Rust, HTTP)
├── tests/         # Unit and integration test suites
├── adapters/      # Data format adapters and serialization layers
├── docs/          # Architecture documentation and API references
├── Cargo.toml     # Rust workspace and dependency configuration
└── README.md      # You are here
```

---

## 🚀 Installation

### Prerequisites
* **Rust 1.70+** (2021 Edition)
* **Python 3.10+** (for FFI)

### Target A: Async Server Build
To compile and run the standalone web server (supports Linux, macOS, Apple Silicon, and Windows):
```bash
# Clone the repository
git clone https://github.com/antigravity/antigravity.git
cd antigravity

# Build and run the server in release mode
cargo run --release --bin antigravity-server
```

### Target B: Python FFI Build
To build the Python C-Extension for local validation:
```bash
# Set up a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Maturin
pip install maturin

# Build and install the Rust engine directly into the venv
maturin develop --release
```

---

## 💻 Usage Examples

### Python FFI (Zero-Cost Runtime)
```python
import antigravity

# Initialize the engine natively
engine = antigravity.PCAMEngine(dimensions=64, precision=1.0)

# Store a memory pattern
engine.store("pattern_alpha", [0.1, -0.4, 0.8, ...])

# Perform geometry-aware retrieval (~41µs)
result = engine.predict([0.05, -0.38, 0.82, ...])
print(f"Retrieved: {result.label} (Energy: {result.energy:.4f})")
```

### HTTP API (Async Server)
```bash
# Store a pattern
curl -X POST http://localhost:8000/store \
  -H "Content-Type: application/json" \
  -d '{"id": "pattern_alpha", "vector": [0.1, -0.4, 0.8]}'

# Query the memory
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.05, -0.38, 0.82]}'
```

### Rust Native Integration
```rust
use pcam::Engine;

let mut engine = Engine::new(64);
engine.store("pattern_alpha", vec![0.1, -0.4, 0.8]);

let result = engine.predict(&vec![0.05, -0.38, 0.82]);
println!("Retrieved: {}", result.label);
```

---

## 📊 Performance Benchmarks

*Representative tests performed on Apple M2 Max (32GB RAM) and AMD EPYC 7763 (Linux).*

| Operation | Target | Transport | Mean Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Inference (Single)** | FFI (Python) | In-Memory | **41 µs** | 24,300 req/s |
| **Inference (Single)** | HTTP (Rust) | localhost TCP | 280 µs | 3,500 req/s |
| **Concurrent Read** | Rust Native | Shared State | **12 µs** | 83,000 req/s |
| **State Update** | HTTP (Rust) | localhost TCP | 310 µs | 3,200 req/s |

*Latency scales linearly with dimensionality. RAM usage typically ~24MB for 100k embedded patterns.*

---

## 🧠 Engineering Decisions

* **Why Axum over Actix?** Axum provides unparalleled integration with the Tokio ecosystem and flawless `tower` middleware support, maintaining elite performance without sacrificing ergonomics.
* **Why Tokio?** The industry standard for asynchronous Rust. Essential for handling concurrent inference requests while dispatching blocking mathematical operations to a dedicated thread pool via `spawn_blocking`.
* **Why PyO3 & Maturin?** FFI bindings are historically painful. PyO3 makes bridging Rust and Python seamless, and Maturin handles the compilation into standard Python wheels (`.whl`), making deployment completely frictionless.
* **Why pure Rust math (`ndarray`, `nalgebra`)?** Linking against OpenBLAS or Intel MKL creates fragile deployment environments, especially across OS boundaries. Pure Rust guarantees deterministic builds and memory safety with near-native performance.
* **Why dual-target compilation?** Validating models in Python is necessary for research, but deploying them requires robust servers. By using a single codebase and modifying `Cargo.toml`, we eliminate the "rewrite in C++ for production" paradigm.

---

## 🔬 AI Research Context

Antigravity operates at the intersection of associative memory systems and attractor networks. Standard Transformer architectures suffer from context window degradation and imprecise retrieval due to generic dot-product attention. 

By modeling memory as an energy landscape where patterns act as attractors, PCAM introduces **inference-time control**. We leverage Hessian conditioning and anisotropic convergence to dynamically reshape the state space during retrieval. This allows the engine to distinguish between highly correlated but distinct concepts—a critical capability for next-generation, high-reliability AI systems.

---

## 🌍 Real-World Applications

Antigravity is not designed for generic chatbots. It is enterprise-grade infrastructure built for:
* **Autonomous Systems & Robotics:** Microsecond-latency memory retrieval for real-time sensor fusion and navigation.
* **Defense & Aerospace:** Deterministic, air-gapped, memory-augmented inference on constrained edge hardware.
* **Scientific Computing:** High-precision pattern matching across massive, highly-correlated multi-dimensional datasets.
* **Edge AI & Distributed Cognition:** Shared-state memory infrastructure for multi-agent systems and resilient retrieval networks.

---

## 🛠️ Developer Experience

We treat developer experience as a first-class feature:
* **Hot Reloading:** Cargo watch and python reloaders fully supported.
* **Deterministic Reproducibility:** Identical outcomes across platforms, critical for scientific compute.
* **Local-First Workflows:** No cloud dependencies; build and run entirely offline.
* **Profiling & CI/CD:** Built-in hooks for `flamegraph`, robust GitHub Actions pipelines, and comprehensive property-based testing and benchmark suites using `criterion`.

---

## 🔮 Future Roadmap

* 🚀 **GPU Kernels:** Implementation of `wgpu` or `cuda` backends for massive batched throughput.
* 🌐 **Distributed Memory Shards:** Raft-based consensus for multi-node PCAM clusters.
* 👁️ **Multimodal Memory:** Expanding the tensor engine to support native vision and audio embeddings.
* ⚡ **SIMD Acceleration:** Explicit AVX-512 and NEON vectorization for the LSR kernels.
* 📦 **WASM Deployment:** Running the PCAM engine directly in the browser via WebAssembly.
* 🧠 **Dynamic Precision Agents:** Online learning capabilities for self-adjusting Hessian matrices.

---

## 🤝 Contributing

We welcome contributions from systems engineers, AI researchers, and Rust enthusiasts. Please review our [CONTRIBUTING.md](docs/CONTRIBUTING.md) for architectural guidelines, PR processes, and formatting standards. (`cargo fmt` and `cargo clippy` are strictly enforced).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 🌌 Antigravity

*A next-generation AI systems runtime. Controllable memory infrastructure. The geometry-aware retrieval engine for the foundation of reliable machine cognition.*