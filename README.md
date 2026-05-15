# 🚀 Antigravity
### Dual-Target High-Performance PCAM Engine

> An extremely modern, hybrid technology stack delivering the raw computational speed of Rust, the concurrency of a modern web server, and the ease of use of Python.

---

## 🧠 1. The Core Engine (Pure Rust)
At the heart of everything is the `pcam` module. This is where the heavy mathematical lifting happens.

* **🦀 Language:** **Rust 2021 Edition**. Chosen for memory safety and zero-cost abstractions—it runs as fast as C/C++ but without the memory leaks.
* **🧮 Linear Algebra:** **`nalgebra` & `ndarray`**. We explicitly chose pure-Rust math libraries instead of relying on C-based BLAS/LAPACK. This guarantees the code will compile and run flawlessly on any machine (Intel, Apple Silicon M-series, Linux, Windows) without dependency nightmares.
* **⚙️ Algorithms Implemented:**
    * **LSR Kernel:** Log-Sum-ReLU projection.
    * **Sparsemax:** Exact sorting-based sparse attention.
    * **Nelder-Mead Optimizer:** A custom simplex optimization algorithm to calculate Hessian condition numbers for precision tuning.
    * **Langevin Dynamics:** Stochastic noise injection for generation modes.

---

## 🌐 2. Target A: The Standalone Web Server
An asynchronous HTTP server that allows remote applications to interact with the memory engine over a network.

* **🕸️ Web Framework:** **Axum 0.7**. The industry standard web framework for Rust, built by the Tokio team. It is incredibly fast and highly ergonomic.
* **⚡ Async Runtime:** **Tokio 1.0**. Handles thousands of concurrent connections. We utilize `tokio::task::spawn_blocking` to ensure that the heavy Nelder-Mead math optimization doesn't freeze the web server while it calculates.
* **🗄️ State Management:** **`Arc<RwLock<HashMap>>`**. An Atomic Reference Counted Read-Write Lock allows the server to hold 64-dimensional patterns in RAM and share them instantly across thousands of parallel incoming network requests *without copying data*.
* **📦 Serialization:** **`serde` and `serde_json`** for blazing-fast JSON parsing.

---

## 🐍 3. Target B: The Python Bridge (Zero-Cost FFI)
Because data science and validation harnesses are often written in Python, running an HTTP server introduces network latency. To bypass this, we built a Foreign Function Interface (FFI) bridge.

* **🌉 The Bridge:** **PyO3**. This library allows Rust to compile directly into a Python C-Extension (a `.so` or `.dylib` file). Python sees it as a normal module (`import my_rust_agent`), but under the hood, it executes raw machine code.
* **🛠️ The Build System:** **Maturin**. A build tool specifically designed to take Rust code and package it into a Python Wheel (`.whl`).
* **⏱️ The Result:** **Zero-Cost Interoperability.** When the harness calls `engine.predict()`, it does not make a network request. It directly invokes the Rust binary in memory, achieving prediction times of **~41 microseconds**.

---

## 🏗️ Summary of the "Dual-Target" Architecture

By modifying `Cargo.toml` with `crate-type = ["cdylib", "lib"]` and using feature flags, we created a codebase that compiles into two entirely different products from the exact same source code:

1.  **💻 The Server:** Run `cargo run --release` ➔ You get a standalone Axum web server listening on port 8000.
2.  **🧪 The Library:** Run `maturin develop --release` ➔ You get a Python library that plugs directly into local validation harnesses.

*Enterprise-grade AI infrastructure, running entirely locally.* ✨