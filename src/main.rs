use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use my_rust_agent::pcam::dynamics::unified_dynamics;
use my_rust_agent::pcam::math::sq_dist;
use my_rust_agent::pcam::optimize::optimize_precision;

// ─── Constants ───────────────────────────────────────────────────────

const MAX_PATTERNS: usize = 1000;
const MAX_DIM: usize = 4096;
const MAX_DYNAMICS_ITERS: usize = 300;
const DYNAMICS_TOL: f64 = 1e-6;

// ─── Domain types ────────────────────────────────────────────────────

#[derive(Clone)]
struct MemorySet {
    patterns: Array2<f64>,        // K × N
    precisions: Vec<Array1<f64>>, // per pattern, each length N
}

#[derive(Clone)]
struct GlobalModel {
    r: Array2<f64>,
    beta: f64,
    eta: f64,
    dim: usize,
}

// Use Arc<MemorySet> to avoid cloning the entire memory set on each retrieve
type SharedState = Arc<RwLock<HashMap<String, Arc<MemorySet>>>>;

// ─── Request / response DTOs ─────────────────────────────────────────

#[derive(Deserialize)]
struct UploadRequest {
    patterns: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct UploadResponse {
    set_id: String,
    message: String,
    num_patterns: usize,
    dimension: usize,
}

#[derive(Deserialize)]
struct RetrieveRequest {
    set_id: String,
    query: Vec<f64>,
    #[serde(default = "default_temperature")]
    temperature: f64,
}

fn default_temperature() -> f64 {
    0.0
}

#[derive(Serialize)]
struct RetrieveResponse {
    cleaned_query: Vec<f64>,
    converged_pattern_id: usize,
    distance: f64,
    confidence: f64,
    iterations_hint: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Safely read from the RwLock, returning a 500 if poisoned instead of panicking.
fn read_state(state: &SharedState) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<MemorySet>>>, (StatusCode, Json<ErrorResponse>)> {
    state.read().map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: "Internal state lock poisoned".into(),
        }))
    })
}

fn write_state(state: &SharedState) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<MemorySet>>>, (StatusCode, Json<ErrorResponse>)> {
    state.write().map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: "Internal state lock poisoned".into(),
        }))
    })
}

fn err_response(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.into() }))
}

// ─── Handlers ────────────────────────────────────────────────────────

async fn health() -> &'static str {
    "OK"
}

async fn upload(
    State((app_state, model)): State<(SharedState, GlobalModel)>,
    Json(payload): Json<UploadRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // ── Input validation ──
    if payload.patterns.is_empty() {
        return Err(err_response(StatusCode::UNPROCESSABLE_ENTITY, "No patterns provided"));
    }
    if payload.patterns[0].is_empty() {
        return Err(err_response(StatusCode::UNPROCESSABLE_ENTITY, "Patterns cannot be empty vectors"));
    }
    if payload.patterns.len() > MAX_PATTERNS {
        return Err(err_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Too many patterns (max {})", MAX_PATTERNS),
        ));
    }

    let n = payload.patterns[0].len();
    if n > MAX_DIM {
        return Err(err_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Dimension too large (max {})", MAX_DIM),
        ));
    }
    if n != model.dim {
        return Err(err_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Pattern dimension {} does not match model dimension {}",
                n, model.dim
            ),
        ));
    }
    if payload.patterns.iter().any(|p| p.len() != n) {
        return Err(err_response(StatusCode::UNPROCESSABLE_ENTITY, "Inconsistent pattern dimensions"));
    }
    // Check for NaN/Inf in patterns
    if payload.patterns.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
        return Err(err_response(StatusCode::UNPROCESSABLE_ENTITY, "Patterns contain NaN or Inf values"));
    }

    let k = payload.patterns.len();
    let patterns_flat: Vec<f64> = payload.patterns.into_iter().flatten().collect();
    let patterns = Array2::from_shape_vec((k, n), patterns_flat)
        .map_err(|e| err_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    // Pre-compute per-pattern precision vectors offline (spawn blocking task)
    let r = model.r.clone();
    let beta = model.beta;
    let patterns_clone = patterns.clone();

    tracing::info!("Optimising precision for {} patterns of dim {}…", k, n);
    let precisions = tokio::task::spawn_blocking(move || {
        (0..k)
            .map(|mu| {
                let xi = patterns_clone.row(mu).to_owned();
                optimize_precision(&xi, &patterns_clone, &r, beta, 200)
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| err_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    tracing::info!("Precision optimisation complete for {} patterns", k);

    let set_id = Uuid::new_v4().to_string();
    let memory_set = Arc::new(MemorySet { patterns, precisions });

    write_state(&app_state)?.insert(set_id.clone(), memory_set);

    Ok(Json(UploadResponse {
        set_id,
        message: format!("Stored {} patterns of dimension {}", k, n),
        num_patterns: k,
        dimension: n,
    }))
}

async fn retrieve(
    State((app_state, model)): State<(SharedState, GlobalModel)>,
    Json(payload): Json<RetrieveRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // ── Input validation ──
    if payload.query.iter().any(|v| !v.is_finite()) {
        return Err(err_response(StatusCode::UNPROCESSABLE_ENTITY, "Query contains NaN or Inf"));
    }
    if payload.temperature < 0.0 {
        return Err(err_response(StatusCode::UNPROCESSABLE_ENTITY, "Temperature must be ≥ 0"));
    }

    // Arc clone — no deep copy of patterns/precisions
    let mem_set = {
        let store = read_state(&app_state)?;
        store
            .get(&payload.set_id)
            .ok_or_else(|| err_response(StatusCode::NOT_FOUND, "Memory set not found"))?
            .clone() // Arc::clone — cheap
    };

    let query = Array1::from_vec(payload.query.clone());
    let n = query.len();
    if n != mem_set.patterns.ncols() {
        return Err(err_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Query dimension {} does not match pattern dimension {}",
                n,
                mem_set.patterns.ncols()
            ),
        ));
    }

    // ── Class-conditional precision blending ──
    // Compute dot-product similarities between query and all patterns
    let sims = mem_set.patterns.dot(&query);
    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Softmax weights (temperature 1.0)
    let exp_sims = sims.mapv(|s| ((s - max_sim) * 1.0).exp());
    let sum_exp = exp_sims.sum();
    let weights = if sum_exp > 0.0 {
        exp_sims / sum_exp
    } else {
        Array1::ones(sims.len()) / (sims.len() as f64)
    };

    // Blend per-pattern precisions
    let mut blended: Array1<f64> = Array1::zeros(n);
    for (mu, &w) in weights.iter().enumerate() {
        if w > 1e-15 {
            let scaled: Array1<f64> = &mem_set.precisions[mu] * w;
            blended = blended + scaled;
        }
    }

    // Reliability mask (down-weight corrupted dimensions)
    let mut weighted_mean: Array1<f64> = Array1::zeros(n);
    for (mu, &w) in weights.iter().enumerate() {
        if w > 1e-15 {
            let pat: Array1<f64> = mem_set.patterns.row(mu).to_owned();
            let scaled: Array1<f64> = pat * w;
            weighted_mean = weighted_mean + scaled;
        }
    }
    let deviation: Array1<f64> = (&query - &weighted_mean).mapv(|x| x.abs());
    let reliability = deviation.mapv(|d| (-2.0 * d).exp()); // λ = 2.0
    blended = blended * &reliability;

    // Run unified dynamics
    let cleaned = unified_dynamics(
        &query,
        &mem_set.patterns,
        &model.r,
        model.beta,
        model.eta,
        &blended,
        payload.temperature,
        MAX_DYNAMICS_ITERS,
        DYNAMICS_TOL,
    );

    // Identify nearest pattern by squared Euclidean distance (proper metric)
    let (best_idx, best_dist) = (0..mem_set.patterns.nrows())
        .map(|mu| {
            let pat = mem_set.patterns.row(mu).to_owned();
            (mu, sq_dist(&cleaned, &pat))
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Normalised confidence: exp(-distance) gives [0,1] range
    let confidence = (-best_dist).exp();

    let mode = if payload.temperature < 1e-12 {
        "deterministic retrieval"
    } else {
        "stochastic generation"
    };

    Ok(Json(RetrieveResponse {
        cleaned_query: cleaned.to_vec(),
        converged_pattern_id: best_idx,
        distance: best_dist,
        confidence,
        iterations_hint: mode.into(),
    }))
}

async fn list_sets(
    State((app_state, _)): State<(SharedState, GlobalModel)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let store = read_state(&app_state)?;
    let sets: Vec<serde_json::Value> = store
        .iter()
        .map(|(id, ms)| {
            serde_json::json!({
                "set_id": id,
                "num_patterns": ms.patterns.nrows(),
                "dimension": ms.patterns.ncols(),
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "memory_sets": sets })))
}

// ─── Entrypoint ──────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Load model parameters from JSON — path configurable via env var
    let config_path = std::env::var("ANTIGRAVITY_CONFIG")
        .unwrap_or_else(|_| "src/model_params.json".into());

    let model_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .unwrap_or_else(|_| panic!("{} not found. Set ANTIGRAVITY_CONFIG env var.", config_path)),
    )
    .expect("Invalid JSON in config file");

    let beta = model_json["beta"].as_f64().unwrap_or(1.0);
    let eta = model_json["eta"].as_f64().unwrap_or(0.1);

    // Support both full matrix and "identity" shorthand for R
    let r = if model_json["R"].is_string() && model_json["R"].as_str() == Some("identity") {
        let dim = model_json["dim"].as_u64().unwrap_or(64) as usize;
        tracing::info!("Using {}×{} identity matrix for R", dim, dim);
        Array2::eye(dim)
    } else {
        let r_data: Vec<Vec<f64>> = serde_json::from_value(model_json["R"].clone())
            .expect("R must be a 2D array or \"identity\"");
        let n = r_data.len();
        let m = r_data.first().map(|r| r.len()).unwrap_or(0);
        if n != m {
            panic!("R must be square, got {}×{}", n, m);
        }
        let r_flat: Vec<f64> = r_data.into_iter().flatten().collect();
        Array2::from_shape_vec((n, n), r_flat).expect("R matrix shape error")
    };

    let dim = r.nrows();
    tracing::info!("Model loaded: dim={}, β={}, η={}", dim, beta, eta);

    let global_model = GlobalModel { r, beta, eta, dim };
    let app_state: SharedState = Arc::new(RwLock::new(HashMap::new()));

    let app = Router::new()
        .route("/health", get(health))
        .route("/upload", post(upload))
        .route("/retrieve", post(retrieve))
        .route("/sets", get(list_sets))
        .with_state((app_state, global_model));

    let port = std::env::var("PORT").unwrap_or_else(|_| "8000".into());
    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Antigravity listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
