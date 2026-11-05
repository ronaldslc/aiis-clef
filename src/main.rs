mod batch;
mod checkpoint;
mod cuda;
mod dequant;
mod encode;
mod engine;
mod gpu;
mod loader;
mod model;
mod placement;
mod swizzle;

mod tile_model;
mod types;

use std::sync::Arc;

use ntex::web::{self, App, HttpServer};

use crate::engine::Engine;
use crate::types::Health;

#[derive(Clone)]
pub struct State {
    pub engine: Arc<dyn Engine>,
    pub batcher: batch::Batcher,
}

#[web::get("/health")]
async fn health(state: web::types::State<State>) -> web::types::Json<Health> {
    web::types::Json(state.engine.health())
}

#[web::post("/v1/systemone")]
async fn systemone(
    state: web::types::State<State>,
    body: web::types::Json<types::SystemOneRequest>,
) -> Result<web::types::Json<types::SystemOneResponse>, web::Error> {
    let request = body.into_inner();
    if let Err(message) = request.validate() {
        return Err(web::error::ErrorBadRequest(message).into());
    }

    let started = std::time::Instant::now();
    let model = request.model_name();

    let questions = request.questions.clone();
    let decided = state.batcher.submit(request).await.map_err(|e| {
        if let Some(over) = e.downcast_ref::<engine::Refused>() {
            return web::Error::from(web::error::ErrorBadRequest(over.to_string()));
        }
        web::Error::from(web::error::ErrorInternalServerError(format!("{e:#}")))
    })?;

    let answers = decided
        .probabilities
        .into_iter()
        .filter_map(|(id, probs)| {
            questions
                .iter()
                .find(|(question_id, _)| question_id == &id)
                .map(|(_, q)| (id, types::systemone_answer(q, &probs)))
        })
        .collect();
    Ok(web::types::Json(types::SystemOneResponse {
        model,
        answers,
        usage: types::Usage {
            input_tokens: decided.input_tokens,
            output_tokens: 0,
        },
        latency_s: started.elapsed().as_secs_f64(),
    }))
}

#[derive(Debug, PartialEq, Eq)]
pub enum Cli {
    Serve,
    Help,
    Version,
    Unknown(String),
}

pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Cli {
    let mut it = args.into_iter();
    let _program = it.next();
    match it.next().as_deref() {
        None => Cli::Serve,
        Some("-h") | Some("--help") => Cli::Help,
        Some("-V") | Some("--version") => Cli::Version,
        Some(other) => Cli::Unknown(other.to_string()),
    }
}

fn usage() -> String {
    format!(
        concat!(
        "aiis-clef {version} — the Clef decision service (SystemOne contract).\n",
        "\n",
        "USAGE\n",
        "  aiis-clef                     serve on $PORT (default 8000)\n",
        "  aiis-clef --version           print the version and exit\n",
        "  aiis-clef --help              print this and exit\n",
        "\n",
        "CONFIGURATION — environment variables, none required\n",
        "  CLEF_MODEL_DIR                checkpoint directory       (default /models/clef-flash-nvfp4)\n",
        "  CLEF_HEAD_DIR                 joint head directory       (default /models/clef-flash)\n",
        "  PORT                          listen port                (default 8000)\n",
        "\n",
        "  CLEF_MAX_LENGTH               longest accepted REQUEST   (default {max_len})\n",
        "  CLEF_MAX_PASS_TOKENS          longest PASS — the chunk   (default {pass})\n",
        "  CLEF_KV_CAPACITY              KV cache, in tokens        (default min(MAX_LENGTH, 512xpass))\n",
        "  CLEF_KV_SLOTS                 concurrent cache segments  (default 1)\n",
        "\n",
        "  CLEF_GRAPH                    0 disables CUDA graphs     (default on)\n",
        "  CLEF_GRAPH_SIZES              capture exactly these sizes (default: dense, both modes)\n",
        "  CLEF_MAX_CONCURRENT           requests per batch         (default 4)\n",
        "\n",
        "The cache budget is CLEF_KV_SLOTS x slot_tokens <= CLEF_KV_CAPACITY, allocated once at load.\n",
        "Depth: docs/research/candle-server/32-CHUNKED-PREFILL.md §10.\n",
        ),
        version = env!("CARGO_PKG_VERSION"),
        max_len = crate::engine::max_input_tokens(),
        pass = crate::gpu::max_chunk_size(),
    )
}

fn apply_host_scheduling() {
    if let Ok(v) = std::env::var("CLEF_CPU_NICE") {
        if let Ok(n) = v.parse::<i32>() {
            let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, n) };
            if rc == 0 {
                tracing::info!(nice = n, "CPU priority set");
            } else {
                tracing::warn!(nice = n, err = std::io::Error::last_os_error().to_string(),
                               "CPU priority NOT set (needs CAP_SYS_NICE)");
            }
        }
    }
    if let Ok(spec) = std::env::var("CLEF_CPU_AFFINITY") {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_ZERO(&mut set) };
        let mut any = false;
        for part in spec.split(',') {
            let part = part.trim();
            let (lo, hi) = match part.split_once('-') {
                Some((a, b)) => (a.parse::<usize>().ok(), b.parse::<usize>().ok()),
                None => (part.parse::<usize>().ok(), part.parse::<usize>().ok()),
            };
            if let (Some(lo), Some(hi)) = (lo, hi) {
                for c in lo..=hi.min(lo + 1024) {
                    if c < libc::CPU_SETSIZE as usize {
                        unsafe { libc::CPU_SET(c, &mut set) };
                        any = true;
                    }
                }
            }
        }
        if any {
            let rc = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
            if rc == 0 {
                tracing::info!(cpus = spec.as_str(), "CPU affinity set");
            } else {
                tracing::warn!(cpus = spec.as_str(), err = std::io::Error::last_os_error().to_string(),
                               "CPU affinity NOT set");
            }
        }
    }
}

#[ntex::main]
async fn main() -> std::io::Result<()> {
    apply_host_scheduling();
    match parse_args(std::env::args()) {
        Cli::Serve => {}
        Cli::Help => {
            println!("{}", usage());
            return Ok(());
        }
        Cli::Version => {
            println!("aiis-clef {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Cli::Unknown(arg) => {
            eprintln!("aiis-clef: unknown argument '{arg}'\n");
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let engine = engine::load().map_err(std::io::Error::other)?;
    let state = State {
        batcher: batch::Batcher::spawn(engine.clone()),
        engine,
    };
    let loaded = state.engine.health();
    tracing::info!(
        "aiis-clef on :{port} — model={} backend={} quant={}",
        loaded.model,
        loaded.backend,
        loaded.quant
    );
    HttpServer::new(move || {
        let state = state.clone();

        let body_limit = state.engine.max_tokens().saturating_mul(16) + 1_048_576;
        tracing::info!(body_limit, max_tokens = state.engine.max_tokens(), "json payload limit");
        async move {
            App::new()
                .state(state)

                .state(web::types::JsonConfig::default().limit(body_limit))
                .service((health, systemone))
        }
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::{parse_args, Cli};

    #[test]
    fn help_and_version_answer_instead_of_serving() {
        let v = |a: &[&str]| parse_args(a.iter().map(|s| s.to_string()));
        assert_eq!(v(&["aiis-clef"]), Cli::Serve);
        assert_eq!(v(&["aiis-clef", "--help"]), Cli::Help);
        assert_eq!(v(&["aiis-clef", "-h"]), Cli::Help);
        assert_eq!(v(&["aiis-clef", "--version"]), Cli::Version);
        assert_eq!(v(&["aiis-clef", "-V"]), Cli::Version);
        assert_eq!(v(&["aiis-clef", "--nope"]), Cli::Unknown("--nope".into()));
    }
}
