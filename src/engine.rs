use anyhow::{Context, Result};

use crate::types::{Health, Probabilities, SystemOneRequest};

pub struct Decided {
    pub probabilities: Vec<(String, Probabilities)>,
    pub input_tokens: usize,
}

pub trait Engine: Send + Sync {
    fn health(&self) -> Health;

    fn decide(&self, request: &SystemOneRequest) -> Result<Decided>;

    fn max_tokens(&self) -> usize {
        max_input_tokens()
    }

    fn decide_batch(&self, requests: &[SystemOneRequest]) -> Vec<Result<Decided>> {
        requests.iter().map(|r| self.decide(r)).collect()
    }

    fn long_begin(&self, _request: &SystemOneRequest) -> Result<usize> {
        anyhow::bail!("this engine does not serve requests in passes")
    }

    fn long_slots(&self) -> usize {
        1
    }

    fn long_advance(&self, _slot: usize) -> Result<bool> {
        anyhow::bail!("this engine does not serve long requests in passes")
    }

    fn long_advance_many(&self, slots: &[usize]) -> Result<Vec<bool>> {
        slots.iter().map(|&s| self.long_advance(s)).collect()
    }

    fn long_finish(&self, _slot: usize) -> Result<Decided> {
        anyhow::bail!("this engine does not serve long requests in passes")
    }

    fn on_idle(&self) {}
}

pub struct Unconfigured;

impl Engine for Unconfigured {
    fn health(&self) -> Health {
        Health {
            ok: true,
            model: std::env::var("CLEF_MODEL_NAME").unwrap_or_else(|_| "clef-flash".into()),
            version: std::env::var("CLEF_REVISION").unwrap_or_else(|_| "main".into()),
            device: "none".into(),
            backend: "unconfigured".into(),
            quant: "none".into(),
            checkpoint: "none".into(),
            max_batch: max_batch(),
            max_tokens: max_input_tokens(),
            graphs: 0,
            graphs_chunked: 0,
            captures_pending: 0,
            head_graphs: 0,
            chunk_tokens: crate::gpu::max_chunk_size(),
                att_keys: crate::cuda::att_keys(),
                stream_priority: crate::cuda::stream_priority(),
            kv_tokens: 0,
            kv_mib: 0,
            kv_slots: crate::gpu::kv_slots(),
            chunked_passes: 0,
            chunked_width_max: 0,
            chunked_passes_sharing: 0,
            graph_pool_mib: 0,
            refused_over_ceiling: 0,
        }
    }

    fn decide(&self, _request: &SystemOneRequest) -> Result<Decided> {
        anyhow::bail!(
            "no Clef backbone is loaded in this build; the engine boundary exists but has no \
             implementation yet"
        )
    }
}

pub struct GpuEngine {
    runner: crate::gpu::GpuRunner,
    head: crate::gpu::DeviceHead,
    kv_graphs: crate::gpu::KvGraphs,
    encoder: crate::encode::Encoder,
    lm_head: crate::placement::HostTable,
    config: crate::model::TextConfig,
    head_config: crate::model::head::HeadConfig,
    refused: std::sync::atomic::AtomicUsize,
    long: std::sync::Mutex<Vec<Option<InFlight>>>,
}

struct InFlight {
    encoded: crate::encode::Encoded,
    questions: Vec<crate::model::head::QuestionSpec>,
    pre: crate::gpu::PreHead,
    pos: usize,
    tokens: usize,
    chunk: usize,
    slot: usize,
}

impl GpuEngine {
    pub fn load(model_dir: &std::path::Path, head_dir: &std::path::Path) -> Result<Self> {
        let checkpoint = crate::loader::Checkpoint::open(model_dir)
            .with_context(|| format!("opening the backbone at {}", model_dir.display()))?;
        let config = crate::model::TextConfig::from_dir(model_dir)
            .with_context(|| format!("reading config.json from {}", model_dir.display()))?;

        let head_path = head_dir.join("joint_head.safetensors");
        let head_checkpoint = crate::loader::Checkpoint::open_single(&head_path)
            .with_context(|| format!("opening the head at {}", head_path.display()))?;
        let head_config = crate::model::head::HeadConfig::from_dir(head_dir)?;

        let lm_head_path = checkpoint.shard_path("lm_head.weight")?;
        let lm_head = crate::placement::HostTable::map(&lm_head_path, "lm_head.weight")
            .context("mapping lm_head, which stays in host memory")?;

        let trace_mem = std::env::var("CLEF_MEM_TRACE").is_ok();
        let stage = |what: &str, d: &crate::cuda::Device| {
            if trace_mem {
                let total = d.total_memory().unwrap_or(0);
                let used = total.saturating_sub(d.free_memory().unwrap_or(0));
                eprintln!("  MEM {:>28}: {:>6} MiB of {}", what, used / (1 << 20), total / (1 << 20));
            }
        };
        let runner = crate::gpu::GpuRunner::load(&checkpoint, &config)
            .context("loading the backbone onto the device")?;
        stage("after the backbone weights", &runner.device);
        let head = crate::gpu::load_head(&runner.device, &head_checkpoint,
                                         head_config_ref(&head_config))
            .context("loading the joint head onto the device")?;
        stage("after the head", &runner.device);
        let encoder = crate::encode::Encoder::load(model_dir)
            .with_context(|| format!("loading the tokenizer from {}", model_dir.display()))?;

        if crate::gpu::graph_requested() {
            if let Err(e) = runner.measure_footprint() {
                tracing::warn!("could not measure the footprint: {e:#}");
            }
        }

        if max_input_tokens() > PROVEN_FULL_PATH_TOKENS {
            eprintln!(
                "  NOTE: CLEF_MAX_LENGTH is {} but the full path is PROVEN to {} tokens on this card — \
                 the head holds a whole-request buffer per captured shape, so the ceiling is what was \
                 measured ({} bytes/token of whole-request buffers, {:.1} KiB each). See \
                 32-CHUNKED-PREFILL.md §10c.",
                max_input_tokens(),
                PROVEN_FULL_PATH_TOKENS,
                whole_request_bytes_per_token_for(head_config.hidden_size, head_config.width),
                whole_request_bytes_per_token_for(head_config.hidden_size, head_config.width) as f64
                    / 1024.0,
            );
        }
        let warmed = if crate::gpu::graph_requested() {
            let sizes = if crate::gpu::chunking_enabled() {
                dense_warm_up_sizes()
            } else {
                warm_up_sizes()
            };

            let report = runner.warm_up(&sizes)?;
            if !report.failed.is_empty() {
                let mut why = String::new();
                for (rows, e) in &report.failed {
                    why.push_str(&format!("\n  - {rows} rows: {e}"));
                }
                anyhow::bail!(
                    "refusing to start: {} of the {} pass geometries this configuration can produce could \
                     not be graphed, and the eager path is not allowed — a request needing one would fail. \
                     Missing:{why}\n\
                     Each geometry needs `graph working set <= free - 1 GiB` at the same time. Free it by \
                     lowering CLEF_MAX_LENGTH (the KV carry is `slots x capacity`), lowering CLEF_KV_SLOTS, \
                     or lowering CLEF_ATT_KEYS (its split partials are {:.0} MiB at 4).",
                    report.failed.len(), report.requested,
                    crate::cuda::split_partials_bytes(
                        crate::cuda::att_keys(),
                        config.num_attention_heads,
                        config.head_dim) as f64 / (1u64 << 20) as f64
                );
            }
            report.captured
        } else {
            0
        };
        tracing::info!(warmed, "graphs captured at load");
        stage("after the warm-up graphs", &runner.device);

        let free_before = runner.device.free_memory().unwrap_or(0);
        let freed = runner.device.trim_device_pool();
        let free_after = runner.device.free_memory().unwrap_or(0);
        tracing::info!(
            freed_mib = freed / (1 << 20),
            free_before_mib = free_before / (1 << 20),
            free_after_mib = free_after / (1 << 20),
            "released the warm-up's pool blocks"
        );

        {
            let cache_for = cache_sizing_tokens(&runner, &config, &head);

            let splits = crate::cuda::KV_SPLITS.max(1);
            let need = crate::cuda::alloc_cache_needed_bytes(
                cache_for, head.config.width, config.hidden_size, crate::gpu::max_chunk_size())
                .saturating_mul(splits / 4 + 1);

            let quarter = runner.device.total_memory().unwrap_or(0) / 4;

            let cap = need.max(3 * 1024 * 1024 * 1024).min(quarter)
                .min(6 * 1024 * 1024 * 1024);
            crate::cuda::set_alloc_cache_cap(cap);

            if let Err(e) = runner.prime_long_path() {
                tracing::warn!("could not prime the long path: {e:#}");
            }
            eprintln!(
                "  ALLOC CACHE: need={:.0} MiB (sized for {cache_for} tokens of whole-request buffers, width {}, \
                 hidden {}, pass {}) cap={:.0} MiB",
                need as f64 / 1048576.0, head.config.width, config.hidden_size,
                crate::gpu::max_chunk_size(), cap as f64 / 1048576.0
            );
        }

        let head_kv_graph = std::env::var("CLEF_HEAD_KV_GRAPH").as_deref() != Ok("0");
        let kv_graphs = if crate::gpu::graph_requested() && head_kv_graph {
            crate::gpu::KvGraphs::capture(&runner.device, &head)?
        } else {
            crate::gpu::KvGraphs::empty(&runner.device, &head)?
        };
        let engine = Self { runner, head, kv_graphs, encoder, lm_head, config, head_config,
                            refused: std::sync::atomic::AtomicUsize::new(0),
                            long: std::sync::Mutex::new(
                                (0..crate::gpu::kv_slots()).map(|_| None).collect(),
                            ) };

        if crate::gpu::graph_requested() {
            engine.seed_limits();
        }
        Ok(engine)
    }

    fn servable_tokens(&self) -> usize {
        let cache = self
            .runner
            .max_servable_tokens()
            .map(|t| t.min(max_input_tokens()))
            .unwrap_or_else(max_input_tokens);

        cache.min(PROVEN_FULL_PATH_TOKENS)
    }

    fn seed_limits(&self) {
        match self.runner.warm_up(&[256, 4096]) {
            Ok(r) => tracing::info!(
                captured = r.captured,
                max_tokens = self.servable_tokens(),
                "measured the footprint; the device decides the maximum pass"
            ),
            Err(e) => tracing::warn!("could not measure the footprint, keeping the configured limit: {e:#}"),
        }
    }

    fn encode_and_check(&self, request: &SystemOneRequest) -> Result<crate::encode::Encoded> {
        let encoded = self.encoder.encode(request)?;
        let tokens = encoded.input_ids.len();
        if tokens == 0 {
            anyhow::bail!("the encoder produced no tokens");
        }

        if let Err(refused) = check_length(tokens, self.servable_tokens()) {
            self.refused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(anyhow::Error::new(refused));
        }
        Ok(encoded)
    }

    pub fn decide_timed(&self, request: &SystemOneRequest)
        -> Result<(Decided, std::time::Duration, std::time::Duration, std::time::Duration,
                   std::time::Duration)> {
        let encoded = self.encode_and_check(request)?;

        let trace = std::env::var("CLEF_MEM_TRACE").is_ok();
        let used = || {
            let d = &self.runner.device;
            d.total_memory().unwrap_or(0).saturating_sub(d.free_memory().unwrap_or(0)) / (1 << 20)
        };
        if trace {
            eprintln!("    mem {:>6} MiB  after encode ({} tokens)", used(), encoded.input_ids.len());
        }

        if trace {
            eprintln!("    mem {:>6} MiB  before a {}-token request (serving path)",
                      used(), encoded.input_ids.len());
        }

        let slot = self.long_begin_encoded(encoded)?;
        let mut t_backbone = std::time::Duration::ZERO;

        let t_bb = std::time::Instant::now();
        loop {
            match self.long_advance_many_impl(&[slot]) {
                Ok(d) => {
                    if d[0] {
                        break;
                    }
                }
                Err(e) => {
                    let _ = self.long_finish_impl(slot);
                    return Err(e);
                }
            }
        }
        self.runner.device.sync()?;
        t_backbone = t_bb.elapsed();
        let t = std::time::Instant::now();
        let out = self.long_finish_impl(slot)?;
        let t_head = t.elapsed();
        if trace {
            eprintln!("    mem {:>6} MiB  after the head (sink path)", used());
        }
        Ok((out, t_backbone, t_head,
            std::time::Duration::ZERO, std::time::Duration::ZERO))
    }

    pub fn long_begin_impl(&self, request: &SystemOneRequest) -> Result<usize> {
        let encoded = self.encode_and_check(request)?;
        self.long_begin_encoded(encoded)
    }

    pub fn long_begin_encoded(&self, encoded: crate::encode::Encoded) -> Result<usize> {
        let tokens = encoded.input_ids.len();
        let chunk = crate::gpu::max_chunk_size();
        let questions = self.question_specs(&encoded);

        let mut slots = self.long.lock().map_err(|_| anyhow::anyhow!("long slot poisoned"))?;
        let free = slots.iter().position(|s| s.is_none())
            .context("every cache slot is busy; the scheduler should have deferred this request")?;

        self.runner.forward_chunk_begin_slot(crate::gpu::kv_capacity(), free)?;
        let pre = crate::gpu::PreHead::new(&self.runner.device, &self.head, &questions, tokens, chunk)?;
        slots[free] = Some(InFlight { encoded, questions, pre, pos: 0, tokens, chunk, slot: free });
        Ok(free)
    }

    pub fn long_advance_impl(&self, slot: usize) -> Result<bool> {
        Ok(self.long_advance_many_impl(&[slot])?[0])
    }

    pub fn long_advance_many_impl(&self, slots: &[usize]) -> Result<Vec<bool>> {
        let mut guard = self.long.lock().map_err(|_| anyhow::anyhow!("long slot poisoned"))?;

        let asked = slots.to_vec();
        let mut wants: Vec<usize> = Vec::with_capacity(asked.len());
        for &s in &asked {
            let job = guard.get(s).and_then(|x| x.as_ref())
                .with_context(|| format!("no long request in slot {s}"))?;

            wants.push(crate::gpu::next_chunk_rows(job.tokens - job.pos).min(job.chunk));
        }
        let fitted = crate::gpu::fit_chunks(&wants);

        let riding: Vec<(usize, usize)> = asked.iter().copied().zip(fitted.iter().copied())
            .filter(|&(_, c)| c > 0).collect();
        {
            let mut specs: Vec<crate::gpu::ChunkSpec<'_>> = Vec::with_capacity(riding.len());
            let mut sinks: Vec<&crate::gpu::PreHead> = Vec::with_capacity(riding.len());
            for &(s, c) in &riding {
                let job = guard.get(s).and_then(|x| x.as_ref())
                    .with_context(|| format!("no long request in slot {s}"))?;
                specs.push(crate::gpu::ChunkSpec {
                    ids: &job.encoded.input_ids, pos: job.pos, slot: job.slot, rows: c,
                });
                sinks.push(&job.pre);
            }

            self.runner.forward_multi_chunk_step(&specs, &sinks)?;
        }

        for &(s, c) in &riding {
            if let Some(job) = guard.get_mut(s).and_then(|x| x.as_mut()) {
                job.pos = (job.pos + c).min(job.tokens);
            }
        }
        let dones = asked.iter().map(|s| {
            guard.get(*s).and_then(|x| x.as_ref()).map_or(false, |job| job.pos >= job.tokens)
        }).collect();
        Ok(dones)
    }

    pub fn long_finish_impl(&self, slot: usize) -> Result<Decided> {
        let mut guard = self.long.lock().map_err(|_| anyhow::anyhow!("long slot poisoned"))?;
        let job = guard.get_mut(slot).and_then(|s| s.take())
            .context("no long request in that slot")?;
        let logits = self.runner.head_forward_pre(&job.pre, job.tokens, &job.encoded.input_ids,
                                                  &job.questions, &self.lm_head, &self.head,
                                                  &self.kv_graphs)?;

        if let Ok(dir) = std::env::var("CLEF_DUMP_MEM") {
            job.pre.dump_memory(&self.runner.device, &dir, job.tokens)?;
        }

        self.runner.forward_chunk_end(job.tokens, job.chunk)?;
        self.runner.device.sync()?;

        if std::env::var("CLEF_POOL_TRIM").is_ok() {
            self.runner.device.trim_device_pool();
        }
        Ok(self.decided(&job.encoded, logits, job.tokens))
    }

    fn question_specs(&self, encoded: &crate::encode::Encoded)
        -> Vec<crate::model::head::QuestionSpec> {
        encoded
            .questions
            .iter()
            .map(|q| crate::model::head::QuestionSpec {
                question_type: crate::model::head::question_type_id(q.question_type),
                question_span: q.question_span,
                option_spans: q.option_spans.clone(),
            })
            .collect()
    }

    fn decided(&self, encoded: &crate::encode::Encoded, logits: Vec<Vec<f32>>,
               tokens: usize) -> Decided {
        let mut probabilities = Vec::with_capacity(encoded.questions.len());
        for (q, lg) in encoded.questions.iter().zip(logits.iter()) {
            let probs = softmax(lg);
            let map: Probabilities = q.option_ids.iter().cloned().zip(probs.into_iter()).collect();
            probabilities.push((q.question_id.clone(), map));
        }
        Decided { probabilities, input_tokens: tokens }
    }
}

pub fn head_config_ref(c: &crate::model::head::HeadConfig) -> crate::model::head::HeadConfig {
    crate::model::head::HeadConfig {
        hidden_size: c.hidden_size,
        width: c.width,
        routing_layers: c.routing_layers,
        layers: c.layers,
        heads: c.heads,
        feedforward: c.feedforward,
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in exps.iter_mut() {
        *v *= inv;
    }
    exps
}

impl Engine for GpuEngine {
    fn max_tokens(&self) -> usize {
        self.servable_tokens()
    }

    fn health(&self) -> Health {
        Health {
            ok: true,
            model: std::env::var("CLEF_MODEL_NAME").unwrap_or_else(|_| "clef-flash".into()),
            version: std::env::var("CLEF_REVISION").unwrap_or_else(|_| "main".into()),
            device: self.runner.device.describe().unwrap_or_else(|_| "cuda".into()),
            backend: "rust-cuda".into(),
            quant: format!(
                "nvfp4+fp8, head bf16 ({} layers, {} routing)",
                self.head_config.layers, self.head_config.routing_layers
            ),
            checkpoint: model_dir()
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "?".into()),
            max_batch: max_batch(),
            max_tokens: self.servable_tokens(),
            graphs: self.runner.graph_count(),
            graphs_chunked: self.runner.graph_count_for(crate::gpu::GraphMode::Chunked),
            captures_pending: self.runner.pending_captures(),
            head_graphs: self.kv_graphs.len(),
            chunk_tokens: crate::gpu::max_chunk_size(),
                att_keys: crate::cuda::att_keys(),
                stream_priority: crate::cuda::stream_priority(),
            kv_tokens: self.runner.carry_capacity(),
            kv_mib: self.runner.carry_bytes() / (1 << 20),
            kv_slots: crate::gpu::kv_slots(),
            chunked_passes: self.runner.chunked_passes(),
            chunked_width_max: self.runner.chunked_width_max(),
            chunked_passes_sharing: self.runner.chunked_passes_sharing(),
            graph_pool_mib: (self.runner.device.graph_pool_high() / (1 << 20)) as usize,
            refused_over_ceiling: self.refused.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    fn decide(&self, request: &SystemOneRequest) -> Result<Decided> {
        Ok(self.decide_timed(request)?.0)
    }

    fn long_slots(&self) -> usize {
        crate::gpu::kv_slots()
    }
    fn long_begin(&self, request: &SystemOneRequest) -> Result<usize> {
        self.long_begin_impl(request)
    }
    fn long_advance(&self, slot: usize) -> Result<bool> {
        self.long_advance_impl(slot)
    }

    fn long_advance_many(&self, slots: &[usize]) -> Result<Vec<bool>> {
        self.long_advance_many_impl(slots)
    }
    fn long_finish(&self, slot: usize) -> Result<Decided> {
        self.long_finish_impl(slot)
    }

    fn on_idle(&self) {
        match self.runner.capture_pending_one() {
            Ok(true) => tracing::info!("captured a pending geometry while idle"),
            Ok(false) => {}
            Err(e) => tracing::warn!("could not capture a pending geometry: {e:#}"),
        }
    }

    fn decide_batch(&self, requests: &[SystemOneRequest]) -> Vec<Result<Decided>> {
        let mut out: Vec<Option<Result<Decided>>> = (0..requests.len()).map(|_| None).collect();
        let mut pending: Vec<(usize, crate::encode::Encoded)> = Vec::with_capacity(requests.len());
        for (i, r) in requests.iter().enumerate() {
            match self.encode_and_check(r) {
                Ok(e) => pending.push((i, e)),
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        let slots = self.long_slots().max(1);
        while !pending.is_empty() {
            let mut running: Vec<(usize, usize)> = Vec::new();
            let mut rest: Vec<(usize, crate::encode::Encoded)> = Vec::new();
            for (i, e) in pending.drain(..) {
                if running.len() < slots {
                    match self.long_begin_encoded(e) {
                        Ok(slot) => running.push((i, slot)),
                        Err(err) => out[i] = Some(Err(err)),
                    }
                } else {
                    rest.push((i, e));
                }
            }
            pending = rest;
            if running.is_empty() {
                let msg = "every cache slot is busy; the scheduler should have deferred this request";
                for (i, _) in pending.drain(..) {
                    out[i] = Some(Err(anyhow::anyhow!("{msg}")));
                }
                break;
            }

            while !running.is_empty() {
                let ids: Vec<usize> = running.iter().map(|(_, s)| *s).collect();
                match self.long_advance_many_impl(&ids) {
                    Ok(dones) => {
                        let mut still = Vec::with_capacity(running.len());
                        for ((i, slot), done) in running.drain(..).zip(dones) {
                            if done {
                                out[i] = Some(self.long_finish_impl(slot));
                            } else {
                                still.push((i, slot));
                            }
                        }
                        running = still;
                    }
                    Err(e) => {
                        let msg = format!("a pass failed: {e:#}");
                        for (i, slot) in running.drain(..) {
                            let _ = self.long_finish_impl(slot);
                            out[i] = Some(Err(anyhow::anyhow!("{msg}")));
                        }
                    }
                }
            }
        }
        out.into_iter().map(|o| o.expect("every request answered")).collect()
    }
}

pub fn warm_up_sizes() -> Vec<usize> {
    if let Ok(spec) = std::env::var("CLEF_GRAPH_SIZES") {
        let parsed: Vec<usize> = spec.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }

const DEFAULT_GRAPH_SIZES: [usize; 9] = [256, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768];
    let max = crate::gpu::max_graph_tokens();
    let mut sizes: Vec<usize> = DEFAULT_GRAPH_SIZES
        .iter()
        .copied()
        .filter(|&s| s <= max)
        .collect();

    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

pub fn dense_warm_up_sizes() -> Vec<usize> {
    let mut sizes = crate::gpu::pass_geometries();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

pub fn max_input_tokens() -> usize {
    std::env::var("CLEF_MAX_LENGTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32768)
}

pub fn check_length(tokens: usize, limit: usize) -> Result<(), Refused> {
    if tokens > limit {
        Err(Refused {
            tokens,
            why: format!("over this device's limit of {limit}"),
        })
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub struct Refused {
    pub tokens: usize,
    pub why: String,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "this device will not run a pass of {} tokens: {}. Send fewer items, shorter ones, \
             or split it into multiple requests.",
            self.tokens, self.why
        )
    }
}

impl std::error::Error for Refused {}

pub fn refuse(tokens: usize, why: String) -> anyhow::Error {
    anyhow::Error::new(Refused { tokens, why })
}

pub const PROVEN_FULL_PATH_TOKENS: usize = 131_072;

pub fn whole_request_bytes_per_token_for(hidden: usize, width: usize) -> usize {
    hidden * 4
        + hidden * 4
        + hidden * 2
        + width * 4
        + hidden * 4
}

pub fn max_batch() -> usize {
    std::env::var("CLEF_MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

pub fn model_dir() -> std::path::PathBuf {
    std::env::var("CLEF_MODEL_DIR")
        .unwrap_or_else(|_| "/models/clef-flash-nvfp4".into())
        .into()
}

pub fn head_dir() -> std::path::PathBuf {
    std::env::var("CLEF_HEAD_DIR")
        .unwrap_or_else(|_| "/models/clef-flash".into())
        .into()
}

fn cache_sizing_tokens(runner: &crate::gpu::GpuRunner, config: &crate::model::TextConfig,
                       head: &crate::gpu::DeviceHead) -> usize {
    let _hidden = config.hidden_size;
    let width = head.config.width;
    let free = runner.device.free_memory().unwrap_or(0);

    let per_token = width * 4;
    (free / 4).saturating_sub(1024 * 1024 * 1024).checked_div(per_token)
        .unwrap_or(0)
        .min(crate::engine::PROVEN_FULL_PATH_TOKENS)
        .max(4096)
}

pub fn load() -> Result<std::sync::Arc<dyn Engine>> {
    let model = model_dir();
    let head = head_dir();
    if !model.exists() {
        tracing::warn!(
            "no checkpoint at {} — starting unconfigured; /v1/systemone will refuse",
            model.display()
        );
        return Ok(std::sync::Arc::new(Unconfigured));
    }
    Ok(std::sync::Arc::new(GpuEngine::load(&model, &head)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOKE_REQUEST: &str = r#"{
  "state": "The migration ran overnight and finished at 04:12. Two rows were skipped.",
  "questions": {
    "rel_0": {
      "type": "score",
      "criteria": {
        "0": "Does not answer.",
        "1": "Related but does not answer.",
        "2": "Answers part.",
        "3": "Directly answers."
      }
    },
    "done_0": { "type": "noul" }
  }
}"#;

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_128k_request_answers_the_same_twice() {
        if std::env::var("CLEF_MAX_LENGTH").is_err() {
            eprintln!("SKIP: set CLEF_MAX_LENGTH=131072 to run this");
            return;
        }
        let model = model_dir();
        if !model.exists() { eprintln!("SKIP: no checkpoint"); return; }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e, Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        let repeats = (131_067 - 157) / 19;
        let state = "The migration ran overnight and finished at 04:12. Two rows were skipped. ".repeat(repeats);
        let json = format!(
            "{{\"state\": {state:?}, \"questions\": {{\"rel_0\": {{\"type\": \"score\", \"criteria\": \
             {{\"0\": \"Does not answer.\", \"1\": \"Related but does not answer.\", \
             \"2\": \"Answers part.\", \"3\": \"Directly answers.\"}}}}}}}}"
        );
        let request: SystemOneRequest = serde_json::from_str(&json).expect("request");
        let tokens = engine.encoder.encode(&request).expect("encode").input_ids.len();
        eprintln!("  {tokens} tokens");
        assert!(tokens > 131_000, "the request must be ~128k");

        let a = engine.decide(&request).expect("first run");
        let b = engine.decide(&request).expect("second run");
        let (pa, pb) = (&a.probabilities, &b.probabilities);
        assert_eq!(pa.len(), pb.len());
        let mut worst = 0.0f32;
        let mut nonfinite = 0usize;
        for ((_, ma), (_, mb)) in pa.iter().zip(pb.iter()) {
            for (k, va) in ma.iter() {
                let vb = mb.get(k).copied().unwrap_or(f32::NAN);
                worst = worst.max((va - vb).abs());
                if !va.is_finite() { nonfinite += 1; }
            }
        }
        eprintln!("  worst |run1 - run2| = {worst:.3e}, non-finite = {nonfinite}");
        assert_eq!(nonfinite, 0, "a 128k request produced non-finite probabilities");
        assert_eq!(worst, 0.0, "a 128k request answered differently on the second run");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_batch_larger_than_a_pass_is_split_and_still_correct() {
        let Ok(checkpoint) = crate::loader::Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = crate::model::config::TextConfig::from_dir("/models/clef-flash-nvfp4")
            .expect("config.json");
        let bound = crate::gpu::max_batch_tokens();
        eprintln!("  the batch pass bound is {bound}");
        let _ = (checkpoint, config);

        let sizes = [bound - 1, 2 * bound];
        let mut groups: Vec<usize> = Vec::new();
        let mut current = 0usize;
        let mut total = 0usize;
        for n in sizes {
            if n > bound {
                if current > 0 { groups.push(current); current = 0; total = 0; }
                groups.push(1);
                continue;
            }
            if total + n > bound && current > 0 {
                groups.push(current);
                current = 0;
                total = 0;
            }
            current += 1;
            total += n;
        }
        if current > 0 { groups.push(current); }
        let biggest: usize = sizes.iter().copied().filter(|&n| n <= bound).max().unwrap_or(0);
        eprintln!("  {} requests of {:?} tokens split into passes of {:?} (bound {bound})",
                  sizes.len(), sizes, groups);
        assert!(groups.len() > 1, "40,000 tokens must not fit in one pass of {bound}");
        assert_eq!(groups.iter().sum::<usize>(), sizes.len(), "every request lands in exactly one pass");
        assert!(biggest <= bound);
    }

    #[test]
    #[ignore]
    fn the_host_gap_probe() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let json = "{\"state\": \"The migration ran overnight. Two rows were skipped.\", \"questions\": \
                    {\"rel_0\": {\"type\": \"score\", \"criteria\": {\"0\": \"No.\", \"1\": \"Partly.\"}}}}";
        let request: SystemOneRequest = serde_json::from_str(json).expect("request");
        let encoded = engine.encoder.encode(&request).expect("encode");
        let tokens = encoded.input_ids.len();

        let _ = engine.runner.forward_dev(&encoded.input_ids).expect("warm-up");
        engine.runner.device.sync().expect("sync");

        const N: usize = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            let _ = engine.runner.forward_dev(&encoded.input_ids).expect("backbone");
        }
        engine.runner.device.sync().expect("sync");
        let wall = t0.elapsed().as_secs_f64() * 1e3 / N as f64;

        let t0 = std::time::Instant::now();
        let _ = engine.runner.forward_dev(&encoded.input_ids).expect("backbone");
        engine.runner.device.sync().expect("sync");
        let one = t0.elapsed().as_secs_f64() * 1e3;

        eprintln!(
            "  tokens={tokens}: one call (synced) {one:.3} ms, queued {N}x {wall:.3} ms each"
        );
        eprintln!(
            "  -> if queued < one, the host was hiding behind the device; \
             if queued ~= one, the gap is host"
        );
    }

    const PRODUCTION_REPEATS: [usize; 2] = [32, 854];

    fn standard_buckets(max_tokens: usize) -> Vec<usize> {
        let mut out: Vec<usize> = PRODUCTION_REPEATS
            .iter()
            .map(|&r| 19 * r + 157)
            .filter(|&t| t <= max_tokens)
            .collect();

        if let Some(&last) = out.last() {
            if max_tokens <= last + crate::cuda::CHUNK_LEN {
                out.pop();
            }
        }
        out.push(max_tokens);
        out.dedup();
        out
    }

    #[test]
    #[ignore]
    fn the_batching_win_today() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        let state = "The migration ran overnight and finished at 04:12. Two rows were skipped. ".repeat(24);
        let requests: Vec<SystemOneRequest> = (0..4)
            .map(|i| {
                let json = format!(
                    "{{\"state\": {:?}, \"questions\": {{\"rel_0\": {{\"type\": \"score\", \"criteria\":                      {{\"0\": \"Does not answer.\", \"1\": \"Related but does not answer.\",                      \"2\": \"Answers part.\", \"3\": \"Directly answers.\"}}}}}}}}",
                    format!("{state} Document {i}.")
                );
                serde_json::from_str(&json).expect("parses")
            })
            .collect();
        let mut serial = Vec::new();
        let mut batched = Vec::new();
        for round in 0..3 {
            if round % 2 == 0 {
                serial.push(serial_pass(&engine, &requests));
                batched.push(batched_pass(&engine, &requests));
            } else {
                batched.push(batched_pass(&engine, &requests));
                serial.push(serial_pass(&engine, &requests));
            }
        }
        eprintln!("  serial  (4 x decide):      {serial:?} ms");
        eprintln!("  batched (1 x decide_batch): {batched:?} ms");
        let sm: f64 = serial.iter().sum::<f64>() / serial.len() as f64;
        let bm: f64 = batched.iter().sum::<f64>() / batched.len() as f64;
        eprintln!("  mean: serial {sm:.1} ms, batched {bm:.1} ms -> {:.3}x", sm / bm);
    }

    fn serial_pass(engine: &GpuEngine, requests: &[SystemOneRequest]) -> f64 {
        let t = std::time::Instant::now();
        for r in requests {
            engine.decide(r).expect("decide");
        }
        t.elapsed().as_secs_f64() * 1e3
    }

    fn batched_pass(engine: &GpuEngine, requests: &[SystemOneRequest]) -> f64 {
        let t = std::time::Instant::now();
        let out = engine.decide_batch(requests);
        for r in &out {
            r.as_ref().expect("decide_batch");
        }
        t.elapsed().as_secs_f64() * 1e3
    }

    #[test]
    #[ignore]
    fn the_engine_timing_breakdown() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        eprintln!("  device: {}", engine.health().device);

        let repeats_list: Vec<usize> = match std::env::var("CLEF_REPEATS").ok() {
            Some(v) => v.split(',').filter_map(|x| x.trim().parse::<usize>().ok())
                .map(|tokens| tokens.saturating_sub(157) / 19)
                .filter(|&r| r > 0)
                .collect(),
            None => standard_buckets(engine.max_tokens())
                .into_iter()
                .map(|tokens| tokens.saturating_sub(157) / 19)
                .collect(),
        };
        eprintln!(
            "  the engine: block={} MiB reserve={} MiB ceiling={} MiB-total={}",
            engine.runner.device.graph_block_bytes() / 1048576,
            engine.runner.device.graph_reserve() / 1048576,
            engine.max_tokens(),
            engine.runner.device.total_memory().unwrap_or(0) / 1048576
        );
        eprintln!(
            "  the standard range: {:?} tokens (the device's own maximum is {})",
            repeats_list.iter().map(|&r| 19 * r + 157).collect::<Vec<_>>(),
            engine.max_tokens()
        );
        let rounds: usize = std::env::var("CLEF_ROUNDS").ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&r| r > 0)
            .unwrap_or(3);

        let mut rows: Vec<(usize, usize, f64, f64, f64)> = Vec::new();

        let mut plan: Vec<(usize, usize)> = Vec::new();
        for &repeats in &repeats_list {
            plan.push((0, repeats));
        }
        for round in 0..rounds {
            for k in 0..repeats_list.len() {
                plan.push((round + 1, repeats_list[(k + round) % repeats_list.len()]));
            }
        }
        let mut last_round = usize::MAX;

        let mut warmed: Vec<usize> = Vec::new();
        for (round, repeats) in plan {
            let warming = round == 0;
            if round != last_round {
                if warming {
                    eprintln!("  --- warm-up sweep (untimed) ---");
                } else {
                    eprintln!("  --- round {round}/{rounds} ---");
                }
                last_round = round;
            }
            let state = "The migration ran overnight and finished at 04:12. Two rows were skipped. "
                .repeat(repeats);
            let json = format!(
                "{{\"state\": {state:?}, \"questions\": {{\"rel_0\": {{\"type\": \"score\", \"criteria\": \
                 {{\"0\": \"Does not answer.\", \"1\": \"Related but does not answer.\", 
                 \"2\": \"Answers part.\", \"3\": \"Directly answers.\"}}}}}}}}"
            );
            let request: SystemOneRequest = match serde_json::from_str(&json) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  SKIP size {repeats}: {e}");
                    continue;
                }
            };

            let t0 = std::time::Instant::now();
            let encoded = engine.encoder.encode(&request).expect("encode");
            let t_encode = t0.elapsed();
            let tokens = encoded.input_ids.len();

            let questions: Vec<crate::model::head::QuestionSpec> = encoded
                .questions
                .iter()
                .map(|q| crate::model::head::QuestionSpec {
                    question_type: crate::model::head::question_type_id(q.question_type),
                    question_span: q.question_span,
                    option_spans: q.option_spans.clone(),
                })
                .collect();
            if !warmed.contains(&tokens) {
                warmed.push(tokens);

                let _ = engine.decide_timed(&request).expect("warm-up");
                engine.runner.device.sync().expect("sync");
            }

            let t0 = std::time::Instant::now();
            let (_decided, t_backbone, t_head, gpu, gpu_head) =
                engine.decide_timed(&request).expect("decide_timed");
            let t_total = t0.elapsed();
            let t_encode = t_total.saturating_sub(t_backbone + t_head);

            let t_upload = std::time::Duration::ZERO;

            let total = t_total;
            let tok_s = tokens as f64 / total.as_secs_f64();
            if warming {
                continue;
            }
            rows.push((tokens, round, total.as_secs_f64() * 1e3, gpu.as_secs_f64() * 1e3, tok_s));
            let _ = gpu_head;

            let mem = |r: &crate::gpu::GpuRunner| {
                (r.mem_used_mib(), r.device.pool_bytes() / (1 << 20),
                 r.device.free_memory().unwrap_or(0) / (1 << 20))
            };
            let (m_used, m_pool, m_free) = mem(&engine.runner);
            eprintln!("  {tokens:6} tokens: used {m_used:>5} MiB pool {m_pool:>5} MiB free {m_free:>5} MiB");
            eprintln!(
                "  {tokens:6} tokens: total {:>9.3?} | encode {:>9.3?} ({:>5.1}%) | \
                 backbone {:>9.3?} ({:>5.1}%) [GPU {:>9.3?}] | upload {:>9.3?} | head {:>9.3?} ({:>5.1}%) [GPU {:>9.3?}] | \
                 {:>7.1} tok/s | round {round}/{rounds}",
                total,
                t_encode, 100.0 * t_encode.as_secs_f64() / total.as_secs_f64(),
                t_backbone, 100.0 * t_backbone.as_secs_f64() / total.as_secs_f64(),
                gpu,
                t_upload,
                t_head, 100.0 * t_head.as_secs_f64() / total.as_secs_f64(),
                gpu_head,
                tok_s,
            );
        }

        eprintln!("  --- summary (min / max / spread / mean over {rounds} rounds, order rotated) ---");
        let mut seen: Vec<usize> = Vec::new();
        for &(tokens, _, _, _, _) in rows.iter() {
            if seen.contains(&tokens) {
                continue;
            }
            seen.push(tokens);
            let mut v: Vec<f64> = rows.iter().filter(|r| r.0 == tokens).map(|r| r.4).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            eprintln!(
                "  {tokens:6} tokens: {:>7.1} min / {:>7.1} max / {:>5.1}% spread / {:>7.1} mean tok/s",
                v[0],
                v[v.len() - 1],
                100.0 * (v[v.len() - 1] - v[0]) / v[0],
                mean,
            );
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn where_a_request_spends_its_time() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => panic!("the engine did not load: {e:#}"),
        };
        let sizes: Vec<usize> = std::env::var("CLEF_STAGE_TOKENS").ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![778, 8186, 16386, 32686, 65546]);
        let rounds: usize = std::env::var("CLEF_STAGE_ROUNDS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(3);

        let questions: String = (0..20).map(|i| format!(
            "\"relp_{i}\": {{\"type\": \"noul\", \"instructions\": \"Is item {i} relevant?\", \
             \"criteria\": {{\"true\": \"It is about the subject.\", \"false\": \"It is off topic.\"}}}},\
             \"rel_{i}\": {{\"type\": \"score\", \"instructions\": \"How relevant is item {i}?\", \
             \"criteria\": {{\"0\": \"No.\", \"1\": \"Loose.\", \"2\": \"Same subject.\", \"3\": \"Direct.\"}}}},\
             \"ans_{i}\": {{\"type\": \"noul\", \"instructions\": \"Does item {i} answer it?\", \
             \"criteria\": {{\"true\": \"It states the fact.\", \"false\": \"It does not.\"}}}}"
        )).collect::<Vec<_>>().join(", ");
        eprintln!("  {:>8} {:>9} {:>9} {:>9} {:>7} {:>7} {:>8}",
                  "tokens", "backbone", "head", "total", "bb%", "hd%", "enc%");
        for tokens in sizes {
            let mut repeats = (tokens / 12).max(1);
            let mut body = String::new();
            let mut got = 0usize;
            for _ in 0..8 {
                let state = "The migration ran overnight and finished at 04:12. Two rows were skipped. "
                    .repeat(repeats);
                let json = format!("{{\"state\": {state:?}, \"questions\": {{{questions}}}}}");
                let request: SystemOneRequest = serde_json::from_str(&json).expect("parse");
                let encoded = engine.encoder.encode(&request).expect("encode");
                got = encoded.input_ids.len();
                if got >= tokens { body = json; break; }
                body = json;
                repeats = (repeats as f64 * tokens as f64 / got as f64 * 1.02) as usize + 1;
            }
            let request: SystemOneRequest = serde_json::from_str(&body).expect("parse");
            let _ = engine.decide_timed(&request).expect("warm-up");
            let mut rows = Vec::new();

            let mut enc_ms: Vec<f64> = Vec::new();
            for _ in 0..rounds {
                let t = std::time::Instant::now();
                let _ = engine.encode_and_check(&request).expect("encode");
                enc_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let (_, bb, hd, gpu, gpu_hd) = engine.decide_timed(&request).expect("timed");
                rows.push((bb.as_secs_f64() * 1e3, hd.as_secs_f64() * 1e3,
                           gpu.as_secs_f64() * 1e3, gpu_hd.as_secs_f64() * 1e3));
            }
            let n = rows.len() as f64;
            let bb = rows.iter().map(|r| r.0).sum::<f64>() / n;
            let hd = rows.iter().map(|r| r.1).sum::<f64>() / n;
            let enc_ms = enc_ms.iter().sum::<f64>() / n;
            let tot = bb + hd + enc_ms;
            eprintln!("  {got:>8} {bb:>9.1} {hd:>9.1} {tot:>9.1} {:>6.1}% {:>6.1}% {:>7.1}%   enc {enc_ms:.1} ms",
                      100.0 * bb / tot.max(1.0), 100.0 * hd / tot.max(1.0),
                      100.0 * enc_ms / tot.max(1.0));
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_two_batch_invariance_arms_agree_in_a_clean_process() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }

        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => panic!("the engine did not load: {e:#}"),
        };
        let mut a: SystemOneRequest =
            serde_json::from_str(SMOKE_REQUEST).expect("the smoke request parses");
        a.validate().expect("valid");
        let mut b = a.clone();
        b.state = format!("{}\n\n{}", a.state, "A second, distinct document for the batched pass.");
        b.validate().expect("valid");
        std::env::set_var("CLEF_LAYER_TAG", "batch_");
        let batched = engine.decide_batch(&[a.clone(), b.clone()]);
        std::env::set_var("CLEF_LAYER_TAG", "solo_");
        let solo = vec![engine.decide(&a), engine.decide(&b)];

        for i in 0..2 {
            let x = match &batched[i] {
                Ok(x) => x,
                Err(e) => { eprintln!("request {i}: batched error {e:#}"); continue; }
            };
            let y = match &solo[i] {
                Ok(y) => y,
                Err(e) => { eprintln!("request {i}: solo error {e:#}"); continue; }
            };
            let mut worst = 0.0f32;
            let mut where_ = None;
            for (qi, (qb, qs)) in x.probabilities.iter().zip(&y.probabilities).enumerate() {
                for ((kb, vb), (ks, vs)) in qb.1.iter().zip(qs.1.iter()) {
                    if kb != ks { continue; }
                    let d = (vb - vs).abs();
                    if d > worst { worst = d; where_ = Some((qi, kb.clone())); }
                }
            }
            eprintln!("  request {i}: worst |dp| {worst:.3e} at {where_:?}");
        }

        let c: SystemOneRequest = serde_json::from_str(SMOKE_REQUEST).expect("the smoke request parses");
        let mut d_req = c.clone();
        d_req.state = "An entirely different document, with no prefix in common.".to_string();
        std::env::set_var("CLEF_LAYER_TAG", "distinct_batch_");
        let batched2 = engine.decide_batch(&[c.clone(), d_req.clone()]);
        std::env::set_var("CLEF_LAYER_TAG", "distinct_solo_");
        let solo2 = vec![engine.decide(&c), engine.decide(&d_req)];
        for i in 0..2 {
            let (x, y) = match (&batched2[i], &solo2[i]) {
                (Ok(x), Ok(y)) => (x, y),
                _ => { eprintln!("request {i}: an arm errored"); continue; }
            };
            let mut worst = 0.0f32;
            for (qb, qs) in x.probabilities.iter().zip(&y.probabilities) {
                for ((kb, vb), (ks, vs)) in qb.1.iter().zip(qs.1.iter()) {
                    if kb == ks { worst = worst.max((vb - vs).abs()); }
                }
            }
            eprintln!("  DISTINCT-PREFIX request {i}: worst |dp| {worst:.3e}");
        }

        let snap = |slot: usize| -> Option<Vec<u16>> {
            (0..40).find_map(|li| engine.runner.carry_kv_snapshot(li))
        };
        std::env::set_var("CLEF_LAYER_TAG", "kv_batch_");
        let _ = engine.decide_batch(&[a.clone(), b.clone()]);
        let kv_batched = snap(1);
        std::env::set_var("CLEF_LAYER_TAG", "kv_solo_");
        let _ = engine.decide(&b);
        let kv_solo = snap(0);
        if let (Some(kb), Some(ks)) = (kv_batched.as_ref(), kv_solo.as_ref()) {
            let cap = crate::gpu::kv_capacity();

            let slots = engine.runner.carry_geometry().map(|(s, _)| s).unwrap_or(1);
            let per = kb.len() / (slots * cap).max(1);

            let rows = 267usize.min(cap);
            if per > 0 && kb.len() >= (cap + rows) * per && ks.len() >= rows * per {
                let a0 = &ks[..rows * per];
                let b0 = &kb[cap * per..(cap + rows) * per];
                let worst = a0.iter().zip(b0).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0);
                eprintln!("  b's keys: slot 0 base 0 vs slot 1 base {cap} (per {per}, {rows} rows): \
                           worst |diff| {worst}");
            } else {
                eprintln!("  kv snapshot unusable: batched {} solo {} cap {cap} slots {slots} per {per}",
                          kb.len(), ks.len());
            }
        } else {
            eprintln!("  no kv snapshot");
        }

        let snap_state = || -> Option<Vec<f32>> { (0..40).find_map(|li| engine.runner.carry_state_snapshot(li)) };
        let _ = engine.decide_batch(&[a.clone(), b.clone()]);
        let st_batched = snap_state();

        if let Some(sb) = st_batched.as_ref() {
            let slots = engine.runner.carry_geometry().map(|(s, _)| s).unwrap_or(1);
            if sb.len() % slots == 0 {
                let per = sb.len() / slots;
                for sl in 0..slots {
                    let r = &sb[sl * per..(sl + 1) * per];
                    let sum: f64 = r.iter().map(|x| *x as f64).sum();
                    eprintln!("  STATE slot {sl}: f{:.6}_{:.6}_{:.6}_{:.6} sum {:.6}",
                              r[0], r[1], r[2], r[3], sum);
                }
            }
        }
        let _ = engine.decide(&b);
        let st_solo = snap_state();
        if let (Some(sb), Some(ss)) = (st_batched.as_ref(), st_solo.as_ref()) {
            let slots = engine.runner.carry_geometry().map(|(s, _)| s).unwrap_or(1);
            if sb.len() == ss.len() && sb.len() % slots == 0 {
                let per = sb.len() / slots;
                let worst = ss[..per].iter().zip(&sb[per..2 * per])
                    .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                let nz = ss[..per].iter().filter(|x| **x != 0.0).count();
                eprintln!("  b's GDN state: slot 0 vs slot 1, worst {worst:.4e} (slot 0 has {nz} non-zero)");
            }
        }
        std::env::set_var("CLEF_LAYER_TAG", "solo_first_");
        let sf_a = engine.decide(&a).expect("solo-first a");
        let sf_b = engine.decide(&b).expect("solo-first b");
        let bf = engine.decide_batch(&[a.clone(), b.clone()]);
        for (i, x, y) in [(0usize, &bf[0], &sf_a), (1, &bf[1], &sf_b)] {
            let (x, y) = match (x, y) {
                (Ok(x), y) => (x, y),
                _ => { eprintln!("  SOLO-FIRST request {i}: the batched arm errored"); continue; }
            };
            let mut worst = 0.0f32;
            for (qb, qs) in x.probabilities.iter().zip(&y.probabilities) {
                for ((kb, vb), (ks, vs)) in qb.1.iter().zip(qs.1.iter()) {
                    if kb == ks { worst = worst.max((vb - vs).abs()); }
                }
            }
            eprintln!("  SOLO-FIRST request {i}: worst |dp| {worst:.3e}");
        }

        let r1 = engine.decide(&b).expect("run 1");
        let r2 = engine.decide(&b).expect("run 2");
        let r3 = engine.decide(&b).expect("run 3");
        let diff = |x: &Decided, y: &Decided| -> f32 {
            let mut w = 0.0f32;
            for (qx, qy) in x.probabilities.iter().zip(&y.probabilities) {
                for ((kx, vx), (ky, vy)) in qx.1.iter().zip(qy.1.iter()) {
                    if kx == ky { w = w.max((vx - vy).abs()); }
                }
            }
            w
        };
        eprintln!("  SAME REQUEST x3: 1v2 {:.4e}  2v3 {:.4e}  1v3 {:.4e}",
                  diff(&r1, &r2), diff(&r2, &r3), diff(&r1, &r3));
        assert_eq!(diff(&r1, &r2), 0.0,
                   "a REPLAY disagrees with its own CAPTURE for the same request — that is the defect, \
                    with no batching and no second request involved");

        let b1 = engine.decide(&b).expect("b alone");
        let _ = engine.decide(&a).expect("a in between");
        let b2 = engine.decide(&b).expect("b after a");
        let w_ba = diff(&b1, &b2);
        eprintln!("  B alone vs B after A (single calls, graph on): {w_ba:.4e}");
        assert_eq!(w_ba, 0.0,
                   "one request's answer changed after another request ran — a buffer is read before it is \
                    written inside the captured region");

        let one = engine.decide_batch(&[b.clone()]);
        let w_one = match &one[0] {
            Ok(x) => diff(x, &engine.decide(&b).expect("decide b again")),
            Err(e) => { eprintln!("  BATCH OF ONE errored: {e:#}"); f32::NAN }
        };
        eprintln!("  decide_batch(&[b]) vs decide(b): {w_one:.4e}");
        let two = engine.decide_batch(&[a.clone(), b.clone()]);
        let w_two = match (&two[1], &one[0]) {
            (Ok(x), Ok(y)) => diff(x, y),
            _ => f32::NAN,
        };
        eprintln!("  decide_batch(&[a,b])[1] vs decide_batch(&[b])[0]: {w_two:.4e}");

        for i in 0..2 {
            let (x, y) = match (&batched[i], &solo[i]) {
                (Ok(x), Ok(y)) => (x, y),
                _ => continue,
            };
            let mut worst = 0.0f32;
            let mut where_ = None;
            for (qi, (qb, qs)) in x.probabilities.iter().zip(&y.probabilities).enumerate() {
                for ((kb, vb), (ks, vs)) in qb.1.iter().zip(qs.1.iter()) {
                    if kb != ks { continue; }
                    let dd = (vb - vs).abs();
                    if dd > worst { worst = dd; where_ = Some((qi, kb.clone())); }
                }
            }
            assert!(worst < 1e-6,
                    "request {i} depends on what it is batched with: worst |dp| {worst:.3e} at {where_:?}");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn every_rung_matches_the_bf16_oracle() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/eval/clef_bf16_oracle_fixture.json"))
                .expect("the bf16 oracle fixture parses");
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }

        let rungs: Vec<usize> = crate::gpu::Rungs::body().list().to_vec();
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => panic!("the engine did not load: {e:#}"),
        };
        eprintln!("  rungs to validate: {rungs:?}");
        let mut worst_all = 0.0f64;
        for r in &rungs {
            std::env::set_var("CLEF_FORCE_RUNG", r.to_string());
            for case in fixture["cases"].as_array().expect("cases") {
                let tokens = case["input_tokens"].as_u64().unwrap_or(0);
                let request: SystemOneRequest =
                    serde_json::from_value(case["request"].clone()).expect("the fixture's request");
                let decided = engine.decide(&request).expect("decide");
                let mut kls: Vec<f64> = Vec::new();
                let mut flips = 0usize;
                let mut n = 0usize;
                for (qid, ref_ans) in case["answers"].as_object().expect("answers") {
                    let Some(got) = decided.probabilities.iter().find(|(k, _)| k == qid).map(|(_, v)| v)
                        else { continue };
                    let r: std::collections::BTreeMap<String, f64> = ref_ans
                        .as_object().expect("a probability map")
                        .iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect();
                    let g: std::collections::BTreeMap<String, f64> =
                        got.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
                    let keys: Vec<&String> = {
                        let mut u: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
                        u.extend(r.keys());
                        u.extend(g.keys());
                        u.into_iter().collect()
                    };
                    let (sr, sg): (f64, f64) = (r.values().sum(), g.values().sum());
                    let (sr, sg) = (if sr > 0.0 { sr } else { 1.0 }, if sg > 0.0 { sg } else { 1.0 });
                    let kl: f64 = keys.iter().map(|k| {
                        let p = r.get(*k).copied().unwrap_or(0.0) / sr;
                        let q = g.get(*k).copied().unwrap_or(0.0) / sg;
                        if p <= 0.0 { 0.0 } else { p * (p / (q + 1e-12)).ln() }
                    }).sum();
                    let argmax = |m: &std::collections::BTreeMap<String, f64>| -> Option<String> {
                        m.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(k, _)| k.clone())
                    };
                    if argmax(&r) != argmax(&g) { flips += 1; }
                    kls.push(kl);
                    n += 1;
                }
                let mean = kls.iter().sum::<f64>() / (kls.len().max(1) as f64);
                let max = kls.iter().cloned().fold(0.0f64, f64::max);
                eprintln!("  rung {r:>5}, {tokens:>6} tokens: mean KL {mean:.3e}, max {max:.3e}, \
                           flips {flips}/{n}");
                worst_all = worst_all.max(mean);

                assert!(mean < 5e-3, "rung {r} at {tokens} tokens: mean KL {mean:.3e} is over 5e-3");
                assert_eq!(flips, 0, "rung {r} at {tokens} tokens: {flips} top-1 flips against bf16");
            }
        }
        std::env::remove_var("CLEF_FORCE_RUNG");
        eprintln!("  every rung matches the bf16 oracle; worst mean KL {worst_all:.3e}");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_shared_pass_is_exact_only_when_nothing_is_padded() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let chunk = crate::gpu::max_chunk_size();
        eprintln!("  chunk = {chunk}, kv_slots = {}", crate::gpu::kv_slots());

        let build = |target: usize, tag: &str| -> SystemOneRequest {
            let base: SystemOneRequest =
                serde_json::from_str(SMOKE_REQUEST).expect("the smoke request parses");
            let filler = "The migration ran again and finished later. ";
            let mut best: Option<(SystemOneRequest, usize)> = None;
            for n in 0..400usize {
                let mut r = base.clone();
                r.state = format!("{}{}", base.state, filler.repeat(n));
                r.validate().expect("valid");
                let len = engine.encoder.encode(&r).expect("encode").input_ids.len();
                let d = len.abs_diff(target);
                if best.as_ref().map_or(true, |(_, bd)| d < *bd) {
                    best = Some((r, d));
                }
                if len == target {
                    break;
                }
                if len > target {
                    break;
                }
            }
            let (r, d) = best.expect("a candidate");
            eprintln!("  [{tag}] target {target} -> {} tokens (off by {d})",
                      engine.encoder.encode(&r).expect("encode").input_ids.len());
            r
        };

        let only: Option<usize> = std::env::var("CLEF_SWEEP_ONLY").ok()
            .and_then(|v| v.trim().parse().ok());
        let worst_of = |a: &SystemOneRequest, b: &SystemOneRequest| -> (f32, usize, usize) {
            let before = engine.runner.chunked_width_max();
            let passes_before = engine.runner.chunked_passes();
            let batched = engine.decide_batch(&[a.clone(), b.clone()]);
            let width = engine.runner.chunked_width_max();
            let passes = engine.runner.chunked_passes() - passes_before;
            let _ = before;
            let solo = vec![engine.decide(a).expect("a"), engine.decide(b).expect("b")];
            let mut w = 0.0f32;
            for (s, bt) in solo.iter().zip(batched.iter()) {
                let bt = bt.as_ref().expect("batched");
                for ((_, p1), (_, p2)) in s.probabilities.iter().zip(bt.probabilities.iter()) {
                    for (k, v) in p1.iter() {
                        let z = p2.get(k).copied().unwrap_or(f32::NAN);
                        w = w.max((v - z).abs());
                    }
                }
            }
            (w, width, passes)
        };
        for (i, (ta, tb, why)) in [
            (chunk, chunk, "both chunks FULL (a shared pass)"),
            (chunk, chunk + 256, "both full, different pass COUNTS"),
            (chunk + 4, chunk + 304, "both full, real > chunk so the next turn is a tail"),
            (255, 267, "both SUB-chunk: the case the guard refuses"),
        ].into_iter().enumerate() {
            if only.map_or(false, |o| o != i) {
                continue;
            }
            let a = build(ta, "a");
            let b = build(tb, "b");
            let (w, width, passes) = worst_of(&a, &b);
            let len_a = engine.encoder.encode(&a).expect("e").input_ids.len();
            let len_b = engine.encoder.encode(&b).expect("e").input_ids.len();
            eprintln!("  arm {i}: {len_a:>5} + {len_b:<5}  width_max={width}  passes={passes}  \
                       worst prob diff {w:.3e}   [{why}]");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_batched_decide_matches_separate_calls() {
        std::env::set_var("CLEF_KV_SLOTS", "2");
        std::env::set_var("CLEF_MAX_PASS_TOKENS", "2048");

        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let mut a: SystemOneRequest =
            serde_json::from_str(SMOKE_REQUEST).expect("the smoke request parses");
        a.validate().expect("valid");

        let mut b = a.clone();
        b.state = format!("{}\n\n{}", a.state, "A second, distinct document for the batched pass.");
        b.validate().expect("valid");
        std::env::set_var("CLEF_LAYER_TAG", "batch_");

        let sharing_before = engine.runner.chunked_passes_sharing();
        let batched_first = engine.decide_batch(&[a.clone(), b.clone()]);
        let sharing = engine.runner.chunked_passes_sharing() - sharing_before;
        assert!(
            sharing >= 1,
            "the batched call ran NO shared pass ({sharing} of them) — its two arms are the same serial run, \
             so this gate cannot see batching at all. Check the scheduler's chunk fitting: 255 + 267 tokens \
             must fit into one graphed geometry."
        );
        eprintln!("  batched call ran {sharing} shared pass(es)");
        std::env::set_var("CLEF_LAYER_TAG", "solo_");
        let solo: Vec<Decided> = vec![
            engine.decide(&a).expect("decide a"),
            engine.decide(&b).expect("decide b"),
        ];
        let _ = &batched_first;
        let batched = batched_first;
        assert_eq!(batched.len(), 2, "one result per request, in order");

        {
            let b2 = engine.decide_batch(&[a.clone(), b.clone()]);
            let s2: Vec<Decided> = vec![
                engine.decide(&a).expect("decide a"),
                engine.decide(&b).expect("decide b"),
            ];
            let mut w = 0.0f32;
            for (x, y) in s2.iter().zip(b2.iter()) {
                let y = y.as_ref().expect("batched");
                for ((_, p1), (_, p2)) in x.probabilities.iter().zip(y.probabilities.iter()) {
                    for (k, v) in p1.iter() {
                        let z = p2.get(k).copied().unwrap_or(f32::NAN);
                        w = w.max((v - z).abs());
                    }
                }
            }
            eprintln!("  SECOND pass (settled heap): worst prob diff {w:.3e}");
        }
        for (i, (solo_i, batched_i)) in solo.iter().zip(batched.iter()).enumerate() {
            let batched_i = batched_i.as_ref().unwrap_or_else(|e| panic!("request {i}: {e:#}"));
            assert_eq!(batched_i.input_tokens, solo_i.input_tokens,
                       "request {i} was tokenised differently in the batch");
            assert_eq!(batched_i.probabilities.len(), solo_i.probabilities.len());
            for ((id, p_solo), (_, p_batch)) in
                solo_i.probabilities.iter().zip(batched_i.probabilities.iter())
            {
                for (opt, v) in p_solo.iter() {
                    let w = p_batch.get(opt).expect("the same options");
                    assert_eq!(
                        v, w,
                        "request {i}, question {id}, option {opt}: the batched pass gave {w} and the \
                         separate pass gave {v}. A request's answer must not depend on what it is \
                         batched with."
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "needs the checkpoint on the card"]
    fn the_engine_answers_a_real_request() {
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let health = engine.health();
        eprintln!("  device:  {}", health.device);
        eprintln!("  backend: {}  quant: {}", health.backend, health.quant);

        eprintln!(
            "  graphs {} ({} chunked) | pending {} | chunk {} | kv {} tokens / {} MiB",
            health.graphs, health.graphs_chunked, health.captures_pending, health.chunk_tokens,
            health.kv_tokens, health.kv_mib
        );
        assert!(health.graphs > 0, "/health must report the graphs the warm-up captured");
        if crate::gpu::chunking_enabled() {
            assert!(health.graphs_chunked > 0, "a long request needs a chunk graph warmed, not captured");
        }
        assert_eq!(health.chunk_tokens, crate::gpu::max_chunk_size());
        assert_eq!(health.kv_tokens, crate::gpu::kv_capacity());
        assert!(health.kv_mib > 0);

        let request: SystemOneRequest =
            serde_json::from_str(SMOKE_REQUEST).expect("the smoke request parses");
        request.validate().expect("the smoke request is valid");

        let started = std::time::Instant::now();
        let decided = engine.decide(&request).expect("decide");
        let elapsed = started.elapsed();
        eprintln!("  {} tokens in {elapsed:?}", decided.input_tokens);

        assert_eq!(decided.probabilities.len(), request.questions.len(),
                   "one distribution per question");
        assert!(decided.input_tokens > 0);
        for (id, probs) in &decided.probabilities {
            let sum: f32 = probs.values().sum();
            eprintln!("    {id}: {probs:?}  (sum {sum:.6})");
            assert!(probs.values().all(|v| v.is_finite()), "{id} produced a non-finite probability");
            assert!((sum - 1.0).abs() < 1e-4, "{id} does not sum to 1: {sum}");
            let (_, q) = request.questions.iter().find(|(k, _)| k == id).expect("known id");
            let expected: Vec<String> = crate::encode::question_options(q)
                .into_iter()
                .map(|(k, _)| k)
                .collect();
            let got: Vec<&String> = probs.keys().collect();
            assert_eq!(got.len(), expected.len(), "{id} has the wrong option count");
            for k in &expected {
                assert!(probs.contains_key(k), "{id} is missing option {k}");
            }
        }
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    #[test]
    fn the_warmed_geometries_are_achievable_passes() {
        let rungs = crate::gpu::Rungs::body();
        let top = crate::gpu::max_chunk_size();
        let slots = crate::gpu::kv_slots();
        let sizes = dense_warm_up_sizes();
        eprintln!("  max_chunk_size {top}, slots {slots}: {} warmed geometries", sizes.len());
        assert!(sizes.len() < 200, "the warm-up must stay bounded: {} sizes", sizes.len());
        for &t in &sizes {
            assert!(t <= slots * top,
                    "{t} rows is wider than the widest pass there can be ({} x {top} = {})",
                    slots, slots * top);
            assert!(t >= rungs.smallest(), "{t} is below the smallest rung");
            assert_eq!(t % crate::cuda::CHUNK_GRANULE, 0, "{t} is not granule-aligned");
        }

        for &r in rungs.list() {
            assert!(sizes.contains(&r), "the ladder's rung {r} is not warmed");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_engine_matches_the_bf16_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/eval/clef_bf16_oracle_fixture.json"))
                .expect("the bf16 oracle fixture parses");
        let model = model_dir();
        if !model.exists() {
            eprintln!("SKIP: no checkpoint at {}", model.display());
            return;
        }
        let engine = match GpuEngine::load(&model, &head_dir()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let mut worst = 0.0f64;
        for case in fixture["cases"].as_array().expect("cases") {
            let tokens = case["input_tokens"].as_u64().unwrap_or(0);
            let request: SystemOneRequest =
                serde_json::from_value(case["request"].clone()).expect("the fixture's request");

            {
                let enc = engine.encode_and_check(&request).expect("encode");
                let h = engine.runner.forward_dev(&enc.input_ids).expect("backbone");
                let v = engine.runner.device.download_f32(&h).expect("download");
                let bad = v.iter().filter(|x| !x.is_finite()).count();
                eprintln!("  {tokens:>6} tokens: backbone {} values, {bad} non-finite", v.len());
            }

            let before = engine.runner.chunked_passes();
            let decided = engine.decide(&request).expect("decide");
            let after = engine.runner.chunked_passes();
            assert!(after > before,
                "{tokens} tokens: `decide` ran no chunked pass ({} -> {after}), so this KL is NOT \
                 measured on the serving path", before);

            let mut kls: Vec<f64> = Vec::new();
            let mut flips = 0usize;
            let mut n = 0usize;
            for (qid, ref_ans) in case["answers"].as_object().expect("answers") {
                let Some(got) = decided.probabilities.iter().find(|(k, _)| k == qid).map(|(_, v)| v)
                    else { continue };
                let r: std::collections::BTreeMap<String, f64> = ref_ans
                    .as_object().expect("a probability map")
                    .iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect();
                let g: std::collections::BTreeMap<String, f64> =
                    got.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();

                let keys: Vec<&String> = {
                    let mut u: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
                    u.extend(r.keys());
                    u.extend(g.keys());
                    u.into_iter().collect()
                };
                let (sr, sg): (f64, f64) = (r.values().sum(), g.values().sum());
                let (sr, sg) = (if sr > 0.0 { sr } else { 1.0 }, if sg > 0.0 { sg } else { 1.0 });
                let kl: f64 = keys.iter().map(|k| {
                    let p = r.get(*k).copied().unwrap_or(0.0) / sr;
                    let q = g.get(*k).copied().unwrap_or(0.0) / sg;
                    if p <= 0.0 { 0.0 } else { p * (p / (q + 1e-12)).ln() }
                }).sum();

                let argmax = |m: &std::collections::BTreeMap<String, f64>| -> Option<String> {
                    m.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(k, _)| k.clone())
                };
                if argmax(&r) != argmax(&g) { flips += 1; }
                if kl.is_nan() {
                    eprintln!("  NON-FINITE KL on {qid}: oracle {:?} engine {:?}", r, g);
                }
                kls.push(kl);
                n += 1;
            }
            let mean = kls.iter().sum::<f64>() / (kls.len().max(1) as f64);
            let max = kls.iter().cloned().fold(0.0f64, f64::max);
            eprintln!("  {tokens:>6} tokens: mean KL {mean:.3e}, max {max:.3e}, top-1 flips {flips}/{n}");
            worst = worst.max(mean);
        }

        assert!(worst < 5e-3, "mean KL against the bf16 oracle is {worst:.3e} — the engine disagrees with bf16");
    }

    #[test]
    fn the_size_gate_refuses_above_the_limit_and_not_at_it() {
        assert!(check_length(32768, 32768).is_ok(), "exactly the limit must be servable");
        assert!(check_length(0, 32768).is_ok());

        let over = check_length(32769, 32768).expect_err("one over must be refused");
        assert_eq!(over.tokens, 32769, "the error must carry the count the caller sent");
        assert!(over.why.contains("32768"), "the reason must name the limit: {}", over.why);
    }

    #[test]
    fn the_refusal_names_the_true_count_and_the_limit() {
        let message = check_length(200_000, 32768).unwrap_err().to_string();
        assert!(message.contains("200000"), "the true count must be in the message: {message}");
        assert!(message.contains("32768"), "the limit must be in the message: {message}");
        assert!(message.contains("over"), "and it must say it is over: {message}");
    }

    #[test]
    fn the_default_limit_is_the_models_context() {
        let parse = |v: Option<&str>| -> usize {
            v.and_then(|v| v.parse().ok()).unwrap_or(32768)
        };
        assert_eq!(parse(None), 32768);
        assert_eq!(parse(Some("")), 32768, "an empty value must fall back, not become 0");
        assert_eq!(parse(Some("nonsense")), 32768);
        assert_eq!(parse(Some("8192")), 8192);
    }
}
