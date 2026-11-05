pub fn gdn_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLEF_GDN_TRACE").is_ok())
}

pub fn gdn_trace_rows() -> usize {
    TRACE_ROWS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn gdn_trace_set_rows(n: usize) {
    TRACE_ROWS.store(n, std::sync::atomic::Ordering::Relaxed);
    if gdn_trace_on() {
        TRACE_PASS_NO.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!("--- PASS {} tokens={n} ---", TRACE_PASS_NO.load(std::sync::atomic::Ordering::Relaxed));
    }
}

pub fn gdn_trace_pass_no() -> usize {
    TRACE_PASS_NO.load(std::sync::atomic::Ordering::Relaxed)
}

static TRACE_PASS_NO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn gdn_trace_set_rows_only(n: usize) {
    TRACE_ROWS.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn gdn_trace_set_layer(li: usize) {
    TRACE_LAYER.store(li, std::sync::atomic::Ordering::Relaxed);
}

pub fn gdn_trace_layer() -> usize {
    TRACE_LAYER.load(std::sync::atomic::Ordering::Relaxed)
}

static TRACE_LAYER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

pub fn gdn_trace_row() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("CLEF_TRACE_ROW").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

static TRACE_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn gdn_trace_set_pass(label: &'static str) {
    if gdn_trace_on() {
        if let Ok(mut p) = TRACE_PASS.lock() {
            *p = label;
        }
    }
}

pub fn gdn_trace_pass() -> &'static str {
    TRACE_PASS.lock().map(|p| *p).unwrap_or("?")
}

static TRACE_PASS: std::sync::Mutex<&'static str> = std::sync::Mutex::new("init");

#[macro_export]
macro_rules! ck {
    ($d:expr, $name:expr, $buf:expr, $hidden:expr) => {
        if $crate::gpu::gdn_trace_on() {
            if std::env::var("CLEF_TRACE_FULL").is_ok() {
                let all = $d.download_f32($buf)?;
                let sum = all.iter().take($buf.len() / 4).fold(0u64, |a, x| a
                    .wrapping_mul(1_000_003).wrapping_add(x.to_bits() as u64));
                eprintln!("    GDN L{:<2} {:<14} FULL {:016x} len={}", $crate::gpu::gdn_trace_layer(),
                          $name, sum, $buf.len() / 4);
            }

            let mut cols = Vec::new();

            let trace_row = $crate::gpu::gdn_trace_row();
            for row in [trace_row, trace_row + 1] {
                let _cols = $hidden;
                if _cols == 0 { continue; }
                let _rows = $buf.len() / (_cols * 4);
                if _rows == 0 { continue; }
                let _row = $crate::gpu::gdn_trace_row();
                if _row + 1 >= _rows { continue; }
                let r = $d.copy_rows_from($buf, row, 1, _cols)?;
                let v = $d.download_f32(&r)?;

                let ncols = std::env::var("CLEF_TRACE_COLS").ok()
                    .and_then(|v| v.parse::<usize>().ok()).unwrap_or(4);

                if std::env::var("CLEF_TRACE_SUM").is_ok() {
                    cols.push(format!("{row}:sum={:016x}", v.iter().fold(0u64, |a, x| a
                        .wrapping_mul(1_000_003)
                        .wrapping_add(x.to_bits() as u64))));
                } else {
                    cols.push(format!("{row}:{:?}",
                        v[..ncols.min(v.len())].iter().map(|x| format!("{x:.9}")).collect::<Vec<_>>()));
                }
            }

            if std::env::var("CLEF_ALLOC_TRACE").is_ok() {
                eprintln!("    GDN L{:<2} {:<14} ptr={:?} len={} p{} {:<7} {}", $crate::gpu::gdn_trace_layer(), $name, $buf.device_ptr(),
                          $buf.len(), $crate::gpu::gdn_trace_pass_no(), $crate::gpu::gdn_trace_pass(),
                          cols.join(" "));
            } else {
                eprintln!("    GDN L{:<2} {:<14} p{} {:<7} rows={} {}", $crate::gpu::gdn_trace_layer(), $name, $crate::gpu::gdn_trace_pass_no(), $crate::gpu::gdn_trace_pass(), $buf.len() / ($hidden * 4), cols.join(" "));
            }
        }
    };
}

use anyhow::{Context, Result};

use crate::cuda::{Buffer, Device};
use crate::loader::{Checkpoint, Projection, Weights};
use crate::model::head::QuestionSpec;
use crate::model::{LayerKind, TextConfig};
use crate::placement::HostTable;

pub enum DeviceProjection {
    Nvfp4 {
        packed: Buffer,
        scales: Buffer,
        rows: usize,
        cols: usize,
        global_scale: f32,
        input_global_scale: Option<f32>,
    },

    Fp8 { values: Buffer, scales: Buffer, rows: usize, cols: usize },

    Bf16 { values: Buffer, rows: usize, cols: usize },
}

impl DeviceProjection {
    pub fn upload(device: &Device, projection: &Projection) -> Result<Self> {
        let (rows, cols) = (projection.rows, projection.cols);
        match &projection.weights {
            Weights::Nvfp4 { packed, scales, global_scale } => {
                let expected = rows * cols / 2;
                if packed.len() != expected {
                    anyhow::bail!("packed weights are {} bytes, expected {expected}", packed.len());
                }

                let swizzled = device
                    .swizzle_scales(scales, rows, cols / 16, false)
                    .context("swizzling the NVFP4 scales")?;
                Ok(Self::Nvfp4 {
                    packed: device.upload(packed)?,
                    scales: swizzled,
                    rows,
                    cols,
                    global_scale: *global_scale,
                    input_global_scale: projection.input_global_scale,
                })
            }
            Weights::Fp8 { values, scales } => Ok(Self::Fp8 {
                values: device.upload(values)?,
                scales: device.upload(scales)?,
                rows,
                cols,
            }),
            Weights::Bf16 { values } => Ok(Self::Bf16 {
                values: device.upload(values)?,
                rows,
                cols,
            }),
        }
    }
}

pub struct DeviceDeltaNet {
    pub input_layernorm: Buffer,
    pub post_attention_layernorm: Buffer,
    pub in_proj_qkv: DeviceProjection,
    pub in_proj_z: DeviceProjection,
    pub in_proj_a: DeviceProjection,
    pub in_proj_b: DeviceProjection,
    pub out_proj: DeviceProjection,
    pub conv_weight: Buffer,
    pub norm_weight: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
    pub mlp_gate: DeviceProjection,
    pub mlp_up: DeviceProjection,
    pub mlp_down: DeviceProjection,
}

pub struct DeviceAttention {
    pub input_layernorm: Buffer,
    pub post_attention_layernorm: Buffer,
    pub q_proj: DeviceProjection,
    pub k_proj: DeviceProjection,
    pub v_proj: DeviceProjection,
    pub o_proj: DeviceProjection,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub mlp_gate: DeviceProjection,
    pub mlp_up: DeviceProjection,
    pub mlp_down: DeviceProjection,
}

pub enum DeviceLayer {
    Linear(Box<DeviceDeltaNet>),
    Full(Box<DeviceAttention>),
}

pub struct DeviceBackbone {
    pub layers: Vec<DeviceLayer>,
    pub embed_tokens: Buffer,
    pub final_norm: Buffer,
    pub config: TextConfig,
}

pub fn load_backbone(device: &Device, checkpoint: &Checkpoint, config: &TextConfig) -> Result<DeviceBackbone> {
    let prefix = checkpoint.text_prefix()?.to_string();
    let hidden = config.hidden_size;

    let upload_projection = |name: &str| -> Result<DeviceProjection> {
        let p = checkpoint.projection(name)?;
        DeviceProjection::upload(device, &p)
    };
    let upload_f32 = |name: &str, len: usize| -> Result<Buffer> {
        let (values, shape) = checkpoint.tensor_f32(name)?;
        if values.len() != len {
            anyhow::bail!("{name} has {} values ({shape:?}), expected {len}", values.len());
        }
        device.upload_f32(&values)
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = config.layer_prefix(checkpoint.text_prefix()?, i);
        let hd = config.head_dim;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;

        layers.push(match config.layer_kind(i) {
            crate::model::LayerKind::LinearAttention => {
                let attn = format!("{p}.linear_attn");
                DeviceLayer::Linear(Box::new(DeviceDeltaNet {
                    input_layernorm: upload_f32(&format!("{p}.input_layernorm.weight"), hidden)?,
                    post_attention_layernorm: upload_f32(
                        &format!("{p}.post_attention_layernorm.weight"), hidden)?,
                    in_proj_qkv: upload_projection(&format!("{attn}.in_proj_qkv"))?,
                    in_proj_z: upload_projection(&format!("{attn}.in_proj_z"))?,
                    in_proj_a: upload_projection(&format!("{attn}.in_proj_a"))?,
                    in_proj_b: upload_projection(&format!("{attn}.in_proj_b"))?,
                    out_proj: upload_projection(&format!("{attn}.out_proj"))?,
                    conv_weight: upload_f32(
                        &format!("{attn}.conv1d.weight"), config.conv_dim() * config.linear_conv_kernel_dim)?,
                    norm_weight: upload_f32(
                        &format!("{attn}.norm.weight"), config.linear_value_head_dim)?,
                    a_log: upload_f32(&format!("{attn}.A_log"), config.linear_num_value_heads)?,
                    dt_bias: upload_f32(&format!("{attn}.dt_bias"), config.linear_num_value_heads)?,
                    mlp_gate: upload_projection(&format!("{p}.mlp.gate_proj"))?,
                    mlp_up: upload_projection(&format!("{p}.mlp.up_proj"))?,
                    mlp_down: upload_projection(&format!("{p}.mlp.down_proj"))?,
                }))
            }
            crate::model::LayerKind::FullAttention => {
                let attn = format!("{p}.self_attn");
                DeviceLayer::Full(Box::new(DeviceAttention {
                    input_layernorm: upload_f32(&format!("{p}.input_layernorm.weight"), hidden)?,
                    post_attention_layernorm: upload_f32(
                        &format!("{p}.post_attention_layernorm.weight"), hidden)?,
                    q_proj: upload_projection(&format!("{attn}.q_proj"))?,
                    k_proj: upload_projection(&format!("{attn}.k_proj"))?,
                    v_proj: upload_projection(&format!("{attn}.v_proj"))?,
                    o_proj: upload_projection(&format!("{attn}.o_proj"))?,
                    q_norm: upload_f32(&format!("{attn}.q_norm.weight"), hd)?,
                    k_norm: upload_f32(&format!("{attn}.k_norm.weight"), hd)?,
                    mlp_gate: upload_projection(&format!("{p}.mlp.gate_proj"))?,
                    mlp_up: upload_projection(&format!("{p}.mlp.up_proj"))?,
                    mlp_down: upload_projection(&format!("{p}.mlp.down_proj"))?,
                }))
            }
        });
        let _ = (heads, kv_heads);
    }
    Ok(DeviceBackbone {
        layers,
        embed_tokens: upload_f32(&format!("{prefix}.embed_tokens.weight"),
                                 config.vocab_size * hidden)?,
        final_norm: upload_f32(&format!("{prefix}.norm.weight"), hidden)?,
        config: config.clone(),
    })
}

pub fn eager_fallback_allowed() -> bool {
    matches!(std::env::var("CLEF_EAGER").as_deref(), Ok("1"))
}

fn eager_or_error<T>(tokens: usize, why: &str, eager: impl FnOnce() -> Result<T>) -> Result<T> {
    if eager_fallback_allowed() {
        tracing::warn!(tokens, why, "running EAGERLY because CLEF_EAGER=1");
        return eager();
    }
    anyhow::bail!(
        "refusing to run {tokens} tokens eagerly: {why}. The eager path is disabled by default — set \
         CLEF_EAGER=1 to allow it, or make the capture fit."
    )
}

pub const SERVABLE_FRACTION: f64 = 0.97;

pub const EXTRAPOLATION_PAD: f64 = 0.25;

pub fn graph_requested() -> bool {
    !matches!(std::env::var("CLEF_GRAPH").as_deref(), Ok("0"))
}

fn gdn_use_chunked() -> bool {
    !matches!(std::env::var("CLEF_GDN").as_deref(), Ok("recurrent"))
}

pub const RESID_FUSED_FROM: usize = 4096;

fn resid_fused(m: usize) -> bool {
    match std::env::var("CLEF_FUSE_RESID").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => m >= RESID_FUSED_FROM,
    }
}

fn run_fp8_projection_from_quantised(d: &Device, q: &Buffer, qs: &Buffer, rows: usize,
                                     proj: &DeviceProjection, add: Option<&Buffer>)
    -> Option<Result<Buffer>> {
    match proj {
        DeviceProjection::Fp8 { values, scales, rows: pr, cols: _ } => {
            let r = if resid_fused(rows) {
                d.fp8_gemm_dev(q, values, qs, scales, rows, *pr, proj_cols(proj), add)
            } else {
                match d.fp8_gemm_dev(q, values, qs, scales, rows, *pr, proj_cols(proj), None) {
                    Ok(c) => match add {
                        Some(a) => d.add_dev(a, &c, rows * *pr),
                        None => Ok(c),
                    },
                    Err(e) => Err(e),
                }
            };
            Some(r)
        }
        _ => None,
    }
}

fn gated_gemm_on() -> bool {
    !matches!(std::env::var("CLEF_GATED_GEMM").as_deref(), Ok("0"))
}

#[allow(clippy::too_many_arguments)]
fn gated_norm_quantised(d: &Device, o: &Buffer, z: &Buffer, weight: &Buffer, tokens: usize,
                        vh: usize, vd: usize, eps: f32, proj: &DeviceProjection)
    -> Result<Option<(Buffer, Buffer)>> {
    if !gated_norm_fused() {
        return Ok(None);
    }
    let rows = tokens * vh;
    match proj {
        DeviceProjection::Nvfp4 { input_global_scale: Some(g), .. } => {
            let tile_m = Device::nvfp4_m_tile_for(tokens);
            Ok(Some(d.rms_norm_gated_quantise_nvfp4_dev(o, z, weight, rows, vd, vh, eps, *g, tile_m)?))
        }
        DeviceProjection::Fp8 { .. } => {
            Ok(Some(d.rms_norm_gated_quantise_fp8_dev(o, z, weight, rows, vd, vh, eps)?))
        }
        _ => Ok(None),
    }
}

fn gated_fp8_on() -> bool {
    matches!(std::env::var("CLEF_GATED_GEMM_FP8").as_deref(), Ok("1"))
}

fn gated_norm_fused() -> bool {
    !matches!(std::env::var("CLEF_FUSE_GATED_NORM").as_deref(), Ok("0"))
}

#[allow(clippy::too_many_arguments)]
fn mlp_gated(d: &Device, y: &Buffer, norm_w: &Buffer, tokens: usize, hidden: usize, eps: f32,
             gate_proj: &DeviceProjection, up_proj: &DeviceProjection,
             down_proj: &DeviceProjection) -> Result<Option<(Buffer, Buffer)>> {
    if !gated_gemm_on() {
        return Ok(None);
    }

    if gated_fp8_on() {
    if let (DeviceProjection::Fp8 { values: fbgp, scales: fbgs, rows: fgn, .. },
            DeviceProjection::Fp8 { values: fbup, scales: fbus, rows: fun, .. },
            DeviceProjection::Fp8 { .. }) = (gate_proj, up_proj, down_proj)
    {
        let finter = *fgn;
        if fgn == fun && finter % 128 == 0 && hidden % 128 == 0 {
            let normed = d.rms_norm_dev(y, norm_w, tokens, hidden, eps)?;
            let (q, qs) = d.quantise_fp8_dev(&normed, tokens, hidden)?;
            let act = d.fp8_gemm_gated_dev(&q, &qs, fbgp, fbgs, fbup, fbus, tokens, finter, hidden)?;
            let (aq, aqs) = d.quantise_fp8_dev(&act, tokens, finter)?;
            return Ok(Some((aq, aqs)));
        }
    }
    }
    let (DeviceProjection::Nvfp4 { packed: bgp, scales: bgs, rows: gn, global_scale: bgg,
                                   input_global_scale: Some(bgigs), .. },
         DeviceProjection::Nvfp4 { packed: bup, scales: bus, rows: un, global_scale: bug,
                                   input_global_scale: Some(buigs), .. },
         DeviceProjection::Nvfp4 { input_global_scale: Some(dnigs), .. }) =
        (gate_proj, up_proj, down_proj)
    else {
        return Ok(None);
    };
    if bgigs != buigs || gn != un {
        return Ok(None);
    }
    let inter = *gn;
    if inter % 128 != 0 || hidden % 64 != 0 {
        return Ok(None);
    }

    let Some((q, qs)) = fused_norm_quantise_nvfp4(d, y, norm_w, tokens, hidden, eps,
                                                  gate_proj, up_proj)? else {
        return Ok(None);
    };
    let (packed, scales) = d.nvfp4_gemm_gated_dev(&q, &qs, bgp, bgs, bup, bus, tokens, inter, hidden,
                                                 *bgigs, *bgg, *bug, *dnigs)?;
    Ok(Some((packed, scales)))
}

fn mlp_gate_up(d: &Device, y: &Buffer, norm_w: &Buffer, tokens: usize, hidden: usize, eps: f32,
               gate_proj: &DeviceProjection, up_proj: &DeviceProjection, trace: bool)
    -> Result<(Buffer, Buffer)> {
    if !trace {
        if let Some((q, qs)) = fused_norm_quantise_nvfp4(d, y, norm_w, tokens, hidden, eps,
                                                         gate_proj, up_proj)? {
            let g = run_nvfp4_projection_from_quantised(d, &q, &qs, tokens, gate_proj, None)
                .context("mlp_gate from the fused quantised activation")??;
            let u = run_nvfp4_projection_from_quantised(d, &q, &qs, tokens, up_proj, None)
                .context("mlp_up from the fused quantised activation")??;
            return Ok((g, u));
        }
    }
    let normed = d.rms_norm_dev(y, norm_w, tokens, hidden, eps)?;

    let shared_q = if matches!(gate_proj, DeviceProjection::Fp8 { .. })
        && matches!(up_proj, DeviceProjection::Fp8 { .. })
    {
        Some(d.quantise_fp8_dev(&normed, tokens, hidden)?)
    } else {
        None
    };
    let proj = |p: &DeviceProjection| -> Result<Buffer> {
        if let Some((q, qs)) = shared_q.as_ref() {
            if let Some(r) = run_fp8_projection_from_quantised(d, q, qs, tokens, p, None) {
                return r;
            }
        }
        run_projection_dev(d, &normed, tokens, p, None)
    };
    let g = proj(gate_proj)?;
    let u = proj(up_proj)?;
    Ok((g, u))
}

fn fused_norm_quantise_nvfp4(d: &Device, y: &Buffer, weight: &Buffer, tokens: usize, hidden: usize,
                             eps: f32, gate: &DeviceProjection, up: &DeviceProjection)
    -> Result<Option<(Buffer, Buffer)>> {
    let (gg, ug) = match (gate, up) {
        (DeviceProjection::Nvfp4 { input_global_scale: Some(g), cols: gc, .. },
         DeviceProjection::Nvfp4 { input_global_scale: Some(u), cols: uc, .. }) if g == u && gc == uc => {
            (*g, *u)
        }
        _ => return Ok(None),
    };
    let _ = ug;
    d.rms_norm_quantise_nvfp4_dev(y, weight, tokens, hidden, eps, gg,
                                  Device::nvfp4_m_tile_for(tokens))
}

fn run_nvfp4_projection_from_quantised(d: &Device, q: &Buffer, qs: &Buffer, rows: usize,
                                       proj: &DeviceProjection, add: Option<&Buffer>)
    -> Option<Result<Buffer>> {
    match proj {
        DeviceProjection::Nvfp4 { packed, scales, rows: pr, cols, global_scale,
                                  input_global_scale } => {
            let act_global = match input_global_scale {
                Some(g) => *g,
                None => {
                    return Some(Err(anyhow::anyhow!(
                        "an NVFP4 projection of shape [{pr}, {cols}] has no input_global_scale; the \
                         activation cannot be quantised against an assumed one"
                    )))
                }
            };
            Some(match d.nvfp4_gemm_dev(q, qs, packed, scales, rows, *pr, *cols, act_global,
                                        *global_scale) {
                Ok(c) => match add {
                    Some(a) => d.add_dev(a, &c, rows * *pr),
                    None => Ok(c),
                },
                Err(e) => Err(e),
            })
        }
        _ => None,
    }
}

fn swiglu_fused_fp8() -> bool {
    !matches!(std::env::var("CLEF_FUSE_SWIGLU").as_deref(), Ok("0"))
        && !matches!(std::env::var("CLEF_FUSE_SWIGLU_FP8").as_deref(), Ok("0"))
}

fn swiglu_fused_nvfp4() -> bool {
    !matches!(std::env::var("CLEF_FUSE_SWIGLU").as_deref(), Ok("0"))
        && !matches!(std::env::var("CLEF_FUSE_SWIGLU_NVFP4").as_deref(), Ok("0"))
}

fn gdn_prep_fused() -> bool {
    !matches!(std::env::var("CLEF_FUSE_GDN_PREP").as_deref(), Ok("0"))
}

fn conv_prep_fused() -> bool {
    !matches!(std::env::var("CLEF_FUSE_CONV_PREP").as_deref(), Ok("0"))
}

fn norm_dual_fused() -> bool {
    !matches!(std::env::var("CLEF_FUSE_NORM_DUAL").as_deref(), Ok("0"))
}

fn qk_norm_rope_fused() -> bool {
    !matches!(std::env::var("CLEF_FUSE_QK_ROPE").as_deref(), Ok("0"))
}

fn proj_cols(proj: &DeviceProjection) -> usize {
    match proj {
        DeviceProjection::Nvfp4 { cols, .. } => *cols,
        DeviceProjection::Fp8 { cols, .. } => *cols,
        DeviceProjection::Bf16 { cols, .. } => *cols,
    }
}

fn mlp_tail(d: &Device, gate: &Buffer, up: &Buffer, tokens: usize, inter: usize,
            proj: &DeviceProjection, add: Option<&Buffer>) -> Result<Buffer> {
    if swiglu_fused_nvfp4() {
        if let DeviceProjection::Nvfp4 { input_global_scale: Some(g), .. } = proj {
            if let Some((q, qs)) = d.swiglu_quantise_nvfp4_dev(
                gate, up, tokens, inter, *g, Device::nvfp4_m_tile_for(tokens))?
            {
                if let Some(r) = run_nvfp4_projection_from_quantised(d, &q, &qs, tokens, proj, add) {
                    return r;
                }
            }
        }
    }
    if swiglu_fused_fp8() {
        if let DeviceProjection::Fp8 { .. } = proj {
            if let Some((q, qs)) = d.swiglu_quantise_fp8_dev(gate, up, tokens, inter)? {
                if let Some(r) = run_fp8_projection_from_quantised(d, &q, &qs, tokens, proj, add) {
                    return r;
                }
            }
        }
    }
    let act = d.swiglu_dev(gate, up, tokens * inter)?;
    run_projection_dev(d, &act, tokens, proj, add)
}

fn run_projection_dev(d: &Device, x: &Buffer, rows: usize, proj: &DeviceProjection,
                      add: Option<&Buffer>) -> Result<Buffer> {
    match proj {
        DeviceProjection::Nvfp4 { packed, scales, rows: pr, cols: pc, global_scale,
                                  input_global_scale } => {
            let act_global = input_global_scale.with_context(|| {
                format!(
                    "an NVFP4 projection of shape [{pr}, {pc}] has no input_global_scale; the \
                     activation cannot be quantised against an assumed one"
                )
            })?;
            let (q, qs) = d.quantise_nvfp4_dev(x, rows, *pc, act_global,
                                                Device::nvfp4_m_tile_for(rows))?;
            crate::ck!(d, "nvfp4_act", &q, *pc / 2);
            crate::ck!(d, "nvfp4_scale", &qs, *pc / 16);
            let c = d.nvfp4_gemm_dev(&q, &qs, packed, scales, rows, *pr, *pc, act_global,
                                     *global_scale)?;
            crate::ck!(d, "nvfp4_out", &c, *pr);

            match add {
                Some(a) => d.add_dev(a, &c, rows * *pr),
                None => Ok(c),
            }
        }
        DeviceProjection::Fp8 { values, scales, rows: pr, cols: pc } => {
            let (q, qs) = d.quantise_fp8_dev(x, rows, *pc)?;

            if resid_fused(rows) {
                d.fp8_gemm_dev(&q, values, &qs, scales, rows, *pr, *pc, add)
            } else {
                let c = d.fp8_gemm_dev(&q, values, &qs, scales, rows, *pr, *pc, None)?;
                match add {
                    Some(a) => d.add_dev(a, &c, rows * *pr),
                    None => Ok(c),
                }
            }
        }
        DeviceProjection::Bf16 { values, rows: pr, cols: pc } => {
            let a = d.to_bf16_dev(x, rows * pc)?;
            let c = d.bf16_gemm_dev(&a, values, rows, *pr, *pc)?;
            match add {
                Some(r) => d.add_dev(r, &c, rows * *pr),
                None => Ok(c),
            }
        }
    }
}

fn run_bf16_projection_from_converted(d: &Device, a: &Buffer, rows: usize, cols: usize,
                                      proj: &DeviceProjection, add: Option<&Buffer>)
    -> Option<Result<Buffer>> {
    match proj {
        DeviceProjection::Bf16 { values, rows: pr, .. } => Some((|| {
            let c = d.bf16_gemm_dev(a, values, rows, *pr, cols)?;
            match add {
                Some(r) => d.add_dev(r, &c, rows * *pr),
                None => Ok(c),
            }
        })()),
        _ => None,
    }
}

pub const GRAPH_BUCKET: usize = 16;

pub fn max_graph_tokens() -> usize {
    return crate::gpu::max_batch_tokens();
}

#[derive(Debug, Clone, Default)]
pub struct WarmReport {
    pub captured: usize,
    pub requested: usize,
    pub failed: Vec<(usize, String)>,
}

fn bytemuck_i32_back(values: Vec<f32>, n: usize) -> Vec<i32> {
    values.iter().take(n).map(|v| v.to_bits() as i32).collect()
}

fn as_refused(e: anyhow::Error, tokens: usize) -> anyhow::Error {
    match e.downcast_ref::<crate::cuda::CaptureTooLarge>() {
        Some(over) => anyhow::Error::new(crate::engine::Refused {
            tokens,
            why: over.to_string(),
        }),
        None => e,
    }
}

pub fn graph_memory_budget(device: &Device) -> usize {
    if let Ok(v) = std::env::var("CLEF_GRAPH_BUDGET") {
        if let Ok(n) = v.parse::<usize>() {
            return n;
        }
    }

    let free = device.free_memory().unwrap_or(0);

    let total = device.total_memory().unwrap_or(0);
    if total > 0 {
        return total / 2;
    }

    const RESERVE: usize = 4 * 1024 * 1024 * 1024;
    free.saturating_sub(RESERVE).max(512 * 1024 * 1024)
}

pub fn graph_size_for(tokens: usize) -> Option<usize> {
    if tokens == 0 || tokens > max_graph_tokens() {
        return None;
    }

    if let Some(fixed) = fixed_graph_size() {
        if tokens <= fixed {
            return Some(fixed);
        }
    }
    Some(crate::gpu::pad_to_chunk(tokens))
}

#[derive(Clone, Debug)]
pub struct Rungs {
    list: Vec<usize>,
}

pub const HEAD_RUNGS: [usize; 8] = [64, 128, 256, 512, 1024, 2048, 4096, 8192];

static ABSORB_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn graph_key(run: usize, mode: GraphMode, _slot: usize) -> (usize, GraphMode) {
    let _ = mode;
    let splits = crate::cuda::splits_hint();
    (run | (splits.min(0xFF) << 24), GraphMode::Chunked)
}

static PASS_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

static DN_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

static STAGE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Rungs {
    pub fn head() -> Self { Rungs { list: HEAD_RUNGS.to_vec() } }

    pub fn body() -> Self { Rungs::body_at(max_chunk_size()) }

    pub fn body_at(top: usize) -> Self {
        if let Some(r) = std::env::var("CLEF_FORCE_RUNG").ok().and_then(|v| v.parse::<usize>().ok()) {
            let r = r.max(crate::cuda::CHUNK_GRANULE);
            return Rungs { list: vec![r] };
        }
        let top = top.max(crate::cuda::CHUNK_GRANULE);

        let mut list: Vec<usize> = Vec::with_capacity(HEAD_RUNGS.len());
        for r in HEAD_RUNGS {
            if r <= top { list.push(r); }
        }

        let slots = kv_slots().max(1);
        for k in 2..=slots {
            for r in HEAD_RUNGS {
                let total = k * r;
                if total <= top { list.push(total); }
            }
        }
        if list.is_empty() || *list.last().unwrap() < top { list.push(top); }
        list.sort_unstable();
        list.dedup();
        Rungs { list }
    }

    pub fn list(&self) -> &[usize] { &self.list }
    pub fn smallest(&self) -> usize { *self.list.first().expect("a ladder is non-empty") }
    pub fn largest(&self) -> usize { *self.list.last().expect("a ladder is non-empty") }
    pub fn contains(&self, r: usize) -> bool { self.list.contains(&r) }

    pub fn at_or_above(&self, need: usize) -> usize {
        self.list.iter().copied().find(|r| *r >= need).unwrap_or_else(|| self.largest())
    }

    pub fn at_or_below(&self, need: usize) -> usize {
        self.list.iter().copied().filter(|r| *r <= need).next_back().unwrap_or_else(|| self.smallest())
    }

    pub fn chunk_for(&self, remaining: usize) -> usize {
        self.cut(remaining)
    }

    fn cut(&self, remaining: usize) -> usize {
        if remaining <= self.smallest() { return self.smallest(); }

        let up = self.at_or_above(remaining);
        if up > remaining && remaining * 2 > up {
            return up;
        }
        self.at_or_below(remaining)
    }

    pub fn decompose(&self, tokens: usize) -> Vec<(usize, usize)> {
        let smallest = self.smallest();
        let mut out = Vec::new();
        let mut base = 0usize;
        while base < tokens {
            let rem = tokens - base;
            let rows = self.list.iter().copied().filter(|r| *r <= rem).max().unwrap_or(smallest);
            out.push((base, rows));
            base += rows;
        }
        out
    }
}

pub fn pass_rungs() -> Vec<usize> { Rungs::body().list().to_vec() }

pub fn pass_geometries() -> Vec<usize> {
    let rungs = Rungs::body();
    let top = max_chunk_size();
    let slots = kv_slots().max(1);
    let mut out: Vec<usize> = rungs.list().to_vec();
    for k in 2..=slots {
        for &r in rungs.list() {
            let total = k * r;
            if total > top && total <= k * top && total <= max_graph_tokens() {
                out.push(total);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

pub fn fit_chunks(want: &[usize]) -> Vec<usize> {
    let granule = crate::cuda::CHUNK_GRANULE;

    let floor = |w: usize| if w == 0 { 0 } else { w.max(granule) / granule * granule };
    let want: Vec<usize> = want.iter().map(|&w| floor(w)).collect();
    let geoms = pass_geometries();
    if want.len() <= 1 {
        let w = want.first().copied().unwrap_or(0);
        if w == 0 { return want; }
        return geoms.iter().copied().find(|&g| g >= w).map(|g| vec![g]).unwrap_or(want);
    }
    let sum: usize = want.iter().sum();
    let Some(target) = geoms.iter().copied().filter(|&g| g <= sum).next_back() else {
        return want;
    };
    if target == sum {
        return want;
    }

    let needed = want.len() * granule;
    let target = if target < needed {
        match geoms.iter().copied().find(|&g| g >= needed) {
            Some(g) => g,
            None => return want,
        }
    } else {
        target
    };

    let units = |level: usize| -> usize { want.iter().map(|&w| w.min(level)).sum() };
    let mut level = *want.iter().max().expect("a pass has at least one chunk");
    while level > 0 && units(level) > target {
        level = level.saturating_sub(granule);
    }
    let mut out: Vec<usize> = want.iter().map(|&w| w.min(level)).collect();

    let mut left = target.saturating_sub(out.iter().sum::<usize>());
    let mut order: Vec<usize> = (0..want.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(want[i]));
    for &i in &order {
        if left == 0 { break; }
        let give = ((want[i] - out[i]) / granule * granule).min(left);
        out[i] += give;
        left -= give;
    }

    if left > 0 {
        if let Some(&i) = order.first() {
            out[i] += left;
        }
    }
    out
}

pub fn pass_rung(need: usize) -> usize { Rungs::body().at_or_above(need) }

pub fn pass_plan(tokens: usize, chunk: usize) -> Vec<(usize, usize)> {
    let granule = crate::cuda::CHUNK_GRANULE;
    let cap = Rungs::body().at_or_below(chunk.max(granule));
    let mut pos = 0usize;
    let mut plan: Vec<(usize, usize)> = Vec::new();
    while pos < tokens {
        let rem = tokens - pos;
        let rows = next_chunk_rows(rem).min(cap);
        let real = rows.min(rem);
        plan.push((pos, pass_rung(real).max(real.div_ceil(granule) * granule)));
        pos += real;
    }
    plan
}

pub fn pass_rows_for(tokens: usize, chunk: usize) -> usize {
    pass_plan(tokens, chunk).iter().map(|(_, r)| *r).sum()
}

pub fn next_chunk_rows(remaining: usize) -> usize { Rungs::body().chunk_for(remaining) }

pub fn max_chunk_size() -> usize {
    let granule = crate::cuda::CHUNK_GRANULE;

    let want = std::env::var("CLEF_MAX_CHUNK")
        .or_else(|_| std::env::var("CLEF_MAX_PASS_TOKENS"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)

        .unwrap_or(8192);
    want.div_ceil(granule) * granule
}

pub fn is_pass_geometry(rows: usize) -> bool {
    let r = pass_rung(rows);
    if rows == r {
        return true;
    }

    rows > r && rows % r == 0 && rows <= kv_slots().max(1) * max_chunk_size()
}

pub fn max_batch_tokens() -> usize {
    let granule = crate::cuda::CHUNK_GRANULE;
    crate::engine::max_input_tokens().div_ceil(granule) * granule
}

pub const MAX_CHUNKS: usize = 512;

pub struct ChunkSpec<'a> {
    pub ids: &'a [u32],
    pub pos: usize,
    pub slot: usize,
    pub rows: usize,
}

pub struct PassMaps {
    pub ids: Vec<u32>,
    pub seq_start: Vec<i32>,
    pub seq_end: Vec<i32>,
    pub cpos: Vec<i32>,
    pub cbase: Vec<i32>,
    pub slot_map: Vec<i32>,
    pub rope_pos: Vec<i32>,
    pub chunk_start: Vec<i32>,
    pub state_slot: Vec<i32>,
    pub hist_save: Vec<(u32, u32)>,
    pub cache_len: i32,
    pub rows_real: Vec<usize>,
    pub rows_pad: Vec<usize>,
    pub offs: Vec<usize>,
    pub total: usize,
}

pub fn pass_layout(chunks: &[ChunkSpec<'_>]) -> (Vec<usize>, Vec<usize>, usize) {
    let granule = crate::cuda::CHUNK_GRANULE;
    let rows_pad: Vec<usize> = chunks.iter().map(|x| x.rows).collect();
    let mut offs: Vec<usize> = Vec::with_capacity(rows_pad.len());
    let mut acc = 0usize;
    for &r in &rows_pad {
        assert!(r > 0 && r % granule == 0,
                "a chunk of {r} rows is not a positive granule multiple");
        offs.push(acc);
        acc += r;
    }
    assert!(
        pass_geometries().binary_search(&acc).is_ok(),
        "a pass of {acc} rows is not a graphed geometry ({:?}) — it would run EAGER. Fit the chunks with \
         `fit_chunks` before building the pass.",
        chunks.iter().map(|x| x.rows).collect::<Vec<_>>()
    );
    (rows_pad, offs, acc)
}

pub fn pass_maps(chunks: &[ChunkSpec<'_>], capacity: usize) -> PassMaps {
    let granule = crate::cuda::CHUNK_GRANULE;
    let cap = capacity as i32;
    let (rows_pad, offs, total) = pass_layout(chunks);
    let rows_real: Vec<usize> = chunks
        .iter()
        .zip(rows_pad.iter())
        .map(|(x, &pad)| pad.min(x.ids.len().saturating_sub(x.pos)))
        .collect();

    let mut m = PassMaps {
        ids: Vec::with_capacity(total),
        seq_start: Vec::with_capacity(total),
        seq_end: Vec::with_capacity(total),
        cpos: Vec::with_capacity(total),
        cbase: Vec::with_capacity(total),
        slot_map: Vec::with_capacity(total),
        rope_pos: Vec::with_capacity(total),
        chunk_start: Vec::with_capacity(total / granule),
        state_slot: Vec::with_capacity(total / granule),
        hist_save: Vec::with_capacity(chunks.len()),
        cache_len: i32::MAX,
        rows_real,
        rows_pad,
        offs,
        total,
    };
    for (i, x) in chunks.iter().enumerate() {
        let (real, pad) = (m.rows_real[i], m.rows_pad[i]);
        let s = x.pos;
        m.ids.extend_from_slice(&x.ids[s..s + real]);

        if pad > real {
            let last = *x.ids.last().expect("a request has at least one token");
            m.ids.extend(std::iter::repeat(last).take(pad - real));
        }
        let base = x.slot as i32 * cap;
        for t in 0..pad {
            m.seq_start.push(m.offs[i] as i32);

            m.seq_end.push((m.offs[i] + real) as i32);
            m.cpos.push(base + x.pos as i32 + t as i32);

            m.cbase.push(base);
            m.slot_map.push(x.slot as i32);
            m.rope_pos.push(x.pos as i32 + t as i32);
        }
        m.cache_len = m.cache_len.min(base + x.pos as i32);
        m.hist_save.push(((m.offs[i] + real - 1) as u32, x.slot as u32));
    }
    for (i, x) in chunks.iter().enumerate() {
        for k in 0..(m.rows_pad[i] / granule) {
            m.chunk_start.push(if k == 0 && i > 0 { 1 } else { 0 });
            m.state_slot.push(x.slot as i32);
        }
    }
    if chunks.is_empty() {
        m.cache_len = 0;
    }
    for (i, a) in chunks.iter().enumerate() {
        for b in &chunks[i + 1..] {
            assert_ne!(a.slot, b.slot, "two chunks of one pass share slot {} — their keys would alias", a.slot);
        }
    }
    m
}

pub fn is_long_request(request: &crate::types::SystemOneRequest) -> bool {
    let chars = request.state.len();
    chars > max_chunk_size()
}

pub fn chunking_enabled() -> bool {
    max_chunk_size() < crate::engine::max_input_tokens()
}

pub fn max_chunked_tokens() -> usize {
    (MAX_CHUNKS * max_chunk_size()).min(crate::engine::max_input_tokens())
}

pub fn kv_slots() -> usize {
    std::env::var("CLEF_KV_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)

        .unwrap_or(1)
}

pub fn kv_capacity() -> usize {
    let cap = std::env::var("CLEF_KV_CAPACITY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| crate::engine::max_input_tokens().min(max_chunked_tokens()));
    cap.div_ceil(crate::cuda::CHUNK_GRANULE) * crate::cuda::CHUNK_GRANULE
}

pub fn head_staging_rows() -> usize {
    kv_capacity().max(*HEAD_RUNGS.last().expect("the head's ladder is non-empty"))
}

pub fn pad_to_chunk(tokens: usize) -> usize {
    tokens.div_ceil(crate::cuda::CHUNK_LEN) * crate::cuda::CHUNK_LEN
}

pub fn graph_covers(tokens: usize) -> bool {
    tokens > 0 && tokens <= max_graph_tokens()
}

pub fn fixed_graph_size() -> Option<usize> {
    std::env::var("CLEF_GRAPH_FIXED_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
}

pub fn bucket_for(tokens: usize) -> usize {
    graph_size_for(tokens).unwrap_or(tokens)
}

struct GraphIo {
    input: Buffer,
    output: Buffer,
    values: usize,
}

struct GraphMaps {
    seq_start: Buffer,
    seq_end: Buffer,
    chunk_start: Buffer,
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

struct GraphedLayers {
    captured: crate::cuda::Captured,
    input: Buffer,
    output: Buffer,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GraphMode {
    Chunked,
}

pub struct PassCarry {
    gdn_state: Vec<Option<Buffer>>,
    state_scratch: Vec<Option<Buffer>>,
    conv_hist: Vec<Option<Buffer>>,
    kv: Vec<Option<(Buffer, Buffer)>>,
    cache_len: Buffer,
    cache_base: Buffer,
    slot_map: Buffer,
    state_slot: Buffer,
    cpos: Buffer,
    cbase: Buffer,
    rope_pos: Buffer,
    state_bytes: usize,
    hist_bytes: usize,
    pub capacity: usize,
    pub slots: usize,
    slot: std::sync::atomic::AtomicUsize,
    real_len: std::sync::atomic::AtomicUsize,
    seq_start: Buffer,
    seq_end: Buffer,
    chunk_start: Buffer,
    real_end: Buffer,
    real_walk: Buffer,
    hist_save: Buffer,
}

pub struct PinnedMapStaging {
    cpos: crate::cuda::PinnedBuf,
    rope_pos: crate::cuda::PinnedBuf,
}

impl PassCarry {
    fn views(&self) -> Vec<crate::cuda::LayerCarry<'_>> {
        let slot = self.slot.load(std::sync::atomic::Ordering::Relaxed);

        let on = |name: &str| std::env::var(name).is_err();
        let (state_on, hist_on, cache_on) = (
            on("CLEF_CHUNK_NO_STATE"),
            on("CLEF_CHUNK_NO_HIST"),
            on("CLEF_CHUNK_NO_CACHE"),
        );
        (0..self.gdn_state.len())
            .map(|li| crate::cuda::LayerCarry {
                gdn_state: if state_on {
                    self.gdn_state[li].as_ref().map(|b| {
                        if std::env::var("CLEF_STATE_SCRATCH").is_ok() {
                            match self.state_scratch.get(li).and_then(|x| x.as_ref()) {
                                Some(sc) => (b, sc),
                                None => (b, b),
                            }
                        } else {
                            (b, b)
                        }
                    })
                } else { None },
                conv_hist: if hist_on { self.conv_hist[li].as_ref() } else { None },
                kv: if cache_on {
                    self.kv[li].as_ref().map(|(k, v)| crate::cuda::KvCache {
                        k,
                        v,
                        len: &self.cache_len,
                        pos_map: Some(&self.cpos),
                        base_map: Some(&self.cbase),
                        slot_map: Some(&self.slot_map),
                        capacity: self.capacity,
                        positions: self.capacity * self.slots,
                    })
                } else {
                    None
                },
                cache_len: Some(&self.cache_len),
                slot_map: Some(&self.slot_map),
                rope_pos: Some(&self.rope_pos),
                slot: if std::env::var("CLEF_CARRY_SLOT0").is_ok() {
                    0
                } else {
                    self.slot.load(std::sync::atomic::Ordering::Relaxed)
                },
                real_len: if std::env::var("CLEF_CARRY_REALLEN0").is_ok() {
                    0
                } else {
                    self.real_len.load(std::sync::atomic::Ordering::Relaxed)
                },
                seq_start: Some(&self.seq_start),
                seq_end: Some(&self.seq_end),
                chunk_start: Some(&self.chunk_start),
                real_end: Some(&self.real_end),
                real_walk: Some(&self.real_walk),
                hist_save: Some(&self.hist_save),
                hist_slots: self.slots,
            })
            .collect()
    }

    fn reset_slot(&self, d: &Device, slot: usize) -> Result<()> {
        for b in self.gdn_state.iter().flatten() {
            d.zero_range(b, slot * self.state_bytes, self.state_bytes)?;
        }
        for b in self.conv_hist.iter().flatten() {
            d.zero_range(b, slot * self.hist_bytes, self.hist_bytes)?;
        }
        d.zero_fill(&self.cache_len)?;
        d.zero_fill(&self.cache_base)?;
        Ok(())
    }

    fn reset(&self, d: &Device) -> Result<()> {
        for b in self.gdn_state.iter().flatten() {
            d.zero_fill(b)?;
        }
        for b in self.conv_hist.iter().flatten() {
            d.zero_fill(b)?;
        }
        d.zero_fill(&self.cache_len)?;
        d.zero_fill(&self.cache_base)?;
        Ok(())
    }
}

pub struct GpuRunner {
    pub device: Device,
    pub backbone: DeviceBackbone,
    graphs: std::sync::Mutex<std::collections::HashMap<(usize, GraphMode), GraphedLayers>>,
    graph_maps: std::sync::Mutex<Option<GraphMaps>>,
    params: Buffer,
    graph_io: std::sync::Mutex<Option<GraphIo>>,
    pending: std::sync::Mutex<std::collections::VecDeque<(usize, GraphMode, usize)>>,
    capturing_pass: std::sync::atomic::AtomicBool,
    chunked_passes: std::sync::atomic::AtomicUsize,
    chunked_width_max: std::sync::atomic::AtomicUsize,
    chunked_passes_sharing: std::sync::atomic::AtomicUsize,
    calib: std::sync::Mutex<Vec<(u64, u64)>>,
    rope: std::sync::Mutex<std::collections::HashMap<usize, (Buffer, Buffer, usize, usize)>>,
    carry: std::sync::Mutex<Option<std::sync::Arc<PassCarry>>>,
}

thread_local! {
    static TRACE_A: std::cell::RefCell<Vec<(String, Vec<f32>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl GpuRunner {
    pub fn load(checkpoint: &Checkpoint, config: &TextConfig) -> Result<Self> {
        let device = Device::open()?;

        device.warm_constants()?;
        let backbone = load_backbone(&device, checkpoint, config)?;

        let params = device.alloc_zeroed(8 * 4)?;
        Ok(Self {
            device,
            backbone,
            graphs: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph_maps: std::sync::Mutex::new(None),
            params,
            graph_io: std::sync::Mutex::new(None),
            pending: std::sync::Mutex::new(std::collections::VecDeque::new()),
            capturing_pass: std::sync::atomic::AtomicBool::new(false),
            chunked_passes: std::sync::atomic::AtomicUsize::new(0),
            chunked_width_max: std::sync::atomic::AtomicUsize::new(0),
            chunked_passes_sharing: std::sync::atomic::AtomicUsize::new(0),
            calib: std::sync::Mutex::new(Vec::new()),
            rope: std::sync::Mutex::new(std::collections::HashMap::new()),
            carry: std::sync::Mutex::new(None),
        })
    }

    pub fn delta_net_layer_dev(&self, x: &Buffer, tokens: usize, w: &DeviceDeltaNet,
                               map: crate::cuda::BatchMap<'_>)
        -> Result<Buffer> {
        Ok(self.delta_net_layer_inner(x, tokens, w, false, map)?.0)
    }

    fn delta_net_layer_inner(&self, x: &Buffer, tokens: usize, w: &DeviceDeltaNet, trace: bool,
                             map: crate::cuda::BatchMap<'_>)
        -> Result<(Buffer, Option<(Vec<f32>, Vec<f32>)>)> {
        let d = &self.device;

        let tracing = std::env::var("CLEF_LAYER_TRACE").is_ok();
        let mut laps: Vec<(&'static str, f64)> = Vec::new();
        let lap = |name: &'static str, t0: std::time::Instant, laps: &mut Vec<(&'static str, f64)>| {
            if tracing {
                laps.push((name, t0.elapsed().as_secs_f64()));
            }
        };
        let c = &self.backbone.config;
        let hidden = c.hidden_size;
        let eps = c.rms_norm_eps;
        let conv_dim = c.conv_dim();

        let _t = std::time::Instant::now();

        let dn_dir = std::env::var("CLEF_DN_DUMP").ok();
        let dn_tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
        let dn_seq = DN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dn_stage = |name: &str, buf: &Buffer, width: usize| {
            if let Some(dir) = dn_dir.as_ref() {
                if let Ok(v) = d.download_f32(buf) {
                    let n = (tokens * width).min(v.len());
                    let bytes: Vec<u8> = v[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(format!("{dir}/{dn_seq:03}_{dn_tag}{name}.f32.bin"), &bytes);
                }
            }
        };

        let fp8_pair = matches!(w.in_proj_qkv, DeviceProjection::Fp8 { .. })
            && matches!(w.in_proj_z, DeviceProjection::Fp8 { .. });
        let bf16_pair = matches!(w.in_proj_a, DeviceProjection::Bf16 { .. })
            && matches!(w.in_proj_b, DeviceProjection::Bf16 { .. });
        let dual = fp8_pair && bf16_pair && norm_dual_fused()
            && !crate::gpu::gdn_trace_on() && dn_dir.is_none();
        let mut fused_norm = if dual {
            d.rms_norm_fp8_bf16_dev(x, &w.input_layernorm, tokens, hidden, eps)
        } else {
            None
        };
        let normed = match fused_norm {
            Some(_) => None,
            None => Some(d.rms_norm_dev(x, &w.input_layernorm, tokens, hidden, eps)?),
        };
        if let Some(n) = normed.as_ref() {
            dn_stage("normed", n, hidden);
            ck!(d, "rms_norm", n, hidden);
        }

        let f32_norm = |what: &str| -> &Buffer {
            normed.as_ref().unwrap_or_else(|| {
                panic!("{what} needs the f32 norm, and the fused path does not produce one")
            })
        };
        lap("input_layernorm", _t, &mut laps);
        let _t = std::time::Instant::now();

        let mut pending_bf: Option<Buffer> = None;
        let shared_q = match fused_norm.take() {
            Some((q, qs, bf)) => {
                pending_bf = Some(bf);
                Some((q, qs))
            }
            None if fp8_pair => Some(d.quantise_fp8_dev(f32_norm("the FP8 hoist"), tokens, hidden)?),
            None => None,
        };
        let mixed = match &shared_q {
            Some((q, qs)) => match run_fp8_projection_from_quantised(d, q, qs, tokens,
                                                                    &w.in_proj_qkv, None) {
                Some(r) => r?,
                None => run_projection_dev(d, f32_norm("in_proj_qkv"), tokens, &w.in_proj_qkv, None)?,
            },
            None => run_projection_dev(d, f32_norm("in_proj_qkv"), tokens, &w.in_proj_qkv, None)?,
        };
        ck!(d, "in_proj_qkv", &mixed, hidden);
        lap("in_proj_qkv", _t, &mut laps);
        let _t = std::time::Instant::now();

        let carry = map.carry;
        let conv_hist = carry.and_then(|c| c.conv_hist);
        let conv_in = mixed;

        let kd = c.linear_key_head_dim;
        let vd = c.linear_value_head_dim;
        let kh = c.linear_num_key_heads;
        let vh = c.linear_num_value_heads;
        let ratio = c.value_heads_per_key_head();
        let fused_prep = gdn_use_chunked() && gdn_prep_fused()
            && kh % 4 == 0 && (vh * kd) % 4 == 0 && kd == 128;
        let fused_conv_prep = fused_prep && conv_prep_fused() && !trace && dn_dir.is_none()
            && conv_dim % 1024 == 0;

        let fused_out = if fused_conv_prep {
            Some(d.gdn_conv_prep_dev(&conv_in, &w.conv_weight, tokens, kh, kd, vh, conv_dim,
                                     c.linear_conv_kernel_dim, map.seq_start, conv_hist,
                                     carry.and_then(|c| c.slot_map))?)
        } else {
            None
        };
        let conv_out = if fused_conv_prep {
            None
        } else {
            Some(d.causal_conv_silu_dev(&conv_in, &w.conv_weight, tokens, conv_dim,
                                        c.linear_conv_kernel_dim, map.seq_start, conv_hist,
                                        carry.and_then(|c| c.slot_map))?)
        };
        if let Some(hist) = conv_hist {
            let k = c.linear_conv_kernel_dim;
            let bytes = (k - 1) * conv_dim * 4;
            match (map.hist_save_dev, map.hist_save) {
                (Some(dev), _) if map.hist_save_n > 0 => {
                    d.conv_hist_save_dev(&conv_in, hist, dev, map.hist_save_n, tokens, k, conv_dim)?;
                }
                _ => match map.hist_save {
                Some(saves) => {
                    for &(row, slot) in saves {
                        let src = (row as usize + 1 - (k - 1)) * conv_dim * 4;
                        if src + bytes <= conv_in.len() {
                            d.copy_bytes_at(hist, slot as usize * bytes, &conv_in, src, bytes)?;
                        }
                    }
                }
                None => {
                    if tokens >= k - 1 {
                        let slot = carry.map(|c| c.slot).unwrap_or(0);
                        d.copy_bytes_at(hist, slot * bytes, &conv_in,
                                        (tokens - (k - 1)) * conv_dim * 4, bytes)?;
                    }
                }
                },
            }
        }
        if let Some(m) = conv_out.as_ref() {
            ck!(d, "conv_silu", m, hidden);
            dn_stage("conv_out", m, conv_dim);
        }
        lap("conv", _t, &mut laps);
        let (prep, q, k, v) = if let Some(p) = fused_out {
            (Some(p), None, None, None)
        } else if fused_prep {
            let m = conv_out.as_ref().expect("the chain ran when the fusion did not");
            (Some(d.gdn_prep_dev(m, tokens, kh, kd, vh, conv_dim)?), None, None, None)
        } else {
            let m = conv_out.as_ref().expect("the chain ran when the fusion did not");
            let (q, k, v) = d.split_qkv_dev(m, tokens, kh, kd, vh, vd)?;

            let q = d.repeat_interleave_dev(&q, tokens, kh, ratio, kd)?;
            let k = d.repeat_interleave_dev(&k, tokens, kh, ratio, kd)?;
            (None, Some(q), Some(k), Some(v))
        };

        let _t = std::time::Instant::now();
        let z = match &shared_q {
            Some((q, qs)) => match run_fp8_projection_from_quantised(d, q, qs, tokens,
                                                                    &w.in_proj_z, None) {
                Some(r) => r?,
                None => run_projection_dev(d, f32_norm("in_proj_z"), tokens, &w.in_proj_z, None)?,
            },
            None => run_projection_dev(d, f32_norm("in_proj_z"), tokens, &w.in_proj_z, None)?,
        };
        ck!(d, "in_proj_z", &z, hidden);
        lap("in_proj_z", _t, &mut laps);

        let shared_bf = match pending_bf.take() {
            Some(bf) => Some(bf),
            None if bf16_pair => Some(d.to_bf16_dev(f32_norm("the BF16 hoist"), tokens * hidden)?),
            None => None,
        };
        let proj_ab = |proj: &DeviceProjection| -> Result<Buffer> {
            if let Some(bf) = shared_bf.as_ref() {
                if let Some(r) = run_bf16_projection_from_converted(d, bf, tokens, hidden, proj, None) {
                    return r;
                }
            }
            run_projection_dev(d, f32_norm("in_proj_a/b"), tokens, proj, None)
        };
        let a = proj_ab(&w.in_proj_a)?;
        ck!(d, "in_proj_a", &a, hidden);
        let b = proj_ab(&w.in_proj_b)?;
        ck!(d, "in_proj_b", &b, hidden);
        let (decay, beta) = d.gdn_gates_dev(&a, &b, &w.a_log, &w.dt_bias, tokens, vh)?;
        ck!(d, "gates_decay", &decay, hidden);

        let _t = std::time::Instant::now();
        let o = if gdn_use_chunked() {
            let state = carry.and_then(|c| c.gdn_state);

            let forced_real = std::env::var("CLEF_REAL_LEN").as_deref() == Ok("tokens");
            let real = if forced_real { tokens } else { carry.map(|c| c.real_len).filter(|&r| r > 0)
                .or_else(|| if map.real_end > 0 && map.real_end < tokens { Some(map.real_end) } else { None })
                .unwrap_or(tokens) };
            match &prep {
                Some(p) => d.gdn_chunk_scan_prepared_state_at_dev(p, &decay, &beta, tokens, real,
                                                                  vh, vd, map, state)?,
                None => d.gdn_chunk_scan_state_at_dev(q.as_ref().expect("the chain produced q"),
                                                      k.as_ref().expect("the chain produced k"),
                                                      v.as_ref().expect("the chain produced v"),
                                                      &decay, &beta, tokens, real, vh, kd, vd,
                                                      map, state)?,
            }
        } else {
            d.gdn_scan_dev(q.as_ref().expect("the chain produced q"),
                           k.as_ref().expect("the chain produced k"),
                           v.as_ref().expect("the chain produced v"),
                           &decay, &beta, tokens, vh, kd, vd)?
        };
        ck!(d, "scan_out", &o, hidden);
        lap("gdn_scan", _t, &mut laps);

        let _t = std::time::Instant::now();
        let gated_q = if trace {
            None
        } else {
            gated_norm_quantised(d, &o, &z, &w.norm_weight, tokens, vh, vd, eps, &w.out_proj)?
        };
        let (gated, y) = if let Some((q, qs)) = gated_q.as_ref() {
            let y = run_nvfp4_projection_from_quantised(d, q, qs, tokens, &w.out_proj, Some(x))
                .or_else(|| run_fp8_projection_from_quantised(d, q, qs, tokens, &w.out_proj, Some(x)))
                .expect("the gated norm only quantises for a projection kind that consumes it")?;
            (None, y)
        } else {
            let gated = d.rms_norm_gated_dev(&o, &z, &w.norm_weight, tokens * vh, vd, eps)?;

            ck!(d, "gated_norm", &gated, vh * vd);
            let y = run_projection_dev(d, &gated, tokens, &w.out_proj, Some(x))?;
            (Some(gated), y)
        };
        lap("gated_norm+quantise", _t, &mut laps);
        let _ = gated;
        let _t = std::time::Instant::now();
        lap("out_proj", _t, &mut laps);
        ck!(d, "out_proj", &y, hidden);

        let inter = c.intermediate_size;
        let _t = std::time::Instant::now();

        let gated = if trace {
            None
        } else {
            mlp_gated(d, &y, &w.post_attention_layernorm, tokens, hidden, eps,
                      &w.mlp_gate, &w.mlp_up, &w.mlp_down)?
        };
        let (gate, up) = if gated.is_some() {
            lap("mlp_gate_up_gated", _t, &mut laps);
            (None, None)
        } else {
            let (g, u) = mlp_gate_up(d, &y, &w.post_attention_layernorm, tokens, hidden, eps,
                                     &w.mlp_gate, &w.mlp_up, trace)?;
            ck!(d, "mlp_gate", &g, hidden);
            lap("mlp_gate", _t, &mut laps);
            let _t = std::time::Instant::now();
            ck!(d, "mlp_up", &u, hidden);
            lap("mlp_up", _t, &mut laps);
            (Some(g), Some(u))
        };
        let _t = std::time::Instant::now();

        let mlp_down_is_fp8 = matches!(w.mlp_down, DeviceProjection::Fp8 { .. });
        let mlp_down_is_nvfp4 = matches!(w.mlp_down, DeviceProjection::Nvfp4 { .. });
        let (act, down) = if let Some((q, qs)) = gated.as_ref() {
            let down = run_nvfp4_projection_from_quantised(d, q, qs, tokens, &w.mlp_down, Some(&y))
                .or_else(|| run_fp8_projection_from_quantised(d, q, qs, tokens, &w.mlp_down, Some(&y)))
                .expect("the gated path is only taken when the down-projection consumes this form")?;
            (None, down)
        } else {
        let (gate, up) = (gate.as_ref().expect("the chain produced the pair"),
                          up.as_ref().expect("the chain produced the pair"));
        let fused = if trace {
            None
        } else if mlp_down_is_fp8 && swiglu_fused_fp8() {
            match d.swiglu_quantise_fp8_dev(&gate, &up, tokens, inter)? {
                Some((q, qs)) => {
                    if std::env::var("CLEF_GDN_TRACE").is_ok() {
                        let raw = d.download(&q)?;
                        eprintln!("    GDN {:<12} {:?}", "quant_act", raw[..8.min(raw.len())].to_vec());
                    }
                    run_fp8_projection_from_quantised(d, &q, &qs, tokens, &w.mlp_down, Some(&y))
                }

                None => None,
            }
        } else if mlp_down_is_nvfp4 && swiglu_fused_nvfp4() {
            let act_global = match &w.mlp_down {
                DeviceProjection::Nvfp4 { input_global_scale, .. } => *input_global_scale,
                _ => None,
            };
            match act_global {
                Some(g) => match d.swiglu_quantise_nvfp4_dev(&gate, &up, tokens, inter, g,
                                                             Device::nvfp4_m_tile_for(tokens))? {
                    Some((q, qs)) => {
                        if std::env::var("CLEF_GDN_TRACE").is_ok() {
                            let raw = d.download(&q)?;
                            eprintln!(
                                "    GDN {:<12} {:?}",
                                "quant_act",
                                raw[..8.min(raw.len())].to_vec()
                            );
                        }
                        run_nvfp4_projection_from_quantised(d, &q, &qs, tokens,
                                                            &w.mlp_down, Some(&y))
                    }
                    None => None,
                },
                None => None,
            }
        } else {
            None
        };
        match fused {
            Some(r) => (None, r?),
            None => {
                let act = d.swiglu_dev(gate, up, tokens * inter)?;

                ck!(d, "swiglu", &act, inter);
                let down = run_projection_dev(d, &act, tokens, &w.mlp_down,
                                              if trace { None } else { Some(&y) })?;
                (Some(act), down)
            }
        }
        };
        lap("swiglu+mlp_down", _t, &mut laps);

        let traced = if trace {
            let act = act
                .as_ref()
                .expect("the trace opts out of the swiglu fusion, so `act` exists");
            Some((d.download_rows_f32(act, tokens, inter)?,
                  d.download_rows_f32(&down, tokens, hidden)?))
        } else {
            None
        };

        ck!(d, "layer_out", &down, hidden);
        let out = if trace { d.add_dev(&y, &down, tokens * hidden)? } else { down };
        if tracing {
            let total: f64 = laps.iter().map(|(_, t)| t).sum();
            for (name, t) in &laps {
                eprintln!("    {name:18} {:8.3} ms  {:5.1}%", t * 1e3, 100.0 * t / total);
            }
            eprintln!("    {:<18} {:8.3} ms", "SUM", total * 1e3);
        }
        Ok((out, traced))
    }

    pub fn delta_net_layer(&self, x: &[f32], tokens: usize, w: &DeviceDeltaNet) -> Result<Vec<f32>> {
        let dev = self.device.upload_f32(x)?;
        let out = self.delta_net_layer_dev(&dev, tokens, w, crate::cuda::BatchMap::SINGLE)?;
        self.device.download_f32(&out)
    }

    pub fn delta_net_layer_traced(
        &self,
        x: &[f32],
        tokens: usize,
        w: &DeviceDeltaNet,
        trace: Option<&mut Option<(Vec<f32>, Vec<f32>)>>,
    ) -> Result<Vec<f32>> {
        let dev = self.device.upload_f32(x)?;
        let (out, traced) = self.delta_net_layer_inner(&dev, tokens, w, trace.is_some(),
                                                       crate::cuda::BatchMap::SINGLE)?;
        if let Some(slot) = trace {
            *slot = traced;
        }
        self.device.download_f32(&out)
    }

    pub fn attention_layer_dev(&self, x: &Buffer, tokens: usize, w: &DeviceAttention,
                               map: crate::cuda::BatchMap<'_>)
        -> Result<Buffer> {
        let table_len = map
            .carry
            .and_then(|c| c.kv)
            .map(|kv| kv.positions)
            .unwrap_or(tokens)
            .max(tokens);
        let (cos, sin) = self.rope_tables_for(table_len)?;
        self.attention_layer_with_rope(x, tokens, w, &cos, &sin, map)
    }

    fn rope_tables_for(&self, tokens: usize) -> Result<(Buffer, Buffer)> {
        let rotary = self.backbone.config.rotary_dim();
        let theta = self.backbone.config.rope_theta;
        let mut cache = self.rope.lock().unwrap();
        if let Some((cos, sin, t, _r)) = cache.get(&tokens) {
            let _ = t;
            return Ok((cos.clone_buffer(), sin.clone_buffer()));
        }
        let positions: Vec<i32> = (0..tokens as i32).collect();
        let pos_dev = self.device.upload_i32(&positions)?;
        let tables = self.device.rope_tables_dev(&pos_dev, tokens, rotary, theta)?;
        cache.insert(tokens, (tables.0.clone_buffer(), tables.1.clone_buffer(), tokens, rotary));
        Ok(tables)
    }

    fn attention_layer_with_rope(&self, x: &Buffer, tokens: usize, w: &DeviceAttention,
                                 cos: &Buffer, sin: &Buffer,
                                 map: crate::cuda::BatchMap<'_>) -> Result<Buffer> {
        crate::gpu::gdn_trace_set_rows_only(tokens);
        let d = &self.device;
        let c = &self.backbone.config;
        let hidden = c.hidden_size;
        let hd = c.head_dim;
        let heads = c.num_attention_heads;
        let kv_heads = c.num_key_value_heads;
        let eps = c.rms_norm_eps;
        let rotary_dim = c.rotary_dim();

        let normed = d.rms_norm_dev(x, &w.input_layernorm, tokens, hidden, eps)?;
        ck!(d, "att_norm", &normed, hidden);

        let shared_q = if matches!(w.q_proj, DeviceProjection::Fp8 { .. })
            && matches!(w.k_proj, DeviceProjection::Fp8 { .. })
            && matches!(w.v_proj, DeviceProjection::Fp8 { .. })
        {
            Some(d.quantise_fp8_dev(&normed, tokens, hidden)?)
        } else {
            None
        };
        let proj_qkv = |proj: &DeviceProjection| -> Result<Buffer> {
            if let Some((q, qs)) = shared_q.as_ref() {
                if let Some(r) = run_fp8_projection_from_quantised(d, q, qs, tokens, proj, None) {
                    return r;
                }
            }
            run_projection_dev(d, &normed, tokens, proj, None)
        };
        let qg = proj_qkv(&w.q_proj)?;
        ck!(d, "att_qg", &qg, heads * hd * 2);
        let k_raw = proj_qkv(&w.k_proj)?;
        let v_raw = proj_qkv(&w.v_proj)?;

        let pos_base = map.carry.and_then(|c| c.cache_len);
        let rope_pos = map.carry.and_then(|c| c.rope_pos);
        let qk_fused = qk_norm_rope_fused() && hd == 256 && !gdn_trace_on()
            && std::env::var("CLEF_ATT_DUMP").is_err();
        let (q, gate, k) = if qk_fused {
            let (q, gate) = d.qk_norm_rope_dev(&qg, &w.q_norm, cos, sin, tokens * heads, heads, hd,
                                               rotary_dim, eps, map.seq_start, pos_base, rope_pos,
                                               true)?;
            let (k, _) = d.qk_norm_rope_dev(&k_raw, &w.k_norm, cos, sin, tokens * kv_heads, kv_heads,
                                            hd, rotary_dim, eps, map.seq_start, pos_base, rope_pos,
                                            false)?;
            (q, gate.expect("the q side always produces a gate"), k)
        } else {
            let (q, gate) = d.split_q_gate_dev(&qg, tokens, heads, hd)?;

            let q = d.rms_norm_dev(&q, &w.q_norm, tokens * heads, hd, eps)?;
            let k = d.rms_norm_dev(&k_raw, &w.k_norm, tokens * kv_heads, hd, eps)?;
            ck!(d, "att_q_norm", &q, hd);
            let q = d.rope_apply_dev(&q, cos, sin, tokens, heads, hd, rotary_dim, map.seq_start,
                                     pos_base, rope_pos)?;
            let k = d.rope_apply_dev(&k, cos, sin, tokens, kv_heads, hd, rotary_dim, map.seq_start,
                                     pos_base, rope_pos)?;
            (q, gate, k)
        };
        ck!(d, "att_q_rope", &q, hd);

        let dump_dir = std::env::var("CLEF_ATT_DUMP").ok();
        let dump = |name: &str, buf: &Buffer, width: usize| {
            if let Some(dir) = &dump_dir {
                if let Ok(all) = d.download_f32(buf) {
                    let n = (tokens * width).min(all.len());
                    let bytes: Vec<u8> = all[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(format!("{dir}/att_{name}.f32.bin"), &bytes);
                }
            }
        };
        dump("qg", &qg, heads * hd * 2);
        dump("q_norm", &q, heads * hd);
        dump("k_norm", &k, kv_heads * hd);
        dump("v", &v_raw, kv_heads * hd);

        let attn = d.attention_dev(&q, &k, &v_raw, tokens, heads, kv_heads, hd, map)?;
        dump("attn", &attn, heads * hd);

        ck!(d, "att_attn", &attn, hd);
        let gated = d.sigmoid_mul_dev(&attn, &gate, tokens * heads * hd)?;
        ck!(d, "att_gated", &gated, hd);
        dump("gated", &gated, heads * hd);
        let y = run_projection_dev(d, &gated, tokens, &w.o_proj, Some(x))?;
        dump("y", &y, hidden);

        let gated = if gdn_trace_on() {
            None
        } else {
            mlp_gated(d, &y, &w.post_attention_layernorm, tokens, hidden, eps,
                      &w.mlp_gate, &w.mlp_up, &w.mlp_down)?
        };
        let down = if let Some((q, qs)) = gated.as_ref() {
            run_nvfp4_projection_from_quantised(d, q, qs, tokens, &w.mlp_down, Some(&y))
                .or_else(|| run_fp8_projection_from_quantised(d, q, qs, tokens, &w.mlp_down, Some(&y)))
                .expect("the gated path is only taken when the down-projection consumes this form")?
        } else {
        let (gate, up) = mlp_gate_up(d, &y, &w.post_attention_layernorm, tokens, hidden, eps,
                                     &w.mlp_gate, &w.mlp_up, crate::gpu::gdn_trace_on())?;
        ck!(d, "mlp_gate", &gate, hidden);
        ck!(d, "mlp_up", &up, hidden);

        if gdn_trace_on() {
            let act = d.swiglu_dev(&gate, &up, tokens * c.intermediate_size)?;
            ck!(d, "swiglu", &act, c.intermediate_size);
            run_projection_dev(d, &act, tokens, &w.mlp_down, Some(&y))?
        } else {
            mlp_tail(d, &gate, &up, tokens, c.intermediate_size, &w.mlp_down, Some(&y))?
        }
        };

        ck!(d, "att_layer_out", &down, hidden);
        Ok(down)
    }

    pub fn attention_layer(&self, x: &[f32], tokens: usize, w: &DeviceAttention)
        -> Result<Vec<f32>> {
        let dev = self.device.upload_f32(x)?;
        let out = self.attention_layer_dev(&dev, tokens, w, crate::cuda::BatchMap::SINGLE)?;
        self.device.download_f32(&out)
    }

    pub fn embed_dev(&self, input_ids: &[u32]) -> Result<Buffer> {
        let ids: Vec<i32> = input_ids.iter().map(|&v| v as i32).collect();
        let ids_dev = self.device.upload_i32(&ids)?;
        self.device.embed_gather_dev(
            &self.backbone.embed_tokens, &ids_dev, input_ids.len(),
            self.backbone.config.hidden_size)
    }

    pub fn layers_dev(&self, h: &Buffer, tokens: usize) -> Result<Buffer> {
        self.layers_dev_batched(h, tokens, crate::cuda::BatchMap::SINGLE)
    }

    pub fn layers_dev_batched(&self, h: &Buffer, tokens: usize,
                              map: crate::cuda::BatchMap<'_>) -> Result<Buffer> {
        self.layers_dev_with(h, tokens, map, None)
    }

    pub fn layers_dev_with(&self, h: &Buffer, tokens: usize,
                           map: crate::cuda::BatchMap<'_>,
                           carry: Option<&[crate::cuda::LayerCarry<'_>]>) -> Result<Buffer> {
        crate::gpu::gdn_trace_set_rows(tokens);
        let c = &self.backbone.config;
        let hidden = c.hidden_size;

        let mut h = h.clone_buffer();

        let row_dump = std::env::var("CLEF_LAYER_ROW").ok().and_then(|v| v.parse::<usize>().ok());
        let n_layers = self.backbone.layers.len();
        let mut rowbuf = match row_dump {
            Some(r) if r < tokens => Some((r, self.device.alloc_zeroed(n_layers * hidden * 4)?)),
            _ => None,
        };

        let trace = std::env::var("CLEF_BATCH_TRACE").is_ok();

        let max_layers = std::env::var("CLEF_LAYERS_MAX").ok().and_then(|v| v.parse::<usize>().ok());
        for (li, layer) in self.backbone.layers.iter().enumerate() {
            crate::gpu::gdn_trace_set_layer(li);
            if max_layers.map_or(false, |n| li >= n) {
                break;
            }

            let lmap = match carry.and_then(|c| c.get(li)) {
                Some(lc) => crate::cuda::BatchMap { carry: Some(*lc), ..map },
                None => map,
            };

            let timed = std::env::var("CLEF_LAYER_TIME").is_ok();
            let t_layer = std::time::Instant::now();
            h = match layer {
                DeviceLayer::Linear(w) => self.delta_net_layer_dev(&h, tokens, w, lmap)?,
                DeviceLayer::Full(w) => self.attention_layer_dev(&h, tokens, w, lmap)?,
            };
            if timed {
                let _ = self.device.sync();
                let kind = if matches!(layer, DeviceLayer::Linear(_)) { "lin" } else { "att" };
                eprintln!("  LTIME {li:2} {kind} rows={tokens} {:.2} ms",
                          t_layer.elapsed().as_secs_f64() * 1e3);
            }
            if let Some((r, buf)) = rowbuf.as_ref() {
                self.device.copy_bytes_at(buf, li * hidden * 4, &h, r * hidden * 4, hidden * 4)?;
            }
            if trace {
                let v = self.device.download_rows_f32(&h, tokens.min(4), hidden)?;
                let kind = if matches!(layer, DeviceLayer::Linear(_)) { "lin" } else { "att" };
                eprintln!(
                    "  LAYER {li:2} {kind} tok0 [0..4] {:?}",
                    v[..4.min(v.len())].iter().map(|x| (x * 1e3).round() / 1e3).collect::<Vec<_>>()
                );

                let dump_wanted = std::env::var("CLEF_LAYER_DUMP_TOKENS").ok()
                    .map_or(true, |n| n.parse::<usize>().map_or(true, |n| n == tokens));
                if let Some(dir) = std::env::var("CLEF_LAYER_DUMP").ok().filter(|_| dump_wanted) {
                    let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();

                    let real = if map.real_end > 0 { map.real_end } else { tokens };
                    let sub = format!("{dir}/t{real}");
                    let _ = std::fs::create_dir_all(&sub);
                    if let Ok(all) = self.device.download_f32(&h) {
                        let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
                        let _ = std::fs::write(format!("{sub}/layer_{tag}{li:02}.f32.bin"), &bytes);
                    }
                }
            }
        }

        if let Some((r, buf)) = rowbuf.as_ref() {
            if let Ok(all) = self.device.download_f32(buf) {
                let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
                let path = format!("/tmp/layerrow_{tag}row{r}_t{tokens}.f32.bin");
                let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(&path, &bytes);
                eprintln!("  rowdump {path} layers={n_layers} hidden={hidden}");
            }
        }

        let final_wanted = std::env::var("CLEF_LAYER_DUMP_TOKENS").ok()
            .map_or(true, |n| n.parse::<usize>().map_or(true, |n| n == tokens));
        if let Some(dir) = std::env::var("CLEF_LAYER_DUMP").ok().filter(|_| final_wanted) {
            if let Ok(n) = self.device.rms_norm_dev(&h, &self.backbone.final_norm, tokens, hidden,
                                                    c.rms_norm_eps) {
                if let Ok(all) = self.device.download_f32(&n) {
                    let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let real = if map.real_end > 0 { map.real_end } else { tokens };
                    let sub = format!("{dir}/t{real}");
                    let _ = std::fs::create_dir_all(&sub);
                    let _ = std::fs::write(format!("{sub}/final.f32.bin"), &bytes);
                }
            }
        }
        crate::ck!(self.device, "final_norm", &h, hidden);

        if std::env::var("CLEF_NORM_TWICE").is_ok() {
            let a = self.device.rms_norm_dev(&h, &self.backbone.final_norm, tokens, hidden, c.rms_norm_eps)?;
            let b = self.device.rms_norm_dev(&h, &self.backbone.final_norm, tokens, hidden, c.rms_norm_eps)?;
            let a = self.device.download_rows_f32(&a, tokens, hidden)?;
            let b = self.device.download_rows_f32(&b, tokens, hidden)?;
            let d = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            let f = a.iter().zip(&b).position(|(p, q)| p != q);
            eprintln!("  NORM TWICE: worst = {d:.3e}, first {f:?}");
        }
        self.device.rms_norm_dev(&h, &self.backbone.final_norm, tokens, hidden, c.rms_norm_eps)
    }

    fn record_calibration(&self, run: u64, used: u64) {
        if let Ok(mut c) = self.calib.lock() {
            if !c.iter().any(|p| p.0 == run) {
                c.push((run, used));
            }
        }
    }

    fn footprint_line(&self) -> Option<(f64, f64)> {
        let points = self.calib.lock().ok()?.clone();
        let lo = *points.iter().min_by_key(|p| p.0)?;
        let hi = *points.iter().max_by_key(|p| p.0)?;
        if hi.0 <= lo.0 {
            return None;
        }
        let b = (hi.1 as f64 - lo.1 as f64) / (hi.0 as f64 - lo.0 as f64);
        let a = lo.1 as f64 - b * lo.0 as f64;
        Some((a, b))
    }

    pub fn max_servable_tokens(&self) -> Option<usize> {
        let chunked = crate::gpu::max_chunked_tokens();
        if crate::gpu::max_chunk_size() < crate::engine::max_input_tokens() {
            let fits = self.carry_capacity_that_fits();
            let t = chunked.min(fits);
            return if t == 0 { None } else { Some(t) };
        }
        let captured = self
            .graphs
            .lock()
            .ok()?
            .keys()
            .filter(|(_, mode)| *mode == GraphMode::Chunked)
            .map(|(run, _)| *run)
            .max()
            .unwrap_or(0);
        let probed = self.calib.lock().ok()?.iter().map(|p| p.0).max().unwrap_or(0) as usize;
        let t = captured.max(probed);
        if t == 0 { None } else { Some(t) }
    }

    pub fn carry_capacity_that_fits(&self) -> usize {
        if let Ok(slot) = self.carry.lock() {
            if let Some(c) = slot.as_ref() {
                return c.capacity;
            }
        }
        let c = &self.backbone.config;
        let n_full = self.backbone.layers.iter().filter(|l| matches!(l, DeviceLayer::Full(_))).count();
        let n_lin = self.backbone.layers.len() - n_full;
        let per_token = 2 * c.num_key_value_heads * c.head_dim * 2 * n_full;
        if per_token == 0 {
            return 0;
        }
        let fixed = crate::cuda::gdn_state_bytes(c.linear_num_value_heads, c.linear_key_head_dim,
                                                 c.linear_value_head_dim) * n_lin
            + (c.linear_conv_kernel_dim - 1) * c.conv_dim() * 4 * n_lin
            + 4;
        let free = self.device.free_memory().unwrap_or(0) / 4 * 3;

        let tokens = free.saturating_sub(fixed) / per_token / crate::gpu::kv_slots().max(1);
        let granule = crate::cuda::CHUNK_GRANULE;
        tokens / granule * granule
    }

    pub fn measure_footprint(&self) -> Result<()> {
        if std::env::var("CLEF_NO_FOOTPRINT").is_ok() { return Ok(()); }

        let top = crate::gpu::max_chunk_size().max(1024);
        for tokens in [1024usize, top] {
            let run = crate::gpu::pad_to_chunk(tokens);
            let ids: Vec<u32> = vec![1; run];
            let h = self.embed_dev(&ids)?;
            let out = self.layers_dev_batched(&h, run, crate::cuda::BatchMap {
                state_slot: None,
                hist_save: None,
                hist_save_dev: None,
                hist_save_n: 0,
                carry: None,
                seq_start: None, chunk_start: None, seq_end: None, real_end: run,
                real_end_dev: None,
            })?;
            self.device.sync()?;
            drop(out);
            let total = self.device.total_memory().unwrap_or(0);
            let used = total.saturating_sub(self.device.free_memory().unwrap_or(0));
            self.record_calibration(run as u64, used as u64);
        }

        if let Some((a, _b)) = self.footprint_line() {
            let total = self.device.total_memory().unwrap_or(0) as f64;
            let shared = (total - a).max(0.0);
            let cap = std::env::var("CLEF_GRAPH_BUDGET")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or((shared / 2.0) as usize);
            self.device.set_graph_block_cap(cap);
            tracing::info!(
                cap_mib = cap / (1 << 20),
                model_mib = a as usize / (1 << 20),
                "the block's cap"
            );
        }
        let reserve = std::env::var("CLEF_GRAPH_RESERVE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1 << 30);
        self.device.set_graph_reserve(reserve);

        let pool_before = self.device.pool_bytes();
        let free_before = self.device.free_memory().unwrap_or(0);
        let freed = self.device.trim_device_pool();
        let free_after = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "  POOL trim: pool {} MiB -> {} MiB | freed {} MiB | free {} MiB -> {} MiB",
            pool_before / (1 << 20),
            self.device.pool_bytes() / (1 << 20),
            freed / (1 << 20),
            free_before / (1 << 20),
            free_after / (1 << 20)
        );
        if let Some((a, b)) = self.footprint_line() {
            eprintln!(
                "  FOOTPRINT a={:.1} MiB b={:.3} MiB/token | block={} MiB reserve={} MiB ceiling={:?}",
                a / 1048576.0,
                b / 1048576.0,
                self.device.graph_block_bytes() / 1048576,
                self.device.graph_reserve() / 1048576,
                self.max_servable_tokens()
            );
        }
        tracing::info!(max = self.max_servable_tokens(), "measured the footprint before capturing");
        Ok(())
    }

    fn capture_fits(&self, tokens: usize) -> Result<()> {
        let total = match self.device.total_memory() {
            Ok(t) if t > 0 => t as f64,
            _ => return Ok(()),
        };

        let Some((a, b)) = self.footprint_line() else {
            return Ok(());
        };
        let run = crate::gpu::pad_to_chunk(tokens) as f64;
        let budget = total * SERVABLE_FRACTION;
        let proven = self.calib.lock().ok().and_then(|c| c.iter().map(|p| p.0).max()).unwrap_or(0) as f64;
        if (run as u64) <= proven as u64 {
            return Ok(());
        }

        let beyond = (run - proven).max(0.0);

        let predicted = a + b * run + EXTRAPOLATION_PAD * b * beyond;
        if predicted > budget {
            return Err(anyhow::Error::new(crate::engine::Refused {
                tokens,
                why: format!(
                    "it is predicted to need about {:.1} GB of the {:.1} GB this device has, measured \
                     from {} captured sizes",
                    predicted / 1e9,
                    total / 1e9,
                    self.calib.lock().map(|c| c.len()).unwrap_or(0)
                ),
            }));
        }
        Ok(())
    }

    fn graph_io_for(&self, values: usize) -> Result<(Buffer, Buffer)> {
        let mut io = self.graph_io.lock().expect("the graph_io lock is never poisoned here");
        let need = match io.as_ref() {
            Some(existing) if existing.values >= values => false,
            _ => true,
        };
        if need {
            let capacity = values.max(io.as_ref().map(|i| i.values * 2).unwrap_or(0));
            let input = self.device.alloc_zeroed(capacity * 4)?;
            let output = self.device.alloc(capacity * 4)?;
            tracing::info!(capacity, "grew the shared graph input/output pair");
            *io = Some(GraphIo { input, output, values: capacity });
        }
        let existing = io.as_ref().expect("just ensured");
        Ok((
            self.device.view(&existing.input, 0, values * 4),
            self.device.view(&existing.output, 0, values * 4),
        ))
    }

    fn graph_maps(&self) -> Result<GraphMaps> {
        let mut maps = self.graph_maps.lock().expect("the graph_maps lock is never poisoned here");
        if let Some(existing) = maps.as_ref() {
            return Ok(GraphMaps {
                seq_start: existing.seq_start.clone_buffer(),
                seq_end: existing.seq_end.clone_buffer(),
                chunk_start: existing.chunk_start.clone_buffer(),
            });
        }
        let ceiling = crate::gpu::max_graph_tokens().div_ceil(crate::gpu::GRAPH_BUCKET)
            * crate::gpu::GRAPH_BUCKET;
        let starts = self.device.alloc_zeroed(ceiling * 4)?;
        let ends = self.device.alloc_zeroed(ceiling * 4)?;
        let chunks = self.device.alloc_zeroed((ceiling.div_ceil(crate::cuda::CHUNK_LEN) + 1) * 4)?;
        *maps = Some(GraphMaps {
            seq_start: starts.clone_buffer(),
            seq_end: ends.clone_buffer(),
            chunk_start: chunks.clone_buffer(),
        });
        Ok(GraphMaps { seq_start: starts, seq_end: ends, chunk_start: chunks })
    }

    fn layers_graphed_with(&self, h: &Buffer, tokens: usize, map: crate::cuda::BatchMap<'_>,
                           carry: Option<&[crate::cuda::LayerCarry<'_>]>)
        -> Result<(Buffer, bool)> {
        let hidden = self.backbone.config.hidden_size;
        let batched = map.seq_start.is_some();

        self.device.warm_attention_scratch(self.backbone.config.num_attention_heads,
                                           self.backbone.config.head_dim)?;

        let mode = GraphMode::Chunked;
        let _ = batched;

        let run = crate::gpu::pad_to_chunk(tokens);
        let values = tokens * hidden;

        let map = crate::cuda::BatchMap {
            carry: None,
            seq_start: map.seq_start,
            chunk_start: map.chunk_start,
            seq_end: map.seq_end,
            state_slot: map.state_slot,
            hist_save: map.hist_save,
            hist_save_dev: map.hist_save_dev,
            hist_save_n: map.hist_save_n,
            real_end: map.real_end,
            real_end_dev: map.real_end_dev,
        };

        if !crate::gpu::graph_covers(tokens) {
            return crate::gpu::eager_or_error(tokens, "no graph covers this size", || {
                Ok((self.layers_dev_with(h, run, map, carry)?, false))
            });
        }

        let key = crate::gpu::graph_key(run, mode, 0);
        let run_values = run * hidden;

        if std::env::var("CLEF_GRAPH_RECAPTURE").is_ok() {
            if let Ok(mut g) = self.graphs.lock() {
                g.remove(&key);
            }
        }

        {
            let graphs = self.graphs.lock().unwrap();
            if let Some(g) = graphs.get(&key) {
                self.device.copy_into(&g.input, h, run_values)?;
                g.captured.launch()?;
                return Ok((self.device.copy_rows(&g.output, tokens, hidden)?, false));
            }
        }
        let (input, output) = if std::env::var("CLEF_GRAPH_SHARED_IO").as_deref() == Ok("0") {
            (self.device.alloc_zeroed(run_values * 4)?, self.device.alloc(run_values * 4)?)
        } else {
            self.graph_io_for(run_values)?
        };

        if let Err(e) = self.capture_fits(tokens) {
            if crate::gpu::eager_fallback_allowed() {
                tracing::warn!(tokens, "running EAGERLY because CLEF_EAGER=1: {e:#}");
                return Ok((self.layers_dev_with(h, run, map, carry)?, false));
            }

            return Err(e);
        }

        let budget = graph_memory_budget(&self.device);
        let held = self.device.graph_pool_high();
        tracing::debug!(tokens, held, budget, "capture pool high-water");

        let table_len = match carry.and_then(|c| c.iter().find_map(|lc| lc.kv)) {
            Some(kv) => kv.positions.max(run),
            None => run,
        };
        let _ = self.rope_tables_for(table_len)?;

        if std::env::var("CLEF_GRAPH_PRIME").is_ok() {
            self.device.begin_priming();
            for _ in 0..3 {
                let warm = self.layers_dev_with(h, run, map, carry)?;
                self.device.sync()?;
                drop(warm);
            }
            self.device.end_priming();
        }

        let mut graph_out: Option<Buffer> = None;
        let captured = match self.device.capture(|| {
            let y = self.layers_dev_with(&input, run, map, carry)?;

            graph_out = Some(y);
            Ok(())
        }) {
            Ok(c) => c,
            Err(e) => return Err(as_refused(e, tokens)),
        };

        let held = self.device.graph_block_bytes();
        tracing::info!(tokens, held, budget, "captured a graph");

        eprintln!(
            "  CAPTURE tokens={tokens} batched={batched} run={run} alloc_nodes={} pool_high={held} free={}",
            captured.alloc_nodes, self.device.free_memory().unwrap_or(0)
        );

        let total = self.device.total_memory().unwrap_or(0);
        let used = total.saturating_sub(self.device.free_memory().unwrap_or(0));
        self.record_calibration(run as u64, used as u64);

        let output = graph_out.unwrap_or(output);
        let mut graphs = self.graphs.lock().unwrap();
        graphs.insert(key, GraphedLayers { captured, input, output });
        let g = graphs.get(&key).expect("just inserted");

        self.device.copy_into(&g.input, h, run_values)?;
        g.captured.launch()?;
        Ok((self.device.copy_rows(&g.output, tokens, hidden)?, true))
    }

    pub fn capture_pending_one(&self) -> Result<bool> {
        let Some((geometry, mode, splits)) = self.pending.lock().ok().and_then(|mut p| p.pop_front())
        else { return Ok(false); };
        struct G(usize);
        impl Drop for G { fn drop(&mut self) { crate::cuda::set_splits_hint(self.0); } }
        let _hg = G(crate::cuda::splits_hint());
        crate::cuda::set_splits_hint(splits);
        struct C<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for C<'_> { fn drop(&mut self) { self.0.store(false, std::sync::atomic::Ordering::SeqCst); } }
        self.capturing_pass.store(true, std::sync::atomic::Ordering::SeqCst);
        let _cg = C(&self.capturing_pass);

        if self.graph_ready_for(geometry, mode) {
            return Ok(false);
        }
        let hidden = self.backbone.config.hidden_size;
        let run = crate::gpu::pad_to_chunk(geometry);
        let h = self.device.alloc_zeroed(run * hidden * 4)?;

        let carry_guard = if mode == GraphMode::Chunked {
            let cap = self
                .carry
                .lock()
                .ok()
                .and_then(|g| g.as_ref().map(|c| c.capacity))
                .unwrap_or_else(crate::gpu::kv_capacity);
            self.ensure_carry(cap)?;
            self.carry().ok()
        } else {
            None
        };
        let chunked_views = carry_guard.as_deref().map(|c| c.views());
        if let Some(c) = carry_guard.as_deref() {
            c.real_len.store(geometry, std::sync::atomic::Ordering::Relaxed);
        }

        let map = if mode == GraphMode::Chunked {
            let c = carry_guard.as_deref().context("a chunked capture needs the carry's maps")?;
            crate::cuda::BatchMap {
                state_slot: Some(&c.state_slot),
                hist_save: None,
                carry: None,
                seq_start: Some(&c.seq_start),
                chunk_start: Some(&c.chunk_start),
                seq_end: Some(&c.seq_end),
                real_end: 0,
                real_end_dev: Some(&c.real_end),
                hist_save_dev: Some(&c.hist_save),
                hist_save_n: c.slots,
            }
        } else {
            crate::cuda::BatchMap {
                state_slot: None,
                hist_save: None,
                hist_save_dev: None,
                hist_save_n: 0,
                carry: None,
                seq_start: None,
                chunk_start: None,
                seq_end: None,
                real_end: 0,
                real_end_dev: Some(&self.params),
            }
        };
        let (_out, captured) = self.layers_graphed_with(&h, geometry, map, chunked_views.as_deref())?;
        Ok(captured)
    }

    pub fn graph_count_for(&self, mode: GraphMode) -> usize {
        self.graphs.lock().map(|g| g.keys().filter(|(_, m)| *m == mode).count()).unwrap_or(0)
    }

    pub fn chunked_passes(&self) -> usize {
        self.chunked_passes.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn chunked_width_max(&self) -> usize {
        self.chunked_width_max.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn chunked_passes_sharing(&self) -> usize {
        self.chunked_passes_sharing.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn pending_captures(&self) -> usize {
        self.pending.lock().map(|p| p.len()).unwrap_or(0)
    }

    pub fn carry_capacity(&self) -> usize {
        self.carry.lock().ok().and_then(|g| g.as_ref().map(|c| c.capacity)).unwrap_or(0)
    }

    pub fn carry_bytes(&self) -> usize {
        let c = &self.backbone.config;
        let cap = self.carry_capacity();
        let n_full = self.backbone.layers.iter().filter(|l| matches!(l, DeviceLayer::Full(_))).count();
        let n_lin = self.backbone.layers.len() - n_full;
        let kv = cap * c.num_key_value_heads * c.head_dim * 2;
        let state = crate::cuda::gdn_state_bytes(c.linear_num_value_heads, c.linear_key_head_dim,
                                                 c.linear_value_head_dim);
        let hist = (c.linear_conv_kernel_dim - 1) * c.conv_dim() * 4;
        2 * kv * n_full + (state + hist) * n_lin + 4
    }

    pub fn graph_count(&self) -> usize {
        self.graphs.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn carry_geometry(&self) -> Option<(usize, usize)> {
        let c = self.carry.lock().ok()?;
        c.as_ref().map(|c| (c.slots, c.capacity))
    }

    pub fn carry_kv_snapshot(&self, layer: usize) -> Option<Vec<u16>> {
        let c = self.carry().ok()?;
        let (k, _v) = c.kv.get(layer)?.as_ref()?;
        let bytes = self.device.download(k).ok()?;
        Some(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    pub fn carry_state_snapshot(&self, layer: usize) -> Option<Vec<f32>> {
        let guard = self.carry.lock().ok()?;
        let c = guard.as_ref()?;
        self.device.download_f32(c.gdn_state.get(layer)?.as_ref()?).ok()
    }

    pub fn graph_memory_nodes(&self, rows: usize, mode: GraphMode) -> Option<(usize, usize, usize)> {
        let g = self.graphs.lock().ok()?;
        if let Some(v) = g.get(&crate::gpu::graph_key(rows, mode, 0)) {
            return Some((v.captured.alloc_nodes, v.captured.free_nodes, v.captured.memcpy_nodes));
        }
        if mode != GraphMode::Chunked {
            return None;
        }

        g.iter()
            .find(|((k, m), _)| *m == mode && (k & 0xFF_FFFF) == rows)
            .map(|(_, v)| (v.captured.alloc_nodes, v.captured.free_nodes, v.captured.memcpy_nodes))
    }

    pub fn graph_ready_for(&self, rows: usize, mode: GraphMode) -> bool {
        self.graphs.lock().map(|g| {
            if mode == GraphMode::Chunked {
                if crate::cuda::splits_hint() > 0 {
                    g.contains_key(&crate::gpu::graph_key(rows, mode, 0))
                } else {
                    g.keys().any(|(k, m)| *m == mode && (k & 0xFF_FFFF) == rows)
                }
            } else {
                g.contains_key(&(rows, mode))
            }
        }).unwrap_or(false)
    }

    pub fn queue_capture_for(&self, rows: usize, mode: GraphMode) {
        let splits = crate::cuda::splits_hint();
        if let Ok(mut pending) = self.pending.lock() {
            if !pending.iter().any(|(r, m, sp)| *r == rows && *m == mode && *sp == splits) {
                pending.push_back((rows, mode, splits));
            }
        }
    }

    pub fn head_forward_pre(&self, pre: &PreHead, tokens: usize, input_ids: &[u32],
                            questions: &[QuestionSpec], lm_head: &HostTable, h: &DeviceHead,
                            kv: &KvGraphs) -> Result<Vec<Vec<f32>>> {
        head_forward_pre_eager_with(&self.device, pre, tokens, input_ids, questions, lm_head, h, kv)
    }

    pub fn mem_used_mib(&self) -> usize {
        self.device
            .total_memory()
            .unwrap_or(0)
            .saturating_sub(self.device.free_memory().unwrap_or(0))
            / (1 << 20)
    }

    pub fn forward_dev(&self, input_ids: &[u32]) -> Result<Buffer> {
        crate::gpu::gdn_trace_set_pass("solo");

        let tokens = input_ids.len();
        let chunk = crate::gpu::max_chunk_size();
        let trace = std::env::var("CLEF_MEM_TRACE").is_ok();
        if trace {
            eprintln!("    mem {:>6} MiB  before a {tokens}-token request", self.mem_used_mib());
        }
        let out = self.forward_chunked_dev(input_ids, chunk, crate::gpu::kv_capacity())?;
        if trace {
            eprintln!("    mem {:>6} MiB  after the backbone ({} pass(es))", self.mem_used_mib(),
                      tokens.div_ceil(chunk));
        }
        Ok(out)
    }

    pub fn prime_long_path(&self) -> Result<()> {
        if std::env::var("CLEF_PRIME_LONG").as_deref() == Ok("0") {
            return Ok(());
        }
        if !crate::gpu::chunking_enabled() {
            return Ok(());
        }

        let t = std::time::Instant::now();
        let cap = crate::gpu::kv_capacity();
        let before = self.device.free_memory().unwrap_or(0);
        self.ensure_carry(cap)?;
        let after = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "  PRIMED the KV cache: {} tokens, {:.0} MiB reserved, free {:.0} MiB, {:.0} ms",
            cap,
            before.saturating_sub(after) as f64 / 1048576.0,
            after as f64 / 1048576.0,
            t.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    fn ensure_carry(&self, capacity: usize) -> Result<()> {
        self.ensure_carry_slots(capacity, crate::gpu::kv_slots())
    }

    fn carry(&self) -> Result<std::sync::Arc<PassCarry>> {
        self.carry.lock()
            .map_err(|_| anyhow::anyhow!("carry lock poisoned"))?
            .clone()
            .context("forward_chunk_begin allocates it")
    }

    fn ensure_carry_slots(&self, capacity: usize, slots: usize) -> Result<()> {
        let slots = slots.max(1);
        let mut slot = self.carry.lock().map_err(|_| anyhow::anyhow!("carry lock poisoned"))?;
        if let Some(c) = slot.as_ref() {
            if c.capacity >= capacity && c.slots >= slots {
                return Ok(());
            }
            tracing::debug!(held = c.capacity, slots = c.slots, capacity, "growing the KV carry; dropping                              the chunked graphs that pointed at the old one");
            slot.take();

            if let Ok(mut g) = self.graphs.lock() {
                g.clear();

                self.device.note_graph_block_cleared();
            }
            if let Ok(mut p) = self.pending.lock() {
                p.retain(|(_, m, _)| *m != GraphMode::Chunked);
            }
        }
        let d = &self.device;
        let c = &self.backbone.config;
        let k = c.linear_conv_kernel_dim;
        let conv_dim = c.conv_dim();
        let state_bytes = crate::cuda::gdn_state_bytes(c.linear_num_value_heads,
                                                       c.linear_key_head_dim,
                                                       c.linear_value_head_dim);

        let kv_bytes = slots * capacity * c.num_key_value_heads * c.head_dim * 2;

        {
            let n_full = self.backbone.layers.iter().filter(|l| matches!(l, DeviceLayer::Full(_))).count();
            let n_lin = self.backbone.layers.len() - n_full;
            let total = 2 * kv_bytes * n_full
                + state_bytes * n_lin
                + (k - 1) * conv_dim * 4 * n_lin
                + 4;
            let free = d.free_memory().context("reading free memory before allocating the carry")?;
            let budget = free / 4 * 3;
            if total > budget {
                anyhow::bail!(
                    "a KV cache for {capacity} tokens needs {} MiB and only {} MiB of the card's \
                     {} MiB is free — that is {:.1} KiB per token over {} full-attention layers. \
                     Lower CLEF_MAX_LENGTH or CLEF_KV_CAPACITY.",
                    total / (1 << 20),
                    free / (1 << 20),
                    d.total_memory().unwrap_or(0) / (1 << 20),
                    (2 * c.num_key_value_heads * c.head_dim * 2) as f64 / 1024.0,
                    n_full
                );
            }
        }
        let mut gdn_state = Vec::with_capacity(self.backbone.layers.len());
        let mut state_scratch = Vec::with_capacity(self.backbone.layers.len());
        let mut conv_hist = Vec::with_capacity(self.backbone.layers.len());
        let mut kv = Vec::with_capacity(self.backbone.layers.len());
        for layer in &self.backbone.layers {
            match layer {
                DeviceLayer::Linear(_) => {
                    gdn_state.push(Some(d.alloc_zeroed(slots * state_bytes)?));

                    state_scratch.push(if std::env::var("CLEF_STATE_SCRATCH").is_ok() {
                        Some(d.alloc_zeroed(slots * state_bytes)?)
                    } else {
                        None
                    });
                    conv_hist.push(Some(d.alloc_zeroed(slots * (k - 1) * conv_dim * 4)?));
                    kv.push(None);
                }
                DeviceLayer::Full(_) => {
                    gdn_state.push(None);
                    state_scratch.push(None);
                    conv_hist.push(None);
                    kv.push(Some((d.alloc_zeroed(kv_bytes)?, d.alloc_zeroed(kv_bytes)?)));
                }
            }
        }
        tracing::info!(
            capacity,
            kv_mib = (kv_bytes / (1 << 20)) as usize,
            state_mib = (state_bytes / (1 << 20)) as usize,
            "chunked prefill: the carry is allocated"
        );

        let max_pass = crate::gpu::max_graph_tokens().max(crate::gpu::max_chunk_size());
        *slot = Some(std::sync::Arc::new(PassCarry { gdn_state, state_scratch, conv_hist, kv, cache_len: d.alloc_zeroed(4)?,
                                 cache_base: d.alloc_zeroed(4)?,
                                 slot_map: d.alloc_zeroed(slots * max_pass * 4)?,
                                 state_slot: d.alloc_zeroed((slots * capacity).div_ceil(64).max(1) * 4)?,
                                 cpos: d.alloc_zeroed(slots * max_pass * 4)?,
                                 cbase: d.alloc_zeroed(slots * max_pass * 4)?,
                                 rope_pos: d.alloc_zeroed(slots * max_pass * 4)?,
                                 seq_start: d.alloc_zeroed(slots * max_pass * 4)?,
                                 seq_end: d.alloc_zeroed(slots * max_pass * 4)?,
                                 chunk_start: d.alloc_zeroed((slots * max_pass).div_ceil(64).max(1) * 4)?,
                                 real_end: d.alloc_zeroed(4)?,
                                 real_walk: d.alloc_zeroed(4)?,
                                 hist_save: d.alloc_zeroed(slots * 2 * 4)?,
                                 state_bytes, hist_bytes: (k - 1) * conv_dim * 4,
                                 capacity, slots,
                                 slot: std::sync::atomic::AtomicUsize::new(0),
                                 real_len: std::sync::atomic::AtomicUsize::new(0) }));
        Ok(())
    }

    pub fn forward_chunked_dev(&self, input_ids: &[u32], chunk: usize, capacity: usize)
        -> Result<Buffer> {
        self.forward_chunked_with(input_ids, chunk, capacity, None)?
            .context("a chunked pass with no sink must produce the hidden states")
    }

    #[cfg(test)]
    pub fn forward_chunked_into(&self, input_ids: &[u32], chunk: usize, capacity: usize,
                                sink: &PreHead) -> Result<()> {
        self.forward_chunked_with(input_ids, chunk, capacity, Some(sink))?;
        Ok(())
    }

    pub fn multi_chunk_pass(&self, chunks: &[ChunkSpec<'_>]) -> Result<Buffer> {
        struct HG(usize);
        impl Drop for HG { fn drop(&mut self) { crate::cuda::set_splits_hint(self.0); } }
        let _hg = HG(crate::cuda::splits_hint());
        if crate::cuda::splits_hint() == 0 {
            if let Some(c) = chunks.first() {
                let splits = if chunks.len() > 1 {
                    crate::cuda::att_keys()
                } else {
                    crate::cuda::splits_for_request(c.ids.len())
                };
                crate::cuda::set_splits_hint(splits);
            }
        }

        let trace = std::env::var("CLEF_PASS_TRACE").is_ok();
        let t_pass = std::time::Instant::now();
        let d = &self.device;
        let c = &self.backbone.config;
        let hidden = c.hidden_size;
        let n = chunks.len();
        if n == 0 {
            return Ok(d.alloc_zeroed(0)?);
        }

        let carry = self.carry()?;
        let cap = carry.capacity;

        self.chunked_passes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.chunked_width_max.fetch_max(chunks.len(), std::sync::atomic::Ordering::Relaxed);
        if chunks.len() > 1 {
            self.chunked_passes_sharing.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let maps = pass_maps(chunks, cap);
        let PassMaps { ids, seq_start, seq_end, cpos, cbase, slot_map, rope_pos, chunk_start,
                       state_slot, hist_save, cache_len, rows_real, rows_pad, offs, total } = maps;

        let bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        if std::env::var("CLEF_BATCH_MAPS").is_ok() {
            for (i, &o) in offs.iter().enumerate() {
                eprintln!("BATCHMAP req {i}: off={o} real={} pad={} cpos0={} cbase0={}",
                          rows_real[i], rows_pad[i], cpos[o], cbase[o]);
            }
            eprintln!("BATCHMAP total={total} n={n} cache_len={cache_len}");
        }
        d.copy_htod_into(&carry.cpos, &bytes(&cpos))?;
        d.copy_htod_into(&carry.cbase, &bytes(&cbase))?;
        d.copy_htod_into(&carry.slot_map, &bytes(&slot_map))?;
        d.copy_htod_into(&carry.rope_pos, &bytes(&rope_pos))?;
        d.copy_htod_into(&carry.state_slot, &bytes(&state_slot))?;

        d.copy_htod_into(&carry.cache_len, &cache_len.to_le_bytes())?;

        let real_end = offs.last().copied().unwrap_or(0)
            + rows_real.last().copied().unwrap_or(0);
        let real_walk = real_end.div_ceil(crate::cuda::CHUNK_GRANULE) * crate::cuda::CHUNK_GRANULE;
        carry.real_len.store(real_walk.min(total), std::sync::atomic::Ordering::Relaxed);

        carry.slot.store(chunks[0].slot, std::sync::atomic::Ordering::Relaxed);

        if std::env::var("CLEF_CARRY_HASH").is_ok() {
            let hash = |b: &Buffer| -> u64 {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                if let Ok(v) = d.download(b) {
                    for x in v.iter().take(1 << 20) {
                        h ^= *x as u64;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                h
            };
            let st = carry.gdn_state.iter().flatten().next().map(hash).unwrap_or(0);
            let hi = carry.conv_hist.iter().flatten().next().map(hash).unwrap_or(0);
            eprintln!("  CARRY state {st:016x} hist {hi:016x} real_len={} slot={}",
                      carry.real_len.load(std::sync::atomic::Ordering::Relaxed),
                      carry.slot.load(std::sync::atomic::Ordering::Relaxed));
        }
        let h = self.embed_dev(&ids)?;
        let views = carry.views();

        d.copy_htod_into(&carry.seq_start, &bytes(&seq_start))?;
        d.copy_htod_into(&carry.seq_end, &bytes(&seq_end))?;
        d.copy_htod_into(&carry.chunk_start, &bytes(&chunk_start))?;
        d.copy_htod_into(&carry.real_end, &(total as i32).to_le_bytes())?;

        d.copy_htod_into(&carry.real_walk, &(real_walk as i32).to_le_bytes())?;

        let mut pairs: Vec<i32> = Vec::with_capacity(carry.slots * 2);
        for &(row, slot) in hist_save.iter() {
            pairs.push(row as i32);
            pairs.push(slot as i32);
        }
        while pairs.len() < carry.slots * 2 {
            pairs.push(-1);
            pairs.push(-1);
        }
        d.copy_htod_into(&carry.hist_save, &bytes(&pairs))?;
        let map = crate::cuda::BatchMap {
            seq_start: Some(&carry.seq_start),
            seq_end: Some(&carry.seq_end),
            chunk_start: Some(&carry.chunk_start),
            state_slot: Some(&carry.state_slot),
            hist_save: Some(&hist_save),
            hist_save_dev: Some(&carry.hist_save),
            hist_save_n: carry.slots,
            real_end: total,
            real_end_dev: Some(&carry.real_end),
            ..crate::cuda::BatchMap::SINGLE
        };

        if let Ok(dir) = std::env::var("CLEF_PASS_INPUT_DUMP") {
            let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
            let seq = PASS_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = std::fs::create_dir_all(&dir);
            let base = format!("{dir}/{seq:03}_{tag}");
            if let Ok(v) = d.download_f32(&h) {
                let n = total * hidden;
                let bytes: Vec<u8> = v[..n.min(v.len())].iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{base}h_p{}.f32.bin", chunks[0].pos), &bytes);
            }
            let mut meta = String::new();
            meta.push_str(&format!("total={total} run={} n={n} cache_len={cache_len}\n",
                                   crate::gpu::pad_to_chunk(total)));
            meta.push_str(&format!("carry.capacity={} slots={} slot={}\n", carry.capacity, carry.slots,
                                   chunks[0].slot));
            let arr = |name: &str, v: &[i32]| {
                let head: Vec<i32> = v.iter().copied().take(8).collect();
                let tail: Vec<i32> = v.iter().rev().copied().take(4).rev().collect();
                format!("{name} len={} head={head:?} tail={tail:?} min={:?} max={:?}\n",
                        v.len(), v.iter().min(), v.iter().max())
            };
            meta.push_str(&arr("cpos", &cpos));
            meta.push_str(&arr("cbase", &cbase));
            meta.push_str(&arr("slot_map", &slot_map));
            meta.push_str(&arr("state_slot", &state_slot));

            let as_i32 = |bytes: Vec<u8>| -> Vec<i32> {
                bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
            };
            if let Ok(b) = d.download(&carry.state_slot) {
                let v = as_i32(b);
                let n = (total / crate::cuda::CHUNK_GRANULE).min(v.len());
                meta.push_str(&format!("state_slot DEVICE first {n}: {:?}\n", &v[..n]));
            }
            if let Ok(b) = d.download(&carry.slot_map) {
                let v = as_i32(b);
                let n = 4.min(v.len());
                meta.push_str(&format!("slot_map DEVICE first {n}: {:?}\n", &v[..n]));
            }
            meta.push_str(&arr("seq_start", &seq_start));
            meta.push_str(&arr("seq_end", &seq_end));
            meta.push_str(&arr("chunk_start", &chunk_start));
            meta.push_str(&arr("rope_pos", &rope_pos));
            meta.push_str(&format!("rows_real={rows_real:?} rows_pad={rows_pad:?} offs={offs:?}\n"));
            meta.push_str(&format!("hist_save={hist_save:?}\n"));
            let _ = std::fs::write(format!("{base}meta_p{}.txt", chunks[0].pos), &meta);
        }
        let shape_trace = std::env::var("CLEF_PASS_SHAPE").is_ok();
        let (pass, graphed) = if std::env::var("CLEF_PASS_GRAPH").as_deref() == Ok("0") {
            (self.layers_dev_with(&h, total, map, Some(&views))?, true)
        } else {
            self.layers_graphed_with(&h, total, map, Some(&views))?
        };
        if shape_trace {
            let rows: Vec<usize> = chunks.iter().map(|c| c.rows).collect();
            eprintln!("SHAPE total={total} run={} n={} rows={rows:?} graphed={graphed}",
                      crate::gpu::pad_to_chunk(total), chunks.len());
        }

        if !graphed && !self.capturing_pass.load(std::sync::atomic::Ordering::SeqCst) {
            self.queue_capture_for(total, GraphMode::Chunked);
        }

        if let Ok(dir) = std::env::var("CLEF_DUMP_PASS") {
            let v = d.download_f32(&pass)?;

            let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
            let path = format!("{dir}/{tag}p{total}_pos{}_{}.bin", chunks[0].pos, std::process::id());
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in &v { bytes.extend_from_slice(&x.to_le_bytes()); }
            std::fs::write(&path, &bytes).ok();
            eprintln!("dump {path} rows={}", v.len() / hidden);
        }
        if trace {
            eprintln!("  PASS rows={} chunks={} host={:.1} ms",
                      chunks.iter().map(|c| c.rows).sum::<usize>(), chunks.len(),
                      t_pass.elapsed().as_secs_f64() * 1e3);
        }
        Ok(pass)
    }

    pub fn forward_multi_chunk_step(&self, chunks: &[ChunkSpec<'_>], sink: &[&PreHead]) -> Result<()> {
        assert_eq!(chunks.len(), sink.len(), "one sink per chunk");
        if chunks.is_empty() {
            return Ok(());
        }
        let d = &self.device;
        let hidden = self.backbone.config.hidden_size;
        let (pass, t_pass) = if std::env::var("CLEF_PASS_TRACE").is_ok() {
            let (p, t) = self.device.timed_dev(|| self.multi_chunk_pass(chunks))?;
            (p, t)
        } else {
            (self.multi_chunk_pass(chunks)?, std::time::Duration::ZERO)
        };
        if std::env::var("CLEF_PASS_TRACE").is_ok() {
            let rows: usize = chunks.iter().map(|c| c.rows).sum();
            let detail: Vec<String> = chunks.iter()
                .map(|c| format!("s{}p{}+{}", c.slot, c.pos, c.rows)).collect();
            eprintln!("  PASS rows={rows} chunks={} [{})] gpu={:.1} ms",
                      chunks.len(), detail.join(" "), t_pass.as_secs_f64() * 1e3);
        }

        if std::env::var("CLEF_PASS_HASH").is_ok() {
            let v = d.download_f32(&pass)?;
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for x in &v {
                for b in x.to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            eprintln!("  PASSHASH {h:016x} values {} first {:?}", v.len(), &v[..3.min(v.len())]);
        }

        assert!(chunks.len() <= kv_slots(),
                "a pass carries {} chunks but only {} KV slots are configured — raise CLEF_KV_SLOTS",
                chunks.len(), kv_slots());
        let (rows_pad, offs, _total) = pass_layout(chunks);
        for (i, sk) in sink.iter().enumerate() {
            let real = rows_pad[i].min(chunks[i].ids.len().saturating_sub(chunks[i].pos));

            assert!(real > 0,
                    "chunk {i} of a {}-chunk pass has 0 real rows (pos {} of {}): a finished request was \
                     left in the pass", chunks.len(), chunks[i].pos, chunks[i].ids.len());
            let view = d.view(&pass, offs[i] * hidden * 4, real * hidden * 4);
            sk.absorb(d, &view, real, chunks[i].pos)?;
        }
        Ok(())
    }

    pub fn forward_chunk_begin(&self, capacity: usize) -> Result<()> {
        self.forward_chunk_begin_slot(capacity, 0)
    }

    pub fn forward_chunk_begin_slot(&self, capacity: usize, slot: usize) -> Result<()> {
        self.ensure_carry_slots(capacity, slot + 1)?;
        let carry = self.carry()?;

        carry.reset_slot(&self.device, slot)?;

        let base = (slot * carry.capacity) as i32;
        self.device.write_i32(&carry.cache_base, &[base])?;

        {
            let run = crate::gpu::max_chunk_size().min(carry.capacity);
            let mbytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            self.device.copy_htod_into(&carry.cbase, &mbytes(&vec![base; run]))?;
            self.device.copy_htod_into(&carry.slot_map, &mbytes(&vec![slot as i32; run]))?;
            self.device.copy_htod_into(&carry.state_slot,
                                       &mbytes(&vec![slot as i32; run.div_ceil(crate::cuda::CHUNK_LEN)]))?;
        }
        Ok(())
    }

    pub fn forward_chunk_end(&self, tokens: usize, chunk: usize) -> Result<()> {
        if tokens > 4 * chunk {
            let freed = self.device.trim_device_pool();
            tracing::debug!(freed, tokens, "released the pool's blocks after a long request");
        }
        Ok(())
    }

    fn forward_chunked_with(&self, input_ids: &[u32], chunk: usize, capacity: usize,
                            sink: Option<&PreHead>) -> Result<Option<Buffer>> {
        let d = &self.device;
        let c = &self.backbone.config;
        let hidden = c.hidden_size;
        let tokens = input_ids.len();
        if tokens > capacity {
            return Err(crate::engine::refuse(
                tokens,
                format!(
                    "{tokens} tokens exceeds the {capacity}-token KV cache the chunked path was given. \
                     The cache is sized at load from CLEF_KV_CAPACITY (default: the server's own \
                     CLEF_MAX_LENGTH), and it cannot grow once a captured graph points at it."
                ),
            ));
        }
        self.forward_chunk_begin(capacity)?;

        let out = match sink {
            Some(_) => None,
            None => Some(d.alloc(tokens * hidden * 4)?),
        };

        let pass_trace = std::env::var("CLEF_PASS_TRACE").is_ok();
        let mut pos = 0;
        let mut pass = 0usize;
        while pos < tokens {
            let t = std::time::Instant::now();

            let rem = tokens - pos;

            let cap = crate::gpu::Rungs::body().at_or_below(chunk.max(crate::cuda::CHUNK_GRANULE));

            let rows = crate::gpu::next_chunk_rows(rem).min(cap);
            if std::env::var("CLEF_CUT_DEBUG").is_ok() {
                eprintln!("    CUT pass={pass} pos={pos} rem={rem} cap={cap} rows={rows}");
            }
            let real = rem.min(rows);
            let spec = ChunkSpec { ids: input_ids, pos, slot: 0, rows };
            let pass_buf = self.multi_chunk_pass(&[spec])?;
            match (sink, out.as_ref()) {
                (Some(sk), _) => sk.absorb(&self.device, &pass_buf, real, pos)?,
                (None, Some(o)) => {
                    self.device.copy_bytes_at(o, pos * hidden * 4, &pass_buf, 0, real * hidden * 4)?;
                }
                (None, None) => unreachable!("one of the two is always set"),
            }
            if pass_trace {
                eprintln!("    pass {pass:>3}  {real:>5} tok (rung {rows:>5})  {:>8.2} ms",
                          t.elapsed().as_secs_f64() * 1e3);
            }
            pos += real;
            pass += 1;
        }
        self.forward_chunk_end(tokens, chunk)?;
        Ok(out)
    }

    pub fn warm_up(&self, sizes: &[usize]) -> Result<WarmReport> {
        let hidden = self.backbone.config.hidden_size;
        let mut captured = 0;
        let mut skipped = 0usize;
        let mut failed: Vec<(usize, String)> = Vec::new();
        for &tokens in sizes {
            if tokens == 0 {
                continue;
            }

            let bucket = match graph_size_for(tokens) {
                Some(b) => b,
                None => continue,
            };

            let budget = graph_memory_budget(&self.device);
            let held = self.device.graph_pool_high();
            tracing::debug!(held, budget, "warm-up: pool high-water against the legacy cap");

            let run = crate::gpu::pad_to_chunk(bucket);
            let ids: Vec<u32> = vec![1; run];
            let h = self.embed_dev(&ids)?;

            let want_chunked = chunking_enabled()
                && is_pass_geometry(run)
                && !self.graph_ready_for(run, GraphMode::Chunked);
            if std::env::var("CLEF_WARM_TRACE").is_ok() {
                eprintln!("  WARM run={run} rung={} chunking={} ready={} -> {want_chunked}",
                          pass_rung(run), chunking_enabled(),
                          self.graph_ready_for(run, GraphMode::Chunked));
            }
            if want_chunked {
                self.ensure_carry(crate::gpu::kv_capacity())?;
                let carry = self.carry()?;
                {
                    let chunk = crate::gpu::max_chunk_size();
                    let k = if run > chunk { run / chunk } else { 1 };
                    let rows_each = if k > 1 { chunk } else { run };
                    if k > 1 {
                        for i in 0..k {
                            self.forward_chunk_begin_slot(crate::gpu::kv_capacity(), i)?;
                        }
                    }
                    let mut specs: Vec<ChunkSpec<'_>> = Vec::with_capacity(k);
                    for i in 0..k {
                        let lo = (i * rows_each).min(ids.len());
                        let hi = (lo + rows_each).min(ids.len());
                        specs.push(ChunkSpec { ids: &ids[lo..hi], pos: 0, slot: i, rows: rows_each });
                    }
                    match self.multi_chunk_pass(&specs).map(|_| ()) {
                        Ok(()) => captured += 1,
                        Err(e) => {
                            eprintln!("  SKIP chunked graph at {run}: {e:#}");
                            failed.push((run, format!("{e:#}")));
                            skipped += 1;
                        }
                    }
                }
            }
        }
        let _ = hidden;
        if skipped > 0 {
            tracing::warn!(skipped, captured, "some warm-up sizes are above the device's graph ceiling");
        }
        Ok(WarmReport { captured, requested: sizes.len(), failed })
    }

    pub fn graph_alloc_nodes(&self, tokens: usize) -> Option<(usize, usize)> {
        let bucket = graph_size_for(tokens)?;
        self.graphs
            .lock()
            .unwrap()
            .get(&(bucket, GraphMode::Chunked))
            .map(|g| (g.captured.alloc_nodes, g.captured.free_nodes))
    }

    fn copy_dev(&self, src: &Buffer, values: usize) -> Result<Buffer> {
        self.device.copy_dev(src, values)
    }

    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let out = self.forward_dev(input_ids)?;
        self.device.download_f32(&out)
    }

    pub fn head_forward(
        &self,
        hidden: &Buffer,
        tokens: usize,
        input_ids: &[u32],
        questions: &[QuestionSpec],
        lm_head: &HostTable,
        h: &DeviceHead,
        kv: &KvGraphs,
    ) -> Result<Vec<Vec<f32>>> {
        head_forward_with(&self.device, hidden, tokens, input_ids, questions, lm_head, h, kv)
    }
}

pub struct HeadIndices {
    pub n_q: usize,
    pub n_o: usize,
    pub tokens: usize,
    pub q_starts: Buffer,
    pub q_ends: Buffer,
    pub o_starts: Buffer,
    pub o_ends: Buffer,
    pub q_of_option: Buffer,
    pub identity_q: Buffer,
    pub zeros_q: Buffer,
    pub zeros_o: Buffer,
    pub types: Buffer,
    pub counts: Buffer,
    pub last_row: Buffer,
}

impl HeadIndices {
    pub fn build(d: &Device, tokens: usize, questions: &[QuestionSpec]) -> Result<Self> {
        let n_q = questions.len();
        let q_starts: Vec<i32> = questions.iter().map(|q| q.question_span.0 as i32).collect();
        let q_ends: Vec<i32> = questions.iter().map(|q| q.question_span.1 as i32).collect();
        let mut o_starts: Vec<i32> = Vec::new();
        let mut o_ends: Vec<i32> = Vec::new();
        let mut q_of_option: Vec<i32> = Vec::new();
        let mut counts: Vec<i32> = Vec::new();
        for (qi, q) in questions.iter().enumerate() {
            counts.push(q.option_spans.len() as i32);
            for span in &q.option_spans {
                o_starts.push(span.0 as i32);
                o_ends.push(span.1 as i32);
                q_of_option.push(qi as i32);
            }
        }
        let n_o = q_of_option.len();
        let a = HeadArrays::host(tokens, questions);
        Ok(Self {
            n_q,
            n_o,
            tokens,
            q_starts: d.upload_i32(&a.q_starts)?,
            q_ends: d.upload_i32(&a.q_ends)?,
            o_starts: d.upload_i32(&a.o_starts)?,
            o_ends: d.upload_i32(&a.o_ends)?,
            q_of_option: d.upload_i32(&a.q_of_option)?,
            identity_q: d.upload_i32(&a.identity_q)?,
            zeros_q: d.upload_i32(&a.zeros_q)?,
            zeros_o: d.upload_i32(&a.zeros_o)?,
            types: d.upload_i32(&a.types)?,
            counts: d.upload_i32(&a.counts)?,
            last_row: d.upload_i32(&a.last_row)?,
        })
    }

    pub fn cached(d: &Device, cache: &std::sync::Mutex<Option<std::sync::Arc<HeadIndices>>>,
                  tokens: usize, questions: &[QuestionSpec]) -> Result<std::sync::Arc<HeadIndices>> {
        let n_q = questions.len();
        let n_o: usize = questions.iter().map(|q| q.option_spans.len()).sum();
        if let Ok(g) = cache.lock() {
            if let Some(h) = g.as_ref() {
                if h.n_q == n_q && h.n_o == n_o && h.tokens == tokens {
                    h.rewrite(d, tokens, questions)?;
                    return Ok(std::sync::Arc::clone(h));
                }
            }
        }
        let built = std::sync::Arc::new(HeadIndices::build(d, tokens, questions)?);
        if let Ok(mut g) = cache.lock() {
            *g = Some(std::sync::Arc::clone(&built));
        }
        Ok(built)
    }

    pub fn rewrite(&self, d: &Device, tokens: usize, questions: &[QuestionSpec]) -> Result<()> {
        assert_eq!(self.n_q, questions.len(), "the head graph is keyed on the question count");
        let a = HeadArrays::host(tokens, questions);
        d.write_i32(&self.q_starts, &a.q_starts)?;
        d.write_i32(&self.q_ends, &a.q_ends)?;
        d.write_i32(&self.o_starts, &a.o_starts)?;
        d.write_i32(&self.o_ends, &a.o_ends)?;
        d.write_i32(&self.q_of_option, &a.q_of_option)?;
        d.write_i32(&self.identity_q, &a.identity_q)?;
        d.write_i32(&self.zeros_q, &a.zeros_q)?;
        d.write_i32(&self.zeros_o, &a.zeros_o)?;
        d.write_i32(&self.types, &a.types)?;
        d.write_i32(&self.counts, &a.counts)?;
        d.write_i32(&self.last_row, &a.last_row)
    }
}

struct HeadArrays {
    q_starts: Vec<i32>, q_ends: Vec<i32>,
    o_starts: Vec<i32>, o_ends: Vec<i32>,
    q_of_option: Vec<i32>, identity_q: Vec<i32>,
    zeros_q: Vec<i32>, zeros_o: Vec<i32>,
    types: Vec<i32>, counts: Vec<i32>, last_row: Vec<i32>,
}

impl HeadArrays {
    fn host(tokens: usize, questions: &[QuestionSpec]) -> Self {
        let n_q = questions.len();
        let mut o_starts = Vec::new();
        let mut o_ends = Vec::new();
        let mut q_of_option = Vec::new();
        let mut counts = Vec::new();
        for (qi, q) in questions.iter().enumerate() {
            counts.push(q.option_spans.len() as i32);
            for span in &q.option_spans {
                o_starts.push(span.0 as i32);
                o_ends.push(span.1 as i32);
                q_of_option.push(qi as i32);
            }
        }
        let n_o = q_of_option.len();

        for &q in &q_of_option {
            assert!((0..n_q as i32).contains(&q), "q_of_option reaches question {q} of {n_q}");
        }
        let q_ends: Vec<i32> = questions.iter().map(|q| q.question_span.1 as i32).collect();
        for &e in o_ends.iter().chain(q_ends.iter()) {
            assert!((0..=tokens as i32).contains(&e), "a span ends at {e} with {tokens} tokens");
        }
        Self {
            q_starts: questions.iter().map(|q| q.question_span.0 as i32).collect(),
            q_ends,
            o_starts,
            o_ends,
            q_of_option,
            identity_q: (0..n_q as i32).collect(),
            zeros_q: vec![0; n_q],
            zeros_o: vec![0; n_o],
            types: questions.iter().map(|q| q.question_type as i32).collect(),
            counts,
            last_row: vec![(tokens.max(1) - 1) as i32],
        }
    }
}

pub struct PreHead {
    norm_w: Buffer,
    norm_b: Buffer,
    memory_proj: Buffer,
    pub memory: Buffer,
    pub spans: Buffer,
    last_norm: Buffer,
    zero_row: Buffer,
    plan: Vec<Vec<(Vec<i32>, usize)>>,
    q_starts_g: Buffer,
    q_ends_g: Buffer,
    o_starts_g: Buffer,
    o_ends_g: Buffer,
    width: usize,
    hidden: usize,
    eps: f32,
    chunk: usize,
    tokens: usize,
    passes: Vec<(usize, usize)>,
}

impl PreHead {
    pub fn from_hidden(d: &Device, h: &DeviceHead, questions: &[QuestionSpec], tokens: usize,
                       hidden: &Buffer) -> Result<Self> {
        let p = Self::with_plan(d, h, questions, tokens, tokens, vec![(0, tokens)])?;
        p.absorb(d, hidden, tokens, 0)?;
        Ok(p)
    }

    pub fn dump_memory(&self, d: &Device, dir: &str, tokens: usize) -> Result<()> {
        let v = d.download_f32(&self.memory)?;

        let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
        let path = format!("{dir}/mem_{tag}t{tokens}_rows{}.bin", v.len() / self.width);
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v { bytes.extend_from_slice(&x.to_le_bytes()); }
        std::fs::write(&path, &bytes).ok();
        eprintln!("dump {path} rows={}", v.len() / self.width);
        Ok(())
    }

    pub fn new(d: &Device, h: &DeviceHead, questions: &[QuestionSpec], tokens: usize, chunk: usize)
        -> Result<Self> {
        Self::with_plan(d, h, questions, tokens, chunk, pass_plan(tokens, chunk))
    }

    fn with_plan(d: &Device, h: &DeviceHead, questions: &[QuestionSpec], tokens: usize,
                 chunk: usize, passes: Vec<(usize, usize)>) -> Result<Self> {
        let (hidden, width, eps) = (h.config.hidden_size, h.config.width, 1e-5f32);
        let n_q = questions.len();

        let mut all: Vec<(usize, usize)> = questions.iter().map(|q| q.question_span).collect();
        for q in questions {
            all.extend(q.option_spans.iter().copied());
        }

        let span_rows: usize = all.iter().map(|(a, b)| b - a).sum();
        let spans = d.alloc(span_rows * hidden * 4)?;

        let ladder_rows = passes.iter().map(|(b, r)| b + r).max().unwrap_or(tokens).max(tokens);
        let memory = d.alloc(ladder_rows * width * 4)?;

        let npasses = passes.len();
        let mut plan: Vec<Vec<(Vec<i32>, usize)>> = vec![Vec::new(); npasses];
        let mut dest = 0usize;
        let mut gq: Vec<(i32, i32)> = Vec::new();
        let mut go: Vec<(i32, i32)> = Vec::new();
        for (si, (start, end)) in all.iter().enumerate() {
            let g0 = dest as i32;
            for (pi, &(base, prows)) in passes.iter().enumerate() {
                let lo = (*start).max(base);
                let hi = (*end).min(base + prows);
                if lo >= hi { continue; }
                let local: Vec<i32> = (lo..hi).map(|r| (r - base) as i32).collect();
                plan[pi].push((local, dest));
                dest += hi - lo;
            }
            let g1 = dest as i32;
            if si < n_q { gq.push((g0, g1)) } else { go.push((g0, g1)) }
        }
        debug_assert_eq!(dest, span_rows, "the plan covers every span row exactly once");
        let dev = |v: &[(i32, i32)]| -> Result<(Buffer, Buffer)> {
            let a: Vec<i32> = v.iter().map(|x| x.0).collect();
            let b: Vec<i32> = v.iter().map(|x| x.1).collect();
            Ok((d.upload_i32(&a)?, d.upload_i32(&b)?))
        };
        let (q_starts_g, q_ends_g) = dev(&gq)?;
        let (o_starts_g, o_ends_g) = dev(&go)?;
        Ok(Self {
            norm_w: h.hidden_norm_w.clone_buffer(),
            norm_b: h.hidden_norm_b.clone_buffer(),
            memory_proj: h.memory_projection.clone_buffer(),
            memory,
            spans,
            last_norm: d.alloc(hidden * 4)?,
            zero_row: d.upload_i32(&[0])?,
            plan,
            q_starts_g,
            q_ends_g,
            o_starts_g,
            o_ends_g,
            width,
            hidden,
            eps,
            chunk: chunk.max(1),
            tokens,
            passes,
        })
    }

    pub fn absorb(&self, d: &Device, hidden_rows: &Buffer, rows: usize, pos: usize) -> Result<()> {
        let tr = std::env::var("CLEF_PASS_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let normalised =
            d.layer_norm_dev(hidden_rows, &self.norm_w, &self.norm_b, rows, self.hidden, self.eps)?;
        let t_norm = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let bf = d.to_bf16_dev(&normalised, rows * self.hidden)?;
        let t_bf = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = std::time::Instant::now();
        let mem = d.bf16_gemm_dev(&bf, &self.memory_proj, rows, self.width, self.hidden)?;
        let t_gemm = t2.elapsed().as_secs_f64() * 1e3;

        if let Ok(dir) = std::env::var("CLEF_ABSORB_DUMP") {
            let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
            let seq = ABSORB_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = std::fs::create_dir_all(&dir);
            for (name, buf, width) in [("norm", &normalised, self.hidden),
                                       ("mem", &mem, self.width)] {
                if let Ok(v) = d.download_f32(buf) {
                    let n = (rows * width).min(v.len());
                    let bytes: Vec<u8> = v[..n].iter().flat_map(|x| x.to_le_bytes()).collect();

                    let fp = format!("f{:.6}_{:.6}_{:.6}_{:.6}", v[0], v[1], v[2], v[3]);
                    let _ = std::fs::write(format!("{dir}/{seq:03}_{tag}{name}_{fp}_p{pos}_r{rows}.f32.bin"), &bytes);
                }
            }
        }
        let t3 = std::time::Instant::now();
        d.copy_bytes_at(&self.memory, pos * self.width * 4, &mem, 0, rows * self.width * 4)?;
        let t_copy = t3.elapsed().as_secs_f64() * 1e3;
        if tr {
            let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
            eprintln!("    absorb{tag} pos={pos} rows={rows} tokens={} npass={} memory_rows={} norm {:.2} bf16 {:.2} gemm {:.2} copy {:.2} ms",
                      self.tokens, self.passes.len(), self.memory.len() / (self.width * 4),
                      t_norm, t_bf, t_gemm, t_copy);
        }
        if pos + rows == self.tokens {
            let g = d.gather_rows_dev(&normalised, &[(rows - 1) as i32], 1, self.hidden)?;
            d.copy_bytes_at(&self.last_norm, 0, &g, 0, self.hidden * 4)?;
        }

        let pi = self.passes.iter().position(|(b, _)| *b == pos)
            .unwrap_or_else(|| pos / self.chunk.max(1));

        if let Some(frags) = self.plan.get(pi) {
            let frags: Vec<&(Vec<i32>, usize)> = frags.iter()
                .filter(|(local, _)| local.iter().all(|r| *r >= 0 && (*r as usize) < rows))
                .collect();
            let total: usize = frags.iter().map(|(local, _)| local.len()).sum();
            if total > 0 {
                let mut idx: Vec<i32> = Vec::with_capacity(total);
                for (local, _) in frags.iter() {
                    idx.extend_from_slice(local);
                }
                let tg = std::time::Instant::now();
                let g = d.gather_rows_dev(&normalised, &idx, total, self.hidden)?;
                if tr { eprintln!("    absorb@{} span-gather {} rows {:.2} ms", pos, total, tg.elapsed().as_secs_f64() * 1e3); }

                let mut contiguous = true;
                let mut expect = frags.first().map(|(_, dest)| *dest).unwrap_or(0);
                for (local, dest) in frags.iter() {
                    if *dest != expect { contiguous = false; break; }
                    expect += local.len();
                }
                if contiguous {
                    let first = frags.first().map(|(_, dest)| *dest).unwrap_or(0);
                    d.copy_bytes_at(&self.spans, first * self.hidden * 4, &g, 0,
                                    total * self.hidden * 4)?;
                } else {
                    let mut at = 0usize;
                    for (local, dest) in frags {
                        d.copy_bytes_at(&self.spans, dest * self.hidden * 4, &g,
                                        at * self.hidden * 4, local.len() * self.hidden * 4)?;
                        at += local.len();
                    }
                }
            }
        }
        Ok(())
    }
}

fn head_body(
    d: &Device,
    hidden: &Buffer,
    tokens: usize,
    idx: &HeadIndices,
    lex_dev: &Buffer,
    logits_out: &Buffer,
    questions: &[QuestionSpec],
    h: &DeviceHead,
    kv: &KvGraphs,
    kreal: &Buffer,
    pre: Option<&PreHead>,
) -> Result<()> {
    {
        let width = h.config.width;
        let hidden_size = h.config.hidden_size;
        let heads = h.config.heads;
        let ff = h.config.feedforward;

        let eps = 1e-5f32;
        let (n_q, n_o) = (idx.n_q, idx.n_o);
        let (normalised, memory, q_means, o_ctx) = match pre {
            Some(p) => (
                None,
                p.memory.clone_buffer(),
                d.mean_span_buf_dev(&p.spans, &p.q_starts_g, &p.q_ends_g, n_q, hidden_size)?,
                d.mean_span_buf_dev(&p.spans, &p.o_starts_g, &p.o_ends_g, n_o, hidden_size)?,
            ),
            None => {
                let normalised = d.layer_norm_dev(hidden, &h.hidden_norm_w, &h.hidden_norm_b, tokens,
                                                  hidden_size, eps)?;

                let memory = d.bf16_gemm_dev(&d.to_bf16_dev(&normalised, tokens * hidden_size)?,
                                             &h.memory_projection, tokens, width, hidden_size)?;

                let q_means =
                    d.mean_span_buf_dev(&normalised, &idx.q_starts, &idx.q_ends, n_q, hidden_size)?;
                let o_ctx =
                    d.mean_span_buf_dev(&normalised, &idx.o_starts, &idx.o_ends, n_o, hidden_size)?;
                (Some(normalised), memory, q_means, o_ctx)
            }
        };

        let a = d.bf16_gemm_dev(&d.to_bf16_dev(&o_ctx, n_o * hidden_size)?,
                                &h.option_context_projection, n_o, width, hidden_size)?;
        let b = d.bf16_gemm_dev(&d.to_bf16_dev(&lex_dev, n_o * hidden_size)?,
                                &h.option_lexical_projection, n_o, width, hidden_size)?;
        let c = d.bf16_gemm_dev(&d.to_bf16_dev(&q_means, n_q * hidden_size)?,
                                &h.option_question_projection, n_q, width, hidden_size)?;
        let mut routed = d.add_dev(&a, &b, n_o * width)?;
        d.accumulate_indexed_buf_dev(&mut routed, &c, &idx.q_of_option, n_o, width)?;

        if std::env::var("CLEF_HEAD_TRACE").is_ok() {
            let stat = |name: &str, v: &[f32]| {
                let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
                let mx = v.iter().fold(f32::MIN, |a, x| a.max(*x));
                let amx = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                eprintln!("    dev {name:20} n={:6} mean={mean:+.6} max={mx:+.6} amax={amx:.6}",
                          v.len());
            };
            if let Some(n) = &normalised {
                stat("normalised", &d.download_f32(n)?);
            }
            stat("memory", &d.download_f32(&memory)?);
            stat("q_means", &d.download_f32(&q_means)?);
            stat("o_ctx", &d.download_f32(&o_ctx)?);
            stat("option_queries", &d.download_f32(&routed)?);
        }

        let __trace = std::env::var("CLEF_HEAD_TRACE").is_ok();
        let __t0 = std::time::Instant::now();
        let mut __mark = __t0;
        macro_rules! __stage { ($name:expr) => { if __trace { eprintln!("    head {:>10} {:>8.3} ms (cum {:>8.3})", $name, __mark.elapsed().as_secs_f64()*1e3, __t0.elapsed().as_secs_f64()*1e3); __mark = std::time::Instant::now(); } } }
        __stage!("prep");

        kv.stage_memory(d, &memory, tokens)?;
        let chunks = Rungs::head().decompose(tokens);
        for (li, layer) in h.evidence.iter().enumerate() {
            let qn = d.layer_norm_dev(&routed, &layer.query_norm_w, &layer.query_norm_b,
                                      n_o, width, eps)?;
            let qp = d.bf16_linear_dev(&d.to_bf16_dev(&qn, n_o * width)?, &layer.q_w, &layer.q_b,
                                       n_o, width, width)?;

            let attn = d.mha_chunks_dev(&qp, n_o, tokens, kreal, width, heads, &chunks,
                                        |dev, c, _rows| {
                let (base, rows) = chunks[c];
                kv.replay(dev, li, base, rows)?;
                Ok((kv.k().clone_buffer(), kv.v().clone_buffer()))
            })?;
            let attn = d.bf16_linear_dev(&d.to_bf16_dev(&attn, n_o * width)?, &layer.out_w,
                                         &layer.out_b, n_o, width, width)?;
            routed = d.add_dev(&routed, &attn, n_o * width)?;

            let nf = d.layer_norm_dev(&routed, &layer.ff_norm_w, &layer.ff_norm_b,
                                      n_o, width, eps)?;
            let f1 = d.bf16_linear_dev(&d.to_bf16_dev(&nf, n_o * width)?, &layer.ff1_w,
                                       &layer.ff1_b, n_o, ff, width)?;
            let f1 = d.gelu_erf_dev(&f1, n_o * ff)?;
            let f2 = d.bf16_linear_dev(&d.to_bf16_dev(&f1, n_o * ff)?, &layer.ff2_w,
                                       &layer.ff2_b, n_o, width, ff)?;
            routed = d.add_dev(&routed, &f2, n_o * width)?;
        }

        let base_fields = d.bf16_gemm_dev(&d.to_bf16_dev(&q_means, n_q * hidden_size)?,
                                          &h.question_projection, n_q, width, hidden_size)?;
        let summaries = d.summary_softmax_buf_dev(&routed, &base_fields, &idx.counts, n_q, width)?;
        let summaries = d.layer_norm_dev(&summaries, &h.option_summary_norm_w,
                                         &h.option_summary_norm_b, n_q, width, eps)?;

        let global_vec = match (&normalised, pre) {
            (Some(n), _) => d.gather_rows_buf_dev(n, &idx.last_row, 1, hidden_size)?,
            (None, Some(p)) => d.gather_rows_buf_dev(&p.last_norm, &p.zero_row, 1, hidden_size)?,
            (None, None) => unreachable!("one of the two is always set"),
        };
        let global_proj = d.bf16_gemm_dev(&d.to_bf16_dev(&global_vec, hidden_size)?,
                                          &h.global_projection, 1, width, hidden_size)?;
        let mut fields = base_fields;
        d.accumulate_indexed_buf_dev(&mut fields, &summaries, &idx.identity_q, n_q, width)?;
        d.accumulate_indexed_buf_dev(&mut fields, &global_proj, &idx.zeros_q, n_q, width)?;
        __stage!("evidence");
        d.accumulate_indexed_buf_dev(&mut fields, &h.type_embedding, &idx.types, n_q, width)?;
        for (li, layer) in h.decoder.iter().enumerate() {
            let n1 = d.layer_norm_dev(&fields, &layer.norm1_w, &layer.norm1_b, n_q, width, eps)?;
            let bf = d.to_bf16_dev(&n1, n_q * width)?;
            let qp = d.bf16_linear_dev(&bf, &layer.self_q_w, &layer.self_q_b, n_q, width, width)?;
            let kp = d.bf16_linear_dev(&bf, &layer.self_k_w, &layer.self_k_b, n_q, width, width)?;
            let vp = d.bf16_linear_dev(&bf, &layer.self_v_w, &layer.self_v_b, n_q, width, width)?;
            let attn = d.mha_dev(&qp, &kp, &vp, n_q, n_q, kreal, width, heads)?;
            __stage!("  self-mha");
            let attn = d.bf16_linear_dev(&d.to_bf16_dev(&attn, n_q * width)?, &layer.self_out_w,
                                         &layer.self_out_b, n_q, width, width)?;
            fields = d.add_dev(&fields, &attn, n_q * width)?;
            __stage!("  self-out");

            let n2 = d.layer_norm_dev(&fields, &layer.norm2_w, &layer.norm2_b, n_q, width, eps)?;
            let qp = d.bf16_linear_dev(&d.to_bf16_dev(&n2, n_q * width)?, &layer.cross_q_w,
                                       &layer.cross_q_b, n_q, width, width)?;

            let attn = d.mha_chunks_dev(&qp, n_q, tokens, kreal, width, heads, &chunks,
                                        |dev, c, _rows| {
                let (base, rows) = chunks[c];
                kv.replay(dev, h.evidence.len() + li, base, rows)?;
                Ok((kv.k().clone_buffer(), kv.v().clone_buffer()))
            })?;
            __stage!("  cross-mha");
            let attn = d.bf16_linear_dev(&d.to_bf16_dev(&attn, n_q * width)?, &layer.cross_out_w,
                                         &layer.cross_out_b, n_q, width, width)?;
            fields = d.add_dev(&fields, &attn, n_q * width)?;

            let n3 = d.layer_norm_dev(&fields, &layer.norm3_w, &layer.norm3_b, n_q, width, eps)?;
            let f1 = d.bf16_linear_dev(&d.to_bf16_dev(&n3, n_q * width)?, &layer.ff1_w,
                                       &layer.ff1_b, n_q, ff, width)?;
            let f1 = d.gelu_erf_dev(&f1, n_q * ff)?;
            __stage!("  cross-out");
            let f2 = d.bf16_linear_dev(&d.to_bf16_dev(&f1, n_q * ff)?, &layer.ff2_w,
                                       &layer.ff2_b, n_q, width, ff)?;
            fields = d.add_dev(&fields, &f2, n_q * width)?;
            __stage!("  ffn");
        }

        __stage!("decoder");
        let fields_normed = d.layer_norm_dev(&fields, &h.field_norm_w, &h.field_norm_b,
                                             n_q, width, eps)?;

        if std::env::var("CLEF_HEAD_TRACE").is_ok() {
            let stat = |name: &str, v: &[f32]| {
                let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
                let mx = v.iter().fold(f32::MIN, |a, x| a.max(*x));
                let amx = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                eprintln!("    dev {name:20} n={:6} mean={mean:+.6} max={mx:+.6} amax={amx:.6}",
                          v.len());
            };
            stat("routed (post-evidence)", &d.download_f32(&routed)?);
            stat("field_normed", &d.download_f32(&fields_normed)?);
        }

        let prior_scale = h.prior_logit_scale.min(100.0f32.ln()).exp();
        let joint_scale = h.joint_logit_scale.min(100.0f32.ln()).exp();
        let gate = 1.0 / (1.0 + (-h.residual_gate).exp());

        let mut anchor = d.gather_rows_buf_dev(&q_means, &idx.identity_q, n_q, hidden_size)?;
        d.accumulate_indexed_buf_dev(&mut anchor, &global_vec, &idx.zeros_q, n_q, hidden_size)?;
        let anchor = d.l2_normalise_dev(&anchor, n_q, hidden_size)?;
        let lex_anchor = d.l2_normalise_dev(&lex_dev, n_o, hidden_size)?;
        let prior = d.head_pair_dot_buf_dev(&lex_anchor, &anchor, &idx.q_of_option, n_o, hidden_size,
                                        prior_scale)?;

        let option_ln = d.layer_norm_dev(&routed, &h.option_norm_w, &h.option_norm_b,
                                         n_o, width, eps)?;
        let option_for_cos = d.l2_normalise_dev(&option_ln, n_o, width)?;
        let field_n = d.l2_normalise_dev(&fields_normed, n_q, width)?;
        let cosine = d.head_pair_dot_buf_dev(&option_for_cos, &field_n, &idx.q_of_option, n_o, width, 1.0)?;

        let field_for_option = d.gather_rows_buf_dev(&fields_normed, &idx.q_of_option, n_o, width)?;
        let features = d.head_features_dev(&field_for_option, &option_ln, n_o, width)?;
        let h1 = d.bf16_linear_dev(&d.to_bf16_dev(&features, n_o * 4 * width)?, &h.residual_w0,
                                   &h.residual_b0, n_o, width, 4 * width)?;
        let h1 = d.gelu_erf_dev(&h1, n_o * width)?;

        let residual = d.head_pair_dot_buf_dev(&h1, &h.residual_w3, &idx.zeros_o, n_o, width, 1.0)?;
        let residual = d.bias_add_dev(&residual, &h.residual_b3, n_o, 1)?;
        if std::env::var("CLEF_HEAD_TRACE").is_ok() {
            let stat = |name: &str, v: &[f32]| {
                let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
                let amx = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                eprintln!("    dev {name:20} n={:6} mean={mean:+.6} amax={amx:.6}", v.len());
            };
            stat("prior", &d.download_f32(&prior)?);
            stat("cosine", &d.download_f32(&cosine)?);
            stat("residual", &d.download_f32(&residual)?);

            let oln = d.download_f32(&option_ln)?;
            for q in 0..n_q {
                let qoo: Vec<i32> = questions.iter().enumerate().flat_map(|(qi, qq)|
                        std::iter::repeat(qi as i32).take(qq.option_spans.len())).collect();
                let rows: Vec<usize> = (0..n_o).filter(|&o| qoo[o] as usize == q).collect();
                let (mut acc, mut mx, mut n) = (0.0f32, 0.0f32, 0usize);
                for &r in &rows {
                    for c in 0..width {
                        let v = oln[r * width + c].abs();
                        acc += v;
                        mx = mx.max(v);
                        n += 1;
                    }
                }
                eprintln!("    dev option_ln q{q}      mean|x|={:.6} max={:.6}",
                          acc / n.max(1) as f32, mx);
            }
            eprintln!("    dev prior values      {:?}", d.download_f32(&prior)?);
            eprintln!("    dev cosine values     {:?}", d.download_f32(&cosine)?);
            eprintln!("    dev residual values   {:?}", d.download_f32(&residual)?);
            eprintln!("    dev scales: prior_scale={prior_scale:.6} joint_scale={joint_scale:.6} \
                       gate={gate:.6} (raw prior_logit_scale={} joint={} gate_raw={})",
                      h.prior_logit_scale, h.joint_logit_scale, h.residual_gate);
        }

        let logits = d.head_logit_dev(&prior, &cosine, &residual, joint_scale, gate)?;

        d.copy_into(logits_out, &logits, n_o)?;
        Ok(())
    }
}

fn lexical_vectors(d: &Device, input_ids: &[u32], questions: &[QuestionSpec], lm_head: &HostTable,
                   h: &DeviceHead) -> Result<Buffer> {
    let hidden_size = h.config.hidden_size;
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for q in questions {
        for span in &q.option_spans {
            spans.push(*span);
        }
    }
    let mut lexical = vec![0.0f32; spans.len() * hidden_size];
    for (o, span) in spans.iter().enumerate() {
        let ids: Vec<u32> = input_ids[span.0..span.1].to_vec();
        let mut rows = vec![0.0f32; ids.len() * hidden_size];
        lm_head.gather_f32(&ids, &mut rows);
        let inv = if ids.is_empty() { 0.0 } else { 1.0 / ids.len() as f32 };
        for r in 0..ids.len() {
            for c in 0..hidden_size {
                lexical[o * hidden_size + c] += rows[r * hidden_size + c] * inv;
            }
        }
    }
    d.upload_f32(&lexical)
}

pub fn head_forward_pre_eager(d: &Device, pre: &PreHead, tokens: usize, input_ids: &[u32],
                              questions: &[QuestionSpec], lm_head: &HostTable, h: &DeviceHead)
    -> Result<Vec<Vec<f32>>> {
    let kv = KvGraphs::empty(d, h)?;
    head_forward_pre_eager_with(d, pre, tokens, input_ids, questions, lm_head, h, &kv)
}

pub fn head_forward(d: &Device, hidden: &Buffer, tokens: usize, input_ids: &[u32],
                    questions: &[QuestionSpec], lm_head: &HostTable, h: &DeviceHead)
    -> Result<Vec<Vec<f32>>> {
    let kv = KvGraphs::empty(d, h)?;
    head_forward_with(d, hidden, tokens, input_ids, questions, lm_head, h, &kv)
}

pub fn head_forward_pre_eager_with(
    d: &Device,
    pre: &PreHead,
    tokens: usize,
    input_ids: &[u32],
    questions: &[QuestionSpec],
    lm_head: &HostTable,
    h: &DeviceHead,
    kv: &KvGraphs,
) -> Result<Vec<Vec<f32>>> {
    let idx = HeadIndices::cached(d, &h.index_cache, tokens, questions)?;
    let lex_dev = lexical_vectors(d, input_ids, questions, lm_head, h)?;
    let logits = d.alloc(idx.n_o * 4)?;
    let kreal = d.upload_i32(&[tokens as i32])?;
    head_body(d, &pre.memory, tokens, &idx, &lex_dev, &logits, questions, h, kv, &kreal, Some(pre))?;
    split_head_logits(d, &logits, questions)
}

pub fn head_forward_with(
    d: &Device,
    hidden: &Buffer,
    tokens: usize,
    input_ids: &[u32],
    questions: &[QuestionSpec],
    lm_head: &HostTable,
    h: &DeviceHead,
    kv: &KvGraphs,
) -> Result<Vec<Vec<f32>>> {
    let timing = std::env::var("CLEF_HEAD_TIME").is_ok();
    let t0 = std::time::Instant::now();
    let idx = HeadIndices::cached(d, &h.index_cache, tokens, questions)?;
    let t_idx = t0.elapsed();
    let t1 = std::time::Instant::now();
    let lex_dev = lexical_vectors(d, input_ids, questions, lm_head, h)?;
    let t_lex = t1.elapsed();
    let logits = d.alloc(idx.n_o * 4)?;
    let t2 = std::time::Instant::now();
    let kreal = d.upload_i32(&[tokens as i32])?;
    head_body(d, hidden, tokens, &idx, &lex_dev, &logits, questions, h, kv, &kreal, None)?;
    let t_body = t2.elapsed();
    let t3 = std::time::Instant::now();
    let out = split_head_logits(d, &logits, questions)?;
    let t_dl = t3.elapsed();
    if timing {
        eprintln!(
            "    head: indices {:>7.3}ms | lexical {:>7.3}ms | body(submit) {:>7.3}ms | \
             download+drain {:>7.3}ms",
            t_idx.as_secs_f64() * 1e3, t_lex.as_secs_f64() * 1e3,
            t_body.as_secs_f64() * 1e3, t_dl.as_secs_f64() * 1e3
        );
    }
    Ok(out)
}

pub struct KvGraphs {
    captured: Vec<Vec<crate::cuda::Captured>>,
    layers: Vec<KvLayer>,
    base: Buffer,
    zero: Buffer,
    memory: Buffer,
    k: Buffer,
    v: Buffer,
    width: usize,
}

struct KvLayer {
    norm_memory: bool,
    nw: Buffer,
    nb: Buffer,
    kw: Buffer,
    kb: Buffer,
    vw: Buffer,
    vb: Buffer,
}

impl KvGraphs {
    pub fn capture(d: &Device, h: &DeviceHead) -> Result<Self> {
        let mut me = Self::buffers(d, h)?;
        for layer in 0..me.layers.len() {
            let mut row = Vec::with_capacity(HEAD_RUNGS.len());
            for &p in HEAD_RUNGS.iter() {
                let body = || me.build(d, layer, p);
                for _ in 0..3 {
                    body()?;
                    d.sync()?;
                }
                row.push(d.capture(body)?);
            }
            me.captured.push(row);
        }
        Ok(me)
    }

    pub fn empty(d: &Device, h: &DeviceHead) -> Result<Self> {
        Self::buffers(d, h)
    }

    fn buffers(d: &Device, h: &DeviceHead) -> Result<Self> {
        let width = h.config.width;
        let pmax = *HEAD_RUNGS.last().expect("ladder is non-empty");
        let base = d.upload_i32(&[0])?;
        let zero = d.upload_i32(&[0])?;

        let rows_max = crate::gpu::head_staging_rows();
        let memory = d.alloc(rows_max * width * 4)?;
        let k = d.alloc(pmax * width * 4)?;
        let v = d.alloc(pmax * width * 4)?;
        let mut layers = Vec::with_capacity(h.evidence.len() + h.decoder.len());
        for l in &h.evidence {
            layers.push(KvLayer {
                norm_memory: true,
                nw: l.memory_norm_w.clone_buffer(), nb: l.memory_norm_b.clone_buffer(),
                kw: l.k_w.clone_buffer(), kb: l.k_b.clone_buffer(),
                vw: l.v_w.clone_buffer(), vb: l.v_b.clone_buffer(),
            });
        }
        for l in &h.decoder {
            layers.push(KvLayer {
                norm_memory: false,
                nw: l.norm1_w.clone_buffer(), nb: l.norm1_b.clone_buffer(),
                kw: l.cross_k_w.clone_buffer(), kb: l.cross_k_b.clone_buffer(),
                vw: l.cross_v_w.clone_buffer(), vb: l.cross_v_b.clone_buffer(),
            });
        }
        Ok(Self { captured: Vec::new(), layers, base, zero, memory, k, v, width })
    }

    fn build(&self, d: &Device, layer: usize, rows: usize) -> Result<()> {
        let l = &self.layers[layer];

        let src = if l.norm_memory {
            d.layer_norm_dev_at(&self.memory, &l.nw, &l.nb, rows, self.width, 1e-5, &self.base)?
        } else {
            d.to_bf16_dev_at(&self.memory, rows * self.width, self.width, &self.base)?
        };
        let bf = if l.norm_memory {
            d.to_bf16_dev_at(&src, rows * self.width, self.width, &self.zero)?
        } else { src };
        let kk = d.bf16_linear_dev(&bf, &l.kw, &l.kb, rows, self.width, self.width)?;
        let vv = d.bf16_linear_dev(&bf, &l.vw, &l.vb, rows, self.width, self.width)?;
        d.copy_into(&self.k, &kk, rows * self.width)?;
        d.copy_into(&self.v, &vv, rows * self.width)?;
        Ok(())
    }

    pub fn stage_memory(&self, d: &Device, src: &Buffer, tokens: usize) -> Result<()> {
        if std::env::var("CLEF_KV_TRACE").is_ok() {
            eprintln!("KVTRACE stage tokens={tokens} src_bytes={} kv_memory_bytes={} width={}",
                      src.len(), self.memory.len(), self.width);
        }

        let n = tokens * self.width * 4;

        if let Ok(dir) = std::env::var("CLEF_KV_STAGE_DUMP") {
            let tag = std::env::var("CLEF_LAYER_TAG").unwrap_or_default();
            let seq = STAGE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = std::fs::create_dir_all(&dir);
            if let Ok(v) = d.download_f32(src) {
                let k = (tokens * self.width).min(v.len());

                let fp = format!("f{:.6}_{:.6}_{:.6}_{:.6}",
                                 v[0], v[1], v[2], v[3]);
                let bytes: Vec<u8> = v[..k].iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/{seq:03}_{tag}{fp}_t{tokens}_rows{k}.f32.bin"), &bytes);
            }
        }
        d.copy_bytes_at(&self.memory, 0, src, 0, n)
    }

    pub fn replay(&self, d: &Device, layer: usize, base: usize, rows: usize) -> Result<()> {
        let read_base = base;

        if std::env::var("CLEF_SKIP_BASE_WRITE").is_err() {
            d.write_i32(&self.base, &[read_base as i32])?;
        }
        if self.captured.is_empty() {
            return self.build(d, layer, rows);
        }
        let rung = HEAD_RUNGS.iter().position(|r| *r == rows)
            .with_context(|| format!("{rows} is not a ladder rung"))?;
        self.captured[layer][rung].launch()?;
        Ok(())
    }

    pub fn k(&self) -> &Buffer { &self.k }
    pub fn v(&self) -> &Buffer { &self.v }
    pub fn len(&self) -> usize { self.captured.iter().map(|r| r.len()).sum() }
}

fn split_head_logits(d: &Device, logits: &Buffer, questions: &[QuestionSpec])
    -> Result<Vec<Vec<f32>>> {
    let flat = d.download_f32(logits)?;
    let mut out = Vec::with_capacity(questions.len());
    let mut off = 0usize;
    for q in questions {
        let n = q.option_spans.len();
        out.push(flat[off..off + n].to_vec());
        off += n;
    }
    Ok(out)
}

pub fn layer_kinds(config: &TextConfig) -> Vec<LayerKind> {
    (0..config.num_hidden_layers).map(|i| config.layer_kind(i)).collect()
}

pub struct DeviceEvidence {
    pub query_norm_w: Buffer,
    pub query_norm_b: Buffer,
    pub memory_norm_w: Buffer,
    pub memory_norm_b: Buffer,
    pub q_w: Buffer,
    pub k_w: Buffer,
    pub v_w: Buffer,
    pub q_b: Buffer,
    pub k_b: Buffer,
    pub v_b: Buffer,
    pub out_w: Buffer,
    pub out_b: Buffer,
    pub ff_norm_w: Buffer,
    pub ff_norm_b: Buffer,
    pub ff1_w: Buffer,
    pub ff1_b: Buffer,
    pub ff2_w: Buffer,
    pub ff2_b: Buffer,
}

pub struct DeviceDecoder {
    pub norm1_w: Buffer,
    pub norm1_b: Buffer,
    pub self_q_w: Buffer,
    pub self_k_w: Buffer,
    pub self_v_w: Buffer,
    pub self_q_b: Buffer,
    pub self_k_b: Buffer,
    pub self_v_b: Buffer,
    pub self_out_w: Buffer,
    pub self_out_b: Buffer,
    pub norm2_w: Buffer,
    pub norm2_b: Buffer,
    pub cross_q_w: Buffer,
    pub cross_k_w: Buffer,
    pub cross_v_w: Buffer,
    pub cross_q_b: Buffer,
    pub cross_k_b: Buffer,
    pub cross_v_b: Buffer,
    pub cross_out_w: Buffer,
    pub cross_out_b: Buffer,
    pub norm3_w: Buffer,
    pub norm3_b: Buffer,
    pub ff1_w: Buffer,
    pub ff1_b: Buffer,
    pub ff2_w: Buffer,
    pub ff2_b: Buffer,
}

pub type HeadIndexCache = std::sync::Mutex<Option<std::sync::Arc<HeadIndices>>>;

pub struct DeviceHead {
    pub hidden_norm_w: Buffer,
    pub hidden_norm_b: Buffer,
    pub memory_projection: Buffer,
    pub question_projection: Buffer,
    pub option_question_projection: Buffer,
    pub global_projection: Buffer,
    pub option_context_projection: Buffer,
    pub option_lexical_projection: Buffer,
    pub type_embedding: Buffer,
    pub index_cache: HeadIndexCache,
    pub evidence: Vec<DeviceEvidence>,
    pub option_summary_norm_w: Buffer,
    pub option_summary_norm_b: Buffer,
    pub decoder: Vec<DeviceDecoder>,
    pub field_norm_w: Buffer,
    pub field_norm_b: Buffer,
    pub option_norm_w: Buffer,
    pub option_norm_b: Buffer,
    pub residual_w0: Buffer,
    pub residual_b0: Buffer,
    pub residual_w3: Buffer,
    pub residual_b3: Buffer,
    pub prior_logit_scale: f32,
    pub joint_logit_scale: f32,
    pub residual_gate: f32,
    pub config: crate::model::head::HeadConfig,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn load_head(device: &Device, ck: &Checkpoint, config: crate::model::head::HeadConfig)
    -> Result<DeviceHead> {
    let width = config.width;
    let hidden = config.hidden_size;
    let heads = config.heads;
    let ff = config.feedforward;

    fn bf16(device: &Device, ck: &Checkpoint, name: &str, values: usize) -> Result<Buffer> {
        let bytes = ck.with_raw(name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::BF16 {
                anyhow::bail!("{name} is {dtype:?}, and the head is BF16 throughout");
            }
            if data.len() != values * 2 {
                anyhow::bail!("{name} has {} bytes ({shape:?}), expected {}", data.len(), values * 2);
            }
            Ok(data.to_vec())
        })?;
        device.upload(&bytes)
    }

    fn f32vec(ck: &Checkpoint, name: &str, values: usize) -> Result<Vec<f32>> {
        ck.with_raw(name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::BF16 {
                anyhow::bail!("{name} is {dtype:?}, and the head is BF16 throughout");
            }
            if data.len() != values * 2 {
                anyhow::bail!("{name} has {} bytes ({shape:?}), expected {}", data.len(), values * 2);
            }
            Ok(data.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
        })
    }

    let vec_dev = |name: &str, values: usize| -> Result<Buffer> {
        device.upload_f32(&f32vec(ck, name, values)?)
    };
    let scalar = |name: &str| -> Result<f32> {
        Ok(f32vec(ck, name, 1)?[0])
    };

    let split_in = |name: &str| -> Result<(Buffer, Buffer, Buffer, Buffer, Buffer, Buffer)> {
        let w_name = format!("{name}.in_proj_weight");
        let b_name = format!("{name}.in_proj_bias");
        let w = ck.with_raw(&w_name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::BF16 {
                anyhow::bail!("{w_name} is {dtype:?}");
            }
            if data.len() != 3 * width * width * 2 {
                anyhow::bail!("{w_name} has {} bytes ({shape:?}), expected {}",
                              data.len(), 3 * width * width * 2);
            }
            Ok(data.to_vec())
        })?;
        let b = f32vec(ck, &b_name, 3 * width)?;
        let block = width * width * 2;
        Ok((
            device.upload(&w[..block])?,
            device.upload(&w[block..2 * block])?,
            device.upload(&w[2 * block..])?,
            device.upload_f32(&b[..width])?,
            device.upload_f32(&b[width..2 * width])?,
            device.upload_f32(&b[2 * width..])?,
        ))
    };

    let mut evidence = Vec::with_capacity(config.routing_layers);
    for i in 0..config.routing_layers {
        let p = format!("evidence_layers.{i}");
        let (q_w, k_w, v_w, q_b, k_b, v_b) = split_in(&format!("{p}.attention"))?;
        evidence.push(DeviceEvidence {
            query_norm_w: vec_dev(&format!("{p}.query_norm.weight"), width)?,
            query_norm_b: vec_dev(&format!("{p}.query_norm.bias"), width)?,
            memory_norm_w: vec_dev(&format!("{p}.memory_norm.weight"), width)?,
            memory_norm_b: vec_dev(&format!("{p}.memory_norm.bias"), width)?,
            q_w, k_w, v_w, q_b, k_b, v_b,
            out_w: bf16(device, ck, &format!("{p}.attention.out_proj.weight"), width * width)?,
            out_b: vec_dev(&format!("{p}.attention.out_proj.bias"), width)?,
            ff_norm_w: vec_dev(&format!("{p}.feedforward_norm.weight"), width)?,
            ff_norm_b: vec_dev(&format!("{p}.feedforward_norm.bias"), width)?,
            ff1_w: bf16(device, ck, &format!("{p}.feedforward.0.weight"), ff * width)?,
            ff1_b: vec_dev(&format!("{p}.feedforward.0.bias"), ff)?,
            ff2_w: bf16(device, ck, &format!("{p}.feedforward.3.weight"), width * ff)?,
            ff2_b: vec_dev(&format!("{p}.feedforward.3.bias"), width)?,
        });
    }

    let mut decoder = Vec::with_capacity(config.layers);
    for i in 0..config.layers {
        let p = format!("layers.{i}");
        let (self_q_w, self_k_w, self_v_w, self_q_b, self_k_b, self_v_b) =
            split_in(&format!("{p}.self_attn"))?;
        let (cross_q_w, cross_k_w, cross_v_w, cross_q_b, cross_k_b, cross_v_b) =
            split_in(&format!("{p}.multihead_attn"))?;
        decoder.push(DeviceDecoder {
            norm1_w: vec_dev(&format!("{p}.norm1.weight"), width)?,
            norm1_b: vec_dev(&format!("{p}.norm1.bias"), width)?,
            self_q_w, self_k_w, self_v_w, self_q_b, self_k_b, self_v_b,
            self_out_w: bf16(device, ck, &format!("{p}.self_attn.out_proj.weight"), width * width)?,
            self_out_b: vec_dev(&format!("{p}.self_attn.out_proj.bias"), width)?,
            norm2_w: vec_dev(&format!("{p}.norm2.weight"), width)?,
            norm2_b: vec_dev(&format!("{p}.norm2.bias"), width)?,
            cross_q_w, cross_k_w, cross_v_w, cross_q_b, cross_k_b, cross_v_b,
            cross_out_w: bf16(device, ck, &format!("{p}.multihead_attn.out_proj.weight"), width * width)?,
            cross_out_b: vec_dev(&format!("{p}.multihead_attn.out_proj.bias"), width)?,
            norm3_w: vec_dev(&format!("{p}.norm3.weight"), width)?,
            norm3_b: vec_dev(&format!("{p}.norm3.bias"), width)?,
            ff1_w: bf16(device, ck, &format!("{p}.linear1.weight"), ff * width)?,
            ff1_b: vec_dev(&format!("{p}.linear1.bias"), ff)?,
            ff2_w: bf16(device, ck, &format!("{p}.linear2.weight"), width * ff)?,
            ff2_b: vec_dev(&format!("{p}.linear2.bias"), width)?,
        });
    }
    let _ = heads;
    Ok(DeviceHead {
        index_cache: std::sync::Mutex::new(None),
        hidden_norm_w: vec_dev("hidden_norm.weight", hidden)?,
        hidden_norm_b: vec_dev("hidden_norm.bias", hidden)?,
        memory_projection: bf16(device, ck, "memory_projection.weight", width * hidden)?,
        question_projection: bf16(device, ck, "question_projection.weight", width * hidden)?,
        option_question_projection: bf16(device, ck, "option_question_projection.weight", width * hidden)?,
        global_projection: bf16(device, ck, "global_projection.weight", width * hidden)?,
        option_context_projection: bf16(device, ck, "option_context_projection.weight", width * hidden)?,
        option_lexical_projection: bf16(device, ck, "option_lexical_projection.weight", width * hidden)?,
        type_embedding: vec_dev("type_embedding.weight", 3 * width)?,
        evidence,
        option_summary_norm_w: vec_dev("option_summary_norm.weight", width)?,
        option_summary_norm_b: vec_dev("option_summary_norm.bias", width)?,
        decoder,
        field_norm_w: vec_dev("field_norm.weight", width)?,
        field_norm_b: vec_dev("field_norm.bias", width)?,
        option_norm_w: vec_dev("option_norm.weight", width)?,
        option_norm_b: vec_dev("option_norm.bias", width)?,
        residual_w0: bf16(device, ck, "residual_scorer.0.weight", width * width * 4)?,
        residual_b0: vec_dev("residual_scorer.0.bias", width)?,
        residual_w3: vec_dev("residual_scorer.3.weight", width)?,
        residual_b3: vec_dev("residual_scorer.3.bias", 1)?,
        prior_logit_scale: scalar("prior_logit_scale")?,
        joint_logit_scale: scalar("joint_logit_scale")?,
        residual_gate: scalar("residual_gate")?,
        config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::Checkpoint;
    use crate::model::layer::{deltanet_layer, DeltaNetWeights};

    #[test]
    fn the_pass_geometry_is_fitted_to_a_graphed_rung() {
        let geoms = pass_geometries();
        let granule = crate::cuda::CHUNK_GRANULE;
        let is_geom = |t: usize| geoms.binary_search(&t).is_ok();

        let fitted = fit_chunks(&[256, 2048]);
        assert_eq!(fitted.iter().sum::<usize>(), 2048,
                   "256 + 2,048 must be fitted DOWN to the graphed 2,048, not padded to 2,304 or 4,096; \
                    got {fitted:?}");
        assert_eq!(fitted, vec![256, 1792],
                   "the SMALLER chunk must be served whole and the larger must absorb the shortfall, so a \
                    short request's slot frees sooner; got {fitted:?}");

        let rung_sizes: Vec<usize> = Rungs::body().list().to_vec();
        let mut cases = 0usize;
        for &x in &rung_sizes {
            for &y in &rung_sizes {
                for n in 2..=4usize {
                    let want: Vec<usize> = if n == 2 { vec![x, y] } else { vec![x, y, x / 2 + granule, y / 2 + granule] };
                    let got = fit_chunks(&want);
                    assert_eq!(got.len(), want.len(), "one entry per chunk: {want:?}");
                    let total: usize = got.iter().sum();
                    assert!(is_geom(total),
                            "{want:?} fitted to {got:?} = {total}, which is NOT a graphed geometry — the \
                             pass would run eager");
                    for (i, (&c, &w)) in got.iter().zip(want.iter()).enumerate() {
                        let bound = w.max(granule);
                        assert!(c <= bound, "{want:?}: chunk {i} was given {c} rows but asked for {w}");
                        assert_eq!(c % granule, 0, "{want:?}: chunk {i} = {c} is not a granule multiple");
                    }

                    assert_eq!(fit_chunks(&got), got, "{want:?}: fit is not idempotent on {got:?}");
                    cases += 1;
                }
            }
        }
        assert!(cases >= 100, "the sweep must actually sweep, ran {cases}");

        assert_eq!(fit_chunks(&[20]), vec![granule], "a sub-granule solo want must ride one granule");

        assert_eq!(fit_chunks(&[192]), vec![pass_rung(192)],
                   "a solo chunk must be padded UP to the next geometry, not shrunk to the one below");
        for &r in &rung_sizes {
            assert_eq!(fit_chunks(&[r]), vec![r], "a solo rung must be left alone: {r}");
        }
    }

    #[test]
    fn pass_maps_must_describe_the_whole_request() {
        let capacity = 4096usize;
        let ids: Vec<u32> = (0..4096).map(|i| (100 + (i % 900)) as u32).collect();

        let cases: Vec<Vec<ChunkSpec<'_>>> = vec![
            vec![ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 4096 }],
            vec![ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 2048 }],
            vec![ChunkSpec { ids: &ids, pos: 2048, slot: 0, rows: 2048 }],
            vec![ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 1024 }],
            vec![ChunkSpec { ids: &ids, pos: 1024, slot: 0, rows: 1024 }],
            vec![ChunkSpec { ids: &ids, pos: 3072, slot: 0, rows: 1024 }],

            vec![ChunkSpec { ids: &ids[..700], pos: 384, slot: 0, rows: 384 }],

            vec![ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 1024 },
                 ChunkSpec { ids: &ids, pos: 1024, slot: 1, rows: 1024 }],
        ];
        for chunks in &cases {
            let want: Vec<usize> = chunks.iter().map(|x| x.rows).collect();
            let fitted = fit_chunks(&want);
            let chunks: Vec<ChunkSpec<'_>> = chunks.iter().enumerate()
                .map(|(i, x)| ChunkSpec { ids: x.ids, pos: x.pos, slot: x.slot, rows: fitted[i] })
                .collect();
            let m = pass_maps(&chunks, capacity);
            assert_eq!(m.total, m.rows_pad.iter().sum::<usize>(), "total is the padded row count");
            assert_eq!(m.cpos.len(), m.total, "one cpos per row");
            assert_eq!(m.cbase.len(), m.total, "one cbase per row");
            assert_eq!(m.seq_end.len(), m.total, "one seq_end per row");
            assert_eq!(m.ids.len(), m.total, "one id per row, pad included");
            assert_eq!(m.chunk_start.len(), m.total / crate::cuda::CHUNK_GRANULE, "one per 64-row chunk");
            for (i, x) in chunks.iter().enumerate() {
                let (off, real) = (m.offs[i], m.rows_real[i]);
                let seg = (x.slot * capacity) as i32;
                for t in 0..m.rows_pad[i] {
                    let row = off + t;

                    assert_eq!(m.cbase[row], seg,
                        "chunk {i} row {t}: cbase is the segment base; `pos` must not enter it");
                    assert_eq!(m.cpos[row], seg + x.pos as i32 + t as i32,
                        "chunk {i} row {t}: the key a row writes is absolute");
                    assert_eq!(m.rope_pos[row], x.pos as i32 + t as i32,
                        "chunk {i} row {t}: RoPE rotates by the position WITHIN the request");
                    assert_eq!(m.seq_end[row], (off + real) as i32,
                        "chunk {i} row {t}: seq_end is the request's own end, which masks the pad");
                }

                let last_real = m.cpos[off + real - 1] + 1;
                for t in 0..real {
                    let row = off + t;
                    let q = m.cpos[row];
                    assert!(m.cbase[row] <= seg + x.pos as i32,
                        "chunk {i} row {t}: the lower bound must not exclude the request's own first key");
                    assert!(q < last_real, "chunk {i} row {t}: a row is inside its own request's key range");
                }
                for (j, y) in chunks.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let other = (y.slot * capacity) as i32;
                    assert!(other >= last_real || other + y.pos as i32 + y.rows as i32 <= m.cbase[off],
                        "chunk {i} must not be able to attend chunk {j}'s keys");
                }
            }
            assert_eq!(m.cache_len,
                       chunks.iter().map(|x| (x.slot * capacity) as i32 + x.pos as i32).min().unwrap(),
                       "cache_len is the pass's FIRST row, as `KvCache::len` and RoPE's `pos_base` document");
        }
    }

    #[test]
    fn the_head_stages_the_longest_request_the_server_admits() {
        let rows = crate::gpu::head_staging_rows();
        let longest = crate::gpu::kv_capacity();
        assert!(rows >= longest,
                "the head stages {rows} rows but the cache admits {longest} tokens: every chunk above \
                 {rows} reads rows nothing wrote");
        let cut = Rungs::head().decompose(longest);
        assert!(cut.len() > 1, "a {longest}-token request is one chunk: this gate would test nothing");
        for (base, r) in &cut {
            assert!(base + r <= rows,
                    "the chunk at {base} for {r} rows reads up to {}, past the {rows}-row staging",
                    base + r);
        }
    }

    #[test]
    fn the_two_ladders_agree_on_their_primitives() {
        for (name, rungs) in [("head", Rungs::head()), ("body", Rungs::body())] {
            let list = rungs.list();
            assert!(!list.is_empty(), "{name}: a ladder is non-empty");
            assert!(list.windows(2).all(|w| w[0] < w[1]), "{name}: the ladder ascends: {list:?}");
            for &r in list {
                assert_eq!(r % crate::cuda::CHUNK_GRANULE, 0,
                           "{name}: rung {r} is not a chunk granule multiple");

                assert_eq!(rungs.at_or_above(r), r, "{name}: at_or_above({r}) is not itself");
                assert_eq!(rungs.at_or_below(r), r, "{name}: at_or_below({r}) is not itself");
                assert!(rungs.contains(r), "{name}: contains({r})");
            }
            assert_eq!(rungs.smallest(), list[0]);
            assert_eq!(rungs.largest(), *list.last().unwrap());

            assert_eq!(rungs.at_or_below(1), rungs.smallest());
            assert_eq!(rungs.at_or_above(rungs.largest() * 4), rungs.largest());
        }

        assert_eq!(Rungs::head().decompose(16_717),
                   vec![(0, 8192), (8192, 8192), (16384, 256), (16640, 64), (16704, 64)],
                   "the head's 16,717 cut is the documented one");
        for (name, rungs, tokens) in [("head", Rungs::head(), 16_717usize), ("body", Rungs::body(), 16_717)] {
            let cut = rungs.decompose(tokens);
            let rows: Vec<usize> = cut.iter().map(|(_, r)| *r).collect();
            let total: usize = rows.iter().sum();
            eprintln!("  {name} decompose({tokens}) = {rows:?} ({total} rows)");
            assert!(total >= tokens, "{name}: the cut must cover the request");
            assert!(total - tokens < rungs.smallest(), "{name}: over-covers by a rung or more");
            assert!(rows.windows(2).all(|w| w[0] >= w[1]), "{name}: largest first: {rows:?}");

            for w in cut.windows(2) { assert_eq!(w[0].0 + w[0].1, w[1].0, "{name}: bases are contiguous"); }
        }

        for top in [2048usize, 8192] {
            let rungs = Rungs::body_at(top);
            let mut remaining = 16_717usize;
            let mut rows = Vec::new();
            while remaining > 0 {
                let r = rungs.chunk_for(remaining);
                rows.push(r);
                remaining = remaining.saturating_sub(r);
            }
            eprintln!("  16,717 at max_chunk_size {top}: {} passes {rows:?}", rows.len());
            assert!(rows.iter().sum::<usize>() >= 16_717, "the cut covers the request");

            let last = *rows.last().expect("a cut has at least one chunk");
            assert!(rows.iter().sum::<usize>() - 16_717 < last,
                   "over-covers by less than the final rung: {rows:?}");

            let fewest = 16_717usize.div_ceil(*rungs.list().iter().rev()
                .find(|&&r| r <= top).expect("a rung fits the cap"));
            assert!(rows.len() <= fewest,
                   "{} passes for 16,717 at cap {top}, but {fewest} suffice: {rows:?}", rows.len());
            assert!(rows.windows(2).all(|w| w[0] >= w[1]), "largest first: {rows:?}");
            for &r in &rows { assert!(r <= top, "{r} exceeds the max chunk"); }
            if top == 8192 {
                assert_eq!(rungs.decompose(16_717).iter().map(|(_, r)| *r).collect::<Vec<_>>(),
                           vec![8192, 8192, 256, 64, 64], "the head's own decomposition");

                let slots = kv_slots().max(1);
                for k in 2..=slots {
                    for r in HEAD_RUNGS {
                        if k * r <= top {
                            assert!(rungs.contains(k * r),
                                    "{k} x {r} = {} is not a rung, and the closure is what makes a \
                                     {slots}-way shared pass land on the ladder", k * r);
                        }
                    }
                }
                let want_6144: Vec<usize> = if slots >= 3 { vec![6144] } else { vec![4096, 2048] };
                assert_eq!(rungs.decompose(6_144).iter().map(|(_, r)| *r).collect::<Vec<_>>(),
                           want_6144,
                           "6,144 at {slots} slots: {} x 2,048 decides whether it is one pass", slots);

                assert_eq!(rungs.decompose(16_717).iter().map(|(_, r)| *r).collect::<Vec<_>>(),
                           vec![8192, 8192, 256, 64, 64], "16,717 is not a rung: the ladder cuts it");

                assert_eq!(rungs.chunk_for(16_717), 8192, "the shipping cut: one chunk of up to the max");
            }
        }
    }

    #[test]
    fn a_pass_is_fitted_to_a_graphed_total() {
        let capacity = 8192usize;
        let ids: Vec<u32> = (0..8192).map(|i| (100 + (i % 900)) as u32).collect();
        let rungs = pass_rungs();
        let geoms = pass_geometries();
        let granule = crate::cuda::CHUNK_GRANULE;
        eprintln!("  ladder: {rungs:?}");
        assert_eq!(*rungs.last().unwrap(), max_chunk_size(), "the top rung is the ceiling");
        assert!(rungs.windows(2).all(|w| w[0] < w[1]), "the ladder ascends: {rungs:?}");
        assert!(rungs.iter().all(|r| geoms.binary_search(r).is_ok()),
                "every rung must be a pass geometry, or a solo pass would run eager");

        let cases: [&[usize]; 7] = [&[2048, 2048, 2048], &[2048, 200, 64], &[300, 300, 300],
                                    &[1000, 2048, 7], &[2048, 2048], &[4096], &[6144]];
        for rows in cases {
            let fitted = fit_chunks(rows);
            let chunks: Vec<ChunkSpec<'_>> = fitted.iter().enumerate()
                .map(|(i, &r)| ChunkSpec { ids: &ids, pos: 0, slot: i, rows: r }).collect();
            let m = pass_maps(&chunks, capacity);

            assert!(geoms.binary_search(&m.total).is_ok(),
                    "{rows:?}: total {} is not a graphed geometry — the pass would run EAGER",
                    m.total);
            let mut sum = 0usize;
            let mut waste = 0usize;
            for (i, &r) in rows.iter().enumerate() {
                let pad = m.rows_pad[i];

                assert!(pad > 0, "{rows:?}: chunk {i} was dropped; the water-fill must give every chunk rows");
                assert_eq!(pad % granule, 0, "{rows:?}: chunk {i} = {pad} is not granule-aligned");
                sum += pad;
                waste += pad.saturating_sub(r);
            }
            assert_eq!(m.total, sum, "{rows:?}: total is the sum of the chunks' fitted rows");
            assert_eq!(m.ids.len(), m.total, "{rows:?}: one id per row");
            if rows.len() == 1 {
                assert_eq!(m.total, *geoms.iter().find(|&&g| g >= rows[0]).unwrap(),
                           "{rows:?}: a solo chunk must take the smallest geometry at or above it");
            } else {
                for (i, &r) in rows.iter().enumerate() {
                    assert!(m.rows_pad[i] <= r.max(granule),
                            "{rows:?}: chunk {i} was given {} rows but asked for {r} — a shared pass must \
                             SHRINK to its geometry, never pad", m.rows_pad[i]);
                }

                assert!(waste <= rows.len() * granule,
                        "{rows:?}: {waste} padding rows is more than one granule per chunk allows");
            }
            eprintln!("  {rows:?} -> fitted {:?}, total {} ({} wanted, {}% padding)",
                      m.rows_pad, m.total, rows.iter().sum::<usize>(),
                      100.0 * waste as f64 / m.total as f64);
        }

        let fitted = fit_chunks(&[2048, 200, 64]);
        let chunks: Vec<ChunkSpec<'_>> = fitted.iter().enumerate()
            .map(|(i, &r)| ChunkSpec { ids: &ids, pos: 0, slot: i, rows: r }).collect();
        let m = pass_maps(&chunks, capacity);
        assert_eq!(m.total, 2048, "[2048,200,64] must fit DOWN to 2,048, not pad up to 2,368 or 6,144");
        assert_eq!(m.rows_pad, vec![1792, 192, 64],
                   "the two tails are served whole and the large chunk absorbs the shortfall");
    }

    #[test]
    #[should_panic(expected = "would run EAGER")]
    fn a_pass_whose_total_is_not_graphed_is_refused() {
        let ids: Vec<u32> = (0..8192).map(|i| (100 + (i % 900)) as u32).collect();

        let chunks: Vec<ChunkSpec<'_>> = (0..3)
            .map(|i| ChunkSpec { ids: &ids, pos: 0, slot: i, rows: 2048 }).collect();
        let _ = pass_maps(&chunks, 8192);
    }

    #[test]
    #[should_panic(expected = "share slot")]
    fn two_chunks_in_one_slot_are_refused() {
        let capacity = 4096usize;
        let ids: Vec<u32> = (0..2048).map(|i| (100 + (i % 900)) as u32).collect();
        let _ = pass_maps(&[ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 1024 },
                            ChunkSpec { ids: &ids, pos: 1024, slot: 0, rows: 1024 }], capacity);
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_six_thousand_row_pass_matches_three_chunk_passes() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let hidden = config.hidden_size;

        let tokens = std::env::var("CLEF_PASS_PROBE").ok()
            .and_then(|v| v.parse::<usize>().ok()).unwrap_or(6144);
        let capacity = crate::gpu::kv_capacity();
        assert!(tokens <= capacity, "{tokens} rows needs a {tokens}-token cache, this one holds {capacity}");
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        let chunk = (tokens / 3).div_ceil(crate::cuda::CHUNK_GRANULE) * crate::cuda::CHUNK_GRANULE;
        let three = runner.forward_chunked_dev(&ids, chunk, capacity).expect("reference chunks");
        let three = runner.device.download_rows_f32(&three, tokens, hidden).expect("dl");
        eprintln!("  reference done: {} x {chunk}", tokens.div_ceil(chunk));

        let one = runner.forward_chunked_dev(&ids, tokens, capacity).expect("1 x tokens");
        let one = runner.device.download_rows_f32(&one, tokens, hidden).expect("dl");
        let worst = three.iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let first = three.iter().zip(&one).position(|(a, b)| a != b);
        eprintln!("  {tokens} rows in one pass against the chunked reference: worst {worst:.3e}, first {first:?}");
        assert_eq!(worst, 0.0, "a pass size must not change the answer");
    }

    #[test]
    fn a_shared_pass_does_not_move_the_bounds_of_a_request() {
        let capacity = 4096usize;
        let ids: Vec<u32> = (0..2048).map(|i| (100 + (i % 900)) as u32).collect();
        let solo = pass_maps(&[ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 1024 }], capacity);
        let shared = pass_maps(&[ChunkSpec { ids: &ids, pos: 0, slot: 0, rows: 1024 },
                                 ChunkSpec { ids: &ids, pos: 1024, slot: 1, rows: 1024 }], capacity);
        assert_eq!(solo.total, 1024);
        assert_eq!(shared.total, 2048, "the shared pass carries both requests' rows");
        for t in 0..solo.total {
            assert_eq!(solo.cpos[t], shared.cpos[t], "row {t}: cpos moved when a neighbour joined");
            assert_eq!(solo.cbase[t], shared.cbase[t], "row {t}: cbase moved when a neighbour joined");
            assert_eq!(solo.slot_map[t], shared.slot_map[t], "row {t}: slot moved");
            assert_eq!(solo.rope_pos[t], shared.rope_pos[t], "row {t}: RoPE position moved");

            assert_eq!(solo.seq_end[t] - solo.offs[0] as i32, shared.seq_end[t] - shared.offs[0] as i32,
                       "row {t}: the request's own end moved when a neighbour joined");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn one_captured_layer_matches_the_direct_one() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let tokens = 64usize;
        let hidden = config.hidden_size;
        let DeviceLayer::Linear(w) = &runner.backbone.layers[0] else {
            panic!("layer 0 is a DeltaNet layer");
        };

        let ids: Vec<u32> = (0..tokens).map(|i| (100 + i) as u32).collect();
        let h = runner.embed_dev(&ids).expect("embed");
        let direct = runner.delta_net_layer_dev(&h, tokens, w, crate::cuda::BatchMap::SINGLE).expect("direct");
        let direct = runner.device.download_rows_f32(&direct, tokens, hidden).expect("dl");

        let input = runner.device.copy_dev(&h, tokens * hidden).expect("copy");
        for _ in 0..3 {
            let warm = runner.delta_net_layer_dev(&input, tokens, w, crate::cuda::BatchMap::SINGLE).expect("warm");
            runner.device.sync().expect("sync");
            drop(warm);
        }
        let output = runner.device.alloc(tokens * hidden * 4).expect("out");
        let captured = runner
            .device
            .capture(|| {
                let y = runner.delta_net_layer_dev(&input, tokens, w, crate::cuda::BatchMap::SINGLE)?;
                runner.device.copy_into(&output, &y, tokens * hidden)
            })
            .expect("capture");
        captured.launch().expect("launch 1");
        captured.launch().expect("launch 2");
        runner.device.sync().expect("sync");
        let got = runner.device.download_rows_f32(&output, tokens, hidden).expect("dl");

        let mut worst = 0.0f32;
        for (a, b) in direct.iter().zip(got.iter()) {
            worst = worst.max((a - b).abs());
        }
        eprintln!("  one layer: worst |captured - direct| = {worst:.3e}");
        assert_eq!(worst, 0.0, "a captured DeltaNet layer differs from the direct run");

        let n = 4usize;
        let run_n = |h: &Buffer| -> Result<Buffer> {
            let mut cur = runner.device.copy_dev(h, tokens * hidden)?;
            for (i, layer) in runner.backbone.layers.iter().take(n).enumerate() {
                cur = match layer {
                    DeviceLayer::Linear(w) => runner.delta_net_layer_dev(&cur, tokens, w, crate::cuda::BatchMap::SINGLE)?,
                    DeviceLayer::Full(w) => runner.attention_layer_dev(&cur, tokens, w, crate::cuda::BatchMap::SINGLE)?,
                };
                let _ = i;
            }
            Ok(cur)
        };
        let direct_n = run_n(&h).expect("direct n");
        let direct_n = runner.device.download_rows_f32(&direct_n, tokens, hidden).expect("dl");

        let input_n = runner.device.copy_dev(&h, tokens * hidden).expect("copy");
        for _ in 0..3 {
            let warm = run_n(&input_n).expect("warm");
            runner.device.sync().expect("sync");
            drop(warm);
        }
        let output_n = runner.device.alloc(tokens * hidden * 4).expect("out");
        let captured_n = runner
            .device
            .capture(|| {
                let y = run_n(&input_n)?;
                runner.device.copy_into(&output_n, &y, tokens * hidden)
            })
            .expect("capture n");
        captured_n.launch().expect("launch 1");
        captured_n.launch().expect("launch 2");
        runner.device.sync().expect("sync");
        let got_n = runner.device.download_rows_f32(&output_n, tokens, hidden).expect("dl");
        let mut worst_n = 0.0f32;
        for (a, b) in direct_n.iter().zip(got_n.iter()) {
            worst_n = worst_n.max((a - b).abs());
        }
        eprintln!("  first {n} layers: worst |captured - direct| = {worst_n:.3e}");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_graph_memory_stays_inside_its_budget() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let budget = graph_memory_budget(&runner.device);
        eprintln!("  budget = {} MiB", budget / (1024 * 1024));

        let sizes: Vec<usize> = (1..=64).map(|i| i * 16).collect();
        let free_before = runner.device.free_memory().unwrap_or(0);
        let captured = runner.warm_up(&sizes).expect("warm-up").captured;

        let held = runner.device.graph_block_bytes();
        let free_after = runner.device.free_memory().unwrap_or(0);

        eprintln!(
            "  warmed {} of {} sizes; {} MiB held of {} MiB budget",
            captured,
            sizes.len(),
            held / (1024 * 1024),
            budget / (1024 * 1024)
        );
        eprintln!(
            "  device free: {} MiB -> {} MiB (spent {} MiB)",
            free_before / (1024 * 1024),
            free_after / (1024 * 1024),
            free_before.saturating_sub(free_after) / (1024 * 1024)
        );

        assert!(
            held <= budget,
            "the engine holds {held} bytes of graph memory against a {budget} budget"
        );

        let geometries: std::collections::BTreeSet<usize> =
            sizes.iter().map(|&t| crate::gpu::pad_to_chunk(t)).collect();

        let chunked = if crate::gpu::chunking_enabled() {
            let mut set = std::collections::BTreeSet::new();
            for &t in sizes.iter() {
                if let Some(b) = crate::gpu::graph_size_for(t) {
                    let run = crate::gpu::pad_to_chunk(b);
                    if crate::gpu::pass_rung(run) == run {
                        set.insert(run);
                    }
                }
            }
            set.len()
        } else {
            0
        };

        let want = chunked;
        assert_eq!(
            captured,
            want,
            "only {captured} of {want} chunked graphs were captured ({} sizes, {chunked} chunked rungs, no \
             solo mode); with one retained block every geometry the device allows must fit",
            sizes.len()
        );
        drop(runner);
        let runner = GpuRunner::load(&checkpoint, &config).expect("reload");

        let default_sizes = if chunking_enabled() {
            crate::engine::dense_warm_up_sizes()
        } else {
            crate::engine::warm_up_sizes()
        };
        let captured_default = runner.warm_up(&default_sizes).expect("warm-up").captured;
        let held = runner.device.graph_block_bytes();
        let budget2 = graph_memory_budget(&runner.device);
        eprintln!(
            "  the default list: {} of {} sizes captured, {} MiB held of a {} MiB budget",
            captured_default,
            default_sizes.len(),
            held / (1024 * 1024),
            budget2 / (1024 * 1024)
        );

        let ceiling = default_sizes
            .iter()
            .copied()
            .filter(|&t| runner.graph_ready_for(t, GraphMode::Chunked))
            .max()
            .expect("no size in the default list was captured, so the ceiling is below 256 tokens");
        let below: Vec<usize> = default_sizes.iter().copied().filter(|&t| t <= ceiling).collect();
        let above: Vec<usize> = default_sizes.iter().copied().filter(|&t| t > ceiling).collect();
        eprintln!(
            "  the default list: ceiling {ceiling}, {} of {} captured ({} above it skipped), {} MiB held \
             of a {} MiB budget",
            below.len(),
            default_sizes.len(),
            above.len(),
            held / (1024 * 1024),
            budget2 / (1024 * 1024)
        );
        for &t in &below {
            assert!(
                runner.graph_ready_for(t, GraphMode::Chunked),
                "{t} is at or below the ceiling ({ceiling}) and has no graph, so the budget stopped the \
                 list partway through sizes the device can run"
            );
        }
        for &t in &above {
            assert!(
                !runner.graph_ready_for(t, GraphMode::Chunked),
                "{t} is above the ceiling ({ceiling}) and was captured anyway — the cap did not hold"
            );
        }
        assert!(held <= budget2, "held {held} against a {budget2} budget");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn each_layer_is_deterministic_against_itself() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let hidden = config.hidden_size;
        let tokens = 8usize;
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + i) as u32).collect();
        let h = runner.embed_dev(&ids).expect("embed");
        for (i, layer) in runner.backbone.layers.iter().enumerate() {
            let mut run = || -> Result<Vec<f32>> {
                let out = match layer {
                    DeviceLayer::Linear(w) => runner.delta_net_layer_dev(&h, tokens, w, crate::cuda::BatchMap::SINGLE)?,
                    DeviceLayer::Full(w) => runner.attention_layer_dev(&h, tokens, w, crate::cuda::BatchMap::SINGLE)?,
                };
                runner.device.download_rows_f32(&out, tokens, hidden)
            };
            let a = run().expect("layer a");
            let b = run().expect("layer b");
            let worst = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let kind = match layer {
                DeviceLayer::Linear(_) => "deltanet",
                DeviceLayer::Full(_) => "attention",
            };
            eprintln!("  layer {i:2} ({kind:9}): worst |a-b| = {worst:.3e}");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_eager_backbone_is_deterministic_against_itself() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let hidden = config.hidden_size;

        {
            let ids: Vec<u32> = (0..8u32).map(|i| 100 + i).collect();
            let e1 = runner.embed_dev(&ids).expect("embed");
            let e1 = runner.device.download_f32(&e1).expect("dl");
            let e2 = runner.embed_dev(&ids).expect("embed");
            let e2 = runner.device.download_f32(&e2).expect("dl");
            let w = e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("  embed_dev twice: worst |a-b| = {w:.3e}   (len {})", e1.len());
        }

        let sizes: Vec<usize> = std::env::var("CLEF_BISECT_SIZES")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![8]);
        for &tokens in &sizes {
            let ids: Vec<u32> = (0..tokens).map(|i| (100 + i) as u32).collect();
            let h = runner.embed_dev(&ids).expect("embed");
            let a = runner.layers_dev(&h, tokens).expect("a");
            let a = runner.device.download_rows_f32(&a, tokens, hidden).expect("dl");
            let b = runner.layers_dev(&h, tokens).expect("b");
            let b = runner.device.download_rows_f32(&b, tokens, hidden).expect("dl");
            let mut worst = 0.0f32;
            let mut first = usize::MAX;
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let d = (x - y).abs();
                if d > worst {
                    worst = d;
                }
                if d > 1e-3 && first == usize::MAX {
                    first = i;
                }
            }
            eprintln!(
                "  tokens {tokens:3}: worst |a-b| = {worst:.3e}   first differing element {} (row {})",
                if first == usize::MAX { "none".into() } else { first.to_string() },
                if first == usize::MAX { "-".into() } else { (first / hidden).to_string() }
            );

            let mut unstable = 0usize;
            for _ in 0..10 {
                let p = runner.layers_dev(&h, tokens).expect("p");
                let p = runner.device.download_rows_f32(&p, tokens, hidden).expect("dl");
                let q = runner.layers_dev(&h, tokens).expect("q");
                let q = runner.device.download_rows_f32(&q, tokens, hidden).expect("dl");
                if p.iter().zip(q.iter()).any(|(x, y)| (x - y).abs() > 1e-3) {
                    unstable += 1;
                }
            }
            eprintln!("              {unstable} of 10 further pairs disagreed");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_eager_answer_does_not_depend_on_the_allocation_layout() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let hidden = config.hidden_size;
        let (from, to) = (200usize, 300usize);

        let run_pair = |runner: &GpuRunner| -> (Vec<f32>, Vec<f32>) {
            let ids: Vec<u32> = (0..to).map(|i| (i % 97) as u32 + 3).collect();
            let h = runner.embed_dev(&ids).expect("embed");
            let ends: Vec<i32> = (0..to).map(|_| from as i32).collect();
            let ends_d = runner.device.upload_i32(&ends).expect("ends");
            let short = runner.layers_dev_batched(&h, from, crate::cuda::BatchMap {
                state_slot: None,
                hist_save: None,
                hist_save_dev: None,
                hist_save_n: 0,
                carry: None,
                seq_start: None, chunk_start: None, seq_end: Some(&ends_d), real_end: 0,
                real_end_dev: None,
            }).expect("short");
            let long = runner.layers_dev_batched(&h, to, crate::cuda::BatchMap {
                state_slot: None,
                hist_save: None,
                hist_save_dev: None,
                hist_save_n: 0,
                carry: None,
                seq_start: None, chunk_start: None, seq_end: Some(&ends_d), real_end: 0,
                real_end_dev: None,
            }).expect("long");
            (
                runner.device.download_rows_f32(&short, from, hidden).expect("dl s"),
                runner.device.download_rows_f32(&long, to, hidden).expect("dl l"),
            )
        };
        let (a1, b1) = run_pair(&runner);

        let mut keep: Vec<crate::cuda::Buffer> = Vec::new();
        for i in 0..48 {
            keep.push(runner.device.alloc(64 * 1024 + i * 4096).expect("shift"));
        }
        let (a2, b2) = run_pair(&runner);

        let worst = |x: &[f32], y: &[f32]| -> f32 {
            x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max)
        };
        let (da, db) = (worst(&a1, &a2), worst(&b1, &b2));
        eprintln!("  shifted layout: the 200-token answer differs by {da:.3e}, the 300-token by {db:.3e}");
        assert_eq!(da, 0.0, "the 200-token answer moved when the allocator moved");
        assert_eq!(db, 0.0, "the 300-token answer moved when the allocator moved");
        let _ = keep;
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_long_request_answers_the_same_twice() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let hidden = config.hidden_size;
        assert!(crate::gpu::RESID_FUSED_FROM > 0, "the fusion threshold is a real size");
        for tokens in [2048usize, 4096, 8192] {
            let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();
            let a = runner.forward_dev(&ids).expect("pass");
            let a = runner.device.download_rows_f32(&a, tokens, hidden).expect("dl");
            let b = runner.forward_dev(&ids).expect("pass");
            let b = runner.device.download_rows_f32(&b, tokens, hidden).expect("dl");
            let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let f = a.iter().zip(&b).position(|(x, y)| x != y);
            let sum = a.iter().fold(0u64, |acc, x| acc.wrapping_mul(1_000_003)
                .wrapping_add(x.to_bits() as u64));
            let diff_rows: Vec<usize> = a
                .iter()
                .zip(&b)
                .enumerate()
                .filter(|(_, (x, y))| x != y)
                .map(|(i, _)| i / hidden)
                .collect();
            if !diff_rows.is_empty() {
                let (lo, hi) = (diff_rows[0], *diff_rows.last().unwrap());
                let uniq = {
                    let mut v = diff_rows.clone();
                    v.dedup();
                    v.len()
                };
                eprintln!(
                    "    {} elements differ over {} rows: rows {}..{} (first/last), chunks {}..{}",
                    diff_rows.len(), uniq, lo, hi,
                    lo / 2048, hi / 2048
                );
            }
            eprintln!(
                "  forward_dev {tokens} against itself: worst = {d:.3e}, first {f:?} sum={sum:016x}"
            );
            assert_eq!(d, 0.0, "{tokens} tokens answered differently on the second run");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn several_requests_share_one_pass() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let head_dir = std::path::Path::new("/models/clef-flash-nvfp4");
        let head_ck = Checkpoint::open_single(&head_dir.join("joint_head.safetensors")).expect("head ck");
        let head_cfg = crate::model::head::HeadConfig::from_dir(head_dir).expect("head cfg");
        let head = crate::gpu::load_head(d, &head_ck, crate::engine::head_config_ref(&head_cfg)).expect("head");
        let lm_head = HostTable::map(&checkpoint.shard_path("lm_head.weight").expect("path"), "lm_head.weight")
            .expect("lm_head");
        let (tokens, chunk, cap) = (4096usize, 2048usize, 4096usize);
        runner.ensure_carry_slots(cap, 3).expect("three slots");
        let q = vec![QuestionSpec { question_type: 1, question_span: (100, 140),
                                    option_spans: vec![(2000, 2050), (3000, 3050)] }];
        let ids: Vec<Vec<u32>> = [100u32, 500, 900].iter()
            .map(|&o| (0..tokens).map(|i| o + (i % 700) as u32).collect()).collect();
        let run_solo = |k: usize| -> Vec<Vec<f32>> {
            runner.forward_chunk_begin_slot(cap, k).expect("begin");
            let pre = PreHead::new(d, &head, &q, tokens, chunk).expect("pre");

            runner.forward_chunked_into(&ids[k], chunk, cap, &pre).expect("alone");
            head_forward_pre_eager(d, &pre, tokens, &ids[k], &q, &lm_head, &head).expect("head")
        };
        let solo: Vec<Vec<Vec<f32>>> = (0..3).map(run_solo).collect();

        let solo_mem: Option<Vec<f32>> = if std::env::var("CLEF_BATCH_MEM").is_ok() {
            runner.forward_chunk_begin_slot(cap, 0).expect("begin");
            let pre = PreHead::new(d, &head, &q, tokens, chunk).expect("pre");
            runner.forward_chunked_into(&ids[0], chunk, cap, &pre).expect("alone");
            d.download_f32(&pre.memory).ok()
        } else {
            None
        };

        let worst = |x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>| x.iter().zip(y)
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        for n in [2usize, 3] {
            std::env::set_var("CLEF_KV_SLOTS", n.to_string());
            for k in 0..n {
                runner.forward_chunk_begin_slot(cap, k).expect("begin");
            }
            let pres: Vec<PreHead> = (0..n)
                .map(|_| PreHead::new(d, &head, &q, tokens, chunk).expect("pre")).collect();

            let mut pos = [0usize; 3];
            while pos[..n].iter().any(|&p| p < tokens) {
                let active: Vec<usize> = (0..n).filter(|&k| pos[k] < tokens).collect();
                let want: Vec<usize> = active.iter()
                    .map(|&k| crate::gpu::next_chunk_rows(tokens - pos[k]).min(chunk))
                    .collect();
                let fitted = crate::gpu::fit_chunks(&want);
                let specs: Vec<ChunkSpec<'_>> = active.iter().zip(fitted.iter()).map(|(&k, &r)| ChunkSpec {
                    ids: &ids[k], pos: pos[k], slot: k, rows: r }).collect();
                let sinks: Vec<&PreHead> = active.iter().map(|&k| &pres[k]).collect();
                runner.forward_multi_chunk_step(&specs, &sinks).expect("multi step");
                for (&k, &r) in active.iter().zip(fitted.iter()) {
                    pos[k] = (pos[k] + r).min(tokens);
                }
            }
            for k in 0..n {
                let got = head_forward_pre_eager(d, &pres[k], tokens, &ids[k], &q, &lm_head, &head).expect("head");
                if let Some(sm) = &solo_mem {
                    if k == 0 {
                        let bm = d.download_f32(&pres[k].memory).expect("mem");
                        let mw = sm.iter().zip(&bm).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                        let first = sm.iter().zip(&bm).position(|(p, q)| p != q);
                        eprintln!("    SINK memory worst = {mw:.3e}  first differing element = {first:?}");
                    }
                }
                let w = worst(&solo[k], &got);
                eprintln!("  {n} in one pass: request {k} worst = {w:.3e}");

                if std::env::var("CLEF_BATCH_DIFF").is_ok() {
                    for (qi, (a, b)) in solo[k].iter().zip(&got).enumerate() {
                        let wq = a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                        if wq > 0.0 {
                            eprintln!("    question {qi}: worst {wq:.3e}  solo[0]={:.6} batched[0]={:.6}",
                                      a.first().copied().unwrap_or(0.0), b.first().copied().unwrap_or(0.0));
                        }
                    }
                }
                assert_eq!(w, 0.0, "{n} in one pass changed request {k}");
            }
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn our_head_on_the_references_hidden() {
        let mut last_mean_kl: Option<f32> = None;
        let dir = std::env::var("CLEF_DUMP_DIR").unwrap_or_else(|_| "/tmp/dump".into());
        let p = std::path::Path::new(&dir);

        if !p.join("hidden_meta.json").exists() {
            eprintln!("  no reference dump at {} — skipping (run layer_probe.py to make one)", p.display());
            return;
        }
        let read = |n: &str| std::fs::read_to_string(p.join(n)).expect("dump file");
        let meta: serde_json::Value = serde_json::from_str(&read("hidden_meta.json")).unwrap();
        let (tokens, hidden) = (meta["tokens"].as_u64().unwrap() as usize,
                                meta["hidden"].as_u64().unwrap() as usize);
        let ids: Vec<u32> = serde_json::from_str::<Vec<u32>>(&read("input_ids.json")).unwrap();
        let qj: serde_json::Value = serde_json::from_str(&read("questions.json")).unwrap();
        let ref_logits: serde_json::Value = serde_json::from_str(&read("logits.json")).unwrap();
        let questions: Vec<QuestionSpec> = qj.as_array().unwrap().iter().map(|q| QuestionSpec {
            question_type: q["type"].as_u64().unwrap() as usize,
            question_span: (q["span"][0].as_u64().unwrap() as usize, q["span"][1].as_u64().unwrap() as usize),
            option_spans: q["options"].as_array().unwrap().iter()
                .map(|o| (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize)).collect(),
        }).collect();
        let raw = std::fs::read(p.join("hidden.f32.bin")).expect("hidden.f32.bin");
        assert_eq!(raw.len(), tokens * hidden * 4, "hidden.f32.bin is not {tokens}x{hidden} f32");
        let mut host = vec![0f32; tokens * hidden];
        for (i, c) in raw.chunks_exact(4).enumerate() { host[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]); }

        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let d = Device::open().expect("device");
        let head_dir = std::path::Path::new("/models/clef-flash-nvfp4");
        let head_ck = Checkpoint::open_single(&head_dir.join("joint_head.safetensors")).expect("head ck");
        let head_cfg = crate::model::head::HeadConfig::from_dir(head_dir).expect("head cfg");
        let head = crate::gpu::load_head(&d, &head_ck, crate::engine::head_config_ref(&head_cfg)).expect("head");
        let lm_head = HostTable::map(&checkpoint.shard_path("lm_head.weight").expect("path"), "lm_head.weight")
            .expect("lm_head");

        let buf = d.upload_f32(&host).expect("upload");
        let pre = PreHead::new(&d, &head, &questions, tokens, tokens).expect("pre");
        pre.absorb(&d, &buf, tokens, 0).expect("absorb");
        let logits = head_forward_pre_eager(&d, &pre, tokens, &ids, &questions, &lm_head, &head).expect("head");

        let mut worst = 0f32;
        let mut worst_q = String::new();
        for (i, q) in qj.as_array().unwrap().iter().enumerate() {
            let qid = q["id"].as_str().unwrap();
            let Some(want) = ref_logits.get(qid) else { continue };
            let want: Vec<f32> = want.as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap() as f32).collect();
            let got = &logits[i];
            assert_eq!(got.len(), want.len(), "{qid}: {} logits vs {}", got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                let diff = (g - w).abs();
                if diff > worst { worst = diff; worst_q = qid.to_string(); }
            }
            if i < 3 {
                eprintln!("  {qid}: ours {:?}", &got[..got.len().min(4)]);
                eprintln!("  {qid}: ref  {:?}", &want[..want.len().min(4)]);
            }
        }
        eprintln!("  OUR HEAD ON THE REFERENCE'S HIDDEN: worst |logit diff| = {worst:.4e} ({worst_q})");

        if let Ok(raw) = std::fs::read_to_string("bench/out/oracle-1item.json") {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
                let want = &v["buckets"]["765"]["answers"];
                let mut kls: Vec<f64> = Vec::new();
                for (i, q) in qj.as_array().unwrap().iter().enumerate() {
                    let qid = q["id"].as_str().unwrap();
                    let Some(w) = want.get(qid).and_then(|x| x.as_object()) else { continue };

                    let mut keys: Vec<&String> = w.keys().collect();
                    if w.contains_key("true") && w.contains_key("false") {
                        keys = vec![w.get_key_value("true").unwrap().0,
                                    w.get_key_value("false").unwrap().0];
                    } else {
                        keys.sort_by_key(|k| k.parse::<i64>().unwrap_or(0));
                    }
                    let wantv: Vec<f64> = keys.iter().map(|k| w[*k].as_f64().unwrap()).collect();
                    let lg = &logits[i];
                    if lg.len() != wantv.len() { continue }
                    let m = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f64> = lg.iter().map(|x| ((*x - m) as f64).exp()).collect();
                    let z: f64 = e.iter().sum();
                    let gotv: Vec<f64> = e.iter().map(|x| x / z).collect();
                    let kl: f64 = wantv.iter().zip(gotv.iter())
                        .map(|(p, q)| p * ((p + 1e-12) / (q + 1e-12)).ln()).sum();
                    kls.push(kl);
                }
                if !kls.is_empty() {
                    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
                    last_mean_kl = Some(mean as f32);
                    eprintln!("  OUR HEAD ALONE vs the oracle: mean KL = {mean:.3e} over {} questions",
                              kls.len());
                }
            }
        }

        let kl_ok = match last_mean_kl {
            Some(k) => k < 5e-3,
            None => true,
        };
        assert!(worst < 1e-1, "our head disagrees with the reference on the reference's own hidden");
        assert!(kl_ok, "our head's mean KL against the reference exceeds 5e-3");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn one_layer_on_the_references_input() {
        let dir = std::env::var("CLEF_REF_DUMP").unwrap_or_else(|_| "/tmp/refdump".into());
        let layer: usize = std::env::var("CLEF_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let hidden_size = 4096usize;

        if !std::path::Path::new(&dir).join(format!("layer_{layer:02}.f32.bin")).exists() {
            eprintln!("SKIP: no reference dump at {dir} — run deploy/bench/layer_probe.py first");
            return;
        }
        let read_rows = |name: &str, rows: usize| -> Vec<f32> {
            let raw = std::fs::read(std::path::Path::new(&dir).join(name)).expect("dump file");
            let mut v: Vec<f32> = raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            v.truncate(rows * hidden_size);
            v
        };

        let raw_len = std::fs::metadata(std::path::Path::new(&dir).join(format!("layer_{layer:02}.f32.bin")))
            .expect("layer file").len() as usize;
        let tokens = raw_len / 4 / hidden_size;
        let input = read_rows(&format!("layer_{:02}.f32.bin", layer - 1), tokens);
        let want = read_rows(&format!("layer_{layer:02}.f32.bin"), tokens);

        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let x = d.upload_f32(&input).expect("upload");

        let out = match &runner.backbone.layers[layer] {
            DeviceLayer::Full(w) => runner.attention_layer_dev(&x, tokens, w, crate::cuda::BatchMap::SINGLE)
                .expect("attention layer"),
            DeviceLayer::Linear(w) => runner.delta_net_layer_dev(&x, tokens, w, crate::cuda::BatchMap::SINGLE)
                .expect("gdn layer"),
        };
        let got = d.download_f32(&out).expect("download");

        let mut all_cos = Vec::new();
        for t in 0..tokens {
            let a = &want[t * hidden_size..(t + 1) * hidden_size];
            let b = &got[t * hidden_size..(t + 1) * hidden_size];
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (p, q) in a.iter().zip(b.iter()) {
                dot += (*p as f64) * (*q as f64);
                na += (*p as f64) * (*p as f64);
                nb += (*q as f64) * (*q as f64);
            }
            all_cos.push(dot / (na.sqrt() * nb.max(1e-12).sqrt()));
        }
        eprintln!("  layer {layer}: mean cosine {:.6}  worst {:.6}  ({} tokens)",
                  all_cos.iter().sum::<f64>() / all_cos.len() as f64,
                  all_cos.iter().cloned().fold(1.0f64, f64::min), all_cos.len());
        let mut report = Vec::new();
        for t in [0usize, 1, 100, 300, tokens - 1] {
            if t >= tokens { continue; }
            let a = &want[t * hidden_size..(t + 1) * hidden_size];
            let b = &got[t * hidden_size..(t + 1) * hidden_size];
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (p, q) in a.iter().zip(b.iter()) {
                dot += (*p as f64) * (*q as f64);
                na += (*p as f64) * (*p as f64);
                nb += (*q as f64) * (*q as f64);
            }
            report.push((t, dot / (na.sqrt() * nb.max(1e-12).sqrt())));
        }
        eprintln!("  LAYER {layer} ON THE REFERENCE'S LAYER {} OUTPUT ({tokens} tokens):",
                  layer - 1);
        for (t, c) in &report {
            eprintln!("      token {t:5}: cosine {c:.6}");
        }
        let worst = report.iter().map(|(_, c)| *c).fold(1.0f64, f64::min);
        eprintln!("  worst cosine {worst:.6}");

        let bf = |v: f32| -> f32 {
            let b = v.to_bits();
            let r = (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16;
            f32::from_bits(r)
        };
        let mut report_bf = Vec::new();
        for t in [0usize, 1, 100, 300, tokens - 1] {
            if t >= tokens { continue; }
            let a = &want[t * hidden_size..(t + 1) * hidden_size];
            let b: Vec<f32> = got[t * hidden_size..(t + 1) * hidden_size].iter().map(|x| bf(*x)).collect();
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (p, q) in a.iter().zip(b.iter()) {
                dot += (*p as f64) * (*q as f64);
                na += (*p as f64) * (*p as f64);
                nb += (*q as f64) * (*q as f64);
            }
            report_bf.push((t, dot / (na.sqrt() * nb.max(1e-12).sqrt())));
        }

        eprintln!("  the CONTRIBUTION (output - input), which is what the layer actually adds:");
        for t in [1usize, 100, 300, tokens - 1] {
            if t >= tokens { continue; }
            let xi = &input[t * hidden_size..(t + 1) * hidden_size];
            let wa: Vec<f32> = want[t * hidden_size..(t + 1) * hidden_size].iter()
                .zip(xi.iter()).map(|(w, x)| w - x).collect();
            let ga: Vec<f32> = got[t * hidden_size..(t + 1) * hidden_size].iter()
                .zip(xi.iter()).map(|(g, x)| g - x).collect();
            let norm = |v: &[f32]| (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (p, q) in wa.iter().zip(ga.iter()) {
                dot += (*p as f64) * (*q as f64);
                na += (*p as f64) * (*p as f64);
                nb += (*q as f64) * (*q as f64);
            }
            eprintln!("      token {t:5}: |ref contrib| {:.3}  |ours contrib| {:.3}  cosine {:.6}",
                      na.sqrt(), nb.sqrt(), dot / (na.sqrt() * nb.max(1e-12).sqrt()));
        }
        eprintln!("  the same, with OUR output rounded to bf16:");
        for (t, c) in &report_bf {
            eprintln!("      token {t:5}: cosine {c:.6}");
        }
        assert!(worst > 0.99, "our layer {layer} does not reproduce the reference on its own input");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn rope_matches_the_reference_formula() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let c = &runner.backbone.config;
        let (rot, theta) = (c.rotary_dim(), c.rope_theta);
        let half = rot / 2;
        let head_dim = c.head_dim;
        let (heads, tokens) = (2usize, 40usize);

        let mut x = vec![0f32; tokens * heads * head_dim];
        let mut state: u32 = 12345;
        for v in x.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0;
        }
        let cos_sin = d.rope_tables_dev(
            &d.upload_i32(&(0..tokens as i32).collect::<Vec<_>>()).unwrap(), tokens, rot, theta,
        ).expect("tables");
        let buf = d.upload_f32(&x).expect("x");
        let got = d.rope_apply_dev(&buf, &cos_sin.0, &cos_sin.1, tokens, heads, head_dim, rot,
                                   None, None, None).expect("rope");
        let got = d.download_f32(&got).expect("download");

        let mut worst = 0f32;
        let mut worst_at = (0usize, 0usize, 0usize);
        for t in 0..tokens {
            for h in 0..heads {
                let base = (t * heads + h) * head_dim;
                for j in 0..head_dim {
                    let want = if j < rot {
                        let i = j % half;
                        let inv = 1.0f64 / (theta as f64).powf(2.0 * i as f64 / rot as f64);
                        let ang = t as f64 * inv;
                        let (co, si) = (ang.cos(), ang.sin());
                        if j < half {
                            x[base + j] as f64 * co - x[base + j + half] as f64 * si
                        } else {
                            x[base + j] as f64 * co + x[base + j - half] as f64 * si
                        }
                    } else {
                        x[base + j] as f64
                    };
                    let diff = (got[base + j] as f64 - want).abs() as f32;
                    if diff > worst { worst = diff; worst_at = (t, h, j); }
                }
            }
        }
        eprintln!("  RoPE vs the reference formula: worst |diff| = {worst:.3e} at token/head/dim {:?}",
                  worst_at);

        assert!(worst < 2e-5, "our RoPE does not match the reference formula");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn rms_norm_matches_the_formula() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let (rows, width, eps) = (37usize, 256usize, 1e-6f32);

        let mut state: u32 = 99991;
        let mut rnd = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let x: Vec<f32> = (0..rows * width).map(|_| rnd()).collect();

        let w: Vec<f32> = (0..width).map(|j| 0.2 + 0.6 * (j as f32 / width as f32)).collect();

        let got = d.rms_norm_dev(&d.upload_f32(&x).unwrap(), &d.upload_f32(&w).unwrap(),
                                 rows, width, eps).expect("rms_norm");
        let got = d.download_f32(&got).expect("download");

        let mut worst = 0f32;
        for r in 0..rows {
            let row = &x[r * width..(r + 1) * width];
            let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / width as f64;
            let inv = 1.0 / (ms + eps as f64).sqrt();
            for j in 0..width {
                let want = row[j] as f64 * inv * (1.0 + w[j] as f64);
                worst = worst.max((got[r * width + j] as f64 - want).abs() as f32);
            }
        }
        let w_rms = (w.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / width as f64).sqrt();
        eprintln!("  rms_norm vs the formula: worst |diff| = {worst:.3e}  (weight RMS {w_rms:.4})");
        assert!(worst < 2e-5, "rms_norm_dev does not match the formula");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_uniform_softmax_returns_the_running_mean_of_v() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let tokens: usize = std::env::var("CLEF_TEST_TOKENS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(37);
        let (heads, kv_heads, hd) = (2usize, 2usize, 256usize);

        let mut state: u32 = 4242;
        let mut rnd = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let q = vec![0f32; tokens * heads * hd];
        let k = vec![0f32; tokens * kv_heads * hd];
        let v: Vec<f32> = (0..tokens * kv_heads * hd).map(|_| rnd()).collect();

        let out = d.attention_dev(&d.upload_f32(&q).unwrap(), &d.upload_f32(&k).unwrap(),
                                  &d.upload_f32(&v).unwrap(), tokens, heads, kv_heads, hd,
                                  crate::cuda::BatchMap::SINGLE).expect("attention");
        let got = d.download_f32(&out).expect("download");

        let mut worst = 0f32;
        let mut worst_at = 0usize;
        for t in 0..tokens {
            for h in 0..heads {
                let kvh = h * kv_heads / heads;
                for j in 0..hd {
                    let mut sum = 0f64;
                    for s in 0..=t {
                        sum += v[(s * kv_heads + kvh) * hd + j] as f64;
                    }
                    let want = (sum / (t + 1) as f64) as f32;
                    let diff = (got[(t * heads + h) * hd + j] - want).abs();
                    if diff > worst { worst = diff; worst_at = t; }
                }
            }
        }
        eprintln!("  uniform softmax: worst |ours - running mean| = {worst:.3e} at token {worst_at}");

        assert!(worst < 5e-3, "the uniform-softmax case is not the running mean of v");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn one_hot_scores_give_the_expected_softmax() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        for (tokens, slope) in [(37usize, 0.1f32), (48, 10.0)] {
            let worst = one_hot_case(d, tokens, slope);
            eprintln!("  one-hot: {tokens} tokens, slope {slope}: worst |ours - expected| = {worst:.3e}");
            assert!(worst < 5e-3, "one-hot scores wrong at {tokens} tokens, slope {slope}");
        }
    }

    fn one_hot_case(d: &Device, tokens: usize, slope: f32) -> f32 {
        let _ = &d;
        let (heads, kv_heads, hd) = (1usize, 1usize, 256usize);

        let mut state: u32 = 777;
        let mut rnd = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let mut q = vec![0f32; tokens * heads * hd];
        let mut k = vec![0f32; tokens * kv_heads * hd];
        for t in 0..tokens {
            q[t * heads * hd] = 1.0;

            k[t * kv_heads * hd] = slope * t as f32;
        }
        let v: Vec<f32> = (0..tokens * kv_heads * hd).map(|_| rnd()).collect();

        let out = d.attention_dev(&d.upload_f32(&q).unwrap(), &d.upload_f32(&k).unwrap(),
                                  &d.upload_f32(&v).unwrap(), tokens, heads, kv_heads, hd,
                                  crate::cuda::BatchMap::SINGLE).expect("attention");
        let got = d.download_f32(&out).expect("download");

        let scale = 1.0f64 / (hd as f64).sqrt();
        let mut worst = 0f32;
        let mut worst_t = 0usize;
        let mut worst_j = 0usize;
        for t in 0..tokens {
            let bf = |x: f32| -> f32 {
                let b = x.to_bits();
                f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16)
            };

            let scores: Vec<f64> = (0..=t)
                .map(|s| bf(slope * s as f32) as f64 * scale)
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();

            let p_bf: Vec<f64> = (0..=t).map(|s| {
                let p = (e[s] / z) as f32;
                let b = p.to_bits();
                let r = (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16;
                f32::from_bits(r) as f64
            }).collect();
            for j in 0..hd {
                let want: f64 = (0..=t).map(|s| p_bf[s] * v[s * hd + j] as f64).sum::<f64>();
                let diff = (got[t * hd + j] as f64 - want).abs() as f32;
                if diff > worst { worst = diff; worst_t = t; worst_j = j; }
            }
        }
        worst
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_grouped_mapping_pairs_each_query_head_with_its_own_kv_head() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let (tokens, heads, kv_heads, hd) = (9usize, 4usize, 2usize, 256usize);

        let mut state: u32 = 31337;
        let mut rnd = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let q = vec![0f32; tokens * heads * hd];
        let k = vec![0f32; tokens * kv_heads * hd];
        let v: Vec<f32> = (0..tokens * kv_heads * hd).map(|_| rnd()).collect();

        let out = d.attention_dev(&d.upload_f32(&q).unwrap(), &d.upload_f32(&k).unwrap(),
                                  &d.upload_f32(&v).unwrap(), tokens, heads, kv_heads, hd,
                                  crate::cuda::BatchMap::SINGLE).expect("attention");
        let got = d.download_f32(&out).expect("download");

        let rep = heads / kv_heads;
        let mut worst = 0f32;
        let mut worst_desc = String::new();
        for h in 0..heads {
            let kvh = h / rep;
            let mut w = 0f32;
            for t in 0..tokens {
                for j in 0..hd {
                    let want: f64 = (0..=t).map(|s| v[(s * kv_heads + kvh) * hd + j] as f64)
                        .sum::<f64>() / (t + 1) as f64;
                    w = w.max((got[(t * heads + h) * hd + j] as f64 - want).abs() as f32);
                }
            }
            if w > worst { worst = w; worst_desc = format!("head {h} (kv {kvh})"); }
        }
        eprintln!("  GQA mapping: worst |ours - running mean of the RIGHT kv head| = {worst:.3e}  ({worst_desc})");
        assert!(worst < 1e-2, "the grouped query/kv head mapping is wrong");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_qk_dot_product_covers_the_whole_head() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let (tokens, heads, kv_heads, hd) = (2usize, 1usize, 1usize, 256usize);

        let dim: usize = std::env::var("CLEF_TEST_DIM").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let mut q = vec![1f32; tokens * heads * hd];
        if dim != usize::MAX { q.iter_mut().for_each(|x| *x = 0.0); }
        let mut k = vec![0f32; tokens * kv_heads * hd];

        let which: usize = std::env::var("CLEF_TEST_KEY").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        for j in 0..hd { k[which * kv_heads * hd + j] = 1.0; }
        if dim != usize::MAX {
            for t in 0..tokens { q[(t * heads) * hd + dim] = 1.0; }
            for j in 0..hd { k[which * kv_heads * hd + j] = 0.0; }
            k[which * kv_heads * hd + dim] = 1.0;
        }
        let v: Vec<f32> = (0..tokens * kv_heads * hd)
            .map(|i| if i < hd { 0.0 } else { 1.0 }).collect();

        let out = d.attention_dev(&d.upload_f32(&q).unwrap(), &d.upload_f32(&k).unwrap(),
                                  &d.upload_f32(&v).unwrap(), tokens, heads, kv_heads, hd,
                                  crate::cuda::BatchMap::SINGLE).expect("attention");
        let got = d.download_f32(&out).expect("download");

        let p1: f64 = got[hd..2 * hd].iter().map(|x| *x as f64).sum::<f64>() / hd as f64;
        let scale = 1.0 / (hd as f64).sqrt();
        let want_p1 = if which == 0 { 1.0 / (1.0 + (hd as f64 * scale).exp()) }
                      else { 1.0 - 1.0 / (1.0 + (hd as f64 * scale).exp()) };
        let dims = if p1 > 0.0 && p1 < 1.0 { (1.0 / p1 - 1.0).ln() / scale } else { f64::NAN };
        eprintln!("  QK^T dim coverage: p(key 1) = {p1:.6e}   expected {want_p1:.6e}");
        eprintln!("  implied dimension count = {dims:.1} of {hd}");
        assert!(p1 < want_p1 * 100.0, "the dot product is summing far fewer dims than the head has");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_request_in_a_higher_slot_equals_the_same_request_in_slot_zero() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let hidden = config.hidden_size;

        let tokens = 267usize;
        let chunk = 2048usize;
        let cap = crate::gpu::kv_capacity();
        runner.ensure_carry_slots(cap, 2).expect("two slots");
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        let worst = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let mut run_in_slot = |slot: usize| -> (Vec<f32>, Option<Vec<u16>>, Option<Vec<f32>>) {
            runner.forward_chunk_begin_slot(cap, slot).expect("begin");
            let out = d.alloc(tokens * hidden * 4).expect("out");
            let mut pos = 0;
            while pos < tokens {
                let rows = crate::gpu::next_chunk_rows(tokens - pos).min(chunk);
                let real = rows.min(tokens - pos);
                let spec = ChunkSpec { ids: &ids, pos, slot, rows };
                let b = runner.multi_chunk_pass(&[spec]).expect("pass");
                d.copy_bytes_at(&out, pos * hidden * 4, &b, 0, real * hidden * 4).expect("copy");
                pos += real;
            }
            d.sync().expect("sync");
            let h = d.download_f32(&out).expect("hidden");
            let kv = (0..40).find_map(|li| runner.carry_kv_snapshot(li));
            let state = (0..40).find_map(|li| runner.carry_state_snapshot(li));
            (h, kv, state)
        };
        let (h0, kv0, st0) = run_in_slot(0);
        let (h1, kv1, st1) = run_in_slot(1);

        let run_multi = |slot: usize| -> (Vec<f32>, Option<Vec<u16>>, Option<Vec<f32>>) {
            runner.forward_chunk_begin_slot(cap, slot).expect("begin multi");
            let mut pos = 0;
            let mut out: Vec<f32> = Vec::with_capacity(tokens * hidden);
            while pos < tokens {
                let rows = crate::gpu::next_chunk_rows(tokens - pos).min(chunk);
                let real = rows.min(tokens - pos);
                let spec = ChunkSpec { ids: &ids, pos, slot, rows };
                let b = runner.multi_chunk_pass(&[spec]).expect("multi pass");
                let v = d.download_f32(&b).expect("download pass");
                out.extend_from_slice(&v[..real * hidden]);
                pos += real;
            }
            let kv = (0..40).find_map(|li| runner.carry_kv_snapshot(li));
            let state = (0..40).find_map(|li| runner.carry_state_snapshot(li));
            (out, kv, state)
        };
        let (hm, kvm, stm) = run_multi(1);
        let w_multi = worst(&h0, &hm);
        let first_multi = h0.iter().zip(&hm).position(|(x, y)| x != y).map(|i| i / hidden);
        eprintln!("multi_chunk_pass in slot 1 vs forward_chunk_step in slot 0: hidden {w_multi:.4e} \
                   (first differing row {first_multi:?})");

        let w_hidden = worst(&h0, &h1);
        let first = h0.iter().zip(&h1).position(|(x, y)| x != y).map(|i| i / hidden);

        let w_kv = match (kv0.as_ref(), kv1.as_ref()) {
            (Some(a), Some(b)) => {
                let per = a.len() / cap.max(1);
                let a0 = &a[..a.len().min(tokens * per)];
                let b0 = &b[b.len().min(cap * per)..];
                let n = a0.len().min(b0.len());
                a0[..n].iter().zip(&b0[..n]).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0)
            }
            _ => -1,
        };

        let w_state = match (st0.as_ref(), st1.as_ref()) {
            (Some(a), Some(b)) if a.len() == b.len() && a.len() % 2 == 0 => {
                let per = a.len() / 2;
                worst(&a[..per], &b[per..2 * per])
            }
            _ => -1.0,
        };
        eprintln!("slot 0 vs slot 1: hidden {w_hidden:.4e} (first differing row {first:?}), \
                   kv |diff| {w_kv}, state {w_state:.4e}");
        assert_eq!(w_hidden, 0.0, "the same pass in slot 1 differs from slot 0 — first row {first:?}");
        assert_eq!(w_kv, 0, "the keys in slot 1 are not at the same offsets inside their segment");
        assert_eq!(w_state, 0.0, "the GDN state after a fresh request in slot 1 is not slot 0's");

        let w_state_m = match (st0.as_ref(), stm.as_ref()) {
            (Some(a), Some(b)) if a.len() == b.len() && a.len() % 2 == 0 => {
                let per = a.len() / 2;
                worst(&a[..per], &b[per..2 * per])
            }
            _ => -1.0,
        };
        let w_kv_m = match (kv0.as_ref(), kvm.as_ref()) {
            (Some(a), Some(b)) => {
                let per = a.len() / cap.max(1);
                let a0 = &a[..a.len().min(tokens * per)];
                let b0 = &b[b.len().min(cap * per)..];
                let n = a0.len().min(b0.len());
                a0[..n].iter().zip(&b0[..n]).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0)
            }
            _ => -1,
        };
        eprintln!("  multi_chunk_pass: kv |diff| {w_kv_m}, state {w_state_m:.4e}");

        let ids_b: Vec<u32> = (0..tokens).map(|i| (500 + (i % 700)) as u32).collect();
        let run_b = |preceded_by_a: bool| -> Vec<f32> {
            if preceded_by_a {
                runner.forward_chunk_begin_slot(cap, 0).expect("begin a");
                let mut pos = 0;
                while pos < tokens {
                    let rows = crate::gpu::next_chunk_rows(tokens - pos).min(chunk);
                    let real = rows.min(tokens - pos);
                    let spec = ChunkSpec { ids: &ids, pos, slot: 0, rows };
                    let _ = runner.multi_chunk_pass(&[spec]).expect("a pass");
                    pos += real;
                }
            }
            runner.forward_chunk_begin_slot(cap, 1).expect("begin b");
            let mut pos = 0;
            let mut out: Vec<f32> = Vec::with_capacity(tokens * hidden);
            while pos < tokens {
                let rows = crate::gpu::next_chunk_rows(tokens - pos).min(chunk);
                let real = rows.min(tokens - pos);
                let spec = ChunkSpec { ids: &ids_b, pos, slot: 1, rows };
                let b = runner.multi_chunk_pass(&[spec]).expect("b pass");
                let v = d.download_f32(&b).expect("download b");
                out.extend_from_slice(&v[..real * hidden]);
                pos += real;
            }
            out
        };
        let b_alone = run_b(false);
        let b_after = run_b(true);
        let w_b = worst(&b_alone, &b_after);
        let first_b = b_alone.iter().zip(&b_after).position(|(x, y)| x != y).map(|i| i / hidden);
        eprintln!("  B alone vs B after A: hidden {w_b:.4e} (first differing row {first_b:?})");

        let run_pair = |a_tokens: usize, b_tokens: usize| -> (Vec<f32>, Vec<f32>) {
            let ids_a: Vec<u32> = (0..a_tokens).map(|i| (100 + (i % 900)) as u32).collect();
            let ids_bb: Vec<u32> = (0..b_tokens).map(|i| (500 + (i % 700)) as u32).collect();
            let run_one = |ids: &[u32], slot: usize| -> Vec<f32> {
                runner.forward_chunk_begin_slot(cap, slot).expect("begin");
                let mut pos = 0;
                let mut out: Vec<f32> = Vec::with_capacity(ids.len() * hidden);
                while pos < ids.len() {
                    let rem = ids.len() - pos;
                    let rows = crate::gpu::next_chunk_rows(rem).min(chunk);
                    let real = rows.min(rem);
                    let spec = ChunkSpec { ids, pos, slot, rows };
                    let b = runner.multi_chunk_pass(&[spec]).expect("pass");
                    let v = d.download_f32(&b).expect("dl");
                    out.extend_from_slice(&v[..real * hidden]);
                    pos += real;
                }
                out
            };
            let b_alone = run_one(&ids_bb, 1);
            let _ = run_one(&ids_a, 0);
            let b_after = run_one(&ids_bb, 1);
            (b_alone, b_after)
        };
        let (x, y) = run_pair(267, 267);
        let w_same = worst(&x, &y);
        let (x2, y2) = run_pair(255, 267);
        let w_diff = worst(&x2, &y2);
        let first_diff = x2.iter().zip(&y2).position(|(p, q)| p != q).map(|i| i / hidden);
        eprintln!("  B after A, SAME size: {w_same:.4e}   DIFFERENT sizes (255 then 267): {w_diff:.4e} \
                   (first differing row {first_diff:?})");

        let head_dir = std::path::Path::new("/models/clef-flash-nvfp4");
        let head_ck = Checkpoint::open_single(&head_dir.join("joint_head.safetensors")).expect("head ck");
        let head_cfg = crate::model::head::HeadConfig::from_dir(head_dir).expect("head cfg");
        let head = crate::gpu::load_head(d, &head_ck, crate::engine::head_config_ref(&head_cfg))
            .expect("load head");
        let q = vec![QuestionSpec { question_type: 1, question_span: (100, 140),
                                    option_spans: vec![(200, 250), (400, 450)] }];
        let ids_a: Vec<u32> = (0..255).map(|i| (100 + (i % 900)) as u32).collect();
        let ids_b2: Vec<u32> = (0..tokens).map(|i| (500 + (i % 700)) as u32).collect();
        let width = head.config.width;

        let mut run_with_sink = |with_a: bool| -> Vec<f32> {
            let pre_b = PreHead::new(d, &head, &q, tokens, chunk).expect("pre b");
            let pre_a = if with_a {
                Some(PreHead::new(d, &head, &q, 255, chunk).expect("pre a"))
            } else {
                None
            };
            if let Some(pa) = pre_a.as_ref() {
                runner.forward_chunk_begin_slot(cap, 0).expect("begin a");
                let mut pos = 0;
                while pos < 255 {
                    let rows = crate::gpu::next_chunk_rows(255 - pos).min(chunk);
                    let real = rows.min(255 - pos);
                    let spec = ChunkSpec { ids: &ids_a, pos, slot: 0, rows };
                    runner.forward_multi_chunk_step(&[spec], &[pa]).expect("a pass");
                    pos += real;
                }
            }
            runner.forward_chunk_begin_slot(cap, 1).expect("begin b");
            let mut pos = 0;
            while pos < tokens {
                let rows = crate::gpu::next_chunk_rows(tokens - pos).min(chunk);
                let real = rows.min(tokens - pos);
                let spec = ChunkSpec { ids: &ids_b2, pos, slot: 1, rows };
                runner.forward_multi_chunk_step(&[spec], &[&pre_b]).expect("b pass");
                pos += real;
            }
            d.download_f32(&pre_b.memory).expect("mem b")
        };
        let mem_alone = run_with_sink(false);
        let mem_with_a = run_with_sink(true);
        let w_sink = worst(&mem_alone, &mem_with_a);
        let first_sink = mem_alone.iter().zip(&mem_with_a).position(|(x, y)| x != y)
            .map(|i| i / width);
        eprintln!("  B's memory alone vs with A's PreHead alive: {w_sink:.4e} \
                   (first differing row {first_sink:?})");

        let run_multi_slot1 = |pre: &PreHead| {
            runner.forward_chunk_begin_slot(cap, 1).expect("begin batch");
            let mut pos = 0;
            while pos < tokens {
                let rows = crate::gpu::next_chunk_rows(tokens - pos).min(chunk);
                let real = rows.min(tokens - pos);
                let spec = ChunkSpec { ids: &ids_b2, pos, slot: 1, rows };
                runner.forward_multi_chunk_step(&[spec], &[pre]).expect("batch pass");
                pos += real;
            }
        };
        let batch_pre = PreHead::new(d, &head, &q, tokens, chunk).expect("pre batch");
        run_multi_slot1(&batch_pre);
        let mem_batch = d.download_f32(&batch_pre.memory).expect("mem batch");
        let solo_pre = PreHead::new(d, &head, &q, tokens, chunk).expect("pre solo");
        runner.forward_chunk_begin_slot(cap, 0).expect("begin solo");
        runner.forward_chunked_into(&ids_b2, chunk, cap, &solo_pre).expect("solo into");
        let mem_solo = d.download_f32(&solo_pre.memory).expect("mem solo");

        let w_entry = worst(&mem_solo[..tokens * width], &mem_batch[..tokens * width]);
        let first_entry = mem_solo[..tokens * width].iter().zip(&mem_batch[..tokens * width])
            .position(|(x, y)| x != y).map(|i| i / width);
        eprintln!("  forward_chunked_into (slot 0) vs forward_multi_chunk_step (slot 1), request rows only: \
                   {w_entry:.4e} (first differing row {first_entry:?})");
        let _ = (&solo_pre, &batch_pre);
        assert_eq!(w_entry, 0.0,
                   "the two arms' entry points disagree inside the request — first differing row \
                    {first_entry:?}");

        let ids_a2: Vec<u32> = (0..255).map(|i| (100 + (i % 900)) as u32).collect();
        let run_a = |pre: &PreHead| {
            runner.forward_chunk_begin_slot(cap, 0).expect("begin a");
            let mut pos = 0;
            while pos < 255 {
                let rows = crate::gpu::next_chunk_rows(255 - pos).min(chunk);
                let real = rows.min(255 - pos);
                let spec = ChunkSpec { ids: &ids_a2, pos, slot: 0, rows };
                runner.forward_multi_chunk_step(&[spec], &[pre]).expect("a pass");
                pos += real;
            }
        };

        let pa1 = PreHead::new(d, &head, &q, 255, chunk).expect("pre a1");
        run_a(&pa1);
        let pb1 = PreHead::new(d, &head, &q, tokens, chunk).expect("pre b1");
        run_multi_slot1(&pb1);
        let mem_h_batch = d.download_f32(&pb1.memory).expect("mem h batch");

        let pa2 = PreHead::new(d, &head, &q, 255, chunk).expect("pre a2");
        run_a(&pa2);
        let pb2 = PreHead::new(d, &head, &q, tokens, chunk).expect("pre b2");
        runner.forward_chunk_begin_slot(cap, 0).expect("begin b solo");
        runner.forward_chunked_into(&ids_b2, chunk, cap, &pb2).expect("b solo into");
        let mem_h_solo = d.download_f32(&pb2.memory).expect("mem h solo");
        let w_h = worst(&mem_h_solo[..tokens * width], &mem_h_batch[..tokens * width]);
        let first_h = mem_h_solo[..tokens * width].iter().zip(&mem_h_batch[..tokens * width])
            .position(|(x, y)| x != y).map(|i| i / width);
        eprintln!("  A present, B in slot 0 vs slot 1: {w_h:.4e} (first differing row {first_h:?})");
        let _ = (&pa1, &pa2, &pb1, &pb2);
        assert_eq!(w_h, 0.0,
                   "with another request present, B's head input depends on its slot — first differing \
                    row {first_h:?}");
        assert_eq!(w_sink, 0.0,
                   "a request's head input depends on whether another request's sink is alive — \
                    first differing row {first_sink:?}");
        assert_eq!(w_diff, 0.0,
                   "a request in slot 1 depends on the SIZE of the request that ran before it in slot 0 — \
                    first differing row {first_diff:?}");
        assert_eq!(w_b, 0.0,
                   "a request in slot 1 depends on whether another request ran in slot 0 — \
                    first differing row {first_b:?}");
        assert_eq!(w_multi, 0.0,
                   "multi_chunk_pass differs from forward_chunk_step for the SAME slot and geometry — \
                    first differing row {first_multi:?}");
        assert_eq!(w_kv_m, 0, "multi_chunk_pass wrote the keys somewhere else");
        assert_eq!(w_state_m, 0.0, "multi_chunk_pass left a different GDN state");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_sink_path_matches_the_whole_tensor_path() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let hidden = config.hidden_size;

        let tokens = 8192usize;
        let chunk = 2048usize;
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        let q = vec![QuestionSpec {
            question_type: 1,
            question_span: (100, 140),
            option_spans: vec![(2000, 2050), (4000, 4050), (8000, 8040)],
        }];

        let head_dir = std::path::Path::new("/models/clef-flash-nvfp4");
        let head_ck = Checkpoint::open_single(&head_dir.join("joint_head.safetensors"))
            .expect("the head checkpoint");
        let head_cfg = crate::model::head::HeadConfig::from_dir(head_dir).expect("head config");
        let head = crate::gpu::load_head(d, &head_ck, crate::engine::head_config_ref(&head_cfg))
            .expect("load head");
        let lm_head = HostTable::map(&checkpoint.shard_path("lm_head.weight").expect("lm_head path"),
                                     "lm_head.weight")
            .expect("map lm_head");

        let whole = runner.forward_dev(&ids).expect("whole-tensor backbone");
        let a = head_forward(d, &whole, tokens, &ids, &q, &lm_head, &head).expect("whole-tensor head");

        let pre = PreHead::new(d, &head, &q, tokens, chunk).expect("prehead");
        runner.forward_chunked_into(&ids, chunk, tokens, &pre).expect("sink backbone");
        let b = head_forward_pre_eager(d, &pre, tokens, &ids, &q, &lm_head, &head).expect("sink head");

        let worst = a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y.iter()))
            .map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        eprintln!("  {} tokens: worst |whole-tensor - sink| = {worst:.3e} over {} options",
                  tokens, a.iter().map(|v| v.len()).sum::<usize>());
        let _ = hidden;
        assert_eq!(worst, 0.0, "the sink path must equal the whole-tensor path exactly");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_cache_the_card_cannot_hold_is_refused() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };

        let err = match runner.ensure_carry(1_048_576) {
            Ok(()) => panic!("a 32 GB cache must be refused on a 32 GB card"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        eprintln!("  refused: {msg}");

        let c = &config;
        let n_full = runner.backbone.layers.iter()
            .filter(|l| matches!(l, DeviceLayer::Full(_))).count();
        let per_token = 2 * c.num_key_value_heads * c.head_dim * 2 * n_full;
        eprintln!(
            "  KV CACHE: {} B/token over {n_full} full-attention layers; \
             the server's gate is {} tokens = {} MiB; this card can hold {} tokens = {} MiB",
            per_token,
            crate::engine::max_input_tokens(),
            per_token * crate::engine::max_input_tokens() / (1 << 20),
            runner.carry_capacity_that_fits(),
            per_token * runner.carry_capacity_that_fits() / (1 << 20),
        );
        assert!(msg.contains("only"), "the refusal must give the numbers: {msg}");
        assert!(msg.contains("CLEF_MAX_LENGTH"), "and say how to fix it: {msg}");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_request_beyond_the_cache_is_refused() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let chunk = 2048usize;
        let cap = 4096usize;
        let tokens = cap + chunk;
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();
        let err = match runner.forward_chunked_dev(&ids, chunk, cap) {
            Ok(_) => panic!("a request beyond the bound must be refused"),
            Err(e) => e,
        };
        let refused = err
            .downcast_ref::<crate::engine::Refused>()
            .unwrap_or_else(|| panic!("the refusal must be a `Refused` (a 400), got: {err:#}"));
        eprintln!("  refused {tokens} tokens: {}", refused.why);
        assert_eq!(refused.tokens, tokens, "the caller is told the true count");
        assert!(
            refused.why.contains("CLEF_KV_CAPACITY"),
            "the reason must name the cache and how to raise it: {}",
            refused.why
        );
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn one_layer_answers_the_same_twice() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let hidden = config.hidden_size;
        let tokens = 16_384usize;
        let DeviceLayer::Linear(w) = &runner.backbone.layers[0] else { panic!("layer 0 is linear") };
        let x: Vec<f32> = (0..tokens * hidden).map(|i| ((i * 37 % 211) as f32 - 105.0) / 97.0).collect();
        let xd = d.upload_f32(&x).expect("upload");
        let map = crate::cuda::BatchMap::SINGLE;
        let a = runner.delta_net_layer_dev(&xd, tokens, w, map).expect("a");
        let b = runner.delta_net_layer_dev(&xd, tokens, w, map).expect("b");
        let a = d.download_rows_f32(&a, tokens, hidden).expect("dl");
        let b = d.download_rows_f32(&b, tokens, hidden).expect("dl");
        let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        let f = a.iter().zip(&b).position(|(p, q)| p != q).map(|i| i / hidden);
        eprintln!("  one layer twice: worst = {worst:.3e}, first row {f:?}");
        assert_eq!(worst, 0.0, "one layer answered differently on the second run");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_first_layer_ops_answer_the_same_twice() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let c = &config;
        let (tokens, hidden) = (16_384usize, c.hidden_size);
        let eps = c.rms_norm_eps;
        let DeviceLayer::Linear(w) = &runner.backbone.layers[0] else { panic!("layer 0 is linear") };
        let x: Vec<f32> = (0..tokens * hidden).map(|i| ((i * 37 % 211) as f32 - 105.0) / 97.0).collect();
        let xd = d.upload_f32(&x).expect("upload");

        const RUNS: usize = 12;
        macro_rules! twice {
            ($name:expr, $run:expr) => {{
                let mut outs = Vec::with_capacity(RUNS);
                for _ in 0..RUNS {
                    outs.push($run);
                }
                let first = d.download_f32(&outs[0]).expect("dl");
                let mut worst = 0.0f32;
                let mut bad = 0usize;
                for o in &outs[1..] {
                    let x = d.download_f32(o).expect("dl");
                    let w = first.iter().zip(&x).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                    if w != 0.0 {
                        bad += 1;
                    }
                    worst = worst.max(w);
                }
                eprintln!("    {:<14} worst = {worst:.3e}  differing runs = {bad}/{RUNS}", $name);
                worst
            }};
        }

        let mut bad = 0.0f32;
        bad = bad.max(twice!("rms_norm", d.rms_norm_dev(&xd, &w.input_layernorm, tokens, hidden, eps).expect("n")));
        let normed = d.rms_norm_dev(&xd, &w.input_layernorm, tokens, hidden, eps).expect("normed");
        bad = bad.max(twice!("in_proj_qkv", run_projection_dev(d, &normed, tokens, &w.in_proj_qkv, None).expect("p")));
        let mixed = run_projection_dev(d, &normed, tokens, &w.in_proj_qkv, None).expect("mixed");
        let conv_dim = c.conv_dim();
        bad = bad.max(twice!("conv", d.causal_conv_silu_dev(&mixed, &w.conv_weight, tokens, conv_dim,
                                                            c.linear_conv_kernel_dim, None, None, None).expect("c")));
        let conv = d.causal_conv_silu_dev(&mixed, &w.conv_weight, tokens, conv_dim,
                                          c.linear_conv_kernel_dim, None, None, None).expect("conv");
        bad = bad.max(twice!("in_proj_z", run_projection_dev(d, &normed, tokens, &w.in_proj_z, None).expect("p")));
        bad = bad.max(twice!("in_proj_a", run_projection_dev(d, &normed, tokens, &w.in_proj_a, None).expect("p")));
        bad = bad.max(twice!("in_proj_b", run_projection_dev(d, &normed, tokens, &w.in_proj_b, None).expect("p")));
        let aa = run_projection_dev(d, &normed, tokens, &w.in_proj_a, None).expect("a");
        let bb = run_projection_dev(d, &normed, tokens, &w.in_proj_b, None).expect("b");
        let vh = c.linear_num_value_heads;
        bad = bad.max(twice!("gates", d.gdn_gates_dev(&aa, &bb, &w.a_log, &w.dt_bias, tokens, vh).expect("g").0));
        let (decay, beta) = d.gdn_gates_dev(&aa, &bb, &w.a_log, &w.dt_bias, tokens, vh).expect("gates");
        let prep = d.gdn_prep_dev(&conv, tokens, c.linear_num_key_heads, c.linear_key_head_dim, vh, conv_dim)
            .expect("prep");
        bad = bad.max(twice!("gdn_scan", d.gdn_chunk_scan_prepared_dev(&prep, &decay, &beta, tokens, vh,
                                                                       c.linear_value_head_dim,
                                                                       crate::cuda::BatchMap::SINGLE).expect("s")));
        let o = d.gdn_chunk_scan_prepared_dev(&prep, &decay, &beta, tokens, vh,
                                              c.linear_value_head_dim, crate::cuda::BatchMap::SINGLE).expect("scan");
        let z = run_projection_dev(d, &normed, tokens, &w.in_proj_z, None).expect("z");
        bad = bad.max(twice!("gated_norm", d.rms_norm_gated_dev(&o, &z, &w.norm_weight, tokens * vh,
                                                                c.linear_value_head_dim, eps).expect("gn")));
        let gated = d.rms_norm_gated_dev(&o, &z, &w.norm_weight, tokens * vh, c.linear_value_head_dim, eps)
            .expect("gated");
        bad = bad.max(twice!("out_proj", run_projection_dev(d, &gated, tokens, &w.out_proj, Some(&xd)).expect("p")));
        let y = run_projection_dev(d, &gated, tokens, &w.out_proj, Some(&xd)).expect("y");
        bad = bad.max(twice!("mlp_norm", d.rms_norm_dev(&y, &w.post_attention_layernorm, tokens, hidden, eps).expect("n")));
        let normed2 = d.rms_norm_dev(&y, &w.post_attention_layernorm, tokens, hidden, eps).expect("n2");
        bad = bad.max(twice!("mlp_gate", run_projection_dev(d, &normed2, tokens, &w.mlp_gate, None).expect("p")));
        bad = bad.max(twice!("mlp_up", run_projection_dev(d, &normed2, tokens, &w.mlp_up, None).expect("p")));
        let gate = run_projection_dev(d, &normed2, tokens, &w.mlp_gate, None).expect("gate");
        let up = run_projection_dev(d, &normed2, tokens, &w.mlp_up, None).expect("up");
        bad = bad.max(twice!("swiglu", d.swiglu_dev(&gate, &up, tokens * c.intermediate_size).expect("s")));
        let act = d.swiglu_dev(&gate, &up, tokens * c.intermediate_size).expect("act");
        bad = bad.max(twice!("mlp_down", run_projection_dev(d, &act, tokens, &w.mlp_down, Some(&y)).expect("p")));

        let inter = c.intermediate_size;
        let act_global = match &w.mlp_down {
            DeviceProjection::Nvfp4 { input_global_scale, .. } => *input_global_scale,
            _ => None,
        };
        if let Some(g) = act_global {
            let tile = Device::nvfp4_m_tile_for(tokens);
            bad = bad.max(twice!("swiglu_fused", d.swiglu_quantise_nvfp4_dev(&gate, &up, tokens, inter, g, tile)
                .expect("fused").expect("the shape is the atom's").0));
            bad = bad.max(twice!("fused_scale", d.swiglu_quantise_nvfp4_dev(&gate, &up, tokens, inter, g, tile)
                .expect("fused").expect("the shape is the atom's").1));
            let (q, qs) = d.swiglu_quantise_nvfp4_dev(&gate, &up, tokens, inter, g, tile)
                .expect("fused").expect("shape");
            bad = bad.max(twice!("fused_gemm", run_nvfp4_projection_from_quantised(d, &q, &qs, tokens,
                                                                                  &w.mlp_down, Some(&y))
                .expect("gemm").expect("the projection produced a buffer")));
        } else {
            eprintln!("    (layer 0's mlp_down is not NVFP4; the fused NVFP4 path was not exercised)");
        }
        eprintln!("  worst over every op: {bad:.3e}");
        assert_eq!(bad, 0.0, "an op in the first layer answered differently on the second run");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_long_pass_above_eight_thousand_is_deterministic() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let hidden = config.hidden_size;
        for tokens in [16_384usize, 24_576] {
            let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

            if std::env::var("CLEF_ONE_RUN").is_ok() {
                let a = runner.forward_dev(&ids).expect("pass");
                let a = runner.device.download_rows_f32(&a, tokens, hidden).expect("dl");
                let sum = a.iter().fold(0u64, |acc, x| acc.wrapping_mul(1_000_003)
                    .wrapping_add(x.to_bits() as u64));
                eprintln!("  ONE RUN {tokens}: sum={sum:016x}");
                continue;
            }
            eprintln!("=== RUN A {tokens} ===");
            let a = runner.forward_dev(&ids).expect("pass");
            let a = runner.device.download_rows_f32(&a, tokens, hidden).expect("dl");
            eprintln!("=== RUN B {tokens} ===");
            let b = runner.forward_dev(&ids).expect("pass");
            let b = runner.device.download_rows_f32(&b, tokens, hidden).expect("dl");
            let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let f = a.iter().zip(&b).position(|(x, y)| x != y);
            let rows: Vec<usize> = a
                .iter()
                .zip(&b)
                .enumerate()
                .filter(|(_, (x, y))| x != y)
                .map(|(i, _)| i / hidden)
                .collect();
            let mut uniq = rows.clone();
            uniq.dedup();
            eprintln!(
                "  forward_dev {tokens} against itself: worst = {d:.3e}, first {f:?} \
                 elements={} rows={} first_row={:?} last_row={:?}",
                rows.len(), uniq.len(), uniq.first(), uniq.last()
            );
            assert_eq!(d, 0.0, "{tokens} tokens answered differently on the second run");
        }
    }

    fn a_ragged_tail_matches_through_both_paths() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else { return };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) { Ok(r) => r, Err(_) => return };
        let d = &runner.device;
        let head_dir = std::path::Path::new("/models/clef-flash-nvfp4");
        let head_ck = Checkpoint::open_single(&head_dir.join("joint_head.safetensors")).expect("head ck");
        let head_cfg = crate::model::head::HeadConfig::from_dir(head_dir).expect("head cfg");
        let head = crate::gpu::load_head(d, &head_ck, crate::engine::head_config_ref(&head_cfg))
            .expect("load head");
        let q = vec![QuestionSpec { question_type: 1, question_span: (100, 140),
                                    option_spans: vec![(2000, 2050), (4000, 4050)] }];
        let (tokens, chunk, cap) = (4096usize, 2048usize, 8192usize);
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        let _ = (tokens, chunk);

        runner.forward_chunk_begin_slot(cap, 0).expect("begin");
        let pre_solo = PreHead::new(d, &head, &q, tokens, chunk).expect("pre solo");
        runner.forward_chunked_into(&ids, chunk, cap, &pre_solo).expect("solo passes");
        let a = d.download_f32(&pre_solo.memory).expect("mem solo");

        runner.forward_chunk_begin_slot(cap, 0).expect("begin");
        let pre_batch = PreHead::new(d, &head, &q, tokens, chunk).expect("pre batch");
        for p0 in (0..tokens).step_by(chunk) {
            let rows = chunk.min(tokens - p0);
            let spec = ChunkSpec { ids: &ids, pos: p0, slot: 0, rows };
            runner.forward_multi_chunk_step(&[spec], &[&pre_batch]).expect("batched pass");
        }
        let b = d.download_f32(&pre_batch.memory).expect("mem batch");

        let w = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let width = head.config.width;
        let first = a.iter().zip(&b).position(|(x, y)| x != y);
        eprintln!("ragged tail: worst {w:.3e}  first differing row {:?} of {}",
                  first.map(|i| i / width), a.len() / width);
        assert_eq!(w, 0.0, "a ragged tail differs between the solo and batched paths");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_whole_range_is_served_in_chunks() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let hidden = config.hidden_size;

        let chunk = 2048usize;

        let tokens = crate::engine::max_input_tokens();
        assert!(tokens % chunk == 0, "an even number of chunks");
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        eprintln!(
            "  chunking_enabled={} max_pass={} max_input={}",
            crate::gpu::chunking_enabled(),
            crate::gpu::max_chunk_size(),
            crate::engine::max_input_tokens()
        );
        let warmed = runner.warm_up(&crate::engine::dense_warm_up_sizes()).expect("warm-up").captured;
        eprintln!("  warm-up captured {warmed} graphs");

        eprintln!(
            "  after load: {} graphs, block {} MiB",
            runner.graph_count(),
            runner.device.graph_pool_high() / (1 << 20)
        );
        let first = runner.forward_chunked_dev(&ids, chunk, tokens).expect("first request");
        runner.device.sync().expect("sync");
        let first = runner.device.download_rows_f32(&first, tokens, hidden).expect("download");

        let pending = runner.capture_pending_one().expect("capture");
        assert!(!pending, "the warm-up should have captured every geometry — nothing was pending");
        assert!(runner.graph_ready_for(chunk, GraphMode::Chunked), "a chunk must be graphed");

        let tail = (tokens % chunk).div_ceil(crate::cuda::CHUNK_GRANULE) * crate::cuda::CHUNK_GRANULE;
        assert_eq!(tail, 0, "this request is an exact number of chunks; use a ragged one below");
        let block = runner.device.graph_pool_high();
        eprintln!(
            "  {tokens} tokens = {} passes of {chunk}; graph pool high-water {:.0} MiB",
            tokens / chunk,
            block as f64 / (1024.0 * 1024.0)
        );

        let long_tokens = tokens / 2;
        assert!(long_tokens % chunk == 0, "an even number of chunks");
        let half: Vec<u32> = (0..long_tokens).map(|i| (100 + (i % 900)) as u32).collect();
        let short_run = runner.forward_chunked_dev(&half, chunk, tokens).expect("half-length request");
        runner.device.sync().expect("sync");
        let after_half = runner.device.graph_pool_high();
        eprintln!(
            "  {long_tokens} tokens = {} passes; block {} MiB — the same as the {tokens}-token request's {} MiB",
            long_tokens / chunk,
            after_half as f64 / (1024.0 * 1024.0),
            block as f64 / (1024.0 * 1024.0)
        );
        assert_eq!(
            after_half, block,
            "the block moved from {} MiB to {} MiB when the REQUEST doubled — it scales with the request, \
             not with the pass",
            block / (1024 * 1024),
            after_half / (1024 * 1024)
        );
        drop(short_run);

        let second = runner.forward_chunked_dev(&ids, chunk, tokens).expect("second request");
        runner.device.sync().expect("sync");
        let second = runner.device.download_rows_f32(&second, tokens, hidden).expect("download");
        let worst = first.iter().zip(&second).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  worst |first request - second request| = {worst:.3e}");
        assert_eq!(worst, 0.0, "two runs of the same request must agree exactly");

        let bad = first.iter().chain(second.iter()).filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite values in a {tokens}-token answer");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_graphed_chunk_matches_the_eager_one() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let hidden = config.hidden_size;
        for (chunk, mult) in [(512usize, 2usize), (2048, 2), (2048, 4), (2048, 6), (2048, 8)] {
        let tokens = mult * chunk;

        let capacity = 16_384usize;
        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        let eager = runner.forward_chunked_dev(&ids, chunk, capacity).expect("eager chunked");
        runner.device.sync().expect("sync");
        let eager = runner.device.download_rows_f32(&eager, tokens, hidden).expect("download");

        let sum = eager.iter().fold(0u64, |a, x| a.wrapping_mul(1_000_003).wrapping_add(x.to_bits() as u64));
        eprintln!("  EAGER CHECKSUM {chunk} = {sum:016x}");

        let already = runner.graph_ready_for(chunk, GraphMode::Chunked);
        assert!(
            already || !runner.graph_ready_for(chunk, GraphMode::Chunked),
            "unreachable"
        );
        let _ = already;

        let mut captured = 0;
        while runner.capture_pending_one().expect("capture") {
            captured += 1;
        }
        eprintln!("  captured {captured} graph(s) on the idle window");
        assert!(runner.graph_ready_for(chunk, GraphMode::Chunked), "the chunk's graph must exist now");
        let before = runner.graph_count();

        let graphed = runner.forward_chunked_dev(&ids, chunk, capacity).expect("graphed chunked");
        runner.device.sync().expect("sync");
        let graphed = runner.device.download_rows_f32(&graphed, tokens, hidden).expect("download");
        let worst = eager.iter().zip(&graphed).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let first = eager.iter().zip(&graphed).position(|(a, b)| a != b);
        eprintln!(
            "  {tokens} tokens in 2 x {chunk}: worst |eager chunk - graphed chunk| = {worst:.3e} \
             (first at flat {first:?} = row {:?} col {:?})",
            first.map(|i| i / hidden),
            first.map(|i| i % hidden)
        );
        assert_eq!(worst, 0.0, "a graphed chunk must equal the eager one exactly");

        assert_eq!(runner.graph_count(), before, "serving captured a graph");
        let nodes = runner.graph_memory_nodes(chunk, GraphMode::Chunked);
        eprintln!("  the chunked graph carries {nodes:?}");
        assert_eq!(nodes, Some((0, 0, 0)), "a captured chunk must carry no alloc, free or memcpy node");
        }
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn a_chunked_request_matches_one_pass() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let hidden = config.hidden_size;

        let tokens = 4096usize;

        let capacity = crate::gpu::kv_capacity();

        let ids: Vec<u32> = (0..tokens).map(|i| (100 + (i % 900)) as u32).collect();

        let one_pass = |ids: &[u32], tokens: usize| -> Vec<f32> {
            runner.forward_chunk_begin(capacity).expect("begin");
            let spec = ChunkSpec { ids, pos: 0, slot: 0, rows: tokens };
            let buf = runner.multi_chunk_pass(&[spec]).expect("one pass");
            runner.forward_chunk_end(tokens, tokens).expect("end");
            runner.device.download_rows_f32(&buf, tokens, hidden).expect("download")
        };
        let whole = one_pass(&ids, tokens);

        {
            let served = runner.forward_dev(&ids).expect("forward_dev");
            let served = runner.device.download_rows_f32(&served, tokens, hidden).expect("download");
            let w = whole.iter().zip(&served).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let first = whole.iter().zip(&served).position(|(a, b)| a != b);
            eprintln!("  one pass vs forward_dev (which chunks at 2048): worst {w:.3e}, first {first:?}");
            assert_eq!(w, 0.0, "the serving entry point must equal one pass exactly");
        }

        {
            let a = runner.forward_chunked_dev(&ids, 2048, capacity).expect("repeat a");
            let a = runner.device.download_rows_f32(&a, tokens, hidden).expect("dl a");
            let b = runner.forward_chunked_dev(&ids, 2048, capacity).expect("repeat b");
            let b = runner.device.download_rows_f32(&b, tokens, hidden).expect("dl b");
            let w = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let first = a.iter().zip(&b).position(|(x, y)| x != y);
            eprintln!("  SAME CALL TWICE: worst {w:.3e}, first differing row {first:?}");
        }

        for chunk in [1024usize, 2048, 512, 256, 128] {
            assert_eq!(chunk % crate::cuda::CHUNK_GRANULE, 0, "and on the granule");
            let cut = runner.forward_chunked_dev(&ids, chunk, capacity).expect("chunked");
            let cut = runner.device.download_rows_f32(&cut, tokens, hidden).expect("download");
            let worst = whole
                .iter()
                .zip(&cut)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let first = whole.iter().zip(&cut).position(|(a, b)| a != b);
            eprintln!(
                "  {tokens} tokens in {} x {chunk}: worst |one pass - chunked| = {worst:.3e}, first differing row {first:?}",
                tokens / chunk
            );
            assert_eq!(worst, 0.0, "a chunked request must equal one pass exactly");
        }

        {
            let big: Vec<u32> = (0..4096).map(|i| (100 + (i % 900)) as u32).collect();

            let plain = runner.forward_dev(&big[..2048]).expect("plain pass");
            let plain = runner.device.download_rows_f32(&plain, 2048, hidden).expect("download");
            let cached = runner.forward_chunked_dev(&big[..2048], 2048, capacity).expect("one chunk");
            let cached = runner.device.download_rows_f32(&cached, 2048, hidden).expect("download");
            let d = plain.iter().zip(&cached).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let f = plain.iter().zip(&cached).position(|(a, b)| a != b);
            eprintln!("  2048 plain vs 2048 one-chunk-cached: worst = {d:.3e}, first {f:?}");
            assert_eq!(d, 0.0, "the cache path must equal the plain pass at cache_len = 0");

            let one = runner.forward_chunked_dev(&big[..2048], 2048, capacity).expect("one chunk");
            let one = runner.device.download_rows_f32(&one, 2048, hidden).expect("download");

            let cut = runner.forward_chunked_dev(&big[..4096], 2048, capacity).expect("chunked");
            let cut = runner.device.download_rows_f32(&cut, 4096, hidden).expect("download");
            let worst = one.iter().zip(&cut).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let first = one.iter().zip(&cut).position(|(a, b)| a != b);
            eprintln!("  2048 first chunk vs 2048 one chunk: worst = {worst:.3e}, first {first:?}");
            assert_eq!(worst, 0.0, "the first chunk must equal the one-chunk pass");
        }

        {
            let ragged: Vec<u32> = (0..700).map(|i| (100 + (i % 900)) as u32).collect();
            let whole = runner.forward_dev(&ragged).expect("one pass");
            let whole = runner.device.download_rows_f32(&whole, 700, hidden).expect("download");
            let cut = runner.forward_chunked_dev(&ragged, crate::gpu::max_chunk_size(), capacity)
                .expect("the engine's own cut");
            let cut = runner.device.download_rows_f32(&cut, 700, hidden).expect("download");
            let worst = whole.iter().zip(&cut).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let first = whole.iter().zip(&cut).position(|(a, b)| a != b);
            eprintln!("  700 tokens at the ENGINE'S cut (max_chunk_size {}): worst {worst:.3e}, first {first:?}",
                      crate::gpu::max_chunk_size());
            assert_eq!(worst, 0.0, "the engine's own decomposition must equal one pass");
        }

        let ragged: Vec<u32> = (0..700).map(|i| (100 + (i % 900)) as u32).collect();
        let whole = runner.forward_dev(&ragged).expect("one pass");
        let whole = runner.device.download_rows_f32(&whole, 700, hidden).expect("download");
        let cut = runner.forward_chunked_dev(&ragged, 384, capacity).expect("chunked");
        let cut = runner.device.download_rows_f32(&cut, 700, hidden).expect("download");
        let worst = whole.iter().zip(&cut).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let first = whole.iter().zip(&cut).position(|(a, b)| a != b);
        eprintln!("  700 tokens in 384 + 316 (padded to 320): worst = {worst:.3e}, first differing row {first:?}");
        assert_eq!(worst, 0.0, "a ragged tail must equal one pass exactly");
    }

    #[test]

    #[test]
    #[ignore = "loads the head onto the device; run on the card"]
    fn the_graphed_head_matches_the_direct_one() {
        use crate::model::head::support;

        let Some(f) = support::load() else {
            eprintln!("SKIP: clef_head_fixture.json not found");
            return;
        };
        let Ok(head_ckpt) = Checkpoint::open_single("/models/clef-flash/joint_head.safetensors")
        else {
            eprintln!("SKIP: joint_head.safetensors not present; run in the aids-cuda container");
            return;
        };
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let config = support::head_config(&f);
        let hidden_size = config.hidden_size;
        let head = load_head(&device, &head_ckpt, config).expect("head onto the device");

        let tokens = f["tokens"].as_u64().unwrap() as usize;
        let hidden = support::lcg(f["seed_hidden"].as_u64().unwrap() as u32, tokens * hidden_size);
        let input_ids = support::input_ids_from(&f);
        let questions = support::questions_from(&f);
        let lm_head = support::lm_head_from(&f, tokens, hidden_size, &input_ids);
        let hidden_dev = device.upload_f32(&hidden).expect("upload hidden");

        let direct = head_forward(&device, &hidden_dev, tokens, &input_ids, &questions, &lm_head,
                                  &head)
            .expect("direct head");

        let kv = KvGraphs::capture(&device, &head).expect("k/v graphs");
        assert_eq!(kv.len(), 6 * HEAD_RUNGS.len(), "one graph per (layer, rung)");
        let pre = PreHead::from_hidden(&device, &head, &questions, tokens, &hidden_dev)
            .expect("sink from hidden");
        let mut worst = 0.0f32;
        for pass in 0..3 {
            let got = head_forward_pre_eager_with(&device, &pre, tokens, &input_ids, &questions,
                                                  &lm_head, &head, &kv)
                .expect("chunked head");
            assert_eq!(got.len(), direct.len(), "a different question count");
            let w = direct.iter().flatten().zip(got.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("  pass {pass}: worst |chunked+graphed - whole-tensor| = {w:.3e}");
            worst = worst.max(w);
        }

        assert!(worst < 1e-3, "the chunked head differs from the whole-tensor one by {worst:.3e}");
    }

    #[test]
    fn the_head_on_the_device_matches_the_reference() {
        use crate::model::head::support;

        let Some(f) = support::load() else {
            eprintln!("SKIP: clef_head_fixture.json not found");
            return;
        };
        let Ok(head_ckpt) = Checkpoint::open_single("/models/clef-flash/joint_head.safetensors")
        else {
            eprintln!("SKIP: joint_head.safetensors not present; run in the aids-cuda container");
            return;
        };
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let config = support::head_config(&f);
        let (hidden_size, width) = (config.hidden_size, config.width);
        let head = load_head(&device, &head_ckpt, config).expect("head onto the device");
        eprintln!("  head loaded onto the device ({width} wide, {hidden_size} hidden)");

        let tokens = f["tokens"].as_u64().unwrap() as usize;
        let hidden = support::lcg(f["seed_hidden"].as_u64().unwrap() as u32,
                                 tokens * hidden_size);
        let input_ids = support::input_ids_from(&f);
        let questions = support::questions_from(&f);
        let lm_head = support::lm_head_from(&f, tokens, hidden_size, &input_ids);

        let hidden_dev = device.upload_f32(&hidden).expect("upload hidden");
        let got = head_forward(&device, &hidden_dev, tokens, &input_ids, &questions, &lm_head,
                               &head)
            .expect("gpu head");

        if std::env::var("CLEF_HEAD_TRACE").is_ok() {
            use crate::model::head::{head_forward as cpu_head_forward, HeadWeights};
            let cfg = support::head_config(&f);
            let w = HeadWeights::load(&head_ckpt, &cfg).expect("cpu head weights");
            let cpu = cpu_head_forward(&hidden, tokens, &input_ids, &questions, &w, &cfg,
                                       &lm_head, 1e-5).expect("cpu head");
            let mut worst_cpu = 0.0f32;
            for (g, c) in got.iter().zip(cpu.iter()) {
                for (a, b) in g.iter().zip(c.iter()) {
                    worst_cpu = worst_cpu.max((a - b).abs());
                }
            }
            eprintln!("  device vs CPU oracle: worst |diff| {worst_cpu:.3e}");
        }

        let want: Vec<Vec<f32>> = f["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect();

        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for (qi, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            for (oi, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                eprintln!("    q{qi} option {oi}: got {a:+.6}  want {b:+.6}  |diff| {:.3e}",
                          (a - b).abs());
                worst = worst.max((a - b).abs());
                scale = scale.max(b.abs());
            }
        }
        eprintln!("  worst |diff| {worst:.3e} against logits of scale {scale:.3e}");

        assert!(
            worst < 1e-2,
            "the head's device path disagrees with the reference by {worst:.3e} — a wiring fault, \
             not BF16 rounding (which measures 3.1e-3)"
        );
    }

    #[test]
    #[ignore = "needs the checkpoint and the card"]
    fn which_m_gives_zero_rows() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let d = &runner.device;
        let inter = config.intermediate_size;
        let hidden = config.hidden_size;
        let idx = (0..config.num_hidden_layers)
            .find(|&i| config.layer_kind(i) == crate::model::LayerKind::FullAttention)
            .expect("a full-attention layer");
        let DeviceLayer::Full(w) = &runner.backbone.layers[idx] else { panic!("not Full") };
        let DeviceProjection::Nvfp4 { packed, scales, global_scale, input_global_scale, .. } =
            &w.mlp_gate else { panic!("mlp_gate is not NVFP4") };
        let gs = input_global_scale.expect("activation scale");

        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0)
        };
        for m in [7usize, 16, 17, 31, 32, 33, 37, 40, 48, 49, 64, 65, 70, 128] {
            let x: Vec<f32> = (0..m * hidden).map(|_| next()).collect();
            let x_d = d.upload_f32(&x).expect("up");
            let tile = Device::nvfp4_m_tile_for(m);
            let (q, qs) = d.quantise_nvfp4_dev(&x_d, m, hidden, gs, tile).expect("quantise");
            let out = d.nvfp4_gemm_dev(&q, &qs, packed, scales, m, inter, hidden, gs, *global_scale)
                .expect("gemm");
            let got = d.download_f32(&out).expect("dl");
            let padded = m.div_ceil(16) * 16;
            let mut zero_rows = Vec::new();
            for r in 0..m {
                let row = &got[r * inter..(r + 1) * inter];
                if row.iter().all(|v| *v == 0.0) {
                    zero_rows.push(r);
                }
            }
            eprintln!("  m {m:4} (padded {padded:4}, tile {tile:3}): zero rows {:?}", zero_rows);

            if m == 37 {
                let blk_mn = crate::swizzle::sf_atom_in_force(tile);
                let rows_padded = crate::swizzle::round_up_public(m, blk_mn);
                let qs_host = d.download(&qs).expect("dl");
                let compact = crate::swizzle::unswizzle(&qs_host, rows_padded, hidden / 16);
                let resw = d.swizzle_scales(&compact, rows_padded, hidden / 16, false)
                    .expect("re-swizzle");
                let o3 = d.nvfp4_gemm_dev(&q, &resw, packed, scales, m, inter, hidden, gs,
                                          *global_scale).expect("g3");
                let v3 = d.download_f32(&o3).expect("dl3");
                let zr3: Vec<usize> = (0..m).filter(|&r| {
                    v3[r * inter..(r + 1) * inter].iter().all(|v| *v == 0.0)
                }).collect();
                eprintln!("      scales RE-SWIZZLED through `swizzle_scales` -> zero rows {:?}", zr3);
                let rs = d.download(&resw).expect("dl");
                let nd = (0..qs_host.len().min(rs.len())).filter(|&i| qs_host[i] != rs[i]).count();
                eprintln!("      the two scale buffers differ in {nd} of {} bytes",
                          qs_host.len().min(rs.len()));
            }
            if m == 37 {
                for t2 in [64usize, 128] {
                    let (q2, qs2) = d.quantise_nvfp4_dev(&x_d, m, hidden, gs, t2).expect("q2");
                    let o2 = d.nvfp4_gemm_dev(&q2, &qs2, packed, scales, m, inter, hidden, gs,
                                              *global_scale).expect("g2");
                    let v2 = d.download_f32(&o2).expect("dl2");
                    let zr: Vec<usize> = (0..m).filter(|&r| {
                        v2[r * inter..(r + 1) * inter].iter().all(|v| *v == 0.0)
                    }).collect();
                    eprintln!("      quantised with tile {t2:3} (atom {:3}, chunk {:3}) -> zero rows {:?}",
                              crate::swizzle::sf_atom_in_force(t2),
                              crate::swizzle::sf_chunk_stride_in_force(t2), zr);
                }
            }
        }
    }

    #[test]
    #[ignore = "needs the checkpoint and the card"]
    fn the_attention_layer_stage_by_stage() {
        use crate::model::attention::AttentionWeights;
        use crate::model::norm::rms_norm as cpu_rms_norm;
        use crate::model::matmul as cpu_matmul;

        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let idx = (0..config.num_hidden_layers)
            .find(|&i| config.layer_kind(i) == crate::model::LayerKind::FullAttention)
            .expect("a full-attention layer");
        let DeviceLayer::Full(w) = &runner.backbone.layers[idx] else { panic!("not a Full layer") };
        let cpu_w = AttentionWeights::load(&checkpoint, &config, idx).expect("cpu weights");
        let (hidden, hd) = (config.hidden_size, config.head_dim);
        let (heads, kv_heads) = (config.num_attention_heads, config.num_key_value_heads);
        let (tokens, eps) = (37usize, config.rms_norm_eps);
        let rotary_dim = config.rotary_dim();
        let mut seed = 0x0DD_BA11u32;
        let x: Vec<f32> = (0..tokens * hidden)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0)
            })
            .collect();
        let x_dev = runner.device.upload_f32(&x).expect("upload");
        let d = &runner.device;

        let rms = |got: &[f32], want: &[f32]| -> f64 {
            let (mut n, mut dd) = (0f64, 0f64);
            for (g, wv) in got.iter().zip(want.iter()) {
                n += ((*g - *wv) as f64).powi(2);
                dd += (*wv as f64).powi(2);
            }
            (n / dd.max(1e-30)).sqrt()
        };
        let dl = |b: &Buffer| d.download_f32(b).expect("dl");
        let mut stage = |name: &str, gpu: &[f32], cpu: &[f32]| {
            eprintln!("  {name:14} rel-rms {:9.3e}   (n {} vs {})", rms(gpu, cpu), gpu.len(), cpu.len());
        };

        let normed_g = dl(&d.rms_norm_dev(&x_dev, &w.input_layernorm, tokens, hidden, eps).unwrap());
        let mut normed_c = vec![0.0f32; tokens * hidden];
        for t in 0..tokens {
            cpu_rms_norm(&x[t * hidden..(t + 1) * hidden], &cpu_w.input_layernorm, eps,
                         &mut normed_c[t * hidden..(t + 1) * hidden]);
        }
        stage("1 pre-norm", &normed_g, &normed_c);

        let normed_d = d.upload_f32(&normed_g).expect("up");
        let qg_g = dl(&run_projection_dev(d, &normed_d, tokens, &w.q_proj, None).unwrap());
        let mut qg_c = vec![0.0f32; tokens * heads * hd * 2];
        cpu_matmul(&cpu_w.q_proj, heads * hd * 2, hidden, &normed_c, tokens, &mut qg_c);
        stage("2 q_proj", &qg_g, &qg_c);

        let k_raw_g = dl(&run_projection_dev(d, &normed_d, tokens, &w.k_proj, None).unwrap());
        let mut k_c = vec![0.0f32; tokens * kv_heads * hd];
        cpu_matmul(&cpu_w.k_proj, kv_heads * hd, hidden, &normed_c, tokens, &mut k_c);
        stage("3 k_proj", &k_raw_g, &k_c);

        let v_g = dl(&run_projection_dev(d, &normed_d, tokens, &w.v_proj, None).unwrap());
        let mut v_c = vec![0.0f32; tokens * kv_heads * hd];
        cpu_matmul(&cpu_w.v_proj, kv_heads * hd, hidden, &normed_c, tokens, &mut v_c);
        stage("4 v_proj", &v_g, &v_c);
        let (q_g, gate_g) = d.split_q_gate_dev(&d.upload_f32(&qg_g).unwrap(), tokens, heads, hd)
            .expect("split");
        let mut q_c = vec![0.0f32; tokens * heads * hd];
        let mut gate_c = vec![0.0f32; tokens * heads * hd];
        for t in 0..tokens {
            for hh in 0..heads {
                let src = t * heads * hd * 2 + hh * hd * 2;
                q_c[(t * heads + hh) * hd..(t * heads + hh + 1) * hd]
                    .copy_from_slice(&qg_c[src..src + hd]);
                gate_c[(t * heads + hh) * hd..(t * heads + hh + 1) * hd]
                    .copy_from_slice(&qg_c[src + hd..src + 2 * hd]);
            }
        }
        stage("5 q/gate split", &dl(&q_g), &q_c);

        let q_norm_g = dl(&d.rms_norm_dev(&q_g, &w.q_norm, tokens * heads, hd, eps).unwrap());
        let mut q_norm_c = vec![0.0f32; tokens * heads * hd];
        for t in 0..tokens {
            for hh in 0..heads {
                let off = (t * heads + hh) * hd;
                cpu_rms_norm(&q_c[off..off + hd], &cpu_w.q_norm, eps,
                             &mut q_norm_c[off..off + hd]);
            }
        }
        stage("6 q_norm", &q_norm_g, &q_norm_c);

        let k_norm_g = dl(&d.rms_norm_dev(&d.upload_f32(&k_raw_g).unwrap(), &w.k_norm,
                                          tokens * kv_heads, hd, eps).unwrap());
        let mut k_norm_c = vec![0.0f32; tokens * kv_heads * hd];
        for t in 0..tokens {
            for hh in 0..kv_heads {
                let off = (t * kv_heads + hh) * hd;
                cpu_rms_norm(&k_c[off..off + hd], &cpu_w.k_norm, eps,
                             &mut k_norm_c[off..off + hd]);
            }
        }
        stage("7 k_norm", &k_norm_g, &k_norm_c);
        let (cos, sin) = runner.rope_tables_for(tokens).expect("rope tables");
        let q_norm_d = d.upload_f32(&q_norm_g).unwrap();
        let q_rope_g = dl(&d.rope_apply_dev(&q_norm_d, &cos, &sin, tokens,
                                            heads, hd, rotary_dim, None, None, None).unwrap());
        let (cos_c, sin_c) = crate::model::attention::rope_tables(
            &(0..tokens).collect::<Vec<_>>(), rotary_dim, config.rope_theta);
        let mut q_rope_c = q_norm_c.clone();
        for t in 0..tokens {
            let (c, sn) = (&cos_c[t * rotary_dim..(t + 1) * rotary_dim],
                           &sin_c[t * rotary_dim..(t + 1) * rotary_dim]);
            for hh in 0..heads {
                let off = (t * heads + hh) * hd;
                crate::model::attention::apply_rope(&mut q_rope_c[off..off + hd], c, sn);
            }
        }
        stage("8 rope(q)", &q_rope_g, &q_rope_c);

        let k_norm_d = d.upload_f32(&k_norm_g).unwrap();
        let k_rope_g = dl(&d.rope_apply_dev(&k_norm_d, &cos, &sin, tokens, kv_heads, hd, rotary_dim, None, None, None)
            .unwrap());
        let mut k_rope_c = k_norm_c.clone();
        for t in 0..tokens {
            let (c, sn) = (&cos_c[t * rotary_dim..(t + 1) * rotary_dim],
                           &sin_c[t * rotary_dim..(t + 1) * rotary_dim]);
            for hh in 0..kv_heads {
                let off = (t * kv_heads + hh) * hd;
                crate::model::attention::apply_rope(&mut k_rope_c[off..off + hd], c, sn);
            }
        }
        stage("8b rope(k)", &k_rope_g, &k_rope_c);

        let v_d = d.upload_f32(&v_g).unwrap();
        let attn_g = dl(&d.attention_dev(&q_rope_g_d(&d, &q_rope_g), &k_rope_g_d(&d, &k_rope_g),
                                         &v_d, tokens, heads, kv_heads, hd, crate::cuda::BatchMap::SINGLE).unwrap());
        let attn_c = crate::model::attention::causal_gqa_attention(
            &q_rope_c, &k_rope_c, &v_c, tokens, heads, kv_heads, hd);
        stage("9 attention", &attn_g, &attn_c);

        let gate_d = d.upload_f32(&dl(&gate_g)).unwrap();
        let gated_g = dl(&d.sigmoid_mul_dev(&d.upload_f32(&attn_g).unwrap(), &gate_d,
                                            tokens * heads * hd).unwrap());
        let gated_c: Vec<f32> = attn_c.iter().zip(gate_c.iter())
            .map(|(a, g)| a * crate::model::norm::sigmoid(*g)).collect();
        stage("10 output gate", &gated_g, &gated_c);

        let gated_d = d.upload_f32(&gated_g).unwrap();
        let y_g = dl(&run_projection_dev(d, &gated_d, tokens, &w.o_proj, Some(&x_dev)).unwrap());
        let mut mixed_c = vec![0.0f32; tokens * hidden];
        cpu_matmul(&cpu_w.o_proj, hidden, heads * hd, &gated_c, tokens, &mut mixed_c);
        let y_c: Vec<f32> = mixed_c.iter().zip(x.iter()).map(|(m, xv)| m + xv).collect();
        stage("11 o_proj+resid", &y_g, &y_c);

        let normed2_g = dl(&d.rms_norm_dev(&d.upload_f32(&y_g).unwrap(), &w.post_attention_layernorm,
                                           tokens, hidden, eps).unwrap());
        let mut normed2_c = vec![0.0f32; tokens * hidden];
        for t in 0..tokens {
            cpu_rms_norm(&y_c[t * hidden..(t + 1) * hidden], &cpu_w.post_attention_layernorm, eps,
                         &mut normed2_c[t * hidden..(t + 1) * hidden]);
        }
        stage("12 post-norm", &normed2_g, &normed2_c);

        let inter = config.intermediate_size;
        let normed2_d = d.upload_f32(&normed2_g).unwrap();
        let mg_g = dl(&run_projection_dev(d, &normed2_d, tokens, &w.mlp_gate, None).unwrap());
        let mut mg_c = vec![0.0f32; tokens * inter];
        cpu_matmul(&cpu_w.mlp_gate, inter, hidden, &normed2_c, tokens, &mut mg_c);
        stage("13 mlp_gate", &mg_g, &mg_c);

        if let DeviceProjection::Nvfp4 { cols: pc, input_global_scale, .. } = &w.mlp_gate {
            let gs = input_global_scale.expect("an NVFP4 projection carries its activation scale");
            let tile = Device::nvfp4_m_tile_for(tokens);
            let (packed, scales) = d.quantise_nvfp4_dev(&normed2_d, tokens, *pc, gs, tile)
                .expect("quantise");
            let pv = d.download(&packed).expect("dl");
            let sv = d.download(&scales).expect("dl");
            let row_len = pc / 2;
            let blk_mn = crate::swizzle::sf_atom_in_force(tile);
            let chunk = crate::swizzle::sf_chunk_stride_in_force(tile);
            for r in 0..tokens.min(6) {
                let nz = pv[r * row_len..(r + 1) * row_len].iter().filter(|b| **b != 0).count();

                let k1_count = (*pc / 16) / 4;
                let blk_sub = blk_mn / 32;
                let m_rest = r / blk_mn;
                let m_atom = (r % 32) * blk_sub + ((r / 32) % blk_sub);
                let si = (m_rest * k1_count + 0) * chunk + m_atom * 4;
                eprintln!("    packed row {r}: {nz} non-zero bytes of {row_len}; scale[{si}] = {}",
                          sv.get(si).copied().unwrap_or(255));
            }
            eprintln!("    (tile {tile}, blk_mn {blk_mn}, chunk {chunk}, scales {} bytes, packed {} bytes)",
                      sv.len(), pv.len());
        }

        {
            let m = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
            let mx = |v: &[f32]| v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            eprintln!("    gpu first 4 {:?}", &mg_g[..4.min(mg_g.len())]);
            eprintln!("    cpu first 4 {:?}", &mg_c[..4.min(mg_c.len())]);
            eprintln!("    gpu mean|.| {:.4e} max|.| {:.4e}   cpu mean|.| {:.4e} max|.| {:.4e}",
                      m(&mg_g), mx(&mg_g), m(&mg_c), mx(&mg_c));

            let n = mg_c.len().min(mg_g.len());
            let first_bad = (0..n).find(|&i| (mg_g[i] - mg_c[i]).abs() > 1e-2);
            eprintln!("    first index with |diff| > 1e-2: {:?}  (row {:?}, col {:?})",
                      first_bad, first_bad.map(|i| i / inter), first_bad.map(|i| i % inter));
            let mut bad_rows = Vec::new();
            for r in 0..tokens {
                let (mut num, mut den) = (0f64, 0f64);
                for c in 0..inter {
                    let (g, wv) = (mg_g[r * inter + c], mg_c[r * inter + c]);
                    num += ((g - wv) as f64).powi(2);
                    den += (wv as f64).powi(2);
                }
                let e = (num / den.max(1e-30)).sqrt();
                if e > 1e-2 { bad_rows.push((r, e)); }
            }
            eprintln!("    rows with rel-rms > 1e-2: {} of {tokens}: {:?}", bad_rows.len(),
                      &bad_rows[..bad_rows.len().min(6)]);

            let mut bad_cols = 0;
            for c in 0..inter {
                let (mut num, mut den) = (0f64, 0f64);
                for r in 0..tokens {
                    let (g, wv) = (mg_g[r * inter + c], mg_c[r * inter + c]);
                    num += ((g - wv) as f64).powi(2);
                    den += (wv as f64).powi(2);
                }
                if (num / den.max(1e-30)).sqrt() > 1e-2 { bad_cols += 1; }
            }
            eprintln!("    columns with rel-rms > 1e-2: {bad_cols} of {inter}");
        }
        let mu_g = dl(&run_projection_dev(d, &normed2_d, tokens, &w.mlp_up, None).unwrap());
        let mut mu_c = vec![0.0f32; tokens * inter];
        cpu_matmul(&cpu_w.mlp_up, inter, hidden, &normed2_c, tokens, &mut mu_c);
        stage("14 mlp_up", &mu_g, &mu_c);

        let act_g = dl(&d.swiglu_dev(&d.upload_f32(&mg_g).unwrap(), &d.upload_f32(&mu_g).unwrap(),
                                     tokens * inter).unwrap());
        let act_c: Vec<f32> = mg_c.iter().zip(mu_c.iter())
            .map(|(g, u)| (g * crate::model::norm::sigmoid(*g)) * u).collect();
        stage("15 swiglu", &act_g, &act_c);

        let act_d = d.upload_f32(&act_g).unwrap();
        let out_g = dl(&run_projection_dev(d, &act_d, tokens, &w.mlp_down, Some(&d.upload_f32(&y_g).unwrap())).unwrap());
        let mut md_c = vec![0.0f32; tokens * hidden];
        cpu_matmul(&cpu_w.mlp_down, hidden, inter, &act_c, tokens, &mut md_c);
        let out_c: Vec<f32> = md_c.iter().zip(y_c.iter()).map(|(m, yv)| m + yv).collect();
        stage("16 mlp_down+res", &out_g, &out_c);
    }

    fn q_rope_g_d(d: &Device, v: &[f32]) -> Buffer { d.upload_f32(v).expect("up") }
    fn k_rope_g_d(d: &Device, v: &[f32]) -> Buffer { d.upload_f32(v).expect("up") }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn one_attention_layer_on_the_gpu_matches_the_cpu_oracle() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let idx = (0..config.num_hidden_layers)
            .find(|&i| config.layer_kind(i) == crate::model::LayerKind::FullAttention)
            .expect("the model has at least one full-attention layer");
        eprintln!("  full-attention layer index {idx}");
        let (tokens, hidden) = (37usize, config.hidden_size);
        let mut seed = 0x0DD_BA11u32;

        let scale: f32 = std::env::var("CLEF_ATT_INPUT_SCALE").ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let x: Vec<f32> = (0..tokens * hidden)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * scale
            })
            .collect();

        let DeviceLayer::Full(dw) = &runner.backbone.layers[idx] else {
            panic!("layer {idx} must be a full-attention layer");
        };
        let x_dev = runner.device.upload_f32(&x).expect("upload");
        let got = runner.device
            .download_f32(&runner.attention_layer_dev(&x_dev, tokens, dw, crate::cuda::BatchMap::SINGLE).expect("gpu attention layer"))
            .expect("download");
        assert!(got.iter().all(|v| v.is_finite()), "the GPU layer produced a non-finite value");

        let w = crate::model::attention::AttentionWeights::load(&checkpoint, &config, idx)
            .expect("cpu attention weights");
        let positions: Vec<usize> = (0..tokens).collect();
        let want = crate::model::attention::attention_layer(&x, &w, &config, &positions);
        let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
        for (a, b) in want.iter().zip(got.iter()) {
            mx = mx.max((a - b).abs());
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        let rms = (num / den.max(1e-30)).sqrt();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!("  tokens {tokens}: max|gpu - cpu| = {mx:.3e}, rel-rms = {rms:.3e}, |ref| max = {scale:.4}");

        let rms_vs = |cand: &[f32]| -> f64 {
            let (mut n, mut d) = (0f64, 0f64);
            for ((g, c), _) in got.iter().zip(cand.iter()).zip(want.iter()) {
                n += ((*g - *c) as f64).powi(2);
                d += (*c as f64).powi(2);
            }
            (n / d.max(1e-30)).sqrt()
        };
        let minus_x: Vec<f32> = want.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let plus_x: Vec<f32> = want.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
        eprintln!("    vs oracle {:.3e} | vs oracle-x {:.3e} | vs oracle+x {:.3e}   (|x| max {:.3e})",
                  rms, rms_vs(&minus_x), rms_vs(&plus_x),
                  x.iter().fold(0.0f32, |m, v| m.max(v.abs())));

        assert!(rms < 1.5e-1, "the attention layer diverged from the CPU oracle: rel-rms {rms:.3e}");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn every_projection_kind_applies_the_residual() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (hidden, inter) = (config.hidden_size, config.intermediate_size);
        let mut seed = 0xA11CEu32;
        let mut gen = |n: usize, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * amp
                })
                .collect()
        };

        let mut checked = 0;
        for tokens in [7usize, 4104] {
            let act = gen(tokens * inter, 0.05);
            let resid = gen(tokens * hidden, 0.002);
            let act_d = runner.device.upload_f32(&act).expect("upload act");
            let resid_d = runner.device.upload_f32(&resid).expect("upload resid");
            for (idx, layer) in runner.backbone.layers.iter().enumerate() {
                let DeviceLayer::Linear(w) = layer else { continue };
                let kind = match &w.mlp_down {
                    DeviceProjection::Nvfp4 { .. } => "nvfp4",
                    DeviceProjection::Fp8 { .. } => "fp8",
                    DeviceProjection::Bf16 { .. } => "bf16",
                };
                if idx != 0 && idx != 6 {
                    continue;
                }
                let without = run_projection_dev(&runner.device, &act_d, tokens, &w.mlp_down, None)
                    .expect("projection without residual");
                let with = run_projection_dev(&runner.device, &act_d, tokens, &w.mlp_down,
                                              Some(&resid_d))
                    .expect("projection with residual");
                let a = runner.device.download_rows_f32(&without, tokens, hidden).expect("dl");
                let b = runner.device.download_rows_f32(&with, tokens, hidden).expect("dl");

                let worst = a
                    .iter()
                    .zip(b.iter())
                    .zip(resid.iter())
                    .fold(0.0f32, |m, ((x, y), r)| m.max(((y - x) - r).abs()));
                let scale = resid.iter().fold(0.0f32, |m, r| m.max(r.abs()));
                eprintln!(
                    "  layer {idx} mlp_down ({kind}, {tokens} rows, epilogue={}): worst \
                     |(with - without) - residual| = {worst:.3e} against a residual of max |r| = \
                     {scale:.3e}",
                    kind == "fp8" && tokens >= RESID_FUSED_FROM
                );
                assert!(
                    worst < 1e-6,
                    "layer {idx}'s {kind} projection did not apply the residual at {tokens} rows \
                     (epilogue={}): worst difference {worst:.3e} against a residual of max |r| \
                     {scale:.3e}",
                    kind == "fp8" && tokens >= RESID_FUSED_FROM
                );
                checked += 1;
            }
        }
        assert!(checked >= 4, "the gate must cover both kinds at both residual paths, got {checked}");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn the_gated_gemm_equals_the_pair() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (hidden, inter) = (config.hidden_size, config.intermediate_size);

        let mut checked = 0;
        for (idx, layer) in runner.backbone.layers.iter().enumerate() {
            let DeviceLayer::Linear(w) = layer else { continue };
            let (DeviceProjection::Nvfp4 { packed: bg_p, scales: bg_s, global_scale: bg_g,
                                           input_global_scale: bg_igs, .. },
             DeviceProjection::Nvfp4 { packed: bu_p, scales: bu_s, global_scale: bu_g,
                                       input_global_scale: bu_igs, .. },
             DeviceProjection::Nvfp4 { input_global_scale: Some(dn_igs), .. }) =
                (&w.mlp_gate, &w.mlp_up, &w.mlp_down) else { continue };
            if idx != 0 { continue; }
            let (Some(bg_igs), Some(bu_igs)) = (bg_igs, bu_igs) else { continue };
            assert_eq!(bg_igs, bu_igs, "the pair must share an input scale for the shared activation");

            for rows in [7usize, 300] {
                let mut seed = 0xC0FFEEu32 ^ (rows as u32);
                let mut next = || {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.9
                };
                let x: Vec<f32> = (0..rows * hidden).map(|_| next()).collect();
                let xd = runner.device.upload_f32(&x).expect("upload x");
                let (q, qs) = runner.device
                    .quantise_nvfp4_dev(&xd, rows, hidden, *bg_igs,
                                        Device::nvfp4_m_tile_for(rows))
                    .expect("quantise the activation");

                let g = run_nvfp4_projection_from_quantised(&runner.device, &q, &qs, rows,
                                                            &w.mlp_gate, None)
                    .expect("gate is nvfp4").expect("gate");
                let u = run_nvfp4_projection_from_quantised(&runner.device, &q, &qs, rows,
                                                            &w.mlp_up, None)
                    .expect("up is nvfp4").expect("up");
                let (pc, sc) = runner.device
                    .swiglu_quantise_nvfp4_dev(&g, &u, rows, inter, *dn_igs,
                                               Device::nvfp4_m_tile_for(rows))
                    .expect("swiglu_quantise")
                    .expect("cols within the kernel's block");
                let (pf, sf) = runner.device
                    .nvfp4_gemm_gated_dev(&q, &qs, bg_p, bg_s, bu_p, bu_s,
                                          rows, inter, hidden, *bg_igs, *bg_g, *bu_g, *dn_igs)
                    .expect("gated gemm");
                let (pf2, sf2) = runner.device
                    .nvfp4_gemm_gated_dev(&q, &qs, bg_p, bg_s, bu_p, bu_s,
                                          rows, inter, hidden, *bg_igs, *bg_g, *bu_g, *dn_igs)
                    .expect("gated gemm, twice");
                for (name, a, b) in [("packed", &pf, &pf2), ("scales", &sf, &sf2)] {
                    let x = runner.device.download(a).expect("dl");
                    let y = runner.device.download(b).expect("dl");
                    match x.iter().zip(y.iter()).position(|(u, w)| u != w) {
                        None => eprintln!("  [{rows} rows] {name:>6} SAME CALL TWICE 0.000e0"),
                        Some(i) => panic!(
                            "{rows} rows: {name} DIFFERS BETWEEN TWO IDENTICAL CALLS at byte {i}: \
                             {:#04x} against {:#04x} — that is a race or an uninitialised read, not an \
                             arithmetic difference", x[i], y[i]
                        ),
                    }
                }
                for (name, fused, chain) in [("packed", &pf, &pc), ("scales", &sf, &sc)] {
                    let a = runner.device.download(fused).expect("dl fused");
                    let b = runner.device.download(chain).expect("dl chain");
                    assert_eq!(a.len(), b.len(), "{rows} rows: {name} sized differently");
                    match a.iter().zip(b.iter()).position(|(x, y)| x != y) {
                        None => eprintln!("  [{rows} rows] {name:>6} worst 0 over {} bytes", a.len()),
                        Some(i) => panic!(
                            "{rows} rows: {name} differs at byte {i} of {}: gated {:#04x} against \
                             chain {:#04x}", a.len(), a[i], b[i]
                        ),
                    }
                }
                checked += 1;
            }
        }
        assert!(checked >= 2, "layer 0's MLP is not NVFP4 in this checkpoint; checked {checked}");
    }

    #[test]
    #[ignore = "loads 10 GB onto the device; run on the card"]
    fn one_layer_on_the_gpu_matches_the_cpu_oracle() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config.json");
        let runner = match GpuRunner::load(&checkpoint, &config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        eprintln!("  backbone loaded onto the device");
        let (tokens, hidden) = (7usize, config.hidden_size);
        let mut seed = 0x5EED_1234u32;
        let x: Vec<f32> = (0..tokens * hidden)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.002
            })
            .collect();

        let DeviceLayer::Linear(dw) = &runner.backbone.layers[0] else {
            panic!("layer 0 must be a DeltaNet layer");
        };
        let mut traced: Option<(Vec<f32>, Vec<f32>)> = None;
        let got = runner
            .delta_net_layer_traced(&x, tokens, dw, Some(&mut traced))
            .expect("gpu layer 0");
        assert!(got.iter().all(|v| v.is_finite()), "the GPU layer produced a non-finite value");

        if let Some((act, down)) = &traced {
            let proj = checkpoint
                .projection(&format!("{}.mlp.down_proj",
                    config.layer_prefix(checkpoint.text_prefix().unwrap(), 0)))
                .expect("down_proj");
            let wf = proj.dequantise();
            let mut cpu = vec![0.0f32; tokens * config.hidden_size];
            crate::model::matmul(&wf, config.hidden_size, config.intermediate_size, act, tokens,
                                 &mut cpu);
            let cm = cpu.iter().map(|v| v.abs()).sum::<f32>() / cpu.len() as f32;
            let gm = down.iter().map(|v| v.abs()).sum::<f32>() / down.len() as f32;
            let am = act.iter().map(|v| v.abs()).sum::<f32>() / act.len() as f32;
            eprintln!("    mlp_down alone: act mean {am:.4e}; CPU(f32 weights) {cm:.4e} vs \
                       GPU(packed) {gm:.4e}  ratio {:.4}", cm / gm.max(1e-30));
        }

        let cw = DeltaNetWeights::load(&checkpoint, &config, 0).expect("cpu weights");
        let mut want = vec![0.0f32; tokens * hidden];
        deltanet_layer(&config, &cw, &x, tokens, &mut want).expect("cpu layer 0");

        if std::env::var("CLEF_GPU_TRACE").is_ok() {
            let st = |name: &str, v: &[f32]| {
                let m = v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
                let mx = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                eprintln!("      cpu {name:16} mean={m:.4e} max={mx:.4e}");
            };
            let mut normed = vec![0.0f32; tokens * hidden];
            for t2 in 0..tokens {
                crate::model::norm::rms_norm(&x[t2 * hidden..(t2 + 1) * hidden], &cw.input_layernorm,
                                             config.rms_norm_eps, &mut normed[t2 * hidden..(t2 + 1) * hidden]);
            }
            st("input_layernorm", &normed);
            let conv_dim = config.conv_dim();
            let mut mixed = vec![0.0f32; tokens * conv_dim];
            crate::model::matmul(&cw.in_proj_qkv, conv_dim, hidden, &normed, tokens, &mut mixed);
            st("in_proj_qkv", &mixed);
            let vdim = config.linear_num_value_heads * config.linear_value_head_dim;
            let mut z = vec![0.0f32; tokens * vdim];
            crate::model::matmul(&cw.in_proj_z, vdim, hidden, &normed, tokens, &mut z);
            st("in_proj_z", &z);

            let mut conv = vec![0.0f32; tokens * conv_dim];
            crate::model::conv::causal_conv1d_silu(&mixed, &cw.conv_weight,
                config.linear_conv_kernel_dim, conv_dim, tokens, &mut conv);
            st("conv_silu", &conv);
            let kd = config.linear_key_head_dim;
            let vd = config.linear_value_head_dim;
            let kh = config.linear_num_key_heads;
            let vh = config.linear_num_value_heads;
            let mut q = vec![0.0f32; tokens * kh * kd];
            let mut k = vec![0.0f32; tokens * kh * kd];
            let mut v = vec![0.0f32; tokens * vh * vd];
            for t2 in 0..tokens {
                let row = &conv[t2 * conv_dim..(t2 + 1) * conv_dim];
                q[t2 * kh * kd..(t2 + 1) * kh * kd].copy_from_slice(&row[..kh * kd]);
                k[t2 * kh * kd..(t2 + 1) * kh * kd].copy_from_slice(&row[kh * kd..2 * kh * kd]);
                v[t2 * vh * vd..(t2 + 1) * vh * vd].copy_from_slice(&row[2 * kh * kd..]);
            }
            let ratio = config.value_heads_per_key_head();
            let rep = |src: &[f32], dim: usize| -> Vec<f32> {
                let mut o = vec![0.0f32; tokens * kh * ratio * dim];
                for t2 in 0..tokens {
                    for h in 0..kh {
                        let s = &src[(t2 * kh + h) * dim..(t2 * kh + h + 1) * dim];
                        for r in 0..ratio {
                            let d = (t2 * kh * ratio + h * ratio + r) * dim;
                            o[d..d + dim].copy_from_slice(s);
                        }
                    }
                }
                o
            };
            let (mut q2, mut k2) = (rep(&q, kd), rep(&k, kd));
            let mut a = vec![0.0f32; tokens * vh];
            let mut b = vec![0.0f32; tokens * vh];
            crate::model::matmul(&cw.in_proj_a, vh, hidden, &normed, tokens, &mut a);
            crate::model::matmul(&cw.in_proj_b, vh, hidden, &normed, tokens, &mut b);
            let mut decay = vec![0.0f32; tokens * vh];
            let mut beta = vec![0.0f32; tokens * vh];
            for t2 in 0..tokens {
                for h in 0..vh {
                    beta[t2 * vh + h] = 1.0 / (1.0 + (-b[t2 * vh + h]).exp());
                    let xv = a[t2 * vh + h] + cw.dt_bias[h];
                    let sp = if xv > 20.0 { xv } else { xv.exp().ln_1p() };
                    decay[t2 * vh + h] = -cw.a_log[h].exp() * sp;
                }
            }
            let mut o = vec![0.0f32; tokens * vh * vd];
            crate::model::deltanet::gated_delta_rule(&mut q2, &mut k2, &v, &decay, &beta, tokens, vh,
                                                     kd, vd, &mut o);
            st("gdn_scan", &o);
            let mut gated = vec![0.0f32; tokens * vh * vd];
            for i in 0..tokens * vh {
                crate::model::norm::rms_norm_gated(&o[i * vd..(i + 1) * vd], &z[i * vd..(i + 1) * vd],
                    &cw.norm_weight, config.rms_norm_eps, &mut gated[i * vd..(i + 1) * vd]);
            }
            st("gated_norm", &gated);
            let mut mo = vec![0.0f32; tokens * hidden];
            crate::model::matmul(&cw.out_proj, hidden, vh * vd, &gated, tokens, &mut mo);
            st("out_proj", &mo);
            let mut y = vec![0.0f32; tokens * hidden];
            for i in 0..tokens * hidden {
                y[i] = x[i] + mo[i];
            }
            st("y (after out_proj)", &y);
            let mut n2 = vec![0.0f32; tokens * hidden];
            for t2 in 0..tokens {
                crate::model::norm::rms_norm(&y[t2 * hidden..(t2 + 1) * hidden],
                    &cw.post_attention_layernorm, config.rms_norm_eps,
                    &mut n2[t2 * hidden..(t2 + 1) * hidden]);
            }
            st("post_attn_layernorm", &n2);
            let inter = config.intermediate_size;
            let mut g = vec![0.0f32; tokens * inter];
            let mut u = vec![0.0f32; tokens * inter];
            crate::model::matmul(&cw.mlp_gate, inter, hidden, &n2, tokens, &mut g);
            crate::model::matmul(&cw.mlp_up, inter, hidden, &n2, tokens, &mut u);
            st("mlp_gate", &g);
            st("mlp_up", &u);
            let act: Vec<f32> = g.iter().zip(u.iter())
                .map(|(a, b)| (a * crate::model::norm::sigmoid(*a)) * b).collect();
            st("swiglu", &act);
            let mut down = vec![0.0f32; tokens * hidden];
            crate::model::matmul(&cw.mlp_down, hidden, inter, &act, tokens, &mut down);
            st("mlp_down", &down);
        }

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (a, b) in got.iter().zip(want.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / got.len() as f64) as f32;
        let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
        eprintln!(
            "  layer 0 GPU vs CPU: max |diff| = {max_abs:.4e}, mean = {mean_abs:.4e}, \
             cpu mean |x| = {scale:.4e}, relative = {:.4e}",
            mean_abs / scale.max(1e-12)
        );

        assert!(
            mean_abs / scale.max(1e-12) < 1.0,
            "the GPU orchestration is wired wrong: {:.3e} relative against the CPU oracle",
            mean_abs / scale.max(1e-12)
        );
    }
}

#[cfg(test)]
mod rung_cut_probe {
    #[test]
    fn print_the_cut() {
        for t in [778usize, 1024, 4096, 6144, 8110, 7846, 8186, 16386, 32686, 65546, 130726] {
            let mut base = 0;
            let mut rows = Vec::new();
            while base < t {
                let rem = t - base;
                let r = crate::gpu::next_chunk_rows(rem);
                rows.push(r);
                base += r;
            }
            eprintln!("  {t:>7} -> {rows:?}  total {} (pad {})", rows.iter().sum::<usize>(),
                      rows.iter().sum::<usize>() - t);
        }
    }
}
