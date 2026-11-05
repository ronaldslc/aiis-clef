use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr,
    LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;


macro_rules! env_once {
    ($name:expr) => {{
        static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        CACHE.get_or_init(|| std::env::var($name).ok()).as_deref()
    }};
}

const SWIZZLE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swizzle.cubin"));

pub const SF_CONFLICT_FREE: bool = true;

const GEMM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.cubin"));

const GEMM_M64_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_m64.cubin"));

const GEMM_GATED_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_gated.cubin"));

const GEMM_GATED_NOEPI_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_gated_noepi.cubin"));

const GEMM_M192_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_m192.cubin"));

const GEMM_M192W8_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_m192w8.cubin"));

const GEMM_STAGES3_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_stages3.cubin"));

const GEMM_ATOM32_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_atom32.cubin"));





const FP8GEMM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm.cubin"));

const FP8GEMM_GATED_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_gated.cubin"));

const FP8GEMM_PLAIN_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_plain.cubin"));

const FP8GEMM_ALIASED_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_aliased.cubin"));

const FP8GEMM_NOPREFETCH_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_noprefetch.cubin"));

const FP8GEMM_REG64_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_reg64.cubin"));

const FP8GEMM_BIG_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_big.cubin"));

const FP8GEMM_EPI1_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_epi1.cubin"));
const FP8GEMM_EPI3_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_epi3.cubin"));
const FP8GEMM_EPIN16_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_epin16.cubin"));
const FP8GEMM_PROD64_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_prod64.cubin"));
const FP8GEMM_PROD24_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_prod24.cubin"));
const FP8GEMM_W2X4_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_w2x4.cubin"));
const FP8GEMM_T128_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_t128x128.cubin"));
const FP8GEMM_N64_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_n64.cubin"));

const FP8GEMM_STAGES2_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_stages2.cubin"));

const FP8GEMM_K64_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_k64.cubin"));
const FP8GEMM_REG160_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_reg160.cubin"));
const FP8GEMM_REG192_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_reg192.cubin"));

const FP8GEMM_ALIASED_PLAIN_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp8gemm_aliased_plain.cubin"));

const FP8_EPI_N: u32 = 32;

fn fp8_box(facts: &crate::model::CubinFacts)
    -> (u32, cudarc::driver::sys::CUtensorMapSwizzle_enum)
{
    use cudarc::driver::sys::CUtensorMapSwizzle_enum as S;
    let ktile = (facts.sub_chunks * 32) as u32;
    let sw = if ktile >= 128 { S::CU_TENSOR_MAP_SWIZZLE_128B } else { S::CU_TENSOR_MAP_SWIZZLE_64B };
    (ktile, sw)
}

pub fn fp8_epi_swizzle(n: usize) -> cudarc::driver::sys::CUtensorMapSwizzle_enum {
    use cudarc::driver::sys::CUtensorMapSwizzle_enum as S;

    if env_once!("CLEF_EPI_SWZ") != Some("0")
        && FP8_EPI_N * 4 == 128 && n >= FP8_EPI_N as usize && n % 32 == 0 {
        S::CU_TENSOR_MAP_SWIZZLE_128B
    } else {
        S::CU_TENSOR_MAP_SWIZZLE_NONE
    }
}

pub fn fp8_padded_m(m: usize, m_tile: usize) -> usize {
    let t = m_tile.max(16);
    m.div_ceil(t) * t
}

fn fp8_log_swizzle(padded_m: usize) -> i32 {
    let _ = padded_m;
    match env_once!("CLEF_SWIZZLE").and_then(|v| v.parse::<i32>().ok()) {
        Some(v) => v,
        None => 0,
    }
}

struct Fp8Variant {
    cubin: &'static [u8],
    name: &'static str,
    facts: crate::model::CubinFacts,
    params: crate::model::Params,
    side: crate::model::Fp8Side,
    ctas_per_sm: usize,
    why: &'static str,
}

fn fp8_variant(m: usize, n: usize, k: usize) -> Fp8Variant {
    fp8_variant_with(
        m, n, k,
        env_once!("CLEF_FP8_ALIASED") == Some("1"),
        env_once!("CLEF_FP8_ATOM") == Some("plain"),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp8Policy { T64x128, T128, T256x64, GemmTuned }

pub fn fp8_policy() -> Option<Fp8Policy> {
    match env_once!("CLEF_FP8_POLICY") {
        Some("t64x128") => Some(Fp8Policy::T64x128),
        Some("t128") => Some(Fp8Policy::T128),
        Some("t256x64") => Some(Fp8Policy::T256x64),
        Some("gemm_tuned") => Some(Fp8Policy::GemmTuned),
        _ => None,
    }
}

pub fn fp8_gemm_tuned(m: usize, n: usize, k: usize) -> Fp8Level {
    if n <= 1024 {
        if m >= 768 { Fp8Level::W2X4 } else { Fp8Level::N64 }
    } else if k >= 8192 {
        if m >= 512 { Fp8Level::S2 } else if m >= 192 { Fp8Level::W2X4 } else { Fp8Level::N64 }
    } else if n >= 8192 && m <= 32768 {
        Fp8Level::T256x64
    } else {
        Fp8Level::T128
    }
}

fn fp8_variant_for_level(l: Fp8Level) -> Fp8Variant {
    use crate::model::{CubinFacts, Fp8Side, Machine, Params};
    let (cubin, name, facts, params, side) = match l {
        Fp8Level::T64x128 => (FP8GEMM_CUBIN, "fp8gemm", CubinFacts::FP8, Params::FP8,
                               Fp8Side::T64x128),
        Fp8Level::T128 => (FP8GEMM_T128_CUBIN, "fp8gemm_t128x128", CubinFacts::FP8_T128,
                           Params::FP8_T128, Fp8Side::T128),
        Fp8Level::T256x64 => (FP8GEMM_BIG_CUBIN, "fp8gemm_big", CubinFacts::FP8_BIG, Params::FP8_BIG,
                          Fp8Side::T256x64),
        Fp8Level::N64 => (FP8GEMM_N64_CUBIN, "fp8gemm_n64", CubinFacts::FP8_N64, Params::FP8_N64,
                          Fp8Side::N64),
        Fp8Level::W2X4 => (FP8GEMM_W2X4_CUBIN, "fp8gemm_w2x4", CubinFacts::FP8_W2X4,
                           Params::FP8_W2X4, Fp8Side::W2X4),
        Fp8Level::S2 => (FP8GEMM_STAGES2_CUBIN, "fp8gemm_stages2", CubinFacts::FP8_STAGES2,
                         Params::FP8_STAGES2, Fp8Side::T64x128),
    };
    let (ctas_per_sm, why) = Machine::RTX_5090.ctas_per_sm(&params);
    Fp8Variant { cubin, name, facts, params, side, ctas_per_sm, why }
}

pub fn fp8_dispatch(m: usize, _n: usize, _k: usize) -> Fp8Level {
    if m <= 256 {
        Fp8Level::T64x128
    } else {
        Fp8Level::T128
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp8Level { T64x128, T256x64, T128, N64, W2X4, S2 }

fn fp8_variant_with(m: usize, n: usize, k: usize, aliased: bool, plain: bool) -> Fp8Variant {
    use crate::model::{CubinFacts, Fp8Side, Machine, Params};

    let prefetch_off = env_once!("CLEF_FP8_PREFETCH") == Some("0");
    let k64 = env_once!("CLEF_FP8_K64") == Some("1");
    let stages2 = env_once!("CLEF_FP8_STAGES2") == Some("1");

    let level: Option<(&'static [u8], &'static str, crate::model::CubinFacts,
                       crate::model::Params, crate::model::Fp8Side)> =
        match env_once!("CLEF_FP8_LEVEL").unwrap_or_default()
    {
        "t64x128" => Some((FP8GEMM_CUBIN, "fp8gemm", CubinFacts::FP8, Params::FP8, Fp8Side::T64x128)),
        "epi1" => Some((FP8GEMM_EPI1_CUBIN, "fp8gemm_epi1", CubinFacts::FP8, Params::FP8_EPI1,
                        Fp8Side::T64x128)),
        "epi3" => Some((FP8GEMM_EPI3_CUBIN, "fp8gemm_epi3", CubinFacts::FP8, Params::FP8_EPI3,
                        Fp8Side::T64x128)),
        "epin16" => Some((FP8GEMM_EPIN16_CUBIN, "fp8gemm_epin16", CubinFacts::FP8, Params::FP8,
                          Fp8Side::T64x128)),
        "prod64" => Some((FP8GEMM_PROD64_CUBIN, "fp8gemm_prod64", CubinFacts::FP8, Params::FP8,
                          Fp8Side::T64x128)),
        "prod24" => Some((FP8GEMM_PROD24_CUBIN, "fp8gemm_prod24", CubinFacts::FP8, Params::FP8,
                          Fp8Side::T64x128)),
        "w2x4" => Some((FP8GEMM_W2X4_CUBIN, "fp8gemm_w2x4", CubinFacts::FP8_W2X4, Params::FP8_W2X4,
                        Fp8Side::W2X4)),
        "t128" => Some((FP8GEMM_T128_CUBIN, "fp8gemm_t128x128", CubinFacts::FP8_T128, Params::FP8_T128,
                        Fp8Side::T128)),
        "n64" => Some((FP8GEMM_N64_CUBIN, "fp8gemm_n64", CubinFacts::FP8_N64, Params::FP8_N64,
                       Fp8Side::N64)),
        _ => None,
    };

    let lvl = match fp8_policy() {
        Some(Fp8Policy::T64x128) => Fp8Level::T64x128,
        Some(Fp8Policy::T128) => Fp8Level::T128,
        Some(Fp8Policy::T256x64) => Fp8Level::T256x64,
        Some(Fp8Policy::GemmTuned) => fp8_gemm_tuned(m, n, k),
        None => fp8_dispatch(m, n, k),
    };
    if level.is_none() {
        return fp8_variant_for_level(lvl);
    }
    let big = env_once!("CLEF_FP8_BIG") == Some("1");

    let regmax: usize = env_once!("CLEF_FP8_REGMAX").and_then(|v| v.parse().ok()).unwrap_or(128);

    let reg_cubin = match regmax {
        64 => Some(FP8GEMM_REG64_CUBIN),
        160 => Some(FP8GEMM_REG160_CUBIN),
        192 => Some(FP8GEMM_REG192_CUBIN),
        _ => None,
    };
    if let Some((c, n, f, p, sd)) = level {
        let (ctas_per_sm, why) = Machine::RTX_5090.ctas_per_sm(&p);
        return Fp8Variant { cubin: c, name: n, facts: f, params: p, side: sd, ctas_per_sm, why };
    }
    let (cubin, name, facts, params, side) = match (aliased, plain, prefetch_off) {
        _ if k64 => (FP8GEMM_K64_CUBIN, "fp8gemm_k64", CubinFacts::FP8_K64, Params::FP8_K64,
                     Fp8Side::T64x128),
        _ if stages2 => (FP8GEMM_STAGES2_CUBIN, "fp8gemm_stages2", CubinFacts::FP8_STAGES2,
                         Params::FP8_STAGES2, Fp8Side::T64x128),
        _ if big => (FP8GEMM_BIG_CUBIN, "fp8gemm_big", CubinFacts::FP8_BIG, Params::FP8_BIG,
                     Fp8Side::T256x64),
        (false, false, false) => (
            reg_cubin.unwrap_or(FP8GEMM_CUBIN),
            if regmax == 128 { "fp8gemm" } else { "fp8gemm_regN" },
            CubinFacts::FP8, Params::FP8, Fp8Side::T64x128,
        ),
        (false, false, true) => (FP8GEMM_NOPREFETCH_CUBIN, "fp8gemm_noprefetch", CubinFacts::FP8,
                                    Params::FP8, Fp8Side::T64x128),
        (false, true, _) => (FP8GEMM_PLAIN_CUBIN, "fp8gemm_plain", CubinFacts::FP8, Params::FP8,
                             Fp8Side::T64x128),
        (true, false, _) => (FP8GEMM_ALIASED_CUBIN, "fp8gemm_aliased", CubinFacts::FP8_ALIASED,
                             Params::FP8_ALIASED, Fp8Side::ALIASED),
        (true, true, _) => (FP8GEMM_ALIASED_PLAIN_CUBIN, "fp8gemm_aliased_plain", CubinFacts::FP8_ALIASED,
                            Params::FP8_ALIASED, Fp8Side::ALIASED),
    };
    let (ctas_per_sm, why) = Machine::RTX_5090.ctas_per_sm(&params);
    Fp8Variant { cubin, name, facts, params, side, ctas_per_sm, why }
}

const GDN_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn.cubin"));

const GDN_CHUNK_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_chunk.cubin"));

pub const CHUNK_LEN: usize = 64;

pub fn gdn_state_bytes(heads: usize, k_dim: usize, v_dim: usize) -> usize {
    assert_eq!(k_dim, CHUNK_KD, "the state carry is sized for linear_key_head_dim = 128");
    assert_eq!(v_dim, CHUNK_VD, "the state carry is sized for linear_value_head_dim = 128");
    heads * CHUNK_VD * CHUNK_KD * 4
}

pub const CHUNK_GRANULE: usize = CHUNK_LEN;
const CHUNK_KD: usize = 128;
const CHUNK_VD: usize = 128;

const CHUNK_V_SPLIT: usize = 4;

pub const fn gdn_state_threads() -> usize {
    let vs = 128 / CHUNK_V_SPLIT;
    let warps_n = if vs >= 32 { 4 } else { 2 };
    (4 * warps_n) * 32
}

fn gdn_segs() -> usize {
    env_once!("CLEF_GDN_SEGS")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1)
}

const ELEMENTWISE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/elementwise.cubin"));

const ATTENTION_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.cubin"));

const ATTENTION_R8_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention_r8.cubin"));

const ATTENTION_R4_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention_r4.cubin"));

const ATTENTION_R12_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention_r12.cubin"));

const ATTENTION_DIRECT_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_direct.cubin"));

const ATTENTION_SMEM_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_smem.cubin"));

const ATTENTION_WIDE_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_wide.cubin"));

const ATTENTION_MMA_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma.cubin"));

const ATTENTION_MMA_MT2_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_mt2.cubin"));

const ATTENTION_MMA_BK64_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_bk64.cubin"));

const ATTENTION_MMA_BK64W_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_bk64w.cubin"));

const ATTENTION_MMA_HD128_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_hd128.cubin"));

const ATTENTION_MMA_QREG96_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_qreg96.cubin"));

const ATTENTION_MMA_PARTIAL_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_partial.cubin"));
const ATTENTION_MMA_PARTIAL256_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_partial256.cubin"));
const ATTENTION_MMA_WS_NOSKIP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_ws_noskip.cubin"));
const ATTENTION_MMA_WS_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_ws.cubin"));
const ATTENTION_MMA_WS_R240_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_ws_r240.cubin"));
const ATTENTION_MMA_WS_SPLIT_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_ws_split.cubin"));
const ATTENTION_MMA_WS_BK48_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_ws_bk48.cubin"));
const ATTENTION_MMA_QREG_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_qreg.cubin"));

const ATTENTION_MMA_GQA_BK32_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_gqa_bk32.cubin"));

const ATTENTION_MMA_BK32_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_bk32.cubin"));

const ATTENTION_MMA_GQA_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_gqa.cubin"));

const ATTENTION_MMA_MSPLIT_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_msplit.cubin"));

const ATTENTION_MMA_BQ160_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attention_mma_bq160.cubin"));

fn att_qreg() -> bool { env_once!("CLEF_ATT_QREG") != Some("0") }
fn att_qreg96() -> bool { env_once!("CLEF_ATT_BK") == Some("96") }

fn att_bk32() -> bool { env_once!("CLEF_ATT_BK") == Some("32") }

fn att_bk48() -> bool { env_once!("CLEF_ATT_BK") == Some("48") }

pub const ATTN_BK: usize = 32;

pub fn split_partials_bytes(splits: usize, heads: usize, head_dim: usize) -> usize {
    let largest_pass = crate::gpu::kv_slots().max(1) * crate::gpu::max_chunk_size();
    splits * largest_pass * heads * (head_dim + 2) * 4
}

pub fn stream_priority() -> i32 {
    STREAM_PRIORITY.load(std::sync::atomic::Ordering::Relaxed)
}

static STREAM_PRIORITY: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(i32::MIN);

static SPLITS_HINT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub fn set_splits_hint(splits: usize) { SPLITS_HINT.store(splits, std::sync::atomic::Ordering::Relaxed); }
pub fn splits_hint() -> usize { SPLITS_HINT.load(std::sync::atomic::Ordering::Relaxed) }
pub fn splits_for_request(tokens: usize) -> usize {
    const THRESHOLD: usize = 40_960;
    if std::env::var("CLEF_ATT_DISPATCH").as_deref() == Ok("0") { return att_keys(); }
    if tokens < THRESHOLD { 1 } else { att_keys() }
}
pub fn att_keys() -> usize {
    let n = env_once!("CLEF_ATT_KEYS").and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(4);
    if n > 1 && !att_ws() { return 1; }
    n
}

fn att_r240() -> bool { env_once!("CLEF_ATT_R240") != Some("0") }

fn att_gqa() -> bool { env_once!("CLEF_ATT_GQA") != Some("0") }

#[allow(dead_code)]
fn att_bq() -> usize { 128 }

fn att_mt() -> usize {
    if env_once!("CLEF_ATT_MT") == Some("2") { 2 } else { 1 }
}

fn att_ncg() -> usize {
    if env_once!("CLEF_ATT_NCG") == Some("2") { 2 } else { 1 }
}

fn att_mma_for(seq: usize) -> bool {
    match env_once!("CLEF_ATT_MMA") {
        Some("1") => true,
        Some("0") => false,

        _ => true,
    }
}

fn att_mma() -> bool {
    att_mma_for(usize::MAX)
}

fn att_wide() -> bool {
    env_once!("CLEF_ATT_WIDE") == Some("1")
}

fn attention_cubin(seq: usize) -> &'static [u8] {
    attention_cubin_for(seq, att_mma_for(seq))
}

fn attention_cubin_for(seq: usize, mma: bool) -> &'static [u8] {
    if mma {
        return if att_qreg96() { ATTENTION_MMA_QREG96_CUBIN }
               else if att_ws() && att_noskip() { ATTENTION_MMA_WS_NOSKIP_CUBIN }
               else if att_ws() && att_keys() > 1 { ATTENTION_MMA_WS_SPLIT_CUBIN }
               else if att_ws() && att_r240() { ATTENTION_MMA_WS_R240_CUBIN }
               else if att_ws() && att_bk48() { ATTENTION_MMA_WS_BK48_CUBIN }
               else if att_ws() { ATTENTION_MMA_WS_CUBIN }
               else if att_qreg() { ATTENTION_MMA_QREG_CUBIN }
               else if att_bk32() && att_gqa() { ATTENTION_MMA_GQA_BK32_CUBIN }
               else if att_bk32() { ATTENTION_MMA_BK32_CUBIN }
               else if att_gqa() { ATTENTION_MMA_GQA_CUBIN }
               else if att_ncg() == 1 { ATTENTION_MMA_MSPLIT_CUBIN }
               else if att_mt() == 2 { ATTENTION_MMA_MT2_CUBIN }
               else { ATTENTION_MMA_CUBIN };
    }
    if att_wide() {
        return ATTENTION_WIDE_CUBIN;
    }
    match att_rows_for(seq) {
        0 => ATTENTION_SMEM_CUBIN,
        8 => ATTENTION_R8_CUBIN,
        _ => ATTENTION_CUBIN,
    }
}

fn attention_module_name(seq: usize) -> &'static str {
    attention_module_name_for(seq, att_mma_for(seq))
}

fn attention_module_name_for(seq: usize, mma: bool) -> &'static str {
    if mma {
        return if att_qreg96() { "attention_mma_qreg96" }
               else if att_ws() && att_noskip() { "attention_mma_ws_noskip" }
               else if att_ws() && att_keys() > 1 { "attention_mma_ws_split" }
               else if att_ws() && att_r240() { "attention_mma_ws_r240" }
               else if att_ws() && att_bk48() { "attention_mma_ws_bk48" }
               else if att_ws() { "attention_mma_ws" }
               else if att_qreg() { "attention_mma_qreg" }
               else if att_bk32() && att_gqa() { "attention_mma_gqa_bk32" }
               else if att_bk32() { "attention_mma_bk32" }
               else if att_gqa() { "attention_mma_gqa" }
               else if att_ncg() == 1 { "attention_mma_msplit" }
               else if att_mt() == 2 { "attention_mma_mt2" }
               else { "attention_mma" };
    }
    if att_wide() {
        return "attention_wide";
    }
    match att_rows_for(seq) {
        0 => "attention_smem",
        8 => "attention_r8",
        _ => "attention",
    }
}

pub fn att_rows() -> u32 {
    att_rows_for(usize::MAX)
}

pub fn att_rows_for(seq: usize) -> u32 {
    if att_wide() {
        return 8;
    }
    match env_once!("CLEF_ATT_ROWS").and_then(|v| v.parse::<u32>().ok()) {
        Some(8) => 8,
        Some(12) => 12,
        Some(0) => 0,
        Some(2) => 2,

        _ => if seq > 2048 { 8 } else { 2 },
    }
}

const QUANTISE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantise.cubin"));


#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorMap([u64; 16]);

unsafe impl cudarc::driver::DeviceRepr for TensorMap {}

const BF16GEMM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bf16gemm.cubin"));

const BF16GEMM_N64_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bf16gemm_n64.cubin"));

const BF16GEMM_N64W4_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bf16gemm_n64w4.cubin"));

pub const KV_SPLITS: usize = 16;

pub const MHA_ROWS_PER_BLOCK: usize = 16;

const HEAD_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/head.cubin"));





pub struct GdnPrep {
    pub v: Buffer,
    pub q_hi: Buffer,
    pub q_lo: Buffer,
    pub k_hi: Buffer,
    pub k_lo: Buffer,
    pub kn: Buffer,
}

pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: std::sync::Mutex<std::collections::HashMap<String, Module>>,
    zero_i32: std::sync::OnceLock<Buffer>,
    scratch: std::sync::Mutex<[Option<Buffer>; 3]>,
    _test_gate: Option<TestGateToken>,
    graph_pool: std::sync::Mutex<usize>,
    capturing: std::sync::atomic::AtomicBool,
    in_capture: std::sync::atomic::AtomicBool,
    graph_pool_allocs: std::sync::atomic::AtomicUsize,
    graph_blocks: std::sync::Mutex<Vec<Arc<CudaSlice<u8>>>>,
    graph_block_has_graphs: std::sync::atomic::AtomicBool,
    graph_bump: std::sync::atomic::AtomicUsize,
    graph_free: Arc<GraphFreeList>,
    graph_block_cap: std::sync::atomic::AtomicUsize,
    graph_reserve: std::sync::atomic::AtomicUsize,
    graph_pool_bytes: std::sync::atomic::AtomicUsize,
    graph_pool_peak: std::sync::atomic::AtomicUsize,
}

struct TestGateToken;

static TEST_GATE_HELD: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
static TEST_GATE_FREE: std::sync::Condvar = std::sync::Condvar::new();

fn test_gate_acquire() -> Option<TestGateToken> {
    if !cfg!(test) {
        return None;
    }

    let mut held = TEST_GATE_HELD.lock().unwrap_or_else(|e| e.into_inner());
    while *held {
        held = TEST_GATE_FREE.wait(held).unwrap_or_else(|e| e.into_inner());
    }
    *held = true;
    Some(TestGateToken)
}

impl Drop for TestGateToken {
    fn drop(&mut self) {
        let mut held = TEST_GATE_HELD.lock().unwrap_or_else(|e| e.into_inner());
        *held = false;
        TEST_GATE_FREE.notify_one();
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.trim_device_pool();
    }
}

impl Device {
    pub fn open() -> Result<Self> {
        if GEMM_CUBIN.is_empty() {
            bail!("the CUDA kernels were not built (no nvcc at build time); GPU tests cannot run");
        }
        let ctx = CudaContext::new(0).context("opening CUDA device 0")?;

        let prio_spec = std::env::var("CLEF_STREAM_PRIORITY").unwrap_or_else(|_| "high".to_string());
        let stream = {
            let range = cudarc::driver::result::stream::get_priority_range().ok();
            let want = match (prio_spec.as_str(), range) {
                ("default", _) | ("0", _) => 0,
                ("high", Some((_, greatest))) => greatest,
                ("low", Some((least, _))) => least,
                (v, Some((least, greatest))) => v.parse::<i32>().map(|n| n.clamp(greatest, least)).unwrap_or(greatest),
                _ => 0,
            };
            STREAM_PRIORITY.store(want, std::sync::atomic::Ordering::Relaxed);
            if want == 0 {
                ctx.new_stream().context("creating the device stream")?
            } else {
                tracing::info!(priority = want, "creating the device stream at an explicit priority");
                ctx.new_stream_with_priority(want)
                    .context("creating the device stream at an explicit priority")?
            }
        };

        unsafe { ctx.disable_event_tracking() };

        if let Some(v) = env_once!("CLEF_POOL_RELEASE") {
            if let Ok(n) = v.parse::<u64>() {
                use cudarc::driver::sys;
                let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

                let r = unsafe { sys::cuDeviceGetMemPool(&mut pool, ctx.cu_device()) };
                if r == sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                    let mut value = n;

                    let r = unsafe {
                        sys::cuMemPoolSetAttribute(
                            pool,
                            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            (&mut value as *mut u64).cast(),
                        )
                    };
                    eprintln!("  CLEF_POOL_RELEASE={n} -> {r:?}");
                }
            }
        }
        let device = Self {
            ctx,
            stream,
            modules: std::sync::Mutex::new(std::collections::HashMap::new()),
            zero_i32: std::sync::OnceLock::new(),
            scratch: std::sync::Mutex::new([None, None, None]),
            _test_gate: test_gate_acquire(),
            graph_pool: std::sync::Mutex::new(0),
            capturing: std::sync::atomic::AtomicBool::new(false),
            in_capture: std::sync::atomic::AtomicBool::new(false),
            graph_pool_allocs: std::sync::atomic::AtomicUsize::new(0),
            graph_pool_bytes: std::sync::atomic::AtomicUsize::new(0),
            graph_pool_peak: std::sync::atomic::AtomicUsize::new(0),
            graph_blocks: std::sync::Mutex::new(Vec::new()),
            graph_block_has_graphs: std::sync::atomic::AtomicBool::new(false),
            graph_bump: std::sync::atomic::AtomicUsize::new(0),
            graph_reserve: std::sync::atomic::AtomicUsize::new(2 << 30),
            graph_block_cap: std::sync::atomic::AtomicUsize::new(0),
            graph_free: GraphFreeList::new(),
};

        device.configure_graph_mempool()?;
        Ok(device)
    }

    pub fn describe(&self) -> Result<String> {
        let name = self.ctx.name().unwrap_or_else(|_| "?".into());
        let major = self.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0);
        let minor = self.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
        Ok(format!("{name} (sm_{major}{minor})"))
    }

    pub fn total_memory(&self) -> Result<usize> {
        let free = self.ctx.mem_get_info().context("mem_get_info")?;
        Ok(free.1)
    }

    pub fn free_memory(&self) -> Result<usize> {
        let free = self.ctx.mem_get_info().context("mem_get_info")?;
        Ok(free.0)
    }

    pub fn swizzle_module(&self) -> Result<Module> {
        self.load(SWIZZLE_CUBIN, "swizzle")
    }

    pub fn gemm_module(&self) -> Result<Module> {
        self.load(GEMM_CUBIN, "gemm")
    }

    pub fn head_module(&self) -> Result<Module> {
        self.load(HEAD_CUBIN, "head")
    }

    pub fn fp8_gemm(&self, a: &[u8], b: &[u8], sa: &[u8], sb: &[u8], m: usize, n: usize, k: usize,
                    add: Option<&Buffer>)
        -> Result<Vec<f32>> {
        assert_eq!(sa.len(), m * 2, "one BF16 scale per output row of A");
        assert_eq!(sb.len(), n * 2, "one BF16 scale per output row of B");

        let v = fp8_variant(m, n, k);
        let n_block: u32 = v.facts.n_block as u32;
        let (cubin, cname) = (v.cubin, v.name);
        let facts = v.facts;

        let mut bad = crate::model::fp8_cubin_matches_launch(
            &facts, facts.m_tile, n_block as usize, facts.stages, 384);

        let work = m.div_ceil(facts.m_tile) * n.div_ceil(n_block as usize);
        let sms = self.sm_count();
        bad.extend(crate::model::fp8_violations(&crate::model::Fp8Side {
            grid_x: work.min(sms * v.ctas_per_sm), grid_y: 1, m, n,
            ctas_per_sm: v.ctas_per_sm, sms,
            ..v.side
        }));
        if !bad.is_empty() {
            anyhow::bail!(
                "the FP8 GEMM launch is inconsistent, and would HANG rather than fail:\n  - {}",
                bad.join("\n  - ")
            );
        }
        let module = self.load(cubin, cname)?;
        let kernel = module.kernel("fp8_gemm")?;
        let dev_a = self.upload(a)?;
        let dev_b = self.upload(b)?;
        let dev_sa = self.upload(sa)?;
        let dev_sb = self.upload(sb)?;

        let padded_m = fp8_padded_m(m, facts.m_tile);
        let dev_c = self.alloc(padded_m * n * 4)?;
        let (box_inner, box_sw) = fp8_box(&facts);
        let map_a = self.make_tensor_map_2d_sw(
            dev_a.device_ptr() as u64, k as u64, m as u64, box_inner, facts.m_tile as u32,
            box_sw)?;
        let map_b = self.make_tensor_map_2d_sw(
            dev_b.device_ptr() as u64, k as u64, n as u64, box_inner, n_block,
            box_sw)?;

        let map_c = if n % 4 == 0 {
            self.make_tensor_map_2d_sw(
                dev_c.device_ptr() as u64, (n * 4) as u64, padded_m as u64, FP8_EPI_N * 4,
                facts.m_tile as u32, fp8_epi_swizzle(n))?
        } else {
            TensorMap([0u64; 16])
        };
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&map_a)
                .arg(&map_b)
                .arg(&dev_a.slice())
                .arg(&dev_b.slice())
                .arg(&dev_sa.slice())
                .arg(&dev_sb.slice())
                .arg(&dev_c.slice())

                .arg(&add.map(|r| r.device_ptr()).unwrap_or(0u64))

                .arg(&map_c)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&mi)

                .arg(&(m as i32))

                .arg(&fp8_log_swizzle(m))

                .launch({
                    let work = m.div_ceil(facts.m_tile) * n.div_ceil(n_block as usize);
                    config_3d(work.min(self.sm_count() * v.ctas_per_sm) as u32, 1, 384)
                })
                .context("launching the FP8 GEMM")?;
        }
        let raw_all = self.download(&dev_c)?;

        let raw = &raw_all[..m * n * 4];
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn gdn_scan(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        seq_len: usize,
        heads: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(q.len(), seq_len * heads * k_dim);
        assert_eq!(k.len(), seq_len * heads * k_dim);
        assert_eq!(v.len(), seq_len * heads * v_dim);
        assert_eq!(decay.len(), seq_len * heads);
        assert_eq!(beta.len(), seq_len * heads);
        assert_eq!(v_dim % 4, 0, "the value dimension is split 4 ways");
        assert_eq!(k_dim % 4, 0, "the key dimension is split 4 ways");

        assert!(k_dim / 8 <= 16, "MAX_ROWS in gdn.cu is 16; k_dim must be <= 128");

        let module = self.load(GDN_CUBIN, "gdn")?;
        let kernel = module.kernel("gdn_scan")?;
        let dev_q = self.upload_f32(q)?;
        let dev_k = self.upload_f32(k)?;
        let dev_v = self.upload_f32(v)?;
        let dev_g = self.upload_f32(decay)?;
        let dev_beta = self.upload_f32(beta)?;
        let dev_out = self.alloc(seq_len * heads * v_dim * 4)?;
        let (sl, nh, kd, vd) = (seq_len as i32, heads as i32, k_dim as i32, v_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_q.slice())
                .arg(&dev_k.slice())
                .arg(&dev_v.slice())
                .arg(&dev_g.slice())
                .arg(&dev_beta.slice())
                .arg(&dev_out.slice())
                .arg(&sl)
                .arg(&nh)
                .arg(&kd)
                .arg(&vd)

                .launch(config_block_2d(heads as u32, 4, v_dim as u32 / 4, 8))
                .context("launching the GDN scan")?;
        }
        let raw = self.download(&dev_out)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn rms_norm(&self, x: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32)
        -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * cols);
        assert_eq!(weight.len(), cols);
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rms_norm_kernel")?;
        let dev_x = self.upload_f32(x)?;
        let dev_w = self.upload_f32(weight)?;
        let dev_out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_x.slice())
                .arg(&dev_w.slice())
                .arg(&dev_out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)

                .launch(config(row_blocks(rows, cols), 256))
                .context("launching rms_norm")?;
        }
        let raw = self.download(&dev_out)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn rms_norm_gated(&self, x: &[f32], gate: &[f32], weight: &[f32], rows: usize, cols: usize,
                          eps: f32) -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * cols);
        assert_eq!(gate.len(), rows * cols);
        assert_eq!(weight.len(), cols);
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rms_norm_gated_kernel")?;
        let dev_x = self.upload_f32(x)?;
        let dev_g = self.upload_f32(gate)?;
        let dev_w = self.upload_f32(weight)?;
        let dev_out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_x.slice())
                .arg(&dev_g.slice())
                .arg(&dev_w.slice())
                .arg(&dev_out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)
                .launch(config(row_blocks(rows, cols), 256))
                .context("launching rms_norm_gated")?;
        }
        let raw = self.download(&dev_out)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        assert_eq!(gate.len(), up.len());
        let total = gate.len();
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("swiglu_kernel")?;
        let dev_g = self.upload_f32(gate)?;
        let dev_u = self.upload_f32(up)?;
        let dev_out = self.alloc(total * 4)?;
        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_g.slice())
                .arg(&dev_u.slice())
                .arg(&dev_out.slice())
                .arg(&n)
                .launch(config(((total as u32) + 1023) / 1024, 256))
                .context("launching swiglu")?;
        }
        let raw = self.download(&dev_out)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn causal_conv_silu(&self, x: &[f32], w: &[f32], rows: usize, channels: usize, k: usize)
        -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * channels);
        assert_eq!(w.len(), channels * k);

        let dev_x = self.upload_f32(x)?;
        let dev_w = self.upload_f32(w)?;
        let dev_out = self.causal_conv_silu_dev(&dev_x, &dev_w, rows, channels, k, None, None, None)?;
        let raw = self.download(&dev_out)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn rope_tables(&self, positions: &[i32], rotary_dim: usize, theta: f32)
        -> Result<(Vec<f32>, Vec<f32>)> {
        let rows = positions.len();
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rope_tables_kernel")?;
        let dev_p = self.upload_i32(positions)?;
        let dev_c = self.alloc(rows * rotary_dim * 4)?;
        let dev_s = self.alloc(rows * rotary_dim * 4)?;
        let (r, rd) = (rows as i32, rotary_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_p.slice())
                .arg(&dev_c.slice())
                .arg(&dev_s.slice())
                .arg(&r)
                .arg(&rd)
                .arg(&theta)
                .launch(config(((rows * rotary_dim) as u32 + 255) / 256, 256))
                .context("launching rope_tables")?;
        }
        let dec = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        Ok((dec(self.download(&dev_c)?), dec(self.download(&dev_s)?)))
    }

    pub fn rope_apply(&self, x: &[f32], cos: &[f32], sin: &[f32], rows: usize, heads: usize,
                      head_dim: usize, rotary_dim: usize) -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * heads * head_dim);
        assert_eq!(cos.len(), rows * rotary_dim);
        assert_eq!(sin.len(), rows * rotary_dim);

        let dev_x = self.upload_f32(x)?;
        let dev_c = self.upload_f32(cos)?;
        let dev_s = self.upload_f32(sin)?;
        let dev_out = self.rope_apply_dev(&dev_x, &dev_c, &dev_s, rows, heads, head_dim, rotary_dim,
                                          None, None, None)?;
        let raw = self.download(&dev_out)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, q: &[f32], k: &[f32], v: &[f32], seq: usize, heads: usize,
                     kv_heads: usize, head_dim: usize) -> Result<Vec<f32>> {
        assert_eq!(q.len(), seq * heads * head_dim);
        assert_eq!(k.len(), seq * kv_heads * head_dim);
        assert_eq!(v.len(), seq * kv_heads * head_dim);
        assert!(head_dim <= 256, "attention.cu supports head_dim <= 256");
        assert_eq!(heads % kv_heads, 0, "grouped query needs a whole ratio");

        let module = self.load(ATTENTION_CUBIN, "attention")?;
        let kernel = module.kernel("attention_kernel")?;
        let dev_q = self.upload_f32(q)?;
        let dev_k = self.upload_f32(k)?;
        let dev_v = self.upload_f32(v)?;
        let dev_out = self.alloc(seq * heads * head_dim * 4)?;
        let (s, h, kh, hd) = (seq as i32, heads as i32, kv_heads as i32, head_dim as i32);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_q.slice())
                .arg(&dev_k.slice())
                .arg(&dev_v.slice())
                .arg(&dev_out.slice())
                .arg(&s)
                .arg(&h)
                .arg(&kh)
                .arg(&hd)
                .arg(&scale).arg(&SEQ_START_NULL)

                .launch(config_block_2d(heads as u32, rows_to_blocks(seq), 64, 1))
                .context("launching attention")?;
        }
        let raw = self.download(&dev_out)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn quantise_nvfp4(&self, x: &[f32], rows: usize, cols: usize, global_scale: f32)
        -> Result<(Vec<u8>, Vec<u8>)> {
        assert_eq!(x.len(), rows * cols);
        assert_eq!(cols % 16, 0, "NVFP4 scales cover 16 values");
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("quantise_nvfp4_kernel")?;
        let dev_x = self.upload_f32(x)?;

        let dev_packed = self.alloc(rows.div_ceil(128) * 128 * (cols / 2))?;
        let blocks_per_row = cols / 16;
        let rows_padded = crate::swizzle::round_up_public(rows, 128);

        let dev_scales = self.alloc(rows_padded * blocks_per_row)?;
        let (r, c) = (rows as i32, cols as i32);
        let blocks_per_row = (cols / 16) as u32;

        let blk_mn = 128i32;
        let chunk = crate::swizzle::sf_chunk_stride_for_tile_m(128) as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_x.slice())
                .arg(&dev_packed.slice())
                .arg(&dev_scales.slice())
                .arg(&r)
                .arg(&c)
                .arg(&global_scale)
                .arg(&blk_mn)
                .arg(&chunk)
                .launch(config_split((blocks_per_row + 63) / 64, rows_padded.min(65535) as u32,
                                     rows_padded.div_ceil(65535) as u32, 64))
                .context("launching quantise_nvfp4")?;
        }
        Ok((self.download(&dev_packed)?, self.download(&dev_scales)?))
    }

    pub fn quantise_fp8(&self, x: &[f32], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        assert_eq!(x.len(), rows * cols);
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("quantise_fp8_kernel")?;
        let dev_x = self.upload_f32(x)?;
        let dev_out = self.alloc(rows * cols)?;
        let dev_scales = self.alloc(rows * 2)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_x.slice())
                .arg(&dev_out.slice())
                .arg(&dev_scales.slice())
                .arg(&r)
                .arg(&c)
                .launch(config(row_blocks(rows, cols), 256))
                .context("launching quantise_fp8")?;
        }
        Ok((self.download(&dev_out)?, self.download(&dev_scales)?))
    }

    pub fn bf16_gemm(&self, a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k * 2, "BF16 is two bytes per value");
        assert_eq!(b.len(), n * k * 2);
        assert_eq!(k % 16, 0, "the BF16 atom contracts over 16");
        let (cubin, name, _threads, _n_block) = Self::bf16_variant(n);
        let module = self.load(cubin, name)?;
        let kernel = module.kernel("bf16_gemm")?;
        let dev_a = self.upload(a)?;
        let dev_b = self.upload(b)?;

        let k_tiles = (k / 64).max(1);

        let n_blocks = ((n + 15) / 16).max(1);

        let n_splits = match env_once!("CLEF_BF16_SPLIT").and_then(|v| v.parse::<usize>().ok())
        {
            Some(forced) => forced.clamp(1, k_tiles.min(32)),

            None => 4usize.clamp(1, k_tiles.min(32)),
        };

        let dev_c = self.alloc(m * n * 4)?;
        let c_elems = m * n;

        let partials = if n_splits > 1 {
            Some(self.alloc_zeroed(n_splits * c_elems * 4)?)
        } else {
            None
        };
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dev_a.slice())
                .arg(&dev_b.slice())
                .arg(&dev_c.slice())
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)

                .arg(&(m as i32))
                .arg(&(n_splits as i32))
                .arg(&partials.as_ref().map(|b| b.slice()).unwrap_or_else(|| dev_a.slice()))
                .arg(&(c_elems as i32))
                .launch(LaunchConfig {
                    grid_dim: (((n + 15) / 16) as u32, ((m + 63) / 64) as u32, n_splits as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("launching the BF16 GEMM")?;
        }
        if let Some(partials) = partials.as_ref() {
            let reduce = module.kernel("bf16_split_reduce")?;
            let elems = c_elems as i32;
            let splits = n_splits as i32;
            unsafe {
                use cudarc::driver::PushKernelArg;
                self.stream
                    .launch_builder(&reduce.function)
                    .arg(&partials.slice())
                    .arg(&dev_c.slice())
                    .arg(&elems)
                    .arg(&splits)
                    .launch(config((c_elems as u32).div_ceil(256), 256))
                    .context("reducing the split-K partials")?;
            }
        }
        let raw = self.download(&dev_c)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    #[allow(clippy::too_many_arguments)]

    #[allow(clippy::too_many_arguments)]

    pub fn nvfp4_m_tile_for(m: usize) -> usize {
        const M_TILE_128: usize = 4 * 32;
        const M_TILE_64: usize = 4 * 16;
        const M_TILE_192: usize = 6 * 32;

        let padded_m = m.div_ceil(16) * 16;
        let wide_rows = padded_m.div_ceil(128) * 128;
        let narrow = (wide_rows - padded_m) as f64 / wide_rows as f64 > 0.20;
        match env_once!("CLEF_NVFP4_TILE") {
            Some("64") => M_TILE_64,
            Some("128") => M_TILE_128,
            Some("192") | Some("192w8") => M_TILE_192,
            _ => {
                let _ = (narrow, wide_rows, padded_m);
                M_TILE_128
            }
        }
    }

    fn launch_nvfp4_gemm(&self, a_packed: &Buffer, b_packed: &Buffer, a_scales: &Buffer,
                         b_scales: &Buffer, out: &Buffer, padded_m: usize, n: usize, k: usize,
                         global_a: f32, global_b: f32, a_rows: usize, n_splits: usize,
                         ) -> Result<()> {
        const M_TILE_128: usize = 4 * 32;
        const M_TILE_64: usize = 4 * 16;
        const M_TILE_192: usize = 6 * 32;
        const M_TILE_256: usize = 8 * 32;

        let m_tile: usize = Self::nvfp4_m_tile_for(padded_m);

        let forced = env_once!("CLEF_NVFP4_TILE");
        let wide8 = matches!(forced.as_deref(), Some("192w8"));

        let stages3 = matches!(forced.as_deref(), Some("stages3"));
        let pick: u32 = if stages3 { 33 } else if wide8 { 1928 } else { m_tile as u32 };

        let m_tile: usize = match pick {
            64 => 64,
            192 | 1928 => 192,
            33 => 128,
            _ => m_tile,
        };
        let module = match pick {
            64 => self.load(GEMM_M64_CUBIN, "gemm_m64")?,
            192 => self.load(GEMM_M192_CUBIN, "gemm_m192")?,
            1928 => self.load(GEMM_M192W8_CUBIN, "gemm_m192w8")?,
            33 => self.load(GEMM_STAGES3_CUBIN, "gemm_stages3")?,
            _ => self.load(GEMM_CUBIN, "gemm")?,
        };

        let cubin = match pick {
            64 => crate::model::CubinFacts::GEMM_M64,
            192 => crate::model::CubinFacts::GEMM_M192,
            1928 => crate::model::CubinFacts::GEMM_M192W8,
            33 => crate::model::CubinFacts::GEMM,
            _ => crate::model::CubinFacts::GEMM,
        };
        let _ = (M_TILE_128, M_TILE_64, M_TILE_192);

        let cta_threads: u32 = if pick == 192 { 512 } else { 384 };
        let kernel = module.kernel("nvfp4_gemm")?;

        const N_BLOCK: u32 = 2 * 64;

        {
            let atom = crate::swizzle::sf_atom_in_force(m_tile);
            let swizzle_side = crate::model::SwizzleSide {
                atom_rows: atom,
                chunk_stride: crate::swizzle::sf_chunk_stride_in_force(m_tile),
                tile_m: m_tile,
                rows_padded: crate::swizzle::round_up_public(a_rows, atom),
            };
            let sub_chunks = 2usize;
            let sf_atoms_m = (m_tile / atom).max(1);

            let chunk_stride_a = crate::swizzle::sf_chunk_stride_in_force(m_tile);
            let kernel_side = crate::model::KernelSide {
                sf_blk_mn: atom,
                sf_chunk_stride_a: chunk_stride_a,
                sf_chunk_stride_b: 512,
                m_tile,
                n_block: 128,
                sub_chunks,
                sf_rows: atom,
                sf_stage_bytes_a_total: sf_atoms_m * sub_chunks * chunk_stride_a,
                sf_stage_bytes_b: sub_chunks * 512,
            };
            let launch_side = crate::model::LaunchSide {
                kernel_threads: cta_threads as usize,
                launched_threads: cta_threads as usize,
                tma_box_rows: m_tile,
                c_box_rows: m_tile,
                kernel_epi_n: EPI_N as usize,
                c_box_inner_bytes: (EPI_N * 4) as usize,
                expect_tx: usize::MAX,
            };
            let mut bad = crate::model::consistency(&swizzle_side, &kernel_side, &launch_side);

            bad.extend(crate::model::cubin_matches_launch(
                &cubin, m_tile, cta_threads as usize, EPI_N as usize, atom));
            if !bad.is_empty() {
                anyhow::bail!(
                    "the GEMM launch is inconsistent, and would hang rather than fail:\n  - {}",
                    bad.join("\n  - ")
                );
            }
        }
        let map_a = self.make_tensor_map_2d_sw(
            a_packed.device_ptr() as u64, (k / 2) as u64, a_rows as u64, 64, m_tile as u32,
            cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_64B)?;
        let map_b = self.make_tensor_map_2d_sw(
            b_packed.device_ptr() as u64, (k / 2) as u64, n as u64, 64, N_BLOCK,
            cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_64B)?;

        const EPI_N: u32 = 32;

        let map_c = if n % 4 == 0 {
            self.make_tensor_map_2d_sw(
                out.device_ptr() as u64, (n * 4) as u64, padded_m as u64, EPI_N * 4, m_tile as u32,
                cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_128B)?
        } else {
            TensorMap([0u64; 16])
        };

        let log_swizzle: i32 = match env_once!("CLEF_SWIZZLE").and_then(|v| v.parse::<i32>().ok()) {
            Some(v) => v,

            None => match padded_m {
                0..=1023 => 0,
                2048..=3071 => 6,
                3072..=6143 => 0,
                6144..=12287 => 3,
                12288..=24575 => 5,
                _ => 3,
            },
        };
        let (mi, ni, ki) = (padded_m as i32, n as i32, k as i32);
        let ar = a_rows as i32;
        let ns = n_splits as i32;

        let work_tiles = (((padded_m + m_tile - 1) / m_tile) * ((n + 127) / 128)) as u32;
        let sms = self
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(0)
            .max(0) as u32;
        let persistent_grid = if sms == 0 { work_tiles } else { work_tiles.min(sms) };
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&map_a)
                .arg(&map_b)
                .arg(&map_c)
                .arg(&a_packed.slice())
                .arg(&b_packed.slice())
                .arg(&a_scales.slice())
                .arg(&b_scales.slice())
                .arg(&out.slice())
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&global_a)
                .arg(&global_b)
                .arg(&ar)
                .arg(&ns)
                .arg(&log_swizzle)

                .launch(config_split(persistent_grid, 1, 1, cta_threads))
                .context("launching the NVFP4 GEMM")?;
        }
        Ok(())
    }

    pub fn nvfp4_gemm_packed(
        &self,
        a_packed: &[u8],
        a_scales: &[u8],
        b_packed: &Buffer,
        b_scales: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        global_a: f32,
        global_b: f32,
        sel_mode: i32,
        b_sel_mode: i32,
        byte_order: i32,
    ) -> Result<Vec<f32>> {
        let dev_a = self.upload(a_packed)?;
        let dev_sa = self.upload(a_scales)?;

        let n_splits = 1usize;

        let dev_c = if n_splits > 1 {
            self.alloc_zeroed(m * n * 4)?
        } else {
            self.alloc(m * n * 4)?
        };
        self.launch_nvfp4_gemm(&dev_a, b_packed, &dev_sa, b_scales, &dev_c, m, n, k, global_a,
                               global_b, m, n_splits)?;
        let raw = self.download(&dev_c)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn bf16_gemm_f32(&self, x: &[f32], w_bytes: &[u8], m: usize, n: usize, k: usize)
        -> Result<Vec<f32>> {
        let round = |v: f32| -> u16 {
            let bits = v.to_bits();
            let lsb = (bits >> 16) & 1;
            ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
        };
        let a: Vec<u8> = x.iter().flat_map(|v| round(*v).to_le_bytes()).collect();
        self.bf16_gemm(&a, w_bytes, m, n, k)
    }

    pub fn download_rows_f32(&self, buffer: &Buffer, rows: usize, cols: usize) -> Result<Vec<f32>> {
        let all = self.download_f32(buffer)?;
        let want = rows * cols;
        if all.len() < want {
            anyhow::bail!("the buffer holds {} values, and {rows}x{cols} needs {want}", all.len());
        }
        Ok(all[..want].to_vec())
    }

    pub fn download_f32(&self, buffer: &Buffer) -> Result<Vec<f32>> {
        Ok(self
            .download(buffer)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn scratch(&self, slot: usize, bytes: usize) -> Result<Buffer> {
        let mut g = self.scratch.lock().map_err(|_| anyhow!("scratch poisoned"))?;
        let need = bytes.max(1);
        let fits = g[slot].as_ref().map(|b| b.len() >= need).unwrap_or(false);
        if !fits {
            if self.capturing.load(std::sync::atomic::Ordering::SeqCst) {
                anyhow::bail!(
                    "scratch slot {slot} needs {need} bytes and this is inside a graph capture: a buffer \
                     that outlives the capture cannot be carved from the capture's block, because the next \
                     capture re-hands that range out. Warm it before the capture opens \
                     (`Device::warm_attention_scratch`).");
            }
            let b = self.alloc(need)?;
            if matches!(env_once!("CLEF_GRAPH_ALLOC_TRACE"), Some(v) if v != "0") {
                eprintln!("SCRATCH slot={slot} need={need} ptr={:#x} off={} block_len={}",
                          b.device_ptr(), b.offset, b.data.len());
            }
            g[slot] = Some(b);
        }
        Ok(g[slot].as_ref().expect("just set").clone_buffer())
    }

    fn split_probe(&self, partials: &Buffer, splits: usize, seq: usize, heads: usize,
                   head_dim: usize) -> Result<()> {
        self.sync()?;
        let raw = self.download_f32(partials)?;
        let per = heads * (head_dim + 2);
        let nan = raw.iter().filter(|v| v.is_nan()).count();
        let zero = raw.iter().filter(|v| **v == 0.0).count();
        let floor = raw.iter().filter(|v| **v == -1.0e30f32).count();
        let mut line = String::new();
        for sp in 0..splits {
            let base = sp * seq * per;
            let m0 = raw.get(base).copied().unwrap_or(f32::NAN);
            let l0 = raw.get(base + 1).copied().unwrap_or(f32::NAN);
            let last = base + (seq - 1) * per;
            let m1 = raw.get(last).copied().unwrap_or(f32::NAN);
            line.push_str(&format!(" sp{sp}[m0={m0:.3e} l0={l0:.3e} mlast={m1:.3e}]"));
        }
        eprintln!("SPLIT-DBG after-launch: elts={} nan={nan} zero={zero} floor={floor}{line}", raw.len());
        Ok(())
    }

    pub fn warm_attention_scratch(&self, heads: usize, head_dim: usize) -> Result<()> {
        let splits = att_keys();
        if splits > 1 {
            let _ = self.scratch(2, split_partials_bytes(splits, heads, head_dim))?;
        }
        Ok(())
    }

    pub fn warm_constants(&self) -> Result<()> {
        self.zero_i32()?;
        Ok(())
    }

    fn zero_i32(&self) -> Result<Buffer> {
        if let Some(b) = self.zero_i32.get() {
            return Ok(b.clone_buffer());
        }
        let b = self.upload_i32(&[0])?;
        let _ = self.zero_i32.set(b.clone_buffer());
        Ok(b)
    }

    pub fn to_bf16_dev(&self, x: &Buffer, total: usize) -> Result<Buffer> {
        let zero = self.zero_i32()?;
        self.to_bf16_dev_at(x, total, 1, &zero)
    }

    pub fn to_bf16_dev_inner(&self, x: &Buffer, total: usize, zero: &Buffer) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("f32_to_bf16_kernel")?;
        let out = self.alloc(total * 2)?;
        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&out.slice())
                .arg(&n)
                .arg(&zero.slice())
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching f32_to_bf16")?;
        }
        Ok(out)
    }

fn bf16_variant(n: usize) -> (&'static [u8], &'static str, u32, usize) {
    match env_once!("CLEF_BF16_TILE") {
        Some("narrow") => (BF16GEMM_CUBIN, "bf16gemm", 256, 16),
        Some("n64") => (BF16GEMM_N64_CUBIN, "bf16gemm_n64", 256, 64),
        Some("n64w4") => (BF16GEMM_N64W4_CUBIN, "bf16gemm_n64w4", 512, 64),

        _ => (BF16GEMM_CUBIN, "bf16gemm", 256, 16),
    }
}

    pub fn bf16_gemm_dev(&self, a: &Buffer, b: &Buffer, m: usize, n: usize, k: usize)
        -> Result<Buffer> {
        self.bf16_gemm_dev_forced(a, b, m, n, k, None)
    }

    pub fn bf16_gemm_dev_forced(&self, a: &Buffer, b: &Buffer, m: usize, n: usize, k: usize,
                                force_wide: Option<bool>) -> Result<Buffer> {
        assert_eq!(k % 16, 0, "the BF16 atom contracts over 16");
        let (cubin, name, threads, n_block) = match force_wide {
            Some(true) => (BF16GEMM_N64W4_CUBIN, "bf16gemm_n64w4", 512u32, 64usize),
            Some(false) => (BF16GEMM_CUBIN, "bf16gemm", 256u32, 16usize),
            None => Self::bf16_variant(n),
        };
        let module = self.load(cubin, name)?;
        let kernel = module.kernel("bf16_gemm")?;

        let padded_m = m.div_ceil(16) * 16;

        let k_tiles = (k / 64).max(1);

        let n_blocks = ((n + 15) / 16).max(1);

        let n_splits = match env_once!("CLEF_BF16_SPLIT").and_then(|v| v.parse::<usize>().ok())
        {
            Some(forced) => forced.clamp(1, k_tiles.min(32)),

            None => 4usize.clamp(1, k_tiles.min(32)),
        };

        let out = self.alloc(padded_m * n * 4)?;

        if let Some(v) = env_once!("CLEF_PAD_FILL") {
            if let Ok(byte) = u8::from_str_radix(v.trim_start_matches("0x"), 16) {
                self.fill_range(&out, m * n * 4, (padded_m - m) * n * 4, byte)?;
            }
        }
        let _ = n_splits;
        let c_elems = padded_m * n;

        let partials = if n_splits > 1 {
            Some(self.alloc_zeroed(n_splits * c_elems * 4)?)
        } else {
            None
        };
        let (mi, ni, ki) = (padded_m as i32, n as i32, k as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&a.slice())
                .arg(&b.slice())
                .arg(&out.slice())
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&(m as i32))
                .arg(&(n_splits as i32))
                .arg(&partials.as_ref().map(|b| b.slice()).unwrap_or_else(|| a.slice()))
                .arg(&(c_elems as i32))

                .launch(LaunchConfig {
                    grid_dim: (n.div_ceil(n_block) as u32, ((padded_m + 63) / 64) as u32,
                               n_splits as u32),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("launching the device BF16 GEMM")?;
        }

        if let Some(partials) = partials.as_ref() {
            let reduce = module.kernel("bf16_split_reduce")?;
            let elems = c_elems as i32;
            let splits = n_splits as i32;
            unsafe {
                use cudarc::driver::PushKernelArg;
                self.stream
                    .launch_builder(&reduce.function)
                    .arg(&partials.slice())
                    .arg(&out.slice())
                    .arg(&elems)
                    .arg(&splits)
                    .launch(config((c_elems as u32).div_ceil(256), 256))
                    .context("reducing the split-K partials")?;
            }
        }
        Ok(out)
    }

    pub fn bf16_linear_dev(&self, a: &Buffer, w: &Buffer, bias: &Buffer, m: usize, n: usize,
                           k: usize) -> Result<Buffer> {
        let gemm = self.bf16_gemm_dev(a, w, m, n, k)?;
        self.bias_add_dev(&gemm, bias, m, n)
    }

    pub fn layer_norm_dev(&self, x: &Buffer, weight: &Buffer, bias: &Buffer, rows: usize,
                          cols: usize, eps: f32) -> Result<Buffer> {
        assert!(cols % 4 == 0, "layer_norm reads float4, so cols must be a multiple of 4, got {cols}");
        let zero = self.zero_i32()?;
        self.layer_norm_dev_at(x, weight, bias, rows, cols, eps, &zero)
    }

    pub fn layer_norm_dev_at(&self, x: &Buffer, weight: &Buffer, bias: &Buffer, rows: usize,
                             cols: usize, eps: f32, base: &Buffer) -> Result<Buffer> {
        assert!(cols % 4 == 0, "layer_norm reads float4, so cols must be a multiple of 4, got {cols}");
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("layer_norm_kernel")?;
        let out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&weight.slice())
                .arg(&bias.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)
                .arg(&base.slice())
                .launch(config(rows as u32, 256))
                .context("launching layer_norm")?;
        }
        Ok(out)
    }

    pub fn to_bf16_dev_at(&self, x: &Buffer, total: usize, cols: usize, base: &Buffer) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("f32_to_bf16_kernel")?;
        let out = self.alloc(total * 2)?;
        let n = total as i32;
        let c = cols as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&out.slice())
                .arg(&n)
                .arg(&c)
                .arg(&base.slice())
                .launch(config(((total as u32) + 255) / 256, 256))

                .with_context(|| format!(
                    "launching f32_to_bf16 (offset): total={total} n={n} cols={cols} grid={} x_len={} out_len={}",
                    ((total as u32) + 255) / 256, x.len(), out.len()))?;
        }
        Ok(out)
    }

    pub fn gelu_erf_dev(&self, x: &Buffer, total: usize) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("gelu_erf_kernel")?;
        let out = self.alloc(total * 4)?;
        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&out.slice())
                .arg(&n)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching gelu_erf")?;
        }
        Ok(out)
    }

    pub fn bias_add_dev(&self, x: &Buffer, bias: &Buffer, rows: usize, cols: usize)
        -> Result<Buffer> {
        let total = rows * cols;
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("bias_add_kernel")?;
        let out = self.alloc(total * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&bias.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .launch(config(((total as u32) + 1023) / 1024, 256))
                .context("launching bias_add")?;
        }
        Ok(out)
    }

    pub fn add_dev(&self, a: &Buffer, b: &Buffer, total: usize) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("add_kernel")?;
        let mut out = self.alloc(total * 4)?;

        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&out.slice())
                .arg(&a.slice())
                .arg(&b.slice())
                .arg(&n)
                .launch(config(((total as u32) + 1023) / 1024, 256))
                .context("launching add")?;
        }
        Ok(out)
    }

    pub fn mha_combine_dev(&self, partial: &Buffer, out: &Buffer, q_len: usize,
                           width: usize, heads: usize, n_splits: usize) -> Result<()> {
        self.mha_combine_for_test(partial, out, q_len, width, heads, n_splits)
    }

    pub fn mha_combine_for_test(&self, partial: &Buffer, out: &Buffer, q_len: usize,
                                width: usize, heads: usize, n_splits: usize) -> Result<()> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let ck = module.kernel("mha_combine_kernel")?;
        let (ql, w, h, n) = (q_len as i32, width as i32, heads as i32, n_splits as i32);
        let groups = ((q_len as u32) + combine_rows_per_block() - 1) / combine_rows_per_block();
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&ck.function)
                .arg(&partial.slice()).arg(&out.slice())
                .arg(&ql).arg(&w).arg(&h).arg(&n)
                .launch(config_3d(heads as u32, groups, 128))
                .context("launching mha combine (test)")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mha_partial_scalar_for_test(&self, q: &Buffer, k: &Buffer, v: &Buffer, partial: &Buffer,
                                       q_len: usize, kv_len: usize, width: usize, heads: usize,
                                       scale: f32, rows: usize) -> Result<()> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let pk = module.kernel("mha_partial_kernel_pl2")?;
        let groups_q = ((q_len as u32) + (MHA_ROWS_PER_BLOCK as u32) - 1) / (MHA_ROWS_PER_BLOCK as u32);
        let (ql, kl, w, h) = (q_len as i32, kv_len as i32, width as i32, heads as i32);
        let kreal = self.upload_i32(&[rows as i32])?;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&pk.function)
                .arg(&q.slice()).arg(&k.slice()).arg(&v.slice()).arg(&partial.slice())
                .arg(&ql).arg(&kl).arg(&w).arg(&h).arg(&scale).arg(&0i32)
                .arg(&kreal.slice()).arg(&(rows as i32))
                .launch(config_grid_3d(heads as u32, groups_q, 1, 128))
                .context("launching mha partial (test)")?;
        }
        Ok(())
    }

    pub fn mha_chunks_dev<F>(&self, q: &Buffer, q_len: usize, kv_len: usize,
                             kv_real: &Buffer, width: usize, heads: usize,
                             chunks: &[(usize, usize)], mut build: F) -> Result<Buffer>
    where
        F: FnMut(&Self, usize, usize) -> Result<(Buffer, Buffer)>,
    {
        let module = self.load(HEAD_CUBIN, "head")?;
        let head_dim = width / heads;
        let n = chunks.len();
        let per = q_len * heads * (head_dim + 2) * 4;

        let partial = self.scratch(0, n * per)?;
        let out = self.scratch(1, q_len * width * 4)?;
        let scale = 1.0f32 / ((width / heads) as f32).sqrt();
        let (ql, kl, w, h) = (q_len as i32, kv_len as i32, width as i32, heads as i32);

        let groups = ((q_len as u32) + combine_rows_per_block() - 1) / combine_rows_per_block();
        let groups_q = ((q_len as u32) + (MHA_ROWS_PER_BLOCK as u32) - 1) / (MHA_ROWS_PER_BLOCK as u32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            let pk = module.kernel(if head_dim / 32 == 2 {
                "mha_partial_kernel_pl2"
            } else {
                "mha_partial_kernel"
            })?;
            for (c, &(base, rows)) in chunks.iter().enumerate() {
                if rows == 0 {
                    continue;
                }
                let (k, v) = build(self, c, rows)?;
                let dst = self.view(&partial, c * per, per);

                if crate::cuda::att_partial_mma() && kv_len > 8192 {
                    let kq = self.to_bf16_dev(&q, q_len * width)?;

                    let elems = rows * width;
                    let kv_at = self.view(&k, 0, elems * 4);
                    let vv_at = self.view(&v, 0, elems * 4);
                    let kb = self.to_bf16_dev(&kv_at, elems)?;
                    let vb = self.to_bf16_dev(&vv_at, elems)?;

                    let real = rows.min(kv_len.saturating_sub(base));

                    self.attention_partial_dev(&kq, &kb, &vb, &dst, q_len, real, heads, head_dim, 0)?;
                } else {
                self.stream
                    .launch_builder(&pk.function)
                    .arg(&q.slice()).arg(&k.slice()).arg(&v.slice()).arg(&dst.slice())
                    .arg(&ql).arg(&kl).arg(&w).arg(&h).arg(&scale).arg(&(base as i32))
                    .arg(&kv_real.slice())

                    .arg(&(rows as i32))
                    .launch(config_grid_3d(heads as u32, groups_q, 1, 128))
                    .context("launching mha partial (ladder chunk)")?;
                }
            }
            let ck = module.kernel("mha_combine_kernel")?;
            self.stream
                .launch_builder(&ck.function)
                .arg(&partial.slice()).arg(&out.slice())
                .arg(&ql).arg(&w).arg(&h).arg(&(n as i32))
                .launch(config_3d(heads as u32, groups, 128))
                .context("launching mha combine (ladder chunk)")?;
        }
        Ok(out)
    }

    #[allow(dead_code)]
    pub fn mha_split_chunked_dev<F>(&self, q: &Buffer, q_len: usize, kv_len: usize,
                                    kv_real: &Buffer, width: usize,
                                    heads: usize, mut build: F) -> Result<Buffer>
    where
        F: FnMut(&Self, usize, usize) -> Result<(Buffer, Buffer)>,
    {
        let module = self.load(HEAD_CUBIN, "head")?;
        let head_dim = width / heads;
        let kv_splits = KV_SPLITS;
        let per = q_len * heads * (head_dim + 2) * 4;
        let partial = self.alloc(kv_splits * per)?;
        let out = self.alloc(q_len * width * 4)?;
        let scale = 1.0f32 / ((width / heads) as f32).sqrt();
        let per_split = kv_len.div_ceil(kv_splits);
        let (ql, kl, w, h) = (q_len as i32, kv_len as i32, width as i32, heads as i32);

        let groups = ((q_len as u32) + combine_rows_per_block() - 1) / combine_rows_per_block();
        let groups_q = ((q_len as u32) + (MHA_ROWS_PER_BLOCK as u32) - 1) / (MHA_ROWS_PER_BLOCK as u32);
        unsafe {
            use cudarc::driver::PushKernelArg;

            let pk = module.kernel(if head_dim / 32 == 2 {
                "mha_partial_kernel_pl2"
            } else {
                "mha_partial_kernel"
            })?;
            for split in 0..kv_splits {
                let rows = per_split.min(kv_len.saturating_sub(split * per_split));
                if rows == 0 {
                    continue;
                }
                let (k, v) = build(self, split, rows)?;
                let dst = self.view(&partial, split * per, per);
                let base = (split * per_split) as i32;
                self.stream
                    .launch_builder(&pk.function)
                    .arg(&q.slice()).arg(&k.slice()).arg(&v.slice()).arg(&dst.slice())
                    .arg(&ql).arg(&kl).arg(&w).arg(&h).arg(&scale).arg(&base).arg(&kv_real.slice())

                    .launch(config_grid_3d(heads as u32, groups_q, 1, 128))
                    .context("launching mha partial (split-chunked)")?;
            }
            let ck = module.kernel("mha_combine_kernel")?;
            self.stream
                .launch_builder(&ck.function)
                .arg(&partial.slice()).arg(&out.slice())
                .arg(&ql).arg(&w).arg(&h).arg(&(kv_splits as i32))
                .launch(config_3d(heads as u32, groups, 128))
                .context("launching mha combine (split-chunked)")?;
        }
        Ok(out)
    }

    pub fn mha_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, q_len: usize, kv_len: usize,
                   kv_real: &Buffer, width: usize, heads: usize) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;

        let kv_splits = KV_SPLITS;
        let head_dim = width / heads;
        let partial = self.alloc(kv_splits * q_len * heads * (head_dim + 2) * 4)?;
        let out = self.alloc(q_len * width * 4)?;
        let scale = 1.0f32 / ((width / heads) as f32).sqrt();
        let (ql, kl, w, h) = (q_len as i32, kv_len as i32, width as i32, heads as i32);

        let groups = ((q_len as u32) + combine_rows_per_block() - 1) / combine_rows_per_block();
        let groups_q = ((q_len as u32) + (MHA_ROWS_PER_BLOCK as u32) - 1) / (MHA_ROWS_PER_BLOCK as u32);
        unsafe {
            use cudarc::driver::PushKernelArg;

            let pk = module.kernel(if head_dim / 32 == 2 {
                "mha_partial_kernel_pl2"
            } else {
                "mha_partial_kernel"
            })?;
            self.stream
                .launch_builder(&pk.function)
                .arg(&q.slice())
                .arg(&k.slice())
                .arg(&v.slice())
                .arg(&partial.slice())
                .arg(&ql)
                .arg(&kl)
                .arg(&w)
                .arg(&h)
                .arg(&scale)
                .arg(&0i32)
                .arg(&kv_real.slice())

                .arg(&(kv_len.div_ceil(kv_splits) as i32))
                .launch(config_grid_3d(heads as u32, groups_q, kv_splits as u32, 128))
                .context("launching mha partial")?;
            let ck = module.kernel("mha_combine_kernel")?;
            self.stream
                .launch_builder(&ck.function)
                .arg(&partial.slice())
                .arg(&out.slice())
                .arg(&ql)
                .arg(&w)
                .arg(&h)
                .arg(&(kv_splits as i32))
                .launch(config_3d(heads as u32, groups, 128))
                .context("launching mha combine")?;
        }
        Ok(out)
    }

    pub fn mean_span_dev(&self, x: &Buffer, starts: &[i32], ends: &[i32], cols: usize)
        -> Result<Buffer> {
        assert_eq!(starts.len(), ends.len());
        let n = starts.len();
        let s = self.upload_i32(starts)?;
        let e = self.upload_i32(ends)?;
        self.mean_span_buf_dev(x, &s, &e, n, cols)
    }

    pub fn mean_span_buf_dev(&self, x: &Buffer, starts: &Buffer, ends: &Buffer, n: usize,
                             cols: usize) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("mean_span_kernel")?;
        let out = self.alloc(n * cols * 4)?;
        let (ni, ci) = (n as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&starts.slice())
                .arg(&ends.slice())
                .arg(&out.slice())
                .arg(&ni)
                .arg(&ci)
                .launch(config(n as u32, 256))
                .context("launching mean_span")?;
        }
        Ok(out)
    }

    pub fn l2_normalise_dev(&self, x: &Buffer, rows: usize, cols: usize) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("l2_normalise_kernel")?;
        let mut out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&out.slice())
                .arg(&x.slice())
                .arg(&r)
                .arg(&c)
                .launch(config(rows as u32, 256))
                .context("launching l2_normalise")?;
        }
        Ok(out)
    }

    pub fn summary_softmax_dev(&self, routed: &Buffer, fields: &Buffer, counts: &[i32],
                               width: usize) -> Result<Buffer> {
        let n = counts.len();
        let c = self.upload_i32(counts)?;
        self.summary_softmax_buf_dev(routed, fields, &c, n, width)
    }

    pub fn summary_softmax_buf_dev(&self, routed: &Buffer, fields: &Buffer, counts: &Buffer,
                                   n: usize, width: usize) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("summary_softmax_kernel")?;
        let out = self.alloc(n * width * 4)?;
        let (nq, w) = (n as i32, width as i32);
        let scale = 1.0f32 / (width as f32).sqrt();
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&routed.slice())
                .arg(&fields.slice())
                .arg(&counts.slice())
                .arg(&out.slice())
                .arg(&nq)
                .arg(&w)
                .arg(&scale)
                .launch(config(n as u32, 256))
                .context("launching summary_softmax")?;
        }
        Ok(out)
    }

    pub fn head_pair_dot_dev(&self, a: &Buffer, b: &Buffer, idx: &[i32], rows: usize, cols: usize,
                             scale: f32) -> Result<Buffer> {
        assert_eq!(idx.len(), rows);
        let i = self.upload_i32(idx)?;
        self.head_pair_dot_buf_dev(a, b, &i, rows, cols, scale)
    }

    pub fn head_pair_dot_buf_dev(&self, a: &Buffer, b: &Buffer, idx: &Buffer, rows: usize,
                                 cols: usize, scale: f32) -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("head_pair_dot_kernel")?;
        let out = self.alloc(rows * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&a.slice())
                .arg(&b.slice())
                .arg(&idx.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&scale)
                .launch(config(rows as u32, 256))
                .context("launching head_pair_dot")?;
        }
        Ok(out)
    }

    pub fn head_features_dev(&self, field: &Buffer, option: &Buffer, rows: usize, width: usize)
        -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("head_features_kernel")?;
        let out = self.alloc(rows * 4 * width * 4)?;
        let (r, w) = (rows as i32, width as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&field.slice())
                .arg(&option.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&w)

                .launch(config_split(4, rows.min(65535) as u32, rows.div_ceil(65535) as u32, 256))
                .context("launching head_features")?;
        }
        Ok(out)
    }

    pub fn accumulate_indexed_dev(&self, dst: &mut Buffer, src: &Buffer, idx: &[i32], rows: usize,
                                  cols: usize) -> Result<()> {
        assert_eq!(idx.len(), rows);

        if let Some(&max) = idx.iter().max() {
            let needed = (max.max(0) as usize + 1) * cols * 4;
            assert!(
                src.len() >= needed,
                "accumulate_indexed: src is {} bytes but idx reaches row {max} of a {cols}-wide row, \
                 which needs {needed}",
                src.len()
            );
        }
        let i = self.upload_i32(idx)?;
        self.accumulate_indexed_buf_dev(dst, src, &i, rows, cols)
    }

    pub fn accumulate_indexed_buf_dev(&self, dst: &mut Buffer, src: &Buffer, idx: &Buffer,
                                      rows: usize, cols: usize) -> Result<()> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("accumulate_indexed_kernel")?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&dst.slice())
                .arg(&src.slice())
                .arg(&idx.slice())
                .arg(&r)
                .arg(&c)
                .launch(config_split(((cols as u32) + 255) / 256, rows.min(65535) as u32, rows.div_ceil(65535) as u32, 256))
                .context("launching accumulate_indexed")?;
        }
        Ok(())
    }

    pub fn gather_rows_dev(&self, src: &Buffer, idx: &[i32], rows: usize, cols: usize)
        -> Result<Buffer> {
        assert_eq!(idx.len(), rows);

        if let Some(&max) = idx.iter().max() {
            let needed = (max.max(0) as usize + 1) * cols * 4;
            assert!(
                src.len() >= needed,
                "gather_rows: src is {} bytes but idx reaches row {max} of a {cols}-wide row, which \
                 needs {needed}",
                src.len()
            );
        }
        let i = self.upload_i32(idx)?;
        self.gather_rows_buf_dev(src, &i, rows, cols)
    }

    pub fn gather_rows_buf_dev(&self, src: &Buffer, idx: &Buffer, rows: usize, cols: usize)
        -> Result<Buffer> {
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("gather_rows_kernel")?;
        let out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&idx.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .launch(config_split(((cols as u32) + 255) / 256, rows.min(65535) as u32, rows.div_ceil(65535) as u32, 256))
                .context("launching gather_rows")?;
        }
        Ok(out)
    }

    pub fn rms_norm_quantise_nvfp4_dev(&self, x: &Buffer, weight: &Buffer, rows: usize, cols: usize,
                                       eps: f32, global_scale: f32, tile_m: usize)
        -> Result<Option<(Buffer, Buffer)>> {
        if cols != 256 * 16 {
            return Ok(None);
        }
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("rms_norm_quantise_nvfp4_kernel")?;
        let blk_mn = crate::swizzle::sf_atom_in_force(tile_m);
        let chunk = crate::swizzle::sf_chunk_stride_in_force(tile_m);
        let rows_padded = crate::swizzle::round_up_public(rows, blk_mn);

        let packed = self.alloc_zeroed(rows_padded * (cols / 2))?;
        let scales = self.alloc_zeroed(rows_padded * (cols / 16))?;
        let (r, c) = (rows as i32, cols as i32);
        let (bm, ch) = (blk_mn as i32, chunk as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&weight.slice())
                .arg(&packed.slice())
                .arg(&scales.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)
                .arg(&global_scale)
                .arg(&bm)
                .arg(&ch)
                .launch(config(rows_padded as u32, 256))
                .context("launching rms_norm_quantise_nvfp4_dev")?;
        }
        Ok(Some((packed, scales)))
    }

    pub fn quantise_nvfp4_dev(
        &self,
        x: &Buffer,
        rows: usize,
        cols: usize,
        global_scale: f32,
        tile_m: usize,
    ) -> Result<(Buffer, Buffer)> {
        assert_eq!(cols % 16, 0, "NVFP4 scales cover 16 values");
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("quantise_nvfp4_kernel")?;

        let blk_mn = crate::swizzle::sf_atom_in_force(tile_m);
        let chunk = crate::swizzle::sf_chunk_stride_in_force(tile_m);

        let rows_padded = crate::swizzle::round_up_public(rows, blk_mn);

        let packed = self.alloc_zeroed(rows_padded * (cols / 2))?;
        let scales = self.alloc_zeroed(rows_padded * (cols / 16))?;
        let (r, c) = (rows as i32, cols as i32);
        let blocks = (cols / 16) as u32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&packed.slice())
                .arg(&scales.slice())
                .arg(&r)
                .arg(&c)
                .arg(&global_scale)
                .arg(&(blk_mn as i32))
                .arg(&(chunk as i32))
                .launch(config_split((blocks + 63) / 64, rows_padded.min(65535) as u32,
                                     rows_padded.div_ceil(65535) as u32, 64))
                .context("launching quantise_nvfp4_dev")?;
        }
        Ok((packed, scales))
    }

    pub fn make_tensor_map_2d(&self, ptr: u64, inner_bytes: u64, rows: u64,
                              box_inner: u32, box_rows: u32) -> Result<TensorMap> {
        self.make_tensor_map_2d_sw(ptr, inner_bytes, rows, box_inner, box_rows,
            cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_128B)
    }

    pub fn make_tensor_map_2d_sw(&self, ptr: u64, inner_bytes: u64, rows: u64,
                                 box_inner: u32, box_rows: u32,
                                 swizzle: cudarc::driver::sys::CUtensorMapSwizzle_enum)
        -> Result<TensorMap> {
        let mut map = cudarc::driver::sys::CUtensorMap { opaque: [0u64; 16] };
        let global_dim: [u64; 2] = [inner_bytes, rows];
        let global_strides: [u64; 1] = [inner_bytes];
        let box_dim: [u32; 2] = [box_inner, box_rows];
        let element_strides: [u32; 2] = [1, 1];
        let res = unsafe {
            cudarc::driver::sys::cuTensorMapEncodeTiled(
                &mut map,
                cudarc::driver::sys::CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                2,
                ptr as *mut std::ffi::c_void,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                cudarc::driver::sys::CUtensorMapInterleave_enum::CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle,
                cudarc::driver::sys::CUtensorMapL2promotion_enum::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                cudarc::driver::sys::CUtensorMapFloatOOBfill_enum::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            bail!("cuTensorMapEncodeTiled returned {res:?}");
        }
        Ok(TensorMap(map.opaque))
    }

    pub fn quantise_fp8_dev(&self, x: &Buffer, rows: usize, cols: usize)
        -> Result<(Buffer, Buffer)> {
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("quantise_fp8_kernel")?;
        let out = self.alloc(rows * cols)?;
        let scales = self.alloc(rows * 2)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&out.slice())
                .arg(&scales.slice())
                .arg(&r)
                .arg(&c)
                .launch(config(row_blocks(rows, cols), 256))
                .context("launching quantise_fp8_dev")?;
        }
        Ok((out, scales))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_gated_quantise_nvfp4_dev(&self, x: &Buffer, gate: &Buffer, weight: &Buffer,
                                             rows: usize, cols: usize, vh: usize, eps: f32,
                                             global_scale: f32, tile_m: usize)
        -> Result<(Buffer, Buffer)> {
        assert_eq!(cols, 128, "the fused gated norm is built for a 128-wide head");
        assert_eq!(cols % 16, 0, "NVFP4 scales cover 16 values");
        let flat_cols = vh * cols;
        let blk_mn = crate::swizzle::sf_atom_in_force(tile_m);
        let chunk = crate::swizzle::sf_chunk_stride_in_force(tile_m);
        let tokens = rows / vh;

        let tokens_padded = crate::swizzle::round_up_public(tokens, blk_mn);
        let rows_padded = tokens_padded * vh;
        let packed = self.alloc_zeroed(rows_padded * (cols / 2))?;
        let scales = self.alloc_zeroed(tokens_padded * (flat_cols / 16))?;
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("rms_norm_gated_quantise_nvfp4_kernel")?;
        let (r, c) = (rows_padded as i32, cols as i32);
        let (bm, ch, v) = (blk_mn as i32, chunk as i32, vh as i32);
        let live = rows as i32;
        let blocks = ((rows_padded + 7) / 8) as u32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice()).arg(&gate.slice()).arg(&weight.slice())
                .arg(&packed.slice()).arg(&scales.slice())
                .arg(&r).arg(&c).arg(&eps).arg(&global_scale)
                .arg(&bm).arg(&ch).arg(&v).arg(&live)
                .launch(config(blocks, 256))
                .context("launching rms_norm_gated_quantise_nvfp4")?;
        }
        Ok((packed, scales))
    }

    pub fn rms_norm_gated_quantise_fp8_dev(&self, x: &Buffer, gate: &Buffer, weight: &Buffer,
                                           rows: usize, cols: usize, vh: usize, eps: f32)
        -> Result<(Buffer, Buffer)> {
        assert_eq!(cols, 128, "the fused gated norm is built for a 128-wide head");
        let tokens = rows / vh;
        let flat_cols = vh * cols;
        let out = self.alloc(rows * cols)?;
        let scales = self.alloc(tokens * 2)?;
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("rms_norm_gated_quantise_fp8_kernel")?;
        let (r, c) = (rows as i32, cols as i32);
        let (v, live) = (vh as i32, rows as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice()).arg(&gate.slice()).arg(&weight.slice())
                .arg(&out.slice()).arg(&scales.slice())
                .arg(&r).arg(&c).arg(&eps).arg(&v).arg(&live)
                .launch(config(tokens as u32, 1024))
                .context("launching rms_norm_gated_quantise_fp8")?;
        }
        let _ = flat_cols;
        Ok((out, scales))
    }

    pub fn rms_norm_fp8_bf16_dev(&self, x: &Buffer, weight: &Buffer, rows: usize, cols: usize,
                                 eps: f32) -> Option<(Buffer, Buffer, Buffer)> {
        if cols != 4096 {
            return None;
        }
        let module = self.load(QUANTISE_CUBIN, "quantise").ok()?;
        let kernel = module.kernel("rms_norm_fp8_bf16_kernel").ok()?;
        let q = self.alloc(rows * cols).ok()?;
        let qs = self.alloc(rows * 2).ok()?;
        let bf = self.alloc(rows * cols * 2).ok()?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&weight.slice())
                .arg(&q.slice())
                .arg(&qs.slice())
                .arg(&bf.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)
                .launch(config(rows as u32, 256))
                .ok()?;
        }
        Some((q, qs, bf))
    }

    pub fn swiglu_quantise_nvfp4_dev(&self, gate: &Buffer, up: &Buffer, rows: usize, cols: usize,
                                     global_scale: f32, tile_m: usize)
        -> Result<Option<(Buffer, Buffer)>> {
        assert_eq!(cols % 16, 0, "NVFP4 scales cover 16 values");

        if cols > 768 * 16 {
            return Ok(None);
        }
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("swiglu_quantise_nvfp4_kernel")?;

        let blk_mn = crate::swizzle::sf_atom_in_force(tile_m);
        let chunk = crate::swizzle::sf_chunk_stride_in_force(tile_m);
        let rows_padded = crate::swizzle::round_up_public(rows, blk_mn);

        let packed = self.alloc_zeroed(rows_padded * (cols / 2))?;

        let scales = self.alloc_zeroed(rows_padded * (cols / 16))?;
        let (r, c) = (rows as i32, cols as i32);
        let per_row = (cols / 16) as u32;
        let threads = per_row.clamp(32, 768);
        let grid_x = per_row.div_ceil(threads);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&gate.slice())
                .arg(&up.slice())
                .arg(&packed.slice())
                .arg(&scales.slice())
                .arg(&r)
                .arg(&c)
                .arg(&global_scale)
                .arg(&(blk_mn as i32))
                .arg(&(chunk as i32))

                .launch(config_split(grid_x, rows_padded.min(65535) as u32,
                                     rows_padded.div_ceil(65535) as u32, threads))
                .context("launching swiglu_quantise_nvfp4_dev")?;
        }
        Ok(Some((packed, scales)))
    }

    pub fn swiglu_quantise_fp8_dev(&self, gate: &Buffer, up: &Buffer, rows: usize, cols: usize)
        -> Result<Option<(Buffer, Buffer)>> {
        let module = self.load(QUANTISE_CUBIN, "quantise")?;
        let kernel = module.kernel("swiglu_quantise_fp8_kernel")?;

        if cols > 768 * 16 {
            return Ok(None);
        }
        let out = self.alloc(rows * cols)?;
        let scales = self.alloc(rows * 2)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&gate.slice())
                .arg(&up.slice())
                .arg(&out.slice())
                .arg(&scales.slice())
                .arg(&r)
                .arg(&c)
                .launch(config(rows as u32, 768))
                .context("launching swiglu_quantise_fp8_dev")?;
        }
        Ok(Some((out, scales)))
    }

    pub fn fp8_gemm_dev(&self, a: &Buffer, b: &Buffer, sa: &Buffer, sb: &Buffer, m: usize,
                        n: usize, k: usize, add: Option<&Buffer>) -> Result<Buffer> {
        self.fp8_gemm_dev_level(a, b, sa, sb, m, n, k, add, None)
    }

    pub fn fp8_gemm_dev_level(&self, a: &Buffer, b: &Buffer, sa: &Buffer, sb: &Buffer, m: usize,
                              n: usize, k: usize, add: Option<&Buffer>,
                              forced: Option<Fp8Level>) -> Result<Buffer> {
        let v = match forced {
            Some(l) => fp8_variant_for_level(l),
            None => fp8_variant(m, n, k),
        };

        if env_once!("CLEF_FP8_LOG").is_some() {
            eprintln!("  fp8_gemm_dev: m={m} n={n} k={k} variant={}", v.name);
        }
        let mut bad = crate::model::fp8_cubin_matches_launch(
            &v.facts, v.facts.m_tile, v.facts.n_block, v.facts.stages, 384);

        let padded = m.div_ceil(16) * 16;
        let work = padded.div_ceil(v.facts.m_tile) * n.div_ceil(v.facts.n_block);
        let sms = self.sm_count();
        bad.extend(crate::model::fp8_violations(&crate::model::Fp8Side {
            grid_x: work.min(sms * v.ctas_per_sm), grid_y: 1, m: padded, n,
            ctas_per_sm: v.ctas_per_sm, sms,
            ..v.side
        }));
        if !bad.is_empty() {
            anyhow::bail!(
                "the FP8 GEMM launch is inconsistent, and would HANG rather than fail:\n  - {}",
                bad.join("\n  - ")
            );
        }
        self.fp8_gemm_dev_tile(v, a, b, sa, sb, m, n, k, add)
    }

    fn fp8_gemm_dev_atom(&self, plain: bool, a: &Buffer, b: &Buffer, sa: &Buffer,
                         sb: &Buffer, m: usize, n: usize, k: usize, add: Option<&Buffer>)
        -> Result<Buffer> {
        self.fp8_gemm_dev_tile(fp8_variant_with(m, n, k, false, plain), a, b, sa, sb, m, n, k, add)
    }

    #[allow(clippy::too_many_arguments)]
    fn fp8_gemm_dev_tile(&self, v: Fp8Variant, a: &Buffer, b: &Buffer,
                         sa: &Buffer, sb: &Buffer, m: usize, n: usize, k: usize, add: Option<&Buffer>)
        -> Result<Buffer> {
        let padded_m = fp8_padded_m(m, v.facts.m_tile);
        let out = self.alloc(padded_m * n * 4)?;
        self.fp8_gemm_dev_tile_into(v, a, b, sa, sb, m, n, k, add, &out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn fp8_gemm_dev_tile_into(&self, v: Fp8Variant, a: &Buffer, b: &Buffer,
                              sa: &Buffer, sb: &Buffer, m: usize, n: usize, k: usize,
                              add: Option<&Buffer>, out: &Buffer) -> Result<()> {
        let module = self.load(v.cubin, v.name)?;
        let kernel = module.kernel("fp8_gemm")?;
        let n_block = v.facts.n_block;
        let ctas_per_sm = v.ctas_per_sm;
        let padded_m = fp8_padded_m(m, v.facts.m_tile);

        assert!(out.len() >= padded_m * n * 4,
                "the caller's buffer must hold at least {padded_m} x {n} floats ({}), got {}",
                padded_m * n * 4, out.len());

        let b_bytes = n * k;

        if env_once!("CLEF_L2_PERSIST") == Some("1") {
            let carve = b_bytes.min(self.max_persisting_l2());
            self.set_persisting_l2(carve)?;
            self.set_l2_window(b, b_bytes, 1.0)?;
        }

        let epi_sw = fp8_epi_swizzle(n);
        let map_c = if n % 4 == 0 {
            self.make_tensor_map_2d_sw(
                out.device_ptr() as u64, (n * 4) as u64, padded_m as u64, FP8_EPI_N * 4,
                v.facts.m_tile as u32, epi_sw)?
        } else {
            TensorMap([0u64; 16])
        };
        let (mi, ni, ki) = (padded_m as i32, n as i32, k as i32);
        let (box_inner, box_sw) = fp8_box(&v.facts);
        let map_a = self.make_tensor_map_2d_sw(
            a.device_ptr() as u64, k as u64, m as u64, box_inner, v.facts.m_tile as u32,
            box_sw)?;
        let map_b = self.make_tensor_map_2d_sw(
            b.device_ptr() as u64, k as u64, n as u64, box_inner, n_block as u32,
            box_sw)?;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&map_a)
                .arg(&map_b)
                .arg(&a.slice())
                .arg(&b.slice())
                .arg(&sa.slice())
                .arg(&sb.slice())
                .arg(&out.slice())
                .arg(&add.map(|r| r.device_ptr()).unwrap_or(0u64))
                .arg(&map_c)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)

                .arg(&(m as i32))
                .arg(&(m as i32))

                .arg(&fp8_log_swizzle(padded_m))

                .launch({
                    let work = padded_m.div_ceil(v.facts.m_tile) * n.div_ceil(n_block);
                    config_3d(work.min(self.sm_count() * ctas_per_sm) as u32, 1, 384)
                })
                .context("launching the device FP8 GEMM")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemm_gated_dev(&self, a: &Buffer, a_scales: &Buffer,
                              bg: &Buffer, bg_scales: &Buffer,
                              bu: &Buffer, bu_scales: &Buffer,
                              m: usize, n: usize, k: usize) -> Result<Buffer> {
        let facts = crate::model::CubinFacts { stages: 2, ..crate::model::CubinFacts::FP8 };
        let (m_tile, n_block) = (facts.m_tile, facts.n_block);
        let module = self.load(FP8GEMM_GATED_CUBIN, "fp8gemm_gated")?;
        let kernel = module.kernel("fp8_gemm")?;
        let padded_m = fp8_padded_m(m, m_tile);
        let out = self.alloc(padded_m * n * 4)?;
        let bad = crate::model::fp8_cubin_matches_launch(&facts, m_tile, n_block, 2, 384);
        if !bad.is_empty() {
            anyhow::bail!("the gated FP8 GEMM launch is inconsistent:\n  - {}", bad.join("\n  - "));
        }
        let (box_inner, box_sw) = fp8_box(&facts);
        let map_a = self.make_tensor_map_2d_sw(
            a.device_ptr() as u64, k as u64, m as u64, box_inner, m_tile as u32, box_sw)?;
        let map_bg = self.make_tensor_map_2d_sw(
            bg.device_ptr() as u64, k as u64, n as u64, box_inner, n_block as u32, box_sw)?;
        let map_bu = self.make_tensor_map_2d_sw(
            bu.device_ptr() as u64, k as u64, n as u64, box_inner, n_block as u32, box_sw)?;

        assert_eq!(n % 4, 0, "the gated FP8 epilogue stores through a TMA descriptor");
        let map_c = self.make_tensor_map_2d_sw(
            out.device_ptr() as u64, (n * 4) as u64, padded_m as u64, FP8_EPI_N * 4,
            m_tile as u32, fp8_epi_swizzle(n))?;
        let (mi, ni, ki) = (padded_m as i32, n as i32, k as i32);
        let rows = m as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&map_a)
                .arg(&map_bg)
                .arg(&map_bu)
                .arg(&a.slice())
                .arg(&bg.slice())
                .arg(&a_scales.slice())
                .arg(&bg_scales.slice())
                .arg(&bu_scales.slice())
                .arg(&out.slice())
                .arg(&0u64)
                .arg(&map_c)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&rows)
                .arg(&rows)
                .arg(&fp8_log_swizzle(padded_m))
                .launch({
                    let work = padded_m.div_ceil(m_tile) * n.div_ceil(n_block);
                    config_3d(work.min(self.sm_count()) as u32, 1, 384)
                })
                .context("launching the gated FP8 GEMM")?;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_gemm_dev(&self, a_packed: &Buffer, a_scales: &Buffer, b_packed: &Buffer,
                          b_scales: &Buffer, m: usize, n: usize, k: usize, global_a: f32,
                          global_b: f32) -> Result<Buffer> {
        let padded_m = m.div_ceil(16) * 16;

        let n_splits = 1usize;

        let out = self.alloc(padded_m * n * 4)?;

        if let Some(v) = env_once!("CLEF_PAD_FILL") {
            if let Ok(byte) = u8::from_str_radix(v.trim_start_matches("0x"), 16) {
                self.fill_range(&out, m * n * 4, (padded_m - m) * n * 4, byte)?;
            }
        }
        let _ = n_splits;
        self.launch_nvfp4_gemm(a_packed, b_packed, a_scales, b_scales, &out, padded_m, n, k,
                               global_a, global_b, m, n_splits)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_gemm_gated_dev(&self, a_packed: &Buffer, a_scales: &Buffer,
                                bg_packed: &Buffer, bg_scales: &Buffer,
                                bu_packed: &Buffer, bu_scales: &Buffer,
                                rows: usize, cols: usize, k: usize,
                                global_a: f32, global_bg: f32, global_bu: f32,
                                out_global: f32) -> Result<(Buffer, Buffer)> {
        const M_TILE: usize = 128;
        const N_BLOCK: usize = 64;
        const CTA_THREADS: u32 = 384;
        assert_eq!(cols % 64, 0, "the gated tile covers 64 columns");
        assert_eq!(k % 64, 0, "KTILE is 128 and the atom's k-chunk is 64");

        let blk_mn = crate::swizzle::sf_atom_in_force(M_TILE);
        let rows_padded = crate::swizzle::round_up_public(rows, blk_mn);
        let packed = self.alloc_zeroed(rows_padded * (cols / 2))?;
        let scales = self.alloc_zeroed(rows_padded * (cols / 16))?;
        let (cubin, name) = match std::env::var("CLEF_GATED_ABLATE").as_deref() {
            Ok("noepi") => (GEMM_GATED_NOEPI_CUBIN, "gemm_gated_noepi"),
            _ => (GEMM_GATED_CUBIN, "gemm_gated"),
        };
        let module = self.load(cubin, name)?;
        let kernel = module.kernel("nvfp4_gemm_gated")?;
        let map_a = self.make_tensor_map_2d_sw(
            a_packed.device_ptr() as u64, (k / 2) as u64, rows as u64, 64, M_TILE as u32,
            cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_64B)?;
        let map_bg = self.make_tensor_map_2d_sw(
            bg_packed.device_ptr() as u64, (k / 2) as u64, cols as u64, 64, N_BLOCK as u32,
            cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_64B)?;
        let map_bu = self.make_tensor_map_2d_sw(
            bu_packed.device_ptr() as u64, (k / 2) as u64, cols as u64, 64, N_BLOCK as u32,
            cudarc::driver::sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_64B)?;

        let bad = crate::model::cubin_matches_launch(
            &crate::model::CubinFacts::GEMM_GATED, M_TILE, CTA_THREADS as usize, 32, blk_mn);
        if !bad.is_empty() {
            anyhow::bail!("the gated GEMM launch is inconsistent:\n  - {}", bad.join("\n  - "));
        }

        let m_bound = crate::swizzle::round_up_public(rows, M_TILE);
        let work_tiles = ((m_bound / M_TILE) * cols.div_ceil(N_BLOCK)) as u32;
        let sms = self
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(0)
            .max(0) as u32;
        let grid = if sms == 0 { work_tiles } else { work_tiles.min(sms) };
        let (mi, ni, ki) = (m_bound as i32, cols as i32, k as i32);
        let ar = rows as i32;

        let log_swizzle: i32 = match m_bound {
            0..=1023 => 0,
            2048..=3071 => 6,
            3072..=6143 => 0,
            6144..=12287 => 3,
            12288..=24575 => 5,
            _ => 3,
        };
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&map_a)
                .arg(&map_bg)
                .arg(&map_bu)
                .arg(&a_packed.slice())
                .arg(&bg_packed.slice())
                .arg(&bu_packed.slice())
                .arg(&a_scales.slice())
                .arg(&bg_scales.slice())
                .arg(&bu_scales.slice())
                .arg(&packed.slice())
                .arg(&scales.slice())
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&global_a)
                .arg(&global_bg)
                .arg(&global_bu)
                .arg(&ar)
                .arg(&log_swizzle)
                .arg(&out_global)
                .launch(config_split(grid, 1, 1, CTA_THREADS))
                .context("launching the gated NVFP4 GEMM")?;
        }
        Ok((packed, scales))
    }

    pub fn rms_norm_dev(&self, x: &Buffer, weight: &Buffer, rows: usize, cols: usize, eps: f32)
        -> Result<Buffer> {
        assert!(cols % 4 == 0, "the norm kernels read float4, so cols must be a multiple of 4, got {cols}");
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rms_norm_kernel")?;
        let out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&weight.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)
                .launch(config(row_blocks(rows, cols), 256))
                .context("launching rms_norm_dev")?;
        }
        Ok(out)
    }

    pub fn rms_norm_gated_dev(&self, x: &Buffer, gate: &Buffer, weight: &Buffer, rows: usize,
                              cols: usize, eps: f32) -> Result<Buffer> {
        assert!(cols % 4 == 0, "the norm kernels read float4, so cols must be a multiple of 4, got {cols}");
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rms_norm_gated_kernel")?;
        let out = self.alloc(rows * cols * 4)?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&gate.slice())
                .arg(&weight.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&eps)
                .launch(config(row_blocks(rows, cols), 256))
                .context("launching rms_norm_gated_dev")?;
        }
        Ok(out)
    }

    pub fn swiglu_dev(&self, gate: &Buffer, up: &Buffer, total: usize) -> Result<Buffer> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("swiglu_kernel")?;
        let out = self.alloc(total * 4)?;
        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&gate.slice())
                .arg(&up.slice())
                .arg(&out.slice())
                .arg(&n)
                .launch(config(((total as u32) + 1023) / 1024, 256))
                .context("launching swiglu_dev")?;
        }
        Ok(out)
    }

    pub fn causal_conv_silu_dev(&self, x: &Buffer, w: &Buffer, rows: usize, channels: usize,
                                   k: usize, seq_start: Option<&Buffer>,
                                   history: Option<&Buffer>,
                                   slot_map: Option<&Buffer>) -> Result<Buffer> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("causal_conv_silu_kernel")?;
        let out = self.alloc(rows * channels * 4)?;
        let (r, c, kk) = (rows as i32, channels as i32, k as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&w.slice())
                .arg(&out.slice())
                .arg(&r)
                .arg(&c)
                .arg(&kk)

                .arg(&seq_start.map(|b| b.device_ptr()).unwrap_or(0u64))

                .arg(&history.map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&slot_map.map(|b| b.device_ptr()).unwrap_or(0u64))
                .launch(config_split(((channels as u32) + 127) / 128, rows.min(65535) as u32, rows.div_ceil(65535) as u32, 128))
                .context("launching causal_conv_silu_dev")?;
        }
        Ok(out)
    }

    pub fn conv_hist_save_dev(&self, src: &Buffer, hist: &Buffer, saves: &Buffer,
                              n_saves: usize, rows: usize, k: usize, channels: usize) -> Result<()> {
        if n_saves == 0 {
            return Ok(());
        }
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("conv_hist_save_kernel")?;
        let (n, r, kk, c) = (n_saves as i32, rows as i32, k as i32, channels as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&hist.slice())
                .arg(&saves.slice())
                .arg(&n)
                .arg(&r)
                .arg(&kk)
                .arg(&c)
                .launch(config(n_saves as u32, 128))
                .context("launching conv_hist_save_dev")?;
        }
        Ok(())
    }

    pub fn sigmoid_mul_dev(&self, x: &Buffer, gate: &Buffer, total: usize) -> Result<Buffer> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("sigmoid_mul_kernel")?;

        let out = self.alloc(total * 4)?;
        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&out.slice())
                .arg(&x.slice())
                .arg(&gate.slice())
                .arg(&n)
                .launch(config(((total as u32) + 1023) / 1024, 256))
                .context("launching sigmoid_mul_dev")?;
        }
        Ok(out)
    }

    pub fn gdn_gates_dev(&self, a: &Buffer, b: &Buffer, a_log: &Buffer, dt_bias: &Buffer,
                         tokens: usize, heads: usize) -> Result<(Buffer, Buffer)> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("gdn_gates_kernel")?;
        let decay = self.alloc(tokens * heads * 4)?;
        let beta = self.alloc(tokens * heads * 4)?;
        let total = tokens * heads;
        let (t, h) = (tokens as i32, heads as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&a.slice())
                .arg(&b.slice())
                .arg(&a_log.slice())
                .arg(&dt_bias.slice())
                .arg(&decay.slice())
                .arg(&beta.slice())
                .arg(&t)
                .arg(&h)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching gdn_gates_dev")?;
        }
        Ok((decay, beta))
    }

    pub fn split_q_gate_dev(&self, qg: &Buffer, tokens: usize, heads: usize, head_dim: usize)
        -> Result<(Buffer, Buffer)> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("split_q_gate_kernel")?;
        let total = tokens * heads * head_dim;
        let q = self.alloc(total * 4)?;
        let gate = self.alloc(total * 4)?;
        let (t, h, hd) = (tokens as i32, heads as i32, head_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&qg.slice())
                .arg(&q.slice())
                .arg(&gate.slice())
                .arg(&t)
                .arg(&h)
                .arg(&hd)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching split_q_gate_dev")?;
        }
        Ok((q, gate))
    }

    pub fn embed_gather_dev(&self, table: &Buffer, ids: &Buffer, tokens: usize, hidden: usize)
        -> Result<Buffer> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("embed_gather_kernel")?;
        let out = self.alloc(tokens * hidden * 4)?;
        let total = tokens * hidden;
        let (t, h) = (tokens as i32, hidden as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&table.slice())
                .arg(&ids.slice())
                .arg(&out.slice())
                .arg(&t)
                .arg(&h)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching embed_gather_dev")?;
        }
        Ok(out)
    }

    pub fn split_qkv_dev(&self, mixed: &Buffer, tokens: usize, kh: usize, kd: usize, vh: usize,
                         vd: usize) -> Result<(Buffer, Buffer, Buffer)> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("split_qkv_kernel")?;
        let q = self.alloc(tokens * kh * kd * 4)?;
        let k = self.alloc(tokens * kh * kd * 4)?;
        let v = self.alloc(tokens * vh * vd * 4)?;
        let conv_dim = 2 * kh * kd + vh * vd;
        let total = tokens * conv_dim;
        let (t, a, b, c, d) = (tokens as i32, kh as i32, kd as i32, vh as i32, vd as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&mixed.slice())
                .arg(&q.slice())
                .arg(&k.slice())
                .arg(&v.slice())
                .arg(&t)
                .arg(&a)
                .arg(&b)
                .arg(&c)
                .arg(&d)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching split_qkv_dev")?;
        }
        Ok((q, k, v))
    }

    pub fn repeat_interleave_dev(&self, src: &Buffer, tokens: usize, kh: usize, ratio: usize,
                                 dim: usize) -> Result<Buffer> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("repeat_interleave_kernel")?;
        let out_heads = kh * ratio;
        let total = tokens * out_heads * dim;
        let out = self.alloc(total * 4)?;
        let (t, h, r, d) = (tokens as i32, kh as i32, ratio as i32, dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&out.slice())
                .arg(&t)
                .arg(&h)
                .arg(&r)
                .arg(&d)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching repeat_interleave_dev")?;
        }
        Ok(out)
    }

    pub fn rope_tables_dev(&self, positions: &Buffer, rows: usize, rotary_dim: usize, theta: f32)
        -> Result<(Buffer, Buffer)> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rope_tables_kernel")?;
        let cos = self.alloc(rows * rotary_dim * 4)?;
        let sin = self.alloc(rows * rotary_dim * 4)?;
        let total = rows * rotary_dim;
        let (r, rd) = (rows as i32, rotary_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&positions.slice())
                .arg(&cos.slice())
                .arg(&sin.slice())
                .arg(&r)
                .arg(&rd)
                .arg(&theta)
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching rope_tables_dev")?;
        }
        Ok((cos, sin))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_dev(&self, x: &Buffer, weight: &Buffer, cos: &Buffer, sin: &Buffer,
                            rows: usize, heads: usize, head_dim: usize, rotary_dim: usize,
                            eps: f32, seq_start: Option<&Buffer>, pos_base: Option<&Buffer>,
                            pos_map: Option<&Buffer>, with_gate: bool)
                            -> Result<(Buffer, Option<Buffer>)> {
        assert_eq!(head_dim, 256, "qk_norm_rope_kernel is built for head_dim = 256");
        assert!(rotary_dim <= head_dim && head_dim % 4 == 0);
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("qk_norm_rope_kernel")?;
        let out = self.alloc(rows * head_dim * 4)?;
        let gate = if with_gate { Some(self.alloc(rows * head_dim * 4)?) } else { None };
        let (r, h, hd, rd) = (rows as i32, heads as i32, head_dim as i32, rotary_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&x.slice())
                .arg(&weight.slice())
                .arg(&cos.slice())
                .arg(&sin.slice())
                .arg(&out.slice())
                .arg(&gate.as_ref().map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&r)
                .arg(&h)
                .arg(&hd)
                .arg(&rd)
                .arg(&eps)
                .arg(&seq_start.map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&pos_base.map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&pos_map.map(|b| b.device_ptr()).unwrap_or(0u64))

                .launch(config(((rows as u32) + 7) / 8, 256))
                .context("launching qk_norm_rope_dev")?;
        }
        Ok((out, gate))
    }

    pub fn rope_apply_dev(&self, x: &Buffer, cos: &Buffer, sin: &Buffer, rows: usize,
                          heads: usize, head_dim: usize, rotary_dim: usize,
                          seq_start: Option<&Buffer>, pos_base: Option<&Buffer>,
                          pos_map: Option<&Buffer>) -> Result<Buffer> {
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("rope_kernel")?;
        let out = self.alloc(rows * heads * head_dim * 4)?;

        let total = rows * heads * (head_dim / 2);
        let (r, h, hd, rd) = (rows as i32, heads as i32, head_dim as i32, rotary_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&out.slice())
                .arg(&x.slice())
                .arg(&cos.slice())
                .arg(&sin.slice())
                .arg(&r)
                .arg(&h)
                .arg(&hd)
                .arg(&rd)
                .arg(&seq_start.map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&pos_base.map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&pos_map.map(|b| b.device_ptr()).unwrap_or(0u64))
                .launch(config(((total as u32) + 255) / 256, 256))
                .context("launching rope_apply_dev")?;
        }
        Ok(out)
    }

    pub fn attention_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, seq: usize, heads: usize,
                         kv_heads: usize, head_dim: usize, map: BatchMap<'_>) -> Result<Buffer> {
        let cache = map.carry.and_then(|c| c.kv);
        if cache.is_some() {
            return self.attention_dev_mma(q, k, v, seq, heads, kv_heads, head_dim, map.seq_start,
                                          map.seq_end, map.real_end, map.real_end_dev, cache);
        }

        if att_mma_for(seq) {
            return self.attention_dev_mma(q, k, v, seq, heads, kv_heads, head_dim, map.seq_start,
                                          map.seq_end, map.real_end, map.real_end_dev, None);
        }

        let rows = if att_wide() { 100 } else { att_rows_for(seq) };
        self.attention_dev_rows(q, k, v, seq, heads, kv_heads, head_dim, rows, map.seq_start)
    }

    fn attention_dev_mma(&self, q: &Buffer, k: &Buffer, v: &Buffer, seq: usize, heads: usize,
                         kv_heads: usize, head_dim: usize,
                         seq_start: Option<&Buffer>, seq_end: Option<&Buffer>,
                         seq_end_scalar: usize, real_end_dev: Option<&Buffer>,
                         cache: Option<KvCache<'_>>) -> Result<Buffer> {
        let bk64 = env_once!("CLEF_ATT_BK") == Some("64w")
            || env_once!("CLEF_ATT_BK") == Some("64q")
            || env_once!("CLEF_ATT_BK")
                .and_then(|v| v.parse::<usize>().ok()).unwrap_or(16) >= 64;

        let hd128 = env_once!("CLEF_ATT_BK") == Some("64q");

        let gqa_pack = att_gqa();

        let gqa_rows = 128usize.div_ceil((heads / kv_heads).max(1)).max(1);
        let bk64w = env_once!("CLEF_ATT_BK") == Some("64w");

        let bk32 = att_bk32();

        let widest = crate::gpu::kv_slots().max(1) * crate::gpu::max_chunk_size();
        let hint = splits_hint();
        let splits = if hint > 0 { hint } else if seq <= widest { att_keys() } else { 1 };
        let module = if att_qreg96() {
            self.load(ATTENTION_MMA_QREG96_CUBIN, "attention_mma_qreg96")?
        } else if att_ws() && att_noskip() {
            self.load(ATTENTION_MMA_WS_NOSKIP_CUBIN, "attention_mma_ws_noskip")?
        } else if att_ws() && splits > 1 {
            self.load(ATTENTION_MMA_WS_SPLIT_CUBIN, "attention_mma_ws_split")?
        } else if att_ws() && att_r240() {
            self.load(ATTENTION_MMA_WS_R240_CUBIN, "attention_mma_ws_r240")?
        } else if att_ws() && att_bk48() {
            self.load(ATTENTION_MMA_WS_BK48_CUBIN, "attention_mma_ws_bk48")?
        } else if att_ws() {
            self.load(ATTENTION_MMA_WS_CUBIN, "attention_mma_ws")?
        } else if att_qreg() {
            self.load(ATTENTION_MMA_QREG_CUBIN, "attention_mma_qreg")?
        } else if gqa_pack {
            self.load(ATTENTION_MMA_GQA_CUBIN, "attention_mma_gqa")?
        } else if bk32 && att_gqa() {
            self.load(ATTENTION_MMA_GQA_BK32_CUBIN, "attention_mma_gqa_bk32")?
        } else if bk32 {
            self.load(ATTENTION_MMA_BK32_CUBIN, "attention_mma_bk32")?
        } else if hd128 {
            self.load(ATTENTION_MMA_HD128_CUBIN, "attention_mma_hd128")?
        } else if bk64w {
            self.load(ATTENTION_MMA_BK64W_CUBIN, "attention_mma_bk64w")?
        } else if bk64 {
            self.load(ATTENTION_MMA_BK64_CUBIN, "attention_mma_bk64")?
        } else {
            self.load(attention_cubin_for(seq, true), attention_module_name_for(seq, true))?
        };
        let kernel = module.kernel("attention_kernel")?;

        let out = self.alloc(seq * heads * head_dim * 4)?;

        if !self.capturing.load(std::sync::atomic::Ordering::SeqCst) {
            self.fill_nan(&out)?;
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (s, h, kh, hd) = (seq as i32, heads as i32, kv_heads as i32, head_dim as i32);
        let row_blocks = if gqa_pack { (seq.div_ceil(gqa_rows)) as u32 }
                         else if hd128 { rows_to_blocks_for(seq, att_rows_for_mma()) }
                         else if bk64 { ((seq + 63) / 64) as u32 }
                         else { rows_to_blocks_for(seq, att_rows_for_mma()) };
        if matches!(env_once!("CLEF_ATT_LAUNCH_TRACE"), Some(v) if v != "0") {
            let gqa = (heads / kv_heads).max(1);
            let rows_per_block = if gqa_pack { gqa_rows } else { 2 * att_rows_for_mma() as usize };
            let covers = rows_per_block * row_blocks as usize;
            eprintln!("ATT launch seq={seq} heads={heads} kv_heads={kv_heads} head_dim={head_dim} \
                       ws={} gqa_pack={gqa_pack} bk64={bk64} splits={splits} gqa={gqa} \
                       rows_per_block={rows_per_block} blocks={row_blocks} covers={covers} \
                       out_elems={}", att_ws(), seq * heads * head_dim);
        }

        let warps = 8u32;

        let threads = if att_ws() { 384 }
                      else if hd128 { (8 * att_ncg() / att_mt() * 32) as u32 }
                      else if bk64w { 256 } else if bk64 { 128 }
                      else { (warps * att_ncg() as u32 / att_mt() as u32 * 32) };

        let conv = module.kernel("to_bf16_kernel")?;

        let scatter = module.kernel("to_bf16_scatter_kernel")?;
        let q16 = self.alloc(seq * heads * head_dim * 2)?;

        if let Some(c) = &cache {
            if c.positions < seq {
                anyhow::bail!("the KV cache holds {} tokens and the pass is {seq}", c.positions);
            }
        }

        let owned_kv = if cache.is_none() {
            Some((self.alloc(seq * kv_heads * head_dim * 2)?,
                  self.alloc(seq * kv_heads * head_dim * 2)?))
        } else {
            None
        };
        let (k16, v16) = match (&cache, &owned_kv) {
            (Some(c), _) => (c.k, c.v),
            (None, Some((k, v))) => (k, v),
            (None, None) => unreachable!("allocated above"),
        };

        let clen = cache.as_ref().map(|c| c.len);
        let stride = (kv_heads * head_dim) as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            for (src, dst, n, off) in [
                (q, &q16, (seq * heads * head_dim) as i64, None),
                (k, k16, (seq * kv_heads * head_dim) as i64, clen),
                (v, v16, (seq * kv_heads * head_dim) as i64, clen),
            ] {
                let blocks = ((n as usize / 8) + 255) / 256;
                match (off.is_some().then(|| cache.as_ref()).flatten().and_then(|c| c.pos_map), off) {
                    (Some(pm), Some(_)) => {
                        self.stream
                            .launch_builder(&scatter.function)
                            .arg(&src.slice()).arg(&dst.slice()).arg(&n)
                            .arg(&pm.device_ptr())
                            .arg(&stride)
                            .launch(config(blocks.max(1) as u32, 256))
                            .context("launching to_bf16_scatter_kernel")?;
                    }
                    _ => {
                        self.stream
                            .launch_builder(&conv.function)
                            .arg(&src.slice()).arg(&dst.slice()).arg(&n)
                            .arg(&off.map(|b| b.device_ptr()).unwrap_or(0u64))
                            .arg(&stride)
                            .launch(config(blocks.max(1) as u32, 256))
                            .context("launching to_bf16_kernel")?;
                    }
                }
            }

            let max_tiles = crate::gpu::kv_capacity() / ATTN_BK + 1;
            let tiles_per_split = max_tiles.div_ceil(splits) as i32;
            if splits > 1 {
                let partials = self.scratch(2, split_partials_bytes(splits, heads, head_dim))?;
                let partials = self.view(&partials, 0, splits * seq * heads * (head_dim + 2) * 4);

                let dbg = matches!(env_once!("CLEF_SPLIT_DEBUG"), Some(v) if v != "0");
                let capturing = self.capturing.load(std::sync::atomic::Ordering::SeqCst);
                if dbg {
                    eprintln!("SPLIT-DBG launch seq={seq} splits={splits} T={tiles_per_split} \
                               capturing={capturing} rows={row_blocks} part_ptr={:#x} out_ptr={:#x}",
                              partials.device_ptr(), out.device_ptr());
                }
                let probe = dbg && !capturing;
                if probe {
                    self.fill_nan(&partials)?;
                    self.sync()?;
                }
                self.stream
                    .launch_builder(&kernel.function)
                    .arg(&q16.slice()).arg(&k16.slice()).arg(&v16.slice()).arg(&out.slice())
                    .arg(&s).arg(&h).arg(&kh).arg(&hd)
                    .arg(&scale)
                    .arg(&seq_start.map(|b| b.device_ptr()).unwrap_or(SEQ_START_NULL))
                    .arg(&seq_end.map(|b| b.device_ptr()).unwrap_or(SEQ_START_NULL))
                    .arg(&(seq_end_scalar as i32))
                    .arg(&real_end_dev.map(|b| b.device_ptr()).unwrap_or(0u64))
                    .arg(&clen.map(|b| b.device_ptr()).unwrap_or(0u64))
                    .arg(&0u64)
                    .arg(&cache.as_ref().and_then(|c| c.pos_map).map(|b| b.device_ptr()).unwrap_or(0u64))
                    .arg(&cache.as_ref().and_then(|c| c.base_map).map(|b| b.device_ptr()).unwrap_or(0u64))
                    .arg(&cache.as_ref().and_then(|c| c.slot_map).map(|b| b.device_ptr()).unwrap_or(0u64))
                    .arg(&partials.slice())
                    .arg(&0i32)
                    .arg(&(seq as i32))

                    .arg(&tiles_per_split)
                    .launch(config_grid_3d(if gqa_pack { kv_heads as u32 } else { heads as u32 },
                                           row_blocks, splits as u32, threads))
                    .context("launching attention_dev_mma (split)")?;
                if probe {
                    self.split_probe(&partials, splits, seq, heads, head_dim)?;
                }
                self.mha_combine_dev(&partials, &out, seq, heads * head_dim, heads, splits)
                    .context("combining the attention's key splits")?;
                if probe {
                    self.sync()?;
                    let o = self.download_f32(&out)?;
                    let nan = o.iter().filter(|v| v.is_nan()).count();
                    let zero = o.iter().filter(|v| **v == 0.0).count();
                    eprintln!("SPLIT-DBG out: nan={nan} zero={zero} of {} first={:?}", o.len(), &o[..4.min(o.len())]);
                }
                return Ok(out);
            }
            self.stream
                .launch_builder(&kernel.function)
                .arg(&q16.slice()).arg(&k16.slice()).arg(&v16.slice()).arg(&out.slice())
                .arg(&s).arg(&h).arg(&kh).arg(&hd)
                .arg(&scale)
                .arg(&seq_start.map(|b| b.device_ptr()).unwrap_or(SEQ_START_NULL))

                .arg(&seq_end.map(|b| b.device_ptr()).unwrap_or(SEQ_START_NULL))
                .arg(&(seq_end_scalar as i32))

                .arg(&real_end_dev.map(|b| b.device_ptr()).unwrap_or(0u64))

                .arg(&clen.map(|b| b.device_ptr()).unwrap_or(0u64))

                .arg(&0u64)
                .arg(&cache.as_ref().and_then(|c| c.pos_map).map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&cache.as_ref().and_then(|c| c.base_map).map(|b| b.device_ptr()).unwrap_or(0u64))
                .arg(&cache.as_ref().and_then(|c| c.slot_map).map(|b| b.device_ptr()).unwrap_or(0u64))

                .launch(config_block_2d(if gqa_pack { kv_heads as u32 } else { heads as u32 },
                                        row_blocks, threads, 1))
                .context("launching attention_dev_mma")?;
        }
        Ok(out)
    }

    pub fn attention_dev_rows(&self, q: &Buffer, k: &Buffer, v: &Buffer, seq: usize, heads: usize,
                              kv_heads: usize, head_dim: usize, rows: u32,
                              seq_start: Option<&Buffer>) -> Result<Buffer> {
        let name = match rows {
            100 => "attention_wide",
            0 => "attention_smem",
            4 => "attention_r4",
            8 => "attention_r8",
            12 => "attention_r12",
            _ => "attention",
        };
        let module = self.load(match rows {
            100 => ATTENTION_WIDE_CUBIN,
            0 => ATTENTION_SMEM_CUBIN,
            4 => ATTENTION_R4_CUBIN,
            8 => ATTENTION_R8_CUBIN,
            12 => ATTENTION_R12_CUBIN,
            _ => ATTENTION_CUBIN,
        }, name)?;
        let kernel = module.kernel("attention_kernel")?;
        let out = self.alloc(seq * heads * head_dim * 4)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (s, h, kh, hd) = (seq as i32, heads as i32, kv_heads as i32, head_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&q.slice())
                .arg(&k.slice())
                .arg(&v.slice())
                .arg(&out.slice())
                .arg(&s)
                .arg(&h)
                .arg(&kh)
                .arg(&hd)
                .arg(&scale)

                .arg(&seq_start.map(|b| b.device_ptr()).unwrap_or(SEQ_START_NULL))

                .launch(config_block_2d(heads as u32,
                                        ((seq as u32) + 2 * match rows { 0 => 2, 100 => 8, r => r } - 1)
                                            / (2 * match rows { 0 => 2, 100 => 8, r => r }), 64, 1))
                .context("launching attention_dev")?;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_partial_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, partial: &Buffer,
                                 q_len: usize, rows: usize, heads: usize, head_dim: usize,
                                 sp: usize) -> Result<()> {
        self.attention_partial_dev_dbg(q, k, v, partial, None, q_len, rows, heads, head_dim, sp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_partial_dev_dbg(&self, q: &Buffer, k: &Buffer, v: &Buffer, partial: &Buffer,
                                     dbg: Option<&Buffer>, q_len: usize, rows: usize, heads: usize,
                                     head_dim: usize, sp: usize) -> Result<()> {
        if q_len == 0 || heads == 0 || rows == 0 {
            return Ok(());
        }

        let module = if matches!(env_once!("CLEF_PARTIAL256"), Some(v) if v != "0") {
            self.load(ATTENTION_MMA_PARTIAL256_CUBIN, "attention_mma_partial256")?
        } else {
            self.load(ATTENTION_MMA_PARTIAL_CUBIN, "attention_mma_partial")?
        };
        let kernel = module.kernel("attention_kernel")?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (s, h, kh, hd) = (q_len as i32, heads as i32, heads as i32, head_dim as i32);
        let (spi, qli) = (sp as i32, q_len as i32);
        let rows_i = rows as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&q.slice())
                .arg(&k.slice())
                .arg(&v.slice())
                .arg(&dbg.unwrap_or(partial).slice())
                .arg(&s)
                .arg(&h)
                .arg(&kh)
                .arg(&hd)
                .arg(&scale)

                .arg(&SEQ_START_NULL)
                .arg(&SEQ_START_NULL)
                .arg(&rows_i)
                .arg(&SEQ_START_NULL)
                .arg(&SEQ_START_NULL)
                .arg(&SEQ_START_NULL)

                .arg(&SEQ_START_NULL)
                .arg(&SEQ_START_NULL)
                .arg(&SEQ_START_NULL)
                .arg(&partial.slice())
                .arg(&spi)
                .arg(&qli)

                .arg(&(1i32 << 24))
                .launch(config_block_2d(heads as u32, ((q_len as u32) + 127) / 128, 256, 1))
                .context("launching attention_partial")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, decay: &Buffer,
                        beta: &Buffer, seq_len: usize, heads: usize, k_dim: usize,
                        v_dim: usize) -> Result<Buffer> {
        let module = self.load(GDN_CUBIN, "gdn")?;
        let kernel = module.kernel("gdn_scan")?;
        let out = self.alloc_zeroed(seq_len * heads * v_dim * 4)?;
        let (sl, nh, kd, vd) = (seq_len as i32, heads as i32, k_dim as i32, v_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&q.slice())
                .arg(&k.slice())
                .arg(&v.slice())
                .arg(&decay.slice())
                .arg(&beta.slice())
                .arg(&out.slice())
                .arg(&sl)
                .arg(&nh)
                .arg(&kd)
                .arg(&vd)
                .launch(config_block_2d(heads as u32, 4, v_dim as u32 / 4, 8))
                .context("launching gdn_scan_dev")?;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]

    fn gdn_chunk_passes_dev(&self, q_hi: &Buffer, q_lo: &Buffer, k_hi: &Buffer, k_lo: &Buffer,
                            kn: &Buffer, v: &Buffer, decay: &Buffer, beta: &Buffer,
                            seq_len: usize, real_len: usize, heads: usize, v_dim: usize,
                            map: BatchMap<'_>,
                            state: Option<(&Buffer, &Buffer)>) -> Result<Buffer> {
        assert_eq!(v_dim, CHUNK_VD, "gdn_chunk.cu is built for linear_value_head_dim = 128");
        let nchunks = seq_len.div_ceil(CHUNK_LEN);
        let idx_count = heads * nchunks;
        let module = self.load(GDN_CHUNK_CUBIN, "gdn_chunk")?;

        let zeroed = real_len < seq_len;
        let force = env_once!("CLEF_ZERO_ALWAYS").unwrap_or_default();
        let z = |name: &'static str, bytes: usize| -> Result<Buffer> {
            if zeroed || force == name { self.alloc_zeroed(bytes) } else { self.alloc(bytes) }
        };

        let l = z("l", idx_count * CHUNK_LEN * CHUNK_LEN * 4)?;
        let gcum = z("gcum", idx_count * CHUNK_LEN * 4)?;
        {
            let kernel = module.kernel("gdn_chunk_l")?;
            let (sl, h, nc) = (seq_len as i32, heads as i32, nchunks as i32);
            unsafe {
                use cudarc::driver::PushKernelArg;
                self.stream
                    .launch_builder(&kernel.function)
                    .arg(&k_hi.slice())
                    .arg(&k_lo.slice())
                    .arg(&decay.slice())
                    .arg(&beta.slice())
                    .arg(&l.slice())
                    .arg(&gcum.slice())
                    .arg(&sl)
                    .arg(&h)
                    .arg(&nc)
                    .launch(config(idx_count as u32, 256))
                    .context("launching gdn_chunk_l")?;
            }
        }

        let w_hi = z("w_hi", idx_count * 2 * CHUNK_LEN * 128)?;
        let w_lo = z("w_lo", idx_count * 2 * CHUNK_LEN * 128)?;
        let u0 = z("u0", idx_count * CHUNK_LEN * CHUNK_VD * 4)?;
        {
            let kernel = module.kernel("gdn_chunk_wu")?;
            let (sl, h, nc) = (seq_len as i32, heads as i32, nchunks as i32);
            unsafe {
                use cudarc::driver::PushKernelArg;
                self.stream
                    .launch_builder(&kernel.function)
                    .arg(&kn.slice())
                    .arg(&v.slice())
                    .arg(&beta.slice())
                    .arg(&l.slice())
                    .arg(&gcum.slice())
                    .arg(&w_hi.slice())
                    .arg(&w_lo.slice())
                    .arg(&u0.slice())
                    .arg(&sl)
                    .arg(&h)
                    .arg(&nc)

                    .launch(config_block_2d(idx_count as u32, 2, 128, 1))
                    .context("launching gdn_chunk_wu")?;
            }
        }

        let delta = z("delta", idx_count * CHUNK_LEN * CHUNK_VD * 4)?;

        let out = self.alloc(seq_len * heads * v_dim * 4)?;
        {
            let kernel = module.kernel("gdn_state")?;
            let segs = gdn_segs();

            let forced = std::env::var("CLEF_GDN_SEGS_FORCE").as_deref() == Ok("1");
            assert!(state.is_none() || segs == 1 || forced,
                    "the GDN state carry needs segs == 1 (CLEF_GDN_SEGS is a timing probe; \
                     CLEF_GDN_SEGS_FORCE=1 overrides it for a TIMING run whose answers are WRONG)");
            let (sl, h, nc, sg) = (seq_len as i32, heads as i32, nchunks as i32, segs as i32);

            let dbg = if std::env::var("CLEF_STATE_DEBUG").is_ok() {
                Some(self.alloc_zeroed(5 * 4)?)
            } else {
                None
            };
            let dbg_ptr = dbg.as_ref().map(|b| b.device_ptr()).unwrap_or(0u64);
            unsafe {
                use cudarc::driver::PushKernelArg;
                self.stream
                    .launch_builder(&kernel.function)
                    .arg(&q_hi.slice())
                    .arg(&q_lo.slice())
                    .arg(&k_hi.slice())
                    .arg(&k_lo.slice())
                    .arg(&w_hi.slice())
                    .arg(&w_lo.slice())
                    .arg(&u0.slice())
                    .arg(&gcum.slice())
                    .arg(&delta.slice())
                    .arg(&out.slice())

                    .arg(&sl)
                    .arg(&h)
                    .arg(&nc)
                    .arg(&sg)

                    .arg(&map.chunk_start.map(|b| b.device_ptr()).unwrap_or(0u64))

                    .arg(&state.map(|(i, _)| i.device_ptr()).unwrap_or(0u64))
                    .arg(&state.map(|(_, o)| o.device_ptr()).unwrap_or(0u64))
                    .arg(&map.state_slot.map(|b| b.device_ptr()).unwrap_or(0u64))
                    .arg(&map.carry.and_then(|c| c.real_walk).map(|b| b.device_ptr()).unwrap_or(0u64))

                    .arg(&dbg_ptr)
                    .launch(LaunchConfig {
                        grid_dim: (heads as u32, CHUNK_V_SPLIT as u32, segs as u32),
                        block_dim: (crate::cuda::gdn_state_threads() as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .context("launching gdn_state")?;
            }
            if let Some(dbg) = dbg.as_ref() {
                let bytes = self.download(dbg)?;
                let v: Vec<i32> = bytes.chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                eprintln!("  STATE_DBG slot0={} last={} slot_stride={} has_out={} nchunks={}",
                          v[0], v[1], v[2], v[3], v[4]);
            }

            if let Some((i, o)) = state {
                if i.device_ptr() != o.device_ptr() {
                    self.copy_bytes_at(i, 0, o, 0, gdn_state_bytes(heads, CHUNK_KD, CHUNK_VD))?;
                }
            }
        }

        {
            let kernel = module.kernel("gdn_chunk_out")?;
            let (sl, h, nc) = (seq_len as i32, heads as i32, nchunks as i32);
            unsafe {
                use cudarc::driver::PushKernelArg;
                self.stream
                    .launch_builder(&kernel.function)
                    .arg(&q_hi.slice())
                    .arg(&q_lo.slice())
                    .arg(&k_hi.slice())
                    .arg(&k_lo.slice())
                    .arg(&delta.slice())
                    .arg(&gcum.slice())
                    .arg(&out.slice())
                    .arg(&sl)
                    .arg(&h)
                                        .arg(&nc)
                    .arg(&map.carry.and_then(|c| c.real_walk).map(|b| b.device_ptr()).unwrap_or(0u64))
                    .launch(config(idx_count as u32, 256))
                    .context("launching gdn_chunk_out")?;
            }
        }
        Ok(out)
    }

    pub fn gdn_prep_dev(&self, mixed: &Buffer, seq_len: usize, kh: usize, kd: usize, heads: usize,
                        conv_dim: usize) -> Result<GdnPrep> {
        let nchunks = seq_len.div_ceil(CHUNK_LEN);
        let idx_count = heads * nchunks;
        let vdim = heads * kd;
        let module = self.load(GDN_CHUNK_CUBIN, "gdn_chunk")?;
        let kernel = module.kernel("gdn_prep_kernel")?;
        let v = self.alloc(seq_len * vdim * 4)?;

        let q_hi = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let q_lo = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let k_hi = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let k_lo = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let kn = self.alloc_zeroed(seq_len * vdim * 4)?;
        let q_scale = 1.0f32 / (kd as f32).sqrt();
        let (sl, nh, nk, nd, nc, cd) = (seq_len as i32, heads as i32, kh as i32, kd as i32,
                                        nchunks as i32, conv_dim as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&mixed.slice())
                .arg(&v.slice())
                .arg(&q_hi.slice())
                .arg(&q_lo.slice())
                .arg(&k_hi.slice())
                .arg(&k_lo.slice())
                .arg(&kn.slice())
                .arg(&sl)
                .arg(&nh)
                .arg(&nk)
                .arg(&nd)
                .arg(&nc)
                .arg(&cd)
                .arg(&q_scale)

                .launch(config(seq_len as u32, 256))
                .context("launching gdn_prep_dev")?;
        }
        Ok(GdnPrep { v, q_hi, q_lo, k_hi, k_lo, kn })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv_prep_dev(&self, conv_in: &Buffer, conv_w: &Buffer,
                             seq_len: usize, kh: usize, kd: usize, heads: usize, conv_dim: usize,
                             k: usize, seq_start: Option<&Buffer>, history: Option<&Buffer>,
                             slot_map: Option<&Buffer>) -> Result<GdnPrep> {
        const CP_ROWS: usize = 16;
        const CP_CH: usize = 1024;
        assert_eq!(kd, CHUNK_KD, "gdn_conv_prep_kernel is built for kd = 128");

        assert_eq!(k, 4, "gdn_conv_prep_kernel's tap window is built for k = 4");
        assert_eq!(conv_dim % CP_CH, 0, "the channel tile must divide conv_dim");
        let nchunks = seq_len.div_ceil(CHUNK_LEN);
        let idx_count = heads * nchunks;
        let vdim = heads * kd;
        let module = self.load(GDN_CHUNK_CUBIN, "gdn_chunk")?;
        let kernel = module.kernel("gdn_conv_prep_kernel")?;
        let v = self.alloc(seq_len * vdim * 4)?;

        let q_hi = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let q_lo = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let k_hi = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let k_lo = self.alloc_zeroed(idx_count * 2 * CHUNK_LEN * 128)?;
        let kn = self.alloc_zeroed(seq_len * vdim * 4)?;
        let q_scale = 1.0f32 / (kd as f32).sqrt();
        let (rows, ch, kk) = (seq_len as i32, conv_dim as i32, k as i32);
        let (nh, nk, nd, nc) = (heads as i32, kh as i32, kd as i32, nchunks as i32);
        let (ss, hist, slots) = (
            seq_start.map(|b| b.device_ptr()).unwrap_or(0u64),
            history.map(|b| b.device_ptr()).unwrap_or(0u64),
            slot_map.map(|b| b.device_ptr()).unwrap_or(0u64),
        );
        let grid = ((conv_dim / CP_CH) as u32, (seq_len.div_ceil(CP_ROWS)) as u32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&conv_in.slice())
                .arg(&conv_w.slice())
                .arg(&v.slice())
                .arg(&q_hi.slice())
                .arg(&q_lo.slice())
                .arg(&k_hi.slice())
                .arg(&k_lo.slice())
                .arg(&kn.slice())
                .arg(&rows)
                .arg(&ch)
                .arg(&kk)
                .arg(&ss)
                .arg(&hist)
                .arg(&slots)
                .arg(&nh)
                .arg(&nk)
                .arg(&nd)
                .arg(&nc)
                .arg(&q_scale)

                .launch(config_3d(grid.0, grid.1, 256))
                .context("launching gdn_conv_prep_dev")?;
        }
        Ok(GdnPrep { v, q_hi, q_lo, k_hi, k_lo, kn })
    }

    pub fn gdn_chunk_scan_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, decay: &Buffer,
                              beta: &Buffer, seq_len: usize, heads: usize, k_dim: usize,
                              v_dim: usize, map: BatchMap<'_>) -> Result<Buffer> {
        self.gdn_chunk_scan_state_dev(q, k, v, decay, beta, seq_len, heads, k_dim, v_dim, map, None)
    }

    pub fn gdn_chunk_scan_state_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, decay: &Buffer,
                                    beta: &Buffer, seq_len: usize, heads: usize, k_dim: usize,
                                    v_dim: usize, map: BatchMap<'_>,
                                    state: Option<(&Buffer, &Buffer)>) -> Result<Buffer> {
        self.gdn_chunk_scan_state_at_dev(q, k, v, decay, beta, seq_len, seq_len, heads, k_dim,
                                         v_dim, map, state)
    }

    pub fn gdn_chunk_scan_state_at_dev(&self, q: &Buffer, k: &Buffer, v: &Buffer, decay: &Buffer,
                                       beta: &Buffer, seq_len: usize, real_len: usize,
                                       heads: usize, k_dim: usize, v_dim: usize,
                                       map: BatchMap<'_>,
                                       state: Option<(&Buffer, &Buffer)>) -> Result<Buffer> {
        assert_eq!(k_dim, CHUNK_KD, "gdn_chunk.cu is built for linear_key_head_dim = 128");
        assert_eq!(v_dim, CHUNK_VD, "gdn_chunk.cu is built for linear_value_head_dim = 128");
        let nchunks = seq_len.div_ceil(CHUNK_LEN);
        let idx_count = heads * nchunks;
        let rows = seq_len * heads;
        let module = self.load(GDN_CHUNK_CUBIN, "gdn_chunk")?;

        let p = self.gdn_norm_qk_dev(q, k, seq_len, heads, k_dim)?;
        self.gdn_chunk_passes_dev(&p.q_hi, &p.q_lo, &p.k_hi, &p.k_lo, &p.kn, v, decay, beta, seq_len,
                                  real_len, heads, v_dim, map, state)
    }

    pub fn gdn_norm_qk_dev(&self, q: &Buffer, k: &Buffer, seq_len: usize, heads: usize,
                           k_dim: usize) -> Result<GdnPrep> {
        let nchunks = seq_len.div_ceil(CHUNK_LEN);
        let idx_count = heads * nchunks;
        let rows = seq_len * heads;
        let module = self.load(GDN_CHUNK_CUBIN, "gdn_chunk")?;
        let kernel = module.kernel("gdn_norm_qk")?;
        let q_hi = self.alloc(idx_count * 2 * CHUNK_LEN * 128)?;
        let q_lo = self.alloc(idx_count * 2 * CHUNK_LEN * 128)?;
        let k_hi = self.alloc(idx_count * 2 * CHUNK_LEN * 128)?;
        let k_lo = self.alloc(idx_count * 2 * CHUNK_LEN * 128)?;
        let kn = self.alloc(k.len())?;
        let (r, c) = (rows as i32, k_dim as i32);
        let q_scale = 1.0f32 / (k_dim as f32).sqrt();
        let (hh, nc) = (heads as i32, nchunks as i32);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&q.slice())
                .arg(&k.slice())
                .arg(&q_hi.slice())
                .arg(&q_lo.slice())
                .arg(&k_hi.slice())
                .arg(&k_lo.slice())
                .arg(&kn.slice())
                .arg(&r)
                .arg(&c)
                .arg(&q_scale)
                .arg(&hh)
                .arg(&nc)
                .launch(config((rows as u32).div_ceil(4), 128))
                .context("launching gdn_norm_qk")?;
        }
        Ok(GdnPrep { v: self.alloc(0)?, q_hi, q_lo, k_hi, k_lo, kn })
    }

    pub fn gdn_chunk_scan_prepared_dev(&self, prep: &GdnPrep, decay: &Buffer, beta: &Buffer,
                                       seq_len: usize, heads: usize, v_dim: usize,
                                       map: BatchMap<'_>) -> Result<Buffer> {
        self.gdn_chunk_scan_prepared_state_dev(prep, decay, beta, seq_len, heads, v_dim, map, None)
    }

    pub fn gdn_chunk_scan_prepared_state_dev(&self, prep: &GdnPrep, decay: &Buffer, beta: &Buffer,
                                             seq_len: usize, heads: usize, v_dim: usize,
                                             map: BatchMap<'_>,
                                             state: Option<(&Buffer, &Buffer)>) -> Result<Buffer> {
        self.gdn_chunk_scan_prepared_state_at_dev(prep, decay, beta, seq_len, seq_len, heads, v_dim,
                                                  map, state)
    }

    pub fn gdn_chunk_scan_prepared_state_at_dev(&self, prep: &GdnPrep, decay: &Buffer, beta: &Buffer,
                                                seq_len: usize, real_len: usize, heads: usize,
                                                v_dim: usize, map: BatchMap<'_>,
                                                state: Option<(&Buffer, &Buffer)>) -> Result<Buffer> {
        self.gdn_chunk_passes_dev(&prep.q_hi, &prep.q_lo, &prep.k_hi, &prep.k_lo, &prep.kn,
                                  &prep.v, decay, beta, seq_len, real_len, heads, v_dim, map, state)
    }

    pub fn head_logit_dev(&self, prior: &Buffer, cosine: &Buffer, residual: &Buffer,
                          joint_scale: f32, gate: f32) -> Result<Buffer> {
        let n = prior.len() / 4;
        let module = self.load(HEAD_CUBIN, "head")?;
        let kernel = module.kernel("head_logit_kernel")?;
        let out = self.alloc(n * 4)?;
        let ni = n as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&prior.slice())
                .arg(&cosine.slice())
                .arg(&residual.slice())
                .arg(&out.slice())
                .arg(&ni)
                .arg(&joint_scale)
                .arg(&gate)
                .launch(config(((n as u32) + 255) / 256, 256))
                .context("launching head_logit")?;
        }
        Ok(out)
    }

    pub fn write_bytes(&self, buf: &Buffer, host: &[u8]) -> Result<()> {
        assert_eq!(host.len(), buf.len(), "write_bytes: {} bytes into a {} byte buffer",
                   host.len(), buf.len());
        self.copy_htod_into(buf, host)
    }

    pub fn write_i32(&self, buf: &Buffer, host: &[i32]) -> Result<()> {
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write_bytes(buf, &bytes)
    }

    pub fn upload_i32(&self, host: &[i32]) -> Result<Buffer> {
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(&bytes)
    }

    pub fn upload_f32(&self, host: &[f32]) -> Result<Buffer> {
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(&bytes)
    }

    pub fn nvfp4_gemm(
        &self,
        a: &[u8],
        b: &[u8],
        sa: &[u8],
        sb: &[u8],
        m: usize,
        n: usize,
        k: usize,
        global_a: f32,
        global_b: f32,
    ) -> Result<Vec<f32>> {
        if m % 16 != 0 || n % 8 != 0 {
            bail!("m must be a multiple of 16 and n of 8; got m={m}, n={n}");
        }
        if k % 64 != 0 {
            bail!("k must be a multiple of 64 for m16n8k64; got {k}");
        }
        if a.len() != m * k / 2 || b.len() != n * k / 2 {
            bail!("packed operand sizes do not match [m,k] and [n,k]");
        }
        if sa.len() != m * k / 16 || sb.len() != n * k / 16 {
            bail!("scale sizes do not match [m,k/16] and [n,k/16]");
        }

        let module = self.gemm_module()?;
        let kernel = module.kernel("nvfp4_gemm")?;

        let dev_a = self.upload(a)?;
        let dev_b = self.upload(b)?;
        let dev_sa = self.swizzle_scales(sa, m, k / 16, SF_CONFLICT_FREE)?;
        let dev_sb = self.swizzle_scales(sb, n, k / 16, false)?;

        let n_splits = 1usize;

        let dev_c = if n_splits > 1 {
            self.alloc_zeroed(m * n * 4)?
        } else {
            self.alloc(m * n * 4)?
        };

        self.launch_nvfp4_gemm(&dev_a, &dev_b, &dev_sa, &dev_sb, &dev_c, m, n, k, global_a,
                               global_b, m, n_splits)?;
        self.stream.synchronize().context("GEMM sync")?;

        let raw = self.download(&dev_c)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn swizzle_scales(&self, scales: &[u8], rows: usize, cols: usize, conflict_free: bool)
        -> Result<Buffer> {
        self.swizzle_scales_for_tile(scales, rows, cols, conflict_free, 128)
    }

    pub fn swizzle_scales_for_tile(
        &self,
        scales: &[u8],
        rows: usize,
        cols: usize,
        conflict_free: bool,
        tile_m: usize,
    ) -> Result<Buffer> {
        let module = self.swizzle_module()?;
        let kernel = module.kernel("swizzle_blockscale")?;

        let blk_mn = crate::swizzle::sf_atom_in_force(tile_m);
        let chunk = crate::swizzle::sf_chunk_stride_in_force(tile_m);
        let rows_padded = crate::swizzle::round_up_public(rows, blk_mn);
        let cols_padded = crate::swizzle::round_up_public(cols, 4);

        let out_bytes = crate::swizzle::swizzled_len_tiled(rows, cols, tile_m);

        let src = self.upload(scales)?;

        let dst = self.alloc_zeroed(out_bytes)?;

        let threads = 256u32;
        let blocks = out_bytes.div_ceil(threads as usize) as u32;

        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&dst.slice())
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(rows_padded as i32))
                .arg(&(cols_padded as i32))
                .arg(&(conflict_free as i32))
                .arg(&(blk_mn as i32))
                .arg(&(chunk as i32))
                .launch(config(blocks, threads))
                .context("launching the swizzle")?;
        }
        self.stream.synchronize().context("swizzle sync")?;
        Ok(dst)
    }

    pub fn load(&self, cubin: &[u8], name: &str) -> Result<Module> {
        if let Some(m) = self.modules.lock().unwrap().get(name) {
            return Ok(Module { module: m.module.clone(), functions: Default::default() });
        }
        let module = self
            .ctx
            .load_module(Ptx::from_binary(cubin.to_vec()))
            .with_context(|| format!("loading cubin {name}"))?;
        let m = Module { module, functions: Default::default() };
        self.modules
            .lock()
            .unwrap()
            .insert(name.to_string(), Module { module: m.module.clone(), functions: Default::default() });
        Ok(m)
    }

    pub fn sm_count(&self) -> usize {
        self.ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(170)
            .max(1) as usize
    }

    #[track_caller]
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        let tail = tail_nan_bytes();

        let granule_mib = env_once!("CLEF_ALLOC_GRANULE_MIB")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);

        let asked = bytes;
        let alloc_bytes = if granule_mib == 0 {
            bytes
        } else {
            let g = granule_mib * 1024 * 1024;

            if bytes >= g { bytes.div_ceil(g) * g } else { bytes.div_ceil(4096) * 4096 }
        };
        let bytes = alloc_bytes;
        let mut b = if self.capturing.load(std::sync::atomic::Ordering::SeqCst)
            && env_once!("CLEF_GRAPH_POOL") != Some("shared")
        {
            self.alloc_from_graph_block(bytes + tail)
                .with_context(|| format!("allocating {bytes} bytes from the capture block"))?
        } else {
            let block_bytes = bytes + tail;
            let data = match alloc_cache_take(block_bytes) {
                Some(block) => block,
                None => std::sync::Arc::new(unsafe { self.stream.alloc::<u8>(block_bytes) }
                    .with_context(|| format!("allocating {bytes} bytes"))?),
            };
            Buffer {
                stream: self.stream.clone(),
                data,
                offset: 0,
                len: asked + tail,
                recycle: None,
            }
        };
        if tail > 0 {
            let full = Buffer {
                stream: self.stream.clone(),
                data: Arc::clone(&b.data),
                offset: b.offset,
                len: bytes + tail,
                recycle: None,
            };
            self.fill_range(&full, asked, tail, 0xFF)?;
            b.len = asked;
        }

        if let Some(v) = env_once!("CLEF_ALLOC_BT_BYTES") {
            if v.parse::<usize>().ok() == Some(bytes) {
                eprintln!("  ALLOC_AT bytes={bytes} {}", std::panic::Location::caller());
            }
        }
        if env_once!("CLEF_ZERO_ALLOC").is_some() || env_once!("CLEF_POISON_ALLOC").is_some() {
            if env_once!("CLEF_POISON_ALLOC").is_some() {
                self.poison_fill(&b)?;
            } else {
                self.zero_fill(&b)?;
            }
            return Ok(b);
        }

        if let Some(v) = env_once!("CLEF_ZERO_MAX_BYTES") {
            if let Ok(limit) = v.parse::<usize>() {
                if bytes <= limit {
                    self.zero_fill(&b)?;
                }
            }
        }

        if let Some(v) = env_once!("CLEF_ZERO_ALLOC_EXACT") {
            if v.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).any(|n| n == bytes) {
                self.zero_fill(&b)?;
            }
        }

        if alloc_trace_on() {
            eprintln!("  ALLOC ptr={:?} len={} slot=None", b.device_ptr(), bytes);
        }
        Ok(b)
    }

    pub fn pool_bytes(&self) -> usize {
        use cudarc::driver::sys;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

        unsafe {
            if sys::cuDeviceGetMemPool(&mut pool, self.ctx.cu_device()) != sys::CUresult::CUDA_SUCCESS
                || pool.is_null()
            {
                return 0;
            }
            let mut v: u64 = 0;
            let r = sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                &mut v as *mut u64 as *mut std::ffi::c_void,
            );
            if r != sys::CUresult::CUDA_SUCCESS {
                return 0;
            }
            v as usize
        }
    }

    pub fn trim_device_pool(&self) -> usize {
        if env_once!("CLEF_POOL_TRIM") != Some("1") {
            return 0;
        }
        use cudarc::driver::sys;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

        unsafe {
            if sys::cuDeviceGetMemPool(&mut pool, self.ctx.cu_device()) != sys::CUresult::CUDA_SUCCESS
                || pool.is_null()
            {
                return 0;
            }
        }

        let budget = env_once!("CLEF_POOL_TRIM_ABOVE_MIB")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096) as usize
            * 1024 * 1024;
        if budget > 0 && self.pool_bytes() <= budget {
            return 0;
        }
        let keep = env_once!("CLEF_POOL_RESERVE_MIB")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2048) as usize
            * 1024 * 1024;
        let before = self.pool_bytes();

        let r = unsafe { sys::cuMemPoolTrimTo(pool, keep) };
        if r != sys::CUresult::CUDA_SUCCESS {
            tracing::warn!(?r, "cuMemPoolTrimTo failed");
            return 0;
        }
        let after = self.pool_bytes();
        before.saturating_sub(after)
    }

    pub fn alloc_zeroed(&self, bytes: usize) -> Result<Buffer> {
        let buf = self.alloc(bytes)?;
        self.zero_fill(&buf)?;
        Ok(buf)
    }

    pub fn poison_fill(&self, buf: &Buffer) -> Result<()> {
        use cudarc::driver::sys;

        let pattern: u32 = env_once!("CLEF_POISON_VALUE")
            .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x3f80_0000);
        let count = (buf.len() / 4) as usize;
        let result = unsafe {
            sys::cuMemsetD32Async(
                buf.device_ptr() as sys::CUdeviceptr,
                pattern,
                count,
                self.stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("filling {count} words with a pattern: {result:?}");
        }
        Ok(())
    }

    pub fn fill_range(&self, buf: &Buffer, offset: usize, bytes: usize, value: u8) -> Result<()> {
        if bytes == 0 || offset + bytes > buf.len() {
            return Ok(());
        }
        use cudarc::driver::sys;

        let r = unsafe {
            sys::cuMemsetD8Async(
                (buf.device_ptr() + offset as u64) as sys::CUdeviceptr,
                value,
                bytes,
                self.stream.cu_stream(),
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemsetD8Async on a range failed: {r:?}");
        }
        Ok(())
    }

    pub fn zero_range(&self, buf: &Buffer, offset: usize, bytes: usize) -> Result<()> {
        if bytes == 0 || offset + bytes > buf.len() {
            return Ok(());
        }
        use cudarc::driver::sys;

        let r = unsafe {
            sys::cuMemsetD8Async(
                (buf.device_ptr() + offset as u64) as sys::CUdeviceptr,
                0,
                bytes,
                self.stream.cu_stream(),
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemsetD8Async on a range failed: {r:?}");
        }
        Ok(())
    }

    pub fn fill_nan(&self, buf: &Buffer) -> Result<()> {
        let pattern = f32::NAN.to_le_bytes();
        let mut bytes = vec![0u8; buf.len()];
        for c in bytes.chunks_exact_mut(4) {
            c.copy_from_slice(&pattern);
        }
        self.copy_htod_into(buf, &bytes)
    }

    pub fn zero_fill(&self, buf: &Buffer) -> Result<()> {
        if env_once!("CLEF_ALLOC_TRACE").is_some() {
            eprintln!("  ZEROFILL ptr={:?} len={}", buf.device_ptr(), buf.len());
        }
        use cudarc::driver::sys;
        let result = unsafe {
            sys::cuMemsetD8Async(
                buf.device_ptr() as sys::CUdeviceptr,
                0,
                buf.len(),
                self.stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemsetD8Async failed: {result:?}");
        }
        Ok(())
    }

    pub fn copy_htod_pinned(&self, dst: &Buffer, src: &PinnedBuf, bytes: usize) -> Result<()> {
        use cudarc::driver::sys;
        let len = bytes.min(src.len()).min(dst.len());
        anyhow::ensure!(len > 0, "a pinned copy with no bytes");

        let r = unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst.device_ptr(), src.as_ptr(), len, self.stream.cu_stream())
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemcpyHtoDAsync (pinned, {len} bytes) failed: {r:?}");
        }
        Ok(())
    }

    pub fn copy_htod_into(&self, dst: &Buffer, host: &[u8]) -> Result<()> {
        use cudarc::driver::sys;

        if self.capturing.load(std::sync::atomic::Ordering::SeqCst) {
            anyhow::bail!(
                "an H2D copy inside a capture bakes a host pointer that nothing keeps alive — the graph \
                 would re-read freed host memory on every replay. Warm this buffer before the capture. \
                 ({} bytes into a {} byte buffer)",
                host.len(),
                dst.len()
            );
        }
        if host.len() > dst.len() {
            anyhow::bail!(
                "the host slice is {} bytes and the destination buffer is {}",
                host.len(),
                dst.len()
            );
        }
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst.device_ptr() as sys::CUdeviceptr,
                host.as_ptr().cast(),
                host.len(),
                self.stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemcpyHtoDAsync_v2 failed: {result:?}");
        }
        Ok(())
    }

    pub fn upload(&self, host: &[u8]) -> Result<Buffer> {
        let buf = self
            .alloc(host.len())
            .with_context(|| format!("allocating {} bytes to upload into", host.len()))?;

        self.copy_htod_into(&buf, host)
            .with_context(|| format!("uploading {} bytes", host.len()))?;
        Ok(buf)
    }

    pub fn download(&self, buffer: &Buffer) -> Result<Vec<u8>> {
        self.stream
            .clone_dtoh(&buffer.slice())
            .context("downloading from device")
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn timed_dev<T>(&self, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<(T, std::time::Duration)> {
        use cudarc::driver::sys::CUevent_flags;
        let start = self.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .context("creating the start event")?;
        let end = self.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .context("creating the end event")?;
        start.record(&self.stream).context("recording the start event")?;
        let out = f()?;
        end.record(&self.stream).context("recording the end event")?;
        end.synchronize().context("waiting for the end event")?;
        let ms = start.elapsed_ms(&end).context("measuring the event pair")?;
        Ok((out, std::time::Duration::from_secs_f64((ms as f64) / 1e3)))
    }

    pub fn begin_priming(&self) {
        self.begin_capture_scope();
        self.capturing.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn end_priming(&self) {
        self.capturing.store(false, std::sync::atomic::Ordering::SeqCst);
        self.end_capture_scope();
    }

    fn begin_capture_scope(&self) {
        let zero = std::env::var("CLEF_GRAPH_ZERO_BLOCK").is_ok();
        let poison = std::env::var("CLEF_GRAPH_POISON_BLOCK").is_ok();
        if zero || poison {
            if let Ok(blocks) = self.graph_blocks.lock() {
                if let Some(block) = blocks.last() {
                    let b = Buffer {
                        stream: self.stream.clone(),
                        data: Arc::clone(block),
                        offset: 0,
                        len: block.len(),
                        recycle: None,
                    };
                    if poison {
                        let _ = self.poison_fill(&b);
                    } else {
                        let _ = self.zero_fill(&b);
                    }
                }
            }
        }
        self.graph_free.clear();
        self.graph_free.open.store(true, std::sync::atomic::Ordering::SeqCst);
        self.graph_bump.store(0, std::sync::atomic::Ordering::SeqCst);
        self.graph_pool_bytes.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    fn end_capture_scope(&self) {
        self.graph_free.open.store(false, std::sync::atomic::Ordering::SeqCst);
        self.graph_free.clear();
        self.graph_bump.store(0, std::sync::atomic::Ordering::SeqCst);
        let used = self.graph_pool_bytes.load(std::sync::atomic::Ordering::SeqCst);
        self.graph_pool_peak.fetch_max(used, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn set_graph_block_cap(&self, bytes: usize) {
        self.graph_block_cap.store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn graph_block_cap(&self) -> usize {
        self.graph_block_cap.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn note_graph_block_cleared(&self) {
        self.graph_block_has_graphs.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn set_graph_reserve(&self, bytes: usize) {
        self.graph_reserve.store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn graph_reserve(&self) -> usize {
        self.graph_reserve.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn graph_block_bytes(&self) -> usize {
        self.graph_blocks
            .lock()
            .map(|b| b.iter().map(|b| b.len()).sum())
            .unwrap_or(0)
    }

    fn current_block_bytes(&self) -> usize {
        self.graph_blocks
            .lock()
            .ok()
            .and_then(|b| b.last().map(|b| b.len()))
            .unwrap_or(0)
    }

    pub fn graph_blocks_retained(&self) -> usize {
        self.graph_blocks.lock().map(|b| b.len()).unwrap_or(0)
    }

    fn current_graph_block(&self) -> Result<Arc<CudaSlice<u8>>> {
        let mut blocks = self
            .graph_blocks
            .lock()
            .map_err(|_| anyhow!("the graph_blocks lock was poisoned"))?;
        if let Some(block) = blocks.last() {
            return Ok(Arc::clone(block));
        }
        let initial = graph_block_initial_bytes();
        let block = Arc::new(self.alloc_from_graph_pool(initial)?);
        tracing::info!(initial, "created the retained capture block");
        blocks.push(Arc::clone(&block));
        Ok(block)
    }

    fn grow_graph_block(&self, needed: usize) -> Result<()> {
        let current = self.current_block_bytes();
        let target = needed.max(current * 2).max(graph_block_initial_bytes());
        let free = self.free_memory().unwrap_or(0);

        let declared = self.graph_block_cap();
        let headroom = free.saturating_sub(1 << 30);
        let cap = if declared > 0 { declared.min(headroom) } else { headroom };

        let target = if target > cap && needed <= cap { needed } else { target };
        if target > cap {
            return Err(anyhow::Error::new(CaptureTooLarge { needed, cap }));
        }
        let block = Arc::new(self.alloc_from_graph_pool(target)?);
        let mut blocks = self
            .graph_blocks
            .lock()
            .map_err(|_| anyhow!("the graph_blocks lock was poisoned"))?;
        if self.graph_block_has_graphs.load(std::sync::atomic::Ordering::SeqCst) {
            tracing::info!(current, target, "growing the capture block; the old one keeps its graphs");
            blocks.push(block);
        } else {
            tracing::info!(current, target, "growing the capture block; the old one held no graphs");
            match blocks.last_mut() {
                Some(slot) => *slot = block,
                None => blocks.push(block),
            }
        }
        self.graph_block_has_graphs.store(false, std::sync::atomic::Ordering::SeqCst);
        self.graph_bump.store(0, std::sync::atomic::Ordering::SeqCst);
        self.graph_free.clear();
        Ok(())
    }

    fn alloc_from_graph_block(&self, bytes: usize) -> Result<Buffer> {
        const ALIGN: usize = 256;
        let aligned = bytes.div_ceil(ALIGN) * ALIGN;
        let block = self.current_graph_block()?;

        let mut block = block;
        let offset = loop {
            let reused = if graph_reuse_enabled() { self.graph_free.take(aligned) } else { None };
            if let Some(offset) = reused {
                break offset;
            }
            let offset = self.graph_bump.fetch_add(aligned, std::sync::atomic::Ordering::SeqCst);
            if offset + aligned <= block.len() {
                break offset;
            }

            self.graph_bump.fetch_sub(aligned, std::sync::atomic::Ordering::SeqCst);
            if self.in_capture.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(anyhow::Error::new(CaptureOverrun { needed: offset + aligned }));
            }
            self.grow_graph_block(offset + aligned)?;
            block = self.current_graph_block()?;
        };
        self.graph_pool_bytes
            .fetch_max(offset + aligned, std::sync::atomic::Ordering::SeqCst);
        if matches!(env_once!("CLEF_GRAPH_ALLOC_TRACE"), Some(v) if v != "0") {
            eprintln!("GRAPH-ALLOC off={offset} len={aligned}");
        }
        Ok(Buffer {
            stream: self.stream.clone(),
            data: block,
            offset,
            len: bytes,
            recycle: Some(Arc::clone(&self.graph_free)),
        })
    }

    pub fn graph_pool_high(&self) -> usize {
        use cudarc::driver::sys;
        let slot = match self.graph_pool.lock() { Ok(g) => *g, Err(_) => return 0 };
        if slot == 0 { return 0; }
        let mut value: usize = 0;

        let r = unsafe {
            sys::cuMemPoolGetAttribute(
                slot as sys::CUmemoryPool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                (&mut value as *mut usize).cast(),
            )
        };
        if r == sys::CUresult::CUDA_SUCCESS { value } else { 0 }
    }

    pub fn graph_pool_peak(&self) -> usize {
        self.graph_pool_peak.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn graph_pool_allocations(&self) -> usize {
        self.graph_pool_allocs.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn graph_pool_reserved(&self) -> usize {
        use cudarc::driver::sys;
        let slot = match self.graph_pool.lock() {
            Ok(g) => *g,
            Err(_) => return 0,
        };
        if slot == 0 {
            return 0;
        }
        let mut value: usize = 0;

        let r = unsafe {
            sys::cuMemPoolGetAttribute(
                slot as sys::CUmemoryPool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                (&mut value as *mut usize).cast(),
            )
        };
        if r == sys::CUresult::CUDA_SUCCESS { value } else { 0 }
    }

    fn graph_pool(&self) -> Result<cudarc::driver::sys::CUmemoryPool> {
        use cudarc::driver::sys;
        let mut slot = self.graph_pool.lock().expect("the graph_pool lock is never poisoned here");
        if *slot != 0 {
            return Ok(*slot as sys::CUmemoryPool);
        }
        self.ctx.bind_to_thread().context("binding the context")?;
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;

        unsafe { props.location.__bindgen_anon_1.id = self.ctx.cu_device() as std::os::raw::c_int };
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

        let result = unsafe { sys::cuMemPoolCreate(&mut pool, &props) };
        if result != sys::CUresult::CUDA_SUCCESS || pool.is_null() {
            anyhow::bail!("cuMemPoolCreate failed: {result:?}");
        }
        for (attr, value) in [
            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 1u64),
            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, 0),
            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, 0),

            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, u64::MAX),
        ] {
            let mut value = value;

            let r = unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut value as *mut u64).cast()) };
            if r != sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("cuMemPoolSetAttribute({attr:?}) failed: {r:?}");
            }
        }

        let mut got: u64 = u64::MAX;

        let r = unsafe {
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut got as *mut u64).cast(),
            )
        };
        tracing::info!(?r, got, "created the dedicated capture pool");
        if r != sys::CUresult::CUDA_SUCCESS || got != u64::MAX {
            anyhow::bail!(
                "the capture pool's release threshold did not take: query {r:?}, value {got} (wanted {})",
                u64::MAX
            );
        }
        *slot = pool as usize;
        Ok(pool)
    }

    fn alloc_from_graph_pool(&self, bytes: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        use cudarc::driver::sys;
        let pool = self.graph_pool()?;
        let mut ptr: sys::CUdeviceptr = 0;

        let result = unsafe {
            sys::cuMemAllocFromPoolAsync(&mut ptr, bytes, pool, self.stream.cu_stream())
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemAllocFromPoolAsync({bytes} bytes) failed: {result:?}");
        }

        self.graph_pool_allocs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(unsafe { self.stream.upgrade_device_ptr(ptr, bytes) })
    }

    pub fn capture(&self, mut f: impl FnMut() -> Result<()>) -> Result<Captured> {
        self.graph_pool()?;

        self.current_graph_block()?;
        let saved = self.pool_release_threshold()?;
        self.set_pool_release_threshold(u64::MAX)?;
        let out = self.capture_attempts(&mut f);
        let _ = self.set_pool_release_threshold(saved);
        out
    }

    fn capture_attempts(&self, f: &mut impl FnMut() -> Result<()>) -> Result<Captured> {
        use cudarc::driver::sys;
        const MAX_ATTEMPTS: usize = 6;
        let mut last: Option<anyhow::Error> = None;
        for _ in 0..MAX_ATTEMPTS {
            self.begin_capture_scope();
            self.capturing.store(true, std::sync::atomic::Ordering::SeqCst);
            self.in_capture.store(true, std::sync::atomic::Ordering::SeqCst);
            let began = self.stream.begin_capture(
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            if let Err(e) = began {
                self.in_capture.store(false, std::sync::atomic::Ordering::SeqCst);
                self.capturing.store(false, std::sync::atomic::Ordering::SeqCst);
                self.end_capture_scope();
                return Err(e).context("beginning graph capture");
            }
            let ran = f();
            let graph = self.stream.end_capture(sys::CUgraphInstantiate_flags(0));
            self.in_capture.store(false, std::sync::atomic::Ordering::SeqCst);
            self.capturing.store(false, std::sync::atomic::Ordering::SeqCst);
            self.end_capture_scope();

            let _ = self.stream.synchronize();
            match ran {
                Ok(()) => {
                    let graph = graph.context("ending graph capture")?;
                    let graph = graph.context("capture produced no graph — was anything launched?")?;
                                    let (alloc_nodes, free_nodes, memcpy_nodes) =
                                        count_memory_nodes(graph.cu_graph());
                                    if memcpy_nodes > 0 {
                                        tracing::warn!(
                                            memcpy_nodes,
                                            "the captured graph carries memcpy nodes; check their sources \
                                             outlive the graph"
                                        );
                                    }
                    if alloc_nodes > 0 || free_nodes > 0 {
                        tracing::warn!(
                            "the captured graph carries {alloc_nodes} allocation and {free_nodes} free                              nodes; each is re-executed on every replay and the addresses will move                              between them"
                        );
                    }
                    self.graph_block_has_graphs.store(true, std::sync::atomic::Ordering::SeqCst);
                    return Ok(Captured { graph, alloc_nodes, free_nodes, memcpy_nodes });
                }
                Err(e) => {
                    let Some(over) = e.downcast_ref::<CaptureOverrun>() else { return Err(e) };
                    let needed = over.needed;
                    tracing::info!(needed, "the capture overran its block; growing and retrying");
                    self.grow_graph_block(needed)?;
                    last = Some(e);
                }
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("the capture never ran")))
            .context("the capture did not fit after growing the block")
    }

    pub fn copy_into(&self, dst: &Buffer, src: &Buffer, values: usize) -> Result<()> {
        if src.len() < values * 4 {
            anyhow::bail!(
                "copy_into: the source is {} values and {values} were asked for — an out-of-bounds read",
                src.len() / 4
            );
        }
        let total = values;
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("copy_kernel")?;
        let n = total as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&dst.slice())
                .arg(&n)
                .launch(config(((total as u32) + 1023) / 1024, 256))
                .context("refreshing a pinned input")?;
        }
        Ok(())
    }

    pub fn copy_bytes_at(&self, dst: &Buffer, dst_off: usize, src: &Buffer, src_off: usize,
                         bytes: usize) -> Result<()> {
        if src_off + bytes > src.len() {
            anyhow::bail!(
                "copy_bytes_at: reading {bytes} at {src_off} from a {} byte source",
                src.len()
            );
        }
        if dst_off + bytes > dst.len() {
            anyhow::bail!(
                "copy_bytes_at: writing {bytes} at {dst_off} into a {} byte destination",
                dst.len()
            );
        }
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("copy_range_u8_kernel")?;
        let (n, so, dof) = (bytes as i32, src_off as i64, dst_off as i64);
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&so)
                .arg(&dst.slice())
                .arg(&dof)
                .arg(&n)
                .launch(config(((bytes as u32) + 255) / 256, 256))
                .context("copying a byte range device to device")?;
        }
        Ok(())
    }

    pub fn copy_bytes(&self, dst: &Buffer, src: &Buffer, bytes: usize) -> Result<()> {
        if src.len() < bytes {
            anyhow::bail!(
                "copy_bytes: the source is {} bytes and {bytes} were asked for — an out-of-bounds read",
                src.len()
            );
        }
        let module = self.load(ELEMENTWISE_CUBIN, "elementwise")?;
        let kernel = module.kernel("copy_u8_kernel")?;
        let n = bytes as i32;
        unsafe {
            use cudarc::driver::PushKernelArg;
            self.stream
                .launch_builder(&kernel.function)
                .arg(&src.slice())
                .arg(&dst.slice())
                .arg(&n)
                .launch(config(((bytes as u32) + 255) / 256, 256))
                .context("copying bytes device to device")?;
        }
        Ok(())
    }

    pub fn view(&self, buf: &Buffer, offset: usize, len: usize) -> Buffer {
        Buffer {
            stream: self.stream.clone(),
            data: Arc::clone(&buf.data),
            offset: buf.offset + offset,
            len,
            recycle: None,
        }
    }

    pub fn copy_rows_from(&self, src: &Buffer, row0: usize, rows: usize, cols: usize) -> Result<Buffer> {
        let out = self.alloc(rows * cols * 4)?;
        let s = Buffer {
            stream: self.stream.clone(),
            data: Arc::clone(&src.data),
            offset: src.offset + row0 * cols * 4,
            len: rows * cols * 4,
            recycle: None,
        };
        self.copy_bytes(&out, &s, rows * cols * 4)?;
        Ok(out)
    }

    pub fn copy_rows(&self, src: &Buffer, rows: usize, cols: usize) -> Result<Buffer> {
        let out = self.alloc(rows * cols * 4)?;
        {
            let src_view = src.slice();
            let s = Buffer {
                stream: self.stream.clone(),
                data: Arc::clone(&src.data),
                offset: src.offset,
                len: rows * cols * 4,
                recycle: None,
            };
            let _ = src_view;
            self.copy_bytes(&out, &s, rows * cols * 4)?;
        }
        Ok(out)
    }

    pub fn copy_dev(&self, src: &Buffer, values: usize) -> Result<Buffer> {
        let out = self.alloc(values * 4)?;
        self.copy_bytes(&out, src, values * 4)?;
        Ok(out)
    }

    fn pool_release_threshold(&self) -> Result<u64> {
        use cudarc::driver::sys;
        self.ctx.bind_to_thread().context("binding the context")?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

        let r = unsafe { sys::cuDeviceGetMemPool(&mut pool, self.ctx.cu_device()) };
        if r != sys::CUresult::CUDA_SUCCESS || pool.is_null() {
            anyhow::bail!("cuDeviceGetMemPool failed: {r:?}");
        }
        let mut value: u64 = 0;

        let r = unsafe {
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut value as *mut u64).cast(),
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemPoolGetAttribute failed: {r:?}");
        }
        Ok(value)
    }

    fn set_pool_release_threshold(&self, value: u64) -> Result<()> {
        use cudarc::driver::sys;
        self.ctx.bind_to_thread().context("binding the context")?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

        let r = unsafe { sys::cuDeviceGetMemPool(&mut pool, self.ctx.cu_device()) };
        if r != sys::CUresult::CUDA_SUCCESS || pool.is_null() {
            anyhow::bail!("cuDeviceGetMemPool failed: {r:?}");
        }
        let mut v = value;

        let r = unsafe {
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut v as *mut u64).cast(),
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {r:?}");
        }
        Ok(())
    }

    pub fn configure_graph_mempool(&self) -> Result<()> {
        use cudarc::driver::sys;
        if !self.ctx.has_async_alloc() {
            return Ok(());
        }
        self.ctx.bind_to_thread().context("binding the context")?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

        let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, self.ctx.cu_device()) };
        if result != sys::CUresult::CUDA_SUCCESS || pool.is_null() {
            anyhow::bail!("cuDeviceGetMemPool failed: {result:?}");
        }
        for (attr, enabled) in [
            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 1),
            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, 0),
            (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, 0),
        ] {
            let mut enabled: i32 = enabled;

            let result = unsafe {
                sys::cuMemPoolSetAttribute(pool, attr, (&mut enabled as *mut i32).cast())
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("cuMemPoolSetAttribute({attr:?}) failed: {result:?}");
            }
        }
        Ok(())
    }

    pub fn l2_cache_size(&self) -> usize {
        self.ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .unwrap_or(0) as usize
    }

    pub fn max_persisting_l2(&self) -> usize {
        self.ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            )
            .map(|v| v as usize)
            .unwrap_or(0)
    }

    pub fn set_persisting_l2(&self, bytes: usize) -> Result<()> {
        use cudarc::driver::sys;
        let r = unsafe { sys::cuCtxSetLimit(sys::CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, bytes) };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuCtxSetLimit(PERSISTING_L2, {bytes}) failed: {r:?}");
        }
        Ok(())
    }

    pub fn set_l2_window(&self, base: &Buffer, bytes: usize, hit_ratio: f32) -> Result<()> {
        use cudarc::driver::sys;
        let mut value: sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
        value.accessPolicyWindow = sys::CUaccessPolicyWindow {
            base_ptr: base.device_ptr() as *mut std::ffi::c_void,
            num_bytes: bytes,
            hitRatio: hit_ratio,
            hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
        };
        let r = unsafe {
            sys::cuStreamSetAttribute(
                self.stream.cu_stream(),
                sys::CUkernelNodeAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &value,
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuStreamSetAttribute(ACCESS_POLICY_WINDOW, {bytes}) failed: {r:?}");
        }
        Ok(())
    }

    pub fn clear_l2_window(&self) -> Result<()> {
        use cudarc::driver::sys;
        let mut value: sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
        value.accessPolicyWindow = sys::CUaccessPolicyWindow {
            base_ptr: std::ptr::null_mut(),
            num_bytes: 0,
            hitRatio: 0.0,
            hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL,
            missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL,
        };
        let r = unsafe {
            sys::cuStreamSetAttribute(
                self.stream.cu_stream(),
                sys::CUkernelNodeAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &value,
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("clearing the access policy window failed: {r:?}");
        }
        Ok(())
    }

    pub fn gpu_time_of(&self, f: impl FnOnce() -> Result<()>) -> Result<std::time::Duration> {
        use cudarc::driver::sys;
        let mut start: sys::CUevent = std::ptr::null_mut();
        let mut end: sys::CUevent = std::ptr::null_mut();
        let stream = self.stream.cu_stream();
        unsafe {
            if sys::cuEventCreate(&mut start, 0) != sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("cuEventCreate failed");
            }
            if sys::cuEventCreate(&mut end, 0) != sys::CUresult::CUDA_SUCCESS {
                sys::cuEventDestroy_v2(start);
                anyhow::bail!("cuEventCreate failed");
            }
            sys::cuEventRecord(start, stream);
        }
        let ran = f();
        unsafe {
            sys::cuEventRecord(end, stream);
            sys::cuEventSynchronize(end);
        }
        let mut ms: f32 = 0.0;
        let got = unsafe { sys::cuEventElapsedTime_v2(&mut ms, start, end) } == sys::CUresult::CUDA_SUCCESS;
        unsafe {
            sys::cuEventDestroy_v2(start);
            sys::cuEventDestroy_v2(end);
        }
        ran?;
        if !got {
            anyhow::bail!("cuEventElapsedTime failed");
        }
        Ok(std::time::Duration::from_secs_f64(ms as f64 / 1000.0))
    }

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().context("waiting for the stream")
    }
}

pub struct Module {
    module: Arc<cudarc::driver::CudaModule>,
    functions: std::sync::Mutex<std::collections::HashMap<String, CudaFunction>>,
}

impl Module {
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        if let Ok(m) = self.functions.lock() {
            if let Some(f) = m.get(name) {
                return Ok(Kernel { function: f.clone() });
            }
        }
        let function = self
            .module
            .load_function(name)
            .with_context(|| format!("no kernel named `{name}` in the module"))?;
        if let Ok(mut m) = self.functions.lock() {
            m.insert(name.to_string(), function.clone());
        }
        Ok(Kernel { function })
    }
}

pub struct Kernel {
    function: CudaFunction,
}

impl Kernel {
    pub fn function(&self) -> &CudaFunction {
        &self.function
    }
}

pub struct PinnedBuf {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    pub fn new(len: usize) -> Result<Self> {
        use cudarc::driver::sys;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        let r = unsafe { sys::cuMemHostAlloc(&mut ptr, len, sys::CU_MEMHOSTALLOC_PORTABLE) };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemHostAlloc({len}) failed: {r:?}");
        }
        Ok(Self { ptr, len })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        if !self.ptr.is_null() {
            unsafe { sys::cuMemFreeHost(self.ptr) };
        }
    }
}

pub fn alloc_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLEF_ALLOC_TRACE").is_ok())
}

pub struct Buffer {
    stream: Arc<CudaStream>,
    data: Arc<CudaSlice<u8>>,
    offset: usize,
    len: usize,
    recycle: Option<Arc<GraphFreeList>>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(list) = &self.recycle {
            list.give_back(self.offset, self.len);
            return;
        }

        if std::sync::Arc::strong_count(&self.data) != 1 {
            return;
        }
        alloc_cache_put(&self.data);
    }
}

static ALLOC_CACHE_HELD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn alloc_cache() -> &'static std::sync::Mutex<std::collections::HashMap<usize, Vec<std::sync::Arc<cudarc::driver::CudaSlice<u8>>>>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, Vec<std::sync::Arc<cudarc::driver::CudaSlice<u8>>>>>>
        = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn alloc_cache_cap() -> usize {
    static CAP: std::sync::OnceLock<std::sync::atomic::AtomicUsize> = std::sync::OnceLock::new();
    let cell = CAP.get_or_init(|| {
        let mib = env_once!("CLEF_ALLOC_CACHE_MIB")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048);
        std::sync::atomic::AtomicUsize::new(mib * 1024 * 1024)
    });
    let installed = ALLOC_CACHE_CAP.load(std::sync::atomic::Ordering::Relaxed);
    if installed > 0 { installed } else { cell.load(std::sync::atomic::Ordering::Relaxed) }
}

pub fn set_alloc_cache_cap(bytes: usize) {
    if env_once!("CLEF_ALLOC_CACHE_MIB").is_some() {
        return;
    }
    ALLOC_CACHE_CAP.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

static ALLOC_CACHE_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn alloc_cache_needed_bytes(tokens: usize, width: usize, hidden: usize, pass: usize) -> usize {
    (5 * tokens * width * 4) + (4 * pass * hidden * 4)
}

fn alloc_cache_take(bytes: usize) -> Option<std::sync::Arc<cudarc::driver::CudaSlice<u8>>> {
    if alloc_cache_cap() == 0 {
        return None;
    }
    let mut m = alloc_cache().lock().ok()?;
    let got = m.get_mut(&bytes).and_then(|v| v.pop());
    if got.is_some() {
        ALLOC_CACHE_HELD.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
    }
    got
}

fn alloc_cache_put(block: &std::sync::Arc<cudarc::driver::CudaSlice<u8>>) {
    let cap = alloc_cache_cap();
    if cap == 0 {
        return;
    }
    let bytes = block.len();
    if bytes > cap {
        return;
    }

    let held = ALLOC_CACHE_HELD.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    if held + bytes > cap {
        ALLOC_CACHE_HELD.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    if let Ok(mut m) = alloc_cache().lock() {
        m.entry(bytes).or_default().push(std::sync::Arc::clone(block));
    } else {
        ALLOC_CACHE_HELD.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

struct GraphFreeList {
    ranges: std::sync::Mutex<Vec<(usize, usize)>>,
    open: std::sync::atomic::AtomicBool,
}

impl GraphFreeList {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            ranges: std::sync::Mutex::new(Vec::new()),
            open: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn give_back(&self, offset: usize, len: usize) {
        if !self.open.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }

        let aligned = len.div_ceil(256) * 256;
        if let Ok(mut ranges) = self.ranges.lock() {
            ranges.push((offset, aligned));
        }
    }

    fn take(&self, bytes: usize) -> Option<usize> {
        let mut ranges = self.ranges.lock().ok()?;
        let mut best: Option<usize> = None;
        for (i, &(_, len)) in ranges.iter().enumerate() {
            if len >= bytes && best.is_none_or(|b| ranges[b].1 > len) {
                best = Some(i);
            }
        }
        let i = best?;
        let (offset, len) = ranges.swap_remove(i);
        if len > bytes {
            ranges.push((offset + bytes, len - bytes));
        }
        Some(offset)
    }

    fn clear(&self) {
        if let Ok(mut ranges) = self.ranges.lock() {
            ranges.clear();
        }
    }
}

#[derive(Debug)]
pub struct CaptureTooLarge {
    pub needed: usize,
    pub cap: usize,
}

impl std::fmt::Display for CaptureTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a graph for this pass needs {} MiB of working set and the graph memory cap is {} MiB",
            self.needed / (1 << 20),
            self.cap / (1 << 20)
        )
    }
}

impl std::error::Error for CaptureTooLarge {}

#[derive(Debug)]
pub struct CaptureOverrun {
    pub needed: usize,
}

impl std::fmt::Display for CaptureOverrun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the capture needed {} bytes and the block is too small", self.needed)
    }
}

impl std::error::Error for CaptureOverrun {}

fn tail_nan_bytes() -> usize {
    static ON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CLEF_TAIL_NAN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(0)
    })
}

fn graph_reuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLEF_GRAPH_REUSE").as_deref() != Ok("0"))
}

fn graph_block_initial_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("CLEF_GRAPH_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v >= 1 << 20)
            .unwrap_or(256 << 20)
    })
}

impl Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device_ptr(&self) -> u64 {
        let view = self.slice();
        let (ptr, guard) = view.device_ptr(&self.stream);
        let ptr = ptr as u64;
        drop(guard);
        ptr
    }

    pub fn slice(&self) -> CudaView<'_, u8> {
        self.data.slice(self.offset..self.offset + self.len)
    }

    pub fn slice_mut(&mut self) -> CudaViewMut<'_, u8> {
        let data = Arc::get_mut(&mut self.data)
            .expect("a shared buffer cannot be written through; take a fresh allocation instead");
        data.slice_mut(self.offset..self.offset + self.len)
    }

    pub fn clone_buffer(&self) -> Buffer {
        Buffer {
            stream: self.stream.clone(),
            data: Arc::clone(&self.data),
            offset: self.offset,
            len: self.len,
            recycle: None,
        }
    }
}

pub struct Captured {
    graph: cudarc::driver::CudaGraph,
    pub memcpy_nodes: usize,
    pub alloc_nodes: usize,
    pub free_nodes: usize,
}

fn count_memory_nodes(graph: cudarc::driver::sys::CUgraph) -> (usize, usize, usize) {
    use cudarc::driver::sys;
    let mut total: usize = 0;

    if unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut total) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return (0, 0, 0);
    }
    if total == 0 {
        return (0, 0, 0);
    }
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); total];

    if unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut total) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return (0, 0, 0);
    }
    let (mut allocs, mut frees) = (0usize, 0usize);
    let mut memcpys = 0usize;

    let mut sizes: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;

        if unsafe { sys::cuGraphNodeGetType(node, &mut kind) } != sys::CUresult::CUDA_SUCCESS {
            continue;
        }
        match kind {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => {
                allocs += 1;
                let mut params: sys::CUDA_MEM_ALLOC_NODE_PARAMS = unsafe { std::mem::zeroed() };

                if unsafe { sys::cuGraphMemAllocNodeGetParams(node, &mut params) }
                    == sys::CUresult::CUDA_SUCCESS
                {
                    *sizes.entry(params.bytesize).or_default() += 1;
                }
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => frees += 1,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => memcpys += 1,
            _ => {}
        }
    }
    if allocs > 0 {
        tracing::warn!(?sizes, "allocation node sizes in the captured graph");
    }
    if memcpys > 0 {
        tracing::warn!(memcpys, "memcpy nodes in the captured graph — check their sources are alive");
    }
    (allocs, frees, memcpys)
}

unsafe impl Send for Captured {}
unsafe impl Sync for Captured {}

impl Captured {
    pub fn launch(&self) -> Result<()> {
        let exec = self.graph.cu_graph_exec() as usize;
        tracing::debug!("cuGraphLaunch on exec {exec:#x}");
        self.graph.launch().context("replaying the captured graph")
    }

    pub fn exec_handle(&self) -> usize {
        self.graph.cu_graph_exec() as usize
    }
}

#[derive(Clone, Copy)]
pub struct KvCache<'a> {
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub len: &'a Buffer,
    pub pos_map: Option<&'a Buffer>,
    pub base_map: Option<&'a Buffer>,
    pub slot_map: Option<&'a Buffer>,
    pub capacity: usize,
    pub positions: usize,
}

#[derive(Clone, Copy, Default)]
pub struct BatchMap<'a> {
    pub seq_start: Option<&'a Buffer>,
    pub seq_end: Option<&'a Buffer>,
    pub real_end: usize,
    pub real_end_dev: Option<&'a Buffer>,
    pub chunk_start: Option<&'a Buffer>,
    pub state_slot: Option<&'a Buffer>,
    pub hist_save: Option<&'a [(u32, u32)]>,
    pub hist_save_dev: Option<&'a Buffer>,
    pub hist_save_n: usize,
    pub carry: Option<LayerCarry<'a>>,
}

#[derive(Clone, Copy)]
pub struct LayerCarry<'a> {
    pub gdn_state: Option<(&'a Buffer, &'a Buffer)>,
    pub conv_hist: Option<&'a Buffer>,
    pub kv: Option<KvCache<'a>>,
    pub cache_len: Option<&'a Buffer>,
    pub slot_map: Option<&'a Buffer>,
    pub rope_pos: Option<&'a Buffer>,
    pub slot: usize,
    pub real_len: usize,
    pub seq_start: Option<&'a Buffer>,
    pub seq_end: Option<&'a Buffer>,
    pub chunk_start: Option<&'a Buffer>,
    pub real_end: Option<&'a Buffer>,
    pub real_walk: Option<&'a Buffer>,
    pub hist_save: Option<&'a Buffer>,
    pub hist_slots: usize,
}

impl<'a> LayerCarry<'a> {
    pub const NONE: LayerCarry<'static> = LayerCarry {
        real_walk: None,
        gdn_state: None,
        conv_hist: None,
        kv: None,
        cache_len: None,
        slot_map: None,
        rope_pos: None,
        seq_start: None,
        seq_end: None,
        chunk_start: None,
        real_end: None,
        hist_save: None,
        hist_slots: 0,
        slot: 0,
        real_len: 0,
    };
}

impl<'a> BatchMap<'a> {
    pub const SINGLE: BatchMap<'static> = BatchMap {
        state_slot: None,
        hist_save: None,
        hist_save_dev: None,
        seq_start: None,
        chunk_start: None,
        seq_end: None,
        real_end: 0,
        real_end_dev: None,
        carry: None,
            hist_save_n: 0,
    };

    pub fn with_carry(mut self, carry: LayerCarry<'a>) -> Self {
        self.carry = Some(carry);
        self
    }
}

pub const SEQ_START_NULL: u64 = 0;

pub fn config(blocks: u32, threads: u32) -> LaunchConfig {
    config_3d(blocks, 1, threads)
}

pub fn row_blocks(rows: usize, cols: usize) -> u32 {
    let tpr = 256.min(32.max(cols.div_ceil(256) * 32));
    let rpb = 256 / tpr;
    rows.div_ceil(rpb) as u32
}

pub fn att_rows_for_mma() -> u32 { 64 }

pub fn att_partial_mma() -> bool {
    !matches!(env_once!("CLEF_HEAD_MMA"), Some(v) if v == "0")
}

pub fn att_noskip() -> bool {
    matches!(env_once!("CLEF_ATT_NOSKIP"), Some(v) if v != "0")
}

pub fn att_ws() -> bool {
    !matches!(env_once!("CLEF_ATT_WS"), Some(v) if v == "0")
}

pub fn combine_rows_per_block() -> u32 {
    4
}

pub fn rows_to_blocks_for(seq: usize, rows: u32) -> u32 {
    let rows_per_block = 2 * rows;
    ((seq as u32) + rows_per_block - 1) / rows_per_block
}

pub fn rows_to_blocks(seq: usize) -> u32 {
    rows_to_blocks_for(seq, att_rows_for(seq))
}

pub fn config_block_2d(x: u32, y: u32, bx: u32, by: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (x, y, 1),
        block_dim: (bx, by, 1),
        shared_mem_bytes: 0,
    }
}

fn split_k_for(m_blocks: usize, n_blocks: usize, k: usize, ktile: usize) -> usize {
    if env_once!("CLEF_SPLIT_K") == Some("1") {
        return 1;
    }

    let _ = m_blocks;
    (680 / n_blocks.max(1)).clamp(1, (k / ktile).max(1).min(32))
}

pub fn config_split(x: u32, y: u32, z: u32, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (x, y, z), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

pub fn config_grid_3d(x: u32, y: u32, z: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (x, y, z),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn config_3d(x: u32, y: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (x, y, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_kernel_carries_the_block_scaled_mma() {
        if GEMM_CUBIN.is_empty() {
            eprintln!("SKIP: kernels were not built (no nvcc at build time)");
            return;
        }
        let dir = std::env::temp_dir().join("aiis-cubin-check");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("gemm.cubin");
        std::fs::write(&path, GEMM_CUBIN).expect("write cubin");

        let out = std::process::Command::new("cuobjdump")
            .arg("-sass")
            .arg(&path)
            .output()
            .expect("cuobjdump");
        let sass = String::from_utf8_lossy(&out.stdout);
        assert!(
            sass.contains("OMMA"),
            "the embedded cubin has no OMMA instruction; SASS was:\n{sass}"
        );

        assert!(
            sass.contains("OMMA.SF.16864") && sass.contains("E2M1") && sass.contains("UE4M3"),
            "the embedded MMA is not the m16n8k64 E2M1/UE4M3 block-scaled form; SASS was:\n{sass}"
        );
    }

                            #[test]
    fn the_device_swizzle_matches_the_cpu_reference() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (rows, cols) in [(4096usize, 256usize), (12288, 256), (130, 6), (4096, 768), (12288, 768)] {
            let src: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();

            for conflict_free in [false, true] {
                let expected = crate::swizzle::swizzle(&src, rows, cols, conflict_free);
                let buffer = device.swizzle_scales(&src, rows, cols, conflict_free).expect("swizzle");
                let got = device.download(&buffer).expect("download");

                assert_eq!(got.len(), expected.len(), "length for {rows}x{cols}");
                if got != expected {
                    let first = got
                        .iter()
                        .zip(expected.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    panic!(
                        "device swizzle differs from the CPU reference for {rows}x{cols} at tile_m 128 \
                         (first at {first}: got {} want {})",
                        got[first], expected[first]
                    );
                }
                eprintln!(
                    "  {rows}x{cols}: {} bytes, matches the CPU reference (cf={conflict_free})",
                    got.len()
                );
            }
        }
        for (rows, cols) in [(768usize, 256usize), (192, 256), (2048, 256), (130, 6), (768, 768)] {
            let src: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();
            for conflict_free in [false, true] {
                let expected = crate::swizzle::swizzle_tiled(&src, rows, cols, 192, conflict_free);
                let buffer = device
                    .swizzle_scales_for_tile(&src, rows, cols, conflict_free, 192)
                    .expect("swizzle at tile_m 192");
                let got = device.download(&buffer).expect("download");

                assert_eq!(
                    got.len(),
                    expected.len(),
                    "length for {rows}x{cols} at tile_m 192 — the 96-row atom pads differently"
                );
                if got != expected {
                    let first = got
                        .iter()
                        .zip(expected.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    panic!(
                        "device swizzle differs from the CPU reference for {rows}x{cols} at tile_m 192 \
                         (first at {first}: got {} want {}) — the kernel is not honouring blk_mn=96",
                        got[first], expected[first]
                    );
                }
                eprintln!(
                    "  {rows}x{cols} @192: {} bytes, matches the CPU reference (cf={conflict_free})",
                    got.len()
                );
            }
        }
    }

    #[test]
    fn an_upload_is_byte_exact_and_does_not_widen() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let src: Vec<u8> = (0..1 << 20).map(|i| (i % 256) as u8).collect();
        let buffer = device.upload(&src).expect("upload");

        assert_eq!(buffer.len(), src.len(), "the device buffer changed size");
        let back = device.download(&buffer).expect("download");
        assert_eq!(back, src, "the upload round trip is not byte-exact");
        eprintln!("  {} bytes round-tripped exactly", src.len());
    }

    fn packed(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..rows * cols / 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xFF) as u8
            })
            .collect()
    }

    fn unit_scales(rows: usize, cols: usize) -> Vec<u8> {
        vec![0x38u8; rows * cols / 16]
    }

    #[test]
    fn the_conflict_free_flag_reaches_both_sides() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (rows, cols) = (16usize, 4usize);
        let src: Vec<u8> = (0..rows * cols).map(|i| (i / 4) as u8).collect();

        for cf in [false, true] {
            let cpu = crate::swizzle::swizzle(&src, rows, cols, cf);
            let gpu = device
                .download(&device.swizzle_scales(&src, rows, cols, cf).expect("swizzle"))
                .expect("download");
            let host_cf = SF_CONFLICT_FREE;
            eprintln!(
                "  cf={cf}: cpu[4]=row1? {} cpu[16]=row4? {} | gpu==cpu {} | SF_CONFLICT_FREE={host_cf}",
                cpu[4], cpu[16], gpu == cpu
            );
        }
    }

    #[test]
    fn the_gemm_matches_the_cpu_reference() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (m, n, k) in [(16usize, 8usize, 64usize), (32, 16, 128), (16, 8, 256)] {
            let a = packed(m, k, 0x1234_5678);
            let b = packed(n, k, 0x9ABC_DEF0);
            let sa = unit_scales(m, k);
            let sb = unit_scales(n, k);

            let reference = crate::model::gemm_ref::matmul(
                &crate::model::gemm_ref::Nvfp4Matrix {
                    packed: a.clone(), scales: sa.clone(), rows: m, cols: k, global_scale: 1.0,
                },
                &crate::model::gemm_ref::Nvfp4Matrix {
                    packed: b.clone(), scales: sb.clone(), rows: n, cols: k, global_scale: 1.0,
                },
            );

            let got = device
                .nvfp4_gemm(&a, &b, &sa, &sb, m, n, k, 1.0, 1.0)
                .expect("gemm");
            assert_eq!(got.len(), m * n);

            let mut worst = (0usize, 0.0f32);
            let mut n_bad = 0usize;
            let mut n_zero = 0usize;
            let mut bad_rows: Vec<usize> = Vec::new();
            let mut bad_cols: Vec<usize> = Vec::new();
            for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                let d = (g - r).abs();
                if d > worst.1 {
                    worst = (i, d);
                }

                if d > 1e-3 {
                    n_bad += 1;
                    if *g == 0.0 {
                        n_zero += 1;
                    }
                    if !bad_rows.contains(&(i / n)) {
                        bad_rows.push(i / n);
                    }
                    if !bad_cols.contains(&(i % n)) {
                        bad_cols.push(i % n);
                    }
                }
            }
            let scale = reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32;
            eprintln!(
                "  {m}x{n}x{k}: max |diff| = {:.3e} at {} (got {:.4}, want {:.4}), ref mean |x| = {:.3e}",
                worst.1, worst.0, got[worst.0], reference[worst.0], scale
            );
            eprintln!(
                "        shape: {n_bad}/{} bad, {n_zero} zero | rows {:?}{} cols {:?}{}",
                got.len(),
                &bad_rows[..bad_rows.len().min(10)],
                if bad_rows.len() > 10 { "..." } else { "" },
                &bad_cols[..bad_cols.len().min(10)],
                if bad_cols.len() > 10 { "..." } else { "" }
            );

            assert!(
                worst.1 < 1e-3 * scale.max(1.0),
                "{m}x{n}x{k} diverges: max |diff| {:.3e} at index {} (got {:.4}, want {:.4})",
                worst.1, worst.0, got[worst.0], reference[worst.0]
            );
        }
    }

    #[test]
    fn derive_the_a_layout() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (16usize, 8usize, 64usize);

        let mut b = vec![0u8; n * k / 2];
        for kk in 0..k {
            b[kk / 2] |= if kk % 2 == 0 { 0x02 } else { 0x20 };
        }
        let sb = unit_scales(n, k);

        let mut found: Vec<(usize, usize, usize)> = Vec::new();
        for row in 0..m {
            for kk in 0..k {
                let mut a = vec![0u8; m * k / 2];

                let byte = row * (k / 2) + kk / 2;
                a[byte] = if kk % 2 == 0 { 0x02 } else { 0x20 };
                let sa = unit_scales(m, k);

                let c = device
                    .nvfp4_gemm(&a, &b, &sa, &sb, m, n, k, 1.0, 1.0)
                    .expect("gemm");
                let nz: Vec<usize> = (0..m * n).filter(|&i| c[i].abs() > 0.5).collect();
                if nz.len() == 1 {
                    found.push((row, kk, nz[0]));
                } else if !nz.is_empty() {
                    eprintln!("  (row {row}, k {kk}) -> {} nonzeros: {nz:?}", nz.len());
                }
            }
        }
        eprintln!("  single-nonzero placements: {} of {}", found.len(), m * k);
        for (row, kk, out) in found.iter().take(24) {
            eprintln!("    A({row:2}, {kk:2}) -> C[{:2}][{}]", out / n, out % n);
        }

        let distinct: std::collections::HashSet<usize> = found.iter().map(|t| t.2).collect();
        eprintln!("  distinct C entries reached: {} of {}", distinct.len(), m * n);
    }

        #[test]
    fn sweep_every_a_position() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (16usize, 8usize, 64usize);
        let sa = unit_scales(m, k);
        let sb = unit_scales(n, k);

        let mut b = vec![0u8; n * k / 2];
        for kk in 0..k {
            b[kk / 2] |= if kk % 2 == 0 { 0x02 } else { 0x20 };
        }

        let mut bad: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for row in 0..m {
            for kk in 0..k {
                let mut a = vec![0u8; m * k / 2];
                let by = row * (k / 2) + kk / 2;
                a[by] = if kk % 2 == 0 { 0x02 } else { 0x20 };
                let c = device.nvfp4_gemm(&a, &b, &sa, &sb, m, n, k, 1.0, 1.0).expect("gemm");
                let nz: Vec<usize> = (0..m * n).filter(|&i| c[i].abs() > 0.5).collect();
                if nz.len() != 1 || nz[0] != row * n {
                    bad.push((row, kk, nz));
                }
            }
        }
        eprintln!("  wrong positions: {} of {}", bad.len(), m * k);
        for (row, kk, nz) in bad.iter().take(8) {
            let where_ = nz.iter().map(|i| format!("({},{})", i / n, i % n)).collect::<Vec<_>>().join(" ");
            eprintln!("    A({row:2},{kk:2}) -> got {where_}");
        }
        for (row, kk, nz) in bad.iter().take(16) {
            let where_ = nz.iter().map(|i| format!("({},{})", i / n, i % n)).collect::<Vec<_>>().join(" ");
            eprintln!("    A({row:2},{kk:2}) -> expected C[{row}][0], got {where_}");
        }

        let bad_set: std::collections::BTreeSet<(usize, usize)> =
            bad.iter().map(|b| (b.0, b.1)).collect();
        for row in [0usize, 1] {
            let right: Vec<usize> = (0..k).filter(|kk| !bad_set.contains(&(row, *kk))).collect();
            eprintln!("  row {row}: CORRECT k = {right:?}");
        }

        let rows_bad: std::collections::BTreeMap<usize, usize> =
            bad.iter().fold(Default::default(), |mut acc, b| { *acc.entry(b.0).or_insert(0) += 1; acc });
        eprintln!("  bad count per row: {rows_bad:?}");
    }

    #[test]
    fn derive_the_b_layout() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (16usize, 8usize, 64usize);
        let sa = unit_scales(m, k);
        let sb = unit_scales(n, k);

        let mut hits: Vec<(usize, usize, usize)> = Vec::new();
        let mut misses = 0usize;
        for n0 in 0..n {
            for k0 in 0..k {
                let mut a = vec![0u8; m * k / 2];
                a[k0 / 2] = if k0 % 2 == 0 { 0x02 } else { 0x20 };

                let mut b = vec![0u8; n * k / 2];
                let byte = n0 * (k / 2) + k0 / 2;
                b[byte] = if k0 % 2 == 0 { 0x02 } else { 0x20 };

                let c = device.nvfp4_gemm(&a, &b, &sa, &sb, m, n, k, 1.0, 1.0).expect("gemm");
                let nz: Vec<usize> = (0..m * n).filter(|&i| c[i].abs() > 0.5).collect();
                if nz.len() == 1 && nz[0] == n0 {
                    hits.push((n0, k0, nz[0]));
                } else {
                    misses += 1;
                    if misses <= 6 {
                        eprintln!("  MISS (n0={n0}, k0={k0}) -> {nz:?}");
                    }
                }
            }
        }
        eprintln!("  B layout: {} of {} (n,k) positions land at C[0][n]", hits.len(), n * k);
        eprintln!("  misses: {misses}");
        for (n0, k0, out) in hits.iter().take(16) {
            eprintln!("    B({n0}, {k0:2}) -> C[0][{out}]");
        }
    }

    #[test]
    fn dump_the_gemm_output() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (16usize, 8usize, 64usize);
        let sa = unit_scales(m, k);
        let sb = unit_scales(n, k);

        let mut b = vec![0u8; n * k / 2];
        for kk in 0..k {
            b[kk / 2] |= if kk % 2 == 0 { 0x02 } else { 0x20 };
        }

        let mut a = vec![0u8; m * k / 2];
        a[0] = 0x02;

        let c = device.nvfp4_gemm(&a, &b, &sa, &sb, m, n, k, 1.0, 1.0).expect("gemm");
        eprintln!("  A(0,0)=1, B(0,*)=1 -> C:");
        for row in 0..m {
            let vals: Vec<String> = (0..n).map(|col| format!("{:6.2}", c[row * n + col])).collect();
            eprintln!("    {row:2}: {}", vals.join(" "));
        }
        eprintln!("  nonzero entries: {}", c.iter().filter(|v| v.abs() > 0.5).count());

        let mut a2 = vec![0u8; m * k / 2];
        a2[8] = 0x02;
        let c2 = device.nvfp4_gemm(&a2, &b, &sa, &sb, m, n, k, 1.0, 1.0).expect("gemm");
        let nz: Vec<usize> = (0..m * n).filter(|&i| c2[i].abs() > 0.5).collect();
        eprintln!("  A(0,16)=1 -> nonzero at {nz:?}");
    }

                #[test]
    #[ignore = "needs the card"]
    fn the_gemms_answer_the_same_twice() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no card");
            return;
        };
        let bf16 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let as_u16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for m in [1024usize, 4096, 8192, 16384] {
            let (n, k) = (4096usize, 4096usize);

            let a: Vec<u8> = (0..m * k).map(|i| [0x38u8, 0x3C, 0x40, 0x38, 0xB8, 0x44][(i * 7 + i / k) % 6]).collect();
            let b: Vec<u8> = (0..n * k).map(|i| [0x38u8, 0x40, 0x3C, 0xB8, 0x38, 0x42][(i * 5 + i / k) % 6]).collect();
            let sa: Vec<u16> = (0..m).map(|i| bf16(1.0 + (i % 5) as f32 * 0.25)).collect();
            let sb: Vec<u16> = (0..n).map(|i| bf16(0.5 + (i % 3) as f32 * 0.5)).collect();
            let (ad, bd) = (device.upload(&a).unwrap(), device.upload(&b).unwrap());
            let (sad, sbd) = (device.upload(&as_u16(&sa)).unwrap(), device.upload(&as_u16(&sb)).unwrap());

            let runs: usize = std::env::var("CLEF_GEMM_RUNS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(if m >= 8192 { 200 } else { 20 });
            let first = device.fp8_gemm_dev(&ad, &bd, &sad, &sbd, m, n, k, None).expect("fp8");
            let first = device.download_rows_f32(&first, m, n).expect("dl");
            let mut worst = 0.0f32;
            let mut bad = 0usize;
            for _ in 0..runs {
                let o = device.fp8_gemm_dev(&ad, &bd, &sad, &sbd, m, n, k, None).expect("fp8");
                let x = device.download_rows_f32(&o, m, n).expect("dl");
                let d = first.iter().zip(&x).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                if d != 0.0 {
                    bad += 1;
                    let elems: Vec<usize> = first.iter().zip(&x).enumerate()
                        .filter(|(_, (p, q))| p != q).map(|(i, _)| i).collect();
                    let rows: Vec<usize> = elems.iter().map(|i| i / n).collect();
                    let (r0, r1) = (rows[0], *rows.last().unwrap());
                    let sample: Vec<(usize, f32, f32)> = elems.iter().take(6)
                        .map(|&i| (i, first[i], x[i])).collect();
                    let cols: Vec<usize> = elems.iter().map(|i| i % n).collect();
                    eprintln!(
                        "  fp8_gemm    [{m}, {n}, {k}] run differs: worst = {d:.3e}, {} elements over {} \
                         rows, rows {r0}..{r1}, cols {}..{}; samples (i, first, again) = {sample:?}",
                        elems.len(), rows.last().unwrap() - r0 + 1,
                        cols[0], *cols.last().unwrap()
                    );
                }
                worst = worst.max(d);
            }
            eprintln!("  fp8_gemm    [{m}, {n}, {k}]: worst = {worst:.3e}, differing runs = {bad}/{runs}");
            assert_eq!(worst, 0.0, "fp8_gemm answered differently on a repeat run");

            let act: Vec<f32> = (0..m * k).map(|i| ((i * 37 % 211) as f32 - 105.0) / 97.0).collect();
            let ad_f32 = device.upload_f32(&act).expect("upload");
            let mut qs = Vec::new();
            for _ in 0..12 {
                qs.push(device.quantise_fp8_dev(&ad_f32, m, k).expect("quantise"));
            }
            let (q0, s0) = (&qs[0].0, &qs[0].1);
            let q0 = device.download(q0).expect("dl");
            let s0 = device.download(s0).expect("dl");
            let mut qbad = 0usize;
            for (q, sc) in &qs[1..] {
                let q1 = device.download(q).expect("dl");
                let s1 = device.download(sc).expect("dl");
                if q1 != q0 || s1 != s0 {
                    qbad += 1;
                }
            }
            eprintln!("  quantise_fp8 [{m}, {k}]: differing runs = {qbad}/12");
            assert_eq!(qbad, 0, "quantise_fp8 answered differently on a repeat run");
        }
    }

    #[test]
    #[ignore = "needs the card"]
    fn the_rms_norm_answers_the_same_twice() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no card");
            return;
        };
        let cols = 4096usize;
        for rows in [1024usize, 4096, 8192, 16384, 24576] {
            let x: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 211) as f32 - 105.0) / 97.0).collect();
            let w: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) / 11.0).collect();
            let xd = device.upload_f32(&x).expect("upload");
            let wd = device.upload_f32(&w).expect("upload");
            let a = device.rms_norm_dev(&xd, &wd, rows, cols, 1e-6).expect("norm");
            let a = device.download_f32(&a).expect("dl");
            let b = device.rms_norm_dev(&xd, &wd, rows, cols, 1e-6).expect("norm");
            let b = device.download_f32(&b).expect("dl");
            let d = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            let f = a.iter().zip(&b).position(|(p, q)| p != q);
            eprintln!("  rms_norm [{rows}, {cols}] twice: worst = {d:.3e}, first {f:?}");
            assert_eq!(d, 0.0, "the RMS norm answered differently on the second run");
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090 and several GB; the M sweep"]
    fn the_fp8_gemm_m_sweep() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let n: usize = std::env::var("CLEF_SWEEP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(12288);
        let k: usize = std::env::var("CLEF_SWEEP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        let ms: Vec<usize> = match std::env::var("CLEF_SWEEP_MS").ok() {
            Some(v) if v == "buckets" => vec![16, 64, 128, 192, 256, 320, 512, 768, 1024],
            Some(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
            None => vec![512, 768, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        };
        let bf16 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let as_u16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        let b: Vec<u8> = (0..n * k)
            .map(|i| [0x38u8, 0x40, 0x3C, 0xB8, 0x38, 0x42][(i * 5 + i / k) % 6])
            .collect();
        let sb: Vec<u16> = (0..n).map(|i| bf16(0.5 + (i % 3) as f32 * 0.5)).collect();
        let b_dev = device.upload(&b).expect("b");
        let sb_dev = device.upload(&as_u16(&sb)).expect("sb");

        let sms = device
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(170)
            .max(1) as usize;

        let ctas_per_sm: usize = std::env::var("CLEF_CTAS_PER_SM")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        eprintln!("  fp8_gemm at n={n} k={k}, {sms} SMs, {ctas_per_sm} CTAs/SM = {} resident CTAs",
                  sms * ctas_per_sm);
        eprintln!("  {:>7}  {:>7}  {:>9}  {:>10}  {:>7}  {:>7}", "m", "ctas", "us", "TFLOP/s", "waves", "us/tile");

        for m in ms {
            let a: Vec<u8> = (0..m * k)
                .map(|i| [0x38u8, 0x3C, 0x40, 0x38, 0xB8, 0x44][(i * 7 + i / k) % 6])
                .collect();
            let sa: Vec<u16> = (0..m).map(|i| bf16(1.0 + (i % 5) as f32 * 0.25)).collect();
            let a_dev = device.upload(&a).expect("a");
            let sa_dev = device.upload(&as_u16(&sa)).expect("sa");

            let run = || {
                device
                    .fp8_gemm_dev(&a_dev, &b_dev, &sa_dev, &sb_dev, m, n, k, None)
                    .expect("gemm")
            };
            for _ in 0..2 {
                let _ = run();
            }
            device.sync().expect("sync");
            let iters = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = run();
            }
            device.sync().expect("sync");
            let dt = t0.elapsed().as_secs_f64() / iters as f64;

            let ctas = m.div_ceil(64) * n.div_ceil(64);
            let resident = sms * ctas_per_sm;
            eprintln!(
                "  {m:7}  {ctas:7}  {:9.1}  {:10.1}  {:7.2}  {:7.1}",
                dt * 1e6,
                2.0 * m as f64 * n as f64 * k as f64 / dt / 1e12,
                ctas as f64 / resident as f64,
                dt * 1e6 / ctas as f64,
            );
        }
    }

    #[test]
    #[ignore]
    fn the_fp8_gemm_k_scaling_probe() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (m, n, k) in [(16usize, 8usize, 64usize), (32, 8, 64), (48, 8, 64), (64, 8, 64),
                          (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 64),
                          (32, 16, 64), (32, 16, 128)] {            let a: Vec<u8> = (0..m * k).map(|i| [0x38u8, 0x3C, 0x40, 0x38][(i * 7 + i / k) % 4]).collect();
            let b: Vec<u8> = (0..n * k).map(|i| [0x38u8, 0x40, 0x3C, 0x38][(i * 5 + i / k) % 4]).collect();

            let sa: Vec<u8> = (0..m)
                .flat_map(|i| { let v = 1.0f32 + (i % 5) as f32 * 0.25; ((v.to_bits() >> 16) as u16).to_le_bytes() })
                .collect();
            let sb: Vec<u8> = (0..n)
                .flat_map(|i| { let v = 0.5f32 + (i % 3) as f32 * 0.5; ((v.to_bits() >> 16) as u16).to_le_bytes() })
                .collect();
            let reference = crate::model::gemm_ref::matmul_fp8(
                &crate::model::gemm_ref::Fp8Matrix {
                    data: a.clone(),
                    scales: sa.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
                    rows: m, cols: k,
                },
                &crate::model::gemm_ref::Fp8Matrix {
                    data: b.clone(),
                    scales: sb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
                    rows: n, cols: k,
                },
            );
            let got = device.fp8_gemm(&a, &b, &sa, &sb, m, n, k, None).expect("fp8 gemm");

            let ratios: Vec<f32> = got.iter().zip(reference.iter())
                .filter(|(_, r)| r.abs() > 1e-6).map(|(g, r)| g / r).collect();
            let mean = ratios.iter().sum::<f32>() / ratios.len().max(1) as f32;
            let zeros = got.iter().filter(|v| v.abs() < 1e-6).count();
            eprintln!("  m={m:3} n={n:3} k={k:3}  mean(got/want) = {mean:.4}  zeros {zeros}/{}  (got[0] {:.3}, want[0] {:.3})",
                      got.len(), got[0], reference[0]);
        }
    }

    #[test]
    #[ignore]
    fn the_two_attention_row_counts_agree_bit_for_bit() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let mut failures = Vec::new();
        for (seq, heads, kv_heads, hd) in
            [(1usize, 16usize, 4usize, 256usize), (2, 16, 4, 256), (3, 16, 4, 256), (8, 16, 4, 256),
             (15, 16, 4, 256), (16, 16, 4, 256), (17, 16, 4, 256), (33, 16, 4, 256),
             (48, 16, 4, 256), (64, 16, 4, 256), (96, 16, 4, 256), (128, 16, 4, 256),
             (37, 16, 4, 256), (129, 16, 4, 256), (765, 16, 4, 256)]
        {
            let mut seed = 0xA77_0001u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
            };
            let q: Vec<f32> = (0..seq * heads * hd).map(|_| rnd()).collect();
            let k: Vec<f32> = (0..seq * kv_heads * hd).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..seq * kv_heads * hd).map(|_| rnd()).collect();
            let (qd, kd, vd) = (
                device.upload_f32(&q).expect("up q"),
                device.upload_f32(&k).expect("up k"),
                device.upload_f32(&v).expect("up v"),
            );

            {
                let (la, lb) = (288usize, 272usize);
                let (h2, kh2, hd2) = (16usize, 4usize, 256usize);
                let mut seed2 = 0x51ed_2701u32;
                let mut rnd2 = || {
                    seed2 = seed2.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed2 >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
                };
                let mut mk = |n: usize, rows: usize| -> Vec<f32> {
                    (0..rows * n * hd2).map(|_| rnd2()).collect()
                };
                let (qa, ka, va) = (mk(h2, la), mk(kh2, la), mk(kh2, la));
                let (qb, kb, vb) = (mk(h2, lb), mk(kh2, lb), mk(kh2, lb));

                let run = |q: &[f32], k: &[f32], v: &[f32], rows: usize| -> Vec<f32> {
                    let (qd, kd, vd) = (
                        device.upload_f32(q).expect("up q"),
                        device.upload_f32(k).expect("up k"),
                        device.upload_f32(v).expect("up v"),
                    );
                    let out = device
                        .attention_dev_mma(&qd, &kd, &vd, rows, h2, kh2, hd2, None, None, 0, None, None)
                        .expect("separate");
                    device.download_f32(&out).expect("dl")
                };
                let ref_a = run(&qa, &ka, &va, la);
                let ref_b = run(&qb, &kb, &vb, lb);

                let cat = |a: &[f32], b: &[f32]| -> Vec<f32> {
                    let mut o = a.to_vec();
                    o.extend_from_slice(b);
                    o
                };
                let (qc, kc, vc) = (cat(&qa, &qb), cat(&ka, &kb), cat(&va, &vb));
                let total = la + lb;

                let mut ss: Vec<i32> = vec![0; la];
                ss.extend(std::iter::repeat(la as i32).take(lb));
                let (qcd, kcd, vcd) = (
                    device.upload_f32(&qc).expect("up qc"),
                    device.upload_f32(&kc).expect("up kc"),
                    device.upload_f32(&vc).expect("up vc"),
                );
                let ssd = device.upload_i32(&ss).expect("up seq_start");
                let out = device
                    .attention_dev_mma(&qcd, &kcd, &vcd, total, h2, kh2, hd2, Some(&ssd), None, 0, None, None)
                    .expect("batched");
                let got = device.download_f32(&out).expect("dl batched");

                let row = hd2 * h2;
                let mut worst_a = 0.0f32;
                let mut worst_b = 0.0f32;
                for t in 0..la {
                    for j in 0..row {
                        worst_a = worst_a.max((got[t * row + j] - ref_a[t * row + j]).abs());
                    }
                }
                for t in 0..lb {
                    for j in 0..row {
                        worst_b = worst_b
                            .max((got[(la + t) * row + j] - ref_b[t * row + j]).abs());
                    }
                }
                assert!(
                    worst_a < 1e-3 && worst_b < 1e-3,
                    "a concatenated pass must reproduce the separate ones: A differs by {worst_a:.3e}, \
                     B by {worst_b:.3e}. If B is the one that moved, the boundary was not applied and B's \
                     queries attended to A's keys - which is exactly what a batched pass must never do."
                );
                eprintln!("OK: batched {la}+{lb} matches separate, worst A {worst_a:.2e} B {worst_b:.2e}");
            }

            let a = device.attention_dev_rows(&qd, &kd, &vd, seq, heads, kv_heads, hd, 2, None)
                .expect("rows 2");

            if std::env::var("CLEF_ATT_MMA_REF").as_deref() == Ok("1") {
                let b = device.attention_dev_mma(&qd, &kd, &vd, seq, heads, kv_heads, hd, None, None, 0, None, None)
                    .expect("the MMA kernel");
                let r = device.download(&b).expect("dl mma");

                let bf = |x: f32| -> f32 {
                    let b = x.to_bits();
                    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
                    f32::from_bits(r & 0xffff_0000)
                };
                let qh: Vec<f32> = q.iter().map(|v| bf(*v)).collect();
                let kh: Vec<f32> = k.iter().map(|v| bf(*v)).collect();
                let vh: Vec<f32> = v.iter().map(|v| bf(*v)).collect();
                let (qs, ks_, vs_) = (1.0f32, 1.0f32, 1.0f32);
                let mut want = vec![0f32; seq * heads * hd];
                let gqa = heads / kv_heads;
                for t in 0..seq {
                    for hh in 0..heads {
                        let kv = hh / gqa;
                        let mut sc: Vec<f32> = (0..=t).map(|sq_| {
                            let mut acc = 0f32;
                            for d in 0..hd {
                                acc += qh[(t * heads + hh) * hd + d] * kh[(sq_ * kv_heads + kv) * hd + d];
                            }
                            acc * (1.0 / (hd as f32).sqrt())
                        }).collect();
                        let m = sc.iter().fold(f32::MIN, |a, &b| a.max(b));
                        let mut l = 0f32;
                        for x in sc.iter_mut() { *x = (*x - m).exp(); l += *x; }

                        let pq: Vec<f32> = sc.iter().map(|p| bf(*p)).collect();
                        let lq: f32 = pq.iter().sum();
                        for d in 0..hd {
                            let mut acc = 0f32;
                            for sq_ in 0..=t { acc += pq[sq_] * vh[(sq_ * kv_heads + kv) * hd + d]; }

                            want[(t * heads + hh) * hd + d] = acc / lq;
                        }
                    }
                }
                let _ = (qs, ks_, vs_);
                let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
                let mut first_bad: Option<(usize, usize, usize, f32, f32)> = None;
                for i in 0..want.len() {
                    let x = want[i];
                    let y = f32::from_le_bytes([r[i*4], r[i*4+1], r[i*4+2], r[i*4+3]]);
                    if (x - y).abs() > 1e-3 && first_bad.is_none() {
                        let d = i % hd;
                        let hh = (i / hd) % heads;
                        let t = i / (hd * heads);
                        first_bad = Some((t, hh, d, x, y));
                    }
                    mx = mx.max((x - y).abs());
                    num += ((x - y) as f64).powi(2);
                    den += (x as f64).powi(2);
                }
                if let Some((t, hh, d, x, y)) = first_bad {
                    eprintln!("            first mismatch: t={t} head={hh} dim={d}  want {x:.6} got {y:.6}");
                }
                eprintln!("  seq {seq:5}: MMA vs QUANTISED HOST REF  max|d| {mx:.3e}  rel-rms {:.3e}",
                          (num / den.max(1e-30)).sqrt());
                if seq <= 3 {
                    let g: Vec<f32> = (0..4).map(|i| f32::from_le_bytes([r[i*4], r[i*4+1], r[i*4+2], r[i*4+3]])).collect();
                    let w: Vec<f32> = (0..4).map(|i| want[i]).collect();
                    eprintln!("            buffer[0..4] = {g:?}\n            want  [0..4] = {w:?}");
                }
                continue;
            }

            if std::env::var("CLEF_ATT_MMA").as_deref() == Ok("1") {
                let b = device.attention_dev_mma(&qd, &kd, &vd, seq, heads, kv_heads, hd, None, None, 0, None, None)
                    .expect("the MMA kernel");
                let (p, r) = (device.download(&a).expect("dl a"), device.download(&b).expect("dl b"));
                let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
                for i in 0..p.len() / 4 {
                    let x = f32::from_le_bytes([p[i*4], p[i*4+1], p[i*4+2], p[i*4+3]]);
                    let y = f32::from_le_bytes([r[i*4], r[i*4+1], r[i*4+2], r[i*4+3]]);
                    mx = mx.max((x - y).abs());
                    num += ((x - y) as f64).powi(2);
                    den += (x as f64).powi(2);
                }
                let rms = (num / den.max(1e-30)).sqrt();
                eprintln!("  seq {seq:5}: MMA max|d| {mx:.3e}  rel-rms {rms:.3e}");
                if !rms.is_finite() || rms > 2e-2 {
                    failures.push(format!("seq {seq}: rel-rms {rms:.3e}"));
                }
                continue;
            }
            let rows_under_test: u32 = match std::env::var("CLEF_ATT_BISECT").as_deref() {
                Ok("4") => 4,
                Ok("12") => 12,
                Ok("0") => 0,
                Ok("wide") => 100,
                _ => 8,
            };
            let b = device.attention_dev_rows(&qd, &kd, &vd, seq, heads, kv_heads, hd, rows_under_test, None)
                .expect("rows under test");
            let (p, r) = (device.download(&a).expect("dl a"), device.download(&b).expect("dl b"));
            assert_eq!(p.len(), r.len(), "seq {seq}: the two row counts wrote different lengths");

            if rows_under_test == 100 {
                let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
                for i in 0..p.len() / 4 {
                    let a = f32::from_le_bytes([p[i*4], p[i*4+1], p[i*4+2], p[i*4+3]]);
                    let b = f32::from_le_bytes([r[i*4], r[i*4+1], r[i*4+2], r[i*4+3]]);
                    mx = mx.max((a - b).abs());
                    num += ((a - b) as f64).powi(2);
                    den += (a as f64).powi(2);
                }
                let rms = (num / den.max(1e-30)).sqrt();
                eprintln!("  seq {seq:5}: wide-lane max|d| {mx:.3e}  rel-rms {rms:.3e}");
                if rms > 1e-4 { failures.push(format!("seq {seq}: rel-rms {rms:.3e}")); }
                continue;
            }

            let mut diff = 0usize;
            let mut worst = (0usize, 0.0f32);
            for (i, (u, w)) in p.chunks_exact(4).zip(r.chunks_exact(4)).enumerate() {
                if u != w {
                    diff += 1;
                    let (fu, fw) = (f32::from_le_bytes([u[0], u[1], u[2], u[3]]),
                                    f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                    if (fu - fw).abs() > worst.1 {
                        worst = (i, (fu - fw).abs());
                    }
                }
            }

            let hd_us = hd;
            let mut rows_diff: std::collections::BTreeSet<usize> = Default::default();
            let mut heads_diff: std::collections::BTreeSet<usize> = Default::default();
            let mut dims_diff: std::collections::BTreeSet<usize> = Default::default();
            for i in 0..p.len() / 4 {
                if p[i * 4..i * 4 + 4] == r[i * 4..i * 4 + 4] { continue; }
                let d = i % hd_us;
                let hh = (i / hd_us) % heads;
                let t = i / (hd_us * heads);
                rows_diff.insert(t);
                heads_diff.insert(hh);
                dims_diff.insert(d);
            }
            eprintln!("  seq {seq:5}: {} of {} words differ{}", diff, p.len() / 4,
                      if diff == 0 { String::new() }
                      else { format!(", worst |d| {:.3e} at word {}", worst.1, worst.0) });
            if diff != 0 {
                let hd_us = hd;
                let show = |i: usize| -> String {
                    let a = f32::from_le_bytes([p[i*4], p[i*4+1], p[i*4+2], p[i*4+3]]);
                    let b = f32::from_le_bytes([r[i*4], r[i*4+1], r[i*4+2], r[i*4+3]]);
                    format!("{a:+.5} vs {b:+.5}")
                };
                let first_bad = *rows_diff.iter().next().unwrap();
                let last_ok = first_bad - 1;
                eprintln!("            row {last_ok} (ok)   : d0 {}  d1 {}  d2 {}",
                          show(last_ok * heads * hd_us), show(last_ok * heads * hd_us + 1),
                          show(last_ok * heads * hd_us + 2));
                eprintln!("            row {first_bad} (BAD) : d0 {}  d1 {}  d2 {}",
                          show(first_bad * heads * hd_us), show(first_bad * heads * hd_us + 1),
                          show(first_bad * heads * hd_us + 2));
                let rows: Vec<usize> = rows_diff.iter().copied().collect();
                let dims: Vec<usize> = dims_diff.iter().copied().collect();
                eprintln!("            rows {:?}{}  heads {} of {}  dims {} of {}",
                          &rows[..rows.len().min(10)], if rows.len() > 10 { "..." } else { "" },
                          heads_diff.len(), heads, dims.len(), hd_us);
            }
            if diff != 0 {
                failures.push(format!("seq {seq}: {diff} of {} words differ", p.len() / 4));
            }
        }
        assert!(failures.is_empty(),
                "ATT_ROWS changed the arithmetic at {} of the sizes tested:\n  {}",
                failures.len(), failures.join("\n  "));
    }

    #[test]
    #[ignore = "needs the RTX 5090; runs both fp8 atoms on real quantised data"]
    fn the_two_fp8_atoms_agree_bit_for_bit() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (m, n, k) in [(16usize, 8usize, 32usize), (32, 16, 64), (129, 65, 128),
                          (765, 4096, 4096), (765, 12288, 4096)] {
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i % 89) as f32 - 44.0) / 440.0).collect();
            let (a, sa) = device.quantise_fp8(&x, m, k).expect("quantise a");
            let (b, sb) = device.quantise_fp8(&w, n, k).expect("quantise w");
            let a_dev = device.upload(&a).expect("up a");
            let b_dev = device.upload(&b).expect("up b");
            let sa_dev = device.upload(&sa).expect("up sa");
            let sb_dev = device.upload(&sb).expect("up sb");

            let plain = device
                .fp8_gemm_dev_atom(true, &a_dev, &b_dev, &sa_dev, &sb_dev, m, n, k, None)
                .expect("plain atom");
            let scaled = device
                .fp8_gemm_dev_atom(false, &a_dev, &b_dev, &sa_dev, &sb_dev, m, n, k, None)
                .expect("block-scaled atom");
            let (p, q) = (device.download(&plain).expect("dl p"), device.download(&scaled).expect("dl q"));
            assert_eq!(p.len(), q.len());

            let mut diff_bits = 0usize;
            let mut worst = (0usize, 0.0f32);
            for (i, (u, v)) in p.chunks_exact(4).zip(q.chunks_exact(4)).enumerate() {
                if u != v {
                    diff_bits += 1;
                    let (fu, fv) = (f32::from_le_bytes([u[0], u[1], u[2], u[3]]),
                                    f32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                    let d = (fu - fv).abs();
                    if d > worst.1 { worst = (i, d); }
                }
            }
            let total = p.len() / 4;
            eprintln!("  {m:4}x{n:5}x{k:5}: {diff_bits} of {total} outputs differ in bit pattern{}",
                if diff_bits > 0 { format!(" (worst |diff| {:.3e} at {})", worst.1, worst.0) } else { String::new() });
            assert_eq!(diff_bits, 0,
                "{m}x{n}x{k}: the two FP8 atoms are NOT bit-identical — {diff_bits} of {total} differ");
        }
    }

    #[test]
    #[ignore]
    fn the_fp8_gemm_reads_every_k_position() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let bf16_one = ((1.0f32.to_bits() >> 16) as u16).to_le_bytes();
        for (m, n, k) in [(16usize, 8usize, 32usize), (16, 8, 64), (16, 8, 128), (16, 8, 192),
                          (64, 64, 128), (64, 64, 64)] {
            let sa: Vec<u8> = std::iter::repeat(bf16_one).take(m).flatten().collect();
            let sb: Vec<u8> = std::iter::repeat(bf16_one).take(n).flatten().collect();
            let mut missing = Vec::new();
            let mut worst_rows = Vec::new();
            for j in 0..k {
                let mut a = vec![0u8; m * k];
                for i in 0..m { a[i * k + j] = 0x38; }
                let mut b = vec![0u8; n * k];
                for i in 0..n { b[i * k + j] = 0x38; }
                let got = device.fp8_gemm(&a, &b, &sa, &sb, m, n, k, None).expect("fp8 gemm");
                let bad = got.iter().filter(|v| (**v - 1.0).abs() > 1e-6).count();
                if bad > 0 {
                    missing.push(j);
                    worst_rows.push(bad);
                }
            }
            let worst = worst_rows.iter().copied().max().unwrap_or(0);
            eprintln!("  {m:3}x{n:3}x{k:4}: {} of {k} k positions wrong (worst {worst} of {} elements)",
                      missing.len(), m * n);
            if !missing.is_empty() && missing.len() <= 16 {
                eprintln!("      wrong k positions: {missing:?}");
            }
        }
    }

    #[test]
    fn the_fp8_gemm_matches_the_cpu_reference() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (m, n, k) in [(16usize, 8usize, 32usize), (32, 16, 64), (16, 8, 128)] {
            let a: Vec<u8> = (0..m * k)
                .map(|i| [0x38u8, 0x3C, 0x40, 0x38, 0xB8, 0x44][(i * 7 + i / k) % 6])
                .collect();
            let b: Vec<u8> = (0..n * k)
                .map(|i| [0x38u8, 0x40, 0x3C, 0xB8, 0x38, 0x42][(i * 5 + i / k) % 6])
                .collect();

            let sa: Vec<u8> = (0..m)
                .flat_map(|i| {
                    let v = 1.0f32 + (i % 5) as f32 * 0.25;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            let sb: Vec<u8> = (0..n)
                .flat_map(|i| {
                    let v = 0.5f32 + (i % 3) as f32 * 0.5;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();

            let reference = crate::model::gemm_ref::matmul_fp8(
                &crate::model::gemm_ref::Fp8Matrix {
                    data: a.clone(), scales: sa.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
                    rows: m, cols: k,
                },
                &crate::model::gemm_ref::Fp8Matrix {
                    data: b.clone(), scales: sb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
                    rows: n, cols: k,
                },
            );

            let got = device.fp8_gemm(&a, &b, &sa, &sb, m, n, k, None).expect("fp8 gemm");
            assert_eq!(got.len(), m * n);

            let mut worst = (0usize, 0.0f32);
            for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                let d = (g - r).abs();
                if d > worst.1 {
                    worst = (i, d);
                }
            }
            let scale = reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32;
            eprintln!(
                "  FP8 {m}x{n}x{k}: max |diff| = {:.3e} at {} (got {:.4}, want {:.4}), ref mean |x| = {:.3e}",
                worst.1, worst.0, got[worst.0], reference[worst.0], scale
            );
            assert!(worst.1 <= 1e-3 * scale.max(1.0), "FP8 GEMM disagrees at {}", worst.0);
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_fp8_gemm_handles_tile_boundaries() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let bf16 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let a_bytes = |m: usize, k: usize| -> Vec<u8> {
            (0..m * k)
                .map(|i| [0x38u8, 0x3C, 0x40, 0x38, 0xB8, 0x44][(i * 7 + i / k) % 6])
                .collect()
        };
        let b_bytes = |n: usize, k: usize| -> Vec<u8> {
            (0..n * k)
                .map(|i| [0x38u8, 0x40, 0x3C, 0xB8, 0x38, 0x42][(i * 5 + i / k) % 6])
                .collect()
        };
        let as_u16 = |v: &[u16]| -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        };

        let check = |m: usize, n: usize, k: usize, rows: Option<(usize, usize)>| {
            let a = a_bytes(m, k);
            let b = b_bytes(n, k);
            let sa: Vec<u16> = (0..m).map(|i| bf16(1.0 + (i % 5) as f32 * 0.25)).collect();
            let sb: Vec<u16> = (0..n).map(|i| bf16(0.5 + (i % 3) as f32 * 0.5)).collect();

            let a_dev = device.upload(&a).expect("a");
            let b_dev = device.upload(&b).expect("b");
            let sa_dev = device.upload(&as_u16(&sa)).expect("sa");
            let sb_dev = device.upload(&as_u16(&sb)).expect("sb");
            let out = device
                .fp8_gemm_dev(&a_dev, &b_dev, &sa_dev, &sb_dev, m, n, k, None)
                .unwrap_or_else(|e| panic!("{m}x{n}x{k} failed to run: {e:#}"));
            let got = device
                .download_rows_f32(&out, m, n)
                .expect("download the caller's rows");
            let (r0, r1) = rows.unwrap_or((0, m));
            let reference = crate::model::gemm_ref::matmul_fp8(
                &crate::model::gemm_ref::Fp8Matrix {
                    data: a[r0 * k..r1 * k].to_vec(),
                    scales: sa[r0..r1].to_vec(),
                    rows: r1 - r0,
                    cols: k,
                },
                &crate::model::gemm_ref::Fp8Matrix {
                    data: b.clone(),
                    scales: sb.clone(),
                    rows: n,
                    cols: k,
                },
            );
            let mut worst = (0usize, 0.0f32);
            for r in r0..r1 {
                for c in 0..n {
                    let d = (got[r * n + c] - reference[(r - r0) * n + c]).abs();
                    if d > worst.1 {
                        worst = (r * n + c, d);
                    }
                }
            }
            let scale = reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32;
            eprintln!(
                "  {m:4}x{n:5}x{k:5} rows {r0}..{r1}: max |diff| = {:.3e} at {} (ref mean |x| {:.3e})",
                worst.1, worst.0, scale
            );
            assert!(
                worst.1 <= 1e-3 * scale.max(1.0),
                "{m}x{n}x{k} diverges over rows {r0}..{r1}: max |diff| {:.3e} at {} (row {}, col {})",
                worst.1,
                worst.0,
                worst.0 / n,
                worst.0 % n
            );
        };
        for (m, n, k) in [
            (1usize, 1usize, 32usize),
            (1, 16, 32),
            (16, 1, 32),
            (13, 13, 32),
            (17, 17, 32),
            (15, 16, 32),
            (16, 15, 32),
            (31, 33, 64),
            (70, 70, 32),
            (65, 129, 32),
            (129, 65, 64),
            (100, 100, 64),
            (7, 200, 32),
        ] {
            check(m, n, k, None);
        }
        for (m, n, k) in [
            (7usize, 12288usize, 4096usize),
            (7, 4096, 12288),
            (7, 8192, 4096),
            (16, 12288, 4096),
        ] {
            check(m, n, k, None);
        }
        for (m, n, k) in [
            (765usize, 12288usize, 4096usize),
            (765, 4096, 12288),
            (765, 8192, 4096),
        ] {
            check(m, n, k, Some((0, 16)));
            check(m, n, k, Some((m - 16, m)));
        }
    }

    #[test]
    fn the_fused_gdn_prep_is_byte_identical_to_the_chain() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (kh, kd, ratio) = (16usize, 128usize, 2usize);
        let vh = kh * ratio;
        let conv_dim = 2 * kh * kd + vh * kd;

        for tokens in [130usize, 128] {
            let mut seed = 0x9D0_9E9u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.05
            };
            let mixed: Vec<f32> = (0..tokens * conv_dim).map(|_| next()).collect();
            let mixed_d = device.upload_f32(&mixed).expect("upload mixed");

            let prep = device
                .gdn_prep_dev(&mixed_d, tokens, kh, kd, vh, conv_dim)
                .expect("gdn_prep_dev");
            let (q, k, v) = device.split_qkv_dev(&mixed_d, tokens, kh, kd, vh, kd).expect("split");
            let q = device.repeat_interleave_dev(&q, tokens, kh, ratio, kd).expect("rep q");
            let k = device.repeat_interleave_dev(&k, tokens, kh, ratio, kd).expect("rep k");
            let chain = device.gdn_norm_qk_dev(&q, &k, tokens, vh, kd).expect("norm_qk");

            let pairs: [(&str, &Buffer, &Buffer); 5] = [
                ("q_hi", &prep.q_hi, &chain.q_hi),
                ("q_lo", &prep.q_lo, &chain.q_lo),
                ("k_hi", &prep.k_hi, &chain.k_hi),
                ("k_lo", &prep.k_lo, &chain.k_lo),
                ("kn", &prep.kn, &chain.kn),
            ];
            for (name, a, b) in pairs {
                let x = device.download(a).expect("dl fused");
                let y = device.download(b).expect("dl chain");
                assert_eq!(x.len(), y.len(), "{name}: the two paths sized it differently");
                match x.iter().zip(y.iter()).position(|(p, q)| p != q) {
                    None => eprintln!("  {tokens} tokens: {name} byte-identical over {} bytes", x.len()),
                    Some(i) => panic!(
                        "{tokens} tokens: {name} differs at byte {i} of {}: fused {:#04x} against \
                         chain {:#04x}",
                        x.len(), x[i], y[i]
                    ),
                }
            }

            let a = device.download(&prep.v).expect("dl v fused");
            let b = device.download(&v).expect("dl v chain");
            match a.iter().zip(b.iter()).position(|(p, q)| p != q) {
                None => eprintln!("  {tokens} tokens: v byte-identical over {} bytes", a.len()),
                Some(i) => panic!("{tokens} tokens: v differs at byte {i}: {:#04x} against {:#04x}",
                                  a[i], b[i]),
            }
        }
    }

    #[test]
    fn the_fused_qk_norm_rope_equals_the_chain() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (heads, kv_heads, hd) = (16usize, 4usize, 256usize);
        let eps = 1e-6f32;
        for (tokens, rotary) in [(1usize, 64usize), (7, 64), (64, 64), (300, 64), (64, 256), (300, 256)] {
            let mut seed = 0x1234_5678u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.8
            };
            let qg: Vec<f32> = (0..tokens * heads * hd * 2).map(|_| next()).collect();
            let kraw: Vec<f32> = (0..tokens * kv_heads * hd).map(|_| next()).collect();
            let wq: Vec<f32> = (0..hd).map(|_| next()).collect();
            let wk: Vec<f32> = (0..hd).map(|_| next()).collect();

            let table_rows = 64usize;
            let cos: Vec<f32> = (0..table_rows * rotary).map(|i| ((i % 97) as f32 / 97.0).cos()).collect();
            let sin: Vec<f32> = (0..table_rows * rotary).map(|i| ((i % 97) as f32 / 97.0).sin()).collect();

            let pos: Vec<i32> = (0..tokens).map(|t| ((t * 3 + 5) % 60) as i32).collect();

            let dqg = device.upload_f32(&qg).expect("qg");
            let dk = device.upload_f32(&kraw).expect("k");
            let dwq = device.upload_f32(&wq).expect("wq");
            let dwk = device.upload_f32(&wk).expect("wk");
            let dc = device.upload_f32(&cos).expect("cos");
            let ds = device.upload_f32(&sin).expect("sin");
            let dp = device.upload_i32(&pos).expect("pos");
            let (cq, cg) = device.split_q_gate_dev(&dqg, tokens, heads, hd).expect("split");
            let cq = device.rms_norm_dev(&cq, &dwq, tokens * heads, hd, eps).expect("nq");
            let ck = device.rms_norm_dev(&dk, &dwk, tokens * kv_heads, hd, eps).expect("nk");
            let cq = device.rope_apply_dev(&cq, &dc, &ds, tokens, heads, hd, rotary, None, None, Some(&dp))
                .expect("rq");
            let ck = device.rope_apply_dev(&ck, &dc, &ds, tokens, kv_heads, hd, rotary, None, None, Some(&dp))
                .expect("rk");
            let (fq, fg) = device.qk_norm_rope_dev(&dqg, &dwq, &dc, &ds, tokens * heads, heads, hd,
                                                   rotary, eps, None, None, Some(&dp), true)
                .expect("fused q");
            let (fk, _) = device.qk_norm_rope_dev(&dk, &dwk, &dc, &ds, tokens * kv_heads, kv_heads, hd,
                                                  rotary, eps, None, None, Some(&dp), false)
                .expect("fused k");
            let fg = fg.expect("the q side produces a gate");
            for (name, a, b) in [("q", &fq, &cq), ("gate", &fg, &cg), ("k", &fk, &ck)] {
                let x = device.download(a).expect("dl fused");
                let y = device.download(b).expect("dl chain");
                assert_eq!(x.len(), y.len(), "{name}: the two paths sized it differently");
                match x.iter().zip(y.iter()).position(|(p, q)| p != q) {
                    None => eprintln!("  [{tokens} tokens] {name:>5} worst 0 over {} bytes", x.len()),
                    Some(i) => panic!(
                        "{tokens} tokens: {name} differs at byte {i} of {}: fused {:#04x} against \
                         chain {:#04x} (element {})",
                        x.len(), x[i], y[i], i / 4
                    ),
                }
            }
        }
    }

    #[test]
    fn the_gated_fp8_gemm_equals_the_pair() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (m, n, k) in [(64usize, 256usize, 128usize), (300, 512, 384)] {
            let mut seed = 0xBEEF_1234u32 ^ ((m * 31 + n) as u32);
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32
            };
            let a: Vec<u8> = (0..m * k).map(|_| (next() * 255.0) as u8).collect();
            let bg: Vec<u8> = (0..n * k).map(|_| (next() * 255.0) as u8).collect();
            let bu: Vec<u8> = (0..n * k).map(|_| (next() * 255.0) as u8).collect();

            let bf = |x: f32| -> u16 {
                let b = x.to_bits();
                let lsb = (b >> 16) & 1;
                ((b.wrapping_add(0x7fff + lsb)) >> 16) as u16
            };
            let sa: Vec<u16> = (0..m).map(|_| bf(1.0 / 64.0 + next() / 64.0)).collect();
            let sbg: Vec<u16> = (0..n).map(|_| bf(1.0 / 64.0 + next() / 64.0)).collect();
            let sbu: Vec<u16> = (0..n).map(|_| bf(1.0 / 64.0 + next() / 64.0)).collect();
            let da = device.upload(&a).expect("a");
            let dbg = device.upload(&bg).expect("bg");
            let dbu = device.upload(&bu).expect("bu");

            let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let dsa = device.upload(&as_bytes(&sa)).expect("sa");
            let dsbg = device.upload(&as_bytes(&sbg)).expect("sbg");
            let dsbu = device.upload(&as_bytes(&sbu)).expect("sbu");

            let g = device.fp8_gemm_dev(&da, &dbg, &dsa, &dsbg, m, n, k, None).expect("gate");
            let u = device.fp8_gemm_dev(&da, &dbu, &dsa, &dsbu, m, n, k, None).expect("up");
            let act = device.swiglu_dev(&g, &u, m * n).expect("swiglu");

            let fused = device
                .fp8_gemm_gated_dev(&da, &dsa, &dbg, &dsbg, &dbu, &dsbu, m, n, k)
                .expect("gated");

            let p = device.download_f32(&fused).expect("dl fused");
            let q = device.download_f32(&act).expect("dl chain");

            let worst = (0..m * n).fold(0.0f32, |w, i| w.max((p[i] - q[i]).abs()));
            let scale = q[..m * n].iter().fold(0.0f32, |w, v| w.max(v.abs()));
            eprintln!("  [{m}x{n}x{k}] gated vs chain: worst |diff| {worst:.3e} against max |act| {scale:.3e}");
            assert!(
                worst == 0.0,
                "{m}x{n}x{k}: the gated FP8 GEMM differs from gate+up+swiglu by {worst:.3e} against a \
                 value of {scale:.3e}"
            );
        }
    }

    #[test]
    fn the_fused_gated_norm_equals_the_chain() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (cols, vh) = (128usize, 32usize);
        let eps = 1e-6f32;
        let flat_cols = vh * cols;
        for tokens in [7usize, 300] {
            let rows = tokens * vh;
            let mut seed = 0x9E37_79B9u32 ^ (tokens as u32);
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 1.1
            };
            let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let g: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let w: Vec<f32> = (0..cols).map(|_| 0.5 + next() * 0.1).collect();
            let dx = device.upload_f32(&x).expect("x");
            let dg = device.upload_f32(&g).expect("gate");
            let dw = device.upload_f32(&w).expect("weight");

            let gated = device.rms_norm_gated_dev(&dx, &dg, &dw, rows, cols, eps).expect("gated");
            let tile_m = Device::nvfp4_m_tile_for(tokens);
            let (pc, sc) = device.quantise_nvfp4_dev(&gated, tokens, flat_cols, 1.0 / 12.5, tile_m)
                .expect("nvfp4 chain");
            let (qc, fsc) = device.quantise_fp8_dev(&gated, tokens, flat_cols).expect("fp8 chain");
            let (pf, sf) = device
                .rms_norm_gated_quantise_nvfp4_dev(&dx, &dg, &dw, rows, cols, vh, eps, 1.0 / 12.5, tile_m)
                .expect("nvfp4 fused");
            let (qf, fsf) = device
                .rms_norm_gated_quantise_fp8_dev(&dx, &dg, &dw, rows, cols, vh, eps)
                .expect("fp8 fused");
            for (name, a, b) in [("nvfp4 packed", &pf, &pc), ("nvfp4 scales", &sf, &sc),
                                 ("fp8 out", &qf, &qc), ("fp8 scales", &fsf, &fsc)] {
                let p = device.download(a).expect("dl fused");
                let q = device.download(b).expect("dl chain");
                assert_eq!(p.len(), q.len(), "{tokens} tokens: {name} sized differently");
                match p.iter().zip(q.iter()).position(|(u, w)| u != w) {
                    None => eprintln!("  [{tokens} tokens] {name:>14} worst 0 over {} bytes", p.len()),
                    Some(i) => panic!(
                        "{tokens} tokens: {name} differs at byte {i} of {}: fused {:#04x} against \
                         chain {:#04x}", p.len(), p[i], q[i]
                    ),
                }
            }
        }
    }

    #[test]
    fn the_fused_norm_dual_equals_the_chain() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let cols = 4096usize;
        let eps = 1e-6f32;
        for rows in [1usize, 7, 300, 1280] {
            let mut seed = 0x51ED_2701u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 1.7
            };
            let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let w: Vec<f32> = (0..cols).map(|_| next() * 0.3).collect();
            let dx = device.upload_f32(&x).expect("x");
            let dw = device.upload_f32(&w).expect("w");

            let normed = device.rms_norm_dev(&dx, &dw, rows, cols, eps).expect("norm");
            let (q, qs) = device.quantise_fp8_dev(&normed, rows, cols).expect("quantise");
            let bf = device.to_bf16_dev(&normed, rows * cols).expect("bf16");
            let (fq, fqs, fbf) = device
                .rms_norm_fp8_bf16_dev(&dx, &dw, rows, cols, eps)
                .expect("the fused norm declines only when cols != 4096");
            for (name, fused, chain) in [("q", &fq, &q), ("q_scales", &fqs, &qs), ("bf16", &fbf, &bf)] {
                let p = device.download(fused).expect("dl fused");
                let c = device.download(chain).expect("dl chain");
                assert_eq!(p.len(), c.len(), "{rows} rows: {name} sized differently");
                match p.iter().zip(c.iter()).position(|(u, w)| u != w) {
                    None => eprintln!("  [{rows} rows] {name:>9} worst 0 over {} bytes", p.len()),
                    Some(i) => panic!(
                        "{rows} rows: {name} differs at byte {i} of {}: fused {:#04x} against chain \
                         {:#04x}",
                        p.len(), p[i], c[i]
                    ),
                }
            }
        }

        let wide = device.upload_f32(&vec![0.0f32; 8 * 2048]).expect("wide");
        let w2048 = device.upload_f32(&vec![0.0f32; 2048]).expect("w2048");
        assert!(
            device.rms_norm_fp8_bf16_dev(&wide, &w2048, 8, 2048, eps).is_none(),
            "the fused norm must decline at cols != 4096"
        );
    }

    #[test]
    fn the_fused_conv_prep_equals_the_chain() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (kh, kd, vh, k) = (16usize, 128usize, 32usize, 4usize);
        let conv_dim = kh * kd * 2 + vh * kd;
        for rows in [64usize, 300, 1280] {
            let mut seed = 0x9e37_79b9u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.7
            };
            let x: Vec<f32> = (0..rows * conv_dim).map(|_| next()).collect();
            let cw: Vec<f32> = (0..conv_dim * k).map(|_| next()).collect();

            let hist: Vec<f32> = (0..2 * (k - 1) * conv_dim).map(|_| next()).collect();
            let boundary = 40.min(rows);
            let seq_start: Vec<i32> = (0..rows)
                .map(|t| if t < boundary { 0 } else { boundary as i32 })
                .collect();
            let slots: Vec<i32> = (0..rows).map(|t| if t < boundary { 0 } else { 1 }).collect();
            let dx = device.upload_f32(&x).expect("x");
            let dw = device.upload_f32(&cw).expect("w");
            let dh = device.upload_f32(&hist).expect("hist");
            let dss = device.upload_i32(&seq_start).expect("seq_start");
            let dsl = device.upload_i32(&slots).expect("slot_map");

            for mapped in [true, false] {
                let (ss, hh, sl) = if mapped {
                    (Some(&dss), Some(&dh), Some(&dsl))
                } else {
                    (None, None, None)
                };
                let mixed = device
                    .causal_conv_silu_dev(&dx, &dw, rows, conv_dim, k, ss, hh, sl)
                    .expect("conv");
                let a = device.gdn_prep_dev(&mixed, rows, kh, kd, vh, conv_dim).expect("chain");
                let b = device
                    .gdn_conv_prep_dev(&dx, &dw, rows, kh, kd, vh, conv_dim, k, ss, hh, sl)
                    .expect("fused");
                let pairs: [(&str, &Buffer, &Buffer); 6] = [
                    ("v", &b.v, &a.v),
                    ("q_hi", &b.q_hi, &a.q_hi),
                    ("q_lo", &b.q_lo, &a.q_lo),
                    ("k_hi", &b.k_hi, &a.k_hi),
                    ("k_lo", &b.k_lo, &a.k_lo),
                    ("kn", &b.kn, &a.kn),
                ];
                for (name, fused, chain) in pairs {
                    let p = device.download(fused).expect("dl fused");
                    let q = device.download(chain).expect("dl chain");
                    assert_eq!(p.len(), q.len(), "{name}: the two paths sized it differently");
                    let ndiff = p.iter().zip(q.iter()).filter(|(u, w)| u != w).count();
                    if ndiff > 0 {
                        eprintln!("  [{rows} rows, mapped={mapped}] {name:>5}: {ndiff} of {} bytes differ ({:.2e})", p.len(), ndiff as f64 / p.len() as f64);
                    }
                    match p.iter().zip(q.iter()).position(|(u, w)| u != w) {
                        None => eprintln!(
                            "  [{rows} rows, mapped={mapped}] {name:>5} worst 0 over {} bytes",
                            p.len()
                        ),
                        Some(i) => {
                            let e = i / 4;
                            let mut ctx = String::new();
                            for k in e.saturating_sub(3)..(e + 4).min(p.len() / 4) {
                                ctx.push_str(&format!(
                                    " [{k}] {:.9e}/{:.9e}",
                                    f32::from_bits(u32::from_le_bytes([p[k*4], p[k*4+1], p[k*4+2], p[k*4+3]])),
                                    f32::from_bits(u32::from_le_bytes([q[k*4], q[k*4+1], q[k*4+2], q[k*4+3]]))
                                ));
                            }
                            panic!(
                            "{rows} rows, mapped={mapped}: {name} differs at byte {i} of {} (element {e}): fused {:#04x} against chain {:#04x}{ctx}",
                            p.len(), p[i], q[i]
                        )
                        },
                    }
                }
            }
        }
    }

    #[test]
    fn the_gdn_scan_matches_the_cpu_oracle() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (seq, heads, k_dim, v_dim) in [(8usize, 4usize, 128usize, 128usize), (17, 2, 128, 128)] {
            let mut seed = 0x2468_1357u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };
            let q: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let k: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..seq * heads * v_dim).map(|_| next()).collect();

            let decay: Vec<f32> = (0..seq * heads).map(|_| -0.05 - 0.4 * (next().abs())).collect();
            let beta: Vec<f32> = (0..seq * heads).map(|_| 0.2 + 0.6 * next().abs()).collect();

            let mut oq = q.clone();
            let mut ok = k.clone();
            let mut want = vec![0.0f32; seq * heads * v_dim];
            crate::model::deltanet::gated_delta_rule(
                &mut oq, &mut ok, &v, &decay, &beta, seq, heads, k_dim, v_dim, &mut want,
            );

            let got = device
                .gdn_scan(&q, &k, &v, &decay, &beta, seq, heads, k_dim, v_dim)
                .expect("gdn");

            let mut worst = (0usize, 0.0f32);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let d = (g - w).abs();
                if d > worst.1 {
                    worst = (i, d);
                }
            }
            let scale = want.iter().map(|x| x.abs()).sum::<f32>() / want.len() as f32;
            eprintln!(
                "  GDN seq={seq} heads={heads} k={k_dim} v={v_dim}: max |diff| = {:.3e} at {} \
                 (got {:.6}, want {:.6}), mean |x| = {:.3e}",
                worst.1, worst.0, got[worst.0], want[worst.0], scale
            );
            assert!(
                worst.1 <= 1e-4 * scale.max(1.0),
                "GDN disagrees at {}: got {} want {}",
                worst.0,
                got[worst.0],
                want[worst.0]
            );
        }
    }

    #[test]
    fn the_chunked_scan_matches_the_recurrent_one() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for (seq, heads, k_dim, v_dim, rate) in [
            (64usize, 4usize, 128usize, 128usize, 0.02f32),
            (17, 2, 128, 128, 0.02),
            (130, 2, 128, 128, 0.02),
            (200, 4, 128, 128, 0.02),
            (200, 4, 128, 128, 3.0),
        ] {
            let mut seed = 0x1357_9bdfu32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };
            let q: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let k: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..seq * heads * v_dim).map(|_| next()).collect();
            let decay: Vec<f32> = (0..seq * heads).map(|_| -(0.01 + rate * next().abs())).collect();
            let beta: Vec<f32> = (0..seq * heads).map(|_| 0.2 + 0.6 * next().abs()).collect();

            let mut oq = q.clone();
            let mut ok = k.clone();
            let mut want = vec![0.0f32; seq * heads * v_dim];
            crate::model::deltanet::gated_delta_rule(
                &mut oq, &mut ok, &v, &decay, &beta, seq, heads, k_dim, v_dim, &mut want,
            );

            let rec = device
                .gdn_scan(&q, &k, &v, &decay, &beta, seq, heads, k_dim, v_dim)
                .expect("gdn");
            let (dq, dk, dv, dd, db) = (
                device.upload_f32(&q).expect("q"),
                device.upload_f32(&k).expect("k"),
                device.upload_f32(&v).expect("v"),
                device.upload_f32(&decay).expect("decay"),
                device.upload_f32(&beta).expect("beta"),
            );
            let raw = device
                .gdn_chunk_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim, crate::cuda::BatchMap::SINGLE)
                .expect("chunked");
            let bytes = device.download(&raw).expect("download");
            let chk: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let scale = want.iter().map(|x| x.abs()).sum::<f32>() / want.len() as f32;
            let report = |name: &str, got: &[f32]| {
                let mut worst = (0usize, 0.0f32);
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    let d = (g - w).abs();
                    if d > worst.1 {
                        worst = (i, d);
                    }
                }
                eprintln!(
                    "  {name} seq={seq} heads={heads} rate={rate}: max |diff| = {:.3e} at {} \
                     (got {:.6}, want {:.6}), mean |x| = {:.3e}",
                    worst.1, worst.0, got[worst.0], want[worst.0], scale
                );

                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "{name} seq={seq} heads={heads} rate={rate}: non-finite at {:?} of {}, \
                     got[{:?}]={:?} want[{:?}]={:?}",
                    got.iter().position(|v| !v.is_finite()),
                    got.len(),
                    got.iter().position(|v| !v.is_finite()),
                    got.iter().find(|v| !v.is_finite()),
                    want.iter().position(|v| !v.is_finite()),
                    want.iter().find(|v| !v.is_finite())
                );
                assert!(
                    worst.1 <= 1e-4 * scale.max(1.0),
                    "{name} disagrees at {}: got {} want {}",
                    worst.0,
                    got[worst.0],
                    want[worst.0]
                );
            };
            report("recurrent", &rec);
            report("chunked  ", &chk);
        }
    }

    #[test]
    #[ignore]
    fn the_gdn_scan_profile_target() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (seq, heads, k_dim, v_dim) = (765usize, 32usize, 128usize, 128usize);
        let mut seed = 0x2468_1357u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..seq * heads * v_dim).map(|_| next()).collect();
        let decay: Vec<f32> = (0..seq * heads).map(|_| -0.002 - 0.02 * next().abs()).collect();
        let beta: Vec<f32> = (0..seq * heads).map(|_| 0.2 + 0.6 * next().abs()).collect();
        let (dq, dk, dv, dd, db) = (
            device.upload_f32(&q).expect("q"),
            device.upload_f32(&k).expect("k"),
            device.upload_f32(&v).expect("v"),
            device.upload_f32(&decay).expect("decay"),
            device.upload_f32(&beta).expect("beta"),
        );

        let _ = device.gdn_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim).expect("rec");
        let _ = device
            .gdn_chunk_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim, crate::cuda::BatchMap::SINGLE)
            .expect("chunked");
        device.sync().expect("sync");
        let _ = device.gdn_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim).expect("rec");
        let _ = device
            .gdn_chunk_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim, crate::cuda::BatchMap::SINGLE)
            .expect("chunked");
        device.sync().expect("sync");
    }

    #[test]
    #[ignore]
    fn the_gdn_scan_timing() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let only: Option<usize> = std::env::var("CLEF_GDN_TIMING_SEQ").ok()
            .and_then(|v| v.parse().ok());
        for (seq, heads, k_dim, v_dim) in
            [(176usize, 32usize, 128usize, 128usize), (309, 32, 128, 128), (765, 32, 128, 128),
             (8192, 32, 128, 128)]
        {
            if let Some(n) = only { if seq != n { continue; } }
            let mut seed = 0x2468_1357u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };
            let q: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let k: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..seq * heads * v_dim).map(|_| next()).collect();
            let decay: Vec<f32> = (0..seq * heads).map(|_| -0.002 - 0.02 * next().abs()).collect();
            let beta: Vec<f32> = (0..seq * heads).map(|_| 0.2 + 0.6 * next().abs()).collect();
            let (dq, dk, dv, dd, db) = (
                device.upload_f32(&q).expect("q"),
                device.upload_f32(&k).expect("k"),
                device.upload_f32(&v).expect("v"),
                device.upload_f32(&decay).expect("decay"),
                device.upload_f32(&beta).expect("beta"),
            );

            const N: usize = 30;
            let mut time = |chunked: bool| -> f64 {
                let run = || {
                    if chunked {
                        device.gdn_chunk_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim, crate::cuda::BatchMap::SINGLE)
                    } else {
                        device.gdn_scan_dev(&dq, &dk, &dv, &dd, &db, seq, heads, k_dim, v_dim)
                    }
                };
                let _ = run().expect("warm-up");
                device.sync().expect("sync");
                let t = std::time::Instant::now();
                for _ in 0..N {
                    let _ = run().expect("run");
                }
                device.sync().expect("sync");
                t.elapsed().as_secs_f64() * 1e6 / N as f64
            };
            let rec = time(false);
            let chk = time(true);
            eprintln!(
                "  GDN seq={seq} heads={heads}: recurrent {rec:8.1} us, chunked {chk:8.1} us \
                 — {:.2}x",
                rec / chk
            );
        }
    }

    #[test]
    fn the_elementwise_kernels_match_the_cpu_oracle() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (rows, cols) = (5usize, 128usize);
        let x: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 11.0).collect();

        let w_plain: Vec<f32> = (0..cols).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let w_gated: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 5) as f32 / 4.0).collect();
        let eps = 1e-6f32;

        let got = device.rms_norm(&x, &w_plain, rows, cols, eps).expect("rms_norm");
        let mut want = vec![0.0f32; rows * cols];
        for r in 0..rows {
            crate::model::norm::rms_norm(
                &x[r * cols..(r + 1) * cols], &w_plain, eps, &mut want[r * cols..(r + 1) * cols],
            );
        }
        let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  rms_norm: max |diff| = {d:.3e}");
        assert!(d < 1e-4, "rms_norm disagrees by {d}");

        let gate: Vec<f32> = (0..rows * cols).map(|i| ((i * 13 % 61) as f32 - 30.0) / 9.0).collect();
        let got = device
            .rms_norm_gated(&x, &gate, &w_gated, rows, cols, eps)
            .expect("rms_norm_gated");
        let mut want = vec![0.0f32; rows * cols];
        for r in 0..rows {
            crate::model::norm::rms_norm_gated(
                &x[r * cols..(r + 1) * cols],
                &gate[r * cols..(r + 1) * cols],
                &w_gated,
                eps,
                &mut want[r * cols..(r + 1) * cols],
            );
        }
        let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  rms_norm_gated: max |diff| = {d:.3e}");
        assert!(d < 1e-4, "rms_norm_gated disagrees by {d}");

        let total = rows * cols - 1;
        let b: Vec<f32> = (0..total).map(|i| ((i * 17 % 43) as f32 - 20.0) / 5.0).collect();
        let xa: Vec<f32> = x[..total].to_vec();
        let xd = device.upload_f32(&xa).expect("upload x");
        let bd = device.upload_f32(&b).expect("upload b");
        let got = device.download_f32(&device.add_dev(&xd, &bd, total).expect("add")).expect("dl");
        let want: Vec<f32> = xa.iter().zip(b.iter()).map(|(p, q)| p + q).collect();
        let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  add: max |diff| = {d:.3e} over {total} elements");
        assert_eq!(got.len(), total, "add returned the wrong length");
        assert!(d < 1e-6, "add disagrees by {d}");

        let up: Vec<f32> = (0..rows * cols).map(|i| ((i * 29 % 71) as f32 - 35.0) / 7.0).collect();
        let got = device.swiglu(&gate, &up).expect("swiglu");
        let want: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g * crate::model::norm::sigmoid(*g)) * u)
            .collect();
        let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  swiglu: max |diff| = {d:.3e}");
        assert!(d < 1e-5, "swiglu disagrees by {d}");
        let (cr, cc, ck) = (6usize, 16usize, 4usize);
        let cx: Vec<f32> = (0..cr * cc).map(|i| ((i * 17 % 43) as f32 - 21.0) / 5.0).collect();
        let cw: Vec<f32> = (0..cc * ck).map(|i| ((i % 9) as f32 - 4.0) / 6.0).collect();
        let got = device.causal_conv_silu(&cx, &cw, cr, cc, ck).expect("conv");
        let mut want = vec![0.0f32; cr * cc];
        crate::model::conv::causal_conv1d_silu(&cx, &cw, ck, cc, cr, &mut want);
        let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  causal_conv_silu: max |diff| = {d:.3e}");
        assert!(d < 1e-5, "causal_conv_silu disagrees by {d}");

        let positions: Vec<i32> = (0..rows as i32).collect();
        let rotary_dim = 64usize;
        let head_dim = 256usize;
        let heads = 3usize;
        let (cos, sin) = device.rope_tables(&positions, rotary_dim, 1e7).expect("tables");
        let (wcos, wsin) = crate::model::attention::rope_tables(
            &positions.iter().map(|&p| p as usize).collect::<Vec<_>>(), rotary_dim, 1e7,
        );
        let dc = cos.iter().zip(wcos.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let ds = sin.iter().zip(wsin.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  rope_tables: max |diff| cos = {dc:.3e}, sin = {ds:.3e}");
        assert!(dc < 1e-5 && ds < 1e-5, "rope tables disagree");

        let rx: Vec<f32> = (0..rows * heads * head_dim).map(|i| ((i * 23 % 97) as f32 - 48.0) / 13.0).collect();
        let got = device
            .rope_apply(&rx, &cos, &sin, rows, heads, head_dim, rotary_dim)
            .expect("rope_apply");

        let mut want = rx.clone();
        for t in 0..rows {
            for h in 0..heads {
                let off = (t * heads + h) * head_dim;
                let c = &wcos[t * rotary_dim..(t + 1) * rotary_dim];
                let s = &wsin[t * rotary_dim..(t + 1) * rotary_dim];
                let half = rotary_dim / 2;
                for j in 0..half {
                    let lo = rx[off + j];
                    let hi = rx[off + j + half];
                    want[off + j] = lo * c[j] - hi * s[j];
                    want[off + j + half] = hi * c[j] + lo * s[j];
                }
            }
        }
        let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  rope_apply: max |diff| = {d:.3e}");
        assert!(d < 1e-5, "rope_apply disagrees by {d}");
    }

    #[test]
    fn the_shipping_attention_mma_matches_the_cpu_oracle() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let seq: usize = std::env::var("CLEF_ATT_ORACLE_SEQ").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(256);
        let (heads, kv_heads, head_dim) = (16usize, 4usize, 256usize);
        let mut seed = 0x1357_9BDFu32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let q: Vec<f32> = (0..seq * heads * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..seq * kv_heads * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..seq * kv_heads * head_dim).map(|_| next()).collect();

        let qb = device.upload_f32(&q).unwrap();
        let kb = device.upload_f32(&k).unwrap();
        let vb = device.upload_f32(&v).unwrap();
        let out = device
            .attention_dev(&qb, &kb, &vb, seq, heads, kv_heads, head_dim, BatchMap::SINGLE)
            .expect("attention_dev");
        let got = device.download_f32(&out).expect("download");

        let group = heads / kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let bf = |x: f32| f32::from_bits((x.to_bits() & 0xFFFF_0000) | ((x.to_bits() >> 16 & 1) << 15));
        let mut scores = vec![0.0f32; seq];
        let mut worst = 0.0f32;
        for t in 0..seq {
            for h in 0..heads {
                let kvh = h / group;
                let q_t = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                let mut max = f32::NEG_INFINITY;
                for s2 in 0..=t {
                    let k_s = &k[(s2 * kv_heads + kvh) * head_dim..(s2 * kv_heads + kvh + 1) * head_dim];
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += bf(q_t[d]) * bf(k_s[d]);
                    }
                    scores[s2] = acc * scale;
                    max = max.max(scores[s2]);
                }
                let mut den = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for s2 in 0..=t {
                    let p = (scores[s2] - max).exp();
                    den += p;
                    let v_s = &v[(s2 * kv_heads + kvh) * head_dim..(s2 * kv_heads + kvh + 1) * head_dim];
                    for d in 0..head_dim {
                        acc[d] += p * bf(v_s[d]);
                    }
                }
                for d in 0..head_dim {
                    let want = acc[d] / den;
                    let g = got[(t * heads + h) * head_dim + d];
                    worst = if !g.is_finite() || !want.is_finite() {
                        f32::INFINITY
                    } else {
                        worst.max((g - want).abs())
                    };
                }
            }
        }
        assert!(worst < 2e-2, "the shipping MMA kernel disagrees with the definition: worst {worst:e}");
    }

    #[test]
    fn the_attention_kernel_matches_the_cpu_oracle() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (seq, heads, kv_heads, head_dim) = (37usize, 16usize, 4usize, 256usize);
        let mut seed = 0x1357_9BDFu32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let q: Vec<f32> = (0..seq * heads * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..seq * kv_heads * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..seq * kv_heads * head_dim).map(|_| next()).collect();

        let got = device
            .attention(&q, &k, &v, seq, heads, kv_heads, head_dim)
            .expect("attention");

        let group = heads / kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut want = vec![0.0f32; seq * heads * head_dim];
        let mut scores = vec![0.0f32; seq];
        for t in 0..seq {
            for h in 0..heads {
                let kvh = h / group;
                let q_t = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                let mut max = f32::NEG_INFINITY;
                for s in 0..=t {
                    let k_s = &k[(s * kv_heads + kvh) * head_dim..(s * kv_heads + kvh + 1) * head_dim];
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q_t[d] * k_s[d];
                    }
                    scores[s] = acc * scale;
                    max = max.max(scores[s]);
                }
                let mut sum = 0.0f32;
                for s in 0..=t {
                    scores[s] = (scores[s] - max).exp();
                    sum += scores[s];
                }
                let o_t = &mut want[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                for s in 0..=t {
                    let p = scores[s] / sum;
                    let v_s = &v[(s * kv_heads + kvh) * head_dim..(s * kv_heads + kvh + 1) * head_dim];
                    for d in 0..head_dim {
                        o_t[d] += p * v_s[d];
                    }
                }
            }
        }

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (a, b) in got.iter().zip(want.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / got.len() as f64) as f32;
        let scale_ref = want.iter().map(|x| x.abs()).sum::<f32>() / want.len() as f32;
        eprintln!(
            "  attention {seq}x{heads}/{kv_heads}x{head_dim}: max |diff| = {max_abs:.3e}, \
             mean = {mean_abs:.3e}, ref mean |x| = {scale_ref:.3e}"
        );
        assert!(max_abs < 1e-4, "attention disagrees by {max_abs}");
    }

    #[test]
    fn the_quantiser_round_trips_through_the_cpu_dequantiser() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (rows, cols) = (7usize, 12288usize);

        let x: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let block = i / 16;
                let within = (i % 16) as f32 / 15.0;
                let base = 2.0 + (block % 5) as f32;
                (base * (0.5 + within)) * if i % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();

        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let global = 448.0 / (amax / 6.0);
        let (packed, scales) = device.quantise_nvfp4(&x, rows, cols, global).expect("quantise");

        assert_eq!(packed.len(), crate::swizzle::round_up_public(rows, 128) * (cols / 2));

        assert_eq!(scales.len(), crate::swizzle::round_up_public(rows, 128) * (cols / 16));

        let compact = crate::swizzle::unswizzle_conflict_free(
            &scales, crate::swizzle::round_up_public(rows, 128), cols / 16);

        let scale_f32: Vec<f32> = compact[..rows * (cols / 16)].iter()
            .map(|&b| crate::model::gemm_ref::e4m3_one(b)).collect();
        let got = crate::dequant::dequantise(&packed, &scale_f32, global, cols, rows);

        let scale_of = x.iter().map(|v| v.abs()).sum::<f32>() / x.len() as f32;
        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut worst = 0usize;
        for (i, (a, b)) in got.iter().zip(x.iter()).enumerate() {
            if (a - b).abs() > max_abs {
                max_abs = (a - b).abs();
                worst = i;
            }

            if b.abs() > scale_of / 8.0 {
                let r = ((a - b) / b).abs();
                max_rel = max_rel.max(r);
            }
        }
        let _ = worst;
        eprintln!("  quantise nvfp4: max |diff| = {max_abs:.4e}, max rel = {max_rel:.4e}, mean |x| = {scale_of:.4e}");

        assert!(
            max_rel < 0.25,
            "relative error {max_rel} exceeds E2M1's worst-case step (2 of 6, i.e. 33% at the top)"
        );

        let max_block_scale = (0..x.len() / 16)
            .map(|b| (0..16).map(|k| x[b * 16 + k].abs()).fold(0.0f32, f32::max) / 6.0)
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= max_block_scale + 1e-3,
            "quantisation error {max_abs} exceeds half a step at the largest block scale ({max_block_scale})"
        );

        let zeros = vec![0.0f32; 16];
        let (p0, s0) = device.quantise_nvfp4(&zeros, 1, 16, global).expect("zero row");

        let values: Vec<f32> = p0
            .iter()
            .take(8)
            .flat_map(|&b| {
                let (lo, hi) = crate::dequant::nibbles(b);
                [lo, hi]
            })
            .collect();
        assert!(
            values.iter().all(|&v| v == 0.0),
            "a zero row must dequantise to zero; values were {values:?}"
        );
        let _ = s0;
    }

    #[test]
    fn the_bf16_gemm_matches_the_cpu_reference() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            let lsb = (bits >> 16) & 1;
            ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
        };
        for (m, n, k) in [(16usize, 8usize, 16usize), (32, 16, 32), (16, 8, 64)] {
            let mut seed = 0x7A3B_1C9Du32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0
            };
            let av: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let bv: Vec<f32> = (0..n * k).map(|_| next()).collect();
            let ab: Vec<u8> = av.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect();
            let bb: Vec<u8> = bv.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect();

            let got = device.bf16_gemm(&ab, &bb, m, n, k).expect("bf16 gemm");

            let aw: Vec<f32> = av.iter().map(|v| f32::from_bits((bf16(*v) as u32) << 16)).collect();
            let bw: Vec<f32> = bv.iter().map(|v| f32::from_bits((bf16(*v) as u32) << 16)).collect();
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += aw[i * k + p] * bw[j * k + p];
                    }
                    want[i * n + j] = acc;
                }
            }
            let d = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
            eprintln!("  BF16 {m}x{n}x{k}: max |diff| = {d:.3e}, ref mean |x| = {scale:.3e}");
            assert!(d <= 1e-3 * scale.max(1.0), "BF16 GEMM disagrees by {d}");
        }
    }

    #[test]
    fn every_row_is_written_above_the_grid_y_cap() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let rows = 70_000usize;
        let mut seed = 0x6_5_535u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.5
        };

        {
            let (channels, k) = (128usize, 4usize);
            let x: Vec<f32> = (0..rows * channels).map(|_| next()).collect();
            let w: Vec<f32> = (0..channels * k).map(|_| next()).collect();
            let x_d = device.upload_f32(&x).expect("upload x");
            let w_d = device.upload_f32(&w).expect("upload w");
            let out = device.causal_conv_silu_dev(&x_d, &w_d, rows, channels, k, None, None, None)
                .expect("conv");
            let got = device.download_f32(&out).expect("download");

            let t = rows - 1;
            let mut worst = 0.0f32;
            for ch in 0..channels {
                let mut acc = 0.0f32;
                for tap in 0..k {
                    let src = t as isize - (k as isize - 1) + tap as isize;
                    if src >= 0 {
                        acc += w[ch * k + tap] * x[src as usize * channels + ch];
                    }
                }
                let want = acc / (1.0 + (-acc).exp());
                worst = worst.max((want - got[t * channels + ch]).abs());
            }
            eprintln!("  causal_conv_silu at {rows} rows: last row worst |diff| = {worst:.3e}");
            assert!(worst < 1e-5, "the last row of causal_conv_silu is wrong: {worst:.3e}");
        }

        {
            let cols = 16usize;
            let x: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let x_d = device.upload_f32(&x).expect("upload x");
            let (packed, _scales) = device.quantise_nvfp4_dev(&x_d, rows, cols, 215.0, 128)
                .expect("quantise");
            let got = device.download(&packed).expect("download");

            let last = &got[(rows - 1) * (cols / 2)..rows * (cols / 2)];
            eprintln!("  quantise_nvfp4 at {rows} rows: last row packed = {:?}", last);
            assert_eq!(last.len(), 8, "the last row's packed bytes are missing");
            assert!(last.iter().any(|b| *b != 0),
                    "the last row was never written — the launch truncated at the grid.y cap");
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn both_bf16_tiles_match_the_reference_at_every_boundary() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            let lsb = (bits >> 16) & 1;
            ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
        };
        let vals = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((s >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.5
                })
                .collect()
        };
        for (m, n, k) in [(16usize, 8usize, 16usize), (17, 13, 32), (15, 15, 64), (7, 17, 64),
                          (64, 100, 128), (128, 512, 128), (256, 1024, 256), (1, 1, 16),
                          (255, 1024, 4096), (765, 1024, 1024), (765, 4096, 4096),
                          (255, 1024, 1024)] {
            let a: Vec<f32> = vals(m * k, 0xA1 ^ (m as u32));
            let b: Vec<f32> = vals(n * k, 0xB2 ^ (n as u32));
            let a_bf: Vec<u16> = a.iter().map(|v| bf16(*v)).collect();
            let b_bf: Vec<u16> = b.iter().map(|v| bf16(*v)).collect();
            let a_d = device.upload(&a_bf.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
                .expect("upload a");
            let b_d = device.upload(&b_bf.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
                .expect("upload b");
            let mut outs = Vec::new();

            let arms: &[bool] = if n >= 512 { &[false, true] } else { &[false] };
            for &wide in arms {
                let c = device.bf16_gemm_dev_forced(&a_d, &b_d, m, n, k, Some(wide))
                    .unwrap_or_else(|e| panic!("{m}x{n}x{k} wide={wide}: {e}"));
                let got = device.download_f32(&c).expect("download");

                let af: Vec<f32> = a_bf.iter().map(|v| f32::from_bits((*v as u32) << 16)).collect();
                let bf_: Vec<f32> = b_bf.iter().map(|v| f32::from_bits((*v as u32) << 16)).collect();
                let mut worst = 0.0f32;
                let mut scale = 0.0f32;
                for row in 0..m {
                    for col in 0..n {
                        let mut acc = 0.0f32;
                        for t in 0..k {
                            acc += af[row * k + t] * bf_[col * k + t];
                        }
                        scale = scale.max(acc.abs());
                        worst = worst.max((acc - got[row * n + col]).abs());
                    }
                }

                let tol = 1e-4 * scale.max(1.0);
                eprintln!("  {m}x{n}x{k} wide={wide}: worst |diff| vs reference = {worst:.3e} \
                           (tolerance {tol:.3e}, |ref| max {scale:.3e})");
                assert!(worst < tol, "{m}x{n}x{k} wide={wide}: worst {worst:.3e} against a {tol:.3e} \
                        tolerance");
                outs.push(got);
            }
            if outs.len() == 2 {
                let mut worst_cross = 0.0f32;
                for (x, y) in outs[0].iter().zip(outs[1].iter()) {
                    worst_cross = worst_cross.max((x - y).abs());
                }
                eprintln!("  {m}x{n}x{k}: the two tiles differ by {worst_cross:.3e}");
                assert_eq!(worst_cross, 0.0, "{m}x{n}x{k}: the two tiles must agree exactly");
            }
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_fp8_gemm_timings_at_the_disputed_points() {
        if std::env::var("CLEF_FP8_MEASURE").is_err() { return; }
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        let shapes: [(usize, usize, usize); 8] = [
            (256, 12288, 4096), (768, 12288, 4096), (8192, 12288, 4096),
            (16384, 12288, 4096), (32768, 12288, 4096), (768, 4096, 4096),
            (16384, 12288, 64), (256, 12288, 64),
        ];
        let levels: [(Fp8Level, &str); 4] = [
            (Fp8Level::T64x128, "SHIP"),
            (Fp8Level::T128, "T128"),
            (Fp8Level::W2X4, "W2X4"),
            (Fp8Level::S2, "S2"),
        ];
        for (m, n, k) in shapes {
            let a: Vec<u8> = (0..m * k)
                .map(|i| [0x38u8, 0x3C, 0x40, 0x38, 0xB8, 0x44][(i * 7 + i / k) % 6])
                .collect();
            let b: Vec<u8> = (0..n * k)
                .map(|i| [0x38u8, 0x40, 0x3C, 0xB8, 0x38, 0x42][(i * 5 + i / k) % 6])
                .collect();
            let bf16 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
            let sa: Vec<u8> = (0..m).flat_map(|i| bf16(1.0 + (i % 5) as f32 * 0.25).to_le_bytes()).collect();
            let sb: Vec<u8> = (0..n).flat_map(|i| bf16(0.5 + (i % 3) as f32 * 0.5).to_le_bytes()).collect();
            let a_d = device.upload(&a).expect("up a");
            let wp_d = device.upload(&b).expect("up b");
            let as_d = device.upload(&sa).expect("up sa");
            let ws = device.upload(&sb).expect("up sb");

            let out = device.alloc(fp8_padded_m(m, 64) * n * 4).expect("out");
            let mut best = vec![f64::MAX; levels.len()];
            for (lvl, _) in levels.iter() {
                for _ in 0..2 {
                    device
                        .fp8_gemm_dev_tile_into(fp8_variant_for_level(*lvl), &a_d, &wp_d, &as_d,
                                                &ws, m, n, k, None, &out)
                        .expect("warm");
                }
            }
            device.sync().expect("sync");
            let rounds = 5usize;
            for _ in 0..rounds {
                for (i, (lvl, _)) in levels.iter().enumerate() {
                    device.sync().expect("sync");
                    let t0 = std::time::Instant::now();
                    device
                        .fp8_gemm_dev_tile_into(fp8_variant_for_level(*lvl), &a_d, &wp_d, &as_d,
                                                &ws, m, n, k, None, &out)
                        .expect("timed");
                    device.sync().expect("sync");
                    let dt = t0.elapsed().as_secs_f64();
                    if dt < best[i] { best[i] = dt; }
                }
            }
            let flops = 2.0 * (m * n * k) as f64;
            eprint!("  m={m:<6} n={n:<6}");
            for (i, (_, name)) in levels.iter().enumerate() {
                eprint!("  {name} {:>9.0} ns {:>5.0} TF/s", best[i] * 1e9, flops / best[i] / 1e12);
            }
            eprintln!();
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn a_capture_allocates_from_its_own_pool() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let a = device.upload_f32(&vec![1.0f32; 256]).expect("a");
        let b = device.upload_f32(&vec![2.0f32; 256]).expect("b");
        let out = device.alloc(256 * 4).expect("out");

        assert_eq!(device.graph_pool_allocations(), 0, "nothing has captured yet, so the pool must not exist");

        let shared_before = device.pool_bytes();
        let captured = device.capture(|| {
            let scratch = device.alloc(4 * 1024 * 1024)?;
            device.copy_into(&scratch, &a, 256)?;
            device.copy_into(&out, &b, 256)?;
            Ok(())
        }).expect("capture");

        let graph_allocs = device.graph_pool_allocations();
        eprintln!("  the capture pool served {graph_allocs} allocations; reserved {} bytes",
                  device.graph_pool_reserved());

        assert!(
            graph_allocs >= 1,
            "the capture allocated nothing from its own pool, so its memory is in the shared one and \
             eager work can take it"
        );

        let shared_after_replay = device.pool_bytes();
        captured.launch().expect("first replay");
        device.sync().expect("sync");
        let shared_grew = device.pool_bytes().saturating_sub(shared_after_replay);
        eprintln!("  a replay grew the shared pool by {shared_grew} bytes");
        assert!(
            shared_grew < 1024 * 1024,
            "a replay grew the shared pool by {shared_grew} bytes, so the graph is allocating from it"
        );

        captured.launch().expect("replay again");
        device.sync().expect("sync");
        let got = device.download_f32(&out).expect("download");
        assert!(got.iter().all(|v| (*v - 2.0).abs() < 1e-6), "the replay must write `b` into `out`");
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn a_captured_graph_carries_no_allocation_nodes() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let a = device.upload_f32(&vec![3.5f32; 1024]).expect("a");
        let out = device.alloc(1024 * 4).expect("out");

        let captured = device.capture(|| {
            let mut live = Vec::new();
            for _ in 0..16 {
                live.push(device.alloc(256 * 1024)?);
            }
            for b in &live {
                device.copy_into(b, &a, 1024)?;
            }
            let first = live.remove(0);
            live.clear();
            let second = device.alloc(256 * 1024)?;
            device.copy_into(&second, &first, 1024)?;
            device.copy_into(&out, &second, 1024)?;
            Ok(())
        }).expect("capture");

        assert_eq!(
            (captured.alloc_nodes, captured.free_nodes, captured.memcpy_nodes),
            (0, 0, 0),
            "the capture recorded {} allocation, {} free and {} MEMCPY nodes. Each is re-executed on every \
             replay: an allocation against a free list that has moved, a memcpy against a source pointer \
             baked at capture. See 29-GRAPH-DESIGN.md 0.6.",
            captured.alloc_nodes, captured.free_nodes, captured.memcpy_nodes
        );
        assert!(
            device.graph_block_bytes() >= 16 * 256 * 1024,
            "the block is {} bytes and the capture asked for {} — the block must hold the working set",
            device.graph_block_bytes(), 16 * 256 * 1024
        );

        captured.launch().expect("replay");
        device.sync().expect("sync");
        let got = device.download_f32(&out).expect("download");
        assert!(
            got.iter().all(|v| (*v - 3.5).abs() < 1e-6),
            "the replay must reproduce what the capture computed"
        );
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn an_htod_copy_inside_a_capture_is_refused() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let host = vec![0u8; 64];
        let err = device
            .capture(|| {
                let dst = device.alloc(64)?;
                device.copy_htod_into(&dst, &host)?;
                Ok(())
            })
            .map(|_| ())
            .expect_err("an H2D copy inside a capture must be refused");
        let text = format!("{err:#}");
        assert!(
            text.contains("H2D copy inside a capture"),
            "the refusal must name the reason, and it said: {text}"
        );
        eprintln!("  refused: {text}");
    }

    #[test]
    fn the_bf16_gemm_handles_tile_boundaries() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            let lsb = (bits >> 16) & 1;
            ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
        };

        let vals = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((s >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * 2.0
                })
                .collect()
        };

        let check = |m: usize, n: usize, k: usize, rows: Option<(usize, usize)>| {
            let av = vals(m * k, 0x7A3B_1C9D);
            let bv = vals(n * k, 0x1234_5678);
            let ab: Vec<u8> = av.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect();
            let bb: Vec<u8> = bv.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect();
            let a_dev = device.upload(&ab).expect("a");
            let b_dev = device.upload(&bb).expect("b");
            let out = device
                .bf16_gemm_dev(&a_dev, &b_dev, m, n, k)
                .unwrap_or_else(|e| panic!("{m}x{n}x{k} failed to run: {e:#}"));
            let got = device
                .download_rows_f32(&out, m, n)
                .expect("download the caller's rows");
            let (r0, r1) = rows.unwrap_or((0, m));

            let aw: Vec<f32> = av[r0 * k..r1 * k]
                .iter()
                .map(|v| f32::from_bits((bf16(*v) as u32) << 16))
                .collect();
            let bw: Vec<f32> = bv
                .iter()
                .map(|v| f32::from_bits((bf16(*v) as u32) << 16))
                .collect();
            let mut worst = (0usize, 0.0f32);
            let mut scale = 0.0f64;
            for i in 0..(r1 - r0) {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += aw[i * k + p] * bw[j * k + p];
                    }
                    scale += acc.abs() as f64;
                    let d = (got[(r0 + i) * n + j] - acc).abs();
                    if d > worst.1 {
                        worst = ((r0 + i) * n + j, d);
                    }
                }
            }
            let scale = (scale / ((r1 - r0) * n) as f64) as f32;
            eprintln!(
                "  {m:4}x{n:5}x{k:5} rows {r0}..{r1}: max |diff| = {:.3e} at {} (ref mean |x| {:.3e})",
                worst.1, worst.0, scale
            );
            assert!(
                worst.1 <= 1e-3 * scale.max(1.0),
                "{m}x{n}x{k} diverges over rows {r0}..{r1}: max |diff| {:.3e} at {} (row {}, col {})",
                worst.1,
                worst.0,
                worst.0 / n,
                worst.0 % n
            );
        };
        for (m, n, k) in [
            (1usize, 1usize, 16usize),
            (1, 16, 16),
            (16, 1, 16),
            (13, 13, 16),
            (17, 17, 16),
            (15, 16, 16),
            (16, 15, 16),
            (31, 33, 32),
            (70, 70, 16),
            (65, 129, 16),
            (129, 65, 32),
            (100, 100, 32),
            (7, 200, 16),
        ] {
            check(m, n, k, None);
        }
        for (m, n, k) in [
            (7usize, 32usize, 4096usize),
            (7, 32, 4096),
            (16, 32, 4096),
            (765, 32, 4096),
            (765, 1024, 4096),
        ] {
            check(m, n, k, None);
        }
        check(765, 12288, 4096, Some((0, 16)));
        check(765, 12288, 4096, Some((749, 765)));
    }

    #[test]
    fn the_fused_swiglu_quantise_is_byte_identical_to_the_pair() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let cols = 12288usize;
        let act_global = 215.0f32;

        for rows in [765usize, 769] {
            let mut seed = 0x5111_61u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0) * 0.05
            };
            let gate: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let up: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let gate_d = device.upload_f32(&gate).expect("upload gate");
            let up_d = device.upload_f32(&up).expect("upload up");

            let tile = Device::nvfp4_m_tile_for(rows);
            let blk_mn = crate::swizzle::sf_atom_in_force(tile);
            let rows_padded = crate::swizzle::round_up_public(rows, blk_mn);
            let (q_f, s_f) = device
                .swiglu_quantise_nvfp4_dev(&gate_d, &up_d, rows, cols, act_global, tile)
                .expect("fused")
                .expect("the fused kernel declined this shape");

            let act = device.swiglu_dev(&gate_d, &up_d, rows * cols).expect("swiglu");
            let (q_c, s_c) = device
                .quantise_nvfp4_dev(&act, rows, cols, act_global, tile)
                .expect("quantise");

            let a = device.download(&q_f).expect("dl packed fused");
            let b = device.download(&q_c).expect("dl packed chain");
            let sa = device.download(&s_f).expect("dl scales fused");
            let sb = device.download(&s_c).expect("dl scales chain");

            let first_diff = |x: &[u8], y: &[u8]| -> Option<usize> {
                x.iter().zip(y.iter()).position(|(p, q)| p != q)
            };
            eprintln!(
                "  rows {rows}: packed {} B, scales {} B (rows_padded {rows_padded}, atom {blk_mn})",
                a.len(),
                sa.len()
            );

            assert_eq!(a.len(), crate::swizzle::round_up_public(rows, 128) * (cols / 2),
                       "the packed buffer is the wrong length");
            assert_eq!(sa.len(), rows_padded * (cols / 16), "the scale buffer is the wrong length");
            assert_eq!(sb.len(), sa.len(), "the two paths sized the scales differently");
            match first_diff(&a, &b) {
                None => eprintln!("    packed: byte-identical over {} bytes", a.len()),
                Some(i) => panic!(
                    "packed differs at byte {i} of {}: fused {:#04x} against chain {:#04x}",
                    a.len(), a[i], b[i]
                ),
            }
            match first_diff(&sa, &sb) {
                None => eprintln!("    scales: byte-identical over {} bytes", sa.len()),
                Some(i) => panic!(
                    "scales differ at byte {i} of {}: fused {:#04x} against chain {:#04x}",
                    sa.len(), sa[i], sb[i]
                ),
            }
        }
    }

    #[test]
    fn the_nvfp4_path_survives_real_scales() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let m: usize = std::env::var("CLEF_NVFP4_PROBE_M").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(16);
        let (n, k) = (4096usize, 12288usize);
        let act_global = 215.0f32;
        let w_global = 4128.0f32;

        let mut seed = 0xABCD_1234u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| next() * (0.01 + (i / k) as f32 * 0.002)).collect();
        let (wp, ws_swizzled) = device.quantise_nvfp4(&w, n, k, w_global).expect("weight quantise");
        let ws_compact = crate::swizzle::unswizzle(
            &ws_swizzled, crate::swizzle::round_up_public(n, 128), k / 16);
        let ws = device.swizzle_scales(&ws_compact[..n * (k / 16)], n, k / 16, false).expect("swizzle");
        let wp_dev = device.upload(&wp).expect("upload weights");
        let (ap, a_scales_swizzled) = device.quantise_nvfp4(&x, m, k, act_global).expect("act quantise");
        let a_compact = crate::swizzle::unswizzle(
            &a_scales_swizzled, crate::swizzle::round_up_public(m, 128), k / 16);
        let a_scales = device
            .swizzle_scales(&a_compact[..m * (k / 16)], m, k / 16, SF_CONFLICT_FREE)
            .expect("swizzle a");
        let a_scales_bytes = device.download(&a_scales).expect("dl a");

        let got = device
            .nvfp4_gemm_packed(&ap, &a_scales_bytes, &wp_dev, &ws, m, n, k,
                               act_global, w_global, 0, 0, 0)
            .expect("gemm");

        let b_uniform = device
            .upload(&vec![0x38u8; crate::swizzle::round_up_public(n, 128) * (k / 16)])
            .expect("uniform");
        let unit_b = device
            .nvfp4_gemm_packed(&ap, &a_scales_bytes, &wp_dev, &b_uniform, m, n, k,
                               act_global, w_global, 0, 0, 0)
            .expect("gemm unit b");
        let b_diff = got.iter().zip(unit_b.iter()).map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            b_diff > 1e-6,
            "changing B's scales changed nothing ({b_diff:.3e}) — the sfb operand is not being applied"
        );

        let asf: Vec<f32> = a_compact[..m * (k / 16)].iter()
            .map(|&b| crate::model::gemm_ref::e4m3_one(b)).collect();
        let wsf: Vec<f32> = ws_compact[..n * (k / 16)].iter()
            .map(|&b| crate::model::gemm_ref::e4m3_one(b)).collect();
        let av = crate::dequant::dequantise(&ap, &asf, act_global, k, m);
        let wv = crate::dequant::dequantise(&wp, &wsf, w_global, k, n);

        let rows = [0usize, 1, 7, 15];
        let cols: Vec<usize> = (0..n).step_by(n / 16).chain(std::iter::once(n - 1)).collect();
        let mut worst = (0usize, 0.0f32);
        let mut ref_sum = 0.0f64;
        for &i in &rows {
            for &j in &cols {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += av[i * k + p] * wv[j * k + p];
                }
                if (got[i * n + j] - acc).abs() > worst.1 {
                    worst = (i * n + j, (got[i * n + j] - acc).abs());
                }
                ref_sum += acc.abs() as f64;
            }
        }
        let ref_mean = (ref_sum / (rows.len() * cols.len()) as f64) as f32;
        eprintln!(
            "  NVFP4 real scales {m}x{n}x{k}: max |diff| = {:.3e} at {}, sampled ref mean {:.3e}",
            worst.1, worst.0, ref_mean
        );
        assert!(
            worst.1 <= 0.05 * ref_mean.max(1e-3),
            "the NVFP4 path disagrees by {} — the scale arrangement is the usual cause",
            worst.1
        );
    }

    #[test]
    fn every_kernel_has_a_cubin_as_new_as_its_source() {
        use std::time::SystemTime;
        let out = std::path::Path::new(env!("OUT_DIR"));
        let kernels = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");

        let variants: &[(&str, &str)] = &[("gemm", "gemm_m64"), ("gemm", "gemm_m192")];

        let mtime = |p: &std::path::Path| -> Option<SystemTime> {
            std::fs::metadata(p).ok().and_then(|m| m.modified().ok())
        };

        let mut missing: Vec<String> = Vec::new();
        let mut stale: Vec<String> = Vec::new();
        let mut checked = 0usize;

        let mut check = |src_name: &str, cubin_name: &str| {
            let src = kernels.join(format!("{src_name}.cu"));
            let cubin = out.join(format!("{cubin_name}.cubin"));
            if !cubin.exists() {
                missing.push(cubin_name.to_string());
                return;
            }
            checked += 1;
            if let (Some(s), Some(c)) = (mtime(&src), mtime(&cubin)) {
                if c + std::time::Duration::from_secs(1) < s {
                    stale.push(format!("{cubin_name}.cubin is older than {src_name}.cu"));
                }
            }
        };

        for entry in std::fs::read_dir(&kernels).expect("kernels directory") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("cu") {
                continue;
            }
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            check(&name, &name);
        }
        for (src, cubin) in variants {
            check(src, cubin);
        }

        assert!(missing.is_empty(), "no cubin was built for: {missing:?}");
        assert!(
            stale.is_empty(),
            "a kernel edit never reached the machine — the build did not recompile it:\n  {}\n\
             `cargo build` reports success regardless, so this is the only thing that catches it.",
            stale.join("\n  ")
        );
        assert!(checked >= 20, "only {checked} cubins checked; the build looks incomplete");
        eprintln!("  {checked} cubins present and at least as new as their sources");
    }

    #[test]
    fn the_a_scale_row_correspondence_is_the_identity() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (64usize, 8usize, 64usize);
        let (one, zero) = (0x38u8, 0x00u8);

        let ap = vec![0x22u8; m * k / 2];
        let bp = vec![0x22u8; n * k / 2];
        let bp_dev = device.upload(&bp).expect("upload b");
        let bs = device.swizzle_scales(&vec![one; n * (k / 16)], n, k / 16, false).expect("swizzle b");

        let mut wrong: Vec<(usize, usize)> = Vec::new();
        for hot in [0usize, 1, 7, 8, 14, 15, 16, 31, 32, 47, 48, 63] {
            let mut sa = vec![zero; m * (k / 16)];
            for c in 0..(k / 16) {
                sa[hot * (k / 16) + c] = one;
            }

            let sa_sw = device
                .swizzle_scales(&sa, m, k / 16, SF_CONFLICT_FREE)
                .expect("swizzle a");
            let got = device
                .nvfp4_gemm_packed(&ap, &device.download(&sa_sw).expect("dl a"), &bp_dev, &bs,
                                   m, n, k, 1.0, 1.0, 0, 0, 0)
                .expect("gemm");

            let lit: Vec<usize> = (0..m).filter(|&r| got[r * n].abs() > 1e-3).collect();
            eprintln!("    one-hot row {hot:2} -> output rows {lit:?}");
            if lit != vec![hot] {
                wrong.push((hot, lit.first().copied().unwrap_or(usize::MAX)));
            }
        }
        assert!(wrong.is_empty(),
                "the A-scale row correspondence is not the identity: {wrong:?} \
                 (one-hot row, first output row that read it)");
    }

    #[test]
    fn the_a_scale_k_block_correspondence_is_the_identity() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (64usize, 8usize, 256usize);
        let (one, zero) = (0x38u8, 0x00u8);

        let bp = vec![0x22u8; n * k / 2];
        let bp_dev = device.upload(&bp).expect("upload b");
        let bs = device.swizzle_scales(&vec![one; n * (k / 16)], n, k / 16, false).expect("swizzle b");

        let blocks = k / 16;
        let mut wrong: Vec<(usize, Vec<usize>)> = Vec::new();
        for hot in 0..blocks {
            let mut ap = vec![0x00u8; m * k / 2];
            for r in 0..m {
                for b in 0..8 {
                    ap[r * (k / 2) + hot * 8 + b] = 0x22;
                }
            }
            let mut sa = vec![zero; m * blocks];
            for r in 0..m {
                sa[r * blocks + hot] = one;
            }
            let sa_sw = device.swizzle_scales(&sa, m, blocks, SF_CONFLICT_FREE).expect("swizzle a");
            let got = device
                .nvfp4_gemm_packed(&ap, &device.download(&sa_sw).expect("dl a"), &bp_dev, &bs,
                                   m, n, k, 1.0, 1.0, 0, 0, 0)
                .expect("gemm");

            let lit: Vec<usize> = (0..m).filter(|&r| got[r * n].abs() > 1e-3).collect();
            eprintln!("    scale block {hot:2} -> output rows {lit:?}");
            if lit.len() != m {
                wrong.push((hot, lit));
            }
        }

        assert!(wrong.is_empty(),
                "some k-blocks never reached the accumulator: {wrong:?}");
    }

    #[test]
    fn the_gemm_applies_a_scale_per_m16_atom() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (m, n, k) = (64usize, 128usize, 256usize);
        let (a_global, b_global) = (448.0f32, 448.0f32);
        let mut seed = 0x5EED_1234u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd() * 8.0).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rnd() * 8.0).collect();
        let (ap, a_scales) = device.quantise_nvfp4(&a, m, k, a_global).expect("a quantise");
        let (bp, b_scales) = device.quantise_nvfp4(&b, n, k, b_global).expect("b quantise");
        let bp_dev = device.upload(&bp).expect("upload b");
        let bs = device.swizzle_scales(&b_scales[..n * (k / 16)], n, k / 16, false).expect("swizzle b");

        let as_sw = device
            .swizzle_scales(&a_scales, m, k / 16, SF_CONFLICT_FREE)
            .expect("swizzle a");
        let a_scales_sw = device.download(&as_sw).expect("dl a");
        let got = device
            .nvfp4_gemm_packed(&ap, &a_scales_sw, &bp_dev, &bs, m, n, k, a_global, b_global, 0, 0, 0)
            .expect("gemm");

        let asf: Vec<f32> = a_scales[..m * (k / 16)].iter()
            .map(|&b| crate::model::gemm_ref::e4m3_one(b)).collect();
        let wsf: Vec<f32> = b_scales[..n * (k / 16)].iter()
            .map(|&b| crate::model::gemm_ref::e4m3_one(b)).collect();
        let av = crate::dequant::dequantise(&ap, &asf, a_global, k, m);
        let wv = crate::dequant::dequantise(&bp, &wsf, b_global, k, n);
        let mut worst = (0usize, 0.0f32);
        let mut ref_sum = 0.0f64;
        for i in [0usize, 1, 15, 16, 17, 31, 32, 47, 63] {
            for j in [0usize, 63, 64, 127] {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += av[i * k + p] * wv[j * k + p];
                }
                let d = (got[i * n + j] - acc).abs();
                if d > worst.1 {
                    worst = (i * n + j, d);
                }

                if d > 0.02 * acc.abs().max(1e-3) {
                    eprintln!("      bad (row {i}, col {j}): got {:.4} want {:.4}", got[i * n + j], acc);
                }
                ref_sum += acc.abs() as f64;
            }
        }
        let ref_mean = (ref_sum / 36.0) as f32;
        eprintln!("  per-atom scale {m}x{n}x{k}: max |diff| = {:.3e} at {}, ref mean {:.3e}",
                  worst.1, worst.0, ref_mean);
        assert!(worst.1 <= 0.02 * ref_mean.max(1e-3),
                "the NVFP4 path disagrees by {} — a scale broadcast across m16 atoms is the usual cause",
                worst.1);
    }

            #[test]
    #[ignore = "timing baseline; run explicitly with --ignored"]
    fn the_gemm_timing_baseline() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let shapes: [(usize, usize, usize, &str); 5] = [
            (512, 12288, 4096, "mlp gate/up"),
            (512, 4096, 12288, "mlp down"),
            (512, 4096, 4096, "attn q/o"),
            (512, 8192, 4096, "deltanet in_proj_qkv"),
            (512, 12288, 4096, "deltanet z"),
        ];
        let act_global = 480.0f32;
        let w_global = 9408.0f32;

        eprintln!("  --- one block, varying k (m=16, n=32) ---");
        for k in [64usize, 512, 4096, 12288] {
            let (m, n) = (16usize, 32usize);
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i % 89) as f32 - 44.0) / 440.0).collect();
            let (ap, asw) = device.quantise_nvfp4(&x, m, k, 480.0).expect("a");
            let (wp, wsw) = device.quantise_nvfp4(&w, n, k, 9408.0).expect("w");
            let wc = crate::swizzle::unswizzle(&wsw, crate::swizzle::round_up_public(n, 128), k / 16);
            let ws = device.swizzle_scales(&wc[..n * (k / 16)], n, k / 16, false).expect("sw");
            let wpd = device.upload(&wp).expect("up");
            for _ in 0..3 { let _ = device.nvfp4_gemm_packed(&ap, &asw, &wpd, &ws, m, n, k, 480.0, 9408.0, 0, 0, 0).expect("w"); }
            let iters = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = device.nvfp4_gemm_packed(&ap, &asw, &wpd, &ws, m, n, k, 480.0, 9408.0, 0, 0, 0).expect("g");
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("  k={k:6}  chunks={:4}  {:8.4} ms   {:8.2} us/chunk",
                      k / 64, dt * 1e3, dt * 1e6 / (k / 64) as f64);
        }
        eprintln!("  --- shapes ---");
        for (m, n, k, name) in shapes {
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i % 89) as f32 - 44.0) / 440.0).collect();
            let (wp, wsw) = device.quantise_nvfp4(&w, n, k, w_global).expect("w");
            let w_compact = crate::swizzle::unswizzle(
                &wsw, crate::swizzle::round_up_public(n, 128), k / 16);
            let ws = device.swizzle_scales(&w_compact[..n * (k / 16)], n, k / 16, false).expect("sw");
            let wp_dev = device.upload(&wp).expect("up");
            let x_dev = device.upload_f32(&x).expect("up x");

            let run = || {
                let (ap, asw) = device
                    .quantise_nvfp4_dev(&x_dev, m, k, act_global, Device::nvfp4_m_tile_for(m))
                    .expect("quantise");
                device
                    .nvfp4_gemm_dev(&ap, &asw, &wp_dev, &ws, m, n, k, act_global, w_global)
                    .expect("gemm")
            };

            for _ in 0..3 {
                let _ = run();
            }
            device.sync().expect("sync");
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = run();
                device.sync().expect("sync");
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!(
                "  {name:22} {m}x{n}x{k}: {:8.3} ms   {:7.1} GFLOP/s",
                dt * 1e3, 2.0 * m as f64 * n as f64 * k as f64 / dt / 1e9
            );
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_kernel_timing_baseline() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        eprintln!("  {}", device.describe().unwrap_or_default());

        const TOKENS: usize = 512;

        let ramp = |n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| (((i % 251) as f32 / 251.0) - 0.5) * scale).collect()
        };

        let time = |label: &str, iters: usize, mut f: Box<dyn FnMut() + '_>| {
            for _ in 0..2 {
                f();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("  {label:34} {:9.4} ms", dt * 1e3);
            dt
        };
        for (m, n, k, name) in [
            (TOKENS, 12288usize, 4096usize, "fp8 mlp gate/up"),
            (TOKENS, 4096, 12288, "fp8 mlp down"),
            (TOKENS, 4096, 4096, "fp8 attn q/o"),
        ] {
            let a: Vec<u8> = (0..m * k).map(|i| (i % 120 + 8) as u8).collect();
            let b: Vec<u8> = (0..n * k).map(|i| (i % 120 + 8) as u8).collect();
            let sa = vec![0u8; m * 2];
            let sb = vec![0u8; n * 2];
            let dt = time(&format!("{name} {m}x{n}x{k}"), 10, Box::new(|| {
                let _ = device.fp8_gemm(&a, &b, &sa, &sb, m, n, k, None).expect("fp8");
            }));
            eprintln!("      -> {:.1} GFLOP/s", 2.0 * m as f64 * n as f64 * k as f64 / dt / 1e9);
        }

        {
            let (vh, kd, vd) = (32usize, 128usize, 128usize);

            let q = ramp(TOKENS * vh * kd, 1.0);
            let k = ramp(TOKENS * vh * kd, 1.0);
            let v = ramp(TOKENS * vh * vd, 1.0);
            let decay: Vec<f32> = (0..TOKENS * vh).map(|i| -0.01 * ((i % 7) as f32 + 1.0)).collect();
            let beta: Vec<f32> = (0..TOKENS * vh).map(|i| 0.3 + 0.4 * ((i % 5) as f32 / 5.0)).collect();
            time(&format!("gdn_scan {TOKENS}x{vh}x{kd}"), 10, Box::new(|| {
                let _ = device.gdn_scan(&q, &k, &v, &decay, &beta, TOKENS, vh, kd, vd)
                    .expect("gdn");
            }));
        }

        {
            let (heads, kv_heads, hd) = (16usize, 4usize, 256usize);
            let q = ramp(TOKENS * heads * hd, 1.0);
            let k = ramp(TOKENS * kv_heads * hd, 1.0);
            let v = ramp(TOKENS * kv_heads * hd, 1.0);
            time(&format!("attention {TOKENS}x{heads}x{hd}"), 5, Box::new(|| {
                let _ = device.attention(&q, &k, &v, TOKENS, heads, kv_heads, hd)
                    .expect("attention");
            }));
        }
        {
            let (rows, cols) = (TOKENS, 4096usize);
            let x = ramp(rows * cols, 2.0);
            let w = ramp(cols, 0.1);
            let dt = time("rms_norm 512x4096", 20, Box::new(|| {
                let _ = device.rms_norm(&x, &w, rows, cols, 1e-6).expect("norm");
            }));
            eprintln!("      -> {:.1} GB/s", 2.0 * (rows * cols * 4) as f64 / dt / 1e9);
        }
        {
            let inter = 12288usize;
            let g = ramp(TOKENS * inter, 3.0);
            let u = ramp(TOKENS * inter, 3.0);
            let dt = time("swiglu 512x12288", 20, Box::new(|| {
                let _ = device.swiglu(&g, &u).expect("swiglu");
            }));
            eprintln!("      -> {:.1} GB/s", 3.0 * (TOKENS * inter * 4) as f64 / dt / 1e9);
        }
        {
            let (heads, hd, rotary) = (16usize, 256usize, 64usize);
            let positions: Vec<i32> = (0..TOKENS as i32).collect();
            let (cos, sin) = device.rope_tables(&positions, rotary, 1e7).expect("tables");
            let x = ramp(TOKENS * heads * hd, 1.0);
            time("rope_apply 512x16x256", 20, Box::new(|| {
                let _ = device.rope_apply(&x, &cos, &sin, TOKENS, heads, hd, rotary).expect("rope");
            }));
        }

        {
            let (rows, cols) = (TOKENS, 4096usize);
            let x = ramp(rows * cols, 2.0);
            let dt = time("quantise_nvfp4 512x4096", 20, Box::new(|| {
                let _ = device.quantise_nvfp4(&x, rows, cols, 480.0).expect("quantise");
            }));
            eprintln!("      -> {:.1} GB/s read", (rows * cols * 4) as f64 / dt / 1e9);
            time("quantise_fp8 512x4096", 20, Box::new(|| {
                let _ = device.quantise_fp8(&x, rows, cols).expect("quantise fp8");
            }));
        }
        {
            let (rows, blocks) = (4096usize, 256usize);
            let scales: Vec<u8> = (0..rows * blocks).map(|i| (i % 200 + 8) as u8).collect();
            time("swizzle_scales 4096x256", 10, Box::new(|| {
                let _ = device.swizzle_scales(&scales, rows, blocks, false).expect("swizzle");
            }));
        }
    }

    #[test]
    #[ignore = "run under ncu"]
    fn the_gemm_profile_target() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let spec = std::env::var("CLEF_GEMM_SHAPE").unwrap_or_else(|_| "512,12288,4096".into());
        let p: Vec<usize> = spec.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() != 3 {
            eprintln!("SKIP: CLEF_GEMM_SHAPE must be m,n,k; got {spec:?}");
            return;
        }

        let kind = std::env::var("CLEF_GEMM_KIND").unwrap_or_else(|_| "nvfp4".into());
        let (m, n, k) = (p[0], p[1], p[2]);
        eprintln!("  profiling {kind} {m}x{n}x{k}");

        let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i % 89) as f32 - 44.0) / 440.0).collect();

        match kind.as_str() {
            "fp8" => {
                let (a, sa) = device.quantise_fp8(&x, m, k).expect("a");
                let (b, sb) = device.quantise_fp8(&w, n, k).expect("w");
                let a_dev = device.upload(&a).expect("up a");
                let b_dev = device.upload(&b).expect("up b");
                let sa_dev = device.upload(&sa).expect("up sa");
                let sb_dev = device.upload(&sb).expect("up sb");
                let run = || {
                    device
                        .fp8_gemm_dev(&a_dev, &b_dev, &sa_dev, &sb_dev, m, n, k, None)
                        .expect("fp8")
                };
                let _ = run();
                let _ = run();
            }
            "bf16" => {
                let round = |v: f32| -> u16 {
                    let bits = v.to_bits();
                    let lsb = (bits >> 16) & 1;
                    ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
                };
                let a: Vec<u8> = x.iter().flat_map(|v| round(*v).to_le_bytes()).collect();
                let b: Vec<u8> = w.iter().flat_map(|v| round(*v).to_le_bytes()).collect();
                let a_dev = device.upload(&a).expect("up a");
                let b_dev = device.upload(&b).expect("up b");
                let run = || device.bf16_gemm_dev(&a_dev, &b_dev, m, n, k).expect("bf16");
                let _ = run();
                let _ = run();
            }
            _ => {
                let (ap, asw) = device.quantise_nvfp4(&x, m, k, 480.0).expect("a");
                let (wp, wsw) = device.quantise_nvfp4(&w, n, k, 9408.0).expect("w");
                let wc =
                    crate::swizzle::unswizzle(&wsw, crate::swizzle::round_up_public(n, 128), k / 16);
                let ws = device
                    .swizzle_scales(&wc[..n * (k / 16)], n, k / 16, false)
                    .expect("sw");
                let wpd = device.upload(&wp).expect("up");
                let run = || {
                    device
                        .nvfp4_gemm_packed(&ap, &asw, &wpd, &ws, m, n, k, 480.0, 9408.0, 0, 0, 0)
                        .expect("nvfp4")
                };
                let _ = run();
                let _ = run();
            }
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_alloc_cost_baseline() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        for mb in [1usize, 8, 38, 64] {
            let bytes = mb * 1024 * 1024;
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let b = device.alloc(bytes).expect("alloc");
                std::hint::black_box(b.device_ptr());
                drop(b);
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("  alloc+free {mb:3} MB: {:9.4} ms", dt * 1e3);
        }

        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let a = device.alloc(2 * 1024 * 1024).expect("a");
            let b = device.alloc(38 * 1024 * 1024).expect("b");
            let c = device.alloc(1024 * 1024).expect("c");
            std::hint::black_box((a.device_ptr(), b.device_ptr(), c.device_ptr()));
            drop((a, b, c));
        }
        eprintln!("  mixed 2+38+1 MB, 3 live: {:9.4} ms",
                  t0.elapsed().as_secs_f64() / iters as f64 * 1e3);

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for bytes in [2usize, 38, 1] {
                let b = device.alloc(bytes * 1024 * 1024).expect("b");
                std::hint::black_box(b.device_ptr());
                drop(b);
            }
        }
        eprintln!("  same 2+38+1 MB, 1 live:  {:9.4} ms",
                  t0.elapsed().as_secs_f64() / iters as f64 * 1e3);
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_alloc_size_sweep() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        const MIB: usize = 1024 * 1024;

        let mut sizes: Vec<usize> = vec![
            12_533_760,
            25_067_520,
            25_165_824,
            37_595_520,
            37_748_736,
        ];
        for mb in [16usize, 20, 24, 26, 28, 30, 32, 34, 36] {
            sizes.push(mb * MIB);
        }
        sizes.sort_unstable();
        sizes.dedup();

        eprintln!("  {:>12}  {:>10}  {:>12}  {:>10}", "bytes", "alloc+sync", "alloc nosync", "MiB");
        for bytes in sizes {
            let time = |sync: bool| -> f64 {
                for _ in 0..3 {
                    let b = device.alloc(bytes).expect("alloc");
                    std::hint::black_box(b.device_ptr());
                    drop(b);
                }
                if sync {
                    device.sync().expect("sync");
                }
                let iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let b = device.alloc(bytes).expect("alloc");
                    std::hint::black_box(b.device_ptr());
                    drop(b);
                    if sync {
                        device.sync().expect("sync");
                    }
                }
                t0.elapsed().as_secs_f64() / iters as f64 * 1e3
            };
            let with_sync = time(true);
            let without = time(false);
            eprintln!(
                "  {bytes:12}  {with_sync:10.4}  {without:12.4}  {:10.2}",
                bytes as f64 / MIB as f64
            );
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_op_overhead_baseline() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        const T: usize = 765;
        let hidden = 4096usize;
        let inter = 12288usize;

        let x = device.upload_f32(&vec![0.25f32; T * hidden]).expect("x");
        let w = device.upload_f32(&vec![1.0f32; hidden]).expect("w");
        let gate = device.upload_f32(&vec![0.5f32; T * inter]).expect("g");
        let up = device.upload_f32(&vec![0.5f32; T * inter]).expect("u");

        let time = |label: &str, iters: usize, mut f: Box<dyn FnMut() + '_>| {
            for _ in 0..3 {
                f();
            }
            device.sync().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
                device.sync().expect("sync");
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("  {label:38} {:9.4} ms", dt * 1e3);
        };
        time("rms_norm_dev [765,4096]", 20, Box::new(|| {
            let _ = device.rms_norm_dev(&x, &w, T, hidden, 1e-6).expect("norm");
        }));
        time("add_dev [765,4096]", 20, Box::new(|| {
            let _ = device.add_dev(&x, &x, T * hidden).expect("add");
        }));
        time("quantise_nvfp4_dev [765,4096]", 20, Box::new(|| {
            let _ = device.quantise_nvfp4_dev(&x, T, hidden, 480.0, Device::nvfp4_m_tile_for(T)).expect("q");
        }));
        time("swiglu_dev [765,12288]", 20, Box::new(|| {
            let _ = device.swiglu_dev(&gate, &up, T * inter).expect("swiglu");
        }));
        time("to_bf16_dev [765,4096]", 20, Box::new(|| {
            let _ = device.to_bf16_dev(&x, T * hidden).expect("bf16");
        }));
        let (wp, wsw) = device.quantise_nvfp4(&vec![0.01f32; inter * hidden], inter, hidden, 9408.0)
            .expect("w");
        let wc = crate::swizzle::unswizzle(&wsw, crate::swizzle::round_up_public(inter, 128), hidden / 16);
        let ws = device.swizzle_scales(&wc[..inter * (hidden / 16)], inter, hidden / 16, false).expect("sw");
        let wpd = device.upload(&wp).expect("up");
        time("quantise + nvfp4_gemm [765,12288,4096]", 20, Box::new(|| {
            let (ap, asw) = device.quantise_nvfp4_dev(&x, T, hidden, 480.0, Device::nvfp4_m_tile_for(T)).expect("q");
            let _ = device.nvfp4_gemm_dev(&ap, &asw, &wpd, &ws, T, inter, hidden, 480.0, 9408.0)
                .expect("gemm");
        }));
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_gemm_m_sweep() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (n, k) = (12288usize, 4096usize);
        let (wp, wsw) = device
            .quantise_nvfp4(&vec![0.01f32; n * k], n, k, 9408.0)
            .expect("w");
        let wc = crate::swizzle::unswizzle(&wsw, crate::swizzle::round_up_public(n, 128), k / 16);
        let ws = device.swizzle_scales(&wc[..n * (k / 16)], n, k / 16, false).expect("sw");
        let wpd = device.upload(&wp).expect("up");

        eprintln!("  {:>6}  {:>12}  {:>9}  {:>9}", "M", "GFLOP/s", "waves", "split");

        for m in [16usize, 64, 128, 192, 256, 320, 512, 768, 1024, 2048, 8192, 32768, 131072] {
            let x = device.upload_f32(&vec![0.25f32; m * k]).expect("x");

            let bench = |mut f: Box<dyn FnMut() + '_>| -> f64 {
                for _ in 0..3 {
                    f();
                }
                device.sync().expect("sync");

                let iters = 20usize;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f();
                }
                device.sync().expect("sync");
                t0.elapsed().as_secs_f64() / iters as f64
            };
            let q_ms = bench(Box::new(|| {
                let _ = device.quantise_nvfp4_dev(&x, m, k, 480.0, Device::nvfp4_m_tile_for(m)).expect("q");
            })) * 1e3;
            let (ap, asw) = device.quantise_nvfp4_dev(&x, m, k, 480.0, Device::nvfp4_m_tile_for(m)).expect("q");
            let g_ms = bench(Box::new(|| {
                let _ = device
                    .nvfp4_gemm_dev(&ap, &asw, &wpd, &ws, m, n, k, 480.0, 9408.0)
                    .expect("gemm");
            })) * 1e3;
            let dt = (q_ms + g_ms) / 1e3;
            let padded = m.div_ceil(16) * 16;
            let waves = (n / 8 * (padded / 16)) as f64 / (170.0 * 24.0);
            eprintln!(
                "  {m:6}  {:12.1}  {waves:9.2}  q {q_ms:7.4}  gemm {g_ms:8.4}",
                2.0 * m as f64 * n as f64 * k as f64 / dt / 1e9
            );
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_sync_cost_baseline() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let bench = |label: &str, iters: usize, mut f: Box<dyn FnMut() + '_>| {
            for _ in 0..3 {
                f();
            }
            device.sync().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("  {label:44} {:9.5} ms", dt * 1e3);
        };
        bench("bare sync, nothing queued", 200, Box::new(|| {
            device.sync().expect("sync");
        }));

        let x = device.upload_f32(&vec![0.25f32; 4096]).expect("x");
        let w = device.upload_f32(&vec![1.0f32; 4096]).expect("w");
        bench("sync after one tiny kernel (rms_norm 1x4096)", 200, Box::new(|| {
            let _ = device.rms_norm_dev(&x, &w, 1, 4096, 1e-6).expect("n");
            device.sync().expect("sync");
        }));

        let big = device.upload_f32(&vec![0.25f32; 765 * 4096]).expect("big");
        let wb = device.upload_f32(&vec![1.0f32; 4096]).expect("wb");
        bench("sync after one large kernel (rms_norm 765x4096)", 200, Box::new(|| {
            let _ = device.rms_norm_dev(&big, &wb, 765, 4096, 1e-6).expect("n");
            device.sync().expect("sync");
        }));
        bench("sync after 25 tiny kernels", 100, Box::new(|| {
            for _ in 0..25 {
                let _ = device.rms_norm_dev(&x, &w, 1, 4096, 1e-6).expect("n");
            }
            device.sync().expect("sync");
        }));
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_pool_must_not_reuse_a_block_within_the_stream() {
        use cudarc::driver::sys;
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        if !device.ctx.has_async_alloc() {
            eprintln!("SKIP: this device has no memory pools, so `alloc` is synchronous");
            return;
        }
        unsafe {
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let r = sys::cuDeviceGetMemPool(&mut pool, device.ctx.cu_device());
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuDeviceGetMemPool failed: {r:?}");
            for (attr, want, why) in [
                (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, 0,
                 "reuse within the same stream without waiting — the whole engine is one stream"),
                (sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, 0,
                 "reuse when the driver believes it can prove safety"),
            ] {
                let mut got: i32 = -1;
                let r = sys::cuMemPoolGetAttribute(pool, attr, (&mut got as *mut i32).cast());
                assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemPoolGetAttribute failed: {r:?}");
                assert_eq!(got, want, "{attr:?} is {got}, expected {want}: {why}");
            }
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_pool_returns_stable_addresses() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let pass = |device: &Device| -> Vec<u64> {
            let a = device.alloc(2 * 1024 * 1024).expect("a");
            let b = device.alloc(38 * 1024 * 1024).expect("b");
            let c = device.alloc(1024 * 1024).expect("c");
            vec![a.device_ptr(), b.device_ptr(), c.device_ptr()]
        };

        let first = pass(&device);
        device.sync().expect("sync");
        let second = pass(&device);
        device.sync().expect("sync");
        let third = pass(&device);
        eprintln!("  pass 1: {first:x?}");
        eprintln!("  pass 2: {second:x?}");
        eprintln!("  pass 3: {third:x?}");
        assert_eq!(
            second, third,
            "the pool did not return the same addresses on a repeated sequence, so a CUDA graph \n             could not be built on it"
        );
    }

    #[test]
    #[ignore = "needs the card"]
    fn the_byte_copy_reproduces_its_source() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no device");
            return;
        };
        let values = 1024usize;
        let host: Vec<f32> = (0..values).map(|i| (i as f32) * 0.5 - 100.0).collect();
        let src = device.upload_f32(&host).expect("upload");

        for _ in 0..3 {
            let copied = device.copy_dev(&src, values).expect("copy_dev");
            let got = device.download_f32(&copied).expect("dl");
            assert_eq!(got.len(), values, "copy_dev returned the wrong length");
            for (i, (a, b)) in host.iter().zip(got.iter()).enumerate() {
                assert_eq!(a, b, "copy_dev differs at {i}: {a} vs {b}");
            }

            let rows = device.copy_rows(&src, 4, 256).expect("copy_rows");
            let got = device.download_f32(&rows).expect("dl");
            assert_eq!(got.len(), 4 * 256, "copy_rows returned the wrong length");
            for (i, (a, b)) in host.iter().take(4 * 256).zip(got.iter()).enumerate() {
                assert_eq!(a, b, "copy_rows differs at {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn a_key_split_does_not_change_the_answer() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no device");
            return;
        };
        std::env::set_var("CLEF_KV_CAPACITY", "8192");
        let (seq, heads, kv_heads, head_dim) = (4096usize, 16usize, 4usize, 256usize);
        let gqa = heads / kv_heads;
        let rph = 128usize.div_ceil(gqa);

        let qh: Vec<f32> = (0..seq * heads * head_dim).map(|i| ((i * 37 % 61) as f32 - 30.0) * 0.05).collect();
        let module = device.load(ATTENTION_MMA_WS_SPLIT_CUBIN, "attention_mma_ws_split")
            .expect("the split cubin");
        let kernel = module.kernel("attention_kernel").expect("attention_kernel");
        let (s, h, kh_i, hd) = (seq as i32, heads as i32, kv_heads as i32, head_dim as i32);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let row_blocks = seq.div_ceil(rph) as u32;

        let run_cl = |cl: usize| -> Vec<(usize, f32, usize)> {
        let kv_rows = cl + seq;
        let kh: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 53 % 59) as f32 - 29.0) * 0.05).collect();
        let vh: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| (i % 97) as f32 * 0.01).collect();
        let q = device.upload_f32(&qh).expect("q");
        let k = device.upload_f32(&kh).expect("k");
        let v = device.upload_f32(&vh).expect("v");
        let q16 = device.to_bf16_dev(&q, seq * heads * head_dim).expect("q16");
        let k16 = device.to_bf16_dev(&k, kv_rows * kv_heads * head_dim).expect("k16");
        let v16 = device.to_bf16_dev(&v, kv_rows * kv_heads * head_dim).expect("v16");
        let clen = device.upload_i32(&[cl as i32]).expect("clen");
        let run = |splits: usize| -> Vec<f32> {
            let out = device.alloc(seq * heads * head_dim * 4).expect("out");
            let partials = device.alloc(splits * seq * heads * (head_dim + 2) * 4).expect("partials");

            device.fill_nan(&partials).expect("poison partials");
            unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream
                    .launch_builder(&kernel.function)
                    .arg(&q16.slice()).arg(&k16.slice()).arg(&v16.slice()).arg(&out.slice())
                    .arg(&s).arg(&h).arg(&kh_i).arg(&hd).arg(&scale)
                    .arg(&0u64).arg(&0u64).arg(&(seq as i32))
                    .arg(&0u64)
                    .arg(&clen.device_ptr())
                    .arg(&0u64)
                    .arg(&0u64).arg(&0u64).arg(&0u64)
                    .arg(&partials.slice()).arg(&0i32).arg(&(seq as i32))

                    .arg(&((crate::gpu::kv_capacity() / ATTN_BK + 1).div_ceil(splits) as i32))
                    .launch(config_grid_3d(kv_heads as u32, row_blocks, splits as u32, 384))
                    .expect("attention launch");
            }
            device.sync().expect("sync");
            device.mha_combine_dev(&partials, &out, seq, heads * head_dim, heads, splits)
                .expect("combine");
            device.sync().expect("sync");
            device.download_f32(&out).expect("download")
        };

        let plain = {
            let plain_mod = device.load(ATTENTION_MMA_WS_R240_CUBIN, "attention_mma_ws_r240").expect("plain cubin");
            let plain_k = plain_mod.kernel("attention_kernel").expect("attention_kernel");
            let out = device.alloc(seq * heads * head_dim * 4).expect("out");
            unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream
                    .launch_builder(&plain_k.function)
                    .arg(&q16.slice()).arg(&k16.slice()).arg(&v16.slice()).arg(&out.slice())
                    .arg(&s).arg(&h).arg(&kh_i).arg(&hd).arg(&scale)
                    .arg(&0u64).arg(&0u64).arg(&(seq as i32))

                    .arg(&0u64)
                    .arg(&clen.device_ptr())
                    .arg(&0u64).arg(&0u64).arg(&0u64).arg(&0u64)
                    .launch(config_grid_3d(kv_heads as u32, row_blocks, 1, 384))
                    .expect("plain launch");
            }
            device.sync().expect("sync");
            device.download_f32(&out).expect("dl plain")
        };

        let mut worsts = Vec::new();
        for splits in [1usize, 2, 3, 4] {
            let many = run(splits);
            assert_eq!(many.len(), plain.len(), "splits={splits}: the combine wrote {} of {} elements",
                       many.len(), plain.len());
            let mut worst = 0.0f32;
            let mut worst_at = 0usize;
            let mut nans = 0usize;
            for (i, (a, b)) in plain.iter().zip(many.iter()).enumerate() {
                if a.is_nan() || b.is_nan() { nans += 1; worst = f32::INFINITY; worst_at = i; continue; }
                let d = (a - b).abs();
                if d > worst { worst = d; worst_at = i; }
            }
            assert_eq!(nans, 0, "splits={splits} produced {nans} NaN elements");
            let row = worst_at / (heads * head_dim);
            let head = (worst_at / head_dim) % heads;
            let dim = worst_at % head_dim;

            let t0 = (row / rph) * rph;
            let bound = t0 + 128;
            eprintln!("  cl={cl} splits={splits}: worst |d| {worst:.3e} at row {row} head {head} dim {dim} \
                       (row block t0={t0}, key bound {bound}, tiles {})", (bound / 32) + 1);
            assert!(worst < 1e-3,
                    "cl={cl} splits={splits} moved the attention by {worst:.3e} at row {row} head {head} dim {dim}");
            worsts.push((splits, worst, row));
        }
        worsts
        };

        for cl in [0usize, 4096] {
            let arms = run_cl(cl);
            eprintln!("  cl={cl}: {arms:?}");
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn a_key_split_does_not_change_a_batched_pass() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no device");
            return;
        };
        std::env::set_var("CLEF_KV_CAPACITY", "4096");
        let (heads, kv_heads, head_dim, cap) = (16usize, 4usize, 256usize, 4096usize);
        let gqa = heads / kv_heads;
        let rph = 128usize.div_ceil(gqa);
        let n0 = 2048usize;
        let n1 = 2048usize;
        let seq = n0 + n1;
        let kv_rows = cap + n1;
        let qh: Vec<f32> = (0..seq * heads * head_dim).map(|i| ((i * 37 % 61) as f32 - 30.0) * 0.05).collect();
        let kh: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 53 % 59) as f32 - 29.0) * 0.05).collect();
        let vh: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| (i % 97) as f32 * 0.01).collect();
        let q = device.upload_f32(&qh).expect("q");
        let k = device.upload_f32(&kh).expect("k");
        let v = device.upload_f32(&vh).expect("v");
        let q16 = device.to_bf16_dev(&q, seq * heads * head_dim).expect("q16");
        let k16 = device.to_bf16_dev(&k, kv_rows * kv_heads * head_dim).expect("k16");
        let v16 = device.to_bf16_dev(&v, kv_rows * kv_heads * head_dim).expect("v16");

        let mut cpos = vec![0i32; seq];
        let mut cbase = vec![0i32; seq];
        let mut cslot = vec![0i32; seq];
        let mut send = vec![0i32; seq];
        for t in 0..seq {
            if t < n0 { cpos[t] = t as i32; cbase[t] = 0; cslot[t] = 0; send[t] = n0 as i32; }
            else { cpos[t] = (cap + (t - n0)) as i32; cbase[t] = cap as i32; cslot[t] = 1; send[t] = seq as i32; }
        }
        let pos = device.upload_i32(&cpos).expect("cpos");
        let base = device.upload_i32(&cbase).expect("cbase");
        let slot = device.upload_i32(&cslot).expect("cslot");
        let se = device.upload_i32(&send).expect("seq_end");
        let clen = device.upload_i32(&[0i32]).expect("clen");
        let module = device.load(ATTENTION_MMA_WS_SPLIT_CUBIN, "attention_mma_ws_split").expect("split cubin");
        let kernel = module.kernel("attention_kernel").expect("attention_kernel");
        let (s_, h_, kh_, hd_) = (seq as i32, heads as i32, kv_heads as i32, head_dim as i32);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let row_blocks = seq.div_ceil(rph) as u32;

        let max_tiles = crate::gpu::kv_capacity() / ATTN_BK + 1;
        let run = |splits: usize| -> Vec<f32> {
            let tps = max_tiles.div_ceil(splits) as i32;
            let out = device.alloc(seq * heads * head_dim * 4).expect("out");
            let partials = device.alloc(splits * seq * heads * (head_dim + 2) * 4).expect("partials");
            device.fill_nan(&partials).expect("poison");
            unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream
                    .launch_builder(&kernel.function)
                    .arg(&q16.slice()).arg(&k16.slice()).arg(&v16.slice()).arg(&out.slice())
                    .arg(&s_).arg(&h_).arg(&kh_).arg(&hd_).arg(&scale)
                    .arg(&0u64)
                    .arg(&se.device_ptr())
                    .arg(&(seq as i32))
                    .arg(&0u64)
                    .arg(&clen.device_ptr())
                    .arg(&0u64)
                    .arg(&pos.device_ptr()).arg(&base.device_ptr()).arg(&slot.device_ptr())
                    .arg(&partials.slice()).arg(&0i32).arg(&(seq as i32)).arg(&tps)
                    .launch(config_grid_3d(kv_heads as u32, row_blocks, splits as u32, 384))
                    .expect("attention launch");
            }
            device.mha_combine_dev(&partials, &out, seq, heads * head_dim, heads, splits).expect("combine");
            device.sync().expect("sync");
            device.download_f32(&out).expect("download")
        };
        let one = run(1);
        for splits in [2usize, 4] {
            let many = run(splits);
            let mut worst = 0.0f32;
            let mut worst_at = 0usize;
            let mut nans = 0usize;
            for (i, (a, b)) in one.iter().zip(many.iter()).enumerate() {
                if a.is_nan() || b.is_nan() { nans += 1; worst_at = i; continue; }
                let d = (a - b).abs();
                if d > worst { worst = d; worst_at = i; }
            }
            let row = worst_at / (heads * head_dim);
            let which = if row < n0 { 0 } else { 1 };
            let tps = max_tiles.div_ceil(splits);
            eprintln!("  batched splits={splits}: worst {worst:.3e} nans {nans} at row {row} (request {which}), T={tps}");
            assert_eq!(nans, 0, "splits={splits} produced {nans} NaNs");
            assert!(worst < 1e-3, "splits={splits} moved a batched pass by {worst:.3e} at row {row}");
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn the_bf16_conversion_reads_the_row_its_base_names() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no device");
            return;
        };
        let (rows, cols, base) = (8usize, 1024usize, 5usize);
        let total = (rows + base) * cols;
        let host: Vec<f32> = (0..total).map(|i| (i / cols) as f32).collect();
        let x = device.upload_f32(&host).expect("x");
        let base_dev = device.upload_i32(&[base as i32]).expect("base");

        let by_base = device.to_bf16_dev_at(&x, rows * cols, cols, &base_dev).expect("at base");
        let view = device.view(&x, base * cols * 4, rows * cols * 4);
        let by_view = device.to_bf16_dev(&view, rows * cols).expect("by view");

        let a = device.download(&by_base).expect("download by_base");
        let b = device.download(&by_view).expect("download by_view");
        assert_eq!(a.len(), b.len(), "the two conversions returned different lengths");
        for i in 0..rows {
            let off = i * cols * 2;
            let want = ((base + i) as f32).to_bits() >> 16;
            let got = u16::from_le_bytes([a[off], a[off + 1]]) as u32;
            let got_view = u16::from_le_bytes([b[off], b[off + 1]]) as u32;
            assert_eq!(got, want,
                       "row {i} of the conversion is {got:#06x}, not the row the base names ({want:#06x})");
            assert_eq!(got, got_view, "the two ways of naming row {i} disagree");
        }
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn a_chunked_head_attention_matches_the_whole_tensor_one() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no device");
            return;
        };
        let (width, heads, q_len, tokens) = (1024usize, 16usize, 8usize, 16_717usize);
        let chunks = crate::gpu::Rungs::head().decompose(tokens);
        assert!(chunks.len() > 1,
                "{tokens} cut into {} chunk(s): a one-chunk arm cannot see a chunk-local read",
                chunks.len());

        let peaks = [100usize, 9_000, 16_500];
        let mut kh = vec![0.0f32; tokens * width];
        let mut vh = vec![0.0f32; tokens * width];
        for t in 0..tokens {
            for c in 0..width {
                vh[t * width + c] = (t % 97) as f32;
            }
        }
        for &t in peaks.iter() {
            for c in 0..width {
                kh[t * width + c] = 1.0;
            }
        }
        let qh = vec![1.0f32; q_len * width];

        let q = device.upload_f32(&qh).expect("q");
        let k = device.upload_f32(&kh).expect("k");
        let v = device.upload_f32(&vh).expect("v");
        let kreal = device.upload_i32(&[tokens as i32]).expect("kv_real");

        let kbuf = device.alloc(tokens * width * 4).expect("k staging");
        let vbuf = device.alloc(tokens * width * 4).expect("v staging");

        let run = |cut: &[(usize, usize)]| {
            device.mha_chunks_dev(&q, q_len, tokens, &kreal, width, heads, cut,
                                  |dev, c, _rows| {
                let (base, rows) = cut[c];

                let here = rows.min(tokens - base);
                dev.copy_bytes_at(&kbuf, 0, &k, base * width * 4, here * width * 4)?;
                dev.copy_bytes_at(&vbuf, 0, &v, base * width * 4, here * width * 4)?;
                Ok((kbuf.clone_buffer(), vbuf.clone_buffer()))
            }).expect("attention")
        };

        let a = device.download_f32(&run(&[(0usize, tokens)])).expect("download whole");
        let b = device.download_f32(&run(&chunks)).expect("download cut");
        let worst = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("  {} tokens, {} chunks, {} peaks: worst |chunked - whole| = {worst:.3e} \
                   (answer's own scale {scale:.1})", tokens, chunks.len(), peaks.len());
        assert!(worst < 1e-2,
                "the cut moved the head's attention by {worst:.3e} ({} chunks: {:?}); the fold order is \
                 9.995e-4 and a chunk read from the wrong rows is 4.221e0", chunks.len(), chunks);
    }

    #[test]
    #[ignore = "needs the RTX 5090"]
    fn a_captured_sequence_replays_the_same_result() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (rows, cols) = (64usize, 4096usize);
        let x = device.upload_f32(&vec![0.25f32; rows * cols]).expect("x");
        let w = device.upload_f32(&vec![1.0f32; cols]).expect("w");

        let run = || -> Result<Buffer> {
            let a = device.rms_norm_dev(&x, &w, rows, cols, 1e-6)?;
            let b = device.to_bf16_dev(&a, rows * cols)?;
            device.rms_norm_dev(&b, &w, rows, cols, 1e-6)
        };

        for _ in 0..3 {
            let warm = run().expect("warm");
            device.sync().expect("sync");
            drop(warm);
        }

        let captured = device.capture(|| {
            let _ = run()?;
            Ok(())
        });
        let captured = match captured {
            Ok(c) => c,
            Err(e) => {
                panic!("capture failed: {e:#}");
            }
        };
        eprintln!("  captured and instantiated");

        captured.launch().expect("first replay");
        device.sync().expect("sync");
        captured.launch().expect("second replay");
        device.sync().expect("sync");
        captured.launch().expect("third replay");
        device.sync().expect("sync");

        let direct = run().expect("direct");
        let d = device.download_f32(&direct).expect("download");
        assert!(
            d.iter().all(|v| v.is_finite()),
            "the captured sequence produced a non-finite value"
        );
        let scale = d.iter().map(|v| v.abs()).sum::<f32>() / d.len() as f32;
        eprintln!("  direct mean |x| = {scale:.6e}, first = {:.6e}", d[0]);
        assert!(scale > 0.0, "the captured sequence produced zeros");
    }

    #[test]
    fn the_gemm_handles_tile_boundaries() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };

        let shapes: [(usize, usize, usize); 17] = [

            (7, 12288, 4096),
            (7, 4096, 12288),
            (7, 8192, 4096),
            (16, 12288, 4096),
            (1, 1, 64),
            (1, 16, 64),
            (16, 1, 64),
            (13, 13, 64),
            (17, 17, 64),
            (15, 16, 64),
            (16, 15, 64),
            (31, 33, 128),
            (70, 70, 64),
            (65, 129, 64),
            (129, 65, 128),
            (100, 100, 192),
            (7, 200, 64),
        ];
        for (m, n, k) in shapes {
            let a = packed(m, k, 0x1234_5678);
            let b = packed(n, k, 0x9ABC_DEF0);
            let sa = unit_scales(m, k);
            let sb = unit_scales(n, k);
            let reference = crate::model::gemm_ref::matmul(
                &crate::model::gemm_ref::Nvfp4Matrix {
                    packed: a.clone(), scales: sa.clone(), rows: m, cols: k, global_scale: 1.0,
                },
                &crate::model::gemm_ref::Nvfp4Matrix {
                    packed: b.clone(), scales: sb.clone(), rows: n, cols: k, global_scale: 1.0,
                },
            );

            let swizzle = |compact: &[u8], rows: usize, cf: bool| -> Buffer {
                let b = device.upload(compact).expect("up scales");
                device.swizzle_scales(&device.download(&b).expect("dl"), rows, k / 16, cf)
                    .expect("swizzle")
            };

            let sa_dev = swizzle(&sa, m, SF_CONFLICT_FREE);
            let sb_dev = swizzle(&sb, n, false);
            let a_dev = device.upload(&a).expect("up a");
            let b_dev = device.upload(&b).expect("up b");
            let out = device
                .nvfp4_gemm_dev(&a_dev, &sa_dev, &b_dev, &sb_dev, m, n, k, 1.0, 1.0)
                .unwrap_or_else(|e| panic!("{m}x{n}x{k} failed to run: {e:#}"));
            let got = device
                .download_rows_f32(&out, m, n)
                .expect("download the caller's rows");
            let mut worst = (0usize, 0.0f32);
            let mut n_bad = 0usize;
            let mut bad_rows: Vec<usize> = Vec::new();
            let mut bad_cols: Vec<usize> = Vec::new();
            let mut n_zero = 0usize;
            for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                let d = (g - r).abs();
                if d > worst.1 {
                    worst = (i, d);
                }

                if d > 1e-3 {
                    n_bad += 1;
                    if *g == 0.0 { n_zero += 1; }
                    let (rr, cc) = (i / n, i % n);
                    if !bad_rows.contains(&rr) { bad_rows.push(rr); }
                    if !bad_cols.contains(&cc) { bad_cols.push(cc); }
                }
            }
            eprintln!(
                "  {m:4}x{n:4}x{k:4}: max |diff| = {:.3e} at {} | {n_bad}/{} bad, {n_zero} zero, \
                 rows {:?}{}, cols {:?}{}",
                worst.1, worst.0, got.len(),
                &bad_rows[..bad_rows.len().min(8)], if bad_rows.len() > 8 { ".." } else { "" },
                &bad_cols[..bad_cols.len().min(8)], if bad_cols.len() > 8 { ".." } else { "" }
            );
            assert!(
                worst.1 < 1e-3,
                "{m}x{n}x{k} diverges: max |diff| {:.3e} at {} (row {}, col {})",
                worst.1, worst.0, worst.0 / n, worst.0 % n
            );
        }
    }

    #[test]
    #[ignore = "loads the device; run on the card"]
    fn the_option_axis_eager_versus_tiled() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        let (width, heads, kv) = (1024usize, 16usize, 8192usize);
        let k = device.upload_f32(&vec![0.05f32; kv * width]).expect("k");
        let v = device.upload_f32(&vec![0.05f32; kv * width]).expect("v");
        let kreal = device.upload_i32(&[kv as i32]).expect("kreal");

        let timed = |q_len: usize, tiles: usize| -> f64 {
            let q = device.upload_f32(&vec![0.05f32; q_len * width]).expect("q");
            for _ in 0..2 {
                for _ in 0..tiles {
                    let _ = device.mha_dev(&q, &k, &v, q_len, kv, &kreal, width, heads).expect("mha");
                }
            }
            device.sync().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                for _ in 0..tiles {
                    let _ = device.mha_dev(&q, &k, &v, q_len, kv, &kreal, width, heads).expect("mha");
                }
            }
            device.sync().expect("sync");
            t0.elapsed().as_secs_f64() * 1e3 / 3.0
        };

        eprintln!("  kv={kv} tokens, width={width}, heads={heads}");
        for &n_o in &[4usize, 256, 5000] {
            let eager = timed(n_o, 1);
            let mut line = format!("  n_o={n_o:5}  eager {eager:8.3} ms |");
            for &bq in &[8usize, 256] {
                let tiles = n_o.div_ceil(bq);
                let t = timed(bq, tiles);
                line.push_str(&format!(" BQ={bq:3} x{tiles:3} {t:8.3} ms ({:+.1}%)",
                                       100.0 * (t - eager) / eager));
            }
            eprintln!("{line}");
        }
    }

    #[test]
    #[ignore = "loads the device; run on the card"]
    fn the_option_axis_graphed_tile_versus_eager() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let (width, heads, kv) = (1024usize, 16usize, 8192usize);
        let k = device.upload_f32(&vec![0.05f32; kv * width]).expect("k");
        let v = device.upload_f32(&vec![0.05f32; kv * width]).expect("v");
        let kreal = device.upload_i32(&[kv as i32]).expect("kreal");

        let eager = |n_o: usize| -> f64 {
            let q = device.upload_f32(&vec![0.05f32; n_o * width]).expect("q");
            for _ in 0..2 { let _ = device.mha_dev(&q, &k, &v, n_o, kv, &kreal, width, heads).unwrap(); }
            device.sync().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..10 { let _ = device.mha_dev(&q, &k, &v, n_o, kv, &kreal, width, heads).unwrap(); }
            device.sync().unwrap();
            t0.elapsed().as_secs_f64() * 1e3 / 10.0
        };

        eprintln!("  kv={kv} tokens, width={width}");
        for &n_o in &[4usize, 256, 5000] {
            let e = eager(n_o);
            let mut line = format!("  n_o={n_o:5}  eager {e:8.3} ms |");
            for &bq in &[4usize, 8, 256, 512, 1024] {
                let tiles = n_o.div_ceil(bq);
                let q = device.upload_f32(&vec![0.05f32; bq * width]).expect("q");

                for _ in 0..3 { let _ = device.mha_dev(&q, &k, &v, bq, kv, &kreal, width, heads).unwrap(); }
                device.sync().unwrap();
                let cap = match device.capture(|| device.mha_dev(&q, &k, &v, bq, kv, &kreal, width, heads).map(|_| ())) {
                    Ok(c) => c,
                    Err(err) => { line.push_str(&format!(" BQ={bq:4} CAPTURE-FAILED({err})")); continue; }
                };
                device.sync().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..10 { for _ in 0..tiles { cap.launch().unwrap(); } }
                device.sync().unwrap();
                let t = t0.elapsed().as_secs_f64() * 1e3 / 10.0;
                line.push_str(&format!(" BQ={bq:4} x{tiles:3} {t:8.3} ms ({:+.1}%)", 100.0 * (t - e) / e));
            }
            eprintln!("{line}");
        }
    }

    #[test]
    #[ignore]
    fn the_head_partial_mma_matches_the_scalar_one() {
        let d = match Device::open() { Ok(d) => d, Err(_) => return };

        let hd: usize = std::env::var("CLEF_PARTIAL_HD").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let q_len: usize = std::env::var("CLEF_PARTIAL_Q").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let rows: usize = std::env::var("CLEF_PARTIAL_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let (_, heads) = (16usize,
                             std::env::var("CLEF_PARTIAL_HEADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1));
        let width = heads * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();

        let known = matches!(env_once!("CLEF_PARTIAL_KNOWN"), Some(v) if v != "0");
        let mut q = vec![0.0f32; q_len * width];
        let mut k = vec![0.0f32; rows * width];
        let mut v = vec![0.0f32; rows * width];
        if known {
            for i in 0..q_len * width { q[i] = 1.0; }
            for d in 0..width { k[d] = 1.0; }

            let uni = matches!(env_once!("CLEF_PARTIAL_UNIFORM"), Some(v) if v != "0");
            for d in 0..width {
                v[d] = if uni { 1.0 } else { (d + 1) as f32 / width as f32 };
            }
            for r in 1..rows { for d in 0..width { k[r * width + d] = 0.0; v[r * width + d] = 0.0; } }
            eprintln!("  KNOWN ANSWER: m must be {:.4} for row 0 (hd {} x scale x log2e)",
                      hd as f32 * scale * std::f32::consts::LOG2_E, hd);
        } else {
            for i in 0..q_len * width { q[i] = ((i * 7 % 13) as f32 - 6.0) * 0.1; }
            for i in 0..rows * width { k[i] = ((i * 5 % 11) as f32 - 5.0) * 0.1;
                                       v[i] = ((i * 3 % 17) as f32 - 8.0) * 0.1; }
        }
        let bytes = |v: &Vec<f32>| -> Vec<u8> {
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }.to_vec()
        };
        let dq = d.upload(&bytes(&q)).unwrap();
        let dk = d.upload(&bytes(&k)).unwrap();
        let dv = d.upload(&bytes(&v)).unwrap();
        let per = q_len * heads * (hd + 2);

        let a = d.alloc_zeroed(per * 4).unwrap();
        let b = d.alloc_zeroed(per * 4).unwrap();

        d.mha_partial_scalar_for_test(&dq, &dk, &dv, &a, q_len, rows, width, heads, scale, rows).unwrap();

        let bq = d.to_bf16_dev(&dq, q_len * width).unwrap();
        let bk = d.to_bf16_dev(&dk, rows * width).unwrap();
        let bv = d.to_bf16_dev(&dv, rows * width).unwrap();

        let dbg = d.alloc_zeroed(256 * 4).unwrap();

        d.attention_partial_dev(&bq, &bk, &bv, &b, q_len, rows, heads, hd, 0).unwrap();
        d.sync().unwrap();
        let to_f32s = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let hv = to_f32s(d.download(&dbg).unwrap());

        let raw = d.download(&bv).unwrap();
        let bf: Vec<f32> = raw.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
        eprintln!("  bf16 INPUT v[0..8]: {:?}", &bf[..8]);
        eprintln!("  KERNEL sees v[0..8]: {:?}", &hv[64..72]);
        eprintln!("  plumbing: dbg[200]={} (want 1234)  raw bits of v[0]={} (want {})",
                  hv[200], hv[201], u16::from_le_bytes([raw[0], raw[1]]));
        let want: Vec<f32> = (0..64).map(|d| {
            let x = if known { if matches!(env_once!("CLEF_PARTIAL_UNIFORM"), Some(v) if v != "0") { 1.0 }
                             else { (d + 1) as f32 / width as f32 } } else { v[d] };

            let bits = x.to_bits();
            f32::from_bits(bits & 0xffff_0000)
        }).collect();
        let bad: Vec<(usize, f32, f32)> = (0..64).filter(|&i| (hv[i] - want[i]).abs() > 1e-4)
            .map(|i| (i, want[i], hv[i])).collect();
        eprintln!("  STAGED V key 0, outdims 0..16: {:?}", &hv[..16]);
        eprintln!("  expected                     : {:?}", &want[..16]);
        if !bad.is_empty() {
            eprintln!("  STAGING MISMATCH at {} of 64 outdims, first 6: {:?}", bad.len(), &bad[..bad.len().min(6)]);
        } else {
            eprintln!("  STAGED V matches for all 64 outdims - the staging is correct");
        }
        let f32s = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let ha = f32s(d.download(&a).unwrap());
        let hb = f32s(d.download(&b).unwrap());
        eprintln!("  scalar (m,l,O): {:?}", &ha[..10]);
        eprintln!("  mma    (m,l,O): {:?}", &hb[..10]);

        let n0 = q_len * heads * (hd + 2);
        let (wi, wv) = ha[..n0].iter().zip(hb[..n0].iter())
            .enumerate()
            .map(|(i, (x, y))| (i, (x - y).abs() / x.abs().max(1.0)))
            .fold((0usize, 0.0f32), |acc, cur| if cur.1 > acc.1 { cur } else { acc });
        eprintln!("  worst at index {wi} (row {}, field {}): scalar {:?} mma {:?}",
                  wi / (heads * (hd + 2)),
                  if wi % (hd + 2) < 2 { format!("m/l {}", wi % (hd + 2)) } else { format!("O[{}]", wi % (hd + 2) - 2) },
                  ha[wi], hb[wi]);

        let n = q_len * heads * (hd + 2);
        let worst = ha[..n].iter().zip(hb[..n].iter())
            .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
            .fold(0.0f32, f32::max);
        eprintln!("  worst relative diff over the whole partial = {worst:.3e}");

        eprintln!("  rows {rows}: scalar m={:.6} l={:.6}   mma m={:.6} l={:.6}",
                  ha[0], ha[1], hb[0], hb[1]);

        eprintln!("  scalar O[0..8]: {:?}", &ha[2..10]);
        eprintln!("  mma    O[0..8]: {:?}", &hb[2..10]);

        if matches!(env_once!("CLEF_PARTIAL_ONLY"), Some(v) if v != "0") {
            eprintln!("  ONLY=1: single launch, skipping the two-split section");
            assert!(worst < 2e-2, "the MMA partial disagrees with the scalar one");
            return;
        }
        let half = rows / 2;
        let pa = d.alloc_zeroed(2 * per * 4).unwrap();
        let pb = d.alloc_zeroed(2 * per * 4).unwrap();
        let oa = d.alloc_zeroed(q_len * width * 4).unwrap();
        let ob = d.alloc_zeroed(q_len * width * 4).unwrap();
        for sp in 0..2 {
            let ko = d.view(&dk, sp * half * width, half * width);
            let vo = d.view(&dv, sp * half * width, half * width);
            let kko = d.view(&bk, sp * half * width, half * width);
            let vvo = d.view(&bv, sp * half * width, half * width);
            let sa = d.view(&pa, sp * per, per);
            let sb = d.view(&pb, sp * per, per);
            d.mha_partial_scalar_for_test(&dq, &ko, &vo, &sa, q_len, half, width, heads, scale, half).unwrap();
            d.attention_partial_dev(&bq, &kko, &vvo, &sb, q_len, half, heads, hd, sp).unwrap();
        }
        d.mha_combine_for_test(&pa, &oa, q_len, width, heads, 2).unwrap();
        d.mha_combine_for_test(&pb, &ob, q_len, width, heads, 2).unwrap();
        d.sync().unwrap();
        let fa = f32s(d.download(&oa).unwrap());
        let fb = f32s(d.download(&ob).unwrap());
        let nw = q_len * width;
        let worst_c = fa[..nw].iter().zip(fb[..nw].iter())

            .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
            .fold(0.0f32, f32::max);
        eprintln!("  TWO SPLITS + COMBINE: scalar {:?}  mma {:?}", &fa[..8], &fb[..8]);
        eprintln!("  combined scalar min/max {:.3e}/{:.3e}  mma min/max {:.3e}/{:.3e}",
                  fa[..nw].iter().cloned().fold(f32::INFINITY, f32::min),
                  fa[..nw].iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                  fb[..nw].iter().cloned().fold(f32::INFINITY, f32::min),
                  fb[..nw].iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        eprintln!("  combined worst relative diff = {worst_c:.3e}");

        assert!(worst < 2e-2, "the MMA partial disagrees with the scalar one");

        let _ = worst_c;
    }

    #[test]
    #[ignore]
    fn what_the_merge_costs() {
        let Ok(device) = Device::open() else {
            eprintln!("SKIP: no device");
            return;
        };
        let (heads, hd, splits) = (16usize, 256usize, 4usize);
        let per = heads * (hd + 2);
        eprintln!("  rows-per-block={} — merge cost per launch, by pass size", combine_rows_per_block());
        eprintln!("  {:>8} {:>7} {:>10} {:>10} {:>9}", "rows", "blocks", "waves", "ms", "GB/s");
        for rows in [512usize, 1024, 2048, 4096, 8192, 16384] {
            let partial = device.alloc(splits * rows * per * 4).expect("partial");
            let out = device.alloc(rows * heads * hd * 4).expect("out");
            let run = || device.mha_combine_dev(&partial, &out, rows, heads * hd, heads, splits);
            for _ in 0..3 { run().expect("warm"); }
            device.sync().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..10 { run().expect("run"); }
            device.sync().expect("sync");
            let ms = t0.elapsed().as_secs_f64() * 1e3 / 10.0;
            let blocks = heads as f64 * (rows as f64 / combine_rows_per_block() as f64).ceil();
            let traffic = (splits * rows * per * 4 + rows * heads * hd * 4) as f64;
            eprintln!("  {rows:>8} {blocks:>7.0} {:>10.2} {ms:>10.3} {:>9.0}",
                      (blocks / 170.0).ceil(), traffic / (ms * 1e-3) / 1e9);
        }
    }

    #[test]
    #[ignore]
    fn the_attention_at_a_chosen_cache_length() {
        let device = match Device::open() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let seq: usize = std::env::var("CLEF_ATT_PROBE_SEQ").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(2048);
        let (heads, kv_heads, hd) = (16usize, 4usize, 256usize);
        let kv_len: usize = std::env::var("CLEF_ATT_PROBE_KV").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(4096);
        assert!(kv_len >= seq, "the cache must hold at least one pass");
        let q = device.upload_f32(&vec![0.05f32; seq * heads * hd]).expect("q");
        let k = device.upload_f32(&vec![0.05f32; seq * kv_heads * hd]).expect("k");
        let v = device.upload_f32(&vec![0.05f32; seq * kv_heads * hd]).expect("v");

        let ck = device.alloc(kv_len * kv_heads * hd * 4).expect("ck");
        let cv = device.alloc(kv_len * kv_heads * hd * 4).expect("cv");
        let clen = device.upload_i32(&[(kv_len - seq) as i32]).expect("clen");
        let cache = KvCache { k: &ck, v: &cv, len: &clen, pos_map: None, base_map: None,
                              slot_map: None, capacity: kv_len, positions: kv_len };
        let run = || device.attention_dev_mma(&q, &k, &v, seq, heads, kv_heads, hd,
                                             None, None, 0, None, Some(cache));
        for _ in 0..2 { let _ = run().expect("mma"); }
        device.sync().expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..3 { let _ = run().expect("mma"); }
        device.sync().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 3.0;
        let flop = 2.0 * seq as f64 * kv_len as f64 * heads as f64 * hd as f64 * 2.0;
        eprintln!("  q_len={seq:5} kv_len={kv_len:6}: {ms:9.3} ms  -> {:8.1} TFLOP/s  {:7.3} ms per 1k queries",
                  flop / (ms * 1e-3) / 1e12, ms * 1000.0 / seq as f64);
    }
}
