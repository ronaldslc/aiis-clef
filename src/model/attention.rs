use crate::loader::Checkpoint;
use crate::model::norm::{rms_norm, sigmoid};
use crate::model::{matmul, TextConfig};
use anyhow::{bail, Result};

pub struct AttentionWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
}

impl AttentionWeights {
    pub fn load(checkpoint: &Checkpoint, config: &TextConfig, index: usize) -> Result<Self> {
        let p = config.layer_prefix(checkpoint.text_prefix()?, index);
        let attn = format!("{p}.self_attn");

        let matrix = |prefix: &str, rows: usize, cols: usize| -> Result<Vec<f32>> {
            let projection = checkpoint.projection(prefix)?;
            if projection.rows != rows || projection.cols != cols {
                bail!(
                    "{prefix} is [{}, {}], expected [{rows}, {cols}]",
                    projection.rows,
                    projection.cols
                );
            }
            Ok(projection.dequantise())
        };
        let vector = |name: &str, len: usize| -> Result<Vec<f32>> {
            let (values, shape) = checkpoint.tensor_f32(name)?;
            if values.len() != len {
                bail!("{name} has {} values ({shape:?}), expected {len}", values.len());
            }
            Ok(values)
        };

        let hidden = config.hidden_size;
        let hd = config.head_dim;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        Ok(Self {
            input_layernorm: vector(&format!("{p}.input_layernorm.weight"), hidden)?,
            post_attention_layernorm: vector(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
            q_proj: matrix(&format!("{attn}.q_proj"), heads * hd * 2, hidden)?,
            k_proj: matrix(&format!("{attn}.k_proj"), kv_heads * hd, hidden)?,
            v_proj: matrix(&format!("{attn}.v_proj"), kv_heads * hd, hidden)?,
            o_proj: matrix(&format!("{attn}.o_proj"), hidden, heads * hd)?,
            q_norm: vector(&format!("{attn}.q_norm.weight"), hd)?,
            k_norm: vector(&format!("{attn}.k_norm.weight"), hd)?,
            mlp_gate: matrix(&format!("{p}.mlp.gate_proj"), config.intermediate_size, hidden)?,
            mlp_up: matrix(&format!("{p}.mlp.up_proj"), config.intermediate_size, hidden)?,
            mlp_down: matrix(&format!("{p}.mlp.down_proj"), hidden, config.intermediate_size)?,
        })
    }
}

pub fn rope_tables(positions: &[usize], rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    let seq = positions.len();
    let mut cos = vec![0.0f32; seq * rotary_dim];
    let mut sin = vec![0.0f32; seq * rotary_dim];
    for (t, &p) in positions.iter().enumerate() {
        for j in 0..rotary_dim {
            let f = p as f32 * inv_freq[j % half];
            cos[t * rotary_dim + j] = f.cos();
            sin[t * rotary_dim + j] = f.sin();
        }
    }
    (cos, sin)
}

#[inline]
fn rotate_half(x: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = Vec::with_capacity(x.len());
    out.extend(x[half..].iter().map(|v| -v));
    out.extend_from_slice(&x[..half]);
    out
}

pub fn apply_rope(vec: &mut [f32], cos: &[f32], sin: &[f32]) {
    let rot = cos.len();
    let rotated = rotate_half(&vec[..rot]);
    for j in 0..rot {
        vec[j] = vec[j] * cos[j] + rotated[j] * sin[j];
    }
}

pub fn causal_gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
) -> Vec<f32> {
    let group = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0.0f32; seq * heads * hd];
    let mut scores = vec![0.0f32; seq];
    for t in 0..seq {
        for hh in 0..heads {
            let kvh = hh / group;
            let q_t = &q[(t * heads + hh) * hd..(t * heads + hh + 1) * hd];
            let mut max = f32::NEG_INFINITY;
            for s_i in 0..=t {
                let k_s = &k[(s_i * kv_heads + kvh) * hd..(s_i * kv_heads + kvh + 1) * hd];
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += q_t[d] * k_s[d];
                }
                scores[s_i] = acc * scale;
                if scores[s_i] > max {
                    max = scores[s_i];
                }
            }
            let mut sum = 0.0f32;
            for s_i in 0..=t {
                scores[s_i] = (scores[s_i] - max).exp();
                sum += scores[s_i];
            }
            let o_t = &mut o[(t * heads + hh) * hd..(t * heads + hh + 1) * hd];
            for s_i in 0..=t {
                let p = scores[s_i] / sum;
                let v_s = &v[(s_i * kv_heads + kvh) * hd..(s_i * kv_heads + kvh + 1) * hd];
                for d in 0..hd {
                    o_t[d] += p * v_s[d];
                }
            }
        }
    }
    o
}

pub fn attention_layer(
    x: &[f32],
    w: &AttentionWeights,
    config: &TextConfig,
    positions: &[usize],
) -> Vec<f32> {
    let seq = positions.len();
    let hidden = config.hidden_size;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let hd = config.head_dim;
    let rotary_dim = config.rotary_dim();
    let group = heads / kv_heads;
    assert_eq!(x.len(), seq * hidden);

    let mut h = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        rms_norm(
            &x[t * hidden..(t + 1) * hidden],
            &w.input_layernorm,
            config.rms_norm_eps,
            &mut h[t * hidden..(t + 1) * hidden],
        );
    }

    let mut qg = vec![0.0f32; seq * heads * hd * 2];
    matmul(&w.q_proj, heads * hd * 2, hidden, &h, seq, &mut qg);
    let mut k_raw = vec![0.0f32; seq * kv_heads * hd];
    matmul(&w.k_proj, kv_heads * hd, hidden, &h, seq, &mut k_raw);
    let mut v_raw = vec![0.0f32; seq * kv_heads * hd];
    matmul(&w.v_proj, kv_heads * hd, hidden, &h, seq, &mut v_raw);

    let mut q = vec![0.0f32; seq * heads * hd];
    let mut gate = vec![0.0f32; seq * heads * hd];
    for t in 0..seq {
        for hh in 0..heads {
            let src = t * heads * hd * 2 + hh * hd * 2;
            q[(t * heads + hh) * hd..(t * heads + hh + 1) * hd]
                .copy_from_slice(&qg[src..src + hd]);
            gate[(t * heads + hh) * hd..(t * heads + hh + 1) * hd]
                .copy_from_slice(&qg[src + hd..src + 2 * hd]);
        }
    }

    let mut k = vec![0.0f32; seq * kv_heads * hd];
    for t in 0..seq {
        for hh in 0..heads {
            let off = (t * heads + hh) * hd;
            let src: Vec<f32> = q[off..off + hd].to_vec();
            rms_norm(&src, &w.q_norm, config.rms_norm_eps, &mut q[off..off + hd]);
        }
        for hh in 0..kv_heads {
            let off = (t * kv_heads + hh) * hd;
            rms_norm(
                &k_raw[off..off + hd],
                &w.k_norm,
                config.rms_norm_eps,
                &mut k[off..off + hd],
            );
        }
    }
    let (cos, sin) = rope_tables(positions, rotary_dim, config.rope_theta);
    for t in 0..seq {
        let c = &cos[t * rotary_dim..(t + 1) * rotary_dim];
        let s = &sin[t * rotary_dim..(t + 1) * rotary_dim];
        for hh in 0..heads {
            let off = (t * heads + hh) * hd;
            apply_rope(&mut q[off..off + hd], c, s);
        }
        for hh in 0..kv_heads {
            let off = (t * kv_heads + hh) * hd;
            apply_rope(&mut k[off..off + hd], c, s);
        }
    }

    let mut o = causal_gqa_attention(&q, &k, &v_raw, seq, heads, kv_heads, hd);
    for (i, g) in gate.iter().enumerate() {
        o[i] *= sigmoid(*g);
    }
    let mut mixed = vec![0.0f32; seq * hidden];
    matmul(&w.o_proj, hidden, heads * hd, &o, seq, &mut mixed);

    let mut out = vec![0.0f32; seq * hidden];
    for i in 0..seq * hidden {
        out[i] = x[i] + mixed[i];
    }
    let mut h2 = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        rms_norm(
            &out[t * hidden..(t + 1) * hidden],
            &w.post_attention_layernorm,
            config.rms_norm_eps,
            &mut h2[t * hidden..(t + 1) * hidden],
        );
    }
    let inter = config.intermediate_size;
    let mut g = vec![0.0f32; seq * inter];
    let mut u = vec![0.0f32; seq * inter];
    matmul(&w.mlp_gate, inter, hidden, &h2, seq, &mut g);
    matmul(&w.mlp_up, inter, hidden, &h2, seq, &mut u);
    let act: Vec<f32> = g
        .iter()
        .zip(u.iter())
        .map(|(a, b)| (a * sigmoid(*a)) * b)
        .collect();
    let mut mlp = vec![0.0f32; seq * hidden];
    matmul(&w.mlp_down, hidden, inter, &act, seq, &mut mlp);
    for i in 0..seq * hidden {
        out[i] += mlp[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_half_pairs_the_far_half_with_the_near_one() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(rotate_half(&x), vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn rope_at_position_zero_changes_nothing() {
        let (cos, sin) = rope_tables(&[0], 64, 1e7);
        let mut v: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let before = v.clone();
        apply_rope(&mut v, &cos, &sin);
        for (a, b) in v.iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 must be the identity: {a} vs {b}");
        }
    }

    #[test]
    fn rope_is_norm_preserving() {
        let (cos, sin) = rope_tables(&[7], 64, 1e7);
        let mut v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut v, &cos, &sin);
        let after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((before - after).abs() < 1e-4, "norm changed: {before} -> {after}");
    }

    #[test]
    fn the_tail_beyond_rotary_dim_is_untouched() {
        let (cos, sin) = rope_tables(&[3], 64, 1e7);
        let mut v: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let tail = v[64..].to_vec();
        apply_rope(&mut v, &cos, &sin);
        assert_eq!(&v[64..], &tail[..], "the pass-through region must not be modified");
    }

    fn fixture() -> Option<(Vec<f32>, Vec<f32>, usize, usize)> {
        let candidates = [
            "tests/eval/clef_attn_fixture.json",
            "../../tests/eval/clef_attn_fixture.json",
            "../tests/eval/clef_attn_fixture.json",
            "tests/eval/clef_attn_fixture.json",
        ];
        let path = candidates.iter().map(std::path::Path::new).find(|p| p.exists())?;
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let decode = |key: &str| -> Vec<f32> {
            use base64::Engine;
            let b64 = json["hidden"][key].as_str().expect("fixture key");
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .expect("base64")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let tokens = json["meta"]["input_ids"].as_array()?.len();
        let input = decode("layer2");
        let hidden = input.len() / tokens;
        Some((input, decode("layer3"), tokens, hidden))
    }

    #[test]
    fn attention_layer_three_matches_the_reference() {
        let Some((input, want, tokens, hidden)) = fixture() else {
            eprintln!("SKIP: clef_attn_fixture.json not found");
            return;
        };

        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: /models/clef-flash not present; run where the checkpoint is mounted");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash").expect("config.json");
        assert_eq!(config.layer_kind(3), crate::model::LayerKind::FullAttention);
        assert_eq!(hidden, config.hidden_size);

        let w = AttentionWeights::load(&checkpoint, &config, 3).expect("layer 3 weights");
        let positions: Vec<usize> = (0..tokens).collect();
        let got = attention_layer(&input, &w, &config, &positions);

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (g, e) in got.iter().zip(want.iter()) {
            let d = (g - e).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / got.len() as f64) as f32;
        let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
        eprintln!("layer 3 (attention): max |diff| = {max_abs:.6e}, mean |diff| = {mean_abs:.6e}, \
                   reference mean |x| = {scale:.6e}");
        eprintln!("          relative to reference scale: {:.3e}", mean_abs / scale);

        assert!(
            max_abs < 1e-3,
            "layer 3 diverges: max |diff| {max_abs:.6e} against a reference scale of {scale:.6e}"
        );
    }
}