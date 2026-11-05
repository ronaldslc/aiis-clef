use crate::loader::Checkpoint;
use crate::model::attention::{attention_layer, AttentionWeights};
use crate::model::layer::{deltanet_layer, DeltaNetWeights};
use crate::model::norm::rms_norm;
use crate::model::{LayerKind, TextConfig};
use anyhow::Result;

pub enum LayerWeights {
    Linear(Box<DeltaNetWeights>),
    Full(Box<AttentionWeights>),
}

pub struct Backbone {
    pub layers: Vec<LayerWeights>,
    pub embed_tokens: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub config: TextConfig,
}

impl Backbone {
    pub fn load(checkpoint: &Checkpoint, config: &TextConfig) -> Result<Self> {
        let prefix = checkpoint.text_prefix()?.to_string();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(match config.layer_kind(i) {
                LayerKind::LinearAttention => {
                    LayerWeights::Linear(Box::new(DeltaNetWeights::load(checkpoint, config, i)?))
                }
                LayerKind::FullAttention => {
                    LayerWeights::Full(Box::new(AttentionWeights::load(checkpoint, config, i)?))
                }
            });
        }
        let (embed_tokens, shape) =
            checkpoint.tensor_f32(&format!("{prefix}.embed_tokens.weight"))?;
        if shape.len() != 2 || shape[0] != config.vocab_size || shape[1] != config.hidden_size {
            anyhow::bail!("embed_tokens.weight has shape {shape:?}, expected [vocab, hidden]");
        }
        let (final_norm, _) = checkpoint.tensor_f32(&format!("{prefix}.norm.weight"))?;
        Ok(Self {
            layers,
            embed_tokens,
            final_norm,
            config: config.clone(),
        })
    }

    pub fn embed(&self, input_ids: &[u32]) -> Vec<f32> {
        let h = self.config.hidden_size;
        let mut out = vec![0.0f32; input_ids.len() * h];
        for (t, &id) in input_ids.iter().enumerate() {
            let row = id as usize * h;
            out[t * h..(t + 1) * h].copy_from_slice(&self.embed_tokens[row..row + h]);
        }
        out
    }

    pub fn forward_all(&self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let tokens = input_ids.len();
        let positions: Vec<usize> = (0..tokens).collect();
        let mut h = self.embed(input_ids);
        let mut states = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            h = match layer {
                LayerWeights::Linear(w) => {
                    let mut out = vec![0.0f32; h.len()];
                    deltanet_layer(&self.config, w, &h, tokens, &mut out)?;
                    out
                }
                LayerWeights::Full(w) => attention_layer(&h, w, &self.config, &positions),
            };
            debug_assert!(h.iter().all(|v| v.is_finite()), "layer {i} produced a non-finite value");
            states.push(h.clone());
        }
        Ok(states)
    }

    pub fn final_norm(&self, h: &[f32]) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let tokens = h.len() / hidden;
        let mut out = vec![0.0f32; h.len()];
        for t in 0..tokens {
            rms_norm(
                &h[t * hidden..(t + 1) * hidden],
                &self.final_norm,
                self.config.rms_norm_eps,
                &mut out[t * hidden..(t + 1) * hidden],
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Option<(Vec<u32>, Vec<Vec<f32>>)> {
        let candidates = [
            "tests/eval/clef_all_layers_fixture.json",
            "../../tests/eval/clef_all_layers_fixture.json",
            "../tests/eval/clef_all_layers_fixture.json",
            "tests/eval/clef_all_layers_fixture.json",
        ];
        let path = candidates.iter().map(std::path::Path::new).find(|p| p.exists())?;
        let json: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let ids: Vec<u32> = json["meta"]["input_ids"]
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as u32)
            .collect();
        let mut states = Vec::new();
        for i in 0..32 {
            let key = format!("layer{i}");
            let b64 = json["hidden"][&key].as_str()?;
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
            states.push(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>(),
            );
        }
        Some((ids, states))
    }

    #[test]
    fn the_activation_dtype_ladder() {
        let Some((ids, _want)) = fixture() else {
            eprintln!("SKIP: clef_all_layers_fixture.json not found");
            return;
        };
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: /models/clef-flash not present; run where the checkpoint is mounted");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash").expect("config.json");
        let backbone = Backbone::load(&checkpoint, &config).expect("backbone weights");

        fn bf16(x: f32) -> f32 {
            let b = x.to_bits();
            let lsb = (b >> 16) & 1;
            f32::from_bits(b.wrapping_add(0x7fff + lsb) & 0xffff_0000)
        }

        fn hi(x: f32) -> f32 { bf16(x) }
        fn hilo(x: f32) -> f32 { let h = bf16(x); h + bf16(x - h) }

        let base = backbone.forward_all(&ids).expect("forward");
        for (name, round) in [("bf16", hi as fn(f32) -> f32), ("bf16 hi/lo", hilo as fn(f32) -> f32)] {
            let tokens = ids.len();
            let positions: Vec<usize> = (0..tokens).collect();
            let mut h = backbone.embed(&ids);
            let mut worst_rel = 0.0f32;
            let mut worst_layer = 0usize;
            for (i, layer) in backbone.layers.iter().enumerate() {
                h = match layer {
                    LayerWeights::Linear(w) => {
                        let mut out = vec![0.0f32; h.len()];
                        deltanet_layer(&config, w, &h, tokens, &mut out).expect("layer");
                        out
                    }
                    LayerWeights::Full(w) => attention_layer(&h, w, &config, &positions),
                };
                for v in h.iter_mut() {
                    *v = round(*v);
                }

                let want = &base[i];

                let dmax = h.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let wmax = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let dsum: f32 = h.iter().zip(want.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                let wsum: f32 = want.iter().map(|v| v * v).sum();
                let rel_max = dmax / wmax.max(f32::MIN_POSITIVE);
                let rel_rms = (dsum / wsum.max(f32::MIN_POSITIVE)).sqrt();
                let _ = rel_max;
                let rel = rel_rms;
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_layer = i;
                }
            }
            eprintln!(
                "  {name:12}: worst RMS-relative error {worst_rel:.3e} at layer {worst_layer} of 32"
            );

            if worst_rel > 0.0 {
                let tokens = ids.len();
                let positions: Vec<usize> = (0..tokens).collect();
                let mut h2 = backbone.embed(&ids);
                let mut line = String::new();
                for (i, layer) in backbone.layers.iter().enumerate() {
                    h2 = match layer {
                        LayerWeights::Linear(w) => {
                            let mut out = vec![0.0f32; h2.len()];
                            deltanet_layer(&config, w, &h2, tokens, &mut out).expect("layer");
                            out
                        }
                        LayerWeights::Full(w) => attention_layer(&h2, w, &config, &positions),
                    };
                    for v in h2.iter_mut() { *v = round(*v); }
                    let want = &base[i];
                    let d: f32 = h2.iter().zip(want.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    let w: f32 = want.iter().map(|v| v * v).sum();
                    line.push_str(&format!("{:.1e} ", (d / w.max(f32::MIN_POSITIVE)).sqrt()));
                }
                eprintln!("              per layer 0..31: {line}");
            }
        }
    }

    #[test]
    fn all_thirty_two_layers_match_the_reference() {
        let Some((ids, want)) = fixture() else {
            eprintln!("SKIP: clef_all_layers_fixture.json not found");
            return;
        };
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: /models/clef-flash not present; run where the checkpoint is mounted");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash").expect("config.json");
        let backbone = Backbone::load(&checkpoint, &config).expect("backbone weights");

        for i in 0..32 {
            let expected = if (i + 1) % config.full_attention_interval == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            };
            assert_eq!(config.layer_kind(i), expected, "layer {i} kind");
        }

        let got = backbone.forward_all(&ids).expect("forward");
        assert_eq!(got.len(), 32);

        let mut worst_relative = 0.0f32;
        let mut worst_layer = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.len(), w.len(), "layer {i} length");
            let mut max_abs = 0.0f32;
            let mut sum_abs = 0.0f64;
            for (a, b) in g.iter().zip(w.iter()) {
                let d = (a - b).abs();
                max_abs = max_abs.max(d);
                sum_abs += d as f64;
            }
            let mean_abs = (sum_abs / g.len() as f64) as f32;
            let scale = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32;
            let relative = mean_abs / scale.max(1e-12);
            if relative > worst_relative {
                worst_relative = relative;
                worst_layer = i;
            }
            let kind = match config.layer_kind(i) {
                LayerKind::LinearAttention => "deltanet",
                LayerKind::FullAttention => "attention",
            };
            eprintln!(
                "  layer {i:2} [{kind:9}] max |diff| = {max_abs:.3e}  mean = {mean_abs:.3e}  \
                 ref mean |x| = {scale:.3e}  relative = {relative:.3e}"
            );
        }
        eprintln!("  worst: layer {worst_layer} at {worst_relative:.3e} relative");
        assert!(
            worst_relative < 1e-4,
            "layer {worst_layer} diverges at {worst_relative:.3e} relative"
        );
    }
}

#[cfg(test)]
mod option_probes {
    use super::*;

    fn bf16(x: f32) -> f32 {
        bf16_round(x, 1.0)
    }

    fn e4m3_round(x: f32, scale: f32) -> f32 {
        let v = x / scale;
        if v == 0.0 {
            return 0.0;
        }
        let sign = if v < 0.0 { -1.0f32 } else { 1.0 };
        let a = v.abs().min(448.0);

        let e = a.log2().floor();
        let (step, biased) = if e < -6.0 {
            (2f32.powi(-9), 0i32)
        } else {
            (2f32.powi(e as i32 - 3), (e as i32 + 7).max(1))
        };
        let q = ((a / step).round() * step).min(448.0);
        let _ = biased;
        sign * q * scale
    }

    fn bf16_round(x: f32, scale: f32) -> f32 {
        let v = x / scale;
        let b = v.to_bits();
        let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
        f32::from_bits(r & 0xffff_0000) * scale
    }

    fn fp8_round(xs: &mut [f32]) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            return;
        }
        let s = amax / 448.0;
        for v in xs.iter_mut() {
            *v = e4m3_round(*v, s);
        }
    }

    fn run(b: &Backbone, ids: &[u32], mode: u8) -> Result<Vec<Vec<f32>>> {
        let tokens = ids.len();
        let positions: Vec<usize> = (0..tokens).collect();
        let mut h = b.embed(ids);
        let mut states = Vec::with_capacity(b.layers.len());
        for layer in b.layers.iter() {
            h = match layer {
                LayerWeights::Linear(w) => {
                    let mut o = vec![0.0f32; h.len()];
                    deltanet_layer(&b.config, w, &h, tokens, &mut o)?;
                    o
                }
                LayerWeights::Full(w) => attention_layer(&h, w, &b.config, &positions),
            };
            match mode {
                1 => h.iter_mut().for_each(|v| *v = bf16(*v)),
                2 => fp8_round(&mut h),
                _ => {}
            }
            states.push(h.clone());
        }
        Ok(states)
    }

    fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
        let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            mx = mx.max((x - y).abs());
            num += ((x - y) as f64) * ((x - y) as f64);
            den += (*x as f64) * (*x as f64);
        }
        (mx, ((num / den.max(1e-30)).sqrt()) as f32)
    }

    #[test]
    #[ignore]
    fn what_the_residual_dtype_costs_in_accuracy() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: checkpoint not present");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash-nvfp4").expect("config");
        let b = match Backbone::load(&checkpoint, &config) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return;
            }
        };
        eprintln!("  backbone loaded (32 layers, host f32)");

        let ids: Vec<u32> = (0..24u32).map(|i| 1000 + i * 37).collect();
        let f32_states = run(&b, &ids, 0).expect("f32 run");
        let bf_states = run(&b, &ids, 1).expect("bf16 run");
        let fp_states = run(&b, &ids, 2).expect("fp8 run");

        eprintln!("\n  layer |        bf16 residual        |        fp8 residual");
        eprintln!("        |   max|d|    rel rms        |   max|d|    rel rms");
        for i in (0..32).step_by(4) {
            let (bm, br) = rel(&f32_states[i], &bf_states[i]);
            let (fm, fr) = rel(&f32_states[i], &fp_states[i]);
            eprintln!("  {i:5} | {bm:9.3e}  {br:9.3e}      | {fm:9.3e}  {fr:9.3e}");
        }
        let (bm, br) = rel(f32_states.last().unwrap(), bf_states.last().unwrap());
        let (fm, fr) = rel(f32_states.last().unwrap(), fp_states.last().unwrap());
        eprintln!("\n  FINAL LAYER   bf16 max|d| {bm:.3e} rel-rms {br:.3e}");
        eprintln!("  FINAL LAYER    fp8 max|d| {fm:.3e} rel-rms {fr:.3e}");

        let scale = f32_states.last().unwrap().iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!("  (reference max|h| = {scale:.4})");
    }

    #[test]
    fn what_chunked_attention_costs_in_accuracy() {
        let (seq, heads, kv_heads, hd) = (512usize, 16usize, 4usize, 256usize);
        let group = heads / kv_heads;
        let mut seed = 0x1234_5678u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
        };
        let q: Vec<f32> = (0..seq * heads * hd).map(|_| rnd()).collect();
        let k: Vec<f32> = (0..seq * kv_heads * hd).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..seq * kv_heads * hd).map(|_| rnd()).collect();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let out_full = |t: usize, h: usize| -> Vec<f32> {
            let kv = h / group;
            let qv = &q[t * heads * hd + h * hd..t * heads * hd + (h + 1) * hd];
            let mut sc: Vec<f32> = (0..=t)
                .map(|s| {
                    let kv_off = s * kv_heads * hd + kv * hd;
                    qv.iter().zip(&k[kv_off..kv_off + hd]).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = sc.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut l = 0f32;
            for s in sc.iter_mut() {
                *s = (*s - m).exp();
                l += *s;
            }
            let mut o = vec![0f32; hd];
            for (s, p) in sc.iter().enumerate() {
                let kv_off = s * kv_heads * hd + kv * hd;
                let p = p / l;
                for d in 0..hd {
                    o[d] += p * v[kv_off + d];
                }
            }
            o
        };

        let out_chunked = |t: usize, h: usize, chunk: usize| -> (Vec<f32>, usize) {
            let mut rescales = 0usize;
            let kv = h / group;
            let qv = &q[t * heads * hd + h * hd..t * heads * hd + (h + 1) * hd];
            let (mut m, mut l) = (f32::MIN, 0f32);
            let mut acc = vec![0f32; hd];
            let mut s0 = 0usize;
            while s0 <= t {
                let s1 = (s0 + chunk).min(t + 1);
                for s in s0..s1 {
                    let kv_off = s * kv_heads * hd + kv * hd;
                    let dot = qv.iter().zip(&k[kv_off..kv_off + hd]).map(|(a, b)| a * b).sum::<f32>()
                        * scale;
                    let m_new = m.max(dot);
                    if m_new > m { rescales += 1; }
                    let corr = (m - m_new).exp();
                    let p = (dot - m_new).exp();
                    l = l * corr + p;
                    for d in 0..hd {
                        acc[d] = acc[d] * corr + p * v[kv_off + d];
                    }
                    m = m_new;
                }
                s0 = s1;
            }
            (acc.iter().map(|a| a / l).collect(), rescales)
        };

        eprintln!("\n  chunked attention vs full softmax, seq={seq} heads={heads} hd={hd}, f32 both sides");
        eprintln!("   chunk |   max|d|     rel-rms        (rescales = times the running max moved)");
        for chunk in [512usize, 256, 128, 64] {
            let (mut mx, mut num, mut den, mut resc) = (0f32, 0f64, 0f64, 0usize);
            for t in [63usize, 127, 255, 383, 511] {
                for h in [0usize, 5, 15] {
                    let a = out_full(t, h);
                    let (b, r) = out_chunked(t, h, chunk);
                    resc += r;
                    for (x, y) in a.iter().zip(b.iter()) {
                        mx = mx.max((x - y).abs());
                        num += ((x - y) as f64).powi(2);
                        den += (*x as f64).powi(2);
                    }
                }
            }
            eprintln!("  {chunk:6} | {mx:9.3e}   {:.3e}   rescales={resc}", (num / den.max(1e-30)).sqrt());
        }
    }

    #[test]
    fn what_bf16_attention_costs_in_accuracy() {
        let (seq, heads, kv_heads, hd) = (512usize, 16usize, 4usize, 256usize);
        let group = heads / kv_heads;
        let mut seed = 0xABCD_1234u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
        };
        let q: Vec<f32> = (0..seq * heads * hd).map(|_| rnd()).collect();
        let k: Vec<f32> = (0..seq * kv_heads * hd).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..seq * kv_heads * hd).map(|_| rnd()).collect();
        let scale = 1.0f32 / (hd as f32).sqrt();

        fn mxfp8_block(xs: &mut [f32], block: usize) {
            for c in xs.chunks_mut(block) {
                let amax = c.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax == 0.0 { continue; }
                let s = 2f32.powf((amax / 448.0).log2().ceil());
                if s == 0.0 || !s.is_finite() { continue; }

                for v in c.iter_mut() { *v = e4m3_round(*v, s); }
            }
        }
        let rnd = |x: f32, mode: u8| -> f32 {
            match mode { 1 => bf16(x), 2 => { let mut b = [x]; mxfp8_block(&mut b, 32); b[0] }, _ => x }
        };

        let out = |t: usize, h: usize, rqkv: bool, rp: bool| -> Vec<f32> {
            let kv = h / group;
            let qv: Vec<f32> = (0..hd)
                .map(|d| { let x = q[t * heads * hd + h * hd + d]; if rqkv { bf16(x) } else { x } })
                .collect();
            let mut sc: Vec<f32> = (0..=t)
                .map(|s| {
                    let o = s * kv_heads * hd + kv * hd;
                    let mut acc = 0f32;
                    for d in 0..hd {
                        let kd = if rqkv { bf16(k[o + d]) } else { k[o + d] };
                        acc += qv[d] * kd;
                    }
                    acc * scale
                })
                .collect();
            let m = sc.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut l = 0f32;
            for x in sc.iter_mut() { *x = (*x - m).exp(); l += *x; }
            let mut o = vec![0f32; hd];
            for (s, p0) in sc.iter().enumerate() {
                let o2 = s * kv_heads * hd + kv * hd;
                let p = if rp { bf16(p0 / l) } else { p0 / l };
                for d in 0..hd {
                    let vd = if rqkv { bf16(v[o2 + d]) } else { v[o2 + d] };
                    o[d] += p * vd;
                }
            }
            o
        };

        let out_mode = |t: usize, h: usize, mode: u8| -> Vec<f32> {
            let kv = h / group;
            let mut qv = q[t * heads * hd + h * hd..t * heads * hd + (h + 1) * hd].to_vec();
            let mut krows: Vec<f32> = (0..=t).flat_map(|s| {
                let o = s * kv_heads * hd + kv * hd;
                k[o..o + hd].to_vec()
            }).collect();
            let mut vrows: Vec<f32> = (0..=t).flat_map(|s| {
                let o = s * kv_heads * hd + kv * hd;
                v[o..o + hd].to_vec()
            }).collect();
            if mode == 1 { qv.iter_mut().for_each(|x| *x = bf16(*x)); krows.iter_mut().for_each(|x| *x = bf16(*x)); vrows.iter_mut().for_each(|x| *x = bf16(*x)); }
            if mode == 2 { mxfp8_block(&mut qv, 32); mxfp8_block(&mut krows, 32); mxfp8_block(&mut vrows, 32); }
            let mut sc: Vec<f32> = (0..=t).map(|s| {
                let ko = s * hd;
                let mut acc = 0f32;
                for d in 0..hd { acc += qv[d] * krows[ko + d]; }
                acc * scale
            }).collect();
            let m = sc.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut l = 0f32;
            for x in sc.iter_mut() { *x = (*x - m).exp(); l += *x; }
            let mut probs: Vec<f32> = sc.iter().map(|p| p / l).collect();
            if mode == 1 { probs.iter_mut().for_each(|x| *x = bf16(*x)); }
            if mode == 2 { mxfp8_block(&mut probs, 32); }
            let mut o = vec![0f32; hd];
            for (s, p) in probs.iter().enumerate() {
                let vo = s * hd;
                for d in 0..hd { o[d] += p * vrows[vo + d]; }
            }
            o
        };

        let out_tensor = |t: usize, h: usize| -> Vec<f32> {
            let kv = h / group;
            let mut qv = q[t * heads * hd + h * hd..t * heads * hd + (h + 1) * hd].to_vec();
            let mut krows: Vec<f32> = (0..=t).flat_map(|s| {
                let o = s * kv_heads * hd + kv * hd; k[o..o + hd].to_vec() }).collect();
            let mut vrows: Vec<f32> = (0..=t).flat_map(|s| {
                let o = s * kv_heads * hd + kv * hd; v[o..o + hd].to_vec() }).collect();

            let sc = |xs: &mut Vec<f32>| -> f32 {
                let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax == 0.0 { return 1.0; }
                let s = amax / 448.0;
                for x in xs.iter_mut() { *x = e4m3_round(*x, s); }
                s
            };
            let (sq, sk, sv) = (sc(&mut qv), sc(&mut krows), sc(&mut vrows));
            let mut scores: Vec<f32> = (0..=t).map(|s| {
                let ko = s * hd;
                let mut acc = 0f32;
                for d in 0..hd { acc += qv[d] * krows[ko + d]; }
                acc * scale
            }).collect();
            let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut l = 0f32;
            for x in scores.iter_mut() { *x = (*x - m).exp(); l += *x; }
            let mut o = vec![0f32; hd];
            for (s, p) in scores.iter().enumerate() {
                let vo = s * hd;
                let p = p / l;
                for d in 0..hd { o[d] += p * vrows[vo + d]; }
            }
            let _ = (sq, sk, sv);
            o
        };

        let out_pvar = |t: usize, h: usize, p_mode: u8| -> Vec<f32> {
            let kv = h / group;
            let mut qv = q[t * heads * hd + h * hd..t * heads * hd + (h + 1) * hd].to_vec();
            let mut krows: Vec<f32> = (0..=t).flat_map(|s| {
                let o = s * kv_heads * hd + kv * hd; k[o..o + hd].to_vec() }).collect();
            let mut vrows: Vec<f32> = (0..=t).flat_map(|s| {
                let o = s * kv_heads * hd + kv * hd; v[o..o + hd].to_vec() }).collect();
            let sc = |xs: &mut Vec<f32>| { let a = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
                if a > 0.0 { let s = a / 448.0; for x in xs.iter_mut() { *x = e4m3_round(*x, s); } } };
            sc(&mut qv); sc(&mut krows); sc(&mut vrows);
            let mut scores: Vec<f32> = (0..=t).map(|s| {
                let ko = s * hd; let mut acc = 0f32;
                for d in 0..hd { acc += qv[d] * krows[ko + d]; } acc * scale }).collect();
            let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut l = 0f32;
            for x in scores.iter_mut() { *x = (*x - m).exp(); l += *x; }

            let mut probs: Vec<f32> = scores.iter().map(|p| p / l).collect();
            match p_mode {
                1 => probs.iter_mut().for_each(|x| *x = bf16(*x)),
                2 => {
                    let a = probs.iter().fold(0f32, |m, v| m.max(v.abs()));
                    if a > 0.0 { let s = a / 448.0;
                        for x in probs.iter_mut() { *x = e4m3_round(*x, s); } }
                }
                _ => {}
            }
            let mut o = vec![0f32; hd];
            for (s, p) in probs.iter().enumerate() {
                let vo = s * hd;
                for d in 0..hd { o[d] += p * vrows[vo + d]; }
            }
            o
        };

        eprintln!("\n  attention accuracy, f32 reference vs tensor-core operands (f32 accumulate)");
        eprintln!("   scheme              |   max|d|     rel-rms");
        for (label, mode) in [("bf16 (FA2/sm120)   ", 1u8), ("MXFP8 block-scaled ", 2u8), ("e4m3 per-TENSOR    ", 3u8)] {
            if mode == 3 {
                let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
                for t in [63usize, 127, 255, 383, 511] {
                    for h in [0usize, 5, 15] {
                        let a = out_mode(t, h, 0);
                        let b = out_tensor(t, h);
                        for (x, y) in a.iter().zip(b.iter()) {
                            mx = mx.max((x - y).abs());
                            num += ((x - y) as f64).powi(2);
                            den += (*x as f64).powi(2);
                        }
                    }
                }
                eprintln!("   {label} | {mx:9.3e}   {:.3e}", (num / den.max(1e-30)).sqrt());
                continue;
            }
            let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
            for t in [63usize, 127, 255, 383, 511] {
                for h in [0usize, 5, 15] {
                    let a = out_mode(t, h, 0);
                    let b = out_mode(t, h, mode);
                    for (x, y) in a.iter().zip(b.iter()) {
                        mx = mx.max((x - y).abs());
                        num += ((x - y) as f64).powi(2);
                        den += (*x as f64).powi(2);
                    }
                }
            }
            eprintln!("   {label} | {mx:9.3e}   {:.3e}", (num / den.max(1e-30)).sqrt());
        }
        let _ = rnd;
        eprintln!("\n  THE PROBABILITIES, with q/k/v held at e4m3 per-tensor:");
        eprintln!("   P precision          |   max|d|     rel-rms");
        for (label, pm) in [("f32 (unquantised)  ", 0u8), ("bf16               ", 1u8), ("e4m3 (per-row)     ", 2u8)] {
            let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
            for t in [63usize, 127, 255, 383, 511] {
                for h in [0usize, 5, 15] {
                    let a = out_mode(t, h, 0);
                    let b = out_pvar(t, h, pm);
                    for (x, y) in a.iter().zip(b.iter()) {
                        mx = mx.max((x - y).abs());
                        num += ((x - y) as f64).powi(2);
                        den += (*x as f64).powi(2);
                    }
                }
            }
            eprintln!("   {label} | {mx:9.3e}   {:.3e}", (num / den.max(1e-30)).sqrt());
        }
        eprintln!("   (per-operand split, bf16):");
        eprintln!("   q,k,v,P rounded?    |   max|d|     rel-rms");
        for (label, rqkv, rp) in [("q,k,v,P  (FA2 full)", true, true),
                                  ("q,k,v only        ", true, false),
                                  ("P only            ", false, true)] {
            let (mut mx, mut num, mut den) = (0f32, 0f64, 0f64);
            for t in [63usize, 127, 255, 383, 511] {
                for h in [0usize, 5, 15] {
                    let a = out(t, h, false, false);
                    let b = out(t, h, rqkv, rp);
                    for (x, y) in a.iter().zip(b.iter()) {
                        mx = mx.max((x - y).abs());
                        num += ((x - y) as f64).powi(2);
                        den += (*x as f64).powi(2);
                    }
                }
            }
            eprintln!("   {label} | {mx:9.3e}   {:.3e}", (num / den.max(1e-30)).sqrt());
        }
    }
}
