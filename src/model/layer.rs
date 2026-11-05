use crate::loader::Checkpoint;
use crate::model::conv::causal_conv1d_silu;
use crate::model::deltanet::{gated_delta_rule, repeat_heads};
use crate::model::norm::{rms_norm, rms_norm_gated, sigmoid, softplus};
use crate::model::{matmul, TextConfig};
use anyhow::{bail, Result};

pub struct DeltaNetWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub in_proj_qkv: Vec<f32>,
    pub in_proj_z: Vec<f32>,
    pub in_proj_a: Vec<f32>,
    pub in_proj_b: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub conv_weight: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
}

impl DeltaNetWeights {
    pub fn load(checkpoint: &Checkpoint, config: &TextConfig, index: usize) -> Result<Self> {
        let p = config.layer_prefix(checkpoint.text_prefix()?, index);
        let attn = format!("{p}.linear_attn");

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
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let heads = config.linear_num_value_heads;
        Ok(Self {
            input_layernorm: vector(&format!("{p}.input_layernorm.weight"), hidden)?,
            post_attention_layernorm: vector(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
            in_proj_qkv: matrix(&format!("{attn}.in_proj_qkv"), key_dim * 2 + value_dim, hidden)?,
            in_proj_z: matrix(&format!("{attn}.in_proj_z"), value_dim, hidden)?,
            in_proj_a: matrix(&format!("{attn}.in_proj_a"), heads, hidden)?,
            in_proj_b: matrix(&format!("{attn}.in_proj_b"), heads, hidden)?,
            out_proj: matrix(&format!("{attn}.out_proj"), hidden, value_dim)?,
            conv_weight: {
                let (v, shape) = checkpoint.tensor_f32(&format!("{attn}.conv1d.weight"))?;
                if shape.len() != 3 || shape[0] != config.conv_dim() || shape[2] != config.linear_conv_kernel_dim {
                    bail!("{attn}.conv1d.weight has shape {shape:?}, expected [conv_dim, 1, kernel]");
                }
                v
            },
            norm_weight: vector(&format!("{attn}.norm.weight"), config.linear_value_head_dim)?,
            a_log: vector(&format!("{attn}.A_log"), heads)?,
            dt_bias: vector(&format!("{attn}.dt_bias"), heads)?,
            mlp_gate: matrix(&format!("{p}.mlp.gate_proj"), config.intermediate_size, hidden)?,
            mlp_up: matrix(&format!("{p}.mlp.up_proj"), config.intermediate_size, hidden)?,
            mlp_down: matrix(&format!("{p}.mlp.down_proj"), hidden, config.intermediate_size)?,
        })
    }
}

pub fn stats(x: &[f32]) -> (f32, f32) {
    let mean = x.iter().map(|v| v.abs()).sum::<f32>() / x.len().max(1) as f32;
    let max = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    (mean, max)
}

pub fn deltanet_layer(
    config: &TextConfig,
    w: &DeltaNetWeights,
    x: &[f32],
    seq_len: usize,
    out: &mut [f32],
) -> Result<()> {
    let hidden = config.hidden_size;
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let k_head = config.linear_key_head_dim;
    let v_head = config.linear_value_head_dim;
    let heads = config.linear_num_value_heads;
    let ratio = config.value_heads_per_key_head();
    let conv_dim = config.conv_dim();

    if x.len() != seq_len * hidden {
        bail!("input is {} values, expected {}", x.len(), seq_len * hidden);
    }

    let mut normed = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        rms_norm(
            &x[t * hidden..(t + 1) * hidden],
            &w.input_layernorm,
            config.rms_norm_eps,
            &mut normed[t * hidden..(t + 1) * hidden],
        );
    }

    let mut mixed = vec![0.0f32; seq_len * conv_dim];
    matmul(&w.in_proj_qkv, conv_dim, hidden, &normed, seq_len, &mut mixed);

    let mut convolved = vec![0.0f32; seq_len * conv_dim];
    causal_conv1d_silu(
        &mixed,
        &w.conv_weight,
        config.linear_conv_kernel_dim,
        conv_dim,
        seq_len,
        &mut convolved,
    );

    let mut q_full = vec![0.0f32; seq_len * key_dim];
    let mut k_full = vec![0.0f32; seq_len * key_dim];
    let mut v_full = vec![0.0f32; seq_len * value_dim];
    for t in 0..seq_len {
        let row = &convolved[t * conv_dim..(t + 1) * conv_dim];
        q_full[t * key_dim..(t + 1) * key_dim].copy_from_slice(&row[..key_dim]);
        k_full[t * key_dim..(t + 1) * key_dim]
            .copy_from_slice(&row[key_dim..key_dim * 2]);
        v_full[t * value_dim..(t + 1) * value_dim].copy_from_slice(&row[key_dim * 2..]);
    }

    let mut z = vec![0.0f32; seq_len * value_dim];
    matmul(&w.in_proj_z, value_dim, hidden, &normed, seq_len, &mut z);

    let mut a = vec![0.0f32; seq_len * heads];
    let mut b = vec![0.0f32; seq_len * heads];
    matmul(&w.in_proj_a, heads, hidden, &normed, seq_len, &mut a);
    matmul(&w.in_proj_b, heads, hidden, &normed, seq_len, &mut b);

    let mut beta = vec![0.0f32; seq_len * heads];
    let mut decay = vec![0.0f32; seq_len * heads];
    for i in 0..seq_len * heads {
        beta[i] = sigmoid(b[i]);
        let h = i % heads;
        decay[i] = -(w.a_log[h].exp()) * softplus(a[i] + w.dt_bias[h]);
    }

    let mut q_rep = vec![0.0f32; seq_len * heads * k_head];
    let mut k_rep = vec![0.0f32; seq_len * heads * k_head];
    repeat_heads(&q_full, seq_len, config.linear_num_key_heads, k_head, ratio, &mut q_rep);
    repeat_heads(&k_full, seq_len, config.linear_num_key_heads, k_head, ratio, &mut k_rep);

    let mut core = vec![0.0f32; seq_len * heads * v_head];
    gated_delta_rule(
        &mut q_rep, &mut k_rep, &v_full, &decay, &beta, seq_len, heads, k_head, v_head, &mut core,
    );

    let mut gated = vec![0.0f32; seq_len * heads * v_head];
    for t in 0..seq_len {
        for h in 0..heads {
            let i = t * heads + h;
            rms_norm_gated(
                &core[i * v_head..(i + 1) * v_head],
                &z[t * value_dim + h * v_head..t * value_dim + (h + 1) * v_head],
                &w.norm_weight,
                config.rms_norm_eps,
                &mut gated[i * v_head..(i + 1) * v_head],
            );
        }
    }

    let mut mixed_out = vec![0.0f32; seq_len * hidden];
    matmul(&w.out_proj, hidden, value_dim, &gated, seq_len, &mut mixed_out);

    let mut y = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        y[i] = x[i] + mixed_out[i];
    }

    let mut mlp_in = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        rms_norm(
            &y[t * hidden..(t + 1) * hidden],
            &w.post_attention_layernorm,
            config.rms_norm_eps,
            &mut mlp_in[t * hidden..(t + 1) * hidden],
        );
    }

    let inter = config.intermediate_size;
    let mut gate = vec![0.0f32; seq_len * inter];
    let mut up = vec![0.0f32; seq_len * inter];
    matmul(&w.mlp_gate, inter, hidden, &mlp_in, seq_len, &mut gate);
    matmul(&w.mlp_up, inter, hidden, &mlp_in, seq_len, &mut up);

    let mut act = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        act[i] = crate::model::norm::silu(gate[i]) * up[i];
    }

    let mut mlp_out = vec![0.0f32; seq_len * hidden];
    matmul(&w.mlp_down, hidden, inter, &act, seq_len, &mut mlp_out);

    for i in 0..seq_len * hidden {
        out[i] = y[i] + mlp_out[i];
    }
    Ok(())
}

#[cfg(test)]
mod reference {
    use super::*;
    use crate::loader::Checkpoint;
    use crate::model::config::TextConfig;
    use crate::model::layer::{deltanet_layer, DeltaNetWeights};

    fn intermediates() -> Option<serde_json::Value> {
        let candidates = [
            "tests/eval/clef_layer0_intermediates.json",
            "../../tests/eval/clef_layer0_intermediates.json",
            "../tests/eval/clef_layer0_intermediates.json",
            "tests/eval/clef_layer0_intermediates.json",
        ];
        let path = candidates.iter().map(std::path::Path::new).find(|p| p.exists())?;
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    fn stage(name: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let mut worst = (0usize, 0.0f32);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        let (i, d) = worst;
        let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
        eprintln!(
            "  {name:<26} max|diff|={d:.3e}  at {i:<6} rel={:.2e}  (ref mean|x|={scale:.3e})",
            d / scale
        );
    }

    struct Fixture {
        input_ids: Vec<i64>,
        embed: Vec<f32>,
        layer0: Vec<f32>,
        tokens: usize,
        hidden: usize,
    }

    fn fixture() -> Option<Fixture> {
        let candidates = [
            "../../tests/eval/clef_hidden_fixture.json",
            "../tests/eval/clef_hidden_fixture.json",
            "tests/eval/clef_hidden_fixture.json",
            "tests/eval/clef_hidden_fixture.json",
        ];
        let path = candidates.iter().map(std::path::Path::new).find(|p| p.exists())?;
        let text = std::fs::read_to_string(path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&text).ok()?;

        let decode = |key: &str| -> Vec<f32> {
            use base64::Engine;
            let b64 = json["full"][key].as_str().expect("fixture key");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .expect("base64");
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        Some(Fixture {
            input_ids: json["input_ids"]
                .as_array()?
                .iter()
                .filter_map(|v| v.as_i64())
                .collect(),
            embed: decode("embed"),
            layer0: decode("layer0"),
            tokens: json["shape"]["tokens"].as_u64()? as usize,
            hidden: json["shape"]["hidden"].as_u64()? as usize,
        })
    }

    fn compare(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g - w).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / got.len() as f64) as f32;
        let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
        (max_abs, mean_abs, scale)
    }

    #[test]
    fn layer_zero_matches_the_reference() {
        let Some(f) = fixture() else {
            eprintln!("SKIP: clef_hidden_fixture.json not found");
            return;
        };

        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: /models/clef-flash not present; run in the aids-cuda container");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash").expect("config.json");

        assert_eq!(config.layer_kind(0), crate::model::LayerKind::LinearAttention);
        let weights = DeltaNetWeights::load(&checkpoint, &config, 0).expect("layer 0 weights");

        let mut got = vec![0.0f32; f.tokens * f.hidden];
        deltanet_layer(&config, &weights, &f.embed, f.tokens, &mut got).expect("forward");
        let (max_abs, mean_abs, scale) = compare(&got, &f.layer0);
        eprintln!("layer 0: max |diff| = {max_abs:.6e}, mean |diff| = {mean_abs:.6e}, reference mean |x| = {scale:.6e}");
        eprintln!("          relative to reference scale: {:.3e}", mean_abs / scale);

        assert!(
            max_abs < 1e-3,
            "layer 0 diverges: max |diff| {max_abs:.6e} against a reference scale of {scale:.6e}"
        );
        assert!(
            mean_abs < 1e-4,
            "layer 0 diverges on average: mean |diff| {mean_abs:.6e} against a reference scale of {scale:.6e}"
        );
    }

    #[test]
    fn the_fixture_input_is_the_embedding_of_the_tokens() {
        let Some(f) = fixture() else {
            eprintln!("SKIP: clef_hidden_fixture.json not found");
            return;
        };

        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: /models/clef-flash not present");
            return;
        };

        let Ok(name) = checkpoint.embed_tokens_name() else {
            eprintln!("SKIP: no embedding table in this checkpoint");
            return;
        };
        if !checkpoint.contains(&name) {
            eprintln!("SKIP: {name} not in this checkpoint");
            return;
        }
        let (table, shape) = checkpoint.tensor_f32(&name).expect("embed_tokens");
        assert_eq!(shape.len(), 2, "embed_tokens should be 2-D");
        let width = shape[1];
        assert_eq!(width, f.hidden);

        let mut got = vec![0.0f32; f.tokens * width];
        for (t, &id) in f.input_ids.iter().enumerate() {
            let row = id as usize * width;
            got[t * width..(t + 1) * width].copy_from_slice(&table[row..row + width]);
        }
        let (max_abs, mean_abs, scale) = compare(&got, &f.embed);
        eprintln!("embed: max |diff| = {max_abs:.6e}, mean |diff| = {mean_abs:.6e}, scale = {scale:.6e}");
        assert!(max_abs < 1e-5, "embedding lookup disagrees: max |diff| {max_abs:.6e}");
    }

    #[test]
    fn dump_intermediates() {
        let Some(f) = fixture() else { return };
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: no checkpoint");
            return;
        };
        let config = TextConfig::from_dir("/models/clef-flash").expect("config");
        let w = DeltaNetWeights::load(&checkpoint, &config, 0).expect("weights");

        let hidden = config.hidden_size;
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let k_head = config.linear_key_head_dim;
        let v_head = config.linear_value_head_dim;
        let heads = config.linear_num_value_heads;
        let ratio = config.value_heads_per_key_head();
        let conv_dim = config.conv_dim();
        let (s, n) = (f.tokens, f.tokens);

        let show = |name: &str, x: &[f32]| {
            let (mean, max) = stats(x);
            eprintln!("  {name:<28} mean={mean:.6} max={max:.6}");
        };

        let mut normed = vec![0.0f32; n * hidden];
        for t in 0..s {
            crate::model::norm::rms_norm(&f.embed[t*hidden..(t+1)*hidden], &w.input_layernorm,
                config.rms_norm_eps, &mut normed[t*hidden..(t+1)*hidden]);
        }
        show("input_layernorm", &normed);

        let mut mixed = vec![0.0f32; n * conv_dim];
        crate::model::matmul(&w.in_proj_qkv, conv_dim, hidden, &normed, s, &mut mixed);
        show("in_proj_qkv", &mixed);

        let mut convolved = vec![0.0f32; n * conv_dim];
        causal_conv1d_silu(&mixed, &w.conv_weight, config.linear_conv_kernel_dim, conv_dim, s, &mut convolved);
        show("conv_out", &convolved);

        let mut q_full = vec![0.0f32; n * key_dim];
        let mut k_full = vec![0.0f32; n * key_dim];
        let mut v_full = vec![0.0f32; n * value_dim];
        for t in 0..s {
            let row = &convolved[t*conv_dim..(t+1)*conv_dim];
            q_full[t*key_dim..(t+1)*key_dim].copy_from_slice(&row[..key_dim]);
            k_full[t*key_dim..(t+1)*key_dim].copy_from_slice(&row[key_dim..key_dim*2]);
            v_full[t*value_dim..(t+1)*value_dim].copy_from_slice(&row[key_dim*2..]);
        }
        show("split_q", &q_full); show("split_k", &k_full); show("split_v", &v_full);

        let mut z = vec![0.0f32; n * value_dim];
        crate::model::matmul(&w.in_proj_z, value_dim, hidden, &normed, s, &mut z);
        show("z", &z);
        let mut a = vec![0.0f32; n * heads];
        let mut b = vec![0.0f32; n * heads];
        crate::model::matmul(&w.in_proj_a, heads, hidden, &normed, s, &mut a);
        crate::model::matmul(&w.in_proj_b, heads, hidden, &normed, s, &mut b);
        show("a_raw", &a); show("b_raw", &b);

        let mut beta = vec![0.0f32; n * heads];
        let mut decay = vec![0.0f32; n * heads];
        for i in 0..n*heads {
            beta[i] = sigmoid(b[i]);
            decay[i] = -(w.a_log[i % heads].exp()) * softplus(a[i] + w.dt_bias[i % heads]);
        }
        show("beta", &beta); show("g", &decay);

        let mut q_rep = vec![0.0f32; n * heads * k_head];
        let mut k_rep = vec![0.0f32; n * heads * k_head];
        repeat_heads(&q_full, s, config.linear_num_key_heads, k_head, ratio, &mut q_rep);
        repeat_heads(&k_full, s, config.linear_num_key_heads, k_head, ratio, &mut k_rep);
        let mut core = vec![0.0f32; n * heads * v_head];
        gated_delta_rule(&mut q_rep, &mut k_rep, &v_full, &decay, &beta, s, heads, k_head, v_head, &mut core);
        show("core_attn_out", &core);

        let mut gated = vec![0.0f32; n * heads * v_head];
        for t in 0..s { for h in 0..heads {
            let i = t*heads + h;
            rms_norm_gated(&core[i*v_head..(i+1)*v_head], &z[t*value_dim+h*v_head..t*value_dim+(h+1)*v_head],
                &w.norm_weight, config.rms_norm_eps, &mut gated[i*v_head..(i+1)*v_head]);
        }}
        show("gated_norm", &gated);

        let mut mo = vec![0.0f32; n * hidden];
        crate::model::matmul(&w.out_proj, hidden, value_dim, &gated, s, &mut mo);
        show("out_proj", &mo);
        let mut y = vec![0.0f32; n*hidden];
        for i in 0..n*hidden { y[i] = f.embed[i] + mo[i]; }
        show("residual1", &y);

        let mut mlp_in = vec![0.0f32; n*hidden];
        for t in 0..s {
            crate::model::norm::rms_norm(&y[t*hidden..(t+1)*hidden], &w.post_attention_layernorm,
                config.rms_norm_eps, &mut mlp_in[t*hidden..(t+1)*hidden]);
        }
        show("post_attention_layernorm", &mlp_in);

        let inter = config.intermediate_size;
        let mut gate = vec![0.0f32; n*inter];
        let mut up = vec![0.0f32; n*inter];
        crate::model::matmul(&w.mlp_gate, inter, hidden, &mlp_in, s, &mut gate);
        crate::model::matmul(&w.mlp_up, inter, hidden, &mlp_in, s, &mut up);
        let mut act = vec![0.0f32; n*inter];
        for i in 0..n*inter { act[i] = crate::model::norm::silu(gate[i]) * up[i]; }
        let mut mout = vec![0.0f32; n*hidden];
        crate::model::matmul(&w.mlp_down, hidden, inter, &act, s, &mut mout);
        show("mlp_out", &mout);
        let mut fin = vec![0.0f32; n*hidden];
        for i in 0..n*hidden { fin[i] = y[i] + mout[i]; }
        show("final", &fin);

        let Some(refs) = intermediates() else {
            eprintln!("  (no reference intermediates; skipping the stage comparison)");
            return;
        };
        let want = |key: &str| -> Vec<f32> {
            use base64::Engine;
            let b64 = refs[key]["b64"].as_str().expect("b64");
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .expect("decode")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        eprintln!("  --- element-wise vs the reference capture ---");
        stage("input_layernorm", &normed, &want("input_layernorm"));
        stage("in_proj_qkv", &mixed, &want("in_proj_qkv"));
        stage("conv_out", &convolved, &want("conv_out"));
        stage("split_q", &q_full, &want("split_q"));
        stage("split_k", &k_full, &want("split_k"));
        stage("split_v", &v_full, &want("split_v"));
        stage("z", &z, &want("z"));
        stage("a_raw", &a, &want("a_raw"));
        stage("b_raw", &b, &want("b_raw"));
        stage("beta", &beta, &want("beta"));
        stage("g", &decay, &want("g"));
        stage("core_attn_out", &core, &want("core_attn_out"));
        stage("gated_norm", &gated, &want("gated_norm"));
        stage("out_proj", &mo, &want("out_proj"));
        stage("residual1", &y, &want("residual1"));
        stage("post_attention_layernorm", &mlp_in, &want("post_attention_layernorm"));
        stage("mlp_out", &mout, &want("mlp_out"));
        stage("final", &fin, &want("final"));
    }
}
