use crate::loader::Checkpoint;
use crate::placement::HostTable;
use anyhow::{Context, Result};

#[inline]
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n as f64;
    let inv = (1.0 / (var + eps as f64).sqrt()) as f32;
    let m = mean as f32;
    for i in 0..n {
        out[i] = (x[i] - m) * inv * weight[i] + bias[i];
    }
}

fn linear(x: &[f32], w: &[f32], bias: Option<&[f32]>, seq: usize, in_dim: usize, out_dim: usize)
    -> Vec<f32> {
    let mut out = vec![0.0f32; seq * out_dim];
    for t in 0..seq {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for (a, b) in row.iter().zip(xt.iter()) {
                acc += a * b;
            }
            out[t * out_dim + o] = acc + bias.map_or(0.0, |b| b[o]);
        }
    }
    out
}

#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn multi_head_attention(
    query: &[f32],
    key_value: &[f32],
    in_proj_weight: &[f32],
    in_proj_bias: &[f32],
    out_proj_weight: &[f32],
    out_proj_bias: &[f32],
    q_len: usize,
    kv_len: usize,
    width: usize,
    heads: usize,
) -> Vec<f32> {
    let hd = width / heads;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let q = linear(query, &in_proj_weight[..width * width], Some(&in_proj_bias[..width]),
                   q_len, width, width);
    let k = linear(key_value, &in_proj_weight[width * width..2 * width * width],
                   Some(&in_proj_bias[width..2 * width]), kv_len, width, width);
    let v = linear(key_value, &in_proj_weight[2 * width * width..],
                   Some(&in_proj_bias[2 * width..]), kv_len, width, width);

    let mut ctx = vec![0.0f32; q_len * width];
    let mut scores = vec![0.0f32; kv_len];
    for t in 0..q_len {
        for h in 0..heads {
            let q_t = &q[t * width + h * hd..t * width + (h + 1) * hd];
            let mut max = f32::NEG_INFINITY;
            for s in 0..kv_len {
                let k_s = &k[s * width + h * hd..s * width + (h + 1) * hd];
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += q_t[d] * k_s[d];
                }
                scores[s] = acc * scale;
                max = max.max(scores[s]);
            }
            let mut sum = 0.0f32;
            for s in 0..kv_len {
                scores[s] = (scores[s] - max).exp();
                sum += scores[s];
            }
            let o = &mut ctx[t * width + h * hd..t * width + (h + 1) * hd];
            for s in 0..kv_len {
                let p = scores[s] / sum;
                let v_s = &v[s * width + h * hd..s * width + (h + 1) * hd];
                for d in 0..hd {
                    o[d] += p * v_s[d];
                }
            }
        }
    }
    linear(&ctx, out_proj_weight, Some(out_proj_bias), q_len, width, width)
}

fn evidence_routing_layer(
    queries: &[f32],
    memory: &[f32],
    w: &EvidenceWeights,
    q_len: usize,
    mem_len: usize,
    width: usize,
    heads: usize,
    ff: usize,
    eps: f32,
) -> Vec<f32> {
    let norm_q: Vec<f32> = (0..q_len)
        .flat_map(|t| {
            let mut o = vec![0.0f32; width];
            layer_norm(&queries[t * width..(t + 1) * width], &w.query_norm_w, &w.query_norm_b, eps, &mut o);
            o
        })
        .collect();
    let norm_m: Vec<f32> = (0..mem_len)
        .flat_map(|t| {
            let mut o = vec![0.0f32; width];
            layer_norm(&memory[t * width..(t + 1) * width], &w.memory_norm_w, &w.memory_norm_b, eps, &mut o);
            o
        })
        .collect();

    let routed = multi_head_attention(
        &norm_q, &norm_m, &w.in_proj_weight, &w.in_proj_bias, &w.out_proj_weight,
        &w.out_proj_bias, q_len, mem_len, width, heads,
    );
    let mut x: Vec<f32> = queries.iter().zip(routed.iter()).map(|(a, b)| a + b).collect();

    let norm_f: Vec<f32> = (0..q_len)
        .flat_map(|t| {
            let mut o = vec![0.0f32; width];
            layer_norm(&x[t * width..(t + 1) * width], &w.ff_norm_w, &w.ff_norm_b, eps, &mut o);
            o
        })
        .collect();
    let h = linear(&norm_f, &w.ff1_w, Some(&w.ff1_b), q_len, width, ff);
    let h: Vec<f32> = h.iter().map(|v| gelu(*v)).collect();
    let h = linear(&h, &w.ff2_w, Some(&w.ff2_b), q_len, ff, width);
    for i in 0..x.len() {
        x[i] += h[i];
    }
    x
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer(
    x: &[f32],
    memory: &[f32],
    w: &DecoderWeights,
    seq: usize,
    mem_len: usize,
    width: usize,
    heads: usize,
    ff: usize,
    eps: f32,
) -> Vec<f32> {
    let norm = |src: &[f32], len: usize, gw: &[f32], gb: &[f32]| -> Vec<f32> {
        (0..len)
            .flat_map(|t| {
                let mut o = vec![0.0f32; width];
                layer_norm(&src[t * width..(t + 1) * width], gw, gb, eps, &mut o);
                o
            })
            .collect()
    };

    let n1 = norm(x, seq, &w.norm1_w, &w.norm1_b);
    let a = multi_head_attention(&n1, &n1, &w.self_in_w, &w.self_in_b, &w.self_out_w,
                                 &w.self_out_b, seq, seq, width, heads);
    let mut y: Vec<f32> = x.iter().zip(a.iter()).map(|(p, q)| p + q).collect();

    let n2 = norm(&y, seq, &w.norm2_w, &w.norm2_b);
    let a = multi_head_attention(&n2, memory, &w.cross_in_w, &w.cross_in_b, &w.cross_out_w,
                                 &w.cross_out_b, seq, mem_len, width, heads);
    for i in 0..y.len() {
        y[i] += a[i];
    }

    let n3 = norm(&y, seq, &w.norm3_w, &w.norm3_b);
    let h = linear(&n3, &w.ff1_w, Some(&w.ff1_b), seq, width, ff);
    let h: Vec<f32> = h.iter().map(|v| gelu(*v)).collect();
    let h = linear(&h, &w.ff2_w, Some(&w.ff2_b), seq, ff, width);
    for i in 0..y.len() {
        y[i] += h[i];
    }
    y
}

pub struct EvidenceWeights {
    pub query_norm_w: Vec<f32>, pub query_norm_b: Vec<f32>,
    pub memory_norm_w: Vec<f32>, pub memory_norm_b: Vec<f32>,
    pub in_proj_weight: Vec<f32>, pub in_proj_bias: Vec<f32>,
    pub out_proj_weight: Vec<f32>, pub out_proj_bias: Vec<f32>,
    pub ff_norm_w: Vec<f32>, pub ff_norm_b: Vec<f32>,
    pub ff1_w: Vec<f32>, pub ff1_b: Vec<f32>,
    pub ff2_w: Vec<f32>, pub ff2_b: Vec<f32>,
}

pub struct DecoderWeights {
    pub norm1_w: Vec<f32>, pub norm1_b: Vec<f32>,
    pub self_in_w: Vec<f32>, pub self_in_b: Vec<f32>,
    pub self_out_w: Vec<f32>, pub self_out_b: Vec<f32>,
    pub norm2_w: Vec<f32>, pub norm2_b: Vec<f32>,
    pub cross_in_w: Vec<f32>, pub cross_in_b: Vec<f32>,
    pub cross_out_w: Vec<f32>, pub cross_out_b: Vec<f32>,
    pub norm3_w: Vec<f32>, pub norm3_b: Vec<f32>,
    pub ff1_w: Vec<f32>, pub ff1_b: Vec<f32>,
    pub ff2_w: Vec<f32>, pub ff2_b: Vec<f32>,
}

impl HeadWeights {
    pub fn load(checkpoint: &Checkpoint, config: &HeadConfig) -> Result<Self> {
        let v = |name: &str, len: usize| -> Result<Vec<f32>> {
            let (values, shape) = checkpoint.tensor_f32(name)?;
            if values.len() != len {
                anyhow::bail!("{name} has {} values ({shape:?}), expected {len}", values.len());
            }
            Ok(values)
        };
        let scalar = |name: &str| -> Result<f32> {
            let (values, _) = checkpoint.tensor_f32(name)?;
            values.first().copied().context("empty scalar")
        };

        let width = config.width;
        let hidden = config.hidden_size;
        let ff = config.feedforward;

        let split_in = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            let w = v(&format!("{name}.in_proj_weight"), 3 * width * width)?;
            let b = v(&format!("{name}.in_proj_bias"), 3 * width)?;
            Ok((w, b))
        };

        let mut evidence = Vec::new();
        for i in 0..config.routing_layers {
            let p = format!("evidence_layers.{i}");
            let (in_proj_weight, in_proj_bias) = split_in(&format!("{p}.attention"))?;
            evidence.push(EvidenceWeights {
                query_norm_w: v(&format!("{p}.query_norm.weight"), width)?,
                query_norm_b: v(&format!("{p}.query_norm.bias"), width)?,
                memory_norm_w: v(&format!("{p}.memory_norm.weight"), width)?,
                memory_norm_b: v(&format!("{p}.memory_norm.bias"), width)?,
                in_proj_weight,
                in_proj_bias,
                out_proj_weight: v(&format!("{p}.attention.out_proj.weight"), width * width)?,
                out_proj_bias: v(&format!("{p}.attention.out_proj.bias"), width)?,
                ff_norm_w: v(&format!("{p}.feedforward_norm.weight"), width)?,
                ff_norm_b: v(&format!("{p}.feedforward_norm.bias"), width)?,
                ff1_w: v(&format!("{p}.feedforward.0.weight"), ff * width)?,
                ff1_b: v(&format!("{p}.feedforward.0.bias"), ff)?,
                ff2_w: v(&format!("{p}.feedforward.3.weight"), width * ff)?,
                ff2_b: v(&format!("{p}.feedforward.3.bias"), width)?,
            });
        }

        let mut decoder = Vec::new();
        for i in 0..config.layers {
            let p = format!("layers.{i}");
            let (self_in_w, self_in_b) = split_in(&format!("{p}.self_attn"))?;
            let (cross_in_w, cross_in_b) = split_in(&format!("{p}.multihead_attn"))?;
            decoder.push(DecoderWeights {
                norm1_w: v(&format!("{p}.norm1.weight"), width)?,
                norm1_b: v(&format!("{p}.norm1.bias"), width)?,
                self_in_w, self_in_b,
                self_out_w: v(&format!("{p}.self_attn.out_proj.weight"), width * width)?,
                self_out_b: v(&format!("{p}.self_attn.out_proj.bias"), width)?,
                norm2_w: v(&format!("{p}.norm2.weight"), width)?,
                norm2_b: v(&format!("{p}.norm2.bias"), width)?,
                cross_in_w, cross_in_b,
                cross_out_w: v(&format!("{p}.multihead_attn.out_proj.weight"), width * width)?,
                cross_out_b: v(&format!("{p}.multihead_attn.out_proj.bias"), width)?,
                norm3_w: v(&format!("{p}.norm3.weight"), width)?,
                norm3_b: v(&format!("{p}.norm3.bias"), width)?,
                ff1_w: v(&format!("{p}.linear1.weight"), ff * width)?,
                ff1_b: v(&format!("{p}.linear1.bias"), ff)?,
                ff2_w: v(&format!("{p}.linear2.weight"), width * ff)?,
                ff2_b: v(&format!("{p}.linear2.bias"), width)?,
            });
        }
        Ok(Self {
            hidden_norm_w: v("hidden_norm.weight", hidden)?,
            hidden_norm_b: v("hidden_norm.bias", hidden)?,
            memory_projection: v("memory_projection.weight", width * hidden)?,
            question_projection: v("question_projection.weight", width * hidden)?,
            option_question_projection: v("option_question_projection.weight", width * hidden)?,
            global_projection: v("global_projection.weight", width * hidden)?,
            option_context_projection: v("option_context_projection.weight", width * hidden)?,
            option_lexical_projection: v("option_lexical_projection.weight", width * hidden)?,
            type_embedding: v("type_embedding.weight", 3 * width)?,
            evidence,
            option_summary_norm_w: v("option_summary_norm.weight", width)?,
            option_summary_norm_b: v("option_summary_norm.bias", width)?,
            decoder,
            field_norm_w: v("field_norm.weight", width)?,
            field_norm_b: v("field_norm.bias", width)?,
            option_norm_w: v("option_norm.weight", width)?,
            option_norm_b: v("option_norm.bias", width)?,
            residual_w0: v("residual_scorer.0.weight", width * width * 4)?,
            residual_b0: v("residual_scorer.0.bias", width)?,
            residual_w3: v("residual_scorer.3.weight", width)?,
            residual_b3: v("residual_scorer.3.bias", 1)?,
            prior_logit_scale: scalar("prior_logit_scale")?,
            joint_logit_scale: scalar("joint_logit_scale")?,
            residual_gate: scalar("residual_gate")?,
        })
    }
}

pub struct QuestionSpec {
    pub question_type: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
}

pub struct HeadWeights {
    pub hidden_norm_w: Vec<f32>, pub hidden_norm_b: Vec<f32>,
    pub memory_projection: Vec<f32>,
    pub question_projection: Vec<f32>,
    pub option_question_projection: Vec<f32>,
    pub global_projection: Vec<f32>,
    pub option_context_projection: Vec<f32>,
    pub option_lexical_projection: Vec<f32>,
    pub type_embedding: Vec<f32>,
    pub evidence: Vec<EvidenceWeights>,
    pub option_summary_norm_w: Vec<f32>, pub option_summary_norm_b: Vec<f32>,
    pub decoder: Vec<DecoderWeights>,
    pub field_norm_w: Vec<f32>, pub field_norm_b: Vec<f32>,
    pub option_norm_w: Vec<f32>, pub option_norm_b: Vec<f32>,
    pub residual_w0: Vec<f32>, pub residual_b0: Vec<f32>,
    pub residual_w3: Vec<f32>, pub residual_b3: Vec<f32>,
    pub prior_logit_scale: f32,
    pub joint_logit_scale: f32,
    pub residual_gate: f32,
}

pub struct HeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

impl HeadConfig {
    pub fn from_dir(dir: &std::path::Path) -> Result<Self> {
        let path = dir.join("joint_head_config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        let get = |k: &str| -> Result<usize> {
            v[k].as_u64()
                .map(|n| n as usize)
                .with_context(|| format!("{} has no integer `{k}`", path.display()))
        };
        Ok(Self {
            hidden_size: get("hidden_size")?,
            width: get("width")?,
            routing_layers: get("routing_layers")?,
            layers: get("layers")?,
            heads: get("heads")?,
            feedforward: get("feedforward")?,
        })
    }
}

pub fn question_type_id(kind: crate::types::QuestionType) -> usize {
    match kind {
        crate::types::QuestionType::Noul => 0,
        crate::types::QuestionType::Choice => 1,
        crate::types::QuestionType::Score => 2,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn head_forward(
    hidden: &[f32],
    tokens: usize,
    input_ids: &[u32],
    questions: &[QuestionSpec],
    weights: &HeadWeights,
    config: &HeadConfig,
    lm_head: &HostTable,
    eps: f32,
) -> Result<Vec<Vec<f32>>> {
    let width = config.width;
    let hidden_size = config.hidden_size;

    let normalised: Vec<f32> = (0..tokens)
        .flat_map(|t| {
            let mut o = vec![0.0f32; hidden_size];
            layer_norm(
                &hidden[t * hidden_size..(t + 1) * hidden_size],
                &weights.hidden_norm_w, &weights.hidden_norm_b, eps, &mut o,
            );
            o
        })
        .collect();

    let memory = linear(&normalised, &weights.memory_projection, None, tokens, hidden_size, width);
    let global_vector = &normalised[(tokens - 1) * hidden_size..tokens * hidden_size];

    let mean_span = |span: (usize, usize)| -> Vec<f32> {
        let (s, e) = span;
        let mut out = vec![0.0f32; hidden_size];
        for t in s..e {
            for d in 0..hidden_size {
                out[d] += normalised[t * hidden_size + d];
            }
        }
        let n = (e - s) as f32;
        for v in out.iter_mut() {
            *v /= n;
        }
        out
    };

    let question_vectors: Vec<Vec<f32>> = questions.iter().map(|q| mean_span(q.question_span)).collect();

    let mut option_queries: Vec<f32> = Vec::new();
    let mut lexical_all: Vec<Vec<f32>> = Vec::new();
    for (qi, q) in questions.iter().enumerate() {
        for span in &q.option_spans {
            let ctx = mean_span(*span);

            let ids: Vec<u32> = input_ids[span.0..span.1].to_vec();

            let mut rows = vec![0.0f32; ids.len() * hidden_size];
            lm_head.gather_f32(&ids, &mut rows);
            let mut lex = vec![0.0f32; hidden_size];
            for r in 0..ids.len() {
                for d in 0..hidden_size {
                    lex[d] += rows[r * hidden_size + d];
                }
            }
            for v in lex.iter_mut() {
                *v /= ids.len() as f32;
            }
            lexical_all.push(lex.clone());

            let a = linear(&ctx, &weights.option_context_projection, None, 1, hidden_size, width);
            let b = linear(&lex, &weights.option_lexical_projection, None, 1, hidden_size, width);
            let c = linear(&question_vectors[qi], &weights.option_question_projection, None, 1,
                           hidden_size, width);
            for d in 0..width {
                option_queries.push(a[d] + b[d] + c[d]);
            }
        }
    }
    let total_options = lexical_all.len();

    let mut routed = option_queries;
    for layer in &weights.evidence {
        routed = evidence_routing_layer(&routed, &memory, layer, total_options, tokens, width,
                                        config.heads, config.feedforward, eps);
    }

    let base_fields = linear(
        &question_vectors.concat(), &weights.question_projection, None, questions.len(),
        hidden_size, width,
    );
    let mut summaries: Vec<f32> = Vec::new();
    let mut offset = 0usize;
    let mut fields: Vec<f32> = Vec::new();
    let global_proj = linear(global_vector, &weights.global_projection, None, 1, hidden_size, width);
    for (qi, q) in questions.iter().enumerate() {
        let count = q.option_spans.len();
        let field = &base_fields[qi * width..(qi + 1) * width];

        let mut weights_opt = vec![0.0f32; count];
        let mut max = f32::NEG_INFINITY;
        for j in 0..count {
            let row = &routed[(offset + j) * width..(offset + j + 1) * width];
            let mut acc = 0.0f32;
            for d in 0..width {
                acc += row[d] * field[d];
            }
            weights_opt[j] = acc / (width as f32).sqrt();
            max = max.max(weights_opt[j]);
        }
        let mut sum = 0.0f32;
        for w in weights_opt.iter_mut() {
            *w = (*w - max).exp();
            sum += *w;
        }
        let mut summary = vec![0.0f32; width];
        for j in 0..count {
            let p = weights_opt[j] / sum;
            let row = &routed[(offset + j) * width..(offset + j + 1) * width];
            for d in 0..width {
                summary[d] += p * row[d];
            }
        }
        summaries.extend_from_slice(&summary);
        offset += count;
    }

    let mut summaries_normed = vec![0.0f32; questions.len() * width];
    for i in 0..questions.len() {
        layer_norm(
            &summaries[i * width..(i + 1) * width],
            &weights.option_summary_norm_w, &weights.option_summary_norm_b, eps,
            &mut summaries_normed[i * width..(i + 1) * width],
        );
    }
    for qi in 0..questions.len() {
        for d in 0..width {
            fields.push(
                base_fields[qi * width + d]
                    + summaries_normed[qi * width + d]
                    + global_proj[d]
                    + weights.type_embedding[questions[qi].question_type * width + d],
            );
        }
    }

    let mut fields = fields;
    for layer in &weights.decoder {
        fields = decoder_layer(&fields, &memory, layer, questions.len(), tokens, width,
                               config.heads, config.feedforward, eps);
    }
    let mut fields_normed = vec![0.0f32; questions.len() * width];
    for i in 0..questions.len() {
        layer_norm(&fields[i * width..(i + 1) * width], &weights.field_norm_w,
                   &weights.field_norm_b, eps, &mut fields_normed[i * width..(i + 1) * width]);
    }

    let prior_scale = weights.prior_logit_scale.min(100.0f32.ln()).exp();
    let joint_scale = weights.joint_logit_scale.min(100.0f32.ln()).exp();
    let gate = 1.0 / (1.0 + (-weights.residual_gate).exp());

    let mut results = Vec::new();
    let mut offset = 0usize;
    let mut trace_terms: Vec<(f32, f32, f32)> = Vec::new();
    for (qi, q) in questions.iter().enumerate() {
        let count = q.option_spans.len();
        let field = &fields_normed[qi * width..(qi + 1) * width];

        let mut anchor = vec![0.0f32; hidden_size];
        for d in 0..hidden_size {
            anchor[d] = question_vectors[qi][d] + global_vector[d];
        }
        normalise(&mut anchor);

        let mut logits = Vec::with_capacity(count);
        for j in 0..count {
            let lex = &lexical_all[offset + j];
            let mut lexical_anchor = lex.clone();
            normalise(&mut lexical_anchor);
            let mut prior = 0.0f32;
            for d in 0..hidden_size {
                prior += lexical_anchor[d] * anchor[d];
            }
            prior *= prior_scale;

            let mut option = vec![0.0f32; width];
            layer_norm(&routed[(offset + j) * width..(offset + j + 1) * width],
                       &weights.option_norm_w, &weights.option_norm_b, eps, &mut option);

            let mut a = field.to_vec();
            let mut b = option.clone();
            normalise(&mut a);
            normalise(&mut b);
            let mut cosine = 0.0f32;
            for d in 0..width {
                cosine += a[d] * b[d];
            }

            let mut features = Vec::with_capacity(width * 4);
            features.extend_from_slice(field);
            features.extend_from_slice(&option);
            for d in 0..width {
                features.push(field[d] * option[d]);
            }
            for d in 0..width {
                features.push((field[d] - option[d]).abs());
            }
            let h = linear(&features, &weights.residual_w0, Some(&weights.residual_b0), 1,
                           width * 4, width);
            let h: Vec<f32> = h.iter().map(|v| gelu(*v)).collect();
            let residual = linear(&h, &weights.residual_w3, Some(&weights.residual_b3), 1, width, 1);

            let joint = joint_scale * cosine + residual[0];
            logits.push(prior + gate * joint);
            trace_terms.push((prior, cosine, residual[0]));
        }
        results.push(logits);
        offset += count;
    }

    if std::env::var("CLEF_HEAD_TRACE").is_ok() {
        eprintln!("    cpu prior values      {:?}",
                  trace_terms.iter().map(|t| t.0).collect::<Vec<_>>());
        eprintln!("    cpu cosine values     {:?}",
                  trace_terms.iter().map(|t| t.1).collect::<Vec<_>>());
        eprintln!("    cpu residual values   {:?}",
                  trace_terms.iter().map(|t| t.2).collect::<Vec<_>>());
    }
    Ok(results)
}

fn normalise(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
pub(crate) mod support {
    use super::*;
    use crate::placement::HostTable;
    use serde_json::Value;

    pub fn lcg(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect()
    }

    pub fn load() -> Option<Value> {
        let candidates = [
            "tests/eval/clef_head_fixture.json",
            "../../tests/eval/clef_head_fixture.json",
            "../tests/eval/clef_head_fixture.json",
            "tests/eval/clef_head_fixture.json",
        ];
        let path = candidates.iter().map(std::path::Path::new).find(|p| p.exists())?;
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    pub fn head_config(f: &Value) -> HeadConfig {
        HeadConfig {
            hidden_size: f["config"]["hidden_size"].as_u64().unwrap() as usize,
            width: f["config"]["width"].as_u64().unwrap() as usize,
            routing_layers: f["config"]["routing_layers"].as_u64().unwrap() as usize,
            layers: f["config"]["layers"].as_u64().unwrap() as usize,
            heads: f["config"]["heads"].as_u64().unwrap() as usize,
            feedforward: f["config"]["feedforward"].as_u64().unwrap() as usize,
        }
    }

    pub fn input_ids_from(f: &Value) -> Vec<u32> {
        f["input_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    }

    pub fn questions_from(f: &Value) -> Vec<QuestionSpec> {
        f["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| QuestionSpec {
                question_type: q["question_type"].as_u64().unwrap() as usize,
                question_span: (
                    q["question_span"][0].as_u64().unwrap() as usize,
                    q["question_span"][1].as_u64().unwrap() as usize,
                ),
                option_spans: q["option_spans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize))
                    .collect(),
            })
            .collect()
    }

    pub fn lm_head_from(f: &Value, tokens: usize, hidden_size: usize, input_ids: &[u32])
        -> HostTable {
        let max_id = *input_ids.iter().max().unwrap() as usize;
        let rows = max_id + 1;
        let lex = lcg(f["seed_lexical"].as_u64().unwrap() as u32, tokens * hidden_size);
        let mut table = vec![0u16; rows * hidden_size];
        for (i, &id) in input_ids.iter().enumerate() {
            for d in 0..hidden_size {
                let v = lex[i * hidden_size + d];

                let bits = v.to_bits();
                let lsb = (bits >> 16) & 1;
                let rounded = bits.wrapping_add(0x7FFF + lsb);
                table[id as usize * hidden_size + d] = (rounded >> 16) as u16;
            }
        }
        let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = std::env::temp_dir().join("aiis_head_lm_head.safetensors");
        let mut header = format!(
            "{{\"lm_head.weight\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{hidden_size}],\
             \"data_offsets\":[0,{}]}}}}",
            bytes.len()
        );
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut file_bytes = (header.len() as u64).to_le_bytes().to_vec();
        file_bytes.extend_from_slice(header.as_bytes());
        file_bytes.extend_from_slice(&bytes);
        std::fs::write(&path, &file_bytes).expect("write table");
        HostTable::map(&path, "lm_head.weight").expect("host table")
    }
}

#[cfg(test)]
mod tests {
    use super::support;
    use super::*;

    #[test]
    fn the_head_matches_the_reference() {
        let Some(f) = support::load() else {
            eprintln!("SKIP: clef_head_fixture.json not found");
            return;
        };
        let Ok(head_ckpt) = Checkpoint::open_single("/models/clef-flash/joint_head.safetensors")
        else {
            eprintln!("SKIP: joint_head.safetensors not present; run where the checkpoint is mounted");
            return;
        };
        let Ok(model_ckpt) = Checkpoint::open("/models/clef-flash") else {
            eprintln!("SKIP: /models/clef-flash not present");
            return;
        };

        let config = support::head_config(&f);
        let tokens = f["tokens"].as_u64().unwrap() as usize;
        let hidden_size = config.hidden_size;

        let weights = HeadWeights::load(&head_ckpt, &config).expect("head weights");
        let hidden = support::lcg(f["seed_hidden"].as_u64().unwrap() as u32, tokens * hidden_size);
        let input_ids = support::input_ids_from(&f);

        let lm_head = support::lm_head_from(&f, tokens, hidden_size, &input_ids);
        let questions = support::questions_from(&f);

        let got = head_forward(
            &hidden, tokens, &input_ids, &questions, &weights, &config, &lm_head, 1e-5,
        )
        .expect("head forward");

        let want: Vec<Vec<f32>> = f["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect();

        let mut worst = 0.0f32;
        for (qi, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            for (oi, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                eprintln!("    q{qi} option {oi}: got {a:+.6}  want {b:+.6}  |diff| {:.3e}", (a - b).abs());
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "head logits disagree by {worst:.3e}");
    }
}
