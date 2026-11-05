use anyhow::{bail, Context, Result};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
    pub full_attention_interval: usize,
    pub attn_output_gate: bool,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
}

impl TextConfig {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("no config.json at {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_str(text).context("config.json is not valid JSON")?;
        let root = raw.get("text_config").unwrap_or(&raw);

        let get = |key: &str| -> Result<&serde_json::Value> {
            root.get(key)
                .with_context(|| format!("config.json has no `{key}`"))
        };
        let usize_of = |key: &str| -> Result<usize> {
            get(key)?
                .as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("`{key}` is not a non-negative integer"))
        };
        let f32_of = |key: &str| -> Result<f32> {
            get(key)?
                .as_f64()
                .map(|v| v as f32)
                .with_context(|| format!("`{key}` is not a number"))
        };

        let rope = root.get("rope_parameters").unwrap_or(root);
        let rope_theta = rope
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1_000_000.0);
        let partial_rotary_factor = rope
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0);
        let mrope_interleaved = rope
            .get("mrope_interleaved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let mrope_section = rope
            .get("mrope_section")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
            .unwrap_or_default();

        let config = TextConfig {
            hidden_size: usize_of("hidden_size")?,
            num_hidden_layers: usize_of("num_hidden_layers")?,
            intermediate_size: usize_of("intermediate_size")?,
            vocab_size: usize_of("vocab_size")?,
            num_attention_heads: usize_of("num_attention_heads")?,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim: usize_of("head_dim")?,
            linear_num_key_heads: usize_of("linear_num_key_heads")?,
            linear_num_value_heads: usize_of("linear_num_value_heads")?,
            linear_key_head_dim: usize_of("linear_key_head_dim")?,
            linear_value_head_dim: usize_of("linear_value_head_dim")?,
            linear_conv_kernel_dim: usize_of("linear_conv_kernel_dim")?,
            rms_norm_eps: f32_of("rms_norm_eps")?,
            rope_theta,
            partial_rotary_factor,
            mrope_section,
            mrope_interleaved,
            full_attention_interval: usize_of("full_attention_interval")?,
            attn_output_gate: root
                .get("attn_output_gate")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            hidden_act: get("hidden_act")?
                .as_str()
                .context("`hidden_act` is not a string")?
                .to_string(),
        };

        if config.hidden_act != "silu" {
            bail!("hidden_act is `{}`, and only `silu` is implemented", config.hidden_act);
        }
        if config.tie_word_embeddings {
            bail!("tie_word_embeddings is true, and the untied path is the only one implemented");
        }
        if config.linear_num_value_heads % config.linear_num_key_heads != 0 {
            bail!(
                "{} value heads do not divide by {} key heads",
                config.linear_num_value_heads,
                config.linear_num_key_heads
            );
        }
        if config.partial_rotary_factor <= 0.0 || config.partial_rotary_factor > 1.0 {
            bail!("partial_rotary_factor {} is outside (0, 1]", config.partial_rotary_factor);
        }
        Ok(config)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as usize
    }

    pub fn value_heads_per_key_head(&self) -> usize {
        self.linear_num_value_heads / self.linear_num_key_heads
    }

    pub fn conv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn layer_kind(&self, index: usize) -> LayerKind {
        if (index + 1) % self.full_attention_interval == 0 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        }
    }

    pub fn layer_prefix(&self, text_prefix: &str, index: usize) -> String {
        format!("{text_prefix}.layers.{index}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_config() -> Option<TextConfig> {
        match TextConfig::from_dir("/models/clef-flash-nvfp4") {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("SKIP: no checkpoint config ({e}); run in the aids-cuda container");
                None
            }
        }
    }

    #[test]
    fn layer_kinds_follow_the_interval() {
        let Some(config) = real_config() else { return };
        let full: Vec<usize> = (0..config.num_hidden_layers)
            .filter(|&i| config.layer_kind(i) == LayerKind::FullAttention)
            .collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31]);
        assert_eq!(config.num_hidden_layers, 32);
    }

    #[test]
    fn the_geometry_matches_the_checkpoint() {
        let Some(config) = real_config() else { return };
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.value_heads_per_key_head(), 2);

        assert_eq!(config.conv_dim(), 8192);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.mrope_section, vec![11, 11, 10]);
        assert!(config.attn_output_gate);
    }

    #[test]
    fn unsupported_configurations_are_refused() {
        let base = r#"{"hidden_size":4096,"num_hidden_layers":32,"intermediate_size":12288,
            "vocab_size":248320,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,
            "linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,
            "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,
            "full_attention_interval":4,"hidden_act":"silu","partial_rotary_factor":0.25}"#;
        assert!(TextConfig::from_json(base).is_ok());
        for (change, why) in [
            (r#""hidden_act":"silu""#, r#""hidden_act":"gelu""#),
            (r#""linear_num_value_heads":32"#, r#""linear_num_value_heads":33"#),
            (r#""partial_rotary_factor":0.25"#, r#""partial_rotary_factor":1.5"#),
        ] {
            let broken = base.replace(change, why);
            assert!(TextConfig::from_json(&broken).is_err(), "should refuse: {why}");
        }
    }
}
