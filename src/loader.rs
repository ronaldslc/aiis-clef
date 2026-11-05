use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

use crate::checkpoint::{storage_of, Storage};
use crate::dequant;

#[derive(Debug, Clone)]
pub enum Weights {
    Nvfp4 { packed: Vec<u8>, scales: Vec<u8>, global_scale: f32 },

    Fp8 { values: Vec<u8>, scales: Vec<u8> },

    Bf16 { values: Vec<u8> },
}

fn e4m3_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| f8_to_f32(&[b])[0]).collect()
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Projection {
    pub weights: Weights,
    pub rows: usize,
    pub cols: usize,
    pub input_global_scale: Option<f32>,
}

impl Projection {
    pub fn storage(&self) -> Storage {
        match self.weights {
            Weights::Nvfp4 { .. } => Storage::Nvfp4,
            Weights::Fp8 { .. } => Storage::Fp8,
            Weights::Bf16 { .. } => Storage::Bf16,
        }
    }

    pub fn bytes(&self) -> usize {
        match &self.weights {
            Weights::Nvfp4 { packed, scales, .. } => packed.len() + scales.len(),
            Weights::Fp8 { values, scales } => values.len() + scales.len(),
            Weights::Bf16 { values } => values.len(),
        }
    }

    pub fn dequantise(&self) -> Vec<f32> {
        match &self.weights {
            Weights::Nvfp4 { packed, scales, global_scale } => {
                let scales = e4m3_to_f32(scales);
                dequant::dequantise(packed, &scales, *global_scale, self.cols, self.rows)
            }
            Weights::Fp8 { values, scales } => {
                let mut out = f8_to_f32(values);
                let scales = bf16_bytes_to_f32(scales);
                for (index, value) in out.iter_mut().enumerate() {
                    let row = index / self.cols;
                    *value *= scales[if scales.len() == self.rows { row } else { 0 }];
                }
                out
            }
            Weights::Bf16 { values } => values
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModelSource {
    Hub { repo: String, revision: Option<String> },
    Dir(PathBuf),
}

impl ModelSource {
    pub fn parse(spec: &str) -> Self {
        if Path::new(spec).is_dir() {
            return ModelSource::Dir(PathBuf::from(spec));
        }

        let parts: Vec<&str> = spec.split('/').collect();
        let repo_shaped = parts.len() == 2
            && !parts[0].is_empty()
            && !parts[1].is_empty()
            && !spec.starts_with(['/', '.']);
        if repo_shaped {
            ModelSource::Hub { repo: spec.to_string(), revision: None }
        } else {
            ModelSource::Dir(PathBuf::from(spec))
        }
    }

    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        if let ModelSource::Hub { revision: slot, .. } = &mut self {
            *slot = Some(revision.into());
        }
        self
    }

    pub fn resolve(&self, cache_dir: Option<&Path>) -> Result<PathBuf> {
        match self {
            ModelSource::Dir(dir) => {
                if !dir.join("model.safetensors.index.json").exists()
                    && !dir.join("model.safetensors").exists()
                {
                    bail!("{} holds no safetensors checkpoint", dir.display());
                }
                Ok(dir.clone())
            }
            ModelSource::Hub { repo, revision } => {
                // **hf-hub 1.0** rewrote the API: `HFClientSync` replaces `Api`, a repo is addressed by
                // (owner, name) rather than a single id, and a file is fetched through a builder whose
                // `send()` returns the path in the cache. The cache directory is a client setting, not a
                // per-call one, so it is applied when the client is built.
                let mut builder = hf_hub::HFClientBuilder::new();
                if let Some(dir) = cache_dir {
                    builder = builder.cache_dir(dir);
                }
                let client = builder.build_sync().context("cannot reach HuggingFace")?;
                let (owner, name) = repo
                    .split_once('/')
                    .with_context(|| format!("{repo:?} is not an owner/name repo id"))?;
                let repo = client.model(owner, name);
                let index = repo
                    .download_file()
                    .filename("model.safetensors.index.json")
                    .maybe_revision(revision.clone())
                    .send()
                    .with_context(|| format!("{repo:?} has no model.safetensors.index.json"))?;
                let dir = index.parent().map(Path::to_path_buf).unwrap_or_default();

                let text = std::fs::read_to_string(&index)?;
                let parsed: serde_json::Value = serde_json::from_str(&text)?;
                if let Some(map) = parsed["weight_map"].as_object() {
                    let mut shards: Vec<&str> = map
                        .values()
                        .filter_map(|v| v.as_str())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    shards.sort_unstable();
                    for shard in shards {
                        repo.download_file()
                            .filename(shard)
                            .maybe_revision(revision.clone())
                            .send()
                            .with_context(|| format!("cannot fetch {shard}"))?;
                    }
                }
                Ok(dir)
            }
        }
    }
}

pub struct Checkpoint {
    dir: PathBuf,
    index: HashMap<String, String>,
    shards: std::cell::RefCell<HashMap<String, Shard>>,
}

struct Shard {
    mmap: Mmap,
    tensors: HashMap<String, (std::ops::Range<usize>, safetensors::Dtype, Vec<usize>)>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;

        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("cannot map {}", path.display()))?;

        if mmap.len() < 8 {
            bail!("{} is too short to be a safetensors shard", path.display());
        }
        let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
        if 8 + header_len > mmap.len() {
            bail!("{} has a header longer than the file", path.display());
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])
            .with_context(|| format!("cannot parse the header of {}", path.display()))?;

        let mut tensors = HashMap::new();
        let object = header
            .as_object()
            .with_context(|| format!("{} header is not an object", path.display()))?;
        for (name, entry) in object {
            let Some(offsets) = entry.get("data_offsets").and_then(|v| v.as_array()) else {
                continue;
            };
            if offsets.len() != 2 {
                bail!("{name} in {} has {} offsets", path.display(), offsets.len());
            }
            let start = 8 + header_len + offsets[0].as_u64().unwrap_or(0) as usize;
            let end = 8 + header_len + offsets[1].as_u64().unwrap_or(0) as usize;
            if end > mmap.len() || start > end {
                bail!("{name} claims bytes {start}..{end} of a {} byte file", mmap.len());
            }
            let dtype = entry
                .get("dtype")
                .and_then(|v| v.as_str())
                .and_then(dtype_from_name)
                .with_context(|| format!("{name} has no usable dtype"))?;
            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
                .unwrap_or_default();
            tensors.insert(name.clone(), (start..end, dtype, shape));
        }
        Ok(Self { mmap, tensors })
    }

    fn get(&self, name: &str) -> Option<(&[u8], safetensors::Dtype, &[usize])> {
        let (range, dtype, shape) = self.tensors.get(name)?;
        Some((&self.mmap[range.clone()], *dtype, shape))
    }
}

fn dtype_from_name(name: &str) -> Option<safetensors::Dtype> {
    use safetensors::Dtype;
    Some(match name {
        "F64" => Dtype::F64,
        "F32" => Dtype::F32,
        "F16" => Dtype::F16,
        "BF16" => Dtype::BF16,
        "I64" => Dtype::I64,
        "I32" => Dtype::I32,
        "I16" => Dtype::I16,
        "I8" => Dtype::I8,
        "U8" => Dtype::U8,
        "BOOL" => Dtype::BOOL,
        "F8_E4M3" => Dtype::F8_E4M3,
        "F8_E5M2" => Dtype::F8_E5M2,
        _ => return None,
    })
}

impl Checkpoint {
    pub fn from_source(source: &ModelSource, cache_dir: Option<&Path>) -> Result<Self> {
        Self::open(source.resolve(cache_dir)?)
    }

    pub fn open_single(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let file_name = path
            .file_name()
            .with_context(|| format!("{} has no file name", path.display()))?
            .to_string_lossy()
            .to_string();
        let shard = Shard::open(&path)?;
        let index: HashMap<String, String> =
            shard.tensors.keys().map(|k| (k.clone(), file_name.clone())).collect();
        if index.is_empty() {
            bail!("{} has no tensors", path.display());
        }
        Ok(Self { dir, index, shards: RefCell::new(HashMap::new()) })
    }

    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("cannot read {}", index_path.display()))?;
        let parsed: serde_json::Value =
            serde_json::from_str(&text).context("index is not valid JSON")?;
        let map = parsed["weight_map"]
            .as_object()
            .context("index has no weight_map")?;
        let index: HashMap<String, String> = map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect();
        if index.is_empty() {
            bail!("{} has an empty weight_map", index_path.display());
        }
        Ok(Self {
            dir,
            index,
            shards: std::cell::RefCell::new(HashMap::new()),
        })
    }

    pub fn shard_path(&self, name: &str) -> Result<std::path::PathBuf> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("{name} is not in the index"))?;
        Ok(self.dir.join(shard))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn storage_of(&self, prefix: &str) -> Option<Storage> {
        storage_of(self.tensor_names(), prefix)
    }

    pub fn for_each_tensor(
        &self,
        mut f: impl FnMut(&str, safetensors::Dtype, usize, bool),
    ) -> Result<()> {
        let missing: std::collections::HashSet<String> = {
            let shards = self.shards.borrow();
            self.index
                .values()
                .filter(|s| !shards.contains_key(*s))
                .cloned()
                .collect()
        };
        for shard_name in missing {
            let opened = Shard::open(&self.dir.join(&shard_name))?;
            self.shards.borrow_mut().insert(shard_name, opened);
        }

        let shards = self.shards.borrow();

        for name in self.index.keys() {
            let Some(shard_name) = self.index.get(name) else { continue };
            let Some(shard) = shards.get(shard_name) else { continue };
            let Some((data, dtype, _)) = shard.get(name) else { continue };

            let has_packed_weight = name
                .strip_suffix(".weight_scale")
                .map(|prefix| self.index.contains_key(&format!("{prefix}.weight_packed")))
                .unwrap_or(false);
            f(name, dtype, data.len(), has_packed_weight);
        }
        Ok(())
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<usize> {
        self.with_raw(name, |data, _, _| Ok(data.len())).ok()
    }

    pub fn tensor_dtype_bytes(&self, name: &str) -> Option<(safetensors::Dtype, usize, f32)> {
        self.with_raw(name, |data, dtype, _| {
            let per_byte = match dtype {
                safetensors::Dtype::U8 => 2.0,
                safetensors::Dtype::F8_E4M3 => 1.0,
                safetensors::Dtype::BF16 => 0.5,
                safetensors::Dtype::F32 => 0.25,
                other => return Err(anyhow::anyhow!("{name} has dtype {other:?}")),
            };
            Ok((dtype, data.len(), per_byte))
        })
        .ok()
    }

    pub fn tensor_storage(&self, name: &str) -> Option<Storage> {
        self.with_raw(name, |_, dtype, _| {
            Ok(match dtype {
                safetensors::Dtype::U8 => Storage::Nvfp4,
                safetensors::Dtype::F8_E4M3 => Storage::Fp8,
                safetensors::Dtype::BF16 => Storage::Bf16,

                _ => return Err(anyhow::anyhow!("{name} has dtype {dtype:?}")),
            })
        })
        .ok()
    }

    pub fn with_raw<T>(
        &self,
        name: &str,
        f: impl FnOnce(&[u8], safetensors::Dtype, &[usize]) -> Result<T>,
    ) -> Result<T> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("{name} is not in the index"))?
            .clone();
        let mut shards = self.shards.borrow_mut();
        if !shards.contains_key(&shard) {
            let opened = Shard::open(&self.dir.join(&shard))?;
            shards.insert(shard.clone(), opened);
        }
        let shard_ref = shards.get(&shard).expect("just inserted");
        let (data, dtype, shape) = shard_ref
            .get(name)
            .with_context(|| format!("{name} is not in shard {shard}"))?;
        f(data, dtype, shape)
    }

    pub fn text_prefix(&self) -> Result<&'static str> {
        for candidate in ["model.language_model", "model", ""] {
            let probe = if candidate.is_empty() {
                "layers.0.input_layernorm.weight".to_string()
            } else {
                format!("{candidate}.layers.0.input_layernorm.weight")
            };
            if self.contains(&probe) {
                return Ok(candidate);
            }
        }
        bail!(
            "no text tower found: neither `model.language_model.layers.0...` nor `model.layers.0...` \
             is in the index, so this checkpoint is not a Qwen3.5 text model"
        )
    }

    pub fn embed_tokens_name(&self) -> Result<String> {
        let prefix = self.text_prefix()?;
        Ok(if prefix.is_empty() {
            "embed_tokens.weight".to_string()
        } else {
            format!("{prefix}.embed_tokens.weight")
        })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.with_raw(name, |data, dtype, shape| {
            let values = match dtype {
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),

                other => bail!("{name} is {other:?}; tensor_f32 reads only F32 and BF16"),
            };
            Ok((values, shape.to_vec()))
        })
    }

    pub fn projection(&self, prefix: &str) -> Result<Projection> {
        let storage = self
            .storage_of(prefix)
            .with_context(|| format!("{prefix} has no weights"))?;
        match storage {
            Storage::Nvfp4 => self.nvfp4_projection(prefix),
            Storage::Fp8 => self.fp8_projection(prefix),
            Storage::Bf16 => self.bf16_projection(prefix),
        }
    }

    fn bf16_projection(&self, prefix: &str) -> Result<Projection> {
        let weight_name = format!("{prefix}.weight");
        let (values, rows, cols) = self.with_raw(&weight_name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::BF16 {
                bail!("{weight_name} is {dtype:?}, expected BF16");
            }
            if shape.len() != 2 {
                bail!("{weight_name} has shape {shape:?}, expected 2 dimensions");
            }
            Ok((data.to_vec(), shape[0], shape[1]))
        })?;
        Ok(Projection {
            weights: Weights::Bf16 { values },
            input_global_scale: None,
            rows,
            cols,
        })
    }

    fn fp8_projection(&self, prefix: &str) -> Result<Projection> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let (weight, rows, cols) = self.with_raw(&weight_name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::F8_E4M3 {
                bail!("{weight_name} is {dtype:?}, expected F8_E4M3");
            }
            if shape.len() != 2 {
                bail!("{weight_name} has shape {shape:?}, expected 2 dimensions");
            }
            Ok((data.to_vec(), shape[0], shape[1]))
        })?;
        let (scales, scale_rows, scale_cols) = self.with_raw(&scale_name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::BF16 {
                bail!("{scale_name} is {dtype:?}, expected BF16");
            }

            let rows = shape.first().copied().unwrap_or(0);
            let cols = shape.get(1).copied().unwrap_or(1);
            Ok((data.to_vec(), rows, cols))
        })?;

        if scale_rows != rows {
            bail!("{scale_name} has {scale_rows} rows for {rows} weight rows");
        }
        let per_row = match scale_cols {
            1 => true,
            n if n == cols => {
                bail!(
                    "{prefix}: {scale_name} is per-column ({n} scales for {cols} columns), which is \
                     not a scheme this loader implements"
                )
            }
            n => bail!("{prefix}: unexpected scale width {n} for {cols} columns"),
        };

        let _ = per_row;
        Ok(Projection {
            weights: Weights::Fp8 { values: weight, scales },
            rows,
            cols,
            input_global_scale: None,
        })
    }

    fn nvfp4_projection(&self, prefix: &str) -> Result<Projection> {
        let packed_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");
        let global_name = format!("{prefix}.weight_global_scale");
        let (packed, rows, packed_cols) = self.with_raw(&packed_name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::U8 {
                bail!("{packed_name} is {dtype:?}, expected U8");
            }
            if shape.len() != 2 {
                bail!("{packed_name} has shape {shape:?}, expected 2 dimensions");
            }
            Ok((data.to_vec(), shape[0], shape[1]))
        })?;
        let (scales, scale_cols) = self.with_raw(&scale_name, |data, dtype, shape| {
            if dtype != safetensors::Dtype::F8_E4M3 {
                bail!("{scale_name} is {dtype:?}, expected F8_E4M3");
            }
            if shape.len() != 2 || shape[0] != rows {
                bail!("{scale_name} has shape {shape:?}, expected [{rows}, _]");
            }
            Ok((data.to_vec(), shape[1]))
        })?;

        let global = self.with_raw(&global_name, |data, dtype, _shape| {
            if dtype != safetensors::Dtype::F32 || data.len() < 4 {
                bail!("{global_name} is {dtype:?} with {} bytes, expected one f32", data.len());
            }
            Ok(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
        })?;

        let cols = packed_cols * 2;
        if scale_cols * dequant::SCALE_BLOCK != cols {
            bail!(
                "{prefix}: {scale_cols} scales x {} do not cover {cols} values",
                dequant::SCALE_BLOCK
            );
        }

        let input_global_scale = self
            .with_raw(&format!("{prefix}.input_global_scale"), |data, dtype, _| {
                if dtype != safetensors::Dtype::F32 || data.len() < 4 {
                    bail!("{prefix}.input_global_scale is {dtype:?}, expected one F32");
                }
                Ok(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            })
            .ok();
        Ok(Projection {
            weights: Weights::Nvfp4 { packed, scales, global_scale: global },
            rows,
            cols,
            input_global_scale,
        })
    }
}

fn f8_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|b| {
            let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let exponent = ((b >> 3) & 0x0F) as i32;
            let mantissa = (b & 0x07) as f32 / 8.0;

            if exponent == 0 {
                sign * mantissa * 2f32.powi(-6)
            } else {
                sign * (1.0 + mantissa) * 2f32.powi(exponent - 7)
            }
        })
        .collect()
}

fn bf16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((bits >> 7) & 0xFF) as i32;
    let mantissa = (bits & 0x7F) as f32 / 128.0;
    match exponent {
        0 => sign * mantissa * 2f32.powi(-126),
        0xFF => f32::NAN,
        _ => sign * (1.0 + mantissa) * 2f32.powi(exponent - 127),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dequantised_layer_matches_the_bf16_checkpoint() {
        let nvfp4_dir = std::env::var("CLEF_NVFP4_DIR").unwrap_or_else(|_| "/models/clef-flash-nvfp4".into());
        let bf16_dir = std::env::var("CLEF_BF16_DIR").unwrap_or_else(|_| "/models/clef-flash".into());
        if !Path::new(&nvfp4_dir).join("model.safetensors.index.json").exists() {
            eprintln!("SKIPPED: no checkpoint at {nvfp4_dir}");
            return;
        }

        let _ = bf16_dir;
        let (nvfp4, bf16) = open_both().expect("both checkpoints");
        let prefix = "model.language_model.layers.0.mlp.gate_proj";

        let got = nvfp4.projection(prefix).expect("dequantise");
        assert_eq!(got.storage(), Storage::Nvfp4);
        assert_eq!(got.rows, 12288);
        assert_eq!(got.cols, 4096);

        let reference = read_bf16(&bf16, prefix);
        let relative = relative_error(&got.dequantise(), &reference);

        assert!(
            (0.02..0.30).contains(&relative),
            "relative error {relative:.4} is not plausible quantisation noise"
        );
    }

    #[test]
    fn an_fp8_layer_matches_the_bf16_checkpoint() {
        let Some((nvfp4, bf16)) = open_both() else { return };

        let prefix = "model.language_model.layers.9.mlp.gate_proj";
        assert_eq!(nvfp4.storage_of(prefix), Some(Storage::Fp8), "layer 9 should be FP8");

        let got = nvfp4.projection(prefix).expect("dequantise");
        assert_eq!(got.storage(), Storage::Fp8);
        assert_eq!((got.rows, got.cols), (12288, 4096));

        let relative = relative_error(&got.dequantise(), &read_bf16(&bf16, prefix));

        assert!(
            (0.005..0.08).contains(&relative),
            "relative error {relative:.4} is not plausible FP8 quantisation noise"
        );
    }

    #[test]
    fn a_model_id_and_a_path_are_told_apart() {
        for id in ["Cloudflare/clef-flash", "alpha-x-ai/clef-flash-NVFP4"] {
            assert!(
                matches!(ModelSource::parse(id), ModelSource::Hub { .. }),
                "{id} should be a repository id"
            );
        }
        std::fs::create_dir_all("models/clef-flash").ok();
        for path in ["/models/clef-flash", "./clef-flash", "models/clef-flash", "/tmp/x/y/z"] {
            assert!(
                matches!(ModelSource::parse(path), ModelSource::Dir(_)),
                "{path} should be a path"
            );
        }

        let with_rev = ModelSource::parse("Cloudflare/clef-flash").with_revision("main");
        assert!(matches!(with_rev, ModelSource::Hub { revision: Some(_), .. }));
        std::fs::remove_dir_all("models/clef-flash").ok();
        std::fs::remove_dir("models").ok();
    }

    #[test]

    #[test]
    fn a_quantized_tensor_is_refused_by_the_scalar_reader() {
        let dir = std::env::var("CLEF_NVFP4_DIR").unwrap_or_else(|_| "/models/clef-flash-nvfp4".into());
        if !Path::new(&dir).join("model.safetensors.index.json").exists() {
            eprintln!("SKIPPED: no checkpoint at {dir}");
            return;
        }
        let checkpoint = Checkpoint::open(&dir).expect("open");
        let err = checkpoint
            .tensor_f32("model.language_model.layers.0.mlp.gate_proj.weight_packed")
            .expect_err("a packed FP4 tensor must not be read as f32");
        assert!(
            err.to_string().contains("reads only F32 and BF16"),
            "unexpected error: {err}"
        );
        let (values, shape) = checkpoint
            .tensor_f32("model.language_model.layers.0.linear_attn.in_proj_a.weight")
            .expect("BF16 must load");
        assert_eq!(shape, vec![32, 4096]);
        assert_eq!(values.len(), 32 * 4096);
    }

    fn open_both() -> Option<(Checkpoint, Checkpoint)> {
        let nvfp4_dir =
            std::env::var("CLEF_NVFP4_DIR").unwrap_or_else(|_| "/models/clef-flash-nvfp4".into());
        let bf16_dir = std::env::var("CLEF_BF16_DIR").unwrap_or_else(|_| "/models/clef-flash".into());
        if !Path::new(&nvfp4_dir).join("model.safetensors.index.json").exists() {
            eprintln!("SKIPPED: no checkpoint at {nvfp4_dir}");
            return None;
        }
        Some((
            Checkpoint::open(&nvfp4_dir).expect("open nvfp4"),
            Checkpoint::open(&bf16_dir).expect("open bf16"),
        ))
    }

    fn read_bf16(checkpoint: &Checkpoint, prefix: &str) -> Vec<f32> {
        checkpoint
            .with_raw(&format!("{prefix}.weight"), |data, dtype, _shape| {
                assert_eq!(dtype, safetensors::Dtype::BF16, "reference is not BF16");
                Ok(data
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            })
            .expect("read reference")
    }

    fn relative_error(got: &[f32], reference: &[f32]) -> f32 {
        assert_eq!(got.len(), reference.len(), "lengths differ");
        let mean_abs = reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32;
        let mean_err =
            got.iter().zip(reference).map(|(a, b)| (a - b).abs()).sum::<f32>() / reference.len() as f32;
        mean_err / mean_abs
    }
}
