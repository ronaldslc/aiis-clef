use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Storage {
    Nvfp4,
    Fp8,
    Bf16,
}

const WEIGHT_PACKED: &str = "weight_packed";
const WEIGHT: &str = "weight";
const WEIGHT_SCALE: &str = "weight_scale";

pub fn storage_of<'a>(names: impl Iterator<Item = &'a str>, prefix: &str) -> Option<Storage> {
    let mut packed = false;
    let mut plain = false;
    let mut scaled = false;
    let want_packed = format!("{prefix}.{WEIGHT_PACKED}");
    let want_plain = format!("{prefix}.{WEIGHT}");
    let want_scale = format!("{prefix}.{WEIGHT_SCALE}");

    for name in names {
        if name == want_packed {
            packed = true;
        } else if name == want_plain {
            plain = true;
        } else if name == want_scale {
            scaled = true;
        }
    }
    match (packed, plain, scaled) {
        (true, _, _) => Some(Storage::Nvfp4),
        (false, true, true) => Some(Storage::Fp8),
        (false, true, false) => Some(Storage::Bf16),
        (false, false, _) => None,
    }
}

#[derive(Debug, Default)]
pub struct LayerMap {
    entries: Vec<(String, String, usize, Storage)>,
}

impl LayerMap {
    pub fn from_tensor_names<'a>(names: impl Iterator<Item = &'a str>) -> Self {
        let names: Vec<&str> = names.collect();
        let mut projections: BTreeSet<(usize, String)> = BTreeSet::new();

        for name in &names {
            if let Some((layer, projection)) = split_layer_projection(name) {
                projections.insert((layer, projection));
            }
        }

        let mut entries = Vec::new();
        for (layer, projection) in projections {
            let prefix = format!("model.language_model.layers.{layer}.{projection}");
            if let Some(storage) = storage_of(names.iter().copied(), &prefix) {
                let family = projection.rsplit_once('.').map(|(f, _)| f).unwrap_or(&projection);
                entries.push((family.to_string(), projection, layer, storage));
            }
        }
        Self { entries }
    }

    pub fn layers_with(&self, projection: &str, storage: Storage) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .entries
            .iter()
            .filter(|(_, p, _, s)| p == projection && *s == storage)
            .map(|(_, _, layer, _)| *layer)
            .collect();
        out.sort_unstable();
        out
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn split_layer_projection(name: &str) -> Option<(usize, String)> {
    const MARKER: &str = "model.language_model.layers.";
    let rest = name.strip_prefix(MARKER)?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;

    let projection = tail
        .strip_suffix(".weight_packed")
        .or_else(|| tail.strip_suffix(".weight_scale"))
        .or_else(|| tail.strip_suffix(".weight_global_scale"))
        .or_else(|| tail.strip_suffix(".input_global_scale"))
        .or_else(|| tail.strip_suffix(".weight"))
        .or_else(|| tail.strip_suffix(".bias"))
        .unwrap_or(tail);
    if projection.is_empty() {
        return None;
    }
    Some((layer, projection.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index_names() -> Option<Vec<String>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/eval/clef_index_fixture.json");
        let text = std::fs::read_to_string(&path).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        Some(
            value["tensor_names"]
                .as_array()?
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
        )
    }

    #[test]
    fn matches_the_real_checkpoint_index() {
        let Some(names) = index_names() else {
            eprintln!("SKIPPED: no tests/eval/clef_index_fixture.json");
            return;
        };
        let map = LayerMap::from_tensor_names(names.iter().map(String::as_str));
        assert!(map.len() > 200, "parsed only {} projections", map.len());

        let nvfp4: Vec<usize> = (0..5).chain(15..32).collect();
        assert_eq!(map.layers_with("mlp.gate_proj", Storage::Nvfp4), nvfp4);
        assert_eq!(map.layers_with("mlp.up_proj", Storage::Nvfp4), nvfp4);
        assert_eq!(map.layers_with("mlp.down_proj", Storage::Nvfp4), nvfp4);
        assert_eq!(map.layers_with("mlp.gate_proj", Storage::Fp8), (5..15).collect::<Vec<_>>());

        assert_eq!(
            map.layers_with("linear_attn.in_proj_qkv", Storage::Nvfp4),
            vec![17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
        );
        assert_eq!(map.layers_with("linear_attn.in_proj_qkv", Storage::Fp8).len(), 13);

        let full: Vec<usize> = (3..32).step_by(4).collect();
        let mut attention: Vec<usize> = map.layers_with("self_attn.q_proj", Storage::Fp8);
        attention.extend(map.layers_with("self_attn.q_proj", Storage::Nvfp4));
        attention.sort_unstable();
        assert_eq!(attention, full);

        assert_eq!(map.layers_with("linear_attn.in_proj_a", Storage::Bf16).len(), 24);
        assert_eq!(map.layers_with("linear_attn.in_proj_b", Storage::Bf16).len(), 24);
        assert!(map.layers_with("linear_attn.in_proj_a", Storage::Nvfp4).is_empty());
    }

    #[test]
    fn weight_scale_does_not_decide_the_scheme() {
        let fp8 = ["p.weight", "p.weight_scale"];
        assert_eq!(storage_of(fp8.into_iter(), "p"), Some(Storage::Fp8));
        let nvfp4 = ["p.weight_packed", "p.weight_scale", "p.weight_global_scale"];
        assert_eq!(storage_of(nvfp4.into_iter(), "p"), Some(Storage::Nvfp4));

        let bf16 = ["p.weight"];
        assert_eq!(storage_of(bf16.into_iter(), "p"), Some(Storage::Bf16));
        let nothing: [&str; 0] = [];
        assert_eq!(storage_of(nothing.into_iter(), "p"), None);
    }

    #[test]
    fn non_layer_tensors_are_ignored() {
        assert_eq!(split_layer_projection("lm_head.weight"), None);
        assert_eq!(split_layer_projection("model.language_model.embed_tokens.weight"), None);
        assert_eq!(
            split_layer_projection("model.language_model.layers.7.mlp.gate_proj.weight"),
            Some((7, "mlp.gate_proj".to_string()))
        );
    }
}
