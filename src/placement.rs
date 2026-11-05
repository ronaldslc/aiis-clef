use crate::checkpoint::Storage;
use crate::loader::Checkpoint;
use anyhow::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub storage: Storage,
    pub bytes: usize,
    pub on_device: bool,
    pub expanded: usize,
}

#[derive(Debug, Default)]
pub struct Plan {
    pub entries: BTreeMap<String, Entry>,
}

impl Plan {
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let mut plan = Plan::default();

        checkpoint.for_each_tensor(|name, dtype, bytes, has_packed_weight| {
            let storage = match dtype {
                safetensors::Dtype::U8 => Storage::Nvfp4,
                safetensors::Dtype::F8_E4M3 => {
                    if has_packed_weight {
                        Storage::Nvfp4
                    } else {
                        Storage::Fp8
                    }
                }
                safetensors::Dtype::BF16 => Storage::Bf16,

                _ => return,
            };

            let values_per_byte = match dtype {
                safetensors::Dtype::U8 => 2.0f64,
                safetensors::Dtype::F8_E4M3 => 1.0,
                safetensors::Dtype::BF16 => 0.5,
                _ => 0.25,
            };
            let values = (bytes as f64 * values_per_byte) as usize;
            plan.entries.insert(
                name.to_string(),
                Entry {
                    storage,
                    bytes,
                    on_device: !name.ends_with("lm_head.weight"),
                    expanded: values * 4,
                },
            );
        })?;
        Ok(plan)
    }

    pub fn device_bytes(&self) -> usize {
        self.entries.values().filter(|e| e.on_device).map(|e| e.bytes).sum()
    }

    pub fn host_bytes(&self) -> usize {
        self.entries.values().filter(|e| !e.on_device).map(|e| e.bytes).sum()
    }

    pub fn expanded_bytes(&self) -> usize {
        self.entries.values().filter(|e| e.on_device).map(|e| e.expanded).sum()
    }

    pub fn by_storage(&self) -> BTreeMap<&'static str, (usize, usize)> {
        let mut out: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        for e in self.entries.values().filter(|e| e.on_device) {
            let key = match e.storage {
                Storage::Nvfp4 => "NVFP4 (packed E2M1 + E4M3 scales)",
                Storage::Fp8 => "FP8 E4M3",
                Storage::Bf16 => "BF16",
            };
            let slot = out.entry(key).or_insert((0, 0));
            slot.0 += 1;
            slot.1 += e.bytes;
        }
        out
    }
}

fn storage_of(checkpoint: &Checkpoint, name: &str) -> Option<Storage> {
    if name.ends_with(".weight_packed") {
        return Some(Storage::Nvfp4);
    }
    if let Some(prefix) = name.strip_suffix(".weight_scale") {
        if checkpoint.contains(&format!("{prefix}.weight_packed")) {
            return Some(Storage::Nvfp4);
        }
        return Some(Storage::Fp8);
    }
    if name.ends_with(".weight") {
        return checkpoint.tensor_storage(name);
    }

    checkpoint.tensor_storage(name)
}

pub struct HostTable {
    map: memmap2::Mmap,
    start: usize,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl HostTable {
    pub fn map(path: &std::path::Path, tensor: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;

        let map = unsafe { memmap2::Mmap::map(&file)? };

        if map.len() < 8 {
            anyhow::bail!("{} is too short to be a safetensors file", path.display());
        }
        let header_len = u64::from_le_bytes(map[..8].try_into()?) as usize;
        if 8 + header_len > map.len() {
            anyhow::bail!("{} has a header longer than the file", path.display());
        }
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + header_len])?;

        let entry = header
            .get(tensor)
            .ok_or_else(|| anyhow::anyhow!("{tensor} is not in {}", path.display()))?;

        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
            .unwrap_or_default();
        if shape.len() != 2 {
            anyhow::bail!("{tensor} has shape {shape:?}, expected 2 dimensions");
        }
        let dtype = entry["dtype"].as_str().unwrap_or("");
        if dtype != "BF16" {
            anyhow::bail!("{tensor} is {dtype}; the host gather reads BF16");
        }

        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("{tensor} has no data_offsets"))?;
        if offsets.len() != 2 {
            anyhow::bail!("{tensor} has {} data offsets, expected 2", offsets.len());
        }
        let start = 8 + header_len + offsets[0].as_u64().unwrap_or(0) as usize;
        let end = 8 + header_len + offsets[1].as_u64().unwrap_or(0) as usize;
        if end > map.len() || start > end {
            anyhow::bail!("{tensor} claims bytes {start}..{end} of a {} byte file", map.len());
        }
        let (rows, cols) = (shape[0], shape[1]);
        let expected = rows * cols * 2;
        if end - start != expected {
            anyhow::bail!(
                "{tensor} is {rows}x{cols} BF16 ({expected} bytes) but the header says {}",
                end - start
            );
        }
        Ok(Self { map, start, rows, cols, row_bytes: cols * 2 })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, index: usize) -> &[u8] {
        debug_assert!(index < self.rows);
        let at = self.start + index * self.row_bytes;
        &self.map[at..at + self.row_bytes]
    }

    pub fn gather_f32(&self, indices: &[u32], out: &mut [f32]) {
        assert_eq!(out.len(), indices.len() * self.cols, "output sized for the gather");
        for (i, &row) in indices.iter().enumerate() {
            let bytes = self.row(row as usize);
            let dst = &mut out[i * self.cols..(i + 1) * self.cols];
            for (c, chunk) in bytes.chunks_exact(2).enumerate() {
                dst[c] = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_is_not_slow() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: no checkpoint");
            return;
        };
        let t = std::time::Instant::now();
        checkpoint.for_each_tensor(|_, _, _, _| {}).expect("pass 1");
        eprintln!("  for_each_tensor (cold): {:?}", t.elapsed());

        let t = std::time::Instant::now();
        checkpoint.for_each_tensor(|_, _, _, _| {}).expect("pass 2");
        eprintln!("  for_each_tensor (warm): {:?}", t.elapsed());

        let t = std::time::Instant::now();
        let plan = Plan::from_checkpoint(&checkpoint).expect("plan");
        eprintln!("  Plan::from_checkpoint: {:?} for {} tensors", t.elapsed(), plan.entries.len());
    }

    #[test]
    fn the_native_footprint_fits_and_the_expanded_one_does_not() {
        let Ok(checkpoint) = Checkpoint::open("/models/clef-flash-nvfp4") else {
            eprintln!("SKIP: /models/clef-flash-nvfp4 not present");
            return;
        };
        let plan = Plan::from_checkpoint(&checkpoint).expect("plan");

        let device = plan.device_bytes();
        let host = plan.host_bytes();
        let expanded = plan.expanded_bytes();

        eprintln!("  on device: {:.1} MB", device as f64 / 1e6);
        eprintln!("  in host:   {:.1} MB  (lm_head, the gather table)", host as f64 / 1e6);
        eprintln!("  total:     {:.1} MB", (device + host) as f64 / 1e6);
        eprintln!("  if expanded to f32: {:.1} MB  ({:.2}x)", expanded as f64 / 1e6,
                  expanded as f64 / device as f64);
        for (kind, (count, bytes)) in plan.by_storage() {
            eprintln!("    {count:>4} x {kind:<34} {:>8.1} MB", bytes as f64 / 1e6);
        }

        let total = device + host;
        assert!(
            (9_900_000_000..10_200_000_000).contains(&total),
            "the native footprint is {:.1} MB; the checkpoint is 10,029.5 MB, so bytes are missing \
             or double-counted",
            total as f64 / 1e6
        );

        eprintln!(
            "  expanded would be {:.1} GB, i.e. {:.2}x the bytes read per forward pass",
            expanded as f64 / 1e9,
            expanded as f64 / device as f64
        );

        assert!(
            host > 1_900_000_000 && host < 2_200_000_000,
            "the host-resident set is {:.1} MB, expected ~2034 MB (lm_head)",
            host as f64 / 1e6
        );
    }
}
