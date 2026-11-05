const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub const SCALE_BLOCK: usize = 16;

#[inline]
pub fn nibbles(byte: u8) -> (f32, f32) {
    (decode(byte & 0x0F), decode((byte >> 4) & 0x0F))
}

#[inline]
fn decode(code: u8) -> f32 {
    let magnitude = E2M1[(code & 0x7) as usize];
    if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub fn dequantise(packed: &[u8], scales: &[f32], global_scale: f32, cols: usize,
                  rows: usize) -> Vec<f32> {
    debug_assert!(
        cols % SCALE_BLOCK == 0,
        "cols ({cols}) must be a multiple of the {SCALE_BLOCK}-value scale block"
    );

    let scales_per_row = cols / SCALE_BLOCK;
    let mut out = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let packed_row = &packed[row * (cols / 2)..(row + 1) * (cols / 2)];
        let scale_row = &scales[row * scales_per_row..(row + 1) * scales_per_row];
        for (index, byte) in packed_row.iter().enumerate() {
            let (low, high) = nibbles(*byte);

            let scale_low = scale_row[(index * 2) / SCALE_BLOCK];
            let scale_high = scale_row[(index * 2 + 1) / SCALE_BLOCK];
            out.push(low * scale_low / global_scale);
            out.push(high * scale_high / global_scale);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_dequantisation() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/eval/clef_dequant_fixture.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: cannot read {} ({e})", path.display());
                return;
            }
        };
        let fixture: serde_json::Value = serde_json::from_str(&text).expect("fixture json");

        let convention = &fixture["convention"];
        assert_eq!(convention["nibble_order"], "low first");
        assert_eq!(convention["global_scale"], "divisor");
        assert_eq!(convention["scale_block"], SCALE_BLOCK);

        let rows = fixture["rows"].as_u64().expect("rows") as usize;
        let packed: Vec<u8> = fixture["weight_packed"]
            .as_array()
            .expect("weight_packed")
            .iter()
            .map(|v| v.as_u64().expect("byte") as u8)
            .collect();
        let scales: Vec<f32> = fixture["weight_scale"]
            .as_array()
            .expect("weight_scale")
            .iter()
            .map(|v| v.as_f64().expect("scale") as f32)
            .collect();
        let global = fixture["weight_global_scale"].as_f64().expect("global") as f32;
        let expected: Vec<f32> = fixture["expected"]
            .as_array()
            .expect("expected")
            .iter()
            .map(|v| v.as_f64().expect("value") as f32)
            .collect();

        let cols = expected.len() / rows;
        assert_eq!(packed.len() * 2, expected.len(), "packed does not cover the same K");
        assert_eq!(scales.len() * SCALE_BLOCK, expected.len(), "scales do not cover the same K");

        let got = dequantise(&packed, &scales, global, cols, rows);
        assert_eq!(got.len(), expected.len());
        for (index, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                "value {index} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn the_nibble_order_is_observable() {
        let mut packed = [0u8; 8];
        packed[0] = 0x01;
        let got = dequantise(&packed, &[1.0], 1.0, SCALE_BLOCK, 1);
        assert_eq!(got.len(), SCALE_BLOCK);
        assert_eq!(got[0], 0.5, "low nibble first: byte 0 holds 0.5 then 0.0");
        assert_eq!(got[1], 0.0);

        assert_ne!((got[0], got[1]), (0.0, 0.5));
    }

    #[test]
    fn scales_apply_per_block() {
        let packed = [0u8; 16];
        let mut values = packed;
        values[0] = 0x02;
        values[8] = 0x02;
        let got = dequantise(&values, &[2.0, 3.0], 1.0, 32, 1);
        assert_eq!(got[0], 2.0, "first block uses the first scale");
        assert_eq!(got[16], 3.0, "second block uses the second scale");
    }

    #[test]
    fn the_sign_bit_is_honoured() {
        assert_eq!(decode(0x9), -0.5);
        assert_eq!(decode(0x1), 0.5);
        assert_eq!(decode(0x0), 0.0);
        assert_eq!(decode(0x7), 6.0);
    }
}
