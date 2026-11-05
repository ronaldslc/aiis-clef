pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
pub fn unpack_nibbles(byte: u8) -> (f32, f32) {
    let low = E2M1[(byte & 0x07) as usize];
    let high = E2M1[((byte >> 4) & 0x07) as usize];
    let low = if byte & 0x08 != 0 { -low } else { low };
    let high = if byte & 0x80 != 0 { -high } else { high };
    (low, high)
}

fn e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let exp = ((b >> 3) & 0x0F) as i32;
            let mant = (b & 0x07) as f32;

            if exp == 0 {
                sign * mant * (2.0f32).powi(-9)
            } else {
                sign * (1.0 + mant / 8.0) * (2.0f32).powi(exp - 7)
            }
        })
        .collect()
}

pub struct Nvfp4Matrix {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub global_scale: f32,
}

impl Nvfp4Matrix {
    pub fn to_f32(&self) -> Vec<f32> {
        assert_eq!(self.cols % 16, 0, "cols must be a multiple of the 16-value block");
        assert_eq!(self.packed.len(), self.rows * self.cols / 2);
        assert_eq!(self.scales.len(), self.rows * self.cols / 16);

        let scales = e4m3(&self.scales);
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let byte = self.packed[r * (self.cols / 2) + c / 2];
                let (lo, hi) = unpack_nibbles(byte);
                let value = if c % 2 == 0 { lo } else { hi };

                let scale = scales[r * (self.cols / 16) + c / 16] / self.global_scale;
                out[r * self.cols + c] = value * scale;
            }
        }
        out
    }
}

pub fn matmul(a: &Nvfp4Matrix, b: &Nvfp4Matrix) -> Vec<f32> {
    assert_eq!(a.cols, b.cols, "the contraction dimension must agree");
    let a_f = a.to_f32();
    let b_f = b.to_f32();
    let (m, k, n) = (a.rows, a.cols, b.rows);

    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (a_f[i * k + p] as f64) * (b_f[j * k + p] as f64);
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

pub struct Fp8Matrix {
    pub data: Vec<u8>,
    pub scales: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
}

impl Fp8Matrix {
    pub fn to_f32(&self) -> Vec<f32> {
        assert_eq!(self.data.len(), self.rows * self.cols);
        assert_eq!(self.scales.len(), self.rows);
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let scale = bf16_to_f32(self.scales[r]);
            for c in 0..self.cols {
                out[r * self.cols + c] = e4m3_one(self.data[r * self.cols + c]) * scale;
            }
        }
        out
    }
}

#[inline]
pub fn e4m3_one(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;

    if exp == 0 {
        sign * mant * (2.0f32).powi(-9)
    } else {
        sign * (1.0 + mant / 8.0) * (2.0f32).powi(exp - 7)
    }
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn matmul_fp8(a: &Fp8Matrix, b: &Fp8Matrix) -> Vec<f32> {
    assert_eq!(a.cols, b.cols, "the contraction dimension must agree");
    let (m, n, k) = (a.rows, b.rows, a.cols);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let sa = bf16_to_f32(a.scales[i]);
        for j in 0..n {
            let sb = bf16_to_f32(b.scales[j]);
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += e4m3_one(a.data[i * k + p]) * e4m3_one(b.data[j * k + p]);
            }
            out[i * n + j] = acc * sa * sb;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibbles_decode_low_first_with_a_sign_bit() {
        assert_eq!(unpack_nibbles(0x21), (0.5, 1.0));

        assert_eq!(unpack_nibbles(0x08), (-0.0, 0.0));
        assert_eq!(unpack_nibbles(0x88), (-0.0, -0.0));

        assert_eq!(unpack_nibbles(0x71), (0.5, 6.0));

        assert_eq!(E2M1[7], 6.0);
    }

    #[test]
    fn e4m3_decodes_exactly() {
        assert_eq!(e4m3(&[0x38])[0], 1.0);

        assert_eq!(e4m3(&[0x40])[0], 2.0);

        assert_eq!(e4m3(&[0x30])[0], 0.5);

        assert_eq!(e4m3(&[0xB8])[0], -1.0);

        assert_eq!(e4m3(&[0x00])[0], 0.0);
    }

    #[test]
    fn the_reference_matches_a_hand_computed_case() {
        let a = Nvfp4Matrix {
            packed: vec![0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            scales: vec![0x38],
            rows: 1,
            cols: 16,
            global_scale: 1.0,
        };
        let b = Nvfp4Matrix {
            packed: vec![0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            scales: vec![0x38],
            rows: 1,
            cols: 16,
            global_scale: 1.0,
        };

        assert_eq!(&a.to_f32()[..4], &[1.0, 2.0, 0.0, 0.0]);
        assert_eq!(b.to_f32()[0], 1.0);

        let c = matmul(&a, &b);
        assert_eq!(c.len(), 1);
        assert!((c[0] - 1.0).abs() < 1e-6, "got {}", c[0]);
    }

    #[test]
    fn the_global_scale_divides() {
        let mut m = Nvfp4Matrix {
            packed: vec![0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            scales: vec![0x38],
            rows: 1,
            cols: 16,
            global_scale: 1.0,
        };
        assert_eq!(m.to_f32()[0], 1.0);

        m.global_scale = 2.0;
        assert_eq!(m.to_f32()[0], 0.5);
    }

    #[test]
    fn scales_apply_per_sixteen_value_block() {
        let mut packed = vec![0x00u8; 16];
        packed[0] = 0x02;
        packed[8] = 0x02;
        let m = Nvfp4Matrix {
            packed,
            scales: vec![0x38, 0x40],
            rows: 1,
            cols: 32,
            global_scale: 1.0,
        };
        let f = m.to_f32();
        assert_eq!(f[0], 1.0, "block 0 uses its own scale");
        assert_eq!(f[16], 2.0, "block 1 uses its own scale");
    }
}