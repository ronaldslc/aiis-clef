pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let sum_squares: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = sum_squares / x.len() as f64;
    let scale = 1.0 / (mean + eps as f64).sqrt();

    for i in 0..x.len() {
        out[i] = (x[i] as f64 * scale) as f32 * (1.0 + weight[i]);
    }
}

pub fn rms_norm_gated(x: &[f32], gate: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), gate.len());
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let sum_squares: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = sum_squares / x.len() as f64;
    let scale = 1.0 / (mean + eps as f64).sqrt();

    for i in 0..x.len() {
        let normalised = (x[i] as f64 * scale) as f32 * weight[i];
        out[i] = normalised * silu(gate[i]);
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
    }
}

pub fn l2_normalise(x: &mut [f32], width: usize, eps: f32) {
    for row in x.chunks_mut(width) {
        let sum_squares: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / (sum_squares + eps as f64).sqrt();
        for v in row.iter_mut() {
            *v = (*v as f64 * inv) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_weight_is_an_identity() {
        let x = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&x, &[0.0; 4], 0.0, &mut out);
        for v in out {
            assert!((v - 1.0).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn the_weight_is_an_offset_from_one() {
        let x = vec![3.0f32, 3.0, 3.0, 3.0];
        let mut out = vec![0.0f32; 4];
        rms_norm(&x, &[0.0; 4], 0.0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6, "got {}", out[0]);
        rms_norm(&x, &[2.0; 4], 0.0, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6, "got {} want 3", out[0]);
    }

    #[test]
    fn the_plain_and_gated_norms_use_different_conventions() {
        let x = vec![3.0f32; 4];
        let w = vec![0.5f32; 4];
        let mut plain = vec![0.0f32; 4];
        let mut gated = vec![0.0f32; 4];
        rms_norm(&x, &w, 0.0, &mut plain);
        rms_norm_gated(&x, &vec![0.0; 4], &w, 0.0, &mut gated);

        assert!((plain[0] - 1.5).abs() < 1e-6, "plain got {}", plain[0]);
        assert!(gated[0].abs() < 1e-9, "gated got {}", gated[0]);
    }

    #[test]
    fn the_gate_is_applied_after_normalising() {
        let x = vec![3.0f32; 4];
        let gate = vec![2.0f32; 4];
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm_gated(&x, &gate, &w, 0.0, &mut out);

        let expected = silu(2.0);
        assert!((out[0] - expected).abs() < 1e-5, "got {} want {expected}", out[0]);

        assert!((out[0] - 1.0).abs() > 1e-3, "this would be the gate-first answer");
    }

    #[test]
    fn silu_is_not_a_mask() {
        assert!(silu(5.0) > 1.0);
        assert!(silu(-5.0) < 0.0);
        assert!(silu(0.0).abs() < 1e-9);
    }

    #[test]
    fn softplus_stays_finite() {
        assert!((softplus(0.0) - 2.0f32.ln()).abs() < 1e-6);
        assert!(softplus(100.0).is_finite());
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
        assert!(softplus(-100.0).is_finite());
        assert!(softplus(-100.0) >= 0.0);
    }

    #[test]
    fn l2_normalise_gives_unit_rows() {
        let mut x = vec![3.0f32, 4.0, 0.0, 0.0];
        l2_normalise(&mut x, 2, 1e-6);
        assert!((x[0] - 0.6).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 0.8).abs() < 1e-6, "got {}", x[1]);
        assert!(x[2] == 0.0 && x[3] == 0.0);
    }
}
