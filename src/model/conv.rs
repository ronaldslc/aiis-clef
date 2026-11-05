use crate::model::norm::silu;

pub fn causal_conv1d_silu(x: &[f32], weight: &[f32], kernel: usize, channels: usize, seq_len: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), seq_len * channels);
    debug_assert_eq!(out.len(), seq_len * channels);
    debug_assert_eq!(weight.len(), channels * kernel);

    let pad = kernel - 1;

    for t in 0..seq_len {
        for c in 0..channels {
            let taps = &weight[c * kernel..(c + 1) * kernel];
            let mut acc = 0.0f32;
            for (k, &w) in taps.iter().enumerate() {
                let src = t as isize + k as isize - pad as isize;
                if src >= 0 && (src as usize) < seq_len {
                    acc += w * x[src as usize * channels + c];
                }
            }
            out[t * channels + c] = silu(acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_tap_kernel_is_pointwise() {
        let x = vec![1.0f32, 2.0, 3.0];
        let weight = vec![2.0f32, 0.5, -1.0];
        let mut out = vec![0.0f32; 3];
        causal_conv1d_silu(&x, &weight, 1, 3, 1, &mut out);
        for c in 0..3 {
            assert!((out[c] - silu(x[c] * weight[c])).abs() < 1e-6);
        }
    }

    #[test]
    fn the_padding_is_on_the_left() {
        let x = vec![10.0f32, 20.0, 30.0];

        let weight = vec![1.0f32, 100.0];
        let mut out = vec![0.0f32; 3];
        causal_conv1d_silu(&x, &weight, 2, 1, 3, &mut out);
        assert!((out[0] - silu(100.0 * 10.0)).abs() < 1e-3, "out[0] = {}", out[0]);
        assert!((out[1] - silu(1.0 * 10.0 + 100.0 * 20.0)).abs() < 1e-3, "out[1] = {}", out[1]);
        assert!((out[2] - silu(1.0 * 20.0 + 100.0 * 30.0)).abs() < 1e-3, "out[2] = {}", out[2]);
    }

    #[test]
    fn the_output_keeps_the_sequence_length() {
        let channels = 4;
        let seq = 7;
        let x = vec![0.5f32; seq * channels];
        let weight = vec![1.0f32; channels * 4];
        let mut out = vec![0.0f32; seq * channels];
        causal_conv1d_silu(&x, &weight, 4, channels, seq, &mut out);
        assert_eq!(out.len(), seq * channels);
    }

    #[test]
    fn channels_do_not_mix() {
        let mut x = vec![0.0f32; 3 * 2];
        x[0] = 5.0;

        let weight = vec![1.0f32, 1.0, 0.0, 0.0];
        let mut out = vec![0.0f32; 3 * 2];
        causal_conv1d_silu(&x, &weight, 2, 2, 3, &mut out);
        assert!(out[0] > 0.0, "channel 0 should see the input");
        for t in 0..3 {
            assert_eq!(out[t * 2 + 1], 0.0, "channel 1 must be untouched at t={t}");
        }
    }
}
