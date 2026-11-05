use crate::model::norm::l2_normalise;

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule(
    query: &mut [f32],
    key: &mut [f32],
    value: &[f32],
    decay: &[f32],
    beta: &[f32],
    seq_len: usize,
    num_heads: usize,
    k_dim: usize,
    v_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(query.len(), seq_len * num_heads * k_dim);
    debug_assert_eq!(key.len(), seq_len * num_heads * k_dim);
    debug_assert_eq!(value.len(), seq_len * num_heads * v_dim);
    debug_assert_eq!(decay.len(), seq_len * num_heads);
    debug_assert_eq!(beta.len(), seq_len * num_heads);
    debug_assert_eq!(out.len(), seq_len * num_heads * v_dim);
    l2_normalise(query, k_dim, 1e-6);
    l2_normalise(key, k_dim, 1e-6);

    let q_scale = 1.0 / (k_dim as f32).sqrt();
    for q in query.iter_mut() {
        *q *= q_scale;
    }

    let mut state = vec![0.0f32; num_heads * k_dim * v_dim];

    for t in 0..seq_len {
        for h in 0..num_heads {
            let q_t = &query[(t * num_heads + h) * k_dim..(t * num_heads + h + 1) * k_dim];
            let k_t = &key[(t * num_heads + h) * k_dim..(t * num_heads + h + 1) * k_dim];
            let v_t = &value[(t * num_heads + h) * v_dim..(t * num_heads + h + 1) * v_dim];
            let decay_t = decay[t * num_heads + h].exp();
            let beta_t = beta[t * num_heads + h];
            let s = &mut state[h * k_dim * v_dim..(h + 1) * k_dim * v_dim];

            for x in s.iter_mut() {
                *x *= decay_t;
            }

            let mut kv = vec![0.0f32; v_dim];
            for (i, &k_i) in k_t.iter().enumerate() {
                if k_i == 0.0 {
                    continue;
                }
                let row = &s[i * v_dim..(i + 1) * v_dim];
                for (j, &sv) in row.iter().enumerate() {
                    kv[j] += k_i * sv;
                }
            }
            for (j, &v_j) in v_t.iter().enumerate() {
                let delta = (v_j - kv[j]) * beta_t;
                if delta == 0.0 {
                    continue;
                }
                for (i, &k_i) in k_t.iter().enumerate() {
                    s[i * v_dim + j] += k_i * delta;
                }
            }

            let o = &mut out[(t * num_heads + h) * v_dim..(t * num_heads + h + 1) * v_dim];
            for (i, &q_i) in q_t.iter().enumerate() {
                if q_i == 0.0 {
                    continue;
                }
                let row = &s[i * v_dim..(i + 1) * v_dim];
                for (j, &sv) in row.iter().enumerate() {
                    o[j] += q_i * sv;
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmaPrecision {
    F32,
    Bf16,
    Tf32,
    Fp8,
    Fp8HiLo,
    Nvfp4,
    Nvfp4HiLo,
    Bf16HiLo,
}

#[inline]
fn to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

#[inline]
fn to_tf32(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 13) & 1;
    f32::from_bits(bits.wrapping_add(0x0fff + lsb) & 0xffff_e000)
}

#[inline]
fn to_e4m3(x: f32) -> f32 {
    let sign = x.to_bits() & 0x8000_0000;
    let a = x.abs();
    if a >= 448.0 {
        return f32::from_bits(sign | 0x43e0_0000);
    }

    let abits = a.to_bits();
    let lsb = (abits >> 20) & 1;
    f32::from_bits(sign | (abits.wrapping_add(0x7_ffff + lsb) & 0xffff_f000))
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
fn e2m1(x: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        return 0.0;
    }
    let a = (x.abs() / scale).min(6.0);
    let mut best = 0.0f32;
    let mut bestd = f32::MAX;
    for m in E2M1 {
        let d = (a - m).abs();
        if d < bestd {
            bestd = d;
            best = m;
        }
    }
    let r = best * scale;
    if x < 0.0 {
        -r
    } else {
        r
    }
}

#[inline]
fn block_scale(a: &[f32], stride: usize, blk: usize, end: usize) -> f32 {
    let mut amax = 0.0f32;
    for p in blk..end {
        amax = amax.max(a[p * stride].abs());
    }
    to_e4m3(amax / 6.0)
}

fn dot_mma(a: &[f32], as_: usize, b: &[f32], bs: usize, n: usize, prec: MmaPrecision) -> f32 {
    match prec {
        MmaPrecision::Nvfp4 | MmaPrecision::Nvfp4HiLo => {
            let hi_lo = prec == MmaPrecision::Nvfp4HiLo;
            let mut s = 0.0f32;
            let mut blk = 0;
            while blk < n {
                let end = (blk + 16).min(n);
                let sa = block_scale(a, as_, blk, end);
                let sb = block_scale(b, bs, blk, end);
                if !hi_lo {
                    for p in blk..end {
                        s += e2m1(a[p * as_], sa) * e2m1(b[p * bs], sb);
                    }
                } else {
                    let (mut ra, mut rb) = (0.0f32, 0.0f32);
                    for p in blk..end {
                        let (x, y) = (a[p * as_], b[p * bs]);
                        ra = ra.max((x - e2m1(x, sa)).abs());
                        rb = rb.max((y - e2m1(y, sb)).abs());
                    }
                    let (ra, rb) = (to_e4m3(ra / 6.0), to_e4m3(rb / 6.0));
                    for p in blk..end {
                        let (x, y) = (a[p * as_], b[p * bs]);
                        let (xh, yh) = (e2m1(x, sa), e2m1(y, sb));
                        let (xl, yl) = (e2m1(x - xh, ra), e2m1(y - yh, rb));
                        s += xh * yh + (xh * yl + xl * yh);
                    }
                }
                blk = end;
            }
            s
        }
        _ => (0..n).map(|p| mm(a[p * as_], b[p * bs], prec)).sum(),
    }
}

#[inline]
fn mm(a: f32, b: f32, p: MmaPrecision) -> f32 {
    match p {
        MmaPrecision::F32 => a * b,
        MmaPrecision::Bf16 => to_bf16(a) * to_bf16(b),
        MmaPrecision::Tf32 => to_tf32(a) * to_tf32(b),
        MmaPrecision::Fp8 => to_e4m3(a) * to_e4m3(b),
        MmaPrecision::Fp8HiLo => {
            let (ah, bh) = (to_e4m3(a), to_e4m3(b));
            let (al, bl) = (to_e4m3(a - ah), to_e4m3(b - bh));
            ah * bh + (ah * bl + al * bh)
        }
        MmaPrecision::Nvfp4 | MmaPrecision::Nvfp4HiLo => {
            unreachable!("e2m1 needs its block scale; use dot_mma")
        }
        MmaPrecision::Bf16HiLo => {
            let (ah, bh) = (to_bf16(a), to_bf16(b));
            let (al, bl) = (to_bf16(a - ah), to_bf16(b - bh));
            ah * bh + (ah * bl + al * bh)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolveMethod {
    Forward,
    Neumann,
}

pub fn neumann_inverse(l: &[f32], n: usize, prec: MmaPrecision) -> Vec<f32> {
    debug_assert_eq!(l.len(), n * n);

    let mut t = vec![0.0f32; n * n];
    for i in 0..n {
        t[i * n + i] = 1.0;
        for j in 0..i {
            t[i * n + j] = -l[i * n + j];
        }
    }
    let mut p = l.to_vec();
    let mut k = 1usize;
    while (1usize << k) < n {
        p = matmul(&p, &p, n, prec);
        let tp = matmul(&t, &p, n, prec);
        for e in 0..n * n {
            t[e] += tp[e];
        }
        k += 1;
    }
    t
}

fn matmul(a: &[f32], b: &[f32], n: usize, prec: MmaPrecision) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = dot_mma(&a[i * n..], 1, &b[j..], n, n, prec);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_gated_delta_rule(
    query: &mut [f32],
    key: &mut [f32],
    value: &[f32],
    decay: &[f32],
    beta: &[f32],
    seq_len: usize,
    num_heads: usize,
    k_dim: usize,
    v_dim: usize,
    chunk: usize,
    prec: MmaPrecision,
    solve: SolveMethod,
    out: &mut [f32],
) {
    assert!(chunk > 0, "chunk must be positive");
    debug_assert_eq!(out.len(), seq_len * num_heads * v_dim);
    l2_normalise(query, k_dim, 1e-6);
    l2_normalise(key, k_dim, 1e-6);
    let q_scale = 1.0 / (k_dim as f32).sqrt();
    for q in query.iter_mut() {
        *q *= q_scale;
    }

    let r = k_dim + v_dim;

    for h in 0..num_heads {
        let mut state = vec![0.0f32; k_dim * v_dim];

        let mut c0 = 0usize;
        while c0 < seq_len {
            let n = chunk.min(seq_len - c0);
            let row = |i: usize, w: usize| ((c0 + i) * num_heads + h) * w;

            let mut gcum = vec![0.0f32; n];
            let mut acc = 0.0f32;
            for i in 0..n {
                acc += decay[(c0 + i) * num_heads + h];
                gcum[i] = acc;
            }

            let mut l = vec![0.0f32; n * n];
            for i in 0..n {
                let bi = beta[(c0 + i) * num_heads + h];
                let ki = row(i, k_dim);
                for j in 0..i {
                    let kj = row(j, k_dim);
                    let dot = dot_mma(&key[kj..], 1, &key[ki..], 1, k_dim, prec);
                    l[i * n + j] = bi * (gcum[i] - gcum[j]).exp() * dot;
                }
            }

            let mut rhs = vec![0.0f32; n * r];
            for i in 0..n {
                let bi = beta[(c0 + i) * num_heads + h];
                let ai = gcum[i].exp();
                let (ki, vi) = (row(i, k_dim), row(i, v_dim));
                for c in 0..r {
                    rhs[i * r + c] = if c < k_dim {
                        bi * ai * key[ki + c]
                    } else {
                        bi * value[vi + c - k_dim]
                    };
                }
            }

            let mut x = vec![0.0f32; n * r];
            match solve {
                SolveMethod::Forward => {
                    for i in 0..n {
                        for c in 0..r {
                            let mut s = rhs[i * r + c];
                            for j in 0..i {
                                s -= l[i * n + j] * x[j * r + c];
                            }
                            x[i * r + c] = s;
                        }
                    }
                }
                SolveMethod::Neumann => {
                    let t = neumann_inverse(&l, n, prec);
                    for i in 0..n {
                        for c in 0..r {
                            x[i * r + c] = dot_mma(&t[i * n..], 1, &rhs[c..], r, n, prec);
                        }
                    }
                }
            }
            let (_w, u0) = (|i: usize, p: usize| x[i * r + p], |i: usize, j: usize| x[i * r + k_dim + j]);

            let mut u = vec![0.0f32; n * v_dim];
            for i in 0..n {
                for j in 0..v_dim {
                    u[i * v_dim + j] =
                        u0(i, j) - dot_mma(&x[i * r..], 1, &state[j..], v_dim, k_dim, prec);
                }
            }

            for i in 0..n {
                let (qi, oi) = (row(i, k_dim), row(i, v_dim));
                let ai = gcum[i].exp();
                for j in 0..v_dim {
                    out[oi + j] = ai * dot_mma(&query[qi..], 1, &state[j..], v_dim, k_dim, prec);
                }

                let mut coefs = vec![0.0f32; i + 1];
                for (jp, coef) in coefs.iter_mut().enumerate() {
                    let kjp = row(jp, k_dim);
                    let dot = dot_mma(&key[kjp..], 1, &query[qi..], 1, k_dim, prec);
                    *coef = (gcum[i] - gcum[jp]).exp() * dot;
                }
                for j in 0..v_dim {
                    out[oi + j] += dot_mma(&coefs, 1, &u[j..], v_dim, i + 1, prec);
                }
            }

            let an = gcum[n - 1].exp();
            let mut next = vec![0.0f32; k_dim * v_dim];

            for j in 0..v_dim {
                let token_stride = num_heads * k_dim;
                let mut dd = vec![0.0f32; n];
                for (i, d) in dd.iter_mut().enumerate() {
                    *d = (gcum[n - 1] - gcum[i]).exp() * u[i * v_dim + j];
                }
                for p in 0..k_dim {
                    next[p * v_dim + j] = an * state[p * v_dim + j]
                        + dot_mma(&key[row(0, k_dim) + p..], token_stride, &dd, 1, n, prec);
                }
            }
            state = next;

            c0 += n;
        }
    }
}

pub fn repeat_heads(x: &[f32], seq_len: usize, heads: usize, dim: usize, ratio: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), seq_len * heads * dim);
    debug_assert_eq!(out.len(), seq_len * heads * ratio * dim);

    for t in 0..seq_len {
        for h in 0..heads {
            let src = &x[(t * heads + h) * dim..(t * heads + h + 1) * dim];
            for r in 0..ratio {
                let dst_head = h * ratio + r;
                let dst = (t * heads * ratio + dst_head) * dim;
                out[dst..dst + dim].copy_from_slice(src);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_beta_writes_nothing() {
        let (s, h, k, v) = (4, 2, 3, 5);
        let mut q = vec![0.5f32; s * h * k];
        let mut kk = vec![0.5f32; s * h * k];
        let val = vec![1.0f32; s * h * v];
        let decay = vec![0.0f32; s * h];
        let beta = vec![0.0f32; s * h];
        let mut out = vec![0.0f32; s * h * v];
        gated_delta_rule(&mut q, &mut kk, &val, &decay, &beta, s, h, k, v, &mut out);
        assert!(out.iter().all(|&x| x == 0.0), "nothing should have been written");
    }

    #[test]
    fn a_repeated_key_produces_no_further_update() {
        let (s, h, k, v) = (2, 1, 2, 2);
        let kvec = [1.0f32, 0.0];
        let vvec = [3.0f32, 4.0];
        let mut q = vec![0.0f32; s * h * k];
        let mut key = vec![0.0f32; s * h * k];
        let mut value = vec![0.0f32; s * h * v];
        for t in 0..s {
            q[t * k..t * k + k].copy_from_slice(&kvec);
            key[t * k..t * k + k].copy_from_slice(&kvec);
            value[t * v..t * v + v].copy_from_slice(&vvec);
        }
        let decay = vec![0.0f32; s * h];
        let beta = vec![1.0f32; s * h];
        let mut out = vec![0.0f32; s * h * v];
        gated_delta_rule(&mut q, &mut key, &value, &decay, &beta, s, h, k, v, &mut out);

        for j in 0..v {
            assert!((out[j] - out[v + j]).abs() < 1e-5, "token 1 moved: {} vs {}", out[j], out[v + j]);
        }

        let q_scale = 1.0f32 / (k as f32).sqrt();
        assert!((out[0] - 3.0 * q_scale).abs() < 1e-4, "got {} want {}", out[0], 3.0 * q_scale);
        assert!((out[1] - 4.0 * q_scale).abs() < 1e-4, "got {} want {}", out[1], 4.0 * q_scale);
    }

    #[test]
    fn full_decay_erases_the_state() {
        let (s, h, k, v) = (1, 1, 2, 2);
        let mut q = vec![1.0f32, 0.0];
        let mut key = vec![1.0f32, 0.0];
        let value = vec![1.0f32, 1.0];
        let decay = vec![-100.0f32];
        let beta = vec![1.0f32];
        let mut out = vec![0.0f32; v];
        gated_delta_rule(&mut q, &mut key, &value, &decay, &beta, s, h, k, v, &mut out);

        assert!(out[0].is_finite());
    }

    #[test]
    fn heads_are_interleaved_not_concatenated() {
        let x = vec![10.0f32, 20.0];
        let mut out = vec![0.0f32; 4];
        repeat_heads(&x, 1, 2, 1, 2, &mut out);
        assert_eq!(out, vec![10.0, 10.0, 20.0, 20.0]);
    }

    #[test]
    fn the_neumann_inverse_is_the_exact_inverse_of_i_plus_l() {
        for n in [1usize, 2, 3, 16, 17, 64] {
            let mut seed = 0x1234_5678u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };

            let mut l = vec![0.0f32; n * n];
            for i in 0..n {
                let b = 0.2 + 0.6 * next().abs();
                for j in 0..i {
                    l[i * n + j] = b * (-0.1 * (i - j) as f32).exp() * next();
                }
            }

            let mul = |a: &[f32], b: &[f32]| -> Vec<f32> { matmul(a, b, n, MmaPrecision::F32) };

            let mut power = l.clone();
            for _ in 1..n {
                power = mul(&power, &l);
            }
            let nilpotent = power.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(nilpotent == 0.0, "n={n}: L^{n} is not zero, max |.| = {nilpotent:e}");

            let t = neumann_inverse(&l, n, MmaPrecision::F32);
            let mut worst = 0.0f32;
            for i in 0..n {
                for j in 0..n {
                    let mut s = t[i * n + j];
                    for p in 0..i {
                        s += l[i * n + p] * t[p * n + j];
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    worst = worst.max((s - want).abs());
                }
            }
            eprintln!("  neumann n={n}: max |(I+L)T - I| = {worst:.3e}");
            assert!(worst <= 1e-4, "n={n}: (I+L)T is not I, worst {worst:e}");
        }
    }

    #[test]
    fn the_chunked_form_matches_the_recurrent_one() {
        for (seq, heads, k_dim, v_dim, chunk) in [
            (64usize, 2usize, 128usize, 128usize, 64usize),
            (17, 2, 128, 128, 64),
            (130, 2, 128, 128, 64),
            (200, 2, 128, 128, 64),
        ] {
            let mut seed = 0x9e37_79b9u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };
            let q: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let k: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..seq * heads * v_dim).map(|_| next()).collect();
            let decay: Vec<f32> = (0..seq * heads).map(|_| -0.002 - 0.02 * next().abs()).collect();
            let beta: Vec<f32> = (0..seq * heads).map(|_| 0.2 + 0.6 * next().abs()).collect();

            let mut oq = q.clone();
            let mut ok = k.clone();
            let mut want = vec![0.0f32; seq * heads * v_dim];
            gated_delta_rule(&mut oq, &mut ok, &v, &decay, &beta, seq, heads, k_dim, v_dim, &mut want);

            for solve in [SolveMethod::Forward, SolveMethod::Neumann] {
                let mut cq = q.clone();
                let mut ck = k.clone();
                let mut got = vec![0.0f32; seq * heads * v_dim];
                chunked_gated_delta_rule(
                    &mut cq, &mut ck, &v, &decay, &beta, seq, heads, k_dim, v_dim, chunk,
                    MmaPrecision::F32, solve, &mut got,
                );

                let mut worst = (0usize, 0.0f32);
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    let d = (g - w).abs();
                    if d > worst.1 {
                        worst = (i, d);
                    }
                }
                let scale = want.iter().map(|x| x.abs()).sum::<f32>() / want.len() as f32;

                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "chunked form ({solve:?}) seq={seq} chunk={chunk}: a non-finite value"
                );
                eprintln!(
                    "  chunked {solve:?} seq={seq} heads={heads} k={k_dim} v={v_dim} chunk={chunk}: \
                     max |diff| = {:.3e} at {} (got {:.6}, want {:.6}), mean |x| = {:.3e}",
                    worst.1, worst.0, got[worst.0], want[worst.0], scale
                );
                assert!(
                    worst.1 <= 1e-4 * scale.max(1.0),
                    "chunked form ({solve:?}) disagrees at {}: got {} want {}",
                    worst.0,
                    got[worst.0],
                    want[worst.0]
                );
            }
        }
    }

    #[test]
    fn the_chunked_precision_ladder() {
        for (seq, heads, rate, chunk) in [
            (200usize, 4usize, 3.0f32, 16usize),
            (200, 4, 3.0, 64),
            (512, 4, 3.0, 16),
            (512, 4, 3.0, 64),
        ] {
            let (k_dim, v_dim) = (128usize, 128usize);
            let mut seed = 0x5bd1_e995u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };
            let q: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let k: Vec<f32> = (0..seq * heads * k_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..seq * heads * v_dim).map(|_| next()).collect();
            let decay: Vec<f32> = (0..seq * heads).map(|_| -(0.01 + rate * next().abs())).collect();
            let beta: Vec<f32> = (0..seq * heads).map(|_| 0.2 + 0.6 * next().abs()).collect();

            let mut oq = q.clone();
            let mut ok = k.clone();
            let mut want = vec![0.0f32; seq * heads * v_dim];
            gated_delta_rule(&mut oq, &mut ok, &v, &decay, &beta, seq, heads, k_dim, v_dim, &mut want);

            let scale = want.iter().map(|x| x.abs()).sum::<f32>() / want.len() as f32;
            let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            eprintln!(
                "  seq={seq} heads={heads} rate={rate} chunk={chunk}: mean |x| = {scale:.3e}, \
                 peak |x| = {peak:.3e}"
            );

            const LADDER: [MmaPrecision; 8] = [
                MmaPrecision::F32,
                MmaPrecision::Tf32,
                MmaPrecision::Bf16,
                MmaPrecision::Bf16HiLo,
                MmaPrecision::Fp8,
                MmaPrecision::Fp8HiLo,
                MmaPrecision::Nvfp4,
                MmaPrecision::Nvfp4HiLo,
            ];

            const NEUMANN_LADDER: [MmaPrecision; 2] = [MmaPrecision::F32, MmaPrecision::Bf16HiLo];
            for (solve, prec) in [
                (SolveMethod::Forward, &LADDER[..]),
                (SolveMethod::Neumann, &NEUMANN_LADDER[..]),
            ]
            .into_iter()
            .flat_map(|(s, ps)| ps.iter().map(move |&p| (s, p)))
            {
                let mut cq = q.clone();
                let mut ck = k.clone();
                let mut got = vec![0.0f32; seq * heads * v_dim];
                chunked_gated_delta_rule(
                    &mut cq, &mut ck, &v, &decay, &beta, seq, heads, k_dim, v_dim, chunk, prec,
                    solve, &mut got,
                );
                let (mut worst, mut worst_rel) = (0.0f32, 0.0f32);
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "{prec:?} seq={seq} chunk={chunk}: a non-finite value"
                );
                for (g, w) in got.iter().zip(want.iter()) {
                    let d = (g - w).abs();
                    worst = worst.max(d);

                    if w.abs() > 1e-3 * scale {
                        worst_rel = worst_rel.max(d / w.abs());
                    }
                }
                eprintln!(
                    "    {solve:?} {prec:?}: max |diff| = {worst:.3e} ({:.2e} of mean |x|), max rel = {worst_rel:.2e}",
                    worst / scale
                );
            }
        }
    }
}
