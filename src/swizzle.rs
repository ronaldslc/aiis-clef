fn round_up(value: usize, n: usize) -> usize {
    value.div_ceil(n) * n
}

pub fn sf_chunk_stride_in_force(tile_m: usize) -> usize {
    sf_atom_in_force(tile_m) * 4
}

pub fn sf_atom_in_force(tile_m: usize) -> usize {
    match std::env::var("CLEF_SF_ATOM").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(a) if a == 128 || a == 96 || a == 64 || a == 32 => a,
        _ => blk_mn_for_tile_m(tile_m),
    }
}

pub fn blk_mn_for_tile_m(tile_m: usize) -> usize {
    if tile_m < 128 {
        return 128;
    }
    for candidate in [128usize, 96, 64, 32] {
        if tile_m % candidate == 0 {
            return candidate;
        }
    }
    unreachable!("every tile we build is a multiple of 32")
}

pub fn sf_chunk_stride_for_tile_m(tile_m: usize) -> usize {
    blk_mn_for_tile_m(tile_m) * 4
}

pub fn round_up_public(value: usize, n: usize) -> usize {
    round_up(value, n)
}

pub fn swizzle(scales: &[u8], rows: usize, cols: usize, conflict_free: bool) -> Vec<u8> {
    swizzle_tiled(scales, rows, cols, 128, conflict_free)
}

pub fn swizzle_tiled(
    scales: &[u8],
    rows: usize,
    cols: usize,
    tile_m: usize,
    conflict_free: bool,
) -> Vec<u8> {
    assert_eq!(scales.len(), rows * cols, "scale matrix is not rows*cols");

    let blk_mn = blk_mn_for_tile_m(tile_m);
    let blk_sub = blk_mn / 32;
    let chunk = sf_chunk_stride_for_tile_m(tile_m);
    let rows_padded = round_up(rows, blk_mn);
    let cols_padded = round_up(cols, 4);

    let mut out = vec![0u8; (rows_padded / blk_mn) * (cols_padded / 4) * chunk];

    let k1_count = cols_padded / 4;
    for m in 0..rows {
        let m1 = m / blk_mn;
        for k in 0..cols {
            let k1 = k / 4;
            let k2 = k % 4;

            let index = if conflict_free {
                ((m1 * k1_count + k1) * blk_mn + (m % blk_mn)) * 4 + k2
            } else {
                let m_atom = (m % 32) * blk_sub + ((m / 32) % blk_sub);
                (m1 * k1_count + k1) * chunk + m_atom * 4 + k2
            };
            out[index] = scales[m * cols + k];
        }
    }
    out
}

pub fn unswizzle(swizzled: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    unswizzle_tiled(swizzled, rows, cols, 128)
}

pub fn unswizzle_conflict_free(swizzled: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    unswizzle_tiled_conflict_free(swizzled, rows, cols, 128)
}

pub fn unswizzle_tiled_conflict_free(swizzled: &[u8], rows: usize, cols: usize, tile_m: usize)
    -> Vec<u8> {
    let blk_mn = blk_mn_for_tile_m(tile_m);
    let chunk = sf_chunk_stride_for_tile_m(tile_m);
    let cols_padded = round_up(cols, 4);
    assert_eq!(swizzled.len(), swizzled_len_tiled(rows, cols, tile_m));

    let k1_count = cols_padded / 4;
    let mut out = vec![0u8; rows * cols];
    for m in 0..rows {
        let m1 = m / blk_mn;
        let m_atom = m % blk_mn;
        for k in 0..cols {
            let k1 = k / 4;
            let k2 = k % 4;
            let index = (m1 * k1_count + k1) * chunk + m_atom * 4 + k2;
            out[m * cols + k] = swizzled[index];
        }
    }
    out
}

pub fn unswizzle_tiled(swizzled: &[u8], rows: usize, cols: usize, tile_m: usize) -> Vec<u8> {
    let blk_mn = blk_mn_for_tile_m(tile_m);
    let blk_sub = blk_mn / 32;
    let chunk = sf_chunk_stride_for_tile_m(tile_m);
    let cols_padded = round_up(cols, 4);
    assert_eq!(swizzled.len(), swizzled_len_tiled(rows, cols, tile_m));

    let k1_count = cols_padded / 4;
    let mut out = vec![0u8; rows * cols];
    for m in 0..rows {
        let m1 = m / blk_mn;
        let m_atom = (m % 32) * blk_sub + ((m / 32) % blk_sub);
        for k in 0..cols {
            let k1 = k / 4;
            let k2 = k % 4;
            let index = (m1 * k1_count + k1) * chunk + m_atom * 4 + k2;
            out[m * cols + k] = swizzled[index];
        }
    }
    out
}

pub fn swizzled_len(rows: usize, cols: usize) -> usize {
    swizzled_len_tiled(rows, cols, 128)
}

pub fn swizzled_len_tiled(rows: usize, cols: usize, tile_m: usize) -> usize {
    let blk_mn = blk_mn_for_tile_m(tile_m);
    let rows_padded = round_up(rows, blk_mn);
    let cols_padded = round_up(cols, 4);
    debug_assert_eq!(
        (rows_padded / blk_mn) * (cols_padded / 4) * (blk_mn * 4),
        rows_padded * cols_padded
    );
    rows_padded * cols_padded
}

#[cfg(test)]
mod index_tests {
    use super::*;

    #[test]
    fn the_gemm_index_matches_the_swizzle() {
        for (rows, cols) in [(128usize, 4usize), (256, 8), (12288, 256), (4096, 256), (4096, 768), (12288, 768)] {
            let compact: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();
            let swizzled = swizzle(&compact, rows, cols, false);
            let mut checked = 0;
            for row in (0..rows).step_by(rows / 8 + 1) {
                for block in (0..cols).step_by(cols / 8 + 1) {
                    let idx = swizzled_index_of(row, block, cols, 128, false);
                    assert!(idx < swizzled.len(), "index {idx} out of range for {rows}x{cols}");
                    assert_eq!(
                        swizzled[idx], compact[row * cols + block],
                        "{rows}x{cols}: the GEMM would read {:#04x} at ({row}, {block}), but the \
                         swizzle wrote {:#04x}",
                        swizzled[idx], compact[row * cols + block]
                    );
                    checked += 1;
                }
            }
            assert!(checked >= 16, "the sampling covered too little");
            eprintln!("  {rows}x{cols}: {checked} positions agree");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn the_round_trip_is_lossless() {
        for (rows, cols) in [(128, 4), (256, 8), (4096, 256), (12288, 256), (130, 6)] {
            let src = ramp(rows * cols);
            let sw = swizzle(&src, rows, cols, false);
            assert_eq!(sw.len(), swizzled_len(rows, cols), "size for {rows}x{cols}");
            let back = unswizzle(&sw, rows, cols);
            assert_eq!(back, src, "round trip for {rows}x{cols}");
        }
    }

    #[test]
    fn the_first_tile_has_the_expected_arrangement() {
        let rows = 128usize;
        let cols = 4usize;
        let src: Vec<u8> = (0..rows * cols).map(|i| i as u8).collect();
        let sw = swizzle(&src, rows, cols, false);

        for m3 in 0..32usize {
            for m2 in 0..4usize {
                for k2 in 0..4usize {
                    let index = (0 * 1 + 0) * 512 + m3 * 16 + m2 * 4 + k2;

                    let expected = ((m2 * 32 + m3) * 4 + 0 * 4 + k2) % 256;
                    assert_eq!(
                        sw[index] as usize, expected,
                        "tile byte at m3={m3} m2={m2} k2={k2}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_conflict_free_tile_has_the_expected_arrangement() {
        let rows = 128usize;
        let cols = 4usize;
        let src: Vec<u8> = (0..rows * cols).map(|i| i as u8).collect();
        let sw = swizzle(&src, rows, cols, true);

        for m in 0..128usize {
            for k2 in 0..4usize {
                let index = m * 4 + k2;
                let expected = (m * 4 + k2) % 256;
                assert_eq!(sw[index] as usize, expected, "cf byte at m={m} k2={k2}");
            }
        }

        assert_ne!(sw[4] as usize, sw[16] as usize, "cf and original must differ at row 1");
    }

    #[test]
    fn padding_is_zero_and_leaves_values_intact() {
        let (rows, cols) = (129usize, 5usize);
        let src = ramp(rows * cols);
        let sw = swizzle(&src, rows, cols, false);
        assert_eq!(sw.len(), 256 * 8, "padded to 256 x 8");

        let back = unswizzle(&sw, rows, cols);
        assert_eq!(back, src, "the real values survive padding");

        let real: std::collections::HashSet<usize> = (0..rows)
            .flat_map(|m| {
                let m1 = m / 128;
                let m2 = (m % 128) / 32;
                let m3 = m % 32;
                (0..cols).map(move |k| ((m1 * 2 + k / 4) * 32 + m3) * 16 + m2 * 4 + k % 4)
            })
            .collect();
        for (i, &b) in sw.iter().enumerate() {
            if !real.contains(&i) {
                assert_eq!(b, 0, "padding at {i} should be zero");
            }
        }
    }

    #[test]
    fn an_aligned_matrix_is_not_padded() {
        assert_eq!(swizzled_len(128, 4), 128 * 4);
        assert_eq!(swizzled_len(4096, 256), 4096 * 256);
        assert_eq!(swizzled_len(12288, 256), 12288 * 256);

        assert_eq!(swizzled_len(129, 256), 256 * 256);
    }

    #[test]
    fn the_generalised_swizzle_reproduces_the_128_row_layout() {
        for (rows, cols) in [(128usize, 4usize), (256, 8), (4096, 256), (12288, 256), (130, 6)] {
            let compact = ramp(rows * cols);
            let got = swizzle_tiled(&compact, rows, cols, 128, false);

            let rows_padded = round_up(rows, 128);
            let cols_padded = round_up(cols, 4);
            let k1_count = cols_padded / 4;
            let mut want = vec![0u8; rows_padded * cols_padded];
            for m in 0..rows {
                for k in 0..cols {
                    let m1 = m / 128;
                    let m2 = (m % 128) / 32;
                    let m3 = m % 32;
                    let index = ((m1 * k1_count + k / 4) * 32 + m3) * 16 + m2 * 4 + (k % 4);
                    want[index] = compact[m * cols + k];
                }
            }
            assert_eq!(got, want, "{rows}x{cols}: the generalised layout moved a byte");
        }
        eprintln!("  the 128-row layout is unchanged across 5 shapes");
    }

    #[test]
    fn the_96_row_atom_matches_the_reference_strides() {
        assert_eq!(blk_mn_for_tile_m(192), 96, "six warps x 16 rows");
        assert_eq!(blk_mn_for_tile_m(128), 128);

        assert_eq!(blk_mn_for_tile_m(64), 128, "a 64-row tile slices the 128-row atom");

        assert_eq!(sf_chunk_stride_for_tile_m(192), 384, "blk_elems = 96*4");
        assert_eq!(sf_chunk_stride_for_tile_m(128), 512);
        assert_eq!(sf_chunk_stride_for_tile_m(64), 512, "the 64-row tile shares the 128-row atom");

        let stride_of = |row: usize| -> usize {
            let blk_sub = 3;
            (row % 32) * blk_sub * 4 + ((row / 32) % blk_sub) * 4
        };
        assert_eq!(stride_of(0), 0);
        assert_eq!(stride_of(32), 4, "row 32 starts the second band");
        assert_eq!(stride_of(64), 8, "row 64 starts the third");

        let k1_count = 256 / 4;
        assert_eq!(swizzled_index_of(0, 0, 256, 192, false), 0, "the first atom starts at zero");
        assert_eq!(swizzled_index_of(0, 4, 256, 192, false), 384, "k1 = 1 is one atom on");
        assert_eq!(
            swizzled_index_of(96, 0, 256, 192, false),
            k1_count * 384,
            "the second atom is one k-width on"
        );
        assert_eq!(
            swizzled_len_tiled(768, 256, 192),
            768 * 256,
            "a smaller atom packs the same bytes differently, it does not need more room"
        );
    }
}

pub fn swizzled_index_of(
    row: usize,
    block: usize,
    blocks: usize,
    tile_m: usize,
    conflict_free: bool,
) -> usize {
    let blk_mn = blk_mn_for_tile_m(tile_m);
    let blk_sub = blk_mn / 32;
    let chunk = sf_chunk_stride_for_tile_m(tile_m);
    let k1_count = blocks / 4;
    let m1 = row / blk_mn;
    let k1 = block / 4;
    let k2 = block % 4;
    if conflict_free {
        (m1 * k1_count + k1) * chunk + (row % blk_mn) * 4 + k2
    } else {
        let m_atom = (row % 32) * blk_sub + ((row / 32) % blk_sub);
        (m1 * k1_count + k1) * chunk + m_atom * 4 + k2
    }
}

pub fn swizzled_index_base(row: usize, blocks: usize, tile_m: usize, conflict_free: bool) -> usize {
    let blk_mn = blk_mn_for_tile_m(tile_m);
    let blk_sub = blk_mn / 32;
    let chunk = sf_chunk_stride_for_tile_m(tile_m);
    let k1_count = blocks / 4;
    if conflict_free {
        (row / blk_mn) * k1_count * chunk + (row % blk_mn) * 4
    } else {
        let m_atom = (row % 32) * blk_sub + ((row / 32) % blk_sub);
        (row / blk_mn) * k1_count * chunk + m_atom * 4
    }
}

pub fn sf_word_expected(
    row: usize,
    ks: usize,
    blocks: usize,
    tile_m: usize,
    conflict_free: bool,
) -> usize {
    swizzled_index_base(row, blocks, tile_m, conflict_free) / 4
        + ks * (sf_chunk_stride_for_tile_m(tile_m) / 4)
}

pub const SWIZZLE_CHUNK_STRIDE: usize = 512;

#[cfg(test)]
mod affine_tests {
    use super::*;

    #[test]
    fn the_affine_index_matches_the_full_one() {
        for blocks in [16usize, 64, 256] {
            for row in 0..256 {
                let base = swizzled_index_base(row, blocks, 128, false);
                for chunk in 0..blocks / 4 {
                    for kb in 0..4 {
                        let block = chunk * 4 + kb;
                        let want = swizzled_index_of(row, block, blocks, 128, false);
                        let got = base + chunk * SWIZZLE_CHUNK_STRIDE + kb;
                        assert_eq!(
                            got, want,
                            "row {row}, chunk {chunk}, kb {kb}, blocks {blocks}: affine {got} vs \
                             full {want}"
                        );
                    }
                }
            }
        }
    }
}
