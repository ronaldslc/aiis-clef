use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hardware {
    pub sms: usize,
    pub regs_per_sm: usize,
    pub smem_per_sm: usize,
    pub smem_per_block: usize,
    pub threads_per_sm: usize,
}

impl Hardware {
    pub const RTX_5090: Hardware = Hardware {
        sms: 170,
        regs_per_sm: 65_536,
        smem_per_sm: 102_400,
        smem_per_block: 101_376,
        threads_per_sm: 1_536,
    };

    pub const RTX_PRO_6000: Hardware = Hardware {
        sms: 188,
        regs_per_sm: 65_536,
        smem_per_sm: 102_400,
        smem_per_block: 101_376,
        threads_per_sm: 1_536,
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficients {
    pub a_us: f64,
    pub c_us_per_element: f64,
}

impl Coefficients {
    pub const FITTED_5090_LARGE_M: Coefficients = Coefficients {
        a_us: 15.0,
        c_us_per_element: 0.0183,
    };

    pub const FITTED_5090_SMALL_M: Coefficients = Coefficients {
        a_us: 9.22,
        c_us_per_element: 0.000666,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub warps_m: usize,
    pub warps_n: usize,
    pub warp_m_rows: usize,
    pub n_tile: usize,
}

impl Geometry {
    pub const SHIPPED: Geometry = Geometry { warps_m: 4, warps_n: 2, warp_m_rows: 32, n_tile: 64 };

    pub const NARROW: Geometry = Geometry { warps_m: 4, warps_n: 2, warp_m_rows: 16, n_tile: 64 };

    pub const V6: Geometry = Geometry { warps_m: 6, warps_n: 2, warp_m_rows: 32, n_tile: 64 };

    pub const V6_WIDE: Geometry = Geometry { warps_m: 4, warps_n: 2, warp_m_rows: 48, n_tile: 64 };

    pub fn mma_warps(&self) -> usize {
        self.warps_m * self.warps_n
    }

    pub fn cta_warps(&self) -> usize {
        self.mma_warps() + 4
    }

    pub fn cta_threads(&self) -> usize {
        self.cta_warps() * 32
    }

    pub fn m_tile(&self) -> usize {
        self.warps_m * self.warp_m_rows
    }

    pub fn n_block(&self) -> usize {
        self.warps_n * self.n_tile
    }

    pub fn m_atoms(&self) -> usize {
        self.warp_m_rows / 16
    }

    pub fn n_sub(&self) -> usize {
        self.n_tile / 8
    }

    pub fn accumulator_regs(&self) -> usize {
        self.m_atoms() * self.n_sub() * 4
    }

    pub fn prefetch_regs(&self) -> usize {
        let (ma, ns) = (self.m_atoms(), self.n_sub());
        8 * ma + 4 * ns + 2 * (ma + ns)
    }

    pub fn regs_available(&self, hw: &Hardware) -> usize {
        hw.regs_per_sm / self.cta_threads()
    }

    pub fn atom_rows(&self) -> usize {
        let m = self.m_tile();
        if m < 128 {
            return 128;
        }
        [128usize, 96, 64, 32].into_iter().find(|c| m % c == 0).unwrap_or(32)
    }

    pub fn penalised(&self) -> bool {
        self.atom_rows() != 128
    }

    pub fn legal_atom(&self) -> bool {
        matches!(self.atom_rows(), 128 | 96)
    }

    pub fn smem_bytes(&self, stages: usize, sub_chunks: usize, epi_n: usize, epi_bufs: usize) -> usize {
        let (m, nb) = (self.m_tile(), self.n_block());
        let a_stage = (m / 2) * 128;
        let b_stage = (nb / 2) * 128;
        let sf_atoms_m = (m / self.atom_rows()).max(1);
        let sf_a = sf_atoms_m * sub_chunks * self.atom_rows() * 4;
        let sf_b = sub_chunks * 512;
        stages * (a_stage + b_stage + sf_a + sf_b) + epi_bufs * m * epi_n * 4
    }

    pub fn k_tiles(&self, k: usize, sub_chunks: usize) -> usize {
        k.div_ceil(64 * sub_chunks)
    }

    pub fn tiles(&self, m: usize, n: usize) -> usize {
        m.div_ceil(self.m_tile()) * n.div_ceil(self.n_block())
    }

    pub fn waves(&self, m: usize, n: usize, hw: &Hardware) -> usize {
        self.tiles(m, n).div_ceil(hw.sms)
    }

    pub fn utilisation(&self, m: usize, n: usize, hw: &Hardware) -> f64 {
        let tiles = self.tiles(m, n);
        tiles as f64 / (self.waves(m, n, hw) * hw.sms) as f64
    }

    pub fn tile_time_us(&self, co: &Coefficients) -> f64 {
        co.a_us + co.c_us_per_element * (self.m_tile() * self.n_block()) as f64
    }

    pub fn makespan_us(&self, m: usize, n: usize, hw: &Hardware, co: &Coefficients) -> f64 {
        self.waves(m, n, hw) as f64 * self.tile_time_us(co)
    }

    pub fn rejection(&self, hw: &Hardware, stages: usize, sub_chunks: usize, epi_n: usize,
                     epi_bufs: usize) -> Option<String> {
        if self.cta_warps() % 4 != 0 {
            return Some(format!("{} warps is not a whole number of warpgroups", self.cta_warps()));
        }
        if self.cta_threads() > hw.threads_per_sm {
            return Some(format!("{} threads exceeds {} per SM", self.cta_threads(), hw.threads_per_sm));
        }

        if self.n_block() % 128 != 0 {
            return Some(format!("N_BLOCK {} is not a multiple of 128 (the weight layout)", self.n_block()));
        }
        let need = self.accumulator_regs() + self.prefetch_regs();
        let have = self.regs_available(hw);
        if need > have {
            return Some(format!("{need} registers needed, {have} available at {} threads",
                                self.cta_threads()));
        }
        let smem = self.smem_bytes(stages, sub_chunks, epi_n, epi_bufs);
        if smem > hw.smem_per_block {
            return Some(format!("{smem} B of shared memory exceeds {} B per block", hw.smem_per_block));
        }
        None
    }

    pub fn best_for(m: usize, n: usize, hw: &Hardware, co: &Coefficients,
                    stages: usize, sub_chunks: usize, epi_n: usize, epi_bufs: usize)
        -> Option<(Geometry, f64)> {
        let mut best: Option<(Geometry, f64)> = None;

        let incumbent = Geometry::SHIPPED;
        if incumbent.rejection(hw, stages, sub_chunks, epi_n, epi_bufs).is_none()
            && !incumbent.penalised()
        {
            best = Some((incumbent, incumbent.makespan_us(m, n, hw, co)));
        }
        for (warps_m, warps_n) in [(1, 4), (2, 2), (2, 4), (4, 1), (4, 2), (4, 3), (6, 2), (8, 1), (8, 2)] {
            let mma = warps_m * warps_n;
            if !(4..=16).contains(&mma) || (mma + 4) % 4 != 0 {
                continue;
            }
            for warp_m_rows in [16usize, 32, 48] {
                for n_tile in [16usize, 32, 64, 128] {
                    let g = Geometry { warps_m, warps_n, warp_m_rows, n_tile };
                    if !matches!(g.n_block(), 128 | 256) {
                        continue;
                    }
                    if g.rejection(hw, stages, sub_chunks, epi_n, epi_bufs).is_some() {
                        continue;
                    }

                    if g.penalised() {
                        continue;
                    }
                    let cost = g.makespan_us(m, n, hw, co);

                    if best.map_or(true, |(_, b)| cost < b * 0.95) {
                        best = Some((g, cost));
                    }
                }
            }
        }
        best
    }
}

impl fmt::Display for Geometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} ({}x{} warps of {}x{})", self.m_tile(), self.n_block(),
               self.warps_m, self.warps_n, self.warp_m_rows, self.n_tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_reproduces_the_shipped_geometry() {
        let g = Geometry::SHIPPED;
        assert_eq!(g.n_block(), 128);
        assert_eq!(g.cta_threads(), 384);
        assert_eq!(g.m_atoms(), 2);
        assert_eq!(g.n_sub(), 8);
        assert_eq!(g.accumulator_regs(), 64, "2 m16 atoms x 8 n8 sub-tiles x 4");
        assert_eq!(g.prefetch_regs(), 68);
        assert_eq!(g.regs_available(&Hardware::RTX_5090), 170);
        assert!(g.rejection(&Hardware::RTX_5090, 3, 2, 32, 2).is_none());

        let smem = g.smem_bytes(3, 2, 32, 2);
        assert!((smem as i64 - 89_216).abs() < 2_048, "model says {smem}, cubin says 89,216");
    }

    #[test]
    fn the_sm_count_is_what_breaks_version_six() {
        let v6 = Geometry::V6;
        let shipped = Geometry::SHIPPED;
        let (m, n) = (2048, 2048);

        assert_eq!(v6.tiles(m, n), 176);
        assert_eq!(shipped.tiles(m, n), 256);
        assert_eq!(v6.waves(m, n, &Hardware::RTX_PRO_6000), 1);
        assert_eq!(shipped.waves(m, n, &Hardware::RTX_PRO_6000), 2);

        assert_eq!(v6.waves(m, n, &Hardware::RTX_5090), 2);
        assert_eq!(shipped.waves(m, n, &Hardware::RTX_5090), 2);

        let co = Coefficients::FITTED_5090_LARGE_M;
        assert!(v6.makespan_us(m, n, &Hardware::RTX_5090, &co)
                > shipped.makespan_us(m, n, &Hardware::RTX_5090, &co));
    }

    #[test]
    fn a_smaller_tile_utilises_better_and_the_model_disagrees_with_measurement() {
        let (m, n) = (768, 12288);
        let (wide, narrow) = (Geometry::SHIPPED, Geometry::NARROW);
        let hw = Hardware::RTX_5090;
        assert!(narrow.utilisation(m, n, &hw) > wide.utilisation(m, n, &hw),
                "the narrow tile must utilise better: {:.1}% vs {:.1}%",
                narrow.utilisation(m, n, &hw) * 100.0, wide.utilisation(m, n, &hw) * 100.0);
        let co = Coefficients::FITTED_5090_LARGE_M;

        assert!(narrow.makespan_us(m, n, &hw, &co) < wide.makespan_us(m, n, &hw, &co),
                "the large-m coefficients are known to favour the narrow tile at m=768");
    }

    #[test]
    fn the_same_tile_costs_fifteen_times_more_at_large_m() {
        let g = Geometry::SHIPPED;
        let work_mflop = (g.m_tile() * g.n_block() * 4096 * 2) as f64 / 1e6;
        assert!((work_mflop - 134.2).abs() < 0.1);

        let small = (768usize, 4usize, 80.6f64);
        let large = (131_072usize, 579usize, 182_530.0f64);
        let rate = |(_m, waves, us): (usize, usize, f64)| waves as f64 * work_mflop / us;
        let (small_rate, large_rate) = (rate(small), rate(large));
        assert!(small_rate > 6.0, "small m runs at {small_rate:.2} TFLOP/s per SM");
        assert!(large_rate < 0.5, "large m runs at {large_rate:.2} TFLOP/s per SM");
        assert!(small_rate / large_rate > 10.0,
                "the collapse is {:.1}x and must be recorded, not rounded away",
                small_rate / large_rate);
    }

    #[test]
    fn the_register_wall_closes_the_bigger_tiles() {
        let hw = Hardware::RTX_5090;

        let v6 = Geometry::V6;
        assert_eq!(v6.regs_available(&hw), 128);
        assert!(v6.accumulator_regs() + v6.prefetch_regs() > v6.regs_available(&hw));
        assert!(v6.rejection(&hw, 2, 2, 32, 2).unwrap().contains("registers"));

        let huge = Geometry { warps_m: 8, warps_n: 2, warp_m_rows: 32, n_tile: 64 };
        assert_eq!(huge.regs_available(&hw), 102);
        assert!(huge.rejection(&hw, 2, 2, 32, 2).is_some());
    }

    #[test]
    fn the_scale_atom_is_reported_and_penalised_ones_are_skipped() {
        for (m_tile, atom) in [(128usize, 128usize), (192, 96), (160, 32), (256, 128), (64, 128)] {
            let g = Geometry { warps_m: 1, warps_n: 4, warp_m_rows: m_tile, n_tile: 32 };
            assert_eq!(g.atom_rows(), atom, "a {m_tile}-row tile's atom");
        }
        assert!(!Geometry::SHIPPED.penalised());
        assert!(Geometry::V6.penalised(), "a 192-row tile's atom is 96, not 128");
    }

    #[test]
    fn the_search_agrees_at_large_m_and_is_out_of_regime_at_small_m() {
        let hw = Hardware::RTX_5090;
        let co = Coefficients::FITTED_5090_LARGE_M;
        let (g, _) = Geometry::best_for(32768, 12288, &hw, &co, 3, 2, 32, 2)
            .expect("a legal geometry must exist");
        assert_eq!(g.m_tile(), 128, "at m=32768 the model picked {g}, and measurement says 128");
        assert_eq!(g.n_block(), 128);
        let (small, _) = Geometry::best_for(768, 12288, &hw, &co, 3, 2, 32, 2).unwrap();
        assert!(small.m_tile() < 128,
                "the large-m coefficients are known to favour a small tile at m=768, got {small}");
    }
}
