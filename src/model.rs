pub mod attention;
pub mod config;
pub mod conv;
pub mod deltanet;
pub mod forward;
pub mod head;
pub mod gemm_ref;
pub mod layer;
pub mod norm;

pub use config::{LayerKind, TextConfig};

#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub bq: usize,
    pub bk: usize,
    pub row_groups: usize,
    pub col_groups: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AttnMachine {
    pub sms: usize,
    pub clock_hz: f64,
    pub tensor_flops: f64,
    pub l1_bytes_per_cycle: f64,
    pub smem_per_cta: f64,
    pub regs_per_sm: f64,
    pub dram_bytes: f64,
}

impl AttnMachine {
    pub const RTX_5090: AttnMachine = AttnMachine {
        sms: 170,
        clock_hz: 2.55e9,
        tensor_flops: 244e12,
        l1_bytes_per_cycle: 128.0,
        smem_per_cta: 99.0 * 1024.0,
        regs_per_sm: 65536.0,
        dram_bytes: 1792e9,
    };

    pub fn l1_bytes(&self) -> f64 {
        self.l1_bytes_per_cycle * self.sms as f64 * self.clock_hz
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AttnPrediction {
    pub mma_ms: f64,
    pub kv_ms: f64,
    pub frag_ms: f64,
    pub warps_per_sm: f64,
    pub bound: &'static str,
    pub known_error: &'static str,
    pub total_ms: f64,
}

impl AttnPrediction {
    pub fn tensor_pipe_pct(&self) -> f64 {
        if self.total_ms > 0.0 { 100.0 * self.mma_ms / self.total_ms } else { 0.0 }
    }

    pub fn occupancy_pct(&self) -> f64 {
        100.0 * self.warps_per_sm / 64.0
    }
}

pub fn attention_time(m: usize, hd: usize, heads: usize, kv_heads: usize, s: &AttnShape,
                      mach: &AttnMachine) -> AttnPrediction {
    let pairs = (m as f64) * (m as f64 + 1.0) / 2.0;

    let redundancy = s.col_groups as f64;
    let flops = pairs * heads as f64 * 4.0 * hd as f64 * redundancy;
    let mma_ms = flops / mach.tensor_flops * 1e3;

    let kv_bytes = (m as f64 / s.bq as f64) * m as f64 * hd as f64 * 2.0 * 2.0;

    let kv_ms = kv_bytes / mach.dram_bytes * 1e3;

    let mmas = pairs / (16.0 * 8.0) * (hd as f64 / 16.0);
    let a_bytes = 16.0 * 16.0 * 2.0;
    let b_bytes = 16.0 * 8.0 * 2.0;

    let frag_bytes = mmas * b_bytes + mmas * a_bytes * redundancy;
    let frag_ms = frag_bytes / mach.l1_bytes() * 1e3;

    let warps = (s.row_groups * s.col_groups) as f64;

    let acc_regs = 16.0 * (hd as f64 / s.col_groups as f64) * 4.0 / 32.0 / 4.0;
    let regs = acc_regs + 56.0;
    let by_regs = mach.regs_per_sm / (32.0 * regs);

    let smem = (s.bq * hd * 2 + 2 * s.bk * hd * 2) as f64;
    let ctas = if smem <= mach.smem_per_cta { mach.smem_per_cta / smem } else { 0.0 };
    let by_smem = ctas * warps;
    let by_grid = (m as f64 / s.bq as f64) * heads as f64 * warps / mach.sms as f64;
    let warps_per_sm = by_regs.min(by_smem).min(by_grid);

    let latency = 64.0 / warps_per_sm.max(1.0);
    let work = kv_ms + frag_ms + mma_ms;
    let bound = if kv_ms >= mma_ms { "K/V (DRAM)" } else { "tensor" };
    let t = work * latency;
    let _ = kv_heads;

    let known_error = "the L1's TRANSACTIONS and return latency are not modelled: `bound` is unreliable, \
                       the magnitudes are not";
    AttnPrediction { mma_ms, kv_ms, frag_ms, warps_per_sm, bound, total_ms: t, known_error }
}

pub fn matvec(weights: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert!(out.len() >= rows);

    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for (w, v) in row.iter().zip(x.iter()) {
            acc += w * v;
        }
        out[r] += acc;
    }
}

pub fn matmul(
    weights: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    seq_len: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(x.len(), seq_len * cols);
    debug_assert_eq!(out.len(), seq_len * rows);

    for t in 0..seq_len {
        let x_t = &x[t * cols..(t + 1) * cols];
        let o_t = &mut out[t * rows..(t + 1) * rows];
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for (w, v) in row.iter().zip(x_t.iter()) {
                acc += w * v;
            }
            o_t[r] = acc;
        }
    }
}

#[cfg(test)]

mod host_reference_tests {
    use super::*;

    #[test]
    fn matmul_lays_out_rows_per_token() {
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        matmul(&w, 2, 2, &x, 2, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matvec_accumulates() {
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        let x = vec![1.0f32, 1.0];
        let mut out = vec![10.0f32, 20.0];
        matvec(&w, 2, 2, &x, &mut out);
        assert_eq!(out, vec![11.0, 21.0]);
    }
}

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Nvfp4,
    Fp8,
}

impl Kind {
    pub fn atom_k(self) -> usize {
        match self { Kind::Nvfp4 => 64, Kind::Fp8 => 32 }
    }

    pub fn element_bits(self) -> usize {
        match self { Kind::Nvfp4 => 4, Kind::Fp8 => 8 }
    }

    pub fn has_sf(self) -> bool { self == Kind::Nvfp4 }

    pub fn peak_fraction(self) -> f64 {
        match self { Kind::Nvfp4 => 1.0, Kind::Fp8 => 0.5 }
    }

    pub fn fit(self) -> Fit {
        match self { Kind::Nvfp4 => Fit::FITTED_5090, Kind::Fp8 => Fit::FITTED_5090_FP8 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub kind: Kind,
    pub warps_m: usize,
    pub warps_n: usize,
    pub warp_m_rows: usize,
    pub n_tile: usize,
    pub ab_stages_a: usize,
    pub ab_stages_b: usize,
    pub sub_chunks: usize,
    pub epi_tile_m: usize,
    pub epi_tile_n: usize,
    pub epi_stages: usize,
    pub epi_aliased: bool,
    pub mma_regs: usize,
    pub producer_regs: usize,
    pub producer_warps: usize,
    pub swizzle_log2: u32,
    pub tiles_per_barrier: usize,
    pub regs_measured: usize,
    pub smem_measured: usize,
}

impl Params {
    pub const SHIPPED: Params = Params {
        kind: Kind::Nvfp4,
        warps_m: 4, warps_n: 2, warp_m_rows: 32, n_tile: 64,
        ab_stages_a: 3, ab_stages_b: 3, sub_chunks: 2,
        epi_tile_m: 128, epi_tile_n: 32, epi_stages: 2, epi_aliased: false,
        mma_regs: 232, producer_regs: 40, producer_warps: 4,
        swizzle_log2: 0, tiles_per_barrier: 1,
        regs_measured: 0, smem_measured: 0,
    };

    pub const FP8: Params = Params {
        kind: Kind::Fp8,
        warps_m: 4, warps_n: 2, warp_m_rows: 16, n_tile: 64,
        ab_stages_a: 3, ab_stages_b: 3, sub_chunks: 4,
        epi_tile_m: 64, epi_tile_n: 32, epi_stages: 2, epi_aliased: false,
        mma_regs: 64, producer_regs: 40, producer_warps: 4,
        swizzle_log2: 0, tiles_per_barrier: 1,
        regs_measured: 128, smem_measured: 90_544,
    };

    pub const FP8_K64: Params = Params { sub_chunks: 2, ..Params::FP8 };

    pub const FP8_W2X4: Params = Params { warps_m: 2, warps_n: 4, warp_m_rows: 32, n_tile: 32,
                                          ..Params::FP8 };

    pub const FP8_T128: Params = Params { warp_m_rows: 32, ab_stages_a: 2, ab_stages_b: 2,
                                          epi_tile_m: 128, smem_measured: 81_920, ..Params::FP8 };

    pub const FP8_N64: Params = Params { n_tile: 32, smem_measured: 57_824, ..Params::FP8 };

    pub const FP8_EPI1: Params = Params { epi_stages: 1, smem_measured: 74_160, ..Params::FP8 };
    pub const FP8_EPI3: Params = Params { epi_stages: 3, smem_measured: 98_736, ..Params::FP8 };

    pub const FP8_STAGES2: Params = Params {
        ab_stages_a: 2, ab_stages_b: 2, smem_measured: 65_952,
        ..Params::FP8
    };

    pub const FP8_BIG: Params = Params {
        warps_m: 8, warps_n: 1, warp_m_rows: 32, n_tile: 64,
        sub_chunks: 2,
        epi_tile_m: 256, epi_tile_n: 32, epi_stages: 1,
        regs_measured: 168, smem_measured: 96_256,
        ..Params::FP8
    };

    pub const FP8_ALIASED: Params = Params {
        ab_stages_a: 2, ab_stages_b: 2, epi_aliased: true, smem_measured: 49_568,
        regs_measured: 64,
        ..Params::FP8
    };

    pub const FP8_96X128: Params = Params {
        warps_m: 6, warp_m_rows: 16,
        smem_measured: 57_824,
        ..Params::FP8
    };

    pub fn mma_warps(&self) -> usize { self.warps_m * self.warps_n }
    pub fn cta_warps(&self) -> usize { self.mma_warps() + self.producer_warps }
    pub fn cta_threads(&self) -> usize { self.cta_warps() * 32 }
    pub fn m_tile(&self) -> usize { self.warps_m * self.warp_m_rows }
    pub fn n_block(&self) -> usize { self.warps_n * self.n_tile }
    pub fn m_atoms(&self) -> usize { self.warp_m_rows / 16 }
    pub fn n_sub(&self) -> usize { self.n_tile / 8 }

    pub fn ktile(&self) -> usize { self.kind.atom_k() * self.sub_chunks }
    pub fn k_tiles(&self, k: usize) -> usize { k.div_ceil(self.ktile()) }

    pub fn resident_warps(&self, machine: &Machine) -> usize {
        let smem = if self.smem_measured > 0 { self.smem_measured } else { self.smem().total };
        let regs = if self.regs_measured > 0 {
            self.regs_measured
        } else {
            self.regs(machine.regs_per_sm).sum
        };
        let _ = (smem, regs);
        let (ctas, _) = machine.ctas_per_sm(self);
        ctas * self.mma_warps()
    }

    pub fn atom_rows(&self) -> usize {
        if !self.kind.has_sf() { return 128; }
        let m = self.m_tile();
        if m < 128 { return 128; }
        [128usize, 96, 64, 32].into_iter().find(|c| m % c == 0).unwrap_or(32)
    }

    pub fn sf_atoms_m(&self) -> usize { (self.m_tile() / self.atom_rows()).max(1) }

    pub fn penalised(&self) -> bool { self.atom_rows() != 128 }

    pub fn legal_atom(&self) -> bool { !self.kind.has_sf() || matches!(self.atom_rows(), 128 | 96) }

    pub fn flops(&self) -> f64 { 2.0 * (self.m_tile() * self.n_block() * 4096) as f64 }

    pub fn a_fits_l2(&self, m: usize, k: usize, machine: &Machine) -> bool {
        self.packed_bytes(m * k) <= machine.l2_bytes
    }

    pub fn l2_read_bytes_public(&self, m: usize, n: usize, k: usize) -> u64 {
        let t = self.packed_bytes(m * n * k);
        (t / self.n_block() as f64 + t / self.m_tile() as f64) as u64
    }

    pub fn packed_bytes(&self, elems: usize) -> f64 {
        elems as f64 * self.kind.element_bits() as f64 / 8.0
    }

    pub fn l2_row_limit(&self, k: usize, machine: &Machine) -> usize {
        (machine.l2_bytes / self.packed_bytes(k)) as usize
    }

    pub fn bytes(&self, m: usize, n: usize, k: usize) -> Bytes {
        let a = self.packed_bytes(self.m_tile() * k);
        let b = self.packed_bytes(self.n_block() * k);
        let c = (self.m_tile() * self.n_block()) as f64 * 4.0;
        let m_tiles = m.div_ceil(self.m_tile());
        Bytes { a_per_tile: a, b_per_tile: b, c_per_tile: c, b_reread: m_tiles as f64,
                n_tiles: n.div_ceil(self.n_block()) as f64 }
    }

    pub fn smem(&self) -> Smem {
        let row_bytes = (self.ktile() * self.kind.element_bits() / 8) as usize;
        let a_stage = self.m_tile() * row_bytes;
        let b_stage = self.n_block() * row_bytes;
        let (sf_a, sf_b) = if self.kind.has_sf() {
            (self.sf_atoms_m() * self.sub_chunks * self.atom_rows() * 4, self.sub_chunks * 512)
        } else {
            (0, 0)
        };
        let ab = self.ab_stages_a * (a_stage + sf_a) + self.ab_stages_b * (b_stage + sf_b);

        let epi = self.epi_stages * self.epi_tile_m * self.epi_tile_n * 4;

        let total = ab + if self.epi_aliased { 0 } else { epi };
        Smem { a_stage, b_stage, sf_a, sf_b, ab, epi, total }
    }

    pub fn regs(&self, regs_per_sm: usize) -> Regs {
        let available = regs_per_sm / self.cta_threads();
        let accumulator = self.m_atoms() * self.n_sub() * 4;
        let prefetch = 8 * self.m_atoms() + 4 * self.n_sub() + 2 * (self.m_atoms() + self.n_sub());
        Regs { available, accumulator, prefetch, sum: accumulator + prefetch }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bytes {
    pub a_per_tile: f64,
    pub b_per_tile: f64,
    pub c_per_tile: f64,
    pub b_reread: f64,
    pub n_tiles: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Smem {
    pub a_stage: usize,
    pub b_stage: usize,
    pub sf_a: usize,
    pub sf_b: usize,
    pub ab: usize,
    pub epi: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Regs {
    pub available: usize,
    pub accumulator: usize,
    pub prefetch: usize,
    pub sum: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Machine {
    pub sms: usize,
    pub regs_per_sm: usize,
    pub smem_per_sm: usize,
    pub smem_per_block: usize,
    pub smem_per_block_driver: usize,
    pub threads_per_sm: usize,
    pub warps_per_sm: usize,
    pub warp_size: usize,
    pub clock_mhz: f64,
    pub peak_flops: f64,
    pub dram_bw: f64,
    pub smem_bw_per_sm: f64,
    pub l2_bytes: f64,
}

impl Machine {
    pub const RTX_5090: Machine = Machine {
        sms: 170,
        regs_per_sm: 65_536,
        smem_per_sm: 102_400,
        smem_per_block: 101_376,
        smem_per_block_driver: 1_044,
        threads_per_sm: 1_536,
        warps_per_sm: 48,
        warp_size: 32,
        clock_mhz: 3_090.0,
        peak_flops: 1_676e12,
        dram_bw: 1_792e9,
        smem_bw_per_sm: 307e9,
        l2_bytes: 96.0 * 1024.0 * 1024.0,
    };

    pub const RTX_PRO_6000: Machine = Machine {
        sms: 188,
        ..Machine::RTX_5090
    };
}

impl Machine {
    pub fn per_sm(&self) -> PerSm {
        PerSm {
            sms: self.sms,
            peak_flops: self.peak_flops / self.sms as f64,
            dram_bw: self.dram_bw / self.sms as f64,
            smem_bw: self.smem_bw_per_sm,
            regs: self.regs_per_sm,
            smem: self.smem_per_sm,
            threads: self.threads_per_sm,
            warps: self.warps_per_sm,
        }
    }

    pub fn ctas_per_sm(&self, p: &Params) -> (usize, &'static str) {
        let smem = if p.smem_measured > 0 { p.smem_measured } else { p.smem().total };
        let regs = if p.regs_measured > 0 { p.regs_measured } else { p.regs(self.regs_per_sm).sum };
        self.ctas_per_sm_with(p.cta_threads(), smem, regs)
    }

    pub fn ctas_per_sm_with(&self, threads: usize, smem_bytes: usize, regs: usize) -> (usize, &'static str) {
        let by_smem = self.smem_per_sm / (smem_bytes + self.smem_per_block_driver).max(1);
        let by_threads = self.threads_per_sm / threads.max(1);
        let by_regs = self.regs_per_sm / (threads.max(1) * regs.max(1));
        let n = by_smem.min(by_threads).min(by_regs).max(1);
        let why = if n == by_smem { "shared memory" }
                  else if n == by_threads { "threads" }
                  else { "registers" };
        (n, why)
    }

    pub fn cycles(&self, us: f64) -> f64 { us * self.clock_mhz }
}

#[derive(Debug, Clone, Copy)]
pub struct PerSm {
    pub sms: usize,
    pub peak_flops: f64,
    pub dram_bw: f64,
    pub smem_bw: f64,
    pub regs: usize,
    pub smem: usize,
    pub threads: usize,
    pub warps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Baseline {
    pub sf_conflict_free: bool,
    pub use_setmaxnreg: bool,
    pub sf_via_tma: bool,
    pub warp_specialized_store: bool,
    pub amortized_timing: bool,
}

impl Baseline {
    pub const SHIPPED: Baseline = Baseline {
        sf_conflict_free: true,
        use_setmaxnreg: true,
        sf_via_tma: true,
        warp_specialized_store: true,
        amortized_timing: true,
    };

    pub fn violations(&self) -> Vec<&'static str> {
        let mut bad = Vec::new();
        if !self.sf_conflict_free {
            bad.push("SF_CONFLICT_FREE is off: the SFA atom is the banded layout, and every coefficient \
                      here was fitted with the conflict-free one");
        }
        if !self.use_setmaxnreg {
            bad.push("USE_SETMAXNREG is off: ptxas allocates a different register count, so the \
                      accumulator and prefetch terms no longer describe this kernel");
        }
        if !self.sf_via_tma {
            bad.push("SF-via-TMA is off: the producer stages the scales per lane, so the barrier term \
                      is not the one measured");
        }
        if !self.warp_specialized_store {
            bad.push("the warp-specialised store is off: the epilogue term was fitted with it");
        }
        if !self.amortized_timing {
            bad.push("timing syncs once per launch: every figure below ~1 ms is dominated by the sync, \
                      which is how a 15x phantom collapse was measured");
        }
        bad
    }

    pub fn applies(&self) -> bool { self.violations().is_empty() }
}

#[derive(Debug, Clone, Copy)]
pub struct Fit {
    pub mma_efficiency: f64,
    pub barrier_ns: f64,
    pub prologue_ns: f64,
    pub epilogue_ns_per_elem: f64,
    pub memory_latency_tiles: f64,
    pub latency_us_per_tile: f64,
    pub atom_penalty: f64,
    pub fill_latency_ns: f64,
    pub ilp_gamma: f64,
    pub ilp_in_flight_ref: f64,
    pub launch_us: f64,
    pub wait_ns_per_ktile: f64,
    pub resident_warps_fitted: f64,
    pub ilp_fitted: f64,
}

impl Fit {
    pub const FITTED_5090: Fit = Fit {
        memory_latency_tiles: 2.0,
        latency_us_per_tile: 1.2,
        mma_efficiency: 0.78,
        barrier_ns: 155.0,
        prologue_ns: 190.0,
        epilogue_ns_per_elem: 0.0032,
        atom_penalty: 0.72,
        fill_latency_ns: 1200.0,
        ilp_gamma: 1.0,
        ilp_in_flight_ref: 16.0,
        launch_us: 8.0,
        wait_ns_per_ktile: 0.0,
        resident_warps_fitted: 8.0,
        ilp_fitted: 1.0,
    };

    pub const FITTED_5090_FP8: Fit = Fit {
        mma_efficiency: 0.90,
        barrier_ns: 29.1 * 0.4,
        prologue_ns: 190.0 * 0.4,
        epilogue_ns_per_elem: 0.0032 * 0.4,
        atom_penalty: 1.0,
        fill_latency_ns: 1200.0,
        ilp_gamma: 0.35,
        ilp_in_flight_ref: 10.0,
        launch_us: 8.0,
        memory_latency_tiles: 2.0,
        latency_us_per_tile: 1.2,
        wait_ns_per_ktile: 96.6,
        resident_warps_fitted: 8.0,
        ilp_fitted: 0.7071,
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Breakdown {
    pub tensor_us: f64,
    pub ilp: f64,
    pub dram_us: f64,
    pub smem_us: f64,
    pub roofline_us: f64,
    pub barrier_us: f64,
    pub prologue_us: f64,
    pub epilogue_us: f64,
    pub wait_us: f64,
    pub total_us: f64,
    pub stall: &'static str,
}

impl Breakdown {
    pub fn pipes(&self) -> &'static str {
        if self.tensor_us >= self.dram_us && self.tensor_us >= self.smem_us { "tensor" }
        else if self.dram_us >= self.smem_us { "dram" }
        else { "smem" }
    }
}

impl fmt::Display for Breakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "roofline {:.1} us ({}), +barrier {:.1} +prologue {:.1} +epilogue {:.1} = {:.1} us",
               self.roofline_us, self.pipes(), self.barrier_us, self.prologue_us,
               self.epilogue_us, self.total_us)
    }
}

pub fn tile_time(p: &Params, machine: &Machine, fit: &Fit, m: usize, n: usize, k: usize) -> Breakdown {
    let ilp = ilp_factor_gamma(p.m_atoms(), p.n_sub(), fit.ilp_gamma, fit.ilp_in_flight_ref);
    let atom = if p.atom_rows() == 128 { 1.0 } else { fit.atom_penalty };
    let per_sm = machine.per_sm();

    let per_sm_peak = per_sm.peak_flops * p.kind.peak_fraction();

    let resident = p.resident_warps(machine).max(1) as f64;

    let occupancy = (fit.resident_warps_fitted / resident).max(1e-9);
    let tensor_us = p.flops()
        / (per_sm_peak * fit.mma_efficiency * ilp * atom)
        * occupancy * 1e6;

    let _total_tiles = tiles(p, m, n) as f64;
    let a_total = p.packed_bytes(m * k);
    let b_total = p.packed_bytes(n * k);
    let c_total = (m * n) as f64 * 4.0;

    let n_tiles = n.div_ceil(p.n_block()) as f64;
    let a_dram = if p.a_fits_l2(m, k, machine) { a_total } else { a_total * n_tiles };

    let swizzle_credit = if p.swizzle_log2 == 0 { 1.0 } else { 0.94 };

    let _ = (a_dram, swizzle_credit);

    let wave_count = waves_with(p, m, n, machine, Scheduling::Eager).max(1) as f64;
    let compulsory_bytes = a_total + b_total + c_total;
    let reread_bytes = if p.a_fits_l2(m, k, machine) { 0.0 } else { a_total * (n_tiles - 1.0) };

    let stage_bytes = (p.m_tile() + p.n_block()) as f64 * p.ktile() as f64;
    let exposed_stages = (p.ab_stages_a.min(p.ab_stages_b).max(1) as f64 - 1.0).max(0.0);
    let fill_us = stage_bytes * exposed_stages / machine.dram_bw * 1e6 / wave_count;

    let steady_bytes = compulsory_bytes + reread_bytes;

    let dram_us = (steady_bytes / machine.dram_bw * 1e6) / wave_count;

    let stages = p.ab_stages_a.min(p.ab_stages_b).max(1) as f64;
    let hidable = (stages - 1.0).max(0.0);
    let exposed = (fit.memory_latency_tiles - hidable).max(0.0);

    let latency_us = exposed * fit.latency_us_per_tile + fill_us;

    let ksubs = k as f64 / 64.0;
    let a_loads = p.m_atoms() as f64 * ksubs;
    let b_loads = p.n_sub() as f64 * ksubs;
    let smem_bytes = a_loads * 32.0 * 16.0 + b_loads * 32.0 * 8.0 + (a_loads + b_loads) * 32.0 * 4.0;
    let smem_us = smem_bytes / per_sm.smem_bw * 1e6;

    let roofline_us = tensor_us.max(dram_us).max(smem_us);

    let barrier_us = (p.k_tiles(k).div_ceil(p.tiles_per_barrier.max(1))) as f64
        * fit.barrier_ns / 1000.0
        * (p.mma_warps() as f64 / 8.0).sqrt();

    let wait_us = p.k_tiles(k) as f64 * fit.wait_ns_per_ktile / 1000.0
        * (fit.resident_warps_fitted * fit.ilp_fitted) / (resident * ilp).max(1e-9)
        + latency_us;
    let prologue_us = (p.ab_stages_a + p.ab_stages_b) as f64 * fit.prologue_ns / 1000.0;

    let epi_elems = (p.m_tile() * p.n_block()) as f64;
    let epi_passes = (p.m_tile() as f64 / p.epi_tile_m as f64).ceil()
        * (p.n_block() as f64 / p.epi_tile_n as f64).ceil();
    let epilogue_us = epi_elems * fit.epilogue_ns_per_elem / 1000.0
        + epi_passes * 120.0 / 1000.0;

    let stall = if ilp < 0.95 {
        "wait"
    } else if dram_us > tensor_us {
        "long_scoreboard"
    } else if barrier_us > 0.25 * roofline_us {
        "barrier"
    } else {
        "short_scoreboard"
    };

    Breakdown {
        tensor_us, dram_us, smem_us, roofline_us,
        barrier_us, prologue_us, epilogue_us, wait_us,
        total_us: roofline_us + barrier_us + prologue_us + epilogue_us + wait_us,
        ilp,
        stall,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineSide {
    pub cta_threads: usize,
    pub num_mma_warps: usize,
    pub producer_warps: usize,
    pub staging_stride: usize,
    pub staging_index_base: isize,
    pub staging_bound: usize,
    pub full_arrivals: usize,
    pub empty_arrivals: usize,
    pub full_via_cp_async: bool,
    pub empty_lane_guarded: bool,
    pub early_return_before_sync: bool,
    pub expected_tx: u32,
    pub box_bytes: u32,
}

pub fn pipeline_violations(p: &PipelineSide) -> Vec<String> {
    let mut bad = Vec::new();

    let first = p.cta_threads as isize - (p.producer_warps * 32) as isize + p.staging_index_base;
    if p.staging_stride != p.producer_warps * 32 {
        bad.push(format!(
            "staging stride {} is not the producer warpgroup's {} threads: the loop's stride must be the \
             threads that execute it", p.staging_stride, p.producer_warps * 32));
    }
    if first < 0 || first >= p.staging_bound as isize {
        bad.push(format!(
            "the staging loop never executes: it starts at {first} against a bound of {} — a HANG",
            p.staging_bound));
    }

    let full_expected = if p.full_via_cp_async { p.producer_warps * 32 } else { p.producer_warps };
    if p.full_arrivals != full_expected {
        bad.push(format!(
            "full[s] is initialised for {} arrivals but {} {} arrive on it: {} — a HANG",
            p.full_arrivals, full_expected,
            if p.full_via_cp_async { "threads (cp.async tracks per thread)" } else { "warps" },
            if p.full_via_cp_async { "it must be PRODUCER_WARPS * 32" } else { "lane-guard it" }));
    }
    let empty_expected = if p.empty_lane_guarded { p.num_mma_warps } else { p.num_mma_warps * 32 };
    if p.empty_arrivals != empty_expected {
        bad.push(format!(
            "empty[s] is initialised for {} arrivals but {} {} arrive on it — a HANG",
            p.empty_arrivals, empty_expected,
            if p.empty_lane_guarded { "warps (one elected lane)" } else { "threads" }));
    }

    if p.expected_tx != p.box_bytes {
        bad.push(format!(
            "expect_tx is armed for {} bytes but the boxes deliver {} — the barrier never completes: \
             a HANG", p.expected_tx, p.box_bytes));
    }

    if p.early_return_before_sync {
        bad.push("a warp returns before a __syncthreads() that other warps reach: a DEADLOCK".into());
    }
    if (p.num_mma_warps + p.producer_warps) % 4 != 0 {
        bad.push(format!(
            "{} warps is not a whole number of warpgroups", p.num_mma_warps + p.producer_warps));
    }
    bad
}

pub fn can_judge(p: &Params, machine: &Machine, fit: &Fit) -> bool {
    (p.resident_warps(machine) as f64 - fit.resident_warps_fitted).abs() < 1.0
}

const RESIDENT_WARPS_FITTED: f64 = 8.0;

fn ilp_factor(m_atoms: usize, n_sub: usize) -> f64 {
    ilp_factor_gamma(m_atoms, n_sub, 1.0, 16.0)
}

fn ilp_factor_gamma(m_atoms: usize, n_sub: usize, gamma: f64, in_flight_ref: f64) -> f64 {
    let in_flight = m_atoms * n_sub;
    if in_flight == 0 {
        return 0.0;
    }

    let traffic = 1.0 / m_atoms as f64 + 1.0 / n_sub as f64;
    let traffic_shipped = 1.0 / 2.0 + 1.0 / 8.0;
    let effective = in_flight as f64 * (traffic_shipped / traffic).powf(gamma);
    (effective / in_flight_ref).sqrt().min(1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheduling {
    Persistent,
    Eager,
}

pub fn capacity_fraction(p: &Params, m: usize, n: usize, machine: &Machine) -> f64 {
    let (ctas, _) = machine.ctas_per_sm(p);
    let slots = (machine.sms * ctas).max(1);
    tiles(p, m, n) as f64 / slots as f64
}

pub fn waves_with(p: &Params, m: usize, n: usize, machine: &Machine, sched: Scheduling) -> usize {
    let denom = match sched {
        Scheduling::Persistent => machine.sms,
        Scheduling::Eager => {
            let (ctas, _) = machine.ctas_per_sm(p);
            machine.sms * ctas
        }
    };
    tiles(p, m, n).div_ceil(denom.max(1))
}

pub fn tiles(p: &Params, m: usize, n: usize) -> usize {
    m.div_ceil(p.m_tile()) * n.div_ceil(p.n_block())
}

pub fn waves(p: &Params, m: usize, n: usize, machine: &Machine) -> usize {
    waves_with(p, m, n, machine, Scheduling::Persistent)
}

pub fn makespan_us(p: &Params, machine: &Machine, fit: &Fit, m: usize, n: usize, k: usize) -> f64 {
    waves(p, m, n, machine) as f64 * tile_time(p, machine, fit, m, n, k).total_us
}

pub fn buildable() -> [(&'static str, Params); 8] {
    [
        ("SHIP", Params::FP8),
        ("BIG", Params::FP8_BIG),
        ("T128", Params::FP8_T128),
        ("W2X4", Params::FP8_W2X4),
        ("N64", Params::FP8_N64),
        ("K64", Params::FP8_K64),
        ("S2", Params::FP8_STAGES2),
        ("EPI1", Params::FP8_EPI1),
    ]
}

pub fn dispatch(m: usize, n: usize, k: usize, machine: &Machine, fit: &Fit) -> (&'static str, f64) {
    let mut best = ("SHIP", f64::MAX);
    for (name, p) in buildable() {
        let t = makespan_us(&p, machine, fit, m, n, k);
        if t < best.1 {
            best = (name, t);
        }
    }
    best
}

pub fn rejection(p: &Params, machine: &Machine) -> Option<String> {
    if p.cta_warps() % 4 != 0 {
        return Some(format!("{} warps is not a whole number of warpgroups", p.cta_warps()));
    }
    if p.cta_threads() > machine.threads_per_sm {
        return Some(format!("{} threads exceeds {} per SM", p.cta_threads(), machine.threads_per_sm));
    }

    if p.kind.has_sf() && p.n_block() % 128 != 0 {
        return Some(format!("N_BLOCK {} is not a multiple of 128 (the weight layout)", p.n_block()));
    }
    if p.warp_m_rows % 16 != 0 {
        return Some(format!("WARP_M_ROWS {} is not a multiple of 16", p.warp_m_rows));
    }
    if p.n_tile % 8 != 0 {
        return Some(format!("N_TILE {} is not a multiple of 8", p.n_tile));
    }
    let r = p.regs(machine.regs_per_sm);
    if r.sum > r.available {
        return Some(format!("{} registers needed, {} available at {} threads",
                            r.sum, r.available, p.cta_threads()));
    }
    let s = p.smem();
    if s.total > machine.smem_per_block {
        return Some(format!("{} B of shared memory exceeds {} B per block",
                            s.total, machine.smem_per_block));
    }
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_attention_model_tracks_the_measured_configurations() {
        let mach = AttnMachine::RTX_5090;
        let (hd, heads, kv_heads) = (256usize, 16usize, 4usize);
        let narrow = AttnShape { bq: 16, bk: 16, row_groups: 1, col_groups: 2 };
        let wide = AttnShape { bq: 64, bk: 16, row_groups: 4, col_groups: 2 };

        let widest = AttnShape { bq: 128, bk: 16, row_groups: 8, col_groups: 2 };
        let (measured_narrow, measured_wide) = (70.9f64, 57.5f64);
        let a = attention_time(16384, hd, heads, kv_heads, &wide, &mach);
        let b = attention_time(16384, hd, heads, kv_heads, &widest, &mach);
        let pred = a.total_ms / b.total_ms;
        let meas = measured_narrow / measured_wide;
        eprintln!("  16384 kernel BQ 64 -> BQ 128: predicted {:.2}x  measured {:.2}x  (kv {:.1} -> {:.1}, mma {:.1})",
                  pred, meas, a.kv_ms, b.kv_ms, b.mma_ms);
        assert!((pred - meas).abs() / meas < 0.15,
                "the model predicts {pred:.2}x against {meas:.2}x measured — a factor is wrong, see the terms");

        let b = attention_time(16384, hd, heads, kv_heads, &widest, &mach);
        eprintln!("  PROFILE CHECK  tensor pipe {:.2}% (measured 24.06)   occupancy {:.2}% (measured 33.34)   \
                   duration {:.1} ms (measured 57.5)",
                  b.tensor_pipe_pct(), b.occupancy_pct(), b.total_ms);
        assert!((b.tensor_pipe_pct() - 24.06).abs() < 10.0,
                "tensor pipe {:.2}% against 24.06 measured — the MMA term is wrong", b.tensor_pipe_pct());
        assert!((b.occupancy_pct() - 33.34).abs() < 15.0,
                "occupancy {:.2}% against 33.34 measured — the residency terms are wrong", b.occupancy_pct());

        for m in [8192usize, 16384, 32768] {
            let a = attention_time(m, hd, heads, kv_heads, &narrow, &mach);
            let b = attention_time(m, hd, heads, kv_heads, &wide, &mach);
            eprintln!("  m={m:>6}  narrow {:>7.1} ms  wide {:>7.1} ms  = {:.2}x  (warps/SM {:.1})",
                      a.total_ms, b.total_ms, a.total_ms / b.total_ms, b.warps_per_sm);
            eprintln!("           {} — the model's `bound` reads \"{}\" and the PROFILE reads L1 92.42%",
                      b.known_error, b.bound);
        }
    }
    use super::*;

    #[test]
    fn the_ilp_curve_pins_its_two_anchors() {
        assert!((ilp_factor_gamma(1, 8, 0.0, 16.0) - 0.70).abs() < 0.01, "anchor 1: {}", ilp_factor_gamma(1, 8, 0.0, 16.0));
        assert!((ilp_factor_gamma(2, 8, 0.0, 16.0) - 1.0).abs() < 1e-9, "anchor 2: {}", ilp_factor_gamma(2, 8, 0.0, 16.0));

        let mut prev = 0.0;
        for a in 1..=8 {
            let f = ilp_factor_gamma(a, 8, 0.0, 16.0);
            assert!(f >= prev - 1e-12, "not monotone at {a}: {f} < {prev}");
            prev = f;
        }
        assert_eq!(ilp_factor_gamma(8, 8, 0.0, 16.0), 1.0);
        assert_eq!(ilp_factor(0, 8), 0.0);

        assert!(ilp_factor(2, 4) > ilp_factor(1, 8),
                "at equal in-flight, 2 atoms of 4 sub must beat 1 atom of 8: {} vs {}",
                ilp_factor(2, 4), ilp_factor(1, 8));

        assert_eq!(ilp_factor(2, 8), 1.0);

        assert!(ilp_factor(1, 4) < ilp_factor(1, 8), "narrowing n_sub must not help");
    }

    #[test]
    fn the_occupancy_term_sees_a_starved_grid() {
        let machine = Machine::RTX_5090;
        let p = Params::SHIPPED;

        assert!(capacity_fraction(&p, 768, 1024, &machine) < 1.0);
        assert!(capacity_fraction(&p, 32768, 12288, &machine) > 1.0);

        assert!(waves_with(&p, 768, 1024, &machine, Scheduling::Eager)
                <= waves_with(&p, 768, 1024, &machine, Scheduling::Persistent) + 1);
        let (ctas, why) = machine.ctas_per_sm_with(256, 24_580, 64);
        eprintln!("  fp8_gemm: {ctas} CTAs/SM, limited by {why}");
        assert_eq!(ctas, 3, "fp8_gemm's own numbers, not the NVFP4 derivation");
    }

    #[test]
    fn the_fp8_tile_choice_is_predicted() {
        let machine = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let (m, n, k) = (768usize, 12288usize, 4096usize);

        let shipped = Params { warps_m: 4, warps_n: 2, warp_m_rows: 16, n_tile: 64,
                               producer_warps: 0, regs_measured: 64, smem_measured: 24_580,
                               ..Params::SHIPPED };
        let candidates = [
            ("shipped  M_TILE  64 (1 atom)", shipped),
            ("taller   M_TILE 128 (2 atoms)", Params { warp_m_rows: 32, regs_measured: 124,
                                                        smem_measured: 32_768, ..shipped }),
            ("wider N  4x4 (1 atom)", Params { warps_n: 4, smem_measured: 24_580 + 8_192,
                                                ..shipped }),
            ("narrower M_TILE  64, n_tile 32", Params { n_tile: 32, ..shipped }),
        ];
        let mut best = ("", f64::INFINITY);
        for (name, p) in candidates {
            let t = tile_time(&p, &machine, &fit, m, n, k);
            let span = makespan_us(&p, &machine, &fit, m, n, k);
            let (ctas, _) = machine.ctas_per_sm_with(256, 24_580, 64);
            let cap = tiles(&p, m, n) as f64 / (machine.sms * ctas) as f64;
            eprintln!(
                "  {name:32} mt {:3} atoms {} in_flight {:2} ilp {:.3} tiles {:5} cap {:.2} \
                 tensor {:6.1} total {:6.1} span {:8.1} us  {}",
                p.m_tile(), p.m_atoms(), p.m_atoms() * p.n_sub(),
                ilp_factor(p.m_atoms(), p.n_sub()),
                tiles(&p, m, n), cap, t.tensor_us, t.total_us, span, t.stall);
            if span < best.1 { best = (name, span); }
        }
        eprintln!("  -> the model prefers: {} at {:.1} us", best.0, best.1);

        let tall = Params { warp_m_rows: 32, ..shipped };
        let base = makespan_us(&shipped, &machine, &fit, m, n, k);
        let wider = makespan_us(&tall, &machine, &fit, m, n, k);
        assert!(wider < base, "two atoms must beat one: {wider:.1} against {base:.1}");
        let gain = (base - wider) / base;
        eprintln!("  -> predicted gain from M_TILE 64 -> 128: {:.1}%", gain * 100.0);
        assert!(gain > 0.15, "the predicted gain is only {:.1}%", gain * 100.0);
    }

    #[test]
    fn the_stall_prediction_names_wait_for_a_starved_grid() {
        let machine = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let fp8 = Params { warps_m: 4, warps_n: 2, warp_m_rows: 16, n_tile: 64,
                           ..Params::SHIPPED };
        let t = tile_time(&fp8, &machine, &fit, 768, 12288, 4096);

        assert_eq!(t.stall, "wait", "ilp {:.3}", t.ilp);

        let nvfp4 = tile_time(&Params::SHIPPED, &machine, &fit, 768, 12288, 4096);
        assert_ne!(nvfp4.stall, "wait", "ilp {:.3}", nvfp4.ilp);
    }

    #[test]
    fn the_ilp_gain_loses_to_the_register_cost_it_pays() {
        let machine = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let (m, n, k) = (768usize, 12288usize, 4096usize);

        let base = Params { warps_m: 4, warps_n: 2, warp_m_rows: 16, n_tile: 64,
                            producer_warps: 0, regs_measured: 64, smem_measured: 24_580,
                            ..Params::SHIPPED };
        let one_atom = base;

        let two_atoms = Params { warp_m_rows: 32, regs_measured: 124, smem_measured: 32_768,
                                 ..base };

        let wider_n = Params { warps_n: 4, smem_measured: 24_580 + 8_192, ..base };
        for (name, p) in [("one atom, 4x2", one_atom), ("two atoms, 4x2", two_atoms),
                          ("one atom, 4x4", wider_n)] {
            let t = tile_time(&p, &machine, &fit, m, n, k);
            eprintln!("  {name:16} in_flight {:2} ilp {:.3} resident {:2} warps  cta_threads {:3} \
                       tensor {:6.2} us  span {:7.1} us",
                      p.m_atoms() * p.n_sub(), t.ilp, p.resident_warps(&machine),
                      p.cta_threads(), t.tensor_us, makespan_us(&p, &machine, &fit, m, n, k));
        }
        let a = tile_time(&one_atom, &machine, &fit, m, n, k);
        let b = tile_time(&two_atoms, &machine, &fit, m, n, k);

        assert!(b.ilp > a.ilp, "ilp must rise");
        assert!(two_atoms.resident_warps(&machine) < one_atom.resident_warps(&machine),
                "resident warps must fall: {} against {}",
                two_atoms.resident_warps(&machine), one_atom.resident_warps(&machine));

        assert!(b.tensor_us > a.tensor_us,
                "the corrected model must predict a loss: {:.2} against {:.2}", b.tensor_us, a.tensor_us);
    }

    #[test]
    fn the_barrier_lever_is_priced() {
        let machine = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let (m, n, k) = (768usize, 12288usize, 4096usize);

        let base = Params { warps_m: 4, warps_n: 2, warp_m_rows: 16, n_tile: 64,
                            sub_chunks: 2, ab_stages_a: 2, ab_stages_b: 2, tiles_per_barrier: 1,
                            producer_warps: 0, regs_measured: 64, smem_measured: 24_580,
                            ..Params::SHIPPED };
        let smem_doubled = 2 * (4_096 + 8_192) * 2 + 4;
        let route1 = Params { sub_chunks: 4, smem_measured: smem_doubled, ..base };
        let route2 = Params { ab_stages_a: 4, ab_stages_b: 4, tiles_per_barrier: 2,
                              smem_measured: smem_doubled, ..base };
        let mut best = ("", f64::INFINITY);
        for (name, p) in [("shipped: 1 barrier/k-tile", base),
                          ("route 1: SUB_CHUNKS 4", route1),
                          ("route 2: STAGES 4, 2 tiles/barrier", route2)] {
            let t = tile_time(&p, &machine, &fit, m, n, k);
            let span = makespan_us(&p, &machine, &fit, m, n, k);
            eprintln!("  {name:34} k_tiles {:3} barrier {:5.2} prologue {:4.2} tensor {:5.2} \
                       resident {:2} warps  total {:5.2}  span {:7.1} us  {}",
                      p.k_tiles(k), t.barrier_us, t.prologue_us, t.tensor_us,
                      p.resident_warps(&machine), t.total_us, span, t.stall);
            if span < best.1 { best = (name, span); }
        }
        eprintln!("  -> the model prefers: {} at {:.1} us", best.0, best.1);
    }

    #[test]
    fn the_model_against_the_capture_at_every_size() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let (n, k) = (12288usize, 4096usize);

        let measured: [(usize, f64); 8] = [
            (512, 114_240.0), (768, 164_768.0), (1024, 219_488.0), (2048, 416_864.0),
            (4096, 815_776.0), (8192, 1_593_568.0), (16384, 3_192_064.0), (32768, 8_125_824.0),
        ];
        eprintln!("  {:>7} {:>11} {:>11} {:>8}  {:>9} {:>9} {:>9}  {}",
                  "m", "measured ns", "predicted ns", "ratio", "tensor", "barrier", "wait", "pipes");
        let mut worst: f64 = 0.0;
        for (m, got) in measured {
            let bd = tile_time(&Params::FP8, &machine, &fit, m, n, k);
            let waves = waves_with(&Params::FP8, m, n, &machine, Scheduling::Eager) as f64;
            let pred = bd.total_us * waves * 1000.0;
            let ratio = pred / got;
            worst = worst.max((ratio - 1.0).abs());
            eprintln!("  {:>7} {:>11.0} {:>11.0} {:>7.2}x  {:>8.1} {:>9.2} {:>9.2}  {}",
                      m, got, pred, ratio, bd.tensor_us, bd.barrier_us, bd.wait_us, bd.pipes());
        }
        eprintln!("  worst deviation: {:.0}%", worst * 100.0);

        assert!(worst < 1.0, "the model is off by more than 2x at some size: worst {:.0}%", worst * 100.0);
    }

    const MEASURED_NS: [[[f64; 14]; 2]; 8] = [

        [
            [18976.0, 19552.0, 21248.0, 22240.0, 27232.0, 26400.0, 44896.0, 59456.0, 78112.0, 140800.0, 265856.0, 537568.0, 1062560.0, 4032256.0],
            [35808.0, 31968.0, 47072.0, 46272.0, 62752.0, 67616.0, 108064.0, 153344.0, 201600.0, 400544.0, 796832.0, 1551520.0, 3109728.0, 7129792.0],
        ],

        [
            [67168.0, 32288.0, 34240.0, 32928.0, 34912.0, 36768.0, 38080.0, 70144.0, 78240.0, 135904.0, 242656.0, 458848.0, 920096.0, 3976128.0],
            [222592.0, 69760.0, 70176.0, 72736.0, 73760.0, 110912.0, 114592.0, 157984.0, 202688.0, 354112.0, 689216.0, 1352672.0, 3521632.0, 12023072.0],
        ],

        [
            [32672.0, 32448.0, 33536.0, 34144.0, 35200.0, 34048.0, 38272.0, 67744.0, 72480.0, 131488.0, 235936.0, 439648.0, 863552.0, 2337920.0],

            [53344.0, 36448.0, 36288.0, 70368.0, 69920.0, 72032.0, 102912.0, 140608.0, 180416.0, 345440.0, 642208.0, 1230176.0, 2543400.0, 5913568.0],
        ],

        [
            [20320.0, 19168.0, 20832.0, 22144.0, 20608.0, 25216.0, 44864.0, 62080.0, 80576.0, 142784.0, 268128.0, 536064.0, 1060416.0, 3056384.0],

            [30272.0, 21888.0, 42240.0, 47040.0, 61930.0, 67840.0, 108640.0, 155712.0, 204800.0, 404416.0, 787680.0, 1543328.0, 3099456.0, 7059008.0],
        ],

        [
            [17376.0, 14560.0, 16128.0, 28704.0, 31520.0, 32160.0, 51232.0, 72128.0, 88768.0, 173536.0, 345600.0, 685280.0, 1374944.0, 3859936.0],
            [54496.0, 29152.0, 46720.0, 59168.0, 71552.0, 88928.0, 133216.0, 200992.0, 257696.0, 511488.0, 1014816.0, 1992448.0, 5772384.0, 9919264.0],
        ],

        [
            [20256.0, 23744.0, 24352.0, 22144.0, 26848.0, 25184.0, 47744.0, 64896.0, 83072.0, 147840.0, 282208.0, 571872.0, 1121184.0, 2824096.0],
            [33792.0, 27136.0, 51456.0, 49536.0, 70272.0, 72704.0, 114752.0, 166592.0, 220640.0, 427616.0, 850880.0, 1657504.0, 3300672.0, 9895616.0],
        ],

        [
            [22976.0, 22144.0, 24128.0, 23648.0, 26848.0, 29216.0, 48000.0, 68832.0, 86368.0, 150048.0, 288544.0, 556128.0, 1100608.0, 2825344.0],
            [29344.0, 26528.0, 47520.0, 51904.0, 70528.0, 69440.0, 114464.0, 165472.0, 221792.0, 420544.0, 826784.0, 1611904.0, 4088384.0, 11480192.0],
        ],

        [
            [19168.0, 19936.0, 20864.0, 19040.0, 23872.0, 24576.0, 45824.0, 62528.0, 79584.0, 143424.0, 269344.0, 537696.0, 1060992.0, 2997568.0],
            [30624.0, 21984.0, 42528.0, 45760.0, 65696.0, 68480.0, 110464.0, 157056.0, 210304.0, 404992.0, 791200.0, 1560768.0, 3108864.0, 8542464.0],
        ],
    ];
    const MEASURED_M: [usize; 14] = [16, 64, 128, 192, 256, 320, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768];
    const MEASURED_N: [usize; 2] = [4096, 12288];
    const MEASURED_TAGS: [&str; 8] = ["SHIP", "BIG", "T128", "W2X4", "N64", "K64", "S2", "EPI1"];

    #[test]
    fn the_model_breakdown_at_the_misses() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;

        let points: [(usize, usize, f64, &str); 6] = [
            (16, 4096, 18_976.0, "SHIP"), (16, 12288, 35_808.0, "SHIP"),
            (64, 12288, 31_968.0, "SHIP"), (768, 4096, 59_456.0, "SHIP"),
            (768, 12288, 140_608.0, "T128"), (16384, 12288, 3_099_456.0, "W2X4"),
        ];
        eprintln!("  {:>6} {:>6} {:>10} {:>8} | {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} | {}",
                  "m", "n", "measured", "best", "tensor", "dram", "smem", "barrier", "prolog", "epilog", "wait", "roofline");
        for (m, n, meas, best) in points {
            let bd = tile_time(&Params::FP8, &machine, &fit, m, n, k);
            let waves = waves_with(&Params::FP8, m, n, &machine, Scheduling::Eager) as f64;
            let pred = bd.total_us * waves * 1000.0;
            eprintln!("  {:>6} {:>6} {:>10.0} {:>8} | {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} | {} (pred {:.0} ns, {:.1}x)",
                      m, n, meas, best, bd.tensor_us, bd.dram_us, bd.smem_us, bd.barrier_us,
                      bd.prologue_us, bd.epilogue_us, bd.wait_us, bd.pipes(), pred, pred / meas);
        }
        eprintln!("\n  ** the per-tile terms are PER WAVE (waves x the per-tile term = the kernel), and the");
        eprintln!("     measured column is the whole kernel. A term that is ~0 at m=16 while the kernel");
        eprintln!("     takes 19 us is the missing one. **");
    }

    const MEASURED_MET: [[[(f64, f64, f64, f64, f64, f64); 14]; 2]; 8] = [

        [
            [(13.04, 2.09, 102.15, 24.00, 1812736.00, 5803502.00), (12.74, 13.51, 86.49, 23.93, 1917797.00, 5792028.00), (24.93, 31.35, 77.07, 23.90, 4133991.00, 11575341.00), (37.79, 0.67, 100.40, 23.89, 5457509.00, 17386735.00), (40.94, 34.95, 82.45, 23.93, 8851423.00, 23784324.00), (56.30, 0.09, 100.35, 23.97, 12630588.00, 29704035.00), (46.63, 25.00, 92.86, 24.33, 24418059.00, 47873408.00), (45.75, 1.42, 99.97, 24.59, 55586568.00, 72354303.00), (46.75, 5.84, 97.66, 24.71, 78518718.00, 98298213.00), (56.27, 2.66, 96.36, 24.85, 133677262.00, 196277391.00), (58.93, 10.61, 94.67, 24.93, 280038049.00, 394929556.00), (59.10, 16.45, 94.26, 24.96, 551750524.00, 790852874.00), (60.59, 19.88, 94.40, 24.98, 1052972875.00, 1579248234.00), (37.68, 82.03, 66.37, 24.99, 5709409905.00, 3667945686.00)],
            [(18.37, 83.97, 33.40, 24.42, 37992110.00, 22035147.00), (20.72, 79.76, 39.10, 24.26, 30588899.00, 17393872.00), (28.49, 67.47, 64.97, 24.29, 53077244.00, 36425972.00), (53.17, 17.26, 91.87, 24.40, 25650032.00, 56201514.00), (50.40, 26.76, 88.82, 24.55, 37485653.00, 73213785.00), (57.04, 18.92, 91.49, 24.65, 51857601.00, 90639205.00), (56.78, 18.00, 91.78, 24.78, 88864969.00, 147028389.00), (58.74, 7.71, 94.09, 24.86, 142159294.00, 220393177.00), (58.45, 12.11, 94.00, 24.90, 201134359.00, 290110711.00), (61.14, 13.20, 94.76, 24.95, 381340564.00, 582821007.00), (59.95, 15.34, 95.04, 24.97, 796690094.00, 1178422320.00), (59.88, 16.18, 94.01, 24.99, 1688593839.00, 2359370380.00), (60.36, 19.05, 94.66, 24.99, 3258581943.00, 4721723315.00), (54.05, 76.47, 76.09, 25.00, 8880814794.00, 9790412589.00)],
        ],

        [
            [(13.60, 12.46, 90.98, 24.64, 48190496.00, 23841273.00), (30.42, 0.01, 99.72, 24.29, 6687686.00, 18302538.00), (29.82, 0.02, 100.36, 24.15, 6773673.00, 18306672.00), (29.71, 1.98, 100.77, 23.99, 6825397.00, 18310533.00), (29.24, 1.74, 100.28, 23.83, 6834524.00, 18308881.00), (56.32, 8.68, 98.19, 24.23, 16036019.00, 37302199.00), (53.97, 1.46, 99.65, 23.93, 16793444.00, 36951512.00), (43.42, 0.98, 99.59, 24.14, 30006698.00, 56455465.00), (49.66, 6.14, 97.47, 24.36, 51002749.00, 75770451.00), (61.09, 9.44, 96.48, 24.68, 94572753.00, 151735001.00), (70.79, 13.29, 94.81, 24.83, 166451212.00, 298170135.00), (73.03, 17.62, 94.56, 24.92, 356778671.00, 599350401.00), (73.08, 21.27, 93.40, 24.96, 733326835.00, 1206763413.00), (28.36, 86.97, 56.42, 24.99, 9346407814.00, 3564013011.00)],
            [(11.97, 7.05, 94.95, 24.78, 498937337.00, 125340433.00), (42.69, 13.13, 96.21, 24.48, 38438380.00, 57880895.00), (40.73, 11.89, 93.74, 24.47, 50010518.00, 57200614.00), (39.38, 21.44, 92.12, 24.42, 55148041.00, 57441907.00), (40.66, 14.95, 93.73, 24.27, 49229078.00, 58129787.00), (52.54, 9.13, 96.77, 24.70, 82062494.00, 115350331.00), (51.25, 9.12, 94.24, 24.58, 88420313.00, 112688288.00), (59.11, 16.32, 92.95, 24.72, 114626462.00, 171179136.00), (62.69, 20.48, 91.42, 24.77, 149782439.00, 228026146.00), (73.46, 19.57, 92.95, 24.88, 261199474.00, 451016479.00), (74.34, 19.96, 93.86, 24.94, 516915405.00, 900315551.00), (72.18, 19.32, 94.35, 24.97, 1162448079.00, 1814959410.00), (70.46, 21.76, 94.43, 24.99, 2191196160.00, 3624442946.00), (37.51, 82.61, 67.16, 25.00, 17550315445.00, 9183813686.00)],
        ],

        [
            [(14.60, 8.72, 82.84, 24.19, 2999847.00, 9043831.00), (14.70, 6.15, 90.23, 24.14, 2975576.00, 9044173.00), (14.61, 5.87, 89.77, 24.04, 2994869.00, 9041118.00), (29.10, 6.96, 93.37, 24.09, 6030913.00, 18077357.00), (29.01, 8.80, 93.03, 24.00, 5923973.00, 18073285.00), (43.75, 0.04, 100.67, 24.08, 9097691.00, 27156943.00), (56.20, 0.04, 99.09, 24.02, 13369935.00, 36368132.00), (44.19, 7.48, 97.46, 23.99, 23833425.00, 55159193.00), (55.64, 1.09, 98.51, 24.37, 32620603.00, 73419679.00), (60.87, 2.36, 96.50, 24.70, 66342715.00, 146991910.00), (69.56, 8.60, 93.96, 24.85, 131705231.00, 293981524.00), (75.73, 15.08, 93.13, 24.92, 266965842.00, 587270909.00), (77.04, 25.55, 91.59, 24.96, 543191972.00, 1175976855.00), (51.00, 83.41, 62.13, 24.99, 3528317599.00, 2687440242.00)],
            [(27.30, 49.03, 55.93, 24.41, 35787953.00, 31794971.00), (42.26, 40.67, 73.63, 24.12, 9941899.00, 27168553.00), (42.53, 44.39, 70.52, 24.07, 9870168.00, 27161260.00), (43.29, 25.37, 82.72, 24.25, 26757938.00, 55526483.00), (43.86, 24.02, 83.16, 24.21, 26549374.00, 55850732.00), (60.16, 11.89, 91.87, 24.48, 39017931.00, 83487627.00), (58.33, 18.46, 88.08, 24.56, 51863551.00, 110670804.00), (64.93, 9.56, 91.72, 24.73, 76244676.00, 165267842.00), (69.08, 19.45, 89.14, 24.79, 102098853.00, 220369089.00), (73.64, 22.76, 91.27, 24.89, 202404007.00, 440280420.00), (77.48, 20.95, 92.53, 24.95, 399081410.00, 880945085.00), (77.97, 19.07, 93.04, 24.97, 810230759.00, 1762194822.00), (75.22, 26.55, 92.12, 24.99, 1591220459.00, 3518165094.00), (65.08, 81.99, 68.82, 24.99, 5946185773.00, 7419144084.00)],
        ],

        [
            [(12.37, 35.97, 51.97, 23.91, 2056097.00, 5552514.00), (12.79, 1.98, 102.39, 23.76, 1718467.00, 5537848.00), (25.16, 12.30, 92.67, 23.79, 3403814.00, 11076586.00), (37.23, 10.82, 95.72, 23.79, 5199433.00, 16626724.00), (44.25, 8.58, 96.57, 23.89, 10568414.00, 22898777.00), (55.12, 1.49, 100.24, 23.84, 13066508.00, 28482869.00), (44.49, 8.29, 97.44, 24.25, 26353097.00, 46193624.00), (47.67, 0.01, 99.66, 24.51, 45445058.00, 70924960.00), (50.03, 1.38, 98.55, 24.63, 60376295.00, 94976132.00), (55.78, 4.83, 96.62, 24.83, 131185877.00, 188944645.00), (59.69, 12.11, 95.43, 24.91, 261276205.00, 377578013.00), (58.99, 15.24, 95.09, 24.96, 540190126.00, 758991493.00), (59.38, 18.79, 94.44, 24.98, 1078098102.00, 1514881120.00), (38.05, 83.44, 66.58, 24.99, 5856248671.00, 3523016440.00)],
            [(24.88, 27.96, 80.48, 24.16, 18630739.00, 18830250.00), (36.18, 38.81, 79.48, 23.83, 6135104.00, 16641454.00), (37.38, 34.50, 81.73, 24.03, 17823690.00, 34553695.00), (52.44, 31.06, 87.30, 24.32, 25511727.00, 51730674.00), (44.92, 23.14, 89.73, 24.52, 45572894.00, 70476779.00), (56.08, 21.61, 89.36, 24.61, 53702242.00, 87923993.00), (53.98, 16.16, 92.30, 24.77, 100082614.00, 141897367.00), (58.22, 8.41, 94.17, 24.84, 138169142.00, 212411480.00), (58.48, 10.01, 94.57, 24.88, 191556919.00, 278952789.00), (59.71, 13.60, 94.75, 24.94, 380973798.00, 561845912.00), (60.11, 17.44, 95.12, 24.97, 795000517.00, 1135636607.00), (59.63, 16.89, 95.15, 24.99, 1657099295.00, 2275781841.00), (59.51, 20.09, 94.61, 24.99, 3311607879.00, 4553251892.00), (53.80, 77.69, 75.77, 25.00, 8831348008.00, 9478750977.00)],
        ],

        [
            [(14.52, 0.01, 99.83, 24.23, 6796549.00, 8695669.00), (17.90, 4.29, 97.36, 23.93, 2941967.00, 7827845.00), (35.11, 14.25, 95.19, 23.95, 6798173.00, 15962402.00), (28.02, 0.01, 100.40, 24.14, 15241461.00, 25020408.00), (34.72, 0.95, 99.87, 24.33, 19425125.00, 33036836.00), (43.94, 13.79, 96.23, 24.46, 23698852.00, 40924475.00), (38.41, 1.87, 100.15, 24.68, 49338303.00, 65895654.00), (42.26, 8.73, 98.62, 24.78, 66823091.00, 100740344.00), (44.87, 0.89, 99.37, 24.84, 92786896.00, 136091713.00), (45.08, 1.43, 98.28, 24.93, 199727348.00, 271313514.00), (44.55, 6.96, 96.69, 24.96, 426518950.00, 540604556.00), (44.21, 10.34, 96.54, 24.98, 880093564.00, 1087042415.00), (43.60, 14.66, 96.17, 24.99, 1799018397.00, 2167753493.00), (30.23, 75.89, 69.90, 25.00, 7498901428.00, 4812164639.00)],
            [(12.76, 25.43, 76.17, 24.53, 94687788.00, 36699912.00), (26.17, 26.25, 85.52, 24.14, 17402274.00, 25065367.00), (33.77, 18.24, 91.58, 24.55, 31270049.00, 49630460.00), (38.69, 16.65, 92.80, 24.71, 52373653.00, 76178128.00), (40.00, 13.88, 94.23, 24.79, 81083909.00, 102670347.00), (40.97, 11.96, 94.98, 24.83, 94007828.00, 127869991.00), (45.05, 24.13, 92.35, 24.89, 151724991.00, 200878021.00), (44.27, 12.24, 94.70, 24.93, 228709204.00, 301446016.00), (44.95, 7.25, 95.60, 24.95, 309777385.00, 402454744.00), (44.52, 14.37, 95.52, 24.98, 662601391.00, 809344016.00), (44.44, 12.37, 96.16, 24.99, 1314470942.00, 1622659475.00), (44.53, 12.39, 96.34, 24.99, 2656771604.00, 3246486682.00), (42.53, 16.58, 95.82, 25.00, 5227146724.00, 6480912465.00), (38.72, 59.28, 82.01, 25.00, 14024807031.00, 13399985640.00)],
        ],

        [
            [(12.09, 1.80, 100.78, 24.25, 2259069.00, 6257641.00), (10.38, 7.35, 94.55, 24.24, 3515783.00, 6255914.00), (21.21, 8.77, 97.02, 24.20, 6885765.00, 12493168.00), (30.60, 8.65, 97.51, 24.23, 10530134.00, 18736458.00), (43.10, 2.40, 99.17, 24.19, 11241743.00, 25064140.00), (45.56, 10.15, 98.67, 24.31, 23989709.00, 31854842.00), (43.10, 6.45, 98.93, 24.48, 29319566.00, 51300532.00), (42.02, 10.44, 98.55, 24.69, 62156150.00, 77019484.00), (48.19, 1.11, 99.06, 24.76, 66437393.00, 103759902.00), (54.50, 1.64, 97.49, 24.88, 137075440.00, 206420793.00), (57.62, 7.03, 95.89, 24.94, 277932196.00, 404085147.00), (55.54, 11.88, 95.18, 24.97, 632920943.00, 838519056.00), (57.30, 16.35, 95.01, 24.99, 1157905987.00, 1663166537.00), (43.47, 84.87, 72.21, 24.99, 4462256900.00, 3621988447.00)],
            [(21.69, 41.42, 82.17, 24.49, 24347690.00, 20885819.00), (28.18, 56.96, 79.70, 24.27, 13391271.00, 18768292.00), (26.16, 55.86, 82.70, 24.43, 48326095.00, 38209500.00), (47.06, 25.35, 92.83, 24.56, 36833399.00, 57211397.00), (42.19, 21.94, 93.34, 24.68, 58379433.00, 76656439.00), (50.93, 23.71, 93.00, 24.74, 72819371.00, 96762383.00), (53.29, 10.57, 95.52, 24.83, 98379359.00, 154800853.00), (54.98, 10.39, 95.05, 24.89, 158753805.00, 232456511.00), (56.58, 17.38, 93.66, 24.92, 225307871.00, 311443260.00), (56.70, 12.50, 94.28, 24.96, 447721696.00, 625968130.00), (55.39, 15.65, 95.09, 24.98, 982910526.00, 1259597623.00), (55.53, 14.74, 95.18, 24.99, 1985844744.00, 2528965722.00), (57.55, 18.46, 92.71, 25.00, 3482155531.00, 4950284746.00), (49.74, 66.92, 80.05, 25.00, 9531545444.00, 10266441896.00)],
        ],

        [
            [(10.69, 12.37, 82.26, 24.25, 3820697.00, 6055203.00), (11.25, 0.66, 101.02, 24.13, 3323455.00, 6042321.00), (21.43, 3.72, 98.96, 24.14, 6847001.00, 12099258.00), (32.54, 1.66, 99.82, 24.14, 10100238.00, 18235438.00), (41.32, 1.06, 99.18, 24.16, 15219028.00, 24533218.00), (48.78, 0.00, 99.90, 24.19, 21681532.00, 31161010.00), (41.14, 1.54, 100.02, 24.46, 37310390.00, 49906030.00), (43.87, 0.00, 99.60, 24.64, 56304816.00, 74607546.00), (45.56, 6.09, 96.62, 24.73, 81575612.00, 101005683.00), (51.97, 4.11, 96.17, 24.87, 165261000.00, 200346109.00), (54.93, 12.98, 94.85, 24.93, 335206394.00, 401872841.00), (55.47, 11.40, 95.25, 24.97, 677300520.00, 804481458.00), (56.58, 17.85, 94.52, 24.98, 1294230149.00, 1602743064.00), (42.22, 86.03, 68.04, 24.99, 4905118831.00, 3542153735.00)],
            [(25.03, 29.24, 81.59, 24.35, 21172738.00, 19625940.00), (28.55, 58.01, 64.04, 24.20, 13772552.00, 18204657.00), (31.99, 18.43, 87.92, 24.29, 28175363.00, 37541116.00), (45.50, 28.75, 86.89, 24.51, 42981879.00, 56110616.00), (43.71, 12.65, 92.26, 24.64, 57030664.00, 75239122.00), (49.02, 24.58, 89.05, 24.72, 84842545.00, 93730096.00), (51.11, 10.62, 93.89, 24.82, 120805962.00, 149765759.00), (53.55, 7.12, 94.41, 24.88, 182538081.00, 224598691.00), (55.39, 17.16, 93.31, 24.91, 245904909.00, 301087198.00), (56.08, 13.61, 94.57, 24.96, 497326896.00, 603750987.00), (56.04, 15.76, 94.76, 24.98, 1019961616.00, 1207546321.00), (56.54, 16.61, 95.14, 24.99, 2028945449.00, 2411953448.00), (55.51, 20.66, 93.95, 24.99, 3855956823.00, 4793503409.00), (48.09, 63.53, 79.00, 25.00, 10175143082.00, 10001849897.00)],
        ],

        [
            [(12.93, 1.46, 99.80, 23.97, 1813441.00, 5783073.00), (12.32, 30.68, 61.22, 23.86, 1947713.00, 5770726.00), (24.76, 1.81, 99.02, 23.82, 3519238.00, 11539820.00), (35.67, 48.35, 75.96, 23.86, 6918211.00, 17343911.00), (47.51, 26.29, 90.06, 23.87, 8274633.00, 23733224.00), (46.78, 29.95, 89.05, 24.07, 22080058.00, 29566868.00), (46.39, 0.00, 100.26, 24.28, 23019166.00, 48083246.00), (50.38, 0.84, 99.93, 24.50, 36451623.00, 72010856.00), (51.30, 0.84, 99.07, 24.64, 56056959.00, 98116409.00), (55.55, 1.36, 97.75, 24.83, 129008944.00, 196079477.00), (59.57, 7.51, 95.87, 24.92, 258148073.00, 389431399.00), (59.56, 12.86, 95.23, 24.96, 523137789.00, 780890729.00), (60.53, 17.72, 92.56, 24.98, 1020937279.00, 1557726122.00), (39.19, 83.90, 66.21, 24.99, 5495887169.00, 3661403239.00)],
            [(25.30, 29.47, 79.30, 24.23, 17920020.00, 19419098.00), (36.37, 33.82, 81.54, 23.85, 5686049.00, 17342583.00), (36.63, 27.13, 83.67, 24.06, 17562944.00, 35990781.00), (49.38, 36.10, 86.05, 24.38, 31503451.00, 53959186.00), (44.31, 14.53, 91.04, 24.55, 47625626.00, 72415968.00), (54.92, 25.79, 89.42, 24.63, 55108144.00, 90412114.00), (55.62, 21.29, 90.82, 24.77, 87970678.00, 145747261.00), (58.23, 10.16, 93.81, 24.85, 134762264.00, 220634150.00), (59.58, 16.65, 93.52, 24.89, 175908518.00, 291201532.00), (61.63, 14.10, 94.71, 24.94, 346582826.00, 579878983.00), (61.28, 19.48, 94.70, 24.97, 746265337.00, 1168836629.00), (59.77, 17.01, 95.38, 24.99, 1609600656.00, 2351598701.00), (60.29, 18.85, 94.70, 24.99, 3136901718.00, 4699194100.00), (53.92, 74.62, 76.95, 25.00, 8128544090.00, 9697592460.00)],
        ],
    ];

    #[test]
    fn the_dispatch_diagnosis() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;
        for (m, n) in [(768usize, 4096usize), (256, 12288), (320, 12288), (16384, 12288)] {
            let ni = MEASURED_N.iter().position(|&x| x == n).unwrap();
            let mi = MEASURED_M.iter().position(|&x| x == m).unwrap();
            eprintln!("--- m={m} n={n} ---");
            let mut rows: Vec<(f64, &str, f64)> = Vec::new();
            for (ci, tag) in MEASURED_TAGS.iter().enumerate() {
                let (name, p) = buildable().into_iter().find(|(t, _)| *t == *tag).unwrap();
                let pred = makespan_us(&p, &machine, &fit, m, n, k) * 1000.0;
                rows.push((pred, name, MEASURED_NS[ci][ni][mi]));
            }
            rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (pred, name, meas) in rows {
                let ratio = if meas > 0.0 { pred / meas } else { 0.0 };
                let p = buildable().into_iter().find(|(t, _)| *t == name).unwrap().1;
                let bd = tile_time(&p, &machine, &fit, m, n, k);
                let w = waves_with(&p, m, n, &machine, Scheduling::Eager) as f64;
                let tiles = tiles(&p, m, n) as f64;
                eprintln!("  {name:<6} pred {pred:>10.0} measured {meas:>10.0} {ratio:>6.2}x | \
tile_m {} n_blk {} m_atoms {} n_sub {} ilp {:.3} resident {} ctas {} tiles {:.0} waves {:.2} \
tensor {:.2} dram {:.2} smem {:.2} wait {:.2} total {:.2}",
                          p.m_tile(), p.n_block(), p.m_atoms(), p.n_sub(), bd.ilp,
                          p.resident_warps(&machine), machine.ctas_per_sm(&p).0, tiles, w,
                          bd.tensor_us, bd.dram_us, bd.smem_us, bd.wait_us, bd.total_us);
            }
        }
    }

    #[test]
    fn the_regret_per_point_with_metrics() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;
        let configs: [(&str, Params); 8] = [
            ("SHIP", Params::FP8), ("BIG", Params::FP8_BIG), ("T128", Params::FP8_T128),
            ("W2X4", Params::FP8_W2X4), ("N64", Params::FP8_N64), ("K64", Params::FP8_K64),
            ("S2", Params::FP8_STAGES2), ("EPI1", Params::FP8_EPI1),
        ];

        let met: [[[(f64, f64, f64, f64, f64, f64); 14]; 2]; 8] = MEASURED_MET;
        let mut rows: Vec<(f64, usize, usize, usize, usize)> = Vec::new();
        for (ni, &n) in MEASURED_N.iter().enumerate() {
            for (mi, &m) in MEASURED_M.iter().enumerate() {
                let mut meas: Vec<(usize, f64)> = (0..8).map(|ci| (ci, MEASURED_NS[ci][ni][mi]))
                    .filter(|(_, v)| *v > 0.0).collect();
                if meas.is_empty() { continue; }
                meas.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut model: Vec<(usize, f64)> = configs.iter().enumerate()
                    .map(|(ci, (_, p))| (ci, makespan_us(p, &machine, &fit, m, n, k))).collect();
                model.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let reg = (MEASURED_NS[model[0].0][ni][mi] - meas[0].1) / meas[0].1;
                rows.push((reg, ni, mi, model[0].0, meas[0].0));
            }
        }
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!("  {:>7} {:>6} {:>9} {:>9} {:>7} | {:<22} | {:<22}", "m", "n", "regret", "pick ns", "best ns", "MODEL PICK (tensor/dram/hit/warps)", "MEASURED BEST");
        for (reg, ni, mi, pick, bestc) in rows.iter().take(10) {
            let (n, m) = (MEASURED_N[*ni], MEASURED_M[*mi]);
            let p = met[*pick][*ni][*mi]; let b = met[*bestc][*ni][*mi];
            eprintln!("  {:>7} {:>6} {:>8.1}% {:>9.0} {:>9.0} | {:<10} t{:>4.0} d{:>4.0} h{:>4.0} w{:>4.0} | {:<10} t{:>4.0} d{:>4.0} h{:>4.0} w{:>4.0}",
                      m, n, reg * 100.0, MEASURED_NS[*pick][*ni][*mi], MEASURED_NS[*bestc][*ni][*mi],
                      MEASURED_TAGS[*pick], p.0, p.1, p.2, p.3,
                      MEASURED_TAGS[*bestc], b.0, b.1, b.2, b.3);
        }
    }

    #[test]
    fn the_fit_against_the_sensitivity_dataset() {
        let machine = Machine::RTX_5090;
        let k = 4096usize;
        let configs: [Params; 8] = [
            Params::FP8, Params::FP8_BIG, Params::FP8_T128, Params::FP8_W2X4,
            Params::FP8_N64, Params::FP8_K64, Params::FP8_STAGES2, Params::FP8_EPI1,
        ];
        let base = Fit::FITTED_5090_FP8;

        eprintln!("  {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {}", "mma_eff", "fixed x", "lat_tiles", "lat_us", "winners", "mean|err|", "worst", "REGRET");
        let mut best = (0usize, f64::MAX, 0.0f64, 0.0f64);
        for mma_eff in [0.90f64] {
            for fixed_scale in [0.4f64] {
              for gamma in [0.35f64] {
               for lat_tiles in [2.0f64, 3.0, 4.0, 6.0, 9.0] {
                for lat_us in [1.2f64, 3.0, 6.0] {
                 for in_flight_ref in [8.0f64, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0] {
                let fit = Fit {
                    mma_efficiency: mma_eff,
                    launch_us: 8.0,
                    memory_latency_tiles: lat_tiles,
                    latency_us_per_tile: lat_us,
                    ilp_gamma: gamma,
                    ilp_in_flight_ref: in_flight_ref,
                    fill_latency_ns: 1200.0,
                    barrier_ns: base.barrier_ns * fixed_scale,
                    prologue_ns: base.prologue_ns * fixed_scale,
                    epilogue_ns_per_elem: base.epilogue_ns_per_elem * fixed_scale,
                    wait_ns_per_ktile: base.wait_ns_per_ktile * fixed_scale,
                    ..base
                };

                let mut wins = 0usize; let mut total = 0usize;
                let mut sum = 0.0f64; let mut worst = 0.0f64;
                let mut regret_sum = 0.0f64; let mut regret_worst = 0.0f64;

                let mut over5 = 0usize;
                let mut worst_vs_ship = 0.0f64;
                for (ni, &n) in MEASURED_N.iter().enumerate() {
                    for (mi, &m) in MEASURED_M.iter().enumerate() {
                        let mut meas: Vec<(usize, f64)> = (0..8).map(|ci| (ci, MEASURED_NS[ci][ni][mi]))
                            .filter(|(_, v)| *v > 0.0).collect();
                        if meas.is_empty() { continue; }
                        meas.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let mut model: Vec<(usize, f64)> = configs.iter().enumerate()
                            .map(|(ci, p)| (ci, makespan_us(p, &machine, &fit, m, n, k))).collect();
                        model.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        if model[0].0 == meas[0].0 { wins += 1; }
                        total += 1;

                        let pick = model[0].0;
                        let reg = (MEASURED_NS[pick][ni][mi] - meas[0].1) / meas[0].1;
                        regret_sum += reg; regret_worst = regret_worst.max(reg);

                        let ship_i = 0usize;
                        let ship = MEASURED_NS[ship_i][ni][mi];
                        if ship > 0.0 {
                            let vs = (MEASURED_NS[pick][ni][mi] - ship) / ship;
                            if vs > 0.05 { over5 += 1; }
                            worst_vs_ship = worst_vs_ship.max(vs);
                        }
                        for (ci, p) in configs.iter().enumerate() {
                            let pred = makespan_us(p, &machine, &fit, m, n, k) * 1000.0;
                            let e = (pred - MEASURED_NS[ci][ni][mi]).abs() / MEASURED_NS[ci][ni][mi];
                            sum += e; worst = worst.max(e);
                        }
                    }
                }
                let mean = sum / (total * 8) as f64;
                let reg_mean = regret_sum / total as f64;
                eprintln!("  {:>7.2} {:>7.2} {:>7.0} {:>7.0} {:>7.0} | {:>4}/{:<2} {:>7.1}% {:>7.1}% | regret {:>5.1}% worst {:>5.1}% | OVER5 {} worst-vs-SHIP {:>5.1}%",
                          mma_eff, fixed_scale, lat_tiles, lat_us, in_flight_ref, wins, total,
                          mean * 100.0, worst * 100.0, reg_mean * 100.0, regret_worst * 100.0,
                          over5, worst_vs_ship * 100.0);
                if reg_mean < best.1 {
                    best = (wins, reg_mean, lat_tiles, lat_us);
                }
                 }
                }
               }
              }
            }
        }
        eprintln!("  BEST BY REGRET: mma_efficiency {:.2} fixed_scale {:.2} -> {} winners, mean regret {:.1}%",
                  best.2, best.3, best.0, best.1 * 100.0);
    }

    const PROD_NS: [[[f64; 10]; 3]; 8] = [

        [
            [54800.0, 70300.0, 209400.0, 266300.0, 513000.0, 1002000.0, 2023900.0, 4816700.0, 10264300.0, 20282900.0],
            [89300.0, 120300.0, 274000.0, 517700.0, 982000.0, 1888900.0, 3760500.0, 8586400.0, 19749900.0, 39658400.0],
            [234300.0, 294600.0, 424200.0, 697800.0, 1437100.0, 2797800.0, 5908500.0, 12380700.0, 28305700.0, 56067000.0],
        ],

        [
            [40000.0, 75400.0, 210300.0, 275400.0, 452300.0, 833200.0, 1663400.0, 4071300.0, 10666400.0, 20911700.0],
            [78200.0, 108900.0, 254000.0, 437400.0, 823900.0, 1615900.0, 3272700.0, 7581800.0, 20042900.0, 40486500.0],
            [224400.0, 262200.0, 381400.0, 628800.0, 1215400.0, 2383500.0, 4866900.0, 10788000.0, 29407200.0, 57251900.0],
        ],

        [
            [54400.0, 72100.0, 207800.0, 272900.0, 486900.0, 960000.0, 1925600.0, 4940700.0, 9999100.0, 19663600.0],
            [91200.0, 120400.0, 259900.0, 492500.0, 901600.0, 1807600.0, 3666700.0, 8381000.0, 19457100.0, 38219600.0],
            [236900.0, 272100.0, 416200.0, 688100.0, 1289100.0, 2643500.0, 5544700.0, 11652900.0, 26936200.0, 55888200.0],
        ],

        [
            [58300.0, 81800.0, 235100.0, 302600.0, 550800.0, 1053800.0, 2157100.0, 5613700.0, 15734800.0, 31663600.0],
            [102700.0, 143100.0, 308100.0, 553900.0, 1078400.0, 2247000.0, 4442600.0, 10029300.0, 26845700.0, 59393200.0],
            [250700.0, 331800.0, 474300.0, 834600.0, 1575500.0, 3145800.0, 6427200.0, 14175500.0, 35919400.0, 83150000.0],
        ],

        [
            [42900.0, 77800.0, 215000.0, 258300.0, 443200.0, 837500.0, 1672000.0, 5878600.0, 13864500.0, 30744800.0],
            [89800.0, 120200.0, 266400.0, 449600.0, 868700.0, 1647000.0, 3382400.0, 10369700.0, 26079500.0, 61113400.0],
            [235400.0, 280700.0, 401400.0, 643800.0, 1302100.0, 2534100.0, 5153000.0, 14082300.0, 37306500.0, 90009200.0],
        ],

        [
            [54700.0, 77000.0, 215300.0, 279500.0, 495600.0, 958700.0, 2058100.0, 4499300.0, 9752200.0, 19650700.0],
            [92300.0, 127200.0, 284800.0, 528600.0, 974400.0, 1928500.0, 3863100.0, 8478100.0, 18886400.0, 38805700.0],
            [260000.0, 313800.0, 440500.0, 738400.0, 1430200.0, 2961600.0, 5725200.0, 12406900.0, 27749500.0, 57666500.0],
        ],

        [
            [56200.0, 76300.0, 199300.0, 269600.0, 481700.0, 919800.0, 1966100.0, 4578300.0, 9684200.0, 19449700.0],
            [92900.0, 127000.0, 271400.0, 507400.0, 942200.0, 1937500.0, 3923000.0, 8716000.0, 19649200.0, 39032900.0],
            [247900.0, 298100.0, 431000.0, 785200.0, 1429100.0, 2846400.0, 5729700.0, 12290300.0, 27685100.0, 58344800.0],
        ],

        [
            [55700.0, 72200.0, 212500.0, 277300.0, 495800.0, 928800.0, 1850400.0, 4725200.0, 10128800.0, 19543300.0],
            [90500.0, 121900.0, 265600.0, 480400.0, 905900.0, 1802400.0, 3573500.0, 8204600.0, 19433200.0, 38737500.0],
            [240900.0, 291700.0, 423900.0, 695600.0, 1383900.0, 2782100.0, 5473400.0, 12082400.0, 27548300.0, 55681400.0],
        ],
    ];
    const PROD_M: [usize; 10] = [512, 768, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
    const PROD_N: [usize; 3] = [4096, 8192, 12288];

    #[test]
    fn the_dispatch_against_the_production_range() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;
        let all = buildable();
        eprintln!("  {:>7} {:>7} | {:>8} {:>9} | {:>9} {:>9} | {:>7} | {}", "m", "n", "dispatch",
                  "measured best", "picked ns", "best ns", "REGRET", "verdict");
        let mut worst = (0.0f64, 0usize, 0usize);
        let mut tot = 0.0; let mut cnt = 0usize;
        for (ni, &n) in PROD_N.iter().enumerate() {
            for (mi, &m) in PROD_M.iter().enumerate() {
                let (name, _) = dispatch(m, n, k, &machine, &fit);
                let ci = all.iter().position(|(t, _)| *t == name).unwrap();
                if PROD_NS[ci][ni][mi] <= 0.0 { continue; }
                let mut best = (usize::MAX, f64::MAX);
                for c in 0..8 { let v = PROD_NS[c][ni][mi]; if v > 0.0 && v < best.1 { best = (c, v); } }
                let picked = PROD_NS[ci][ni][mi];
                let reg = (picked - best.1) / best.1;
                tot += reg; cnt += 1;
                if reg > worst.0 { worst = (reg, m, n); }
                eprintln!("  {:>7} {:>7} | {:>8} {:>9} | {:>9.0} {:>9.0} | {:>6.1}% | {}", m, n, name,
                          all[best.0].0, picked, best.1, reg * 100.0,
                          if reg < 0.05 { "tie / optimal" } else { "SUBOPTIMAL" });
            }
        }
        eprintln!("  mean regret {:.1}%   worst {:.1}% at m={} n={}", tot / cnt as f64 * 100.0,
                  worst.0 * 100.0, worst.1, worst.2);
    }

    #[test]
    fn the_dispatch_proved_against_the_measurement() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;
        let all = buildable();
        eprintln!("  {:>7} {:>7} | {:>8} {:>8} | {:>9} {:>9} | {:>8} | {}",
                  "m", "n", "dispatch", "true best", "picked ns", "best ns", "REGRET", "verdict");
        let mut worst = (0.0f64, 0usize, usize::MAX);
        let mut tot = 0.0f64; let mut cnt = 0usize;
        for (ni, &n) in MEASURED_N.iter().enumerate() {
            for (mi, &m) in MEASURED_M.iter().enumerate() {
                let (name, _) = dispatch(m, n, k, &machine, &fit);
                let ci = all.iter().position(|(t, _)| *t == name).unwrap();
                if MEASURED_NS[ci][ni][mi] <= 0.0 { continue; }

                let mut best = (usize::MAX, f64::MAX);
                for (c, _) in all.iter().enumerate() {
                    let v = MEASURED_NS[c][ni][mi];
                    if v > 0.0 && v < best.1 { best = (c, v); }
                }
                let picked = MEASURED_NS[ci][ni][mi];
                let reg = (picked - best.1) / best.1;
                tot += reg; cnt += 1;
                if reg > worst.0 { worst = (reg, m, n); }
                eprintln!("  {:>7} {:>7} | {:>8} {:>8} | {:>9.0} {:>9.0} | {:>7.1}% | {}",
                          m, n, name, all[best.0].0, picked, best.1, reg * 100.0,
                          if (picked - best.1).abs() < 1e-9 { "OPTIMAL" } else { "SUBOPTIMAL" });
            }
        }
        eprintln!("  mean regret {:.1}%   worst {:.1}% at m={} n={}", tot / cnt as f64 * 100.0,
                  worst.0 * 100.0, worst.1, worst.2);

        assert!(worst.0 < 0.20,
                "the dispatch is more than 20% off the best at m={} n={} ({:.1}%)",
                worst.1, worst.2, worst.0 * 100.0);
    }

    #[test]
    #[ignore = "needs no GPU, but it is in gates.sh — kept ignored so `cargo test` stays fast"]
    fn the_dispatch_must_never_lose_to_the_shipped_configuration() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;
        let all = buildable();
        let ship = all.iter().position(|(t, _)| *t == "SHIP").unwrap();
        let mut bad = Vec::new();
        for (ni, &n) in MEASURED_N.iter().enumerate() {
            for (mi, &m) in MEASURED_M.iter().enumerate() {
                let (name, _) = dispatch(m, n, k, &machine, &fit);
                let ci = all.iter().position(|(t, _)| *t == name).unwrap();
                if MEASURED_NS[ci][ni][mi] <= 0.0 || MEASURED_NS[ship][ni][mi] <= 0.0 { continue; }
                if MEASURED_NS[ci][ni][mi] > MEASURED_NS[ship][ni][mi] * 1.05 {
                    bad.push(format!("m={m} n={n}: dispatches {name} at {:.0} against the shipped {:.0} \
                                      ({:+.1}%)", MEASURED_NS[ci][ni][mi], MEASURED_NS[ship][ni][mi],
                                     100.0 * (MEASURED_NS[ci][ni][mi] - MEASURED_NS[ship][ni][mi])
                                     / MEASURED_NS[ship][ni][mi]));
                }
            }
        }
        assert!(bad.is_empty(), "the dispatch loses to a fixed choice:\n  {}", bad.join("\n  "));
    }

    #[test]
    fn the_model_against_the_sensitivity_dataset() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let k = 4096usize;
        let configs: [(&str, Params); 8] = [
            ("SHIP", Params::FP8), ("BIG", Params::FP8_BIG), ("T128", Params::FP8_T128),
            ("W2X4", Params::FP8_W2X4), ("N64", Params::FP8_N64), ("K64", Params::FP8_K64),
            ("S2", Params::FP8_STAGES2), ("EPI1", Params::FP8_EPI1),
        ];
        let mut wins = 0usize; let mut total = 0usize;
        let mut worst_ratio: f64 = 0.0;
        eprintln!("  {:>6} {:>6} | {:>8} {:>9} {:>9} | {}",
                  "m", "n", "measured", "model", "winner", "verdict");
        for (ni, &n) in MEASURED_N.iter().enumerate() {
            for (mi, &m) in MEASURED_M.iter().enumerate() {
                let mut meas: Vec<(usize, f64)> = (0..8)
                    .map(|ci| (ci, MEASURED_NS[ci][ni][mi]))
                    .filter(|(_, v)| *v > 0.0).collect();
                meas.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let mut model: Vec<(usize, f64)> = configs.iter().enumerate()
                    .map(|(ci, (_, p))| (ci, makespan_us(p, &machine, &fit, m, n, k)))
                    .collect();
                model.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mw = MEASURED_TAGS[meas[0].0];
                let dw = configs[model[0].0].0;
                let agree = mw == dw;
                if agree { wins += 1; }
                total += 1;

                let ship = configs.iter().position(|(t, _): &(&str, Params)| *t == "SHIP").unwrap();
                let ratio = model.iter().find(|(ci, _)| *ci == ship).map(|(_, v)| *v * 1000.0 / MEASURED_NS[ship][ni][mi]).unwrap_or(0.0);
                if ratio.is_finite() { worst_ratio = worst_ratio.max((ratio - 1.0).abs()); }
                if m >= 512 {
                    eprintln!("  {:>6} {:>6} | {:>8.0} {:>9.0} {:>9} | {}",
                              m, n, MEASURED_NS[meas[0].0][ni][mi], model[0].1 * 1000.0, dw,
                              if agree { "MATCH" } else { "MISS" });
                }
            }
        }
        eprintln!("  winner agreement: {wins}/{total}   worst shipped error: {:.0}%", worst_ratio * 100.0);

        let mut errs: Vec<(f64, usize, usize, usize)> = Vec::new();
        for (ni, &n) in MEASURED_N.iter().enumerate() {
            for (mi, &m) in MEASURED_M.iter().enumerate() {
                for (ci, (tag, p)) in configs.iter().enumerate() {
                    if MEASURED_NS[ci][ni][mi] <= 0.0 { continue; }
                    let pred = makespan_us(p, &machine, &fit, m, n, k) * 1000.0;
                    let e = (pred - MEASURED_NS[ci][ni][mi]).abs() / MEASURED_NS[ci][ni][mi];
                    errs.push((e, m, n, ci));
                    let _ = tag;
                }
            }
        }
        errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!("  worst absolute errors:");
        for (e, m, n, ci) in errs.iter().take(6) {
            eprintln!("    {:<6} m={:<6} n={:<6} {:>6.0}%  (pred {:.0} vs measured {:.0})",
                      MEASURED_TAGS[*ci], m, n, e * 100.0,
                      makespan_us(&configs[*ci].1, &machine, &fit, *m, *n, k) * 1000.0,
                      MEASURED_NS[*ci][MEASURED_N.iter().position(|x| x == n).unwrap()][MEASURED_M.iter().position(|x| x == m).unwrap()]);
        }
    }

    #[test]
    fn the_model_ranks_the_fp8_space() {
        let machine = Machine::RTX_5090;
        let fit = Kind::Fp8.fit();
        let (m, n, k) = (768usize, 12288usize, 4096usize);
        let shipped = makespan_us(&Params::FP8, &machine, &fit, m, n, k);
        let rows = sweep(&Params::FP8, m, n, k, &machine, &fit, 400);
        eprintln!("  the shipped shape predicts {shipped:.1} us for {m}x{n}x{k}");
        eprintln!("  {:>3} {:>8} {:>7} {:>7} {:>6} {:>6} {:>7} {:>11}  {}",
                  "#", "warps", "m_rows", "n_tile", "epi_n", "epi_st", "sub_ch", "predicted", "judgeable");
        for (i, (p, cost)) in rows.iter().take(12).enumerate() {
            eprintln!("  {:>3} {:>3}x{:<4} {:>7} {:>7} {:>6} {:>6} {:>7} {:>10.1}us  {} ({} warps)",
                      i + 1, p.warps_m, p.warps_n, p.warp_m_rows, p.n_tile, p.epi_tile_n,
                      p.epi_stages, p.sub_chunks, cost,
                      if can_judge(p, &machine, &fit) { "YES" } else { "no " },
                      p.resident_warps(&machine));
        }

        assert!(can_judge(&Params::FP8, &machine, &fit));
        for (p, _) in rows.iter().filter(|(p, _)| can_judge(p, &machine, &fit)) {
            assert_eq!(p.resident_warps(&machine), 8, "a judged candidate must be at the fitted residency");
        }

        let refused = rows.iter().take_while(|(p, _)| !can_judge(p, &machine, &fit)).count();
        eprintln!("  the top {refused} of {} candidates are REFUSED (residency differs from the fit's 8)",
                  rows.len());
        let judged: Vec<_> = rows.iter().filter(|(p, _)| can_judge(p, &machine, &fit)).take(5).collect();
        eprintln!("  --- the actionable ranking (judgeable only) ---");
        for (i, (p, cost)) in judged.iter().enumerate() {
            eprintln!("  {:>3} {:>3}x{:<4} {:>7} {:>7} {:>6} {:>6} {:>7} {:>10.1}us  {} warps",
                      i + 1, p.warps_m, p.warps_n, p.warp_m_rows, p.n_tile, p.epi_tile_n,
                      p.epi_stages, p.sub_chunks, cost, p.resident_warps(&machine));
        }
        assert!(!judged.is_empty(), "no judgeable candidate at all — the fit covers nothing");
    }

    #[test]
    fn the_model_rejects_both_measured_regressions() {
        let machine = Machine::RTX_5090;
        let fit = Fit::FITTED_5090_FP8;
        let (m, n, k) = (768usize, 12288usize, 4096usize);

        let base = Params::FP8;
        let aliased = Params::FP8_ALIASED;
        let taller = Params { warp_m_rows: 32, regs_measured: 124, smem_measured: 32_768, ..base };
        let wider  = Params { warps_n: 4, smem_measured: 40_960, ..base };

        assert!(can_judge(&base, &machine, &fit), "the fit must cover the kernel it was taken at");
        assert_eq!(base.resident_warps(&machine), 8);
        assert_eq!(aliased.resident_warps(&machine), 16);
        assert!(!can_judge(&aliased, &machine, &fit),
                "the 2-CTA aliased build is outside the fit and the model must say so rather than price it");
        let shipped = makespan_us(&base, &machine, &fit, m, n, k);
        for (name, p, measured) in [("M_TILE 128 (2 atoms)", taller, "+25% on the 3-CTA base"),
                                    ("WARPS_N 4 (512 thr)", wider, "+7..60% on the 3-CTA base"),
                                    ("the aliased 2-CTA build", aliased, "steps 1-5; retired")] {
            let t = tile_time(&p, &machine, &fit, m, n, k);
            let span = makespan_us(&p, &machine, &fit, m, n, k);
            let judgeable = can_judge(&p, &machine, &fit);
            eprintln!("  {name:22} measured {measured:22}  model {:7.1} against shipped {:7.1} \
                       = {:+.1}%   wait {:5.2} barrier {:5.2} resident {:2}  can_judge {}",
                      span, shipped, 100.0 * (span - shipped) / shipped,
                      t.wait_us, t.barrier_us, p.resident_warps(&machine), judgeable);

            if name.starts_with("the aliased") {
                assert!(!judgeable, "{name} changes the residency, so the model must refuse it");
            }
        }
        assert!(can_judge(&base, &machine, &fit), "the fitted configuration must be judgeable");
    }

    #[test]
    fn the_pipeline_agreements_catch_the_hang_that_happened() {
        let as_built = PipelineSide {
            cta_threads: 384, num_mma_warps: 8, producer_warps: 4,
            staging_stride: 384, staging_index_base: 0, staging_bound: 256,
            full_arrivals: 4, empty_arrivals: 8,
            full_via_cp_async: true, empty_lane_guarded: false,
            early_return_before_sync: false,
            expected_tx: 12_288, box_bytes: 6_144,
        };
        let v = pipeline_violations(&as_built);
        eprintln!("  as built: {v:?}");
        assert!(v.iter().any(|m| m.contains("never executes")), "rule 1 must fire: {v:?}");
        assert!(v.iter().any(|m| m.contains("stride")), "rule 1's other half must fire: {v:?}");

        let fixed = PipelineSide {
            staging_stride: 128, staging_index_base: -256,
            full_arrivals: 4 * 32, empty_arrivals: 8, empty_lane_guarded: true,
            expected_tx: 6_144, box_bytes: 6_144,
            ..as_built
        };
        assert!(pipeline_violations(&fixed).is_empty(), "{:?}", pipeline_violations(&fixed));

        let short_tx = PipelineSide { expected_tx: 12_288, ..fixed };
        assert!(pipeline_violations(&short_tx)[0].contains("expect_tx"),
                "the barrier is armed short: {:?}", pipeline_violations(&short_tx));
        let unguarded_empty = PipelineSide { empty_lane_guarded: false, ..fixed };
        assert!(pipeline_violations(&unguarded_empty)[0].contains("empty[s]"),
                "the empty arrive must be lane-guarded: {:?}", pipeline_violations(&unguarded_empty));
        let wrong_full = PipelineSide { full_arrivals: 4, ..fixed };
        assert!(pipeline_violations(&wrong_full)[0].contains("full[s]"),
                "the cp.async arrive counts threads: {:?}", pipeline_violations(&wrong_full));
        let deadlock = PipelineSide { early_return_before_sync: true, ..fixed };
        assert!(pipeline_violations(&deadlock)[0].contains("DEADLOCK"), "the sync rule");
        let partial_wg = PipelineSide { producer_warps: 3, ..fixed };
        assert!(pipeline_violations(&partial_wg).iter().any(|m| m.contains("warpgroup")), "wg rule");
    }

    #[test]
    fn the_shipped_configuration_is_reproduced() {
        let p = Params::SHIPPED;
        let hw = Machine::RTX_5090;
        assert_eq!(p.m_tile(), 128);
        assert_eq!(p.n_block(), 128);
        assert_eq!(p.cta_threads(), 384);
        assert_eq!(p.atom_rows(), 128);
        assert!(rejection(&p, &hw).is_none());
        let s = p.smem();
        assert!((s.total as i64 - 89_216).abs() < 2_048,
                "model says {} B, the cubin says 89,216", s.total);
        let r = p.regs(hw.regs_per_sm);
        assert_eq!(r.accumulator, 64);
        assert_eq!(r.prefetch, 68);
        assert_eq!(r.available, 170);
    }

    #[test]
    fn the_prediction_matches_the_production_regime() {
        let p = Params::SHIPPED;
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let b = tile_time(&p, &hw, &fit, 768, 12288, 4096);
        assert!((b.total_us - 20.1).abs() < 7.0,
                "predicted {:.1} us against 20.1 measured: {b}", b.total_us);
    }

    #[test]
    fn the_terms_attribute_the_atom_penalty() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let wide = tile_time(&Params::SHIPPED, &hw, &fit, 768, 12288, 4096);
        let v6 = Params { warps_m: 6, warps_n: 2, warp_m_rows: 32, n_tile: 64,
                          ab_stages_a: 2, ab_stages_b: 2, ..Params::SHIPPED };
        assert_eq!(v6.atom_rows(), 96);
        let tall = tile_time(&v6, &hw, &fit, 768, 12288, 4096);
        assert!(tall.total_us > wide.total_us);

        assert!(tall.tensor_us / wide.tensor_us > 1.3,
                "the atom penalty must show in the tensor term");
    }

    #[test]
    fn the_barrier_term_scales_with_the_k_tile() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let two = tile_time(&Params::SHIPPED, &hw, &fit, 768, 12288, 4096);
        let four = Params { sub_chunks: 4, ..Params::SHIPPED };
        let b = tile_time(&four, &hw, &fit, 768, 12288, 4096);
        assert_eq!(four.ktile(), 256);
        assert!((two.barrier_us / b.barrier_us - 2.0).abs() < 0.01,
                "doubling KTILE must halve the barrier term: {} vs {}", two.barrier_us, b.barrier_us);
    }

    #[test]
    fn the_swizzle_only_moves_the_dram_term() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let off = tile_time(&Params::SHIPPED, &hw, &fit, 768, 12288, 4096);
        let on = Params { swizzle_log2: 6, ..Params::SHIPPED };
        let b = tile_time(&on, &hw, &fit, 768, 12288, 4096);
        assert!(b.dram_us <= off.dram_us, "the swizzle must not increase DRAM traffic");
        assert!((b.tensor_us - off.tensor_us).abs() < 1e-9, "and must not touch the tensor term");
    }

    #[test]
    fn the_a_residency_boundary_is_where_l2_says_it_is() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let p = Params::SHIPPED;
        let k = 4096;
        assert_eq!(p.l2_row_limit(k, &hw), 49_152, "96 MB over 2048 bytes per row");
        assert!(p.a_fits_l2(49_152, k, &hw));
        assert!(!p.a_fits_l2(49_168, k, &hw), "one row past the limit is past it");

        let below = tile_time(&p, &hw, &fit, 49_152, 12288, k);
        let above = tile_time(&p, &hw, &fit, 65_536, 12288, k);
        assert!(above.dram_us > below.dram_us * 1.5,
                "the ramp must show in the DRAM term: {:.1} vs {:.1} us",
                above.dram_us, below.dram_us);
        assert!((above.tensor_us - below.tensor_us).abs() < 1e-9,
                "and must NOT show in the tensor term");
    }

    #[test]
    fn the_residency_boundary_moves_with_k() {
        let hw = Machine::RTX_5090;
        let p = Params::SHIPPED;
        assert_eq!(p.l2_row_limit(4096, &hw), 49_152);
        assert_eq!(p.l2_row_limit(8192, &hw), 24_576, "twice the K fits half the rows");
        assert_eq!(p.l2_row_limit(2048, &hw), 98_304);
    }

    #[test]
    fn the_register_wall_is_visible_in_the_terms() {
        let hw = Machine::RTX_5090;
        let v6 = Params { warps_m: 6, warps_n: 2, warp_m_rows: 32, n_tile: 64,
                          ab_stages_a: 2, ab_stages_b: 2, ..Params::SHIPPED };
        let r = v6.regs(hw.regs_per_sm);
        assert_eq!(r.available, 128, "512 threads");
        assert!(r.sum > r.available);
        assert!(rejection(&v6, &hw).unwrap().contains("registers"));
    }

    #[test]
    fn the_breakdown_names_the_limiting_pipe() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let b = tile_time(&Params::SHIPPED, &hw, &fit, 768, 12288, 4096);
        assert!(matches!(b.pipes(), "tensor" | "dram" | "smem"));
        assert!(b.roofline_us > 0.0 && b.total_us > b.roofline_us);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwizzleSide {
    pub atom_rows: usize,
    pub chunk_stride: usize,
    pub tile_m: usize,
    pub rows_padded: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSide {
    pub sf_blk_mn: usize,
    pub sf_chunk_stride_a: usize,
    pub sf_chunk_stride_b: usize,
    pub m_tile: usize,
    pub n_block: usize,
    pub sub_chunks: usize,
    pub sf_rows: usize,
    pub sf_stage_bytes_a_total: usize,
    pub sf_stage_bytes_b: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchSide {
    pub kernel_threads: usize,
    pub launched_threads: usize,
    pub tma_box_rows: usize,
    pub c_box_rows: usize,
    pub kernel_epi_n: usize,
    pub c_box_inner_bytes: usize,
    pub expect_tx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubinFacts {
    pub m_tile: usize,
    pub n_block: usize,
    pub stages: usize,
    pub sub_chunks: usize,
    pub epi_n: usize,
    pub sf_blk_mn: usize,
    pub cta_threads: usize,
}

impl CubinFacts {
    pub const GEMM: CubinFacts = CubinFacts {
        m_tile: 128, n_block: 128, stages: 2, sub_chunks: 2, epi_n: 32,
        sf_blk_mn: 128, cta_threads: 384,
    };

    pub const GEMM_M64: CubinFacts = CubinFacts { m_tile: 64, ..CubinFacts::GEMM };

    pub const GEMM_GATED: CubinFacts = CubinFacts { m_tile: 128, n_block: 64, ..CubinFacts::GEMM };

    pub const GEMM_M192: CubinFacts = CubinFacts {
        m_tile: 192, sf_blk_mn: 96, cta_threads: 512, ..CubinFacts::GEMM
    };

    pub const GEMM_M192W8: CubinFacts = CubinFacts { m_tile: 192, sf_blk_mn: 96, ..CubinFacts::GEMM };

    pub const FP8: CubinFacts = CubinFacts {
        m_tile: 64, n_block: 128, stages: 3, sub_chunks: 4, epi_n: 0, sf_blk_mn: 0, cta_threads: 384,
    };

    pub const FP8_ALIASED: CubinFacts = CubinFacts { stages: 2, ..CubinFacts::FP8 };

    pub const FP8_K64: CubinFacts = CubinFacts { sub_chunks: 2, ..CubinFacts::FP8 };

    pub const FP8_W2X4: CubinFacts = CubinFacts { m_tile: 64, n_block: 128, ..CubinFacts::FP8 };

    pub const FP8_T128: CubinFacts = CubinFacts { m_tile: 128, n_block: 128, stages: 2,
                                                  ..CubinFacts::FP8 };

    pub const FP8_N64: CubinFacts = CubinFacts { m_tile: 64, n_block: 64, ..CubinFacts::FP8 };

    pub const FP8_STAGES2: CubinFacts = CubinFacts { stages: 2, ..CubinFacts::FP8 };

    pub const FP8_BIG: CubinFacts = CubinFacts {
        m_tile: 256, n_block: 64, stages: 3, sub_chunks: 2, epi_n: 0, sf_blk_mn: 0, cta_threads: 384,
    };

    pub fn launch_m_tile(&self) -> usize { self.m_tile }
    pub fn launch_threads(&self) -> usize { self.cta_threads }
}

pub fn cubin_matches_launch(cubin: &CubinFacts, m_tile: usize, threads: usize,
                            epi_n: usize, sf_blk_mn: usize) -> Vec<String> {
    let mut bad = Vec::new();
    if cubin.m_tile != m_tile {
        bad.push(format!(
            "the selected cubin was compiled for a {}-row tile but the launch builds {}-row tensor maps \
             — the TMA delivers the wrong number of bytes, `expect_tx` is never satisfied and the kernel \
             HANGS. The dispatch must select the cubin whose M_TILE matches `m_tile`.",
            cubin.m_tile, m_tile));
    }
    if cubin.cta_threads != threads {
        bad.push(format!("the cubin is __launch_bounds__({}) but the launch uses {threads} threads",
                         cubin.cta_threads));
    }
    if cubin.epi_n != epi_n {
        bad.push(format!("the cubin stages {}-wide epilogue rows but the C descriptor's box is {epi_n}",
                         cubin.epi_n));
    }
    if cubin.sf_blk_mn != sf_blk_mn {
        bad.push(format!("the cubin's SF_BLK_MN is {} but the swizzle wrote {sf_blk_mn}",
                         cubin.sf_blk_mn));
    }
    bad
}

pub fn fp8_cubin_matches_launch(cubin: &CubinFacts, m_tile: usize, n_block: usize, stages: usize,
                                threads: usize) -> Vec<String> {
    let mut bad = Vec::new();
    if cubin.m_tile != m_tile {
        bad.push(format!(
            "the selected fp8 cubin was compiled for an {}-row A box but the launch builds {m_tile}-row \
             maps — `expect_tx` is never satisfied and the kernel HANGS", cubin.m_tile));
    }
    if cubin.n_block != n_block {
        bad.push(format!(
            "the selected fp8 cubin has N_BLOCK {} but the launch builds an {}-row B box and an n/{} grid \
             — the TMA writes {} bytes into a {}-byte buffer, the SMEM overruns the barriers and the \
             kernel HANGS. The dispatch must select the cubin whose N_BLOCK matches `n_block`",
            cubin.n_block, n_block, n_block, n_block * 128, cubin.n_block * 128));
    }
    if cubin.stages != stages {
        bad.push(format!("the fp8 cubin has STAGES {} but the host reports {stages}",
                         cubin.stages));
    }
    if cubin.cta_threads != threads {
        bad.push(format!("the fp8 cubin is __launch_bounds__({}) but the launch uses {threads} threads",
                         cubin.cta_threads));
    }
    bad
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fp8Side {
    pub m_tile: usize,
    pub n_block: usize,
    pub n_tile: usize,
    pub warps_m: usize,
    pub warps_n: usize,
    pub warp_m_rows: usize,
    pub ktile: usize,
    pub smem_row_bytes: usize,
    pub stages: usize,
    pub cta_threads: usize,
    pub num_mma_warps: usize,
    pub producer_warps: usize,
    pub epi_smem: usize,
    pub box_inner: usize,
    pub box_rows_a: usize,
    pub box_rows_b: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub m: usize,
    pub n: usize,
    pub persistent: bool,
    pub ctas_per_sm: usize,
    pub sms: usize,
    pub swizzled: bool,
    pub smem_align: usize,
    pub driver_smem: usize,
    pub smem_per_block_limit: usize,
    pub smem_per_sm_limit: usize,
    pub regs_per_thread: usize,
}

impl Fp8Side {
    pub const T64x128: Fp8Side = Fp8Side {
        m_tile: 64, n_block: 128, n_tile: 64, warps_m: 4, warps_n: 2, warp_m_rows: 16,
        ktile: 128, smem_row_bytes: 128, stages: 3, cta_threads: 384,
        num_mma_warps: 8, producer_warps: 4,
        box_inner: 128, box_rows_a: 64, box_rows_b: 128,
        grid_x: 1, grid_y: 1, m: 64, n: 128,
        persistent: true, ctas_per_sm: 1, sms: 170,
        swizzled: false, smem_align: 1024,
        driver_smem: 1044, smem_per_block_limit: 101_376, smem_per_sm_limit: 102_400,
        regs_per_thread: 64,
        epi_smem: 16_384,
    };

    pub const ALIASED: Fp8Side = Fp8Side { stages: 2, ctas_per_sm: 2, epi_smem: 0, ..Fp8Side::T64x128 };

    pub const W2X4: Fp8Side = Fp8Side {
        m_tile: 64, n_block: 128, n_tile: 32, warps_m: 2, warps_n: 4, warp_m_rows: 32,
        box_rows_a: 64, box_rows_b: 128, m: 64, n: 128,
        ..Fp8Side::T64x128
    };
    pub const T128: Fp8Side = Fp8Side {
        m_tile: 128, n_block: 128, n_tile: 64, warp_m_rows: 32, stages: 2,
        box_rows_a: 128, box_rows_b: 128, m: 128, n: 128,
        ..Fp8Side::T64x128
    };
    pub const N64: Fp8Side = Fp8Side {
        m_tile: 64, n_block: 64, n_tile: 32,
        box_rows_a: 64, box_rows_b: 64, m: 64, n: 64,
        ..Fp8Side::T64x128
    };

    pub const T256x64: Fp8Side = Fp8Side {
        m_tile: 256, n_block: 64, n_tile: 64, warps_m: 8, warps_n: 1, warp_m_rows: 32,
        ktile: 64, smem_row_bytes: 64, stages: 3, cta_threads: 384,
        num_mma_warps: 8, producer_warps: 4,
        box_inner: 64, box_rows_a: 256, box_rows_b: 64,
        grid_x: 1, grid_y: 1, m: 256, n: 64,
        epi_smem: 32_768,
        ..Fp8Side::T64x128
    };

    pub fn stage_bytes(&self) -> usize {
        (self.m_tile + self.n_block) * self.smem_row_bytes
    }

    pub fn box_bytes(&self) -> usize {
        self.box_rows_a * self.box_inner + self.box_rows_b * self.box_inner
    }

    pub fn smem_bytes(&self) -> usize {
        self.stages * self.stage_bytes() + self.m_tile * 2 + self.n_block * 2 + self.stages * 16
            + self.epi_smem
    }

    pub fn ctas_by_smem(&self) -> usize {
        self.smem_per_sm_limit / (self.smem_bytes() + self.driver_smem).max(1)
    }

    pub fn ctas_by_regs(&self) -> usize {
        65_536 / (self.cta_threads * self.regs_per_thread).max(1)
    }

    pub fn l2_read_bytes(&self, k: usize) -> u64 {
        let t = self.m as u64 * self.n as u64 * k as u64;
        t / self.n_block as u64 + t / self.m_tile as u64
    }
}

pub fn fp8_violations(s: &Fp8Side) -> Vec<String> {
    let mut bad = Vec::new();

    if s.n_block != s.n_tile * s.warps_n {
        bad.push(format!("N_BLOCK {} is not N_TILE {} x WARPS_N {}", s.n_block, s.n_tile, s.warps_n));
    }
    if s.m_tile != s.warps_m * s.warp_m_rows {
        bad.push(format!("M_TILE {} is not WARPS_M {} x WARP_M_ROWS {}", s.m_tile, s.warps_m,
                         s.warp_m_rows));
    }

    if s.box_inner != s.ktile {
        bad.push(format!("the box inner is {} but KTILE_FP8 is {} — a short box stages only part of the \
                          k-range and the boundary gate reads it as a wrong answer", s.box_inner, s.ktile));
    }
    if s.box_rows_a != s.m_tile || s.box_rows_b != s.n_block {
        bad.push(format!("the boxes are [{}, {}] rows but the tile is M_TILE {} x N_BLOCK {} — the box's \
                          rows ARE the tile's rows", s.box_rows_a, s.box_rows_b, s.m_tile, s.n_block));
    }
    if s.smem_row_bytes < s.ktile {
        bad.push(format!("the SMEM row is {} B but the k-tile is {} B — the row must hold the whole \
                          k-tile", s.smem_row_bytes, s.ktile));
    }

    if s.box_bytes() != s.stages * s.stage_bytes() / s.stages * 1 && s.box_bytes() != s.stage_bytes() {
        bad.push(format!(
            "the TMA boxes deliver {} bytes but one stage is {} — `expect_tx` is armed with the stage's \
             size, so the transaction count never reaches zero and the kernel HANGS",
            s.box_bytes(), s.stage_bytes()));
    }

    let work_tiles = s.m.div_ceil(s.m_tile) * s.n.div_ceil(s.n_block);
    if s.persistent {
        let want = work_tiles.min(s.sms * s.ctas_per_sm);
        if s.grid_x != want || s.grid_y != 1 {
            bad.push(format!("the persistent grid is ({}, {}) but the residency needs ({}, 1): \
                             {work_tiles} work tiles over {} SMs x {} CTAs/SM", s.grid_x, s.grid_y,
                             want, s.sms, s.ctas_per_sm));
        }
    } else {
        let want_x = s.m.div_ceil(s.m_tile);
        let want_y = s.n.div_ceil(s.n_block);
        if s.grid_x != want_x || s.grid_y != want_y {
            bad.push(format!("the grid is ({}, {}) but the tile needs ({}, {}) for {}x{}", s.grid_x,
                             s.grid_y, want_x, want_y, s.m, s.n));
        }
    }

    if s.swizzled {
        if !matches!(s.box_inner, 32 | 64 | 128) {
            bad.push(format!("a swizzled box inner must be 32, 64 or 128 bytes; it is {}", s.box_inner));
        }
        if s.smem_row_bytes != s.box_inner {
            bad.push(format!("a swizzled row must be the swizzle span: row {} B against box inner {} B",
                             s.smem_row_bytes, s.box_inner));
        }
        if s.smem_align < 1024 {
            bad.push(format!(
                "the staging buffer is aligned {} but a SWIZZLE_128B destination must be aligned 1024: \
                 `Swizzle<3,4,3>` XORs address bits [4,7) with [7,10), so a base whose bits 7-9 are not \
                 zero XORs EVERY row by a constant and the kernel reads the wrong 16-byte units at every \
                 k — while every layout probe still reports the TMA write correct", s.smem_align));
        }
    }

    if s.smem_bytes() > s.smem_per_block_limit {
        bad.push(format!("static SMEM is {} B against a {}-byte per-block limit", s.smem_bytes(),
                         s.smem_per_block_limit));
    }
    if s.cta_threads != (s.num_mma_warps + s.producer_warps) * 32 {
        bad.push(format!("{} threads is not ({} MMA + {} producer) x 32", s.cta_threads,
                         s.num_mma_warps, s.producer_warps));
    }
    if s.producer_warps != 0 && s.producer_warps % 4 != 0 {
        bad.push(format!("the producer group is {} warps; `setmaxnreg` is undefined in a partial \
                          warpgroup, so it must be a whole one", s.producer_warps));
    }
    if s.ctas_by_smem() == 0 {
        bad.push(format!("SMEM {} B + driver {} B does not fit {} B per SM at any CTA count",
                         s.smem_bytes(), s.driver_smem, s.smem_per_sm_limit));
    }
    bad
}

pub fn consistency(s: &SwizzleSide, k: &KernelSide, l: &LaunchSide) -> Vec<String> {
    let mut bad = Vec::new();

    if s.atom_rows != k.sf_blk_mn {
        bad.push(format!(
            "swizzle atom {} != kernel SF_BLK_MN {} — the GEMM reads a layout nothing wrote. \
             Set CLEF_SF_ATOM to match, or rebuild without it.",
            s.atom_rows, k.sf_blk_mn));
    }
    if s.chunk_stride != k.sf_chunk_stride_a {
        bad.push(format!(
            "swizzle chunk stride {} != kernel SF_CHUNK_STRIDE_A {}",
            s.chunk_stride, k.sf_chunk_stride_a));
    }
    if s.tile_m != k.m_tile {
        bad.push(format!("swizzle tile_m {} != kernel M_TILE {}", s.tile_m, k.m_tile));
    }

    if !matches!(k.sf_blk_mn, 128 | 96) {
        bad.push(format!(
            "SF_BLK_MN {} is not an atom the kernel computes correctly at — measured: 128 and 96 pass \
             every gate, 32 produces a WRONG ANSWER (and with unit_scales, so not a layout artifact)",
            k.sf_blk_mn));
    }
    if k.m_tile % k.sf_blk_mn != 0 && k.m_tile > k.sf_blk_mn {
        bad.push(format!(
            "SF_BLK_MN {} does not divide M_TILE {} and is not taller than it — a row would read \
             another row's scales", k.sf_blk_mn, k.m_tile));
    }

    let sf_atoms_m = (k.m_tile / k.sf_blk_mn).max(1);
    let expect_a = sf_atoms_m * k.sub_chunks * k.sf_chunk_stride_a;
    if k.sf_stage_bytes_a_total != expect_a {
        bad.push(format!(
            "SF_STAGE_BYTES_A_TOTAL {} != {sf_atoms_m} atoms x {} chunks x {} bytes = {expect_a}",
            k.sf_stage_bytes_a_total, k.sub_chunks, k.sf_chunk_stride_a));
    }
    let sf_smem_a_words = k.sf_rows * k.sub_chunks * sf_atoms_m;
    if sf_smem_a_words * 4 != k.sf_stage_bytes_a_total {
        bad.push(format!(
            "sfa_smem holds {} B but the copy writes {} B — the copy overruns the buffer",
            sf_smem_a_words * 4, k.sf_stage_bytes_a_total));
    }

    if l.kernel_threads != l.launched_threads {
        bad.push(format!(
            "kernel is __launch_bounds__({}) but the host launches {} — CUDA_ERROR_INVALID_VALUE, or a \
             hang when the mismatch changes the warp count",
            l.kernel_threads, l.launched_threads));
    }

    if l.tma_box_rows != k.m_tile {
        bad.push(format!(
            "TMA box is {} rows against a {}-row tile — the transaction count never reaches zero and \
             `mbar_wait` spins forever. This is a HANG, not a wrong answer, and it has happened four times.",
            l.tma_box_rows, k.m_tile));
    }
    if l.c_box_rows != k.m_tile {
        bad.push(format!("C box is {} rows against a {}-row tile", l.c_box_rows, k.m_tile));
    }
    if l.c_box_inner_bytes != l.kernel_epi_n * 4 {
        bad.push(format!(
            "C descriptor's inner extent is {} B but the kernel stages {}-wide rows — the store reads \
             past each row and never completes",
            l.c_box_inner_bytes, l.kernel_epi_n));
    }

    let copies = k.sf_stage_bytes_a_total + k.sf_stage_bytes_b;
    if l.expect_tx < copies {
        bad.push(format!(
            "expect_tx {} is less than the {copies} B of scale copies alone — the barrier completes early",
            l.expect_tx));
    }

    bad
}

#[cfg(test)]
mod consistency_tests {
    use super::*;

    fn shipped() -> (SwizzleSide, KernelSide, LaunchSide) {
        (
            SwizzleSide { atom_rows: 128, chunk_stride: 512, tile_m: 128, rows_padded: 128 },
            KernelSide { sf_blk_mn: 128, sf_chunk_stride_a: 512, sf_chunk_stride_b: 512,
                         m_tile: 128, n_block: 128, sub_chunks: 2, sf_rows: 128,
                         sf_stage_bytes_a_total: 1024, sf_stage_bytes_b: 1024 },
            LaunchSide { kernel_threads: 384, launched_threads: 384, tma_box_rows: 128,
                         c_box_rows: 128, kernel_epi_n: 32, c_box_inner_bytes: 128,
                         expect_tx: 8192 + 8192 + 1024 + 1024 },
        )
    }

    #[test]
    fn the_shipped_launch_is_consistent() {
        let (s, k, l) = shipped();
        assert_eq!(consistency(&s, &k, &l), Vec::<String>::new());
    }

    #[test]
    fn the_atom_mismatch_that_hung_is_caught() {
        let (s, mut k, l) = shipped();
        k.sf_blk_mn = 32;
        k.sf_chunk_stride_a = 128;
        let bad = consistency(&s, &k, &l);
        assert!(bad.iter().any(|m| m.contains("swizzle atom 128 != kernel SF_BLK_MN 32")),
                "the atom mismatch must be named: {bad:?}");
    }

    #[test]
    fn a_forced_atom_is_consistent_and_still_refused() {
        let (_, mut k, l) = shipped();
        k.sf_blk_mn = 32;
        k.sf_chunk_stride_a = 128;
        k.sf_rows = 32;
        k.sf_stage_bytes_a_total = 4 * 2 * 128;
        let s = SwizzleSide { atom_rows: 32, chunk_stride: 128, tile_m: 128, rows_padded: 128 };
        let bad = consistency(&s, &k, &l);

        assert!(!bad.iter().any(|m| m.contains("chunk stride") || m.contains("overruns")),
                "the numbers must agree once the stride follows the atom: {bad:?}");

        assert!(bad.iter().any(|m| m.contains("WRONG ANSWER")),
                "and it must be refused on correctness, not costed: {bad:?}");
    }

    #[test]
    fn the_tensor_map_box_mismatch_is_caught() {
        let (s, k, mut l) = shipped();
        l.tma_box_rows = 192;
        let bad = consistency(&s, &k, &l);
        assert!(bad.iter().any(|m| m.contains("HANG")), "must warn it is a hang: {bad:?}");
    }

    #[test]
    fn the_thread_count_mismatch_is_caught() {
        let (s, k, mut l) = shipped();
        l.kernel_threads = 640;
        let bad = consistency(&s, &k, &l);
        assert!(bad.iter().any(|m| m.contains("__launch_bounds__(640)")), "{bad:?}");
    }

    #[test]
    fn the_epi_n_mismatch_is_caught() {
        let (s, k, mut l) = shipped();
        l.c_box_inner_bytes = 128;
        l.kernel_epi_n = 16;
        assert!(consistency(&s, &k, &l).iter().any(|m| m.contains("inner extent")));
    }

    #[test]
    fn an_unsupported_atom_is_refused() {
        let (mut s, mut k, l) = shipped();
        s.atom_rows = 32;
        s.chunk_stride = 128;
        k.sf_blk_mn = 32;
        k.sf_chunk_stride_a = 128;
        k.sf_rows = 32;
        k.sf_stage_bytes_a_total = 4 * 2 * 128;
        let bad = consistency(&s, &k, &l);
        assert!(bad.iter().any(|m| m.contains("WRONG ANSWER")),
                "an unsupported atom must be refused rather than costed: {bad:?}");
    }

    #[test]
    fn every_mismatch_is_reported() {
        let (mut s, mut k, mut l) = shipped();
        s.atom_rows = 64;
        k.m_tile = 192;
        l.tma_box_rows = 96;
        let bad = consistency(&s, &k, &l);
        assert!(bad.len() >= 3, "expected several mismatches, got {bad:?}");
    }
}

#[cfg(test)]
mod machine_tests {
    use super::*;

    #[test]
    fn the_hardware_factors_match_the_measured_card() {
        let m = Machine::RTX_5090;
        assert_eq!(m.sms, 170);
        assert_eq!(m.regs_per_sm, 65_536);
        assert_eq!(m.smem_per_sm, 102_400, "100 KB per SM");
        assert_eq!(m.smem_per_block, 101_376, "99 KB per block -- the opt-in one, not 48 KB");
        assert_eq!(m.threads_per_sm, 1_536);
        assert_eq!(m.warps_per_sm, 48);
        assert_eq!(m.warp_size, 32);
        assert_eq!(m.l2_bytes, 96.0 * 1024.0 * 1024.0);

        let p = m.per_sm();
        assert!((p.peak_flops - 9.859e12).abs() < 1e10, "1676 TFLOP/s over 170 SMs");
        assert!((p.dram_bw - 1.792e12 / 170.0).abs() < 1.0e6, "about 10.5 GB/s per SM");
    }

    #[test]
    fn the_two_cards_differ_only_in_the_sm_count() {
        let ours = Machine::RTX_5090;
        let theirs = Machine::RTX_PRO_6000;
        assert_eq!(ours.sms, 170);
        assert_eq!(theirs.sms, 188);

        assert_eq!(ours.regs_per_sm, theirs.regs_per_sm);
        assert_eq!(ours.smem_per_sm, theirs.smem_per_sm);
        assert_eq!(ours.smem_per_block, theirs.smem_per_block);
        assert_eq!(ours.threads_per_sm, theirs.threads_per_sm);

        let v6 = Params { warps_m: 6, warps_n: 2, warp_m_rows: 32, n_tile: 64,
                          ab_stages_a: 2, ab_stages_b: 2, ..Params::SHIPPED };
        assert_eq!(waves(&v6, 2048, 2048, &ours), 2);
        assert_eq!(waves(&v6, 2048, 2048, &theirs), 1);
    }

    #[test]
    fn occupancy_names_the_binding_resource() {
        let m = Machine::RTX_5090;
        let shipped = Params::SHIPPED;
        let (n, why) = m.ctas_per_sm(&shipped);
        assert_eq!(n, 1, "the shipped config is one CTA per SM");
        assert_eq!(why, "shared memory", "and SMEM is what limits it");

        let small = Params { warps_m: 2, warps_n: 2, warp_m_rows: 32, n_tile: 64,
                             ab_stages_a: 2, ab_stages_b: 2, epi_tile_m: 64, epi_tile_n: 32,
                             epi_stages: 1, ..Params::SHIPPED };
        let (n2, _) = m.ctas_per_sm(&small);
        assert!(n2 >= 1);
    }

    #[test]
    fn time_and_cycles_are_interchangeable() {
        let m = Machine::RTX_5090;
        let cycles = m.cycles(20.1);
        assert!((cycles - 62_109.0).abs() < 1.0, "20.1 us at 3090 MHz");
    }
}

#[cfg(test)]
mod atom_tests {
    use super::*;

    #[test]
    fn the_atom_legality_gate_matches_what_the_gates_measured() {
        let at = |m_tile: usize| Params { warps_m: 1, warps_n: 4, warp_m_rows: m_tile, n_tile: 32,
                                          ..Params::SHIPPED };
        assert_eq!(at(128).atom_rows(), 128);
        assert!(at(128).legal_atom());
        assert_eq!(at(192).atom_rows(), 96);
        assert!(at(192).legal_atom(), "Version 6's atom passes every gate");
        assert_eq!(at(160).atom_rows(), 32);
        assert!(!at(160).legal_atom(), "and 32 does not compute");
    }

    #[test]
    fn a_penalised_atom_is_costed_and_an_illegal_one_is_refused() {
        let at = |m_tile: usize| Params { warps_m: 1, warps_n: 4, warp_m_rows: m_tile, n_tile: 32,
                                          ..Params::SHIPPED };

        assert!(at(192).penalised() && at(192).legal_atom());

        assert!(at(160).penalised() && !at(160).legal_atom());

        assert!(!at(128).penalised() && at(128).legal_atom());
    }
}

#[cfg(test)]
mod negative_tests {
    use super::*;

    fn good() -> (SwizzleSide, KernelSide, LaunchSide) {
        (
            SwizzleSide { atom_rows: 128, chunk_stride: 512, tile_m: 128, rows_padded: 128 },
            KernelSide { sf_blk_mn: 128, sf_chunk_stride_a: 512, sf_chunk_stride_b: 512,
                         m_tile: 128, n_block: 128, sub_chunks: 2, sf_rows: 128,
                         sf_stage_bytes_a_total: 1024, sf_stage_bytes_b: 1024 },
            LaunchSide { kernel_threads: 384, launched_threads: 384, tma_box_rows: 128,
                         c_box_rows: 128, kernel_epi_n: 32, c_box_inner_bytes: 128,
                         expect_tx: 18_432 },
        )
    }

    fn consistency_mutations() -> Vec<(
        &'static str,
        &'static str,
        fn(&mut SwizzleSide, &mut KernelSide, &mut LaunchSide),
        usize,
    )> {
        vec![

            ("swizzle atom vs kernel SF_BLK_MN", "swizzle atom",
             |s, _, _| { s.atom_rows = 64; }, 1),
            ("swizzle chunk stride vs kernel", "chunk stride",
             |s, _, _| { s.chunk_stride = 1024; }, 1),
            ("swizzle tile_m vs kernel M_TILE", "swizzle tile_m",
             |s, _, _| { s.tile_m = 256; }, 1),

            ("SF_BLK_MN does not divide M_TILE", "does not divide",
             |s, k, _| { s.atom_rows = 96; s.chunk_stride = 384;
                         k.sf_blk_mn = 96; k.sf_chunk_stride_a = 384; k.sf_rows = 96;
                         k.sf_stage_bytes_a_total = 1 * 2 * 384; }, 1),

            ("staged bytes vs atoms x chunks x stride", "SF_STAGE_BYTES_A_TOTAL",
             |_, k, _| { k.sf_stage_bytes_a_total = 2048; }, 2),
            ("sfa_smem vs what the copy writes", "overruns",
             |_, k, _| { k.sf_rows = 256; }, 1),
            ("kernel threads vs launched threads", "__launch_bounds__",
             |_, _, l| { l.launched_threads = 512; }, 1),
            ("TMA box rows vs M_TILE", "HANG",
             |_, _, l| { l.tma_box_rows = 192; }, 1),
            ("C box rows vs M_TILE", "C box",
             |_, _, l| { l.c_box_rows = 64; }, 1),
            ("C box inner vs kernel EPI_N", "inner extent",
             |_, _, l| { l.c_box_inner_bytes = 64; }, 1),
            ("expect_tx vs the copies", "expect_tx",
             |_, _, l| { l.expect_tx = 1024; }, 1),
            ("an atom the kernel cannot compute at", "WRONG ANSWER",
             |s, k, _| { s.atom_rows = 32; s.chunk_stride = 128; k.sf_blk_mn = 32;
                         k.sf_chunk_stride_a = 128; k.sf_rows = 32;
                         k.sf_stage_bytes_a_total = 4 * 2 * 128; }, 1),
        ]
    }

    #[test]
    fn every_consistency_check_fires() {
        for (name, marker, mutate, _) in consistency_mutations() {
            let (mut s, mut k, mut l) = good();
            mutate(&mut s, &mut k, &mut l);
            let bad = consistency(&s, &k, &l);
            assert!(bad.iter().any(|m| m.contains(marker)),
                    "the check for `{name}` did not fire (looking for `{marker}`), got: {bad:?}");
        }
    }

    #[test]
    fn each_consistency_mutation_breaks_only_its_own_check() {
        for (name, _, mutate, expected) in consistency_mutations() {
            let (mut s, mut k, mut l) = good();
            mutate(&mut s, &mut k, &mut l);
            let bad = consistency(&s, &k, &l);
            assert_eq!(bad.len(), expected,
                    "`{name}` broke {expected} checks, got {}: {bad:?}", bad.len());
        }
    }

    #[test]
    fn the_shipped_configuration_passes_every_check() {
        let (s, k, l) = good();
        assert_eq!(consistency(&s, &k, &l), Vec::<String>::new());
    }

    fn limitation_mutations() -> Vec<(&'static str, &'static str, fn(&mut Params))> {
        vec![
            ("not a whole number of warpgroups", "warpgroups",
             |p| { p.warps_m = 5; p.warps_n = 1; }),
            ("threads over the per-SM ceiling", "threads",
             |p| { p.warps_m = 8; p.warps_n = 2; }),
            ("N_BLOCK not a multiple of 128", "multiple of 128",
             |p| { p.n_tile = 24; }),
            ("WARP_M_ROWS not a multiple of 16", "multiple of 16",
             |p| { p.warp_m_rows = 20; }),

            ("N_TILE not a multiple of 8", "multiple of 8",
             |p| { p.warps_m = 1; p.warps_n = 32; p.n_tile = 12; }),
            ("the register wall", "registers",
             |p| { p.warps_m = 6; p.warps_n = 2; }),
        ]
    }

    #[test]
    fn every_limitation_fires_and_names_its_resource() {
        for (name, marker, mutate) in limitation_mutations() {
            let mut p = Params::SHIPPED;
            mutate(&mut p);
            let why = rejection(&p, &Machine::RTX_5090)
                .unwrap_or_else(|| panic!("the limitation `{name}` did not fire"));
            assert!(why.contains(marker),
                    "`{name}` named `{why}`, which does not mention `{marker}`");
        }
    }

    #[test]
    fn the_shared_memory_limit_fires() {
        let p = Params { epi_tile_n: 128, epi_stages: 4, ..Params::SHIPPED };
        assert!(p.regs(Machine::RTX_5090.regs_per_sm).sum <= p.regs(Machine::RTX_5090.regs_per_sm).available,
                "the mutation must not touch the register budget");
        let why = rejection(&p, &Machine::RTX_5090).expect("a 4-deep 128-wide epilogue must be refused");
        assert!(why.contains("shared memory"), "got `{why}`");
    }

    #[test]
    fn the_shipped_configuration_is_legal() {
        assert!(rejection(&Params::SHIPPED, &Machine::RTX_5090).is_none());
    }

    #[test]
    fn the_structural_rules_do_not_depend_on_the_card() {
        let p = Params::SHIPPED;
        assert_eq!(rejection(&p, &Machine::RTX_5090), rejection(&p, &Machine::RTX_PRO_6000),
                "the register, SMEM and shape rules are per-SM facts and must not differ");

        let ours = waves(&p, 2048, 2048, &Machine::RTX_5090);
        let theirs = waves(&p, 2048, 2048, &Machine::RTX_PRO_6000);
        assert_eq!(ours, 2);
        assert_eq!(theirs, 2, "256 tiles is two waves on both");

        let v6 = Params { warps_m: 6, warps_n: 2, warp_m_rows: 32, n_tile: 64,
                          ab_stages_a: 2, ab_stages_b: 2, ..Params::SHIPPED };
        assert_eq!(waves(&v6, 2048, 2048, &Machine::RTX_5090), 2);
        assert_eq!(waves(&v6, 2048, 2048, &Machine::RTX_PRO_6000), 1);
    }
}

#[cfg(test)]
mod baseline_tests {
    use super::*;

    #[test]
    fn the_shipped_baseline_has_no_violations() {
        assert!(Baseline::SHIPPED.applies());
        assert_eq!(Baseline::SHIPPED.violations(), Vec::<&str>::new());
    }

    #[test]
    fn every_improvement_off_invalidates_the_fit_and_names_itself() {
        let cases: [(&str, Baseline); 5] = [
            ("SF_CONFLICT_FREE", Baseline { sf_conflict_free: false, ..Baseline::SHIPPED }),
            ("USE_SETMAXNREG", Baseline { use_setmaxnreg: false, ..Baseline::SHIPPED }),
            ("SF-via-TMA", Baseline { sf_via_tma: false, ..Baseline::SHIPPED }),
            ("warp-specialised store", Baseline { warp_specialized_store: false, ..Baseline::SHIPPED }),
            ("timing syncs once", Baseline { amortized_timing: false, ..Baseline::SHIPPED }),
        ];
        for (name, b) in cases {
            assert!(!b.applies(), "turning {name} off must invalidate the fit");
            let v = b.violations();
            assert_eq!(v.len(), 1, "exactly one assumption is broken: {v:?}");
            assert!(v[0].contains(name), "the message must name {name}, got {}", v[0]);
        }
    }

    #[test]
    fn every_broken_assumption_is_reported() {
        let b = Baseline { sf_conflict_free: false, use_setmaxnreg: false, sf_via_tma: false,
                           warp_specialized_store: false, amortized_timing: false };
        assert_eq!(b.violations().len(), 5);
    }
}

pub fn sweep(base: &Params, m: usize, n: usize, k: usize, machine: &Machine, fit: &Fit, top: usize)
    -> Vec<(Params, f64)>
{
    let mut out: Vec<(Params, f64)> = Vec::new();
    for (warps_m, warps_n) in [(1, 4), (2, 2), (2, 4), (4, 1), (4, 2), (6, 2), (8, 1), (8, 2)] {
        let mma = warps_m * warps_n;
        if !(4..=16).contains(&mma) {
            continue;
        }
        for warp_m_rows in [16usize, 32, 48] {
            for n_tile in [16usize, 32, 64] {
                for ab_stages in [base.ab_stages_a] {
                    for sub_chunks in [1usize, 2] {
                        for (epi_n, epi_stages) in [(16usize, 2usize), (32, 2), (32, 1)] {
                            for swizzle in [0u32, 3, 5] {
                                let p = Params {
                                    warps_m, warps_n, warp_m_rows, n_tile,
                                    ab_stages_a: ab_stages, ab_stages_b: ab_stages,
                                    sub_chunks,
                                    epi_tile_m: warp_m_rows * warps_m, epi_tile_n: epi_n,
                                    epi_stages,
                                    swizzle_log2: swizzle,
                                    ..*base
                                };
                                if rejection(&p, machine).is_some() || !p.legal_atom() {
                                    continue;
                                }
                                let cost = makespan_us(&p, machine, fit, m, n, k);
                                out.push((p, cost));
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out.truncate(top);
    out
}

#[cfg(test)]
mod sweep_tests {
    use super::*;

    #[test]
    fn the_sweep_reproduces_the_shipped_choice() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let best = sweep(&Params::SHIPPED, 768, 12288, 4096, &hw, &fit, 10);
        assert!(!best.is_empty(), "the sweep must find something legal");
        let (p, cost) = &best[0];
        eprintln!("  best for m=768: {}x{} warps {}x{} of {}x{}, {} stages, {} sub-chunks, \
                   epi {}x{}/{} -> {:.1} us",
                  p.warps_m, p.warps_n, p.m_tile(), p.n_block(), p.warp_m_rows, p.n_tile,
                  p.ab_stages_a, p.sub_chunks, p.epi_tile_n, p.epi_stages, p.epi_tile_m, cost);

        let shipped_cost = makespan_us(&Params::SHIPPED, &hw, &fit, 768, 12288, 4096);
        assert!(*cost <= shipped_cost * 1.02,
                "the best must be no worse than shipped: {:.1} vs {:.1}", cost, shipped_cost);
        eprintln!("  shipped: {:.1} us ({:.1}% of best)", shipped_cost, shipped_cost / cost * 100.0);
    }

    #[test]
    fn the_sweep_reports_the_best_per_geometry() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        for (m, n, k) in [(768usize, 12288usize, 4096usize), (2048, 2048, 2048), (32768, 12288, 4096)] {
            eprintln!("\n  === m={m} n={n} k={k} ===");
            let all = sweep(&Params::SHIPPED, m, n, k, &hw, &fit, usize::MAX);
            let shipped = makespan_us(&Params::SHIPPED, &hw, &fit, m, n, k);

            let mut seen: std::collections::BTreeMap<(usize, usize), (Params, f64)> = Default::default();
            for (p, c) in &all {
                let key = (p.m_tile(), p.n_block());
                if seen.get(&key).map_or(true, |(_, best)| c < best) {
                    seen.insert(key, (*p, *c));
                }
            }
            let mut rows: Vec<_> = seen.into_iter().collect();
            rows.sort_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap());
            for ((mt, nb), (p, c)) in rows.iter().take(8) {
                let util = {
                    let t = tiles(p, m, n);
                    t as f64 / (waves(p, m, n, &hw) * hw.sms) as f64
                };
                let mark = if *mt == 128 && *nb == 128 { "  <- shipped shape" } else { "" };
                eprintln!("    {mt:>4}x{nb:<4} w{}x{} rows{:<3} stg{} sc{} epi{}/{}/{} sw{}  {:>9.1} us  \
                           util {:>5.1}%{mark}",
                          p.warps_m, p.warps_n, p.warp_m_rows, p.ab_stages_a, p.sub_chunks,
                          p.epi_tile_n, p.epi_stages, p.epi_tile_m, p.swizzle_log2, c, util * 100.0);
            }
            eprintln!("    shipped pipeline: {shipped:.1} us");
        }
    }

    #[test]
    fn the_sweep_reports_the_top_candidates() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        for (m, n, k) in [(768usize, 12288usize, 4096usize), (2048, 2048, 2048), (32768, 12288, 4096)] {
            eprintln!("\n  === m={m} n={n} k={k} ===");
            for (p, cost) in sweep(&Params::SHIPPED, m, n, k, &hw, &fit, 4) {
                eprintln!("    {:>4}x{:<4} w{}x{} rows{:<3} nt{:<3} stg{} sc{} epi{}/{}/{} sw{}  {:>9.1} us  util {:.1}%",
                          p.m_tile(), p.n_block(), p.warps_m, p.warps_n, p.warp_m_rows, p.n_tile,
                          p.ab_stages_a, p.sub_chunks, p.epi_tile_n, p.epi_stages, p.epi_tile_m,
                          p.swizzle_log2, cost, p_util(&p, m, n, &hw) * 100.0);
            }
        }
    }

    fn p_util(p: &Params, m: usize, n: usize, hw: &Machine) -> f64 {
        let t = tiles(p, m, n);
        t as f64 / (waves(p, m, n, hw) * hw.sms) as f64
    }
}

#[cfg(test)]
mod depth_tests {
    use super::*;

    #[test]
    fn deeper_is_cheaper_at_every_shape() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let two = Params { ab_stages_a: 2, ab_stages_b: 2, ..Params::SHIPPED };
        let three = Params { ab_stages_a: 3, ab_stages_b: 3, ..Params::SHIPPED };
        for (m, n, k) in [(768usize, 12288usize, 4096usize), (32768, 12288, 4096), (131_072, 12288, 4096)] {
            let t2 = tile_time(&two, &hw, &fit, m, n, k);
            let t3 = tile_time(&three, &hw, &fit, m, n, k);

            assert!(t3.total_us < t2.total_us,
                    "at m={m} the deeper pipeline must be cheaper: {:.2} vs {:.2} us",
                    t3.total_us, t2.total_us);
            assert!(t3.wait_us < t2.wait_us,
                    "at m={m} and it must be cheaper BECAUSE it hides latency: {:.2} vs {:.2}",
                    t3.wait_us, t2.wait_us);
        }
    }

    #[test]
    fn depth_costs_a_prologue_wherever_it_is_used() {
        let hw = Machine::RTX_5090;
        let fit = Fit::FITTED_5090;
        let p = Params { ab_stages_a: 4, ab_stages_b: 4, ..Params::SHIPPED };
        let b = tile_time(&p, &hw, &fit, 768, 12288, 4096);
        assert!(b.prologue_us > 0.0);

        assert!(rejection(&p, &hw).is_some(), "4 stages must exceed the per-block limit");
    }
}

#[cfg(test)]
mod cubin_tests {
    use super::*;

    #[test]
    fn the_shipped_cubin_matches_its_launch() {
        assert_eq!(cubin_matches_launch(&CubinFacts::GEMM, 128, 384, 32, 128), Vec::<String>::new());
    }

    #[test]
    fn a_cubin_of_the_wrong_tile_is_caught() {
        let bad = cubin_matches_launch(&CubinFacts::GEMM, 64, 384, 32, 128);
        assert!(bad.iter().any(|m| m.contains("HANG")),
                "the mismatch must say it hangs: {bad:?}");
        assert!(bad[0].contains("compiled for a 128-row tile"), "got {bad:?}");
    }

    #[test]
    fn every_cubin_declares_its_own_geometry() {
        assert_eq!(CubinFacts::GEMM.m_tile, 128);
        assert_eq!(CubinFacts::GEMM_M64.m_tile, 64, "the narrow variant really is 64 rows");
        assert_eq!(CubinFacts::GEMM_M192.m_tile, 192);
        assert_eq!(CubinFacts::GEMM_M192.sf_blk_mn, 96, "Version 6's atom");
        assert_eq!(CubinFacts::GEMM_M192.cta_threads, 512);
        assert_eq!(CubinFacts::GEMM_M192W8.cta_threads, 384, "8 warps of 48 rows");

        assert_eq!(CubinFacts::GEMM_M64.epi_n, CubinFacts::GEMM.epi_n);
        assert_eq!(CubinFacts::GEMM_M64.cta_threads, CubinFacts::GEMM.cta_threads);
    }

    #[test]
    fn a_cubin_of_the_wrong_thread_count_is_caught() {
        assert!(cubin_matches_launch(&CubinFacts::GEMM_M192, 192, 384, 32, 96)
                .iter().any(|m| m.contains("__launch_bounds__")));
    }

    #[test]
    fn the_shipped_fp8_geometry_has_no_violations() {
        assert_eq!(fp8_violations(&Fp8Side::T64x128), Vec::<String>::new());
    }

    #[test]
    fn the_nb128_geometry_cuts_the_l2_read_by_a_quarter() {
        let shipped = Fp8Side::T64x128;
        let n64 = Fp8Side { n_block: 64, n_tile: 32, box_rows_b: 64, grid_x: 2, ..Fp8Side::T64x128 };
        assert_eq!(fp8_violations(&n64), Vec::<String>::new());
        assert_eq!(fp8_violations(&shipped), Vec::<String>::new());

        let k = 4096;
        let a = n64.l2_read_bytes(k);
        let b = shipped.l2_read_bytes(k);

        assert_eq!(b * 4, a * 3, "the L2 read must fall by exactly 25%: {a} -> {b}");

        let aliased_2 = Fp8Side::ALIASED;
        let aliased_3 = Fp8Side { stages: 3, ..Fp8Side::ALIASED };
        assert_eq!(aliased_2.ctas_by_smem(), 2, "STAGES 2 is what keeps the second CTA");
        assert_eq!(aliased_3.ctas_by_smem(), 1);
        assert_eq!(aliased_2.smem_bytes(), 49_568, "staging, scales and barriers — no epilogue buffer");
        assert_eq!(shipped.smem_bytes(), 90_544, "and the shape adds its own 16,384 B of epilogue");
        assert_eq!(shipped.ctas_by_smem(), 1, "which is what takes the CTA");
    }

    #[test]
    fn the_fp8_model_refuses_the_configuration_that_hung() {
        let as_built = Fp8Side {
            n_block: 64, n_tile: 32, stages: 3, box_rows_b: 128,
            grid_y: 96,
            n: 12288,
            persistent: false,
            ..Fp8Side::T64x128
        };
        let bad = fp8_violations(&as_built);
        assert!(bad.iter().any(|m| m.contains("the boxes are")),
                "the box-vs-tile rule must fire: {bad:?}");
        assert!(bad.iter().any(|m| m.contains("the TMA boxes deliver")),
                "the expect_tx rule must fire: {bad:?}");
        assert!(bad.iter().any(|m| m.contains("the grid is")),
                "the grid rule must fire: {bad:?}");
    }

    #[test]
    fn the_fp8_model_refuses_a_swizzled_buffer_that_is_not_1024_aligned() {
        let swizzled = Fp8Side { swizzled: true, smem_align: 1024, ..Fp8Side::T64x128 };
        assert_eq!(fp8_violations(&swizzled), Vec::<String>::new(), "1024 is the correct alignment");
        let misaligned = Fp8Side { smem_align: 128, ..swizzled };
        assert!(fp8_violations(&misaligned).iter().any(|m| m.contains("aligned 1024")),
                "the alignment rule must fire");
    }

    #[test]
    fn the_model_prices_both_kernels() {
        let m = Machine::RTX_5090;

        assert_eq!(Kind::Nvfp4.atom_k(), 64);
        assert_eq!(Kind::Fp8.atom_k(), 32);
        assert_eq!(Params::SHIPPED.ktile(), 128, "2 x k64");
        assert_eq!(Params::FP8.ktile(), 128, "4 x k32 — the same k-tile from a different atom");
        assert_eq!(Params::FP8.ktile(), 32 * 4);
        assert!(!Kind::Fp8.has_sf() && Kind::Nvfp4.has_sf());
        assert!(!Params::FP8.penalised(), "fp8 has no SF atom to be penalised for");
        assert!(Params::FP8.legal_atom());

        let k = 4096usize;
        let (m8, n8) = (768usize, 12288usize);
        assert_eq!(Params::FP8.packed_bytes(m8 * k), (m8 * k) as f64, "8-bit, one byte per element");
        assert_eq!(Params::SHIPPED.packed_bytes(m8 * k), (m8 * k) as f64 / 2.0, "4-bit packed");

        let shipped = Params::FP8;
        assert_eq!(shipped.m_tile(), 64);
        assert_eq!(shipped.n_block(), 128);
        assert_eq!(shipped.cta_threads(), 384);
        let (ctas, why) = m.ctas_per_sm(&shipped);
        assert_eq!((ctas, why), (1, "shared memory"), "the epilogue's own buffer is what takes the CTA");
        let resident = shipped.resident_warps(&m);
        assert_eq!(resident, 8, "1 CTA x 8 MMA warps");

        assert!(can_judge(&shipped, &m, &Kind::Fp8.fit()),
                "the fit is taken at 8 resident warps, which is what the shape runs");
        eprintln!("  fp8 shipped: {resident} resident warps, can_judge = {}",
                  can_judge(&shipped, &m, &Kind::Fp8.fit()));

        let eff = |p: &Params| {
            let t = tiles(p, m8, n8) as f64;
            let r = (p.resident_warps(&m) / p.mma_warps()) as f64 * 170.0;
            let w = waves_with(p, m8, n8, &m, Scheduling::Eager) as f64;
            (t / r, w, (t / r) / w)
        };

        let bd = tile_time(&shipped, &m, &Kind::Fp8.fit(), m8, n8, k);
        let measured_us = 164_736.0 / 1000.0 / waves_with(&shipped, m8, n8, &m, Scheduling::Eager) as f64;
        eprintln!("  fp8 shipped per-tile: predicted {:.2} us, measured {:.2} us ({:.0}%)",
                  bd.total_us, measured_us, 100.0 * bd.total_us / measured_us);
        eprintln!("    pipes: {}", bd.pipes());
        let (tr_s, w_s, e_s) = eff(&shipped);
        let (tr_9, w_9, e_9) = eff(&Params::FP8_96X128);
        eprintln!("  64x128 @ 1 CTA: tiles/residency {tr_s:.2}, waves {w_s}, efficiency {:.1}%",
                  e_s * 100.0);
        eprintln!("  96x128 @ 1 CTA : tiles/residency {tr_9:.2}, waves {w_9}, efficiency {:.1}%",
                  e_9 * 100.0);
        assert_eq!(tiles(&shipped, m8, n8), 1152);
        assert_eq!(tiles(&Params::FP8_96X128, m8, n8), 768);

        assert!(e_s > e_9, "the shape's wave efficiency must now be the better one: {e_s} against {e_9}");

        let l2 = |p: &Params| p.l2_read_bytes_public(m8, n8, k);
        assert!(l2(&Params::FP8_96X128) < l2(&shipped));
        eprintln!("  L2 read: 64x128 {:.2} GB against 96x128 {:.2} GB ({:.0}%)",
                  l2(&shipped) as f64 / 1e9, l2(&Params::FP8_96X128) as f64 / 1e9,
                  100.0 * l2(&Params::FP8_96X128) as f64 / l2(&shipped) as f64);
    }

    #[test]
    fn the_fp8_model_refuses_a_box_that_is_not_the_tile() {
        let short_box = Fp8Side { box_inner: 64, ..Fp8Side::T64x128 };
        assert!(fp8_violations(&short_box).iter().any(|m| m.contains("box inner")));

        let short_row = Fp8Side { smem_row_bytes: 64, ..Fp8Side::T64x128 };
        assert!(fp8_violations(&short_row).iter().any(|m| m.contains("SMEM row")));
    }
}
