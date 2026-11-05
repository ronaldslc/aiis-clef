use std::path::{Path, PathBuf};
use std::process::Command;

const ARCH: &str = "sm_120a";

const KERNELS: &[&str] = &["swizzle", "gemm", "fp8gemm", "gdn", "gdn_chunk", "elementwise", "attention", "quantise", "bf16gemm", "head", "attention_direct", "attention_smem", "attention_wide"];

const NO_MMA_KERNELS: &[&str] = &["swizzle", "gdn", "gdn_chunk", "elementwise", "attention", "quantise", "bf16gemm", "head", "attention_direct", "attention_smem", "attention_wide"];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let kernels = manifest.join("kernels");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    let nvcc = find_nvcc();
    if nvcc.is_none() {
        println!(
            "cargo:warning=nvcc not found; the CUDA kernels were NOT built and their tests will skip"
        );

        for name in KERNELS {
            std::fs::write(out.join(format!("{name}.cubin")), b"").expect("write empty cubin");
        }
        std::fs::write(out.join("gemm_m64.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("gemm_gated.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("gemm_gated_noepi.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("gemm_m192.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_plain.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_gated.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_aliased.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_noprefetch.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_reg64.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_reg160.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_k64.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_stages2.cubin"), b"").expect("write empty cubin");
        for v in ["epi1", "epi3", "epin16", "prod64", "prod24", "w2x4", "t128x128", "n64"] {
            std::fs::write(out.join(format!("fp8gemm_{v}.cubin")), b"").expect("write empty cubin");
        }
        std::fs::write(out.join("fp8gemm_big.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_reg192.cubin"), b"").expect("write empty cubin");
        std::fs::write(out.join("fp8gemm_aliased_plain.cubin"), b"").expect("write empty cubin");
        println!("cargo:rustc-cfg=no_cuda");
        return;
    }
    let nvcc = nvcc.expect("checked");

    let mut variants: Vec<(&str, &str, &[&str])> = KERNELS
        .iter()
        .filter(|n| **n != "gemm")
        .map(|n| (*n, *n, &[][..]))
        .collect();
    variants.extend([
        ("gemm", "gemm", &[][..]),
        ("gemm", "gemm_m64", &["-DWARP_M_ROWS=16"][..]),

        ("gemm", "gemm_gated", &["-DGATED_GEMM=1", "-DN_TILE=32"][..]),

        ("gemm", "gemm_gated_noepi", &["-DGATED_GEMM=1", "-DN_TILE=32", "-DGATED_NO_EPI=1"][..]),

        ("fp8gemm", "fp8gemm", &[][..]),

        ("fp8gemm", "fp8gemm_plain", &["-DFP8_PLAIN_ATOM=1"][..]),

        ("fp8gemm", "fp8gemm_gated", &["-DGATED_GEMM_FP8=1", "-DSTAGES=2"][..]),

        ("attention", "attention", &[][..]),

        ("attention_direct", "attention_direct", &[][..]),

        ("attention_smem", "attention_smem", &[][..]),

        ("attention_wide", "attention_wide", &[][..]),

        ("attention_mma", "attention_mma", &[][..]),

        ("attention_mma", "attention_mma_mt2", &["-DM_TILES=2"][..]),

        ("attention_mma", "attention_mma_msplit", &["-DNCG=1"][..]),

        ("attention_mma", "attention_mma_bk32", &["-DBK=32", "-DNCG=1"][..]),

        ("attention_mma", "attention_mma_qreg96",
         &["-DQ_IN_REGS", "-DBK=96", "-DATT_GQA_PACK", "-DNCG=1"][..]),

        ("attention_mma", "attention_mma_partial256",
         &["-DQ_IN_REGS", "-DBK=32", "-DATT_GQA_PACK", "-DNCG=1",
           "-DATT_PARTIAL", "-DATT_NONCAUSAL", "-DMMA_REG_MAX=232", "-DPRODUCER_REG_MAX=40"][..]),
        ("attention_mma", "attention_mma_partial",
         &["-DQ_IN_REGS", "-DBK=32", "-DHD_MAX=64", "-DATT_GQA_PACK", "-DNCG=1",

           "-DATT_PARTIAL", "-DATT_NONCAUSAL",
           "-DMMA_REG_MAX=232", "-DPRODUCER_REG_MAX=40"][..]),

        ("attention_mma", "attention_mma_ws_noskip",
         &["-DQ_IN_REGS", "-DBK=32", "-DATT_GQA_PACK", "-DNCG=1", "-DATT_WS=1",
           "-DMMA_REG_MAX=232", "-DPRODUCER_REG_MAX=40", "-DATT_NO_SKIP"][..]),

        ("attention_mma", "attention_mma_ws",
         &["-DQ_IN_REGS", "-DBK=32", "-DATT_GQA_PACK", "-DNCG=1", "-DATT_WS=1", "-DMMA_REG_MAX=232", "-DPRODUCER_REG_MAX=40"][..]),

        ("attention_mma", "attention_mma_ws_split",
         &["-DQ_IN_REGS", "-DBK=32", "-DATT_GQA_PACK", "-DNCG=1", "-DATT_WS=1", "-DATT_PARTIAL",
           "-DMMA_REG_MAX=240", "-DPRODUCER_REG_MAX=24"][..]),

        ("attention_mma", "attention_mma_ws_r240",
         &["-DQ_IN_REGS", "-DBK=32", "-DATT_GQA_PACK", "-DNCG=1", "-DATT_WS=1", "-DMMA_REG_MAX=240",
           "-DPRODUCER_REG_MAX=24"][..]),

        ("attention_mma", "attention_mma_ws_bk48",
         &["-DQ_IN_REGS", "-DBK=48", "-DATT_GQA_PACK", "-DNCG=1", "-DATT_WS=1", "-DMMA_REG_MAX=232",
           "-DPRODUCER_REG_MAX=40"][..]),

        ("attention_mma", "attention_mma_qreg",

         &["-DQ_IN_REGS", "-DBK=64", "-DATT_GQA_PACK", "-DNCG=1"][..]),
        ("attention_mma", "attention_mma_gqa_bk32",
         &["-DATT_GQA_PACK", "-DBK=32", "-DNCG=1"][..]),

        ("attention_mma", "attention_mma_gqa", &["-DATT_GQA_PACK", "-DNCG=1"][..]),

        ("attention_mma", "attention_mma_bk64", &["-DBQ=64", "-DBK=64", "-DNCG=1"][..]),

        ("attention_mma", "attention_mma_bk64w", &["-DBQ=64", "-DBK=64", "-DNCG=2"][..]),

        ("attention_mma", "attention_mma_hd128", &["-DHD_MAX=128", "-DBK=64", "-DNCG=1"][..]),

        ("attention_mma", "attention_mma_bq160", &["-DBQ=160", "-DNCG=1"][..]),
        ("attention", "attention_r8", &["-DATT_ROWS=8"][..]),

        ("attention", "attention_r4", &["-DATT_ROWS=4"][..]),

        ("attention", "attention_r12", &["-DATT_ROWS=12"][..]),

        ("bf16gemm", "bf16gemm_n64", &["-DN_TILE=32"][..]),

        ("bf16gemm", "bf16gemm_n64w4", &["-DN_TILE=16", "-DWARPS_N=4"][..]),

        ("fp8gemm", "fp8gemm_noprefetch", &["-DFP8_PREFETCH=0"][..]),

        ("fp8gemm", "fp8gemm_reg64", &["-DMMA_REG_MAX=64"][..]),

        ("fp8gemm", "fp8gemm_k64", &["-DSUB_CHUNKS=2"][..]),

        ("fp8gemm", "fp8gemm_stages2", &["-DSTAGES=2"][..]),

        ("fp8gemm", "fp8gemm_epi1", &["-DEPI_BUFS=1"][..]),
        ("fp8gemm", "fp8gemm_epi3", &["-DEPI_BUFS=3"][..]),

        ("fp8gemm", "fp8gemm_epin16", &["-DEPI_N=16"][..]),

        ("fp8gemm", "fp8gemm_prod64", &["-DPRODUCER_REG_MAX=64"][..]),
        ("fp8gemm", "fp8gemm_prod24", &["-DPRODUCER_REG_MAX=24"][..]),

        ("fp8gemm", "fp8gemm_w2x4",
         &["-DWARPS_M=2", "-DWARPS_N=4", "-DWARP_M_ROWS=32", "-DN_TILE=32"][..]),

        ("fp8gemm", "fp8gemm_t128x128",
         &["-DWARP_M_ROWS=32", "-DSTAGES=2"][..]),

        ("fp8gemm", "fp8gemm_n64", &["-DN_TILE=32"][..]),

        ("fp8gemm", "fp8gemm_big",
         &["-DWARPS_M=8", "-DWARPS_N=1", "-DWARP_M_ROWS=32", "-DN_TILE=64", "-DSUB_CHUNKS=2",
           "-DEPI_BUFS=1", "-DMMA_REG_MAX=192"][..]),
        ("fp8gemm", "fp8gemm_reg160", &["-DMMA_REG_MAX=160"][..]),

        ("fp8gemm", "fp8gemm_reg160", &["-DMMA_REG_MAX=160"][..]),

        ("fp8gemm", "fp8gemm_k64", &["-DSUB_CHUNKS=2"][..]),
        ("fp8gemm", "fp8gemm_big",
         &["-DWARPS_M=8", "-DWARPS_N=1", "-DWARP_M_ROWS=32", "-DN_TILE=64", "-DSUB_CHUNKS=2",
           "-DEPI_BUFS=1", "-DMMA_REG_MAX=192"][..]),
        ("fp8gemm", "fp8gemm_reg192", &["-DMMA_REG_MAX=192"][..]),

        ("fp8gemm", "fp8gemm_aliased", &["-DSTAGES=2", "-DEPI_SEPARATE=0", "-DMMA_REG_MAX=64"][..]),

        ("fp8gemm", "fp8gemm_aliased_plain",
         &["-DSTAGES=2", "-DEPI_SEPARATE=0", "-DMMA_REG_MAX=64", "-DFP8_PLAIN_ATOM=1"][..]),

        (
            "gemm",
            "gemm_m192",
            &["-DWARPS_M=6", "-DMMA_REG_MAX=152", "-DSTAGES=2"][..],
        ),

        (
            "gemm",
            "gemm_m192w8",
            &["-DWARPS_M=4", "-DWARP_M_ROWS=48", "-DWARPS_N=2", "-DMMA_REG_MAX=208",
              "-DSTAGES=2"][..],
        ),

        (
            "gemm",
            "gemm_stages3",
            &["-DSTAGES=3"][..],
        ),

        (
            "gemm",
            "gemm_atom32",
            &["-DSF_BLK_MN=32"][..],
        ),

    ]);
    for (name, out_name, extra) in variants {
        let src = kernels.join(format!("{name}.cu"));
        println!("cargo:rerun-if-changed={}", src.display());
        if !src.exists() {
            panic!("kernel source missing: {}", src.display());
        }
        let cubin = out.join(format!("{out_name}.cubin"));

        let built = Command::new(&nvcc)
            .args([
                "-arch", ARCH, "-cubin", "-O3", "-lineinfo", "--ptxas-options=-v", "-o",
            ])
            .arg(&cubin)
            .args(extra)
            .arg(&src)
            .output()
            .unwrap_or_else(|e| panic!("could not run nvcc: {e}"));

        if !built.status.success() {
            panic!(
                "nvcc failed on {}:\n{}",
                src.display(),
                String::from_utf8_lossy(&built.stderr)
            );
        }

        let stderr = String::from_utf8_lossy(&built.stderr);
        if stderr.contains("redefined") {
            panic!(
                "{out_name}: a -D flag was IGNORED — the kernel source redefined it and the source wins. \
                 Guard the knob with #ifndef in the .cu, or the variant is a lie.\n{}",
                stderr.lines().filter(|l| l.contains("redefined")).collect::<Vec<_>>().join("\n")
            );
        }
        report_resource_budget(out_name, &stderr);

        if NO_MMA_KERNELS.contains(&name) {
            continue;
        }
        let sass = dump_sass(&nvcc, &cubin);

        if !sass.contains("OMMA") && !sass.contains("QMMA") && !sass.contains("HMMA") {
            panic!(
                "{} compiled without a tensor-core MMA (OMMA, QMMA or HMMA) instruction.\n\
                 The target is {ARCH}; if this is a new kernel that legitimately has no MMA, add it \
                 to NO_MMA_KERNELS in build.rs rather than removing this check.\n\
                 SASS:\n{sass}",
                src.display()
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn report_resource_budget(name: &str, ptxas: &str) {
    let number_before = |needles: &[&str]| -> Option<usize> {
        for needle in needles {
            if let Some(idx) = ptxas.find(needle) {
                if let Some(word) = ptxas[..idx].split_whitespace().next_back() {
                    let digits = word.trim_end_matches(|c: char| !c.is_ascii_digit());
                    if let Ok(v) = digits.parse::<usize>() {
                        return Some(v);
                    }
                }
            }
        }
        None
    };
    let registers = number_before(&["registers"]);
    let stack = number_before(&["bytes stack frame", "stack frame"]);
    let spills = number_before(&["bytes spill stores", "spill stores"]);
    if let Some(r) = registers {
        if r > 42 {
            println!(
                "cargo:warning={name}: {r} registers per thread — above the 42 that 6 blocks/SM at 8 \
                 warps needs, so occupancy drops below the 48 warp slots. See build.rs."
            );
        }
    }
    if stack.unwrap_or(0) > 0 || spills.unwrap_or(0) > 0 {
        println!(
            "cargo:warning={name}: {} bytes of stack frame and {} of spills — a per-thread array is in \
             local memory. A runtime loop bound is the usual cause; a compile-time bound and an \
             unrolled loop with a uniform guard promote it to registers. See build.rs.",
            stack.unwrap_or(0),
            spills.unwrap_or(0)
        );
    }
}

fn find_nvcc() -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(dir) = std::env::var(var) {
            let candidate = Path::new(&dir).join("bin/nvcc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    let conventional = Path::new("/usr/local/cuda/bin/nvcc");
    if conventional.exists() {
        return Some(conventional.to_path_buf());
    }

    let probe = Command::new("nvcc").arg("--version").output();
    match probe {
        Ok(o) if o.status.success() => Some(PathBuf::from("nvcc")),
        _ => None,
    }
}

fn dump_sass(nvcc: &Path, cubin: &Path) -> String {
    let cuobjdump = nvcc
        .parent()
        .map(|p| p.join("cuobjdump"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("cuobjdump"));

    let out = Command::new(&cuobjdump)
        .arg("-sass")
        .arg(cubin)
        .output()
        .unwrap_or_else(|e| panic!("could not run cuobjdump ({}): {e}", cuobjdump.display()));

    if !out.status.success() {
        panic!(
            "cuobjdump failed on {}: {}",
            cubin.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}
