# aiis-clef — AI inference server for the Clef decision model

A single-binary, CUDA-native inference server for **Clef**, a hybrid Gated-DeltaNet / attention
decision model, served over the SystemOne contract (`POST /v1/systemone`). Rust for the host, our
own CUDA kernels for the math, no Python in the serving path.

The model answers typed questions about a `state` in **one forward pass** — it generates no tokens.
Every question is scored jointly, so a request is a prefill plus a small decision head.

## Headline: how it compares

One RTX 5090, one sitting, three rotated rounds per arm, every arm on the **same NVFP4 checkpoint**
except the full-precision reference. Median end-to-end latency, and the ratio against this server:

| input tokens | 778 | 8.2k | 16.4k | 32.7k | 65.5k | 130.7k |
| --- | --- | --- | --- | --- | --- | --- |
| **aiis-clef** | **40 ms** | **320** | **691** | **1,525** | **3,978** | **10,897** |
| vLLM | 53 | 426 | 778 | 1,672 | 4,692 | 13,025 |
| llama.cpp | 53 | 561 | 1,235 | 2,884 | — ¹ | — ¹ |
| full precision (bf16) | 85 | 769 | 1,510 | 3,145 | 7,262 | 20,544 |
| **vs vLLM** | **1.32×** | **1.33×** | **1.13×** | **1.10×** | **1.18×** | **1.20×** |
| **vs llama.cpp** | **1.31×** | **1.75×** | **1.77×** | **1.90×** | — | — |
| **vs bf16** | **2.10×** | **2.40×** | **2.18×** | **2.06×** | **1.83×** | **1.89×** |

**Accuracy**: mean KL against the full-precision answer is **1.6e-05 → 4.1e-03** from 778 to 130.7k
tokens, with **0 of 60 questions changing their top answer** at every size. The 4-bit weights are the
checkpoint's own NVFP4 tensors — nothing is re-quantised at load.

¹ llama.cpp's decision path evaluates a prompt in one physical batch, so its ubatch *is* its maximum
prompt; above ~32k the buffers do not fit on a 32 GB card. Its weights are the same NVFP4 tensors,
converted with the E4M3 scale bits preserved.

Full methodology, per-size charts, concurrency and the accuracy scatter: **[report/benchmark.html](report/benchmark.html)**.

## Requirements

| | |
| --- | --- |
| **GPU** | NVIDIA RTX 5090 (sm_120). The kernels are compiled for `sm_120` only — a different card needs `ARCH` in `build.rs` changed, and the tile shapes are tuned for this one |
| **Driver** | any driver that supports CUDA 13.0 (the build links `cuda-13030`) |
| **CUDA toolkit** | CUDA 13.x, for `nvcc` at **build** time only. The runtime uses the driver API through `cudarc`, so the toolkit is not needed on the serving host |
| **Rust** | 1.75+ (2021 edition) |
| **VRAM** | ~10 GB for the checkpoint plus the KV carry: **4,146 MiB** at a 131,072-token context, ~1.5 GB of graph pool. A 131k request peaks near **30 GB** |
| **Checkpoint** | a Clef NVFP4 directory (compressed-tensors `safetensors` + `joint_head_config.json` + tokenizer), pointed at by `CLEF_MODEL_DIR` |

## Build

```bash
cargo build --release          # needs nvcc on PATH; compiles the kernels to cubins and embeds them
./target/release/aiis-clef --help
```

`build.rs` finds `nvcc` (`$CUDARC_CUDA_VERSION`, `/usr/local/cuda/bin`, `PATH`), compiles each kernel
for `sm_120`, and fails the build if a `-D` it passes is discarded by the source — a silently ignored
define is how a kernel once shipped with the wrong tile.

## Run

```bash
export CLEF_MODEL_DIR=/models/clef-flash-nvfp4     # the checkpoint directory
export PORT=8000
./target/release/aiis-clef
```

Then:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "User question: where does agent identity live?\nItems:\nitem 0:\nid: doc-0\ntitle: Note\ntext: ...",
  "questions": {
    "ans_0": {"type": "noul", "instructions": "Does the item answer the question?"}
  }
}'
```

The answer carries `probabilities` for `choice`/`score` questions and `noul` (P(true)) for `noul`.

## Configuration

None of these are required; the defaults are what the benchmark above measured.

| variable | default | what it does |
| --- | --- | --- |
| `CLEF_MODEL_DIR` | `/models/clef-flash-nvfp4` | checkpoint directory |
| `CLEF_HEAD_DIR` | `/models/clef-flash` | directory holding the joint head |
| `PORT` | `8000` | listen port |
| `CLEF_MAX_LENGTH` | the device's measured ceiling | longest request accepted; a request above it is refused with a 400 naming the numbers |
| `CLEF_KV_SLOTS` | `1` | requests in flight, i.e. the widest shared pass. A memory decision: a second slot does not fit the graph budget at the 8,192 chunk |
| `CLEF_MAX_CHUNK` | `8192` | the longest chunk one request contributes, and the top of the graph's geometry set |
| `CLEF_MAX_CONCURRENT` | `4` | how many requests the queue may coalesce |

## How it works, in one paragraph

A request is cut into rung-sized chunks (64 … 8,192 rows) and each pass runs the whole layer stack
once, replayed from a captured CUDA graph keyed on the pass geometry. Every per-request bound is a
**per-token map** (`cpos`, `cbase`, `seq_end`, `state_slot`), so a pass can carry several requests and
a request's answer does not depend on what it was batched with. Weights stay resident in their packed
NVFP4/FP8 form; the GEMMs read them directly. The three carries across a chunk boundary — the GDN's
recurrent state, the conv's history and attention's KV — are explicit buffers, which is what makes a
chunked request bit-identical to one pass.

## Attribution

The kernels and the engine were written for this project. The tiling, the NVFP4 block-scaled MMA
layout and the Gated-DeltaNet formulation follow the published work of the reference implementations
surveyed in the parent research tree (candle, llama.cpp, vLLM, TensorRT-LLM and the Qwen technical
reports). `cudarc` is the CUDA driver binding; `tokenizers` and `safetensors` are Hugging Face crates.

## License

MIT — see [LICENSE](LICENSE).
