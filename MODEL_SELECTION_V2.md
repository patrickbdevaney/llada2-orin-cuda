# Model selection v2: measured Orin bandwidth plus the wider small-dLLM field (2026-10-24)

This supersedes the model recommendation in RESEARCH_AND_PLAN.md §2. The engine design (§6) and
the phased plan (§7) carry over, with the model-specific parts swapped.

## 1. Measured on this Orin Nano Super (MAXN_SUPER, GPU 1020 MHz)

Benchmark: `bench/bw_stream.cu`, read-only streaming kernels, random data so
compression cannot inflate the result.

| access pattern | 64–256 MB | 512 MB – 3 GB |
|---|---|---|
| grid-stride, interleaved across the whole buffer | 99–101 GB/s | **75–82 GB/s**, from GPU TLB thrash |
| each CTA streams its own contiguous chunk | 99–101 GB/s | **99–100 GB/s** |
| 24 separate 128 MB buffers back-to-back | 100.4 GB/s | |

Findings:
- **The real ceiling is ~100 GB/s, 98% of the 102.4 GB/s spec.** Planning number: ~90 GB/s
  effective.
- **Kernel rule:** every weight-streaming kernel must have each CTA walk a contiguous slab, in a
  Marlin-style tile-major layout. Never stride the whole matrix across the grid. The difference
  is worth +25%.
- NVMe sequential read is 2.8 GB/s, 1/36 of DRAM. Offloading experts to NVMe is not viable for
  decode.

## 2. Candidates that can run on 8 GB

| model | type | quality class | license | prefill cost | fits on Orin as | projected decode @ 90 GB/s |
|---|---|---|---|---|---|---|
| **LLaDA2.2-mini** (inclusionAI, Sep 2026) | MoE dLLM: 16.3B total / 1.4B active, 20 layers, 256 experts (top-8 + 1 shared), block 32, **block routing ≤48 experts/layer/block**, T2T + Levenshtein edits | ≈ Qwen3-8B (no-think). The 2.0-mini scored avg 71.7 vs 70.2: IFEval 80.8, BFCL-live 70.9, HumanEval 86.6. 2.2 adds agentic RL: τ²-Bench 57.5, BFCL-v4 47.7, LongBench-v2 35.0 | Apache-2.0 | ~1×, about 2.2 GFLOP/token | experts at ~2.5–3.0 bpw → 5.3–6.2 GB, or prune + 3.5–4 bpw → 4.6–4.7 GB | **~115–260 tok/s** (TPF 2–4) |
| Nemotron-Labs-Diffusion-3B (NVIDIA, Mar 2026) | dense, 26 layers, h3072. AR, diffusion and self-speculation modes | ≈ Qwen3-4B. IFEval 69 | NVIDIA Open | 1× | W4 → 2.0 GB | ~100–135 tok/s |
| ALoDLM-8B (Amazon, Oct 2026) | looped dense Qwen3-8B | ≥ Qwen3-8B on its suite; no IFEval or tool-use evaluation | CC BY-NC | **3.17×** | W4 → 4.3 GB | ~35–60 tok/s |
| ALoDLM-1.7B | looped dense | ≈ Qwen3-1.7B | CC BY-NC | 4× | W4 → 1.1 GB | ~130–210 tok/s |
| Nemotron-Labs-Diffusion-8B / VLM-8B | dense, 34 layers | ≈ Qwen3-8B. IFEval 69 | NVIDIA Open / NSCL (VLM) | 1× | W4 → ~4.6 GB | ~40–55 tok/s |
| DiffusionGemma 26B-A4B, LLaDA2.2-flash 100B | MoE | | | | do not fit (≥ 7.9 GB even at 2.5 bpw) | – |

Notes on the quality column:
- Each family uses its own benchmark suite, so the quality-class column is approximate.
- The LLaDA2.2-mini card compares only against earlier LLaDA versions. Its Qwen3-8B parity is
  inherited from the 2.0-mini table.
- The 2.2 release trades some IFBench and GPQA for BFCL-v4, agentic and long-context scores.

**Verdict:** LLaDA2.2-mini is the only option that plausibly delivers all three goals at once:
- **intelligence:** Qwen3-8B class, trained for tool use and agents, 128K context;
- **prefill:** 1.4B active, so roughly 10× ALoDLM-8B prompt throughput, and a 4K agent prompt
  should take ~2–4 s;
- **decode >100 tok/s:** only ~1.4–1.6 GB is read per forward because block routing caps the
  expert working set, with 2–4 tokens per forward.

**The catch is fitting 16.3B parameters into ~6.5 GB.** The whole project hinges on whether
~2.75 bpw experts, or experts pruned to fit, keep agentic quality. That has to be measured first.

## 3. LLaDA2.2-mini memory and speed model

Parameter breakdown:
- experts: 15.30B
- non-expert: 0.31B (attention 0.21B, shared experts, dense layer 0, routers)
- lm_head: 0.32B, kept INT8
- embeddings: 0.32B, held in a host mmap with only touched rows resident

Bytes per forward assume one 32-token block, at most 48 experts per layer:

| expert format | resident | bytes/fwd | fwd/s | tok/s @ TPF 2 / 3 / 4 |
|---|---|---|---|---|
| 2.5 bpw uniform | 5.28 GB | 1.39 GB | 65 | 129 / 194 / 258 |
| **2.75 bpw mixed (hot experts at 3–4 bit, cold at 2 bit)** | 5.76 GB | 1.48 GB | 61 | 121 / 182 / 243 |
| 3.0 bpw | 6.23 GB | 1.57 GB | 57 | 114 / 172 / 229 (too tight with OS + KV) |
| 4.25 bpw (Q4) | 8.63 GB | – | – | does not fit; the existing Q4_K_M GGUF is 9.9 GB |
| prune 37% of experts + 3.5 bpw | 4.71 GB | 1.75 GB | 51 | 103 / 154 / 205 |
| prune 50% + 4.25 bpw | 4.56 GB | 2.02 GB | 45 | 89 / 134 / 178 |

- KV cache is tiny: 20 layers × 4 KV heads = 40 KB/token in BF16, so 32K context is 1.3 GB.
- LLaDA2.1 reported TPF of 3.1 in its quality mode and 5.9 in its speed mode, for the flash model.
  The mini's TPF on *your* traffic is unmeasured. The T2T/edit post-steps count as forwards.

## 4. Quantization strategy for the MoE

1. **Frequency- and sensitivity-weighted mixed precision per expert.**
   - Run the BF16 model on Thor over real executor traffic.
   - Log the block-routing pools and per-expert activation counts.
   - Assign bits per expert by usage × Hessian sensitivity: hot experts at 4 bit, warm at 3,
     cold at 2. Target an average of ~2.75 bpw.
   - Use GPTQ error compensation with group 64 and FP16 scales.
   - Attention, shared experts, routers and dense layer 0 stay at W4 or better. They are only
     0.31B parameters but run on every token.
   - lm_head stays INT8; routers stay FP16/FP32. Routing flips are the main low-bit failure mode.
2. **Pruning (REAP-style) as the alternative bit budget.** Drop the experts least used on your
   domain, then spend the freed bytes on more bits. Evaluate 25% and 37% prune points.
3. **Choose plain bit-packed INT2/3/4 over trellis/QTIP or codebook formats.** Orin has only
   1024 CUDA cores. At ~4.3 G weights per forward, a trellis decode of ~6 ops per weight costs
   ~13 ms of ALU time, the same as the memory time. Bit-unpack plus the lop3 dequant is about
   1–2 ops per weight and hides under the memory stream. Confirm with a microbenchmark before
   ruling trellis out; your laguna TRELLIS_VERDICT.md is the prior art here.
4. **Gate:** the quantized model must stay within ~2 points of BF16 on your executor task
   success and on BFCL / IFEval, and TPF must not drop by more than 10%. Low-bit noise lowers
   confidence, which means fewer tokens clear the threshold per step.

## 5. Kernels this changes, relative to the dense ALoDLM plan

- **Block-routed MoE.** Per forward: router → per-block top-48 pool → per-token top-8 inside the
  pool. Then run a *grouped skinny GEMM over the ≤48 pooled experts*, each expert streamed
  contiguously exactly once with its ≤32 token rows gathered. This is the bandwidth-critical
  kernel. Fuse the gate/up projections with SiLU·mul, and fuse down with a weighted scatter-add.
- **Mixed-bit expert storage.** Each expert carries its own bit-width tag. Compile the kernel
  per bit-width (2, 3 and 4), with one launch per bit class, or a persistent kernel with a work
  queue sorted by expert.
- **Attention:** GQA 16:4, head_dim 128, partial RoPE (64 of 128 dims), qk-norm. Bidirectional
  inside the current block, causal to the prefix KV.
- **Decoder state machine on the GPU:**
  - M2T: fill masks above the threshold, with a per-step floor.
  - T2T and Levenshtein DELETE/INSERT edits within the active block.
  - Post-edit steps.
  - The block is committed to KV only when it is final, so edits never touch cached KV.
- **Prefill:** large-M grouped GEMM over all experts, which is compute-bound. Use FP16
  mma.sync, and W4A8 INT8 mma for the ≥3-bit experts later on.
- **Free n-gram drafting.** T2T editing can rewrite tokens, so mask slots can be pre-filled
  with prompt-lookup continuations as provisional tokens. The model's own edit pass then
  accepts or rewrites them. This works for copy-heavy executor output such as code edits,
  paths and JSON. Experimental.

## 6. Revised phase 0 (on Thor, before any Orin CUDA work)

1. **BF16 LLaDA2.2-mini reference.** Use the HF `generate` with the SGLang defaults: block 32,
   threshold 0.5, edit 0.0, post-steps 16. Run it on ~100 captured executor prompts, recording
   success, tool-call validity, TPF, forwards per block, and expert-pool statistics.
2. **Same prompts on Nemotron-Diffusion-3B and ALoDLM-8B**, for the quality/TPF comparison.
3. **Simulated quantization sweep on LLaDA2.2-mini.** Fake-quant experts at 2.5, 2.75 and 3.0
   bpw mixed, plus prune25 + 3.5 and prune37 + 3.5. Re-measure success and TPF.
4. **Decision:**
   - If the 2.75 bpw variant holds quality, build LLaDA2.2-mini.
   - If it fails but prune + 3.5 holds, build that.
   - If both fail, fall back to Nemotron-Diffusion-3B W4: easy fit, ~100–135 tok/s,
     Qwen3-4B-class quality.

Orin prerequisites that are done or measured: MAXN_SUPER is active, bandwidth is measured, and
a local nvcc 12.6 exists in the session scratchpad. Your system nvcc install was not visible
yet; check that `/usr/local/cuda-12.6/bin/nvcc` exists. The DreamBlooms GGUF repo ships a
llama.cpp `0002-llada2.patch`. That makes a quick functional, non-optimized baseline possible
once a ≤6 GB GGUF quant exists, though Q4_K_M at 9.9 GB does not fit.
