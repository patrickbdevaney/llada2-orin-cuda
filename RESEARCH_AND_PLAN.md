# ALoDLM on Jetson Orin Nano Super (8 GB): research findings and implementation plan

Date: 2026-10-24. Target: a single-stream pure-CUDA engine plus an OpenAI-compatible server,
built for the fastest decode on Orin Nano Super. It follows the same pattern as
laguna-s1 / qwen38-3090 / glm-5.3-flash.

---

## 1. What ALoDLM actually is (from the configs, code and paper, not the marketing)

Paper: arXiv 2610.04198 (2026-10-03, Amazon). Code: github.com/amazon-science/ALoDLM.
Weights: **CC BY-NC 4.0** (non-commercial).

- **Backbone:** a stock Qwen3 checkpoint (`Qwen3ForCausalLM`, GQA with 8 KV heads, head_dim 128,
  vocab 151936, untied lm_head). The diffusion behaviour comes entirely from the decoder plus a
  4 KB linear `exit_gate.pt`, configured by `alodlm_config.json`.
- **Looped layers:**
  | | layers | prelude | recurrent core (≤4×) | coda | KV buffer pairs |
  |---|---|---|---|---|---|
  | 1.7B | 28 | – | **all 28** `[0,28)` | – | 112 |
  | 8B   | 36 | `[0,10)` | `[10,26)` | `[26,36)` | 84 |
- **Decode loop:** WeDLM-style, with causal attention and topological reordering.
  - Each *outer step* builds a window of the committed-but-not-yet-cached tokens followed by
    up to 16 `[MASK]` slots.
  - The step then runs 1–4 *recurrent passes*. Each pass is core + coda + lm_head on the mask
    rows. After each pass:
    - tokens commit if `entropy + 0.02·relpos < τ`;
    - the gate's survival CDF over uncommitted tokens decides whether to stop the step (`≥ q`).
  - Committed tokens are fed back as embeddings, and unresolved tokens keep their latent state.
  - On the next outer step, committed tokens are recomputed as observed inputs before their K/V
    enters the prefix cache.
- **KV is depth-aware.** Recurrent layers keep one K/V per depth. Depths that were never run
  inherit the last completed depth.
- **Prefill runs the recurrent core at all 4 depths.** That is 4.0× the AR prefill FLOPs at
  1.7B and 3.17× at 8B. The paper lists the resulting long time-to-first-token as a known
  limitation.
- **Speed (paper, single stream, 1×B200, GSM8K):**
  - ALoDLM-8B: 612 tok/s at 93.25% accuracy, about 2.7× vLLM Qwen3-8B (~228 tok/s).
  - Sweep: τ 0.1 → 279 tok/s at 93.8%, τ 0.6 → 508 tok/s at 92.3% (q = 0.5);
    q 0.1 → 455 tok/s, q 0.9 → 309 tok/s (τ = 0.2).
  - No 1.7B throughput is published, and no tokens-per-pass figure is published.
  - Back-solving from the B200 numbers gives roughly **~2 committed tokens per recurrent pass**
    on GSM8K in the speed modes. This is an inference, not a published number.
- **Quality (Table 1, 11-benchmark average):**
  | | ALoDLM-1.7B | Qwen3-1.7B | ALoDLM-8B | Qwen3-8B |
  |---|---|---|---|---|
  | Average | 65.5 | 63.8 | 80.3 | 78.5 |
  | MMLU-Pro | 39.6 | 39.0 | 63.9 | 56.8 |
  | HumanEval+ | 67.7 | 60.4 | 84.2 | 79.3 |
  | GPQA-D | 41.4 | 31.3 | 49.5 | 47.0 |
  - Caveat: these are quality-preset numbers (`left1`, one token per pass), so they show
    quality at AR-like speed. Fast entropy mode costs roughly 1–3 points on GSM8K.
  - **Not evaluated at all:** IFEval / instruction following, tool calling (BFCL), long
    context, multi-turn. The model card says generation length beyond 4096 tokens and long
    context are "not established".

## 2. Is this the best diffusion LM for a fast Orin executor in October 2026?

**Short answer:** ALoDLM-8B is the newest model and has the best *published* quality among
open DLMs on its own suite. It is not a clear win for an *agent executor*. The one real
competitor is NVIDIA **Nemotron-Labs-Diffusion** (arXiv 2607.05722, weights out since Mar 2026,
NVIDIA Open Model License, so commercial use is OK).

| | ALoDLM-8B | ALoDLM-1.7B | Nemotron-Labs-Diffusion-3B (~3.8B actual) | Nemotron-Labs-Diffusion-8B |
|---|---|---|---|---|
| Quality class | ≥ Qwen3-8B | ≈ Qwen3-1.7B | ≈ Qwen3-4B (avg 55.5 vs 53.2) | ≈ Qwen3-8B |
| IFEval | not reported | not reported | 69 (Qwen3-4B 72) | 69 (Qwen3-8B 87) ← weak |
| Parallel mode | entropy commit (lossy, tunable τ) | same | diffusion, **or AR-verified self-speculation** (TPF 4.4–5.4) | same (TPF 6.0–6.4) |
| Prefill cost | 3.17× AR | 4× AR | 1× | 1× |
| KV per token (int8) | 168 KB | 224 KB | 52 KB | ~70 KB |
| Context | 4K tested | 4K tested | 262K configured | 262K configured |
| License | CC BY-NC | CC BY-NC | NVIDIA Open | NVIDIA Open |
| W4 resident on Orin | ~4.3 GB | ~1.1 GB | ~2.0 GB | ~4.6 GB |

Other options ruled out:
- DiffusionGemma 26B-A4B: 13 GB+ at 4-bit, so it does not fit.
- LLaDA2 mini: 16B MoE, does not fit.
- WeDLM-8B, SDAR, Dream, LLaDA-8B: all dominated by ALoDLM-8B.
- FLARE 2B/4B: release status is unclear.
- Open-dCoder 0.5B: too weak.

**Verdict for Orin:**
1. **ALoDLM-1.7B is fast (~110–175 tok/s projected) but too weak for delegated executor
   work.** MMLU-Pro is 39.6, instruction following and tool use are unmeasured, the KV cost per
   token is the largest of any option, and it has the worst prefill multiplier. Use it only as
   a classifier, router or drafter.
2. **ALoDLM-8B W4 is the quality pick: ~40–50 tok/s projected, about 2.5× AR Qwen3-8B W4 on the
   same Orin (~18 tok/s).**
   - It fits only if the machine runs headless.
   - With long agent prompts, the 3.17× prefill (~150–180 tok/s optimistic) makes TTFT the
     bottleneck: a 4K-token prompt takes ~25 s cold. A persistent prefix cache is mandatory.
3. **Nemotron-Labs-Diffusion-3B is the speed/quality hedge.**
   - It is better than ALoDLM-1.7B and about as fast (~80–110 tok/s projected).
   - Prefill costs 1×, and AR verification guards output quality.
   - Its license is commercial-OK and its context is long.
4. Both families are dense GQA models with head_dim 128, 8 KV heads, SwiGLU and RMSNorm.
   **One kernel library serves both.** Build the engine for ALoDLM-8B first and keep the model
   layer thin so a Nemotron-3B port costs days, not weeks.

**Do not commit to a model until Phase 0 below has run on Thor against your own executor
prompts.** The unmeasured items (tool calls, instruction following, tokens per pass on code and
JSON) decide this choice more than any published benchmark does.

## 3. Orin roofline and cost model

Hardware facts:
- Ampere sm_87 with 8 SMs. FP16 dense tensor throughput is ~16.7 TFLOPS at 1020 MHz
  (MAXN_SUPER); INT8 dense is ~33 TOPS. There is no FP8/FP4.
- LPDDR5 at 102 GB/s spec, shared with the CPU. The model below assumes **80 GB/s achieved**;
  measure this in Phase 0.

**Key insight: a diffusion window is nearly free compute on Orin.**
- The ridge point is ~200 FLOP/B for BF16 weights and ~50 rows for W4A16 tensor-core GEMM.
- A 16–48-row window therefore stays memory-bound. Each recurrent pass costs about one full
  weight read, whether it carries 1 row or 32.
- So the speed formula is `tok/s ≈ BW / bytes_per_pass × tokens_per_pass`.

Projected decode speed. These are pure bandwidth figures; expect 75–85% of them after launch
and attention overheads.

| model / weights | resident | bytes/pass | pass/s | tok/s @ 1.5 / 2.0 / 2.5 tok/pass |
|---|---|---|---|---|
| ALoDLM-8B W4g128 + int8 head | 4.33 GB | 3.82 GB | 21 | 31 / 42 / 52 |
| ALoDLM-8B W4g128 + int4 head | 4.02 GB | 3.51 GB | 23 | 34 / 46 / 57 |
| ALoDLM-8B ~3.5 bpw mixed | 3.4–3.7 GB | 3.0–3.3 GB | 25–27 | 37–41 / 49–54 / 61–68 |
| ALoDLM-1.7B W4 + int8 head | 1.07 GB | 1.07 GB | 75 | 112 / 150 / 187 |
| Nemotron-Diff-3B W4 + int8 head | 2.02 GB | 2.02 GB | 40 | 59 / 79 / 99 |
| AR baseline Qwen3-8B W4 | 4.4 GB | 4.4 GB | 18 | 18 |

How the 8B bytes/pass is computed:
- One pass is core + coda + lm_head.
- The prelude runs once per outer step, amortised over ~2 passes.

BF16 for 8B (15 GB) is impossible. W8 for 8B (7.8 GB) does not fit.

## 4. Quantization and format choice

**Weights: W4A16, group 128, symmetric int4 with FP16 scales (~4.13–4.25 bpw).**
- Use GPTQ with act-order, or AWQ.
- Pack offline into a Marlin-style tile layout pre-permuted for `mma.m16n8k16` and `ldmatrix`.
  Dequantize with the lop3 trick, using the INT4→FP16 magic-number conversion.

Calibration must account for the loop:
- The recurrent core sees different activation statistics at depths 1–4.
- Collect calibration Hessians/activations from *real recurrent rollouts at all depths*,
  weighted by how often each depth runs. Calibrating only on a plain Qwen3 forward is wrong.
- Quantization error compounds across up to 4 core applications.

**Measure quantization by tokens per pass as well as accuracy.**
- Quantization noise raises output entropy, which means fewer tokens clear τ, which means
  fewer tokens per pass, which means lower speed.
- Track the triple (accuracy, tokens per pass, passes per outer step) against BF16 on Thor.
  Re-tune τ/q for the quantized model.

Precision for the other components:

| component | format | why |
|---|---|---|
| lm_head | **INT8 per-channel**, with W4g64 as an experiment | Entropy-based commit decisions depend on calibrated logits. INT8 costs ~8% speed on 8B. Try int4 and accept it only if tokens per pass and accuracy hold. |
| embeddings | INT8 rows in a host mmap (unified memory) | Only ~32 rows are touched per step, so the 0.6–1.2 GB table never needs to be hot. |
| norms, q/k-norm, exit gate, rotary | FP32 / FP16 | Tiny. The gate is 4096 + 4 parameters, and its decisions are threshold-sensitive. |
| KV cache | INT8, per-token per-head scale (FP16 when context ≤ 2K) | The depth-aware KV is 2.3–4× normal size. |
| activations | FP16 with FP32 accumulate | Decode is memory-bound, so A8 buys nothing there. |
| prefill (optional, phase 6) | W4A8 using INT8 mma, with per-token activation scale | Prefill is compute-bound at 3.17× AR FLOPs, and INT8 tensor cores give 2× FP16 throughput. |

What not to use:
- 3-bit or 2-bit everywhere: error compounds through the recurrence and the entropy collapse
  costs more speed than the smaller weights save.
- Mixed precision as an experiment only: ~3.5 bpw on prelude and coda MLPs, keeping W4 on the
  recurrent core and attention.
- 2:4 sparsity needs fine-tuning and only helps compute.

**On-disk format:** your own pre-packed blob. One file holds a header, tensors already in kernel
tile layout, and scales, plus a JSON manifest. Load it with mmap → `cudaMemcpy` into a single
arena. Not GGUF: no GGUF tooling knows about the loop or gate.

## 5. Speculative decoding: what actually helps here

ALoDLM's entropy commit already is draft-free speculation, without a verifier. A classic
separate draft model would mostly duplicate it. Ranked by expected value on Orin:

1. **Free self-verification.**
   - ALoDLM was trained with an auxiliary next-token (AR) loss on the clean stream, and the
     window attention is causal over the reordered rows.
   - Committed-but-uncached tokens are re-run as observed rows on the next outer step, and
     their readouts are computed anyway.
   - Apply lm_head to those rows as well (the extra rows are compute-free on Orin) and check
     `argmax_AR(row i) == committed token(i+1)`. On a mismatch, truncate and re-mask from there.
   - This turns the lossy entropy commit into AR-checked commits at **zero extra weight
     reads**. It lets you raise τ (more tokens per pass) while keeping quality.
   - *Unproven:* ALoDLM's AR-mode quality is never reported. Measure the AR accuracy of
     depth-d readouts in Phase 0.
2. **N-gram / prompt-lookup drafting injected as provisional observed tokens.**
   - Executor work (editing code, echoing file content, JSON tool arguments, repeating paths)
     copies a lot from the prompt.
   - Fill the window tail with n-gram continuations from the prompt and the generation so far
     as *provisional* committed tokens, and verify them with mechanism 1.
   - This is cheap, needs no extra memory, and the upside is large on edit-heavy traffic.
3. **Entropy commit tuning (τ, q, window size, position penalty), per workload.**
   - The window could grow from 16 to 32, since it is still memory-bound. Note the model was
     trained with 16-token blocks, so this needs validation.
   - Expose τ/q per request: in the request body, or as a "fast" / "careful" preset.
4. **Separate drafter, ALoDLM-1.7B drafting for 8B (same tokenizer). Lowest priority.**
   - Resident memory would be ~4.3 + 1.1 GB plus two depth-aware KV caches, which leaves no
     headroom.
   - The projected gain is about 45 → 55 tok/s, and only if acceptance is at least 5 tokens.
   - Revisit only if mechanisms 1 and 2 disappoint.

If Nemotron-Labs-Diffusion-3B wins Phase 0, its built-in linear self-speculation (diffusion
draft plus AR verify with shared KV, plus the optional LoRA drafter) is exactly mechanism 1,
already trained for it.

## 6. Engine architecture (pure CUDA C++, single stream)

```
server/   OpenAI /v1/chat/completions + /v1/completions, SSE streaming, Qwen3 chat template,
          <tool_call> JSON parsing, per-request tau/q, persistent prefix cache keyed by token-hash
engine/   arena allocator (one cudaMalloc), weight loader (mmap blob), depth-aware KV manager,
          outer-step scheduler, prefill (chunked), detokenizer (tokenizers C or own BPE)
kernels/  w4a16_skinny_gemm (M∈{16,32,48}, split-K for 8 SMs), fused QKV+qk-norm+RoPE+KV-int8-write,
          window_attention (M queries × paged KV with per-token depth indirection, split-KV),
          gate_up_silu_mul, down_proj+residual, rmsnorm(+recurrent-carry norm),
          lm_head_fused (tile over vocab, online max/sumexp/Σp·logit → argmax+entropy, never
          materialize M×151936 fp32 logits) + exit-gate dot + commit/stop decision kernel
graphs/   one CUDA graph per (window M bucket) per outer step, recurrent loop as a
          conditional WHILE node (CUDA ≥12.3) → zero host syncs inside an outer step
```

Design specifics:
- **KV with depth deduplication.**
  - Each cached token stores `d_t`, the number of depths that actually ran in the step where
    it was cached.
  - Only depths `1..d_t` are physically stored.
  - A query at depth j reads key depth `min(j, d_t)`. This matches the reference "inherit last
    completed depth" rule exactly.
  - With ~2 passes per step, this saves ~45% of the recurrent-layer KV. It also means prompt
    tokens (prefill = 4 depths) are the only full-depth entries.
- **Decision logic on the GPU.** Entropy, gate hazard, survival, commit mask, stop flag and
  re-masking all run on device. The host reads back one small pinned struct per outer step for
  streaming.
- **Orin specifics:**
  - Unified memory: use `cudaMalloc` for the hot arena and a host mmap for embeddings and the
    tokenizer.
  - Run in MAXN_SUPER with `jetson_clocks`.
  - Keep the CPU quiet during decode, because it shares the DRAM bandwidth.
  - Run headless.
- **Correctness oracle.** Port `alodlm/inference.py` and check bitwise-structurally, then within
  tolerance: same commits, depths and stop passes on greedy runs against BF16 on Thor, then
  against W4.

## 7. Phased plan

**Phase 0: measure before building (on Thor, which has 128 GB; plus Orin probes).**
- Orin prerequisites:
  - nvcc is not installed on this Orin: `sudo apt install cuda-nvcc-12-6 cuda-libraries-dev-12-6`
    (or `nvidia-jetpack`).
  - It is currently in 25 W mode: `sudo nvpmodel -m 2 && sudo jetson_clocks`.
  - 4.9 GB is available with the desktop up; going headless (`sudo systemctl set-default
    multi-user.target`) should free ~1.5 GB.
- Orin: STREAM-style bandwidth probe and an mma peak probe. These are the real roofline inputs.
- Thor: run the reference portable decoder in BF16 for ALoDLM-8B, ALoDLM-1.7B and
  Nemotron-Diff-3B on **your actual executor traffic**: ~100 captured prompts from your agent
  stack with tool schemas and real tool calls. Record:
  - task success
  - tool-call JSON validity
  - generated / recurrent_passes (tokens per pass)
  - passes per outer step
  - prompt length distribution, which drives the TTFT cost
- Thor: measure the AR next-token accuracy of ALoDLM observed-row readouts, which is the
  go/no-go for self-verification.
- **Gate:** pick the model. Proceed with 8B if its executor success clearly beats the 3B
  options and the median prompt is ≤ 2–3K tokens or cacheable.

**Phase 1: quantization pipeline (Thor).**
- Depth-aware GPTQ/AWQ calibration.
- Sweep variants: W4g128 vs g64, and head int8 vs int4.
- Re-tune τ/q.
- Report accuracy and tokens per pass against BF16.
- Emit the packed blob.

**Phase 2: kernels and microbenchmarks (Orin).**
- Build the W4A16 skinny GEMM and fused lm_head-entropy kernels first; together they are about
  95% of bytes.
- Target ≥ 85% of measured bandwidth at M = 16/32.

**Phase 3: single-pass forward plus the oracle.** Match a reference pass (logits, gate hazards,
K/V per depth) within tolerance.

**Phase 4: full adaptive decoder.**
- Outer steps, depth-dedup KV, device-side decisions, conditional-node graphs.
- Greedy parity with the reference on 50 prompts.

**Phase 5: server.**
- OpenAI endpoints, SSE, tool-call parsing, persistent prefix cache with disk spill.
- `bench_openai.py`, reused from the qwen38 repo.

**Phase 6: speed features.**
- Self-verification, then n-gram injection, then a window-32 trial, then W4A8 prefill, then
  CUDA-graph M buckets.
- Measure each on the Phase 0 traffic.

**Phase 7 (optional):** port to Nemotron-Labs-Diffusion-3B using the same kernels, with its
self-speculation mode. Head-to-head on executor success × tok/s × TTFT.

## 8. Risks

- **Executor competence.** ALoDLM has no instruction-following or tool-use evaluation. This is
  the biggest risk, and Phase 0 addresses it first.
- **TTFT.** The prefill multiplier makes long prompts slow, so the prefix cache and W4A8
  prefill matter.
- **Memory.** The 8B is tight: ~4.3 GB weights + ~0.4–0.7 GB KV at 4K + workspace, out of
  ~6.5 GB usable headless. The 16K context needs about 1.5 GB of KV even with int8 and dedup.
- **Quantization and parallelism.** Quantization can cut tokens per pass more than it saves
  bytes. Measure this; do not assume it.
- **License.** CC BY-NC is fine for personal use, but blocks any commercial use of the 8B and
  1.7B weights.

Sources: huggingface.co/amazon/ALoDLM-8B, huggingface.co/amazon/ALoDLM-1.7B,
arxiv.org/abs/2610.04198, github.com/amazon-science/ALoDLM, arxiv.org/abs/2607.05722,
huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B, huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B,
github.com/ai4s-research/awesome-diffusion-llm, recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it.
