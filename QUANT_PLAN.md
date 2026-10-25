# LLaDA2.2-mini on Orin Nano Super: compression decision (2026-10-24)

Model: inclusionAI/LLaDA2.2-mini (Apache 2.0).
- 16.3B total parameters, of which 15.30B are routed experts (19 MoE layers × 256 experts ×
  3×2048×512).
- 0.31B non-expert parameters: attention, 1 shared expert per layer, dense layer 0, routers.
- lm_head is 0.32B; the embedding table is 0.32B.
- Block routing caps each 32-token block at 48 of 256 experts per layer.

## 1. The memory budget sets the bit budget

Usable RAM is about 6.6 GB, assuming headless operation, orin-explorer stopped, and no Claude
Code session on the box. Fixed costs come to ~1.2 GB:

| item | size |
|---|---|
| CUDA context | 0.35 GB |
| non-expert weights at W4 | 0.17 GB |
| lm_head INT8 | 0.33 GB |
| workspace | 0.15 GB |
| server process | 0.2 GB |
| embeddings | mmap; only touched rows are resident |

Context length decides what is left for the experts:

| context | KV | expert budget | avg bpw over all experts | code bits per kept weight (0.19 bpw scale overhead) at 0% / 15% / 25% pruned |
|---|---|---|---|---|
| 8K INT8 KV | 0.17 GB | 5.23 GB | 2.74 | 2.55 / 3.03 / 3.46 |
| **16K INT8 KV** | 0.34 GB | 5.07 GB | 2.65 | 2.46 / **2.93** / 3.34 |
| 32K INT8 KV | 0.67 GB | 4.73 GB | 2.47 | 2.28 / 2.72 / 3.11 |
| 32K BF16 KV | 1.34 GB | 4.06 GB | 2.12 | 1.93 / 2.31 / 2.64 |

**Every 0.33 GB freed is worth about +0.17 bpw on the experts.** Cheap ways to free it:
- INT8 KV.
- A 4-bit lm_head, saving 0.16 GB. Only do this after confirming the threshold behaviour holds.
- Trimming the lm_head vocabulary: the 157K vocab is the Ling tokenizer, CJK-heavy. Keeping
  ~80K English/code tokens saves ~0.16 GB.
- Loading weights with O_DIRECT, so the file is not also held in the page cache.

## 2. Mixed precision vs pruning vs both: the evidence

- **ProQuant, arXiv 2610.30779 (Oct 2026).** This is the closest prior art: a fine-grained MoE
  (Qwen3-30B-A3B, 128 experts), GPTQ g128 asymmetric, attention at 4 bit. An ILP chooses each
  expert's bit-width from {0 = prune, 2, 3}, minimising routing-weighted output error. Then a
  KL fine-tune of the RMSNorm scales only, called RoCo.
  - Qwen3-30B-A3B average over GSM8K, MATH-500, GPQA-D, IFEval, MMLU-Redux and BFCL-v3:
    | variant | score |
    |---|---|
    | BF16 | 80.94 |
    | **2.5-bit** | **78.45** (−2.5) |
    | 2.0-bit | 74.49 |
    | uniform GPTQ 2.0-bit | 68.64 |
    | best mixed-precision-only baseline at 2.5 bits (MxMoE) | 62.41 |
  - Ablation: the candidate set {0,2,3} beat {1,2,3} and {0,1,2,3}. **Pruning a weak expert
    beats keeping it at 1 bit.**
  - Code: github.com/ThrowBug/ProQuant
- **REAP, ICLR 2026.** Router-weighted expert-activation pruning. It is near-lossless on code
  and tool calling at 50% pruning for *large* MoEs; across five models the mean coding drop was
  1.9%.
  - LLaDA2.2-mini is small, with 1.4B active and 512-wide experts, so it has less redundancy.
  - Pruning alone to fit 4-bit would need ~45–50% removed. That is too aggressive for
    general/agentic quality on a model this small.
- **Pure mixed precision** (MC-MoE, GEMQ, MxMoE, EAC-MoE): frequency or sensitivity-weighted
  2/3/4-bit. It is fine around 3 bpw. Uniform 3-bit nearly holds quality, while uniform 2-bit
  degrades badly (arXiv 2604.06515). In ProQuant's numbers, the mixed-only methods collapse
  around 2–2.5 bits.
- **Diffusion-LM specifics:**
  - arXiv 2508.14896 found massive activation outliers in the second FFN linear (down_proj) of
    LLaDA. Rotation-based methods survive where plain methods fail.
  - STaR-Quant (2606.04945) found that masked and unmasked tokens have different activation
    statistics within a denoising step, and that errors accumulate across steps. So calibrate
    in the decode regime, not on clean text.
  - CoDA (2604.20079) suggests dLLMs are, if anything, *more* robust than AR models at 2–4 bits.

**Decision: joint pruning plus mixed precision (ProQuant-style), not either one alone.**
- Target: **16K context, ~10–20% of experts pruned, survivors at mostly 3 bit, hot or sensitive
  experts at 4 bit, the weakest kept ones at 2 bit.** That is ~2.65 bpw averaged over all expert
  weights.
- This sits between ProQuant's 2.5 and 3.0 settings, so expect roughly **≤2 points** of loss
  against BF16. Rotation (§3) should claw some of that back.

## 3. The recipe (all on the Orin, with no Thor needed)

Install CUDA torch from `pypi.jetson-ai-lab.io/jp6/cu126`, which has cp310 wheels from 2.8 to
2.11. Use a separate venv: install `python3.10-venv`, or use `virtualenv`. Do not touch the CPU
torch 2.14 in `~/.local`.

1. **Rotation, offline and free at runtime.**
   - Fuse a randomized Hadamard rotation of the residual stream into the weights, QuaRot/SpinQuant
     style:
     - embedding and all residual-reading weights (q/k/v, router, gate/up, shared expert,
       lm_head);
     - all residual-writing weights (o_proj, every expert down_proj);
     - RMSNorm γ folded into the following weights first.
   - Routing stays exact up to FP error, because the router is rotated with the stream.
   - Add an **online FWHT-512 on each expert's down_proj input**. It is 512-dim, a power of
     two, and nearly free in-kernel. This is the outlier site the dLLM PTQ study flagged.
   - Optionally add an FWHT-128 per head on the o_proj input.
2. **Diffusion-regime calibration data.**
   - About 1–2K sequences, drawn from your executor traffic, tool-call transcripts, code edits
     and some general chat.
   - Build each as prompt + response, with the response's current block masked at ratios
     sampled from U(0,1). Use the model's own block-causal attention and block routing.
   - Record per-expert routed inputs, gate weights and pool-membership counts.
3. **Sequential layer-wise GPTQ with weight streaming.** BF16 is 32 GB and RAM is 8 GB, so for
   each layer:
   - mmap that layer's shards (~1.6 GB of BF16 MoE weights);
   - run the calibration activations through it on the GPU;
   - write the outputs to NVMe for the next layer.
   - Total I/O is about 32 GB read once, roughly 15 s at 2.8 GB/s. Compute dominates.
   - Per expert, accumulate a 2048² Hessian for gate/up and a 512² Hessian for down. Process
     experts in chunks, since 256 × 16 MB = 4 GB at once is too much.
4. **Per-expert cost table.** For each expert e and bit-width b ∈ {2, 3, 4}, record the GPTQ
   quantised, routing-weighted output error. For b = 0, use the routing-weighted output norm,
   i.e. the cost of dropping the expert. Normalise by activation count, following ProQuant.
5. **ILP.**
   - Minimise total cost subject to Σ bytes ≤ the budget in §1, with a per-layer prune cap
     ρ ≈ 0.25.
   - Keep at least 48 experts per layer. The block pool size is C = 48, so if pruning goes deeper
     the pool capacity must be lowered to the number of survivors.
   - Pruned experts are removed by masking their router logits to −inf. No retraining is needed.
6. **RoCo.** KL fine-tune of the RMSNorm γ only, against BF16 teacher logits cached during
   step 3. It is small enough to run on Orin with the quantised model resident.
7. **Gates before any kernel tuning** (run with the CUDA engine once Phase 3 exists):
   - success on your executor-prompt set;
   - BFCL-v3 subset and IFEval;
   - **tokens per forward (TPF)**;
   - pass criteria: ≤2–3 points below the BF16 reference numbers, and TPF ≥ 90% of BF16.
   - A BF16 reference for TPF/quality on Orin is possible but slow: stream the experts from an
     mmap. Block routing keeps each forward to ~5.7 GB of experts. Use a 10–20-prompt subset.

## 4. Packing format: plain bit-sliced scalar codes, no trellis or vector codebooks

| option | quality at ~2.5–3 bpw | Orin decode cost per weight | verdict |
|---|---|---|---|
| **GPTQ scalar INT2/3/4 + Hadamard, group 128, asymmetric, FP16 scale + u8 zero** | good, with rotation closing much of the gap | ~1–1.5 ops using lop3 / magic-number FP16 conversion | **use this** |
| non-uniform 3-bit LUT per group (NF3/any3-style, register LUT via `__byte_perm`) | slightly better than uniform | ~2–3 ops | optional v2 upgrade |
| QTIP / EXL3 trellis | best on big models; lost most of its MATH score at 2-bit on Llama-3.2-1B/3B | ~6–8 ops | **no.** Each forward streams ~3.5 G weights, so trellis decode is ~20–28 G ops. That is 10–28 ms of ALU time on 1024 CUDA cores, more than the 16 ms memory time. |
| AQLM / codebook VQ | good | smem codebook lookups, poor on 8 SMs | no |

Layout:
- **Bit-sliced planes:** 4-bit = one nibble plane; 3-bit = a 2-bit plane + a 1-bit plane;
  2-bit = a 2-bit plane. All loads stay 128-bit aligned.
- **Expert-major, tile-major contiguous slabs.** Each CTA streams one slab. Measured: ~100 GB/s
  this way vs 75–82 GB/s with interleaved access.
- Every expert carries a bit-width tag. The grouped-GEMM kernel is templated per bit-width, and
  the work queue is sorted by (bit-width, expert).

## 5. A finding that changes the kernel plan: Orin is *not* purely memory-bound here

One 32-token block forward does about:
- 31 GFLOP of expert work: 32 tokens × 8 experts, spread over ≤48 experts, so ~5 rows per
  expert;
- 20 GFLOP of dense work;
- up to 21 GFLOP for lm_head on the masked rows.

With naive m16 tensor-core tiles, the experts pad 5 → 16 rows. That gives about **133 GFLOP per
forward, ~13 ms at a realistic 60% of the 16.7 TFLOPS FP16 peak.** That is as long as the 16 ms
memory time. Mitigations:
- **Swap-AB mma for experts.** Put the weight rows on the M=16 side and the tokens on N=8, so
  padding drops 5 → 8 rows and expert FLOPs fall to ~46 GFLOP.
- Run lm_head only over the still-masked rows; there are fewer every step. Swap-AB applies here
  too.
- Put the dequant (~1.5 ops/weight × 3.5 G ≈ 5 G ops) in the same warp as the loads, so it
  overlaps the memory stream.
- Revised forward estimate: about 17–20 ms, giving 50–60 forwards/s and **~100–165 tok/s at
  TPF 2–3**. TPF on your traffic is still the big unknown.

## 6. Later option: lazy expert residency instead of hard pruning

- Pruned experts are written to NVMe. If a block's pool picks one, finish the current forward
  without it (exactly as if it were pruned) and async-load it at 2.8 GB/s: ~1 MB takes
  ~0.4 ms.
- It then serves the remaining ~5–15 forwards of that block. Routing within a block is stable,
  because the block is denoised in place.
- The result: quality approaches the unpruned model, with no stalls. Evaluate this only after the
  baseline works.

Sources: ProQuant arxiv.org/abs/2610.30779; REAP arxiv.org/abs/2510.13999 and
github.com/CerebrasResearch/reap; dLLM PTQ study arxiv.org/abs/2508.14896; STaR-Quant
arxiv.org/abs/2606.04945; CoDA quantization arxiv.org/abs/2604.20079; MoE quantization
generalization arxiv.org/abs/2604.06515; EAC-MoE arxiv.org/abs/2508.01625; GEMQ
arxiv.org/abs/2605.23078; MODE arxiv.org/abs/2606.17118; QTIP arxiv.org/abs/2406.11235;
LLaDA2.2 tech report (github.com/inclusionAI/LLaDA2.X).

## 7. KV cache sizing (LLaDA2.2-mini: 20 layers x 4 KV heads x 128 x K,V = 20,480 values/token)

| format | per token | notes |
|---|---|---|
| BF16 | 40 KB | |
| INT8 + scales | 20.6 KB | default |
| INT4 (K per-channel, V per-token, KIVI-style) + scales | 10.6 KB | needs a quality check on the agent set |

Max context = (6.6 GB usable − 1.2 GB fixed − experts) / per-token:

| experts avg bpw | KV room | BF16 | INT8 | INT4 |
|---|---|---|---|---|
| 2.40 | 0.81 GB | 19K | 37K | 73K |
| 2.50 | 0.62 GB | 15K | 29K | 56K |
| 2.65 | 0.33 GB | 8K | 15K | 30K |

The diffusion block reads the whole prefix KV on every forward, so long context also costs speed:
- at 32K: INT8 runs at ~68% of short-context speed, INT4 at ~80%;
- at 64K: INT8 ~51%, INT4 ~67%.

**Every 16K of INT8 context costs 0.33 GB, which is 0.17 bpw of experts.** Choose 16K INT8 or 32K INT4 as the
serving point, and expose it as a launch flag. The expert byte budget is solved after that choice.

## 8. Pipeline status (quant/)

- s02_corpus.py: done.
  - 541 rows, 1.20M tokens. Mix: agentic 0.26, code 0.24, general 0.18, math 0.10, science 0.06,
    longctx 0.10, multilingual 0.06.
  - 78.5% of tokens are assistant spans eligible for masking.
  - Sources, provenance and licence rules are vendored from deepseek-v4.1-flash-reap-thor; xlam is
    excluded as an unverified generator.
- s03_teacher_pass.py: BF16 layer-streaming teacher, in progress.
  - Each row runs as clean + noisy streams. Outputs: REAP/HOPE/Super-Expert banks per layer, split by
    stream x bucket, plus held-out top-32 teacher targets.
  - Smoke test: masked-token accuracy 0.54 on code and 0.37 on prose (random mask ratios), peak GPU 5.0 GiB.
- Next stages:
  - s04: keep-set candidates (HOPE/REAP/freq/random) at 10/15/20/25% pruning.
  - s05: GPTQ per expert at 2/3/4 bits on Hadamard-rotated weights, giving a cost table, then an ILP over
    {0,2,3,4} under the byte budget.
  - s06: replay the arms against the teacher's top-32 on held-out rows (top-1 and kl32).
  - Then pack for the CUDA engine.

## 9. Measured results (2026-10-24)

### s03 teacher: masked-token accuracy at random mask ratios
| bucket | accuracy |
|---|---|
| code | 0.784 |
| math | 0.708 |
| agentic | 0.622 |
| longctx | 0.458 |
| science | 0.435 |
| general | 0.378 |
| multilingual | 0.333 |

Structured output is the most predictable, which is where parallel decoding commits the most tokens per pass.
The residual stream peaks at |x| ≈ 38k, under fp16's 65,504, so the fp16 hidden-state memmaps are safe.

### s05 per-layer distortion
Distortion is the expert quantization error as a fraction of the cost of pruning every expert in the layer.
- Layer 1: 2b 5.7%, 3b 1.05%, 4b 0.13%.
- Layers 2–18: mostly 2b 11–18%, 3b 2.5–4%, 4b 0.6–1.1%.
- Exceptions: layers 5, 7 and 8 sit near 2b 5–8%. Layer 19 is 2b 7.5%.
- Every extra bit cuts distortion 4–8×.

### s05 allocation sweep (proxy objective; lower is better)
| point | objective | expert bpw | pruned |
|---|---|---|---|
| 32K int4 | 0.0153 | 2.64 | 10.5% |
| 16K int8 | 0.0152 | 2.65 | 10.5% |
| 64K int4 | 0.0194 | 2.45 | 12.8% |
| 64K int8 | 0.0316 | 2.10 | 17.5% |
| uniform 3-bit | 0.0303 | 3.0 | 0% — 6.10 GB, does not fit |

### s06 run 1: held-out replay vs the in-run teacher
Readout was in bf16 (see the caveat below).

| arm | top-1 | kl32 | truth |
|---|---|---|---|
| teacher | 0.950* | 0 | 0.602 |
| rotid | 0.924 | 0.0068 | 0.602 |
| u4 | 0.901 | 0.0213 | 0.600 |
| prune10 (REAP, BF16) | 0.870 | 0.0538 | 0.598 |
| 16K int8 alloc | 0.813 | 0.1334 | 0.588 |
| 32K int4 alloc | 0.813 | 0.1341 | 0.588 |
| 64K int4 alloc | 0.793 | 0.1598 | 0.586 |
| 64K int8 alloc | 0.764 | 0.2371 | 0.576 |

\* bf16 logit ties. The readout is fp32 from run 2 on.

- 16K-int8 and 32K-int4 are indistinguishable (McNemar z = 0.36). **int4 KV doubles the context at no expert cost.**
- **The proxy under-weights pruning by ~3.5×.** Measured dKL / objective is 9.2 for pruning and 2.6 for 4-bit.
- **The proxy is also super-additive in 2-bit.** Even with separate ratios it predicts 0.054 for the 32K allocation;
  0.127 was measured.
- rotid's gap (the numerics of the fp32 rotated path against bf16) is the baseline for every quantized arm.
- For scale: deepseek-v4.1 REAP-192 shipped at kl32 0.298 against its teacher.

### s06 run 2 (running): calibrating the allocator weights at 32K int4
Arms: teacher, rotid, u4, u3, v1 allocation, no-prune, prune×3.5, prune×3.5 + 2-bit×3. Readout in fp32.

### s06 run 2: fp32 readout, 32K int4 budget (5.05 GB experts)
| arm | top-1 | kl32 [95% CI] | truth | agentic | code | multilingual |
|---|---|---|---|---|---|---|
| teacher | 1.000 | 0 | 0.6022 | 1 | 1 | 1 |
| rotid (floor) | 0.959 | 0.0062 | 0.6023 | 0.953 | 0.976 | 0.935 |
| uniform 4-bit (8.0 GB, does not fit) | 0.922 | 0.0207 | 0.5996 | 0.927 | 0.953 | 0.845 |
| uniform 3-bit (6.1 GB, does not fit) | 0.877 | 0.0534 | 0.5957 | 0.888 | 0.921 | 0.780 |
| **32K int4, no pruning** (61% 2b / 34% 3b / 6% 4b) | **0.826** | **0.112** [0.090, 0.137] | **0.5903** | 0.839 | 0.897 | 0.662 |
| 32K, prune x3.5 | 0.824 | 0.121 | 0.5884 | 0.844 | 0.895 | 0.639 |
| 32K, prune x3.5 + 2-bit x3 | 0.823 | 0.121 | 0.5891 | 0.839 | 0.896 | 0.611 |
| 32K, v1 allocation (prune x1) | 0.818 | 0.134 | 0.5878 | 0.824 | 0.894 | 0.581 |

**Decision (pending owner): 32K int4 KV, no-pruning allocation** (`s05/alloc/32768_int4_noprune.npy`).
- Re-weighting the allocator barely moves the result, because the byte budget binds.
- No pruning is best on every overall metric and the simplest to serve: all experts stay routable and the
  router is unchanged.

**Next levers, cheapest first:**
1. Measure the real fixed overhead on the Orin; the assumption is 6.6 GB usable and 1.2 GB fixed.
2. 2-bit at group 64.
3. RoCo-style norm KD.
4. NVMe-backed cold experts at higher bits (engine feature).

**Owner decision (2026-10-24):** go with 32K, no pruning, and evaluate the int4 KV cache.

## 10. KV cache format (s06kv, held-out, 54 rows, vs in-run BF16 teacher)

| arm | top-1 | kl32 [95% CI] | truth |
|---|---|---|---|
| int8 per token | 0.940 | 0.0185 [0.013, 0.027] | 0.6029 |
| K8 / V4, 128-pt rotation | 0.935 | 0.0172 [0.013, 0.023] | 0.6022 |
| **K4 per channel over each 32-token block / V4 per token ("4c/4")** | **0.918** | **0.0275** [0.021, 0.034] | **0.5996** |
| 4c/4 + rotation | 0.921 | 0.0250 [0.020, 0.031] | 0.6000 |
| int4 per token + rotation | 0.862 | 0.0813 | 0.5945 |
| int4 per token, no rotation (smoke) | 0.44 | 1.31 | — |
| experts (32K no-prune) alone | 0.826 | 0.112 | 0.5903 |
| **experts + 4c/4** | **0.821** | **0.124** [0.100, 0.151] | **0.5880** |

- Plain per-token int4 keys fail because the keys carry per-channel outliers.
- Per-channel keys over the committed 32-token block (KIVI-style) fix it. That group is exactly the diffusion
  block, quantized once at commit.
- The cost is 11.8 KB/token (vs 10.9 for plain int4), and KV adds only KL +0.011 on top of the experts.
- The rotation is not needed for 4c/4.
- **Chosen: 4c/4.** Implemented in `engine/attn.cu` (`kv_commit`, `attn_partial`).

## 11. Non-expert precision and A8 (s05n + s06 run 3, in progress)
s05n GPTQ errors (relative output energy on held-out tokens):
- 4-bit: 1e-3 to 8e-3, with the attention dense projection the worst.
- 8-bit: ~1e-5.
- lm_head: 1.4e-3 at 4-bit, 5e-6 at 8-bit.

On the smoke row:
- 8-bit sits at the noise floor (KL 0.009).
- 4-bit costs KL 0.054.
- 4-bit lm_head adds 0.004.
- A8 expert activations add about 0.001 per layer.

Run 3 (`s06ne`, held-out, vs in-run teacher, all on the 32K alloc + 4c/4):

| arm | top-1 | kl32 | truth | longctx top-1 |
|---|---|---|---|---|
| experts + 4c/4 (BF16 non-experts) | 0.818 | 0.1225 | 0.5885 | 0.748 |
| **+ 8-bit non-experts, 8-bit lm_head** | 0.820 | 0.1226 | 0.5885 | 0.751 |
| + 4-bit non-experts | 0.812 | 0.1328 | 0.5858 | 0.748 |
| + 8-bit non-experts, 4-bit lm_head | 0.791 | 0.1264 | 0.5882 | **0.583** |
| **+ 8-bit non-experts + A8 expert activations** | 0.820 | 0.1212 | 0.5897 | 0.751 |

**Chosen (as packed): 8-bit non-experts, 8-bit lm_head, A8.** 8-bit is free; 4-bit non-experts cost +0.010; a
4-bit lm_head breaks long context.

Engine reality check (engine/ENGINE_DESIGN.md): on real prompts the simulated quantized model is ~28% rel err
from BF16 at the last layer and its first-step argmax agrees with the teacher on ~60% of masked positions.
Outputs are coherent (correct tool calls, code, math) with occasional repetition artifacts ("and and", one
"the the the" run in a long-context summary). The measured engine footprint leaves ~0.6-0.9 GB unused at
4-16K context: the plan assumed 1.2 GB fixed + 6.6 GB usable; spending ~0.5 GB more on experts (~+0.25 bpw)
is the obvious quality lever.
