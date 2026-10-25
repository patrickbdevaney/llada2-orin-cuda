# LLaDA2.2-mini Orin engine: design notes (v1)

Status 2026-10-25 (late): 32K context, 4c/4 KV, experts as allocated (owner: context over bpw). OpenAI server
verified at --max-ctx 32768 (5.71 GB weights + 0.39 GB KV, ~0.6 GB still free with a Claude Code session on the box).
Clocks pinned (sudo jetson_clocks); tools/jetson-clocks.service runs it at boot once installed:
`sudo cp tools/jetson-clocks.service /etc/systemd/system/ && sudo systemctl enable --now jetson-clocks.service`.

Step cost (tests/bench_step: the captured step graph replayed, best of 5): 24.7 ms at 128 tokens of context,
25.3 ms at 1K, 26.9 ms at 3.2K, 30.6 ms at 8K, 34.4 ms at 16K (this morning: 26.4 / 27.1 / 28.8 at 128 / 1K / 3.2K;
~29 ms/step before that).
tests/eval_knobs.py --n-math 96 (116 scored tasks): server default (fast: threshold 0.3, steps 16) 113-116/116 at
180-183 tok/s; model card (threshold 0.5, steps 32, `--reference`) 115-116/116 at 151 tok/s. Every miss so far has
correct arithmetic and a doubled final digit ("Answer: 66" for 6) or no Answer line: the quantized model's
repetition artifact, on both presets (more often at 0.3); which tasks hit it moves with any numeric change.
Prefill: batched, 16 blocks (512 tokens) per pass: 2.5K tok/s.

Decode-speed work of 2026-10-25 (late), each checked against tests/test_layers, test_moe_i8 / test_dense_i8 /
test_attn and eval_knobs:
- Device strip layout (moe8::relayout_strips at load, files unchanged): per strip [all tile codes][all tile metas],
  metas lane-ordered (u32 scale pair, u16 zero pair). The interleaved 48 B meta after every 512 B of codes cost ~20%
  of DRAM throughput (pattern microbenchmark: 84 -> 103 GB/s). Bit-exact.
- GEMM body: per-lane load pointers with compile-time offsets inside a ring pass (one advance per pass, one select
  for the last-pass wrap): ~20-30% of the loop's instructions were 64-bit address arithmetic. 2-bit gate_up 75 -> 80
  GB/s; the 32-token variant no longer spills. Activation scales/sums [kt][token] (one 64-bit LDS, no bank
  conflict); I2F for the conversions in the expert GEMMs (NT <= 2), the magic-number path in the 32-token ones.
- Routing: router on the tensor cores (bf16 hi + lo split of the fp32 input, 2 MMAs per tile: fp32-grade, checked
  against fp64 by tests/test_router), K split 8 ways, partials summed by route_core in fixed order: 41 -> 19 us/layer.
  Rank loops with 4 independent accumulators.
- One work item per routed expert (<= 8 / <= 16 / <= 32 tokens -> NT1 / NT2 / NT4): an expert's weights stream once.
  The shared expert runs on its own stream (st4, low priority) from the post-attention norm, overlapping routing.
- Readout: the screen GEMM writes fp16 values (readout check unchanged vs fp32), 4 strip groups: screen 2.88 -> 2.58
  ms, candidate pass 0.37 -> 0.26 ms.
- Attention: lazy softmax rescale (reference max moves only when a block's max exceeds it by > 8 in log2 units;
  probabilities exponentiated in fp32, then fp16) -- O is rescaled only when a row moved. More accurate than the
  fp16 exp it replaced (test_attn 6.1 -> 5.9e-4 at 100 blocks); 0.4-0.6 ms/step at 3-16K context.
- MMA accumulators start at (int->float bias - z * Sxq): no integer work left after the expert/dense MMAs.
- Fusions: attention combine writes the dense layer's int8 input directly (quant_rows_q8 gone); the MoE combine runs
  the next layer's input norm (moe_combine_norm, Engine::in_ready); attention combine restructured (warp per row).
- Tuned strip groups of the expert launches against real routed work lists (tests/bench_stage).

Tried and rejected (measured): cp.async shared-memory weight ring for the expert GEMMs (slower than the register
ring, with either layout); deeper register rings / 4 CTAs per SM / 4 strips per warp (spills); dedicated 8-bit
32-token kernel, 16-warp CTAs, K-split (no gain); persistent CTAs with an atomic task queue (slower than the
hardware scheduler); fusing fwht512 into the expert down GEMM (duplicated per CTA: slower); every expert as 8-token items, hot experts as
16-token halves (slower in the pipeline though faster in isolation); per-row int8 lm_head (KL 5x the
GPTQ g128 head vs BF16); 2-pass 16-token screen; 3-stream priority tricks. Note: isolated repeats of a < 5 MB
matrix run partly from L2 -- judge small-kernel changes with tests/bench_step, not in isolation.

Where the time goes (PROF=1, ~26 ms/pass): routed+shared expert GEMMs ~14 ms (2/3-bit at 70-80 GB/s against a
~95 GB/s ceiling for this access pattern; instruction-bound: unpack + per-group correction ~85-120 instr per tile),
qkv + dense 8-bit ~4.6 ms (~65 GB/s), readout ~2.9 ms, attention ~1 ms per 1K tokens of context, routing ~1 ms.
Option not taken (owner's call, costs quality): 4-bit non-experts (s06ne: +0.010 KL) would save ~2.4 ms/pass.

Clocks matter: these were measured under the default dynamic GPU governor, which idles at 306 MHz and does not
fully ramp for short bursts (tensor rate 5.7 vs 8.1 TMAC/s pinned). Run `sudo jetson_clocks` for consistent
numbers; sustained decode reaches 1020 MHz either way.

What changed (2026-10-25), each verified against tests/test_layers (BF16 and simulated-quant cases):
- Batched prefill (Engine::prefill): layer by layer, all blocks' K/V committed first, then block j attends cache
  blocks < ctx0 + j and itself (= sequential semantics). Routing per 32-token block, slots merged per expert
  (ops::route_batch_*), NT4 work items, GEMM grid "swap" (items fastest) for L2 reuse.
- Attention: one barrier per block (dequantize block b+1 while computing b), half2 dequant, half2 exp2, 4 warps
  x 32 query rows (each K/V fragment feeds two mma), ~4 chunks per head with the fresh block folded into the last;
  batch dimension for prefill. ~1.07 us per cached block per layer (fp16 mma floor ~0.5): attention costs
  ~0.4 ms/pass per 1K tokens of context. K8 per token + int8 Q was simulated and rejected (no better than 4c/4,
  +33% KV).
- CUDA graph per denoising step; between blocks only attention/position nodes are patched
  (cudaGraphExecKernelNodeSetParams). LLADA_NOGRAPH=1 disables.
- lm_head screening (quant/s07d_lm_split.py -> lm_head_split.bin): stream the high-nibble plane (half the bytes),
  keep each row's top-8 + softmax sum, rescore the 8 exactly from both planes. vs the exact 8-bit readout:
  top-1 99.92-99.97%, top-2 99.6-99.7%, |dp1| mean 3e-3. LLADA_NO_LM_SPLIT=1 uses the 8-bit head;
  LLADA_LM_CHECK=1 loads both and reports agreement (needs +0.33 GB).
- MoE: NT1/NT2 launches run concurrently on two streams; int->float in the epilogue without I2F.
- Router, schedule (rank-based top-8, popc slot placement), RoPE (warp per head, sincos table), FWHT512+A8,
  combine and norm+rotation (warp FWHT + one transpose) rewritten: ~2.5 ms/pass saved together.
- Decode knobs exposed: llada_generate(..., steps, ...), server request fields threshold / steps /
  editing_threshold / max_post_steps, server flags --reference (model card 0.5/32; default is fast 0.3/16) / --threshold / --steps.

Per-pass profile (code prompt, PROF=1, ~28.7 ms pinned clocks): MoE gate_up 9.7, down 5.2, lm_head screen 2.8,
readout 0.5-0.7, qkv 2.5, dense 2.1, router 1.0, attention 1.0 at ~600 tokens, the rest < 1 ms each.
Bandwidth floor at 82 GB/s is ~16.5 ms; the expert GEMMs (2/3-bit, ~70 GB/s) are instruction-bound.

Earlier status (2026-10-24): RoPE-position race in prefill fixed (pinned staging buffer reused before the async
copy ran; every prompt with >= 3 full blocks was prefilled at wrong positions).

Verification tools: tests/test_layers <model> <case> [self|batch] against tests/gen_layer_case.py --prompt
(BF16) or quant/student_case.py (s06-simulated quantized model); quant/teacher_probe.py (streamed BF16);
tests/bench_kern (per-kernel GB/s); PROF=1 / PREFILL_BENCH=1 bench_gen; LLADA_TRACE=1 (per-step trace);
LLADA_SYNC=1 (sync + check after every prefill launch); tests/eval_knobs.py.

## Data flow per layer (one 32-token diffusion block, one denoising step)

```
h ──rmsnorm(γ_in)──► qkv (W4/W8, x'=n·Q) ─► q/k-norm ─► partial RoPE ─► attention over [KV cache | fresh block]
  ◄──────────── + dense(a', W' = W_o·blockdiag(R)) ◄──────────────┘          (V folded with R: a' = a·R per head)
h ──rmsnorm──► router (bf16 W, fp32 math): sigmoid + bias, block max → pool 48, token top-8 in pool, ×2.5 normalised
          └─► n·Q (FWHT2048 + signs) ─► moe_gate_up (2/3/4-bit tiles) ─► fwht512 ─► moe_down ─► moe_combine
                                       └► shared expert (same rotated path)
acc (Q basis) ─► ·Qᵀ (FWHT2048 + signs) ─► + h
```

## Weight formats
- Routed experts: `quant/packlib.py` tiles (16 rows × 128 cols, A-fragment order, fp16 scale + u8 zero per
  row-group), gate_up rows interleaved 8 gate / 8 up per strip. `quant/s07_pack_experts.py` →
  `~/models/llada2mini-orin/experts_L*.bin` + `experts.json`.
- Non-experts (attention, shared, layer-0 MLP, lm_head): `quant/s05n_nonexpert.py` GPTQ at 4 and 8 bits in the
  rotated bases; precision chosen by s06 `+ne:` arms. The same tile kernel serves them (a dense matrix is one
  "expert" with all tokens).
- Embeddings (322M params, 0.64 GB BF16): NOT resident. mmap'd file, 32 rows gathered per step on the host.
  The s05 memory model never counted them, so they must stay off the heap.

## KV cache
The cache format is 4c/4 (QUANT_PLAN §10): K is 4-bit per channel over each committed 32-token block, V is
4-bit per token, at 11.8 KB/token. On held-out data it costs KL +0.011 on top of the experts.
A block's K/V stay BF16 while it is being denoised and are quantized once when the block commits. That is
exactly the 32-token per-channel group of "4c".

## Step budget (bandwidth, 100 GB/s contiguous)
experts 48 × 19 × ~1.0 MB ≈ 0.95 GB · attention+shared W4 0.17 GB (W8 0.31) · lm_head INT8 0.33 GB ·
KV 11.8 KB/token (4c/4) → 0.39 GB at 32K. ≈ 1.5 GB/step at 4K context ≈ 15 ms → ~65 steps/s. At 2-4 tokens per
step (threshold decoding) that is 130-260 tok/s before overlap. lm_head is a third of the expert stream, so it is
the next lever: logits only for still-masked positions, or 4-bit.
