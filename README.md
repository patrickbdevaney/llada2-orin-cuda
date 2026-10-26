# llada2-orin-cuda

A pure-CUDA inference engine and OpenAI-compatible server for
[LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini), a 16B-parameter mixture-of-experts diffusion LLM
with 1.4B active parameters. It targets the 8 GB Jetson Orin Nano Super. The repository holds the engine, the server,
and the compression pipeline that fit the model into 5.7 GB. There is no PyTorch at runtime: hand-written kernels, a
C++ block decoder and a small Python HTTP front end.

## Status (2026-10-25)

Research release from a single maintainer. It is usable for local evaluation and agent-executor experiments on an
Orin Nano.

The served model is a heavily quantized student, not a lossless copy. Its top-1 agreement with the BF16 model is
0.82 on held-out text. It shows a known repetition artifact, for example a doubled final digit or "and and"
(see [Limitations](#limitations)).

| Artifact | State |
|---|---|
| Weights: [`patrickbdevaney/LLaDA2.2-mini-orin-nano`](https://huggingface.co/patrickbdevaney/LLaDA2.2-mini-orin-nano) | Public, 6.7 GB download |
| Engine ([`engine/`](engine/)) and server ([`server/`](server/)) | Working at 32K context; verified against BF16 and simulated-quantization layer tests |
| Compression pipeline ([`quant/`](quant/)) | Complete for the shipped configuration; scripts assume the author's paths |

## Results

**Speed.** Measured on a Jetson Orin Nano Super with pinned clocks (`sudo jetson_clocks`).

| Metric | Value |
|---|---|
| One denoising step (32-token block) | 24.7 ms at 128 tokens of context, 25.3 ms at 1K, 30.6 ms at 8K, 34.4 ms at 16K |
| Decode, server default (threshold 0.3, 16 steps) | ~180 tok/s on math, code and tool-call tasks |
| Decode, model-card settings (`--reference`: threshold 0.5, 32 steps) | ~150 tok/s on the same tasks; prose 40–80 tok/s |
| Prefill | ~2.9K tok/s (16 blocks of 32 tokens per pass) |
| Context | 32K tokens (int4 KV cache, 11.8 KB/token) |

**Task check.** [`engine/tests/eval_knobs.py`](engine/tests/eval_knobs.py) runs 96 generated math problems, 12 code
tasks executed against tests, and 8 tool calls. It scores 113–116 / 116 at either decode setting. Every miss so far
has correct arithmetic but a repetition artifact in the final answer.

**Quality vs the BF16 teacher.** Held-out rows, teacher-forced in the diffusion regime. Top-1 is agreement with the
teacher's argmax. kl32 is KL(teacher ‖ model) over the teacher's top 32 tokens. Brackets are 95% bootstrap
intervals. The source for each row is [`QUANT_PLAN.md`](QUANT_PLAN.md) §9–11.

| Configuration | Expert bytes | top-1 | kl32 |
|---|---|---|---|
| BF16 teacher | 30.6 GB | 1.000 | 0 |
| Rotated fp32 path, unquantized (numerical floor) | — | 0.959 | 0.006 |
| Uniform 4-bit experts | 8.0 GB (does not fit) | 0.922 | 0.021 |
| Uniform 3-bit experts | 6.1 GB (does not fit) | 0.877 | 0.053 |
| Mixed 2/3/4-bit experts, no pruning | 5.05 GB | 0.826 | 0.112 [0.090, 0.137] |
| **Shipped:** mixed experts + int4 KV + 8-bit non-experts + int8 activations | 5.05 GB | **0.820** | **0.121** |

**Pruning finding.** REAP/HOPE expert pruning was built, measured and rejected for this model.
- Under the fixed 8 GB byte budget, every allocation that pruned experts to pay for more bits on the survivors scored
  below the no-pruning allocation: 0.818–0.824 against 0.826 top-1.
- The allocator's proxy objective under-weighted pruning damage by about 3.5× relative to measured KL.
- Pruning 10% alone at BF16 already costs 0.054 kl32. At 512-wide experts and 1.4B active parameters, this model has
  less expert redundancy than the large MoEs where REAP is near-lossless.
- The decision record is in [`QUANT_PLAN.md`](QUANT_PLAN.md) §2 and §9.

## Method

1. **Bit budget from memory.** About 6.6 GB is usable and about 1.2 GB is fixed, which leaves 5.05 GB for 15.3B
   routed-expert parameters at 32K context (2.64 bits per weight on average). Every 0.33 GB saved elsewhere buys
   about +0.17 bits per expert weight ([`QUANT_PLAN.md`](QUANT_PLAN.md) §1).
2. **Rotation.** An offline randomized Hadamard rotation of the residual stream is fused into the weights. An online
   FWHT-512 runs on each expert's down-projection input, the outlier site reported for LLaDA.
3. **Diffusion-regime calibration.** 541 rows (1.20M tokens) are calibrated as they are decoded: the response's
   current block is masked at random ratios. The mix is agentic, code, general, math, science, long-context and
   multilingual.
4. **Per-expert allocation.** GPTQ per expert at 2, 3 and 4 bits gives a cost table. An ILP over {0 = prune, 2, 3, 4}
   then allocates bits under the byte budget, following ProQuant (arXiv 2610.30779). The shipped allocation is 61%
   2-bit, 34% 3-bit and 6% 4-bit, with no experts pruned.
5. **KV cache "4c/4".** Keys are 4-bit per channel over each committed 32-token block; values are 4-bit per token.
   This adds +0.011 KL on top of the experts. Plain per-token int4 keys fail (top-1 0.44), because keys carry
   per-channel outliers ([`QUANT_PLAN.md`](QUANT_PLAN.md) §10).
6. **Non-experts at 8 bits.** This is effectively free (+0.0001 KL). A 4-bit lm_head was rejected because it breaks
   long context (long-context top-1 0.583 vs 0.751).

**Engine** ([`engine/ENGINE_DESIGN.md`](engine/ENGINE_DESIGN.md)):
- **Decoding.** LLaDA2.2 joint threshold/insert-delete block decoding, with a CUDA graph per denoising step.
- **Expert GEMMs.** W2/3/4/8 × A8 int8 tensor-core GEMMs over packed tiles, run as one work item per routed expert so
  each expert's weights stream once.
- **Router.** A tensor-core router with fp32-grade accuracy.
- **lm_head readout.** A screen over the high-nibble plane keeps each row's top-8 candidates, then rescores them
  exactly. It matches the exact 8-bit readout on 99.92–99.97% of top-1 tokens.
- **Where the time goes.** The routed and shared expert GEMMs take about 14 of 26 ms per step. They run at
  70–80 GB/s, are instruction-bound on unpacking, and the access-pattern ceiling is about 95 GB/s. The design notes
  list each optimization with its measured effect, plus the alternatives that were tried and rejected.

## Getting started

Requires a Jetson Orin Nano Super 8GB (JetPack 6, CUDA 12.6 `nvcc` for `sm_87`) and Python 3.10.

```bash
cd engine && make                    # bench_gen, libllada.so, tests/test_layers (set NVCC=... if needed)
pip install tokenizers jinja2 huggingface_hub
huggingface-cli download patrickbdevaney/LLaDA2.2-mini-orin-nano --local-dir ~/models/llada2mini-orin
cd ../server && python3 server.py --model ~/models/llada2mini-orin --engine ~/models/llada2mini-orin
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python function that reverses a string."}]}'
```

- **Request fields:** `threshold`, `steps`, `editing_threshold` and `max_post_steps` override the decode defaults for
  one request. `--reference` switches the server to the model card's settings.
- **API:** tool calls and streaming are supported.
- **Concurrency:** requests are served one at a time.
- **Memory:** weights take about 5.7 GB of the 7.4 GiB shared memory, so run nothing else GPU-heavy alongside.
- **Clocks:** pin them for consistent speed. [`tools/jetson-clocks.service`](tools/jetson-clocks.service) does it at
  boot.

## Compatibility

| Platform | State |
|---|---|
| Jetson Orin Nano Super 8GB, JetPack 6, CUDA 12.6 | Tested |
| Other Jetson Orin modules (`sm_87`) with ≥ 8 GB | Expected to work; not tested |
| Other GPUs and architectures | Not supported: the kernels are tuned for `sm_87` and Orin's memory system |

## Limitations

- **Repetition artifact.** The quantized model sometimes repeats a token: a doubled final digit ("Answer: 66" for 6),
  "and and", and once a "the the the" run in a long summary. It happens more often at the fast 0.3 threshold. Which
  tasks it hits changes with any numeric change.
- **Fidelity.** Top-1 agreement with BF16 is 0.82 overall. Multilingual drops furthest (0.66 vs 0.90 on code).
  Quality numbers are teacher agreement plus a 116-task check, not standard benchmarks.
- **Memory headroom.** About 0.6–0.9 GB is unused at 4–16K context. Spending it on experts (about +0.25 bpw) is the
  open quality lever.
- **Single stream.** There is no batching across requests.

## Repository map

| Path | Contents |
|---|---|
| [`engine/`](engine/) | CUDA kernels (`moe_i8.cu`, `attn.cu`, `ops.cu`), forward pass (`engine.cu`), block decoder (`decode.cu`), C API (`libllada.cu`), tests and benchmarks |
| [`server/server.py`](server/server.py) | OpenAI-style `/v1/chat/completions` over `libllada.so` |
| [`quant/`](quant/) | Compression pipeline: corpus (`s02`), BF16 layer-streamed teacher pass (`s03`), keep-sets (`s04`), GPTQ cost table and ILP allocation (`s05`), held-out arbitration (`s06`), packing (`s07*`) |
| [`QUANT_PLAN.md`](QUANT_PLAN.md) | Compression decisions and every measured result |
| [`engine/ENGINE_DESIGN.md`](engine/ENGINE_DESIGN.md) | Engine design, profile and measured optimizations |
| [`MODEL_SELECTION_V2.md`](MODEL_SELECTION_V2.md), [`RESEARCH_AND_PLAN.md`](RESEARCH_AND_PLAN.md) | Model choice: measured Orin bandwidth and the small diffusion-LLM field (2026-10-24) |
| [`bench/`](bench/), [`tools/`](tools/) | Bandwidth probe; clock-pinning unit |

## Reproducing

The pipeline was designed to run on the Orin itself ([`QUANT_PLAN.md`](QUANT_PLAN.md) §3); the BF16 teacher streams
one layer at a time from NVMe. [`quant/chain.sh`](quant/chain.sh) runs the stages after the teacher pass in order. The scripts hard-code the
author's directories and virtual environment; edit the variables at the top of each script.
`engine/tests/test_layers` checks the engine against BF16 cases from `gen_layer_case.py` and against simulated
quantization cases from `quant/student_case.py`.

## Contributing and contact

Issues are welcome for reproducibility problems and wrong numbers. Pull requests are not being accepted yet.

## License

Apache-2.0 ([`LICENSE`](LICENSE)), the same as the base model,
[inclusionAI/LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini). The quantized weights inherit the base
model's license.

## Citation

```bibtex
@misc{devaney2026llada2orin,
  title  = {llada2-orin-cuda: LLaDA2.2-mini on an 8 GB Jetson Orin Nano},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/llada2-orin-cuda}
}
```

This work builds on LLaDA2.2 by inclusionAI and on ProQuant (arXiv 2610.30779), REAP (arXiv 2510.13999), GPTQ and
QuaRot-style rotation. Related: [deepseek-v4.1-flash-reap-thor](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor),
whose corpus sources and licence rules this pipeline reuses.
