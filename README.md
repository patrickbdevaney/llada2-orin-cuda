# llada2-orin-cuda

A pure-CUDA inference engine and OpenAI-compatible server for **[LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini)**
(16B-parameter MoE diffusion LLM, 1.4B active) on a **Jetson Orin Nano Super 8GB** (sm_87, 7.4 GiB shared memory).
No PyTorch at runtime: hand-written kernels, a C++ block decoder, and a small Python HTTP front end.

Weights: **[patrickbdevaney/LLaDA2.2-mini-orin-nano](https://huggingface.co/patrickbdevaney/LLaDA2.2-mini-orin-nano)**
(6.3 GB packed: experts mostly 2-bit with some 3/4-bit, GPTQ, Hadamard-rotated; attention, shared expert and
lm_head 8-bit; int4 KV cache).

## Performance (Orin Nano Super, `sudo jetson_clocks`)

| | |
|---|---|
| denoising step (32-token block) | 24.7 ms at 128 tokens of context, 25.3 ms at 1K, 30.6 ms at 8K, 34.4 ms at 16K |
| decode, server default (threshold 0.3, 16 steps) | ~180 tok/s on math/code/tool-call tasks |
| decode, model-card settings (`--reference`) | ~150 tok/s on the same tasks; prose is slower (40-80 tok/s) |
| prefill | ~2.9K tok/s (16 blocks per pass) |
| context | 32K tokens (int4 KV, 11.8 KB/token) |

Quality check: `engine/tests/eval_knobs.py` (96 generated math problems, 12 code tasks run against tests, 8 tool
calls) scores 113-116 / 116 at either setting. Misses have correct arithmetic and a doubled final digit, a
repetition artifact of the quantized model. See `engine/ENGINE_DESIGN.md` for where the time goes and what was tried.

## Layout

- `engine/` — CUDA engine: `moe_i8.cu` (W2/3/4/8 x A8 int8 tensor-core GEMMs over packed tiles), `attn.cu` (attention
  over the int4 KV cache), `ops.cu` (norms, Hadamard rotations, tensor-core router, routing/scheduling), `engine.cu`
  (forward pass, batched prefill, CUDA graphs), `decode.cu` (LLaDA2.2 JointThresholdInDel block decoding with a
  screened lm_head readout), `libllada.cu` (C API). Tests and benchmarks in `engine/tests/`.
- `server/server.py` — OpenAI-style `/v1/chat/completions` (tools, streaming) on top of `libllada.so`.
- `quant/` — the quantization pipeline that produced the weights (calibration corpus, teacher pass, per-expert bit
  allocation, GPTQ, packing). Plans and measurements: `QUANT_PLAN.md`.

## Run

```bash
# toolchain: CUDA 12.6 nvcc for sm_87 (set NVCC=... if it is not at the Makefile's default)
cd engine && make            # bench_gen, libllada.so, tests/test_layers
pip install tokenizers jinja2 huggingface_hub
huggingface-cli download patrickbdevaney/LLaDA2.2-mini-orin-nano --local-dir ~/models/llada2mini-orin
cd ../server && python3 server.py --model ~/models/llada2mini-orin --engine ~/models/llada2mini-orin
```

Then `POST http://127.0.0.1:8000/v1/chat/completions`. Per request, `threshold`, `steps`, `editing_threshold` and
`max_post_steps` override the decode defaults; `--reference` switches the server to the model card's settings.

The weights take ~5.7 GB of the Orin's 7.4 GiB; run nothing else GPU-heavy alongside. Pin clocks for consistent
speed (`tools/jetson-clocks.service` does it at boot).

## License

Apache-2.0, as is the base model, [inclusionAI/LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini).
