NVCC ?= $(HOME)/cuda-nvcc/usr/local/cuda-12.6/bin/nvcc
FLAGS = -O3 -arch=sm_87 -std=c++17 -I.
SRC = engine.cu decode.cu moe_i8.cu attn.cu ops.cu

all: bench_gen libllada.so tests/test_layers

bench_gen: bench_gen.cu $(SRC)
	$(NVCC) $(FLAGS) $< -o $@
libllada.so: libllada.cu $(SRC)
	$(NVCC) $(FLAGS) -Xcompiler -fPIC -shared $< -o $@
tests/test_layers: tests/test_layers.cu $(SRC)
	$(NVCC) $(FLAGS) $< -o $@
.PHONY: all
tests/bench_kern: tests/bench_kern.cu $(SRC)
	$(NVCC) $(FLAGS) $< -o $@
