#!/usr/bin/env bash
# s05n (non-expert GPTQ 4/8) -> s06 run 3: non-expert precision, lm_head precision and A8 expert activations,
# all on top of the chosen 32K expert allocation with the 4c/4 KV cache.
set -u
source ~/venvs/orin-quant/env.sh
cd ~/alodlm-orin-cuda/quant
LOG=~/models/llada-work/logs
PY=~/venvs/orin-quant/bin/python
A=~/models/llada-work/s05/alloc/32768_int4_noprune.npy
$PY s05n_nonexpert.py > $LOG/s05n.log 2>&1 || { echo "s05n rc=$?" >> $LOG/chain_ne.log; exit 1; }
echo "s05n done $(date +%T)" >> $LOG/chain_ne.log
$PY s06_arbitrate.py --out ~/models/llada-work/s06ne --reference ng_kv \
  --arm teacher --arm ng_kv=alloc:$A+kv:4c/4 --arm ng_kv_ne8=alloc:$A+kv:4c/4+ne:8/8 \
  --arm ng_kv_ne4=alloc:$A+kv:4c/4+ne:4/8 --arm ng_kv_ne8lm4=alloc:$A+kv:4c/4+ne:8/4 \
  --arm ng_kv_ne8_a8=alloc:$A+kv:4c/4+ne:8/8+act:8 > $LOG/s06ne.log 2>&1
echo "s06ne rc=$? $(date +%T)" >> $LOG/chain_ne.log
