#!/usr/bin/env bash
# s05 costs (resumable) -> s05 allocation sweep, for re-runs after s04 and the smoke gate have passed.
set -u
source ~/venvs/orin-quant/env.sh
cd ~/alodlm-orin-cuda/quant
LOG=~/models/llada-work/logs
PY=~/venvs/orin-quant/bin/python
step() { echo "[$(date +%T)] $1" >> $LOG/chain.log; }
fail() { echo "[$(date +%T)] FAILED $1 rc=$2" >> $LOG/chain.log; exit 1; }
step "s05 costs (resume)"; $PY s05_costs.py >> $LOG/s05.log 2>&1 || fail s05 $?
step "s05 alloc"; $PY s05_alloc.py > $LOG/s05_alloc.log 2>&1 || fail s05_alloc $?
step "chain done"
