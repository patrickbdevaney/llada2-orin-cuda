#!/usr/bin/env bash
# Stages after s03, one at a time (each needs the shared 7.4 GiB heap alone).
# s04 -> smoke gate (s05 one layer + s06 teacher/rotid/keep/uniform arms on the 4-row smoke corpus) -> s05 costs
# -> s05 allocation sweep. Any failure stops the chain and is logged in logs/chain.log.
set -u
source ~/venvs/orin-quant/env.sh
cd ~/alodlm-orin-cuda/quant
LOG=~/models/llada-work/logs
PY=~/venvs/orin-quant/bin/python
SMOKE=~/models/llada-smoke
step() { echo "[$(date +%T)] $1" >> $LOG/chain.log; }
fail() { echo "[$(date +%T)] FAILED $1 rc=$2" >> $LOG/chain.log; exit 1; }
DONE=~/models/llada-work/s03/teacher_done.json
until [ -f $DONE ]; do                     # completion marker, not the log (a resumed run appends after old rc= lines)
  sleep 30
  if [ ! -f $DONE ] && ! pgrep -f s03_teacher_pass.py > /dev/null; then fail s03 "no process, no $DONE"; fi
done
step "s04"; $PY s04_select.py > $LOG/s04.log 2>&1 || fail s04 $?
step "smoke s05"
rm -rf $SMOKE/s05 $SMOKE/hs5 $SMOKE/s06
LLADA_WORK=$SMOKE $PY s05_costs.py --corpus $SMOKE/corpus --row-stride 1 --cap 1024 --layers 1 \
  > $LOG/smoke_s05.log 2>&1 || fail smoke_s05 $?
step "smoke s06"
LLADA_WORK=$SMOKE $PY s06_arbitrate.py --corpus $SMOKE/corpus --arm teacher --arm rotid \
  --arm keep:$SMOKE/s04/keepsets/hope_sum_k205.npy --arm uniform:4@1 --arm uniform:2@1 \
  --reference teacher > $LOG/smoke_s06.log 2>&1 || fail smoke_s06 $?
step "s05 costs"; $PY s05_costs.py > $LOG/s05.log 2>&1 || fail s05 $?
step "s05 alloc"; $PY s05_alloc.py > $LOG/s05_alloc.log 2>&1 || fail s05_alloc $?
step "chain done"
