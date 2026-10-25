#!/bin/bash
# bench_gen over tests/prompts/*.bin; prints per-prompt timing lines; generated text into $OUT/<name>.txt
D=$(dirname "$0"); OUT=${OUT:-/tmp}; MAXNEW=${MAXNEW:-512}
for p in $D/prompts/*.bin; do
  n=$(basename $p .bin)
  $D/../bench_gen ${ENGINE:-$HOME/models/llada2mini-orin} $p $MAXNEW 8192 > $OUT/$n.raw
  echo "$n: $(grep '\[run 0\]' $OUT/$n.raw | sed 's/.*prompt/prompt/')"
  echo "$n: $(grep '\[run 1\]' $OUT/$n.raw | sed 's/.*prompt/prompt/')"
  python3 -c "
from tokenizers import Tokenizer; import pathlib, sys
t=Tokenizer.from_file(str(pathlib.Path.home()/'models/LLaDA2.2-mini/tokenizer.json'))
ids=[int(x) for x in open('$OUT/$n.raw').read().split('\n')[3 - 1].split()]
open('$OUT/$n.txt','w').write(t.decode(ids, skip_special_tokens=False))"
done
