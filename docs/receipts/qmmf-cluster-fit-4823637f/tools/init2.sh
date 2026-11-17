#!/bin/bash
# usage: init2.sh TREE_A TREE_B PREFIX: one serve each (each binary first load times its expert forms); prints both forms.
cd ~/glm-perf; O=out/$3-forms.txt; : > $O
for T in $1 $2; do
  L=$3-init-$T
  ./up.sh $T $L 65536 -- --parallel 1 --prompt-cache-gib 0 && python3 one_sha.py "hello" 16 > /dev/null 2>&1; ./down.sh $L
  echo "$T $(grep -h "windows (kept\|windows (timed" out/$L/server-rank0.log | tail -1)" >> $O
done
echo ALLDONE >> $O
