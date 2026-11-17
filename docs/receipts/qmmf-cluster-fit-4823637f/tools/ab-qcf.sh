#!/bin/bash
# #3681 receipt A/B: main (qcfbase) vs branch (qcf), ABBA twice, 5 plain + 3 drafted 128-token runs a serve, shas, forms.
cd ~/glm-perf; O=out/qab.txt; : > $O
for L in qab-a1 qab-b1 qab-b2 qab-a2 qab-a3 qab-b3 qab-b4 qab-a4; do
  T=qcfbase; case $L in qab-b*) T=qcf;; esac
  ./up.sh $T $L 65536 -- --parallel 1 --prompt-cache-gib 0 || { echo "$L UPFAIL" >> $O; ./down.sh $L; continue; }
  python3 one_sha.py "Write a long story about a lighthouse keeper." 64 > /dev/null 2>&1
  for i in 1 2 3 4 5; do python3 one_sha.py "Write a long story about a lighthouse keeper." 128; done > out/$L/runs.txt 2>&1
  for i in 1 2 3; do python3 one.py "Write a long story about a lighthouse keeper." 128 1; done > out/$L/drafted.txt 2>&1
  python3 one_sha.py "$(python3 -c "print(open(\"one.py\").read()*20)")" 64 > out/$L/long.txt 2>&1
  ./down.sh $L
  echo "$L $T plain $(awk "{print \$1}" out/$L/runs.txt | tr "\n" " ") drafted $(awk "{print \$1}" out/$L/drafted.txt | tr "\n" " ") shas $(awk "{print \$2}" out/$L/runs.txt | sort -u | tr "\n" " ") long $(awk "{print \$2}" out/$L/long.txt) $(grep -h "experts" out/$L/server-rank0.log | grep -o "experts [a-z]*" | tail -1)" >> $O
done
echo ALLDONE >> $O
