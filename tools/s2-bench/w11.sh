#!/usr/bin/env bash
# w11.sh TAG REPS "n1:off1 n2:off2 ..."  -- cliff map: for each (n,offset) disjoint set: prime once, then REPS back-to-back warm passes, instrumented
cd /home/kevin/projects/lanes/s2-speed; export ESTATE_FR=$PWD/fr
TAG=${1:-cliff}; REPS=${2:-4}; SETS=${3:-"4:0 6:4 8:10 10:18 12:28"}
python3 sampler.py ${TAG}_sampler.csv & SP=$!
echo "[]" > ${TAG}_passes.json
add() { python3 - "$@" <<P
import json,sys
m=json.load(open("${TAG}_passes.json")); m.append(dict(name=sys.argv[1],t0=float(sys.argv[2]),t1=float(sys.argv[3]),file=sys.argv[4])); json.dump(m,open("${TAG}_passes.json","w"))
P
}
for s in $SETS; do n=${s%%:*}; off=${s##*:}
  EL="python3 estate_load.py --n $n --offset $off --min-tok 18000 --max-tok 36000"
  t0=$(date +%s.%N); $EL --max-tokens 1 >/dev/null 2>&1; $EL --max-tokens 256 --out ${TAG}_n${n}_prime.json > /dev/null 2>&1; t1=$(date +%s.%N); add n${n}_prime $t0 $t1 ${TAG}_n${n}_prime.json
  for i in $(seq 1 $REPS); do t0=$(date +%s.%N); $EL --max-tokens 256 --out ${TAG}_n${n}_r$i.json > /dev/null 2>&1; t1=$(date +%s.%N); add n${n}_r$i $t0 $t1 ${TAG}_n${n}_r$i.json; python3 analyze_cliff.py $TAG | tail -1; done
done
kill $SP
