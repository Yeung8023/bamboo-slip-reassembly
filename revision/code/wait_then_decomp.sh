#!/usr/bin/env bash
# The decomposition timings are the one measurement that needs a quiet machine,
# so this waits for every other experiment to finish before starting.
set -u
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd "$ROOT"

busy() {
  for x in $(pgrep -f "revision/code/exp_"); do
    [ -r "/proc/$x/comm" ] || continue
    grep -q python "/proc/$x/comm" || continue
    case "$(tr '\0' ' ' < /proc/$x/cmdline)" in
      *exp_decomp*) : ;;
      *) return 0 ;;
    esac
  done
  for x in $(pgrep -x python3 -x python 2>/dev/null); do :; done
  return 1
}

quiet() {
  # the solve times reported in the manuscript have to come from an idle
  # machine, and this one is shared, so wait for other work as well as ours
  awk '{exit ($1 < 6.0) ? 0 : 1}' /proc/loadavg
}

while busy || ! quiet; do sleep 60; done
echo "machine free at $(date +%H:%M), load $(cut -d' ' -f1-3 /proc/loadavg)"

nice -n 5 $PY -u revision/code/exp_decomp.py \
    --sizes 150,400,900 --seeds 3 --short 240 --long 1500 --workers 10 \
    > revision/logs/e3_decomp.log 2>&1
echo "E3 small exit $?"

nice -n 5 $PY -u revision/code/exp_decomp.py \
    --sizes 1200 --seeds 1 --short 240 --long 1800 --workers 10 --case \
    --out revision/results/decomposition_large.csv \
    > revision/logs/e3_decomp_large.log 2>&1
echo "E3 large exit $?"
