#!/usr/bin/env bash
# What remains after the scaling sizes.
#
# The 1200-slip size is dropped: the Shuihudi-scale case is 3253 fragments and
# answers the same question, and the scaling curve is already measured at four
# sizes. The frontier control block runs alongside the case because its numbers
# do not depend on wall-clock time, only the scaling timings did.
set -u
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd "$ROOT"

running() {   # true while any experiment python is alive, excluding this script
  pgrep -x python -x python3 -u "$(id -u)" 2>/dev/null | while read -r x; do
    [ -r "/proc/$x/cmdline" ] || continue
    case "$(tr '\0' ' ' < "/proc/$x/cmdline")" in
      *exp_decomp.py*) echo y ;;
    esac
  done | grep -q y
}

while running; do sleep 60; done
echo "scaling sizes finished at $(date +%H:%M)"

nice -n 10 "$PY" -u revision/code/exp_notch_roll.py --slips 400 --cal-slips 120 \
    --seeds 3 --time-limit 180 --workers 6 --only-control \
    > revision/logs/e2c_control.log 2>&1 &
control=$!

nice -n 5 "$PY" -u revision/code/exp_decomp.py --sizes 1155 --seeds 1 \
    --short 240 --long 2400 --workers 10 \
    --out revision/results/decomposition_case.csv \
    > revision/logs/e3_case.log 2>&1
echo "Shuihudi case exit $? at $(date +%H:%M)"

wait "$control"
echo "control block exit $? at $(date +%H:%M)"
echo "ALL EXPERIMENTS DONE"
