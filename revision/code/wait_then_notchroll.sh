#!/usr/bin/env bash
set -u
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd "$ROOT"
# wait for the tolerance sweep to release its slot
while pgrep -f "revision/code/exp_notch.py" | while read x; do
        [ -r "/proc/$x/comm" ] && grep -q python "/proc/$x/comm" && echo y; done | grep -q y; do
  sleep 60
done
echo "tolerance sweep finished at $(date +%H:%M); starting the roll-length run"
nice -n 10 $PY -u revision/code/exp_notch_roll.py \
    --slips 400 --cal-slips 120 --seeds 3 --time-limit 180 --workers 6 \
    > revision/logs/e2b_notch_roll.log 2>&1
echo "E2b exit $?"
