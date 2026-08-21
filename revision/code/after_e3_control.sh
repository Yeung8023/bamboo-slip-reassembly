#!/usr/bin/env bash
# The control block of the frontier: the same five variants on corpora with no
# roll standard at all, which is what shows that the gain comes from the
# structure of slip length and not from the extra variables.
# It runs after the decomposition experiment, which needs the quiet machine.
set -u
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd "$ROOT"

# wait for the decomposition experiment to appear and then to finish
for _ in $(seq 1 120); do
  pgrep -f "exp_decomp.py" > /dev/null && break
  sleep 60
done
while pgrep -f "exp_decomp.py" | while read x; do
        [ -r "/proc/$x/comm" ] && grep -q python "/proc/$x/comm" && echo y; done | grep -q y; do
  sleep 60
done
echo "decomposition finished at $(date +%H:%M); running the frontier control block"

nice -n 10 $PY -u revision/code/exp_notch_roll.py \
    --slips 400 --cal-slips 120 --seeds 3 --time-limit 180 --workers 8 \
    --only-control > revision/logs/e2c_control.log 2>&1
echo "control block exit $?"
