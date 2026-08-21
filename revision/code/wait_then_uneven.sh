#!/usr/bin/env bash
# Part B of the realism experiment is rerun after the fix to merge(), which
# stopped sub-corpora sharing excavation-unit labels they could not share.
set -u
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd "$ROOT"
while pgrep -f "revision/code/exp_realism.py" | while read x; do
        [ -r "/proc/$x/comm" ] && grep -q python "/proc/$x/comm" && echo y; done | grep -q y; do
  sleep 60
done
echo "first realism run finished at $(date +%H:%M); rerunning the uneven-preservation part"
nice -n 12 $PY -u revision/code/exp_realism.py --slips 400 --cal-slips 120 \
    --seeds 3 --time-limit 180 --workers 5 --only uneven \
    --out revision/results/realism_uneven.csv \
    > revision/logs/e6b_uneven.log 2>&1
echo "E6b exit $?"
