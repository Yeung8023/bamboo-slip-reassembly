#!/usr/bin/env bash
# Let the roll-standard block finish all three seeds, then stop the frontier so
# the decomposition timings can be taken on a quiet machine. The control block
# is rerun afterwards if there is time; it is our own check, not a reviewer's.
set -u
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
cd "$ROOT"
while [ "$(grep -c 'roll_typical. seed 2' revision/logs/e2b_notch_roll.log)" -lt 5 ]; do
  sleep 45
  pgrep -f exp_notch_roll.py > /dev/null || break
done
echo "roll-standard block complete at $(date +%H:%M); stopping the frontier"
for x in $(pgrep -f exp_notch_roll.py); do
  [ -r "/proc/$x/comm" ] && grep -q python "/proc/$x/comm" && kill "$x"
done
