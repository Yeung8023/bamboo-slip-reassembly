#!/usr/bin/env bash
# Every experiment of the revision, in the order they were run.
#
# The decomposition timings (E3) must be measured on an idle machine, so that
# one runs last and alone. Everything else can share the machine.
set -euo pipefail

PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
ROOT=/home/sam/paper_ws/paper-nature/yunmeng
cd "$ROOT"
mkdir -p revision/results revision/figs revision/logs

# The published behaviour must be reproduced exactly before anything else is
# believed, and the decomposition must be exact.
$PY revision/code/test_invariants.py

# E1  excavation context under the deposition model            (R2-2, R1-1)
nice -n 10 $PY -u revision/code/exp_dispersion.py \
    --slips 400 --cal-slips 120 --seeds 3 --time-limit 180 --workers 7 \
    > revision/logs/e1_dispersion.log 2>&1 &

# E2  binding notches and the pooled slip length               (R1-2, R2-5)
nice -n 10 $PY -u revision/code/exp_notch.py \
    --slips 400 --cal-slips 120 --seeds 3 --time-limit 180 --workers 7 \
    > revision/logs/e2_notch.log 2>&1 &

# E4  global reassembly baselines                              (R2-4)
nice -n 12 $PY -u revision/code/exp_baselines.py \
    --states P2,P3,P4,P5 --slips 400 --cal-slips 120 --seeds 3 \
    --time-limit 180 --workers 5 > revision/logs/e4_baselines.log 2>&1 &

# E6  misspecified taphonomy and uneven preservation           (R2-1)
nice -n 14 $PY -u revision/code/exp_realism.py \
    --slips 400 --cal-slips 120 --seeds 3 --time-limit 180 --workers 4 \
    > revision/logs/e6_realism.log 2>&1 &
wait

# E5  photographs of excavated slips                           (R2-1)
# Fetch the images first; see revision/README.md for the archive and its DOI.
nice -n 12 $PY -u revision/code/exp_real.py \
    --slips 300 --cal-slips 120 --seeds 3 --time-limit 180 --workers 5 \
    > revision/logs/e5_real.log 2>&1

# E3  decomposition and proven optimality                      (R2-3)
# Alone, on an idle machine: the timings in the manuscript come from here.
nice -n 5 $PY -u revision/code/exp_decomp.py \
    --sizes 150,400,900,1800,2600 --seeds 3 --time-limit 900 --workers 10 \
    --case > revision/logs/e3_decomp.log 2>&1

$PY revision/code/summarise.py | tee revision/logs/summary.txt
$PY revision/code/figures_rev.py
$PY revision/code/tables_rev.py
cd revision/paper && latexmk -pdf main.tex && latexmk -pdf supplementary.tex \
    && latexmk -pdf response.tex
