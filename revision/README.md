# Revision — corpus scale reassembly of fragmented bamboo slips

Everything produced in response to the two reviews lives here. The submitted
package in `../submission/` and the submitted manuscript sources in `../paper/`
are left untouched.

```
revision/
├── code/          new experiments and the analysis that turns them into figures
├── results/       CSV output of every experiment
├── figs/          figures for the revised manuscript and the SI
├── paper/         revised manuscript, revised SI, response letter
├── logs/          run logs
└── data_real/     DeepJiandu photographs (not redistributed; see below)
```

## What changed in the model

Three additions to `../src/assemble.py` and one to `../src/synth.py`. Every new
option defaults to the behaviour of the submitted version, so
`../results/*.csv` still regenerates exactly. This is checked by
`code/test_invariants.py`, which also asserts the exactness of the
decomposition.

| Change | Where | Answers |
|---|---|---|
| Deposition and excavation recording (`DepositionSpec`) | `src/synth.py` | R2-2 |
| Graded contextual evidence (`strat_mode="soft"`, `fit_context_lr`) | `src/assemble.py` | R1-1, R2-2 |
| Slip length pooled over a roll (`use_roll_length`) | `src/assemble.py` | R1-2, R2-5 |
| Notch cord ordering (`notch_order`) | `src/assemble.py` | R2-5 |
| Exact decomposition (`solve_decomposed`) | `src/assemble.py` | R2-3 |
| Global puzzle-solving baselines | `code/global_baselines.py` | R2-4 |

The published behaviour is reproduced bit for bit: `code/test_invariants.py`
regenerates a corpus and a solution with the defaults and compares them against
the submitted results.

## Reproducing

```bash
PY=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd /home/sam/paper_ws/paper-nature/yunmeng

$PY revision/code/test_invariants.py                       # must pass first

$PY -u revision/code/exp_dispersion.py --slips 400 --seeds 3   # E1
$PY -u revision/code/exp_notch.py      --slips 400 --seeds 3   # E2
$PY -u revision/code/exp_decomp.py     --seeds 3               # E3
$PY -u revision/code/exp_baselines.py  --slips 400 --seeds 3   # E4
$PY -u revision/code/exp_real.py       --slips 200 --seeds 3   # E5
$PY -u revision/code/exp_realism.py    --slips 400 --seeds 3   # E6

$PY revision/code/figures_rev.py
$PY revision/code/tables_rev.py
cd revision/paper && latexmk -pdf main.tex && latexmk -pdf response.tex
```

## Real photographs

`code/exp_real.py` builds corpora whose substrate is the DeepJiandu dataset:
7,416 infrared photographs of excavated Qin and Han slips, DOI
`10.57760/sciencedb.08560`, CC BY 4.0.

```bash
mkdir -p revision/data_real && cd revision/data_real
curl -L -o DeepJiandu.zip "https://china.scidb.cn/download?fileId=ad765583840b679209eb21842436b1f0"
unzip -q DeepJiandu.zip -d images
```

The photographs are not redistributed here. `code/exp_real.py` rebuilds the
corpus from the archive, and `data_real/screened.json` records exactly which
files passed segmentation, so the corpus is reproducible from the DOI alone.

## Note on the threshold grid

The graded context term is referenced to the same-square case
(`fit_context_lr` subtracts `lr[0]`). Without that, a term common to every arc
would shift one method along the threshold grid relative to another, and the
comparison would measure the grid rather than the model. Every experiment
reports whether a tuned threshold landed on a grid endpoint.

## Headline results are rerun under the deposition model

Table 1, Table 2, the exchange rate, the case study and the scaling curve in the
revised manuscript all come from `results/rerun_*.csv`, produced by
`code/exp_main.py` with fragments deposited, dispersed and recorded as
`DepositionSpec(model="spatial")` describes, and with the excavation record read
as graded evidence. The submitted versions of those numbers rested on one unit
drawn per slip, never missing and never wrong, which the revision itself shows is
not a record any excavation delivers.

Four experiments predate that rerun and select each method's threshold on join
F1 rather than on the partition index: the comparison with global puzzle solving,
the real material corpora, the misspecification sweep and the test under uneven
preservation. They are internally consistent, every method within them is treated
alike, and the manuscript says so where they are reported.

## Reproducing the headline

```bash
$PY -u revision/code/exp_main.py --what main     --slips 400 --seeds 5
$PY -u revision/code/exp_main.py --what ablation --slips 400 --seeds 5 --states P4
$PY -u revision/code/exp_main.py --what main --slips 150,400,900,1800 --seeds 3 \
       --states P4 --out revision/results/rerun_scaling.csv
```
