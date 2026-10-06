#!/usr/bin/env bash
# Large-network validation (100 nodes, up to 10 000 samples), after Novelli et al. (2019).
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
R=validation/results
L=${LOGDIR:-/tmp}
P=(.venv/bin/python -u validation/novelli.py --nodes 100 --reps 1 --alpha 0.001 --prescreen 0.1)
"${P[@]}" --kind var --samples 1000 3000 10000 --threads 4 --device mps --out $R/val100_var.nc > $L/val100_var.log 2>&1
"${P[@]}" --kind logistic --samples 1000 3000 10000 --threads 4 --device mps --out $R/val100_logistic.nc > $L/val100_logistic.log 2>&1
"${P[@]}" --kind var --in-degree 0 --samples 10000 --threads 4 --device mps --out $R/val100_empty.nc > $L/val100_empty.log 2>&1
"${P[@]}" --kind var --samples 10000 --null strata --statistic debiased --threads 4 --device mps --out $R/val100_var_strata.nc > $L/val100_var_strata.log 2>&1
"${P[@]}" --kind logistic --samples 10000 --estimator ksg --n-targets 20 --threads 1 --out $R/val100_logistic_ksg.nc > $L/val100_logistic_ksg.log 2>&1
echo LARGE DONE
