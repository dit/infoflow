#!/usr/bin/env bash
# Large-network validation with the linear-Gaussian estimator (100 nodes), after Novelli et al. (2019).
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
R=validation/results
L=${LOGDIR:-/tmp}
P=(.venv/bin/python -u validation/novelli.py --nodes 100 --reps 1 --estimator gaussian --threads 4 --no-shared)
"${P[@]}" --kind var --samples 1000 3000 10000 --out $R/val100_var_gaussian.nc > $L/val100_var_gaussian.log 2>&1
"${P[@]}" --kind var --in-degree 0 --samples 10000 --out $R/val100_empty_gaussian.nc > $L/val100_empty_gaussian.log 2>&1
"${P[@]}" --kind logistic --samples 10000 --out $R/val100_logistic_gaussian.nc > $L/val100_logistic_gaussian.log 2>&1
echo GAUSSIAN DONE
