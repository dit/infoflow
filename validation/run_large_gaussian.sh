#!/usr/bin/env bash
# Large-network validation with the linear-Gaussian estimator (100 nodes), after Novelli et al. (2019).
# One file per configuration; rerunning skips configurations already written.
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
R=validation/results/alpha001
L=${LOGDIR:-/tmp}
mkdir -p $R
P=(.venv/bin/python -u validation/novelli.py --nodes 100 --reps 1 --alpha 0.001 --estimator gaussian --threads 4 --no-shared)
run() {
    local name=$1
    shift
    [ -e $R/$name.nc ] || "${P[@]}" "$@" --out $R/$name.nc > $L/$name.log 2>&1
}
for T in 1000 3000 10000; do
    run val100_var_gaussian_T$T --kind var --samples $T
done
run val100_empty_gaussian --kind var --in-degree 0 --samples 10000
run val100_logistic_gaussian --kind logistic --samples 10000
echo GAUSSIAN DONE
