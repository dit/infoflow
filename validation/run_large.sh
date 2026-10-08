#!/usr/bin/env bash
# Large-network validation (100 nodes, up to 10 000 samples), after Novelli et al. (2019).
# One file per configuration; rerunning skips configurations already written.
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
R=validation/results/alpha001
L=${LOGDIR:-/tmp}
mkdir -p $R
P=(.venv/bin/python -u validation/novelli.py --nodes 100 --reps 1 --alpha 0.001 --prescreen 0.1)
run() {
    local name=$1
    shift
    [ -e $R/$name.nc ] || "${P[@]}" "$@" --out $R/$name.nc > $L/$name.log 2>&1
}
for T in 1000 3000 10000; do
    run val100_var_T$T --kind var --samples $T --threads 4 --device mps
    run val100_logistic_T$T --kind logistic --samples $T --threads 4 --device mps
done
run val100_empty --kind var --in-degree 0 --samples 10000 --threads 4 --device mps
run val100_var_strata --kind var --samples 10000 --null strata --statistic debiased --threads 4 --device mps
run val100_logistic_ksg5 --kind logistic --samples 10000 --estimator ksg --n-targets 5 --threads 1
echo LARGE DONE
