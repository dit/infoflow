#!/usr/bin/env bash
# Symbol estimators with fewer degrees of freedom than the plug-in CMI, on the same
# 100-node networks as run_large.sh: the stratified ordinal trend and coarse symbols.
# One file per configuration; rerunning skips configurations already written.
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
R=validation/results/alpha001
L=${LOGDIR:-/tmp}
mkdir -p $R
P=(.venv/bin/python -u validation/novelli.py --nodes 100 --reps 1 --alpha 0.001 --prescreen 0.1 --threads 4)
run() {
    local name=$1
    shift
    [ -e $R/$name.nc ] || "${P[@]}" "$@" --out $R/$name.nc > $L/$name.log 2>&1
}
for est in trend coarse; do
    # coarse keeps plug-in CMI, whose permutation batches run on the GPU
    D=(--estimator $est)
    [ $est = coarse ] && D+=(--device mps)
    for T in 1000 3000 10000; do
        run val100_var_${est}_T$T --kind var --samples $T "${D[@]}"
    done
    run val100_empty_${est} --kind var --in-degree 0 --samples 10000 "${D[@]}"
    run val100_logistic_${est} --kind logistic --samples 10000 "${D[@]}"
done
echo OPTIONS DONE
