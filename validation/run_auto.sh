#!/usr/bin/env bash
# infer(balanced) against its member estimators on small and 100-node scenarios.
# One file per (scenario, arm); rerunning skips what is already written.
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
R=validation/results/auto
L=${LOGDIR:-/tmp}
mkdir -p $R
run() {
    local scenario=$1 arm=$2
    local name=${scenario}_${arm}
    [ -e $R/$name.nc ] || .venv/bin/python -u validation/auto.py --scenario $scenario --arm $arm --out $R/$name.nc \
        > $L/auto_$name.log 2>&1
}
for scenario in mute orthogonal; do
    for arm in auto gaussian trend coarse plugin; do run $scenario $arm; done
done
for scenario in xor common_driver chain; do
    for arm in auto coarse plugin; do run $scenario $arm; done
done
for scenario in var_T3000 logistic_T10000 empty_T10000 var_T10000; do
    for arm in auto gaussian trend coarse; do run $scenario $arm; done
done
echo AUTO DONE
