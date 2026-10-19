#!/usr/bin/env bash
# Hierarchical Bayesian estimators (infoflow.bayes): motif layer accuracy, conditioning
# scaling, and Bayes-factor vs permutation tests of TE = 0.
# Results: validation/results/bayes/<part>.json. Resumable per part with --parts.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --extra bayes python validation/bayes_benchmark.py "$@"
