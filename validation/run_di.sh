#!/usr/bin/env bash
# infoflow vs directed-information baselines (pairwise DI, DIG with bounded in-degree,
# oCSE) on the tent ring, noisy ECA rules 90/110, and random VAR/logistic networks.
# Results: validation/results/di/results.json. Resumable per case with --cases.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python validation/di_benchmark.py "$@"
