"""
Tests for per-layer intervals, p-values, and FDR adjustment.
"""

import numpy as np
import pytest

from infoflow.layers import adjust_layer_pvalues, layer_statistics


def _samples(kind, n, rng):
    x = rng.integers(0, 2, n)
    w = rng.integers(0, 2, n)
    if kind == "copy":
        return np.where(rng.random(n) < 0.9, x, 1 - x), x, w
    if kind == "xor":
        return x ^ w, x, w
    if kind == "none":
        return rng.integers(0, 2, n), x, w
    z = rng.integers(0, 2, n)
    return z ^ (rng.random(n) < 0.1), z ^ (rng.random(n) < 0.1), z


@pytest.mark.parametrize(
    ("kind", "significant"),
    [("copy", {"intrinsic"}), ("xor", {"synergistic"}), ("shared", {"shared"}), ("none", set())],
)
def test_layer_pvalues_identify_the_right_layer(kind, significant):
    stats = layer_statistics(*_samples(kind, 3000, np.random.default_rng(1)), n_boot=39, n_null=39, prng=0)
    detected = {layer for layer, p in stats.pvalue.items() if p <= 0.05}
    assert detected == significant


def test_copy_interval_covers_truth():
    truth = 1 + 0.9 * np.log2(0.9) + 0.1 * np.log2(0.1)
    stats = layer_statistics(*_samples("copy", 3000, np.random.default_rng(2)), n_boot=100, n_null=0, prng=0)
    low, high = stats.ci["intrinsic"]
    assert low < truth < high
    assert stats.ci["te"][0] <= stats.flow.te <= stats.ci["te"][1]


def test_intrinsic_null_size():
    """
    Under a pure-synergy null (intrinsic flow zero), the intrinsic test rarely rejects.
    """
    rejections = 0
    for seed in range(20):
        stats = layer_statistics(*_samples("xor", 500, np.random.default_rng(seed)), n_boot=0, n_null=19, prng=seed)
        rejections += stats.pvalue["intrinsic"] <= 0.05
    assert rejections <= 3


def test_adjust_layer_pvalues():
    pvalues = {"intrinsic": np.array([0.001, 0.5, np.nan, 0.01]), "shared": np.array([np.nan] * 4)}
    significant, adjusted = adjust_layer_pvalues(pvalues, alpha=0.05)
    assert significant["intrinsic"].tolist() == [True, False, False, True]
    assert np.isnan(adjusted["intrinsic"][2])
    assert not significant["shared"].any()
    dependent, _ = adjust_layer_pvalues(pvalues, alpha=0.05, dependent=True)
    assert dependent["intrinsic"].sum() <= significant["intrinsic"].sum()
