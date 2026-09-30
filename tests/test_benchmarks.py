"""
Tests for the conditioning-scaling benchmark.
"""

import numpy as np
import pytest

from infoflow.benchmarks import conditioning_scaling, scaling_data


def test_analytic_values():
    assert scaling_data(10, 0, "gaussian")[3] == pytest.approx(-0.5 * np.log2(1 - 0.36))
    h2 = lambda p: -p * np.log2(p) - (1 - p) * np.log2(1 - p)  # noqa: E731
    assert scaling_data(10, 0, "discrete")[3] == pytest.approx(h2(0.34) - h2(0.1))
    assert scaling_data(10, 3, "discrete", coupled=False)[3] == 0.0
    ds = conditioning_scaling("discrete", ("plugin",), n_samples=(200_000,), dims=(0,), n_reps=1, prng=0)
    assert float(ds["bias"].squeeze()) == pytest.approx(0, abs=0.005)


def test_symbol_estimates_inflate_then_collapse():
    ds = conditioning_scaling(
        "gaussian", ("plugin",), n_samples=(1000,), dims=(0, 4, 12), n_reps=3, coupled=False, prng=0
    )
    spurious = ds["mean"].sel(estimator="plugin", n=1000).values
    assert spurious[0] < 0.02 and spurious[1] > 0.3 and spurious[2] < 0.05


def test_ksg_never_manufactures_dependence_and_decays_downward():
    null = conditioning_scaling(
        "gaussian", ("ksg",), n_samples=(1000,), dims=(0, 4, 8), n_reps=3, coupled=False, prng=0
    )
    assert np.all(np.abs(null["mean"].values) < 0.03)
    ds = conditioning_scaling("gaussian", ("ksg",), n_samples=(1000,), dims=(0, 8), n_reps=3, prng=0)
    truth = ds.attrs["truth"]
    near, far = ds["mean"].sel(estimator="ksg", n=1000).values
    assert near == pytest.approx(truth, abs=0.06)
    assert 0.1 < far < truth
