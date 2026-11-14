"""
Tests for the synergy generators and validations (Hourican & Quax 2026 mechanisms).
"""

import numpy as np
import pytest

from infoflow import datasets
from infoflow.benchmarks import planted_module_validation, synergy_validation
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=50, n_perm_min_stat=50, n_perm_omnibus=100, n_perm_max_seq=100, n_perm_pairs=50, n_perm_tdmi=50
)


def test_lagged_generators():
    x = datasets.xor_descendant(4000, seed=0)
    assert x.shape == (4000, 4)
    assert np.mean(x[1:, 2] == x[:-1, 0] ^ x[:-1, 1]) == pytest.approx(0.9, abs=0.02)
    assert np.mean(x[1:, 3] == x[:-1, 2]) == pytest.approx(0.9, abs=0.02)
    a = datasets.asymmetric_interaction(20000, seed=0)
    assert a[:, 0].mean() == pytest.approx(0.35, abs=0.01) and a[:, 1].mean() == pytest.approx(0.6, abs=0.01)
    cell = (a[:-1, 0] == 1) & (a[:-1, 1] == 0)
    assert a[1:, 2][cell].mean() == pytest.approx(0.8, abs=0.03)
    g = datasets.or_gate(4000, seed=0)
    assert np.mean(g[1:, 2] == g[:-1, 0] | g[:-1, 1]) == pytest.approx(0.9, abs=0.02)
    m = datasets.xor_among(4000, seed=0)
    assert m.shape == (4000, 12)
    assert np.mean(m[3:, 11] == m[:-3, 8] ^ m[1:-2, 10]) == pytest.approx(0.9, abs=0.02)


@pytest.mark.parametrize("kind", datasets.CONTEMPORANEOUS_MOTIFS)
def test_contemporaneous_motif_samples_its_law(kind):
    law = datasets.contemporaneous_motif(kind=kind, law=True)
    sample = datasets.contemporaneous_motif(20000, kind, seed=0)
    assert sample.shape[1] == law.outcome_length()
    for i in range(sample.shape[1]):
        assert sample[:, i].mean() == pytest.approx(law.marginal([i])[(1,)], abs=0.015)


def test_contemporaneous_motif_kind():
    with pytest.raises(ValueError, match="kind"):
        datasets.contemporaneous_motif(kind="and")


def test_synergy_validation():
    result = synergy_validation(n_samples=2000, n_reps=1, settings=FAST, prng=0)
    assert set(result.data_vars) == {"rate", "ci_low", "ci_high", "successes", "trials"}
    assert result.sizes["check"] == 14 and (result.trials == 1).all()
    assert ((result.ci_low <= result.rate) & (result.rate <= result.ci_high)).all()
    for check in (
        "xor, flip 0.1: both parents",
        "xor among 12 processes: both parents",
        "lag-0 xor: the population family",
    ):
        assert float(result.rate.sel(check=check)) == 1


def test_planted_module_validation():
    background = datasets.noisy_eca(2000, n_cells=4, seed=1)
    result = planted_module_validation(background, modules=("xor", "independent"), n_reps=1, settings=FAST, prng=0)
    assert list(result.module.values) == ["xor", "independent"]
    assert float(result.recovered.sel(module="xor")) == 1
    assert ((result.exact <= result.recovered) & (result.false_parents >= 0)).all()
    assert (result.recovered_ci.sel(bound="low") <= result.recovered).all()
    assert result.attrs["background_processes"] == 4
    with pytest.raises(ValueError, match="module"):
        planted_module_validation(background, modules=("nand",), n_reps=1, settings=FAST)
