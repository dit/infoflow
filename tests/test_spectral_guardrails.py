"""Numerical/provenance safeguards for the opt-in spectral discovery route."""

import numpy as np
import pytest

from infoflow import (
    BackboneSettings,
    SpectralSettings,
    backbone_from_rates,
    orient_backbone,
    rates_from_spectrum,
    spectral_rates,
)

from .test_backbone import _rates


def test_extreme_measurement_units_do_not_change_rates():
    f = np.linspace(0, 1, 33)
    raw = np.broadcast_to([[1.0, 0.5], [0.5, 1.0]], (len(f), 2, 2))
    scales = np.array([1e-100, 1e100])
    converted = raw * scales[None, :, None] * scales[None, None, :]
    a, b = rates_from_spectrum(f, raw), rates_from_spectrum(f, converted)
    for name in ("rate", "conditional_rate", "condition_number"):
        assert np.allclose(a.dataset[name], b.dataset[name], equal_nan=True)


def test_shrinkage_cannot_disguise_a_materially_indefinite_spectrum():
    # Every individual pair has valid coherence, but the joint matrix is indefinite.
    s = np.broadcast_to([[1, 0.8, 0.8], [0.8, 1, -0.8], [0.8, -0.8, 1]], (2, 3, 3))
    with pytest.raises(ValueError, match="before regularization"):
        rates_from_spectrum([0, 1], s, settings=SpectralSettings(shrinkage=1))


def test_raw_rank_and_regularized_rank_are_distinct():
    x = np.random.default_rng(8).normal(size=(128, 6))
    r = spectral_rates(x, settings=SpectralSettings(nperseg=128, shrinkage=0.2))
    assert np.max(r.dataset.raw_spectral_rank.values) == 1
    assert np.min(r.dataset.spectral_rank.values) == 6


def test_population_zero_oracle_rejects_empirical_estimates():
    rates = spectral_rates(np.random.default_rng(3).normal(size=(1024, 3)))
    with pytest.raises(ValueError, match="not population zeros"):
        backbone_from_rates(rates, assume_forest=True)
    settings = BackboneSettings()
    object.__setattr__(settings, "dependent", "no")
    with pytest.raises(ValueError, match="boolean"):
        settings.check()


def test_numeric_timescales_cannot_silently_conflict_with_orientation_order():
    rates = _rates([(0, 1)])
    b = backbone_from_rates(rates, assume_forest=True)
    labels = np.ones((3, 3))
    with pytest.raises(ValueError, match="agree"):
        orient_backbone(b, timescale_order=[2, 1, 0], labels=labels)
    b.rates.dataset["timescale"][:] = 1
    with pytest.raises(ValueError, match="strictly ordered"):
        orient_backbone(b, timescale_order=[0, 1, 2], labels=labels)


def test_unresolvable_monte_carlo_floor_does_not_delete_a_candidate():
    from infoflow import infer_backbone

    x = np.random.default_rng(2).normal(size=(1024, 2))
    x[:, 1] += 10 * x[:, 0]
    with pytest.warns(UserWarning, match="budget"):
        b = infer_backbone(
            x,
            settings=BackboneSettings(n_null=3),
            spectral_settings=SpectralSettings(nperseg=64),
            assume_forest=True,
            prng=0,
        )
    assert b.dataset.pvalue.values[0, 1] == 0.25
    assert b.dataset.budget_unresolved.values[0, 1]
    assert b.dataset.status.values[0, 1] == "unresolved"
    assert not b.dataset.detected.values[0, 1]
    assert b.candidate_sources() == {0: [1], 1: [0]}
