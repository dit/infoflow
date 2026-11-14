"""Spectral normalization, numerical validity, and analytic process oracles."""

from dataclasses import replace

import numpy as np
import pytest
from dit.inference import Trials
from scipy.signal import csd

from infoflow.data import DiscreteData
from infoflow.datasets import multiscale_ou
from infoflow.spectral import SpectralSettings, conditional_coherence, rates_from_spectrum, spectral_rates

from ._spectral import MOTIFS, coupling, ou_spectrum, sampled_spectrum


def test_white_noise_normalization_and_subset_conditioning():
    fs, rho = 10, 0.6
    f = np.linspace(0, fs / 2, 129)
    s = np.broadcast_to([[1, rho], [rho, 1]], (len(f), 2, 2))
    r = rates_from_spectrum(f, s)
    truth = -0.5 * np.log2(1 - rho**2) * fs
    assert r.dataset.rate.values[0, 1] == pytest.approx(truth)
    assert r.dataset.conditional_rate.values[0, 1] == pytest.approx(truth)
    assert np.isnan(np.diag(r.dataset.rate)).all()
    assert np.allclose(conditional_coherence(s, [0, 1], []), rho**2)
    assert np.allclose(conditional_coherence(s, [0, 1]), rho**2)
    independent = rates_from_spectrum(f, np.broadcast_to(np.eye(3), (len(f), 3, 3)))
    assert independent.dataset.rate.values[0, 1] == 0
    assert independent.dataset.conditional_rate.values[0, 1] == 0


@pytest.mark.parametrize("nfft", [127, 128, 256])
def test_welch_matches_scipy_and_trial_weighting(nfft):
    trials = [np.random.default_rng(k).normal(size=(n, 3)) for k, n in enumerate((1000, 601))]
    settings = SpectralSettings(nperseg=127, nfft=nfft, noverlap=43)
    result = spectral_rates(Trials(trials), fs=7, settings=settings)
    counts = [(len(t) - 127) // (127 - 43) + 1 for t in trials]
    f, cross = csd(trials[0][:, 0], trials[0][:, 1], fs=7, nperseg=127, nfft=nfft, noverlap=43)
    _, cross2 = csd(trials[1][:, 0], trials[1][:, 1], fs=7, nperseg=127, nfft=nfft, noverlap=43)
    expected = (counts[0] * cross + counts[1] * cross2) / sum(counts)
    assert np.allclose(result.dataset.frequency, f)
    assert np.allclose(result.dataset.spectrum.values[:, 0, 1], expected.conj())
    assert result.dataset.segment_count.values.tolist() == counts
    assert result.settings["n_segments"] == sum(counts)
    assert result.trial_lengths == (1000, 601)


def test_measurement_units_and_process_permutation():
    x = np.random.default_rng(5).normal(size=(5000, 3))
    x[:, 1] += 0.7 * x[:, 0]
    settings = SpectralSettings(nperseg=128)
    a = spectral_rates(x, fs=2, timescales=[1, 5, 10], names=["A", "B", "C"], settings=settings)
    b = spectral_rates(x * [1e-6, 1e6, -3], fs=20, timescales=[0.1, 0.5, 1], settings=settings)
    for key in ("rate", "conditional_rate"):
        assert np.allclose(b.dataset[key], 10 * a.dataset[key], equal_nan=True)
        assert np.allclose(b.dataset[f"rescaled_{key}"], a.dataset[f"rescaled_{key}"], equal_nan=True)
    order = [2, 0, 1]
    c = spectral_rates(x[:, order], fs=2, settings=settings)
    assert np.allclose(c.dataset.conditional_rate, a.dataset.conditional_rate.values[order][:, order], equal_nan=True)
    assert "bits/time" in a.summary()


def test_underidentified_conditioning_and_explicit_shrinkage():
    x = np.random.default_rng(2).normal(size=(256, 5))
    s = SpectralSettings(nperseg=256, noverlap=0)
    result = spectral_rates(x, settings=s)
    assert np.isnan(result.dataset.conditional_rate.values).all()
    assert not result.dataset.conditional_valid.values.any()
    regularized = spectral_rates(x, settings=replace(s, shrinkage=0.2))
    assert regularized.dataset.conditional_valid.values.all()
    assert np.isfinite(regularized.dataset.conditional_rate.values[0, 1])
    assert regularized.dataset.attrs["shrinkage_coherence_change"] > 0
    # More FFT bins are not more spectral replicates.
    padded = spectral_rates(x, settings=replace(s, nfft=512))
    assert np.isnan(padded.dataset.conditional_rate.values).all()


def test_perfect_coherence_is_not_silently_capped():
    f = np.linspace(0, 1, 33)
    s = np.ones((len(f), 2, 2))
    result = rates_from_spectrum(f, s)
    assert np.isinf(result.dataset.rate.values[0, 1])
    assert np.isnan(result.dataset.conditional_rate.values[0, 1])


@pytest.mark.parametrize("fast_to_slow", [False, True])
@pytest.mark.parametrize("ratio", [2, 5, 20])
@pytest.mark.parametrize("a", [1, -0.6])
def test_two_node_closed_forms(fast_to_slow, ratio, a):
    tau, D = np.array([1.0, ratio]), np.array([0.7, 1.3])
    A = np.zeros((2, 2))
    sender, receiver = (0, 1) if fast_to_slow else (1, 0)
    A[receiver, sender] = a
    f = np.r_[0, np.geomspace(1e-8, 1e5, 10000)]
    result = rates_from_spectrum(f, ou_spectrum(f, A, tau, D), timescales=tau)
    c = a**2 * D[sender] / D[receiver]
    # Paper S23/S36 use natural logarithms.
    expected = (
        ratio / 2 * np.expm1(0.5 * np.log1p(c / ratio)) if fast_to_slow else np.expm1(0.5 * np.log1p(c * ratio)) / 2
    )
    actual = result.dataset.rescaled_rate.values[0, 1]
    assert actual * np.log(2) == pytest.approx(expected, rel=3e-5)
    assert np.allclose(result.dataset.rate, result.dataset.conditional_rate, equal_nan=True)


@pytest.mark.parametrize("edges,labels", MOTIFS)
def test_schur_precision_and_finite_ratio_patterns(edges, labels):
    f = np.r_[0, np.geomspace(1e-6, 100, 2000)]
    spectrum = ou_spectrum(f, coupling(edges, weight=-0.8), [1, 3, 9], [0.5, 1, 2])
    ds = rates_from_spectrum(f, spectrum).dataset
    for i, j in ((0, 1), (1, 2), (0, 2)):
        schur = conditional_coherence(spectrum, [i, j])
        assert np.allclose(schur, ds.partial_coherence.values[:, i, j], atol=1e-12)
        linked = ds.rate.values[i, j] > 1e-10 and ds.conditional_rate.values[i, j] > 1e-10
        assert bool(linked) == ((i, j) in edges or (j, i) in edges)


def test_ou_sampling_reference_and_stationarity():
    A, tau, dt = coupling([(0, 1)], P=2, weight=1.2), [1.0, 5.0], 0.1
    x = multiscale_ou(1000000, A, tau, dt=dt, seed=6)
    r = spectral_rates(x, fs=1 / dt, settings=SpectralSettings(nperseg=1024))
    f = r.dataset.frequency.values
    truth = rates_from_spectrum(f, sampled_spectrum(f, A, tau, dt)).dataset.rate.values[0, 1]
    assert r.dataset.rate.values[0, 1] == pytest.approx(truth, abs=0.012)
    assert np.allclose(x.var(axis=0), [1.0, 1.24], atol=0.15)
    assert np.array_equal(multiscale_ou(100, A, tau, seed=3), multiscale_ou(100, A, tau, seed=3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nperseg": 3},
        {"nperseg": 4.5},
        {"noverlap": 256},
        {"nfft": 32},
        {"detrend": "linear"},
        {"shrinkage": -1},
        {"condition_limit": 1},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        spectral_rates(np.ones((512, 2)), settings=SpectralSettings(**kwargs))


@pytest.mark.parametrize(
    "data",
    [
        np.ones((512, 2)),
        np.full((512, 2), np.nan),
        np.ones((512, 2), dtype=complex),
        [np.ones((512, 2)), np.ones((512, 3))],
    ],
)
def test_invalid_observations(data):
    with pytest.raises(ValueError):
        spectral_rates(data)


def test_invalid_spectra_names_and_parameters():
    x = np.random.default_rng(1).normal(size=(512, 2))
    for kwargs in ({"fs": 0}, {"fs": np.inf}, {"names": ["a", "a"]}, {"timescales": [1, -1]}):
        with pytest.raises(ValueError):
            spectral_rates(x, **kwargs)
    with pytest.raises(ValueError, match="nperseg"):
        spectral_rates(x[:10])
    with pytest.raises(ValueError, match="symbol codes"):
        spectral_rates(DiscreteData.from_discrete(np.zeros((10, 2), dtype=int)))
    f = [0, 1]
    for s in (np.array([[[1, 2], [2, 1]]] * 2), np.array([[[1, 0.5j], [0, 1]]] * 2)):
        with pytest.raises(ValueError):
            rates_from_spectrum(f, s)
    with pytest.raises(ValueError):
        rates_from_spectrum([1, 0], np.broadcast_to(np.eye(2), (2, 2, 2)))
    for pair, cond in (([0, 0], None), ([0, 2], None), ([0, 1], [0])):
        with pytest.raises(ValueError):
            conditional_coherence(np.broadcast_to(np.eye(2), (2, 2, 2)), pair, cond)
    for kwargs in ({"dt": 0}, {"timescales": [1, 0]}, {"coupling": [[0, 2], [2, 0]]}):
        params = dict(n=10, coupling=np.zeros((2, 2)), timescales=[1, 5]) | kwargs
        with pytest.raises(ValueError):
            multiscale_ou(**params)
