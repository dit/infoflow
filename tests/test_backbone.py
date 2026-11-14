"""Spectral graph evidence, null construction and conservative orientation."""

import warnings

import numpy as np
import pytest
from scipy.stats import binom

from infoflow.backbone import (
    BackboneSettings,
    _conditional_null,
    _GaussianSpectrum,
    backbone_from_rates,
    infer_backbone,
    orient_backbone,
    rate_scaling,
)
from infoflow.datasets import multiscale_ou
from infoflow.parallel import thread_map
from infoflow.spectral import SpectralSettings, conditional_coherence, rates_from_spectrum
from infoflow.stats import benjamini_hochberg

from ._spectral import MOTIFS, coupling, ou_spectrum


def _rates(edges, P=3, ratio=3, permutation=None):
    f = np.r_[0, np.geomspace(1e-6, 1000, 2000)]
    tau = ratio ** np.arange(P, dtype=float)
    s = ou_spectrum(f, coupling(edges, P), tau)
    if permutation is not None:
        s, tau = s[:, permutation][:, :, permutation], tau[permutation]
    return rates_from_spectrum(f, s, timescales=tau)


@pytest.mark.parametrize("edges,labels", MOTIFS)
@pytest.mark.parametrize("ratio", [1.2, 5])
def test_population_forest_and_exact_orientation(edges, labels, ratio):
    result = backbone_from_rates(_rates(edges, ratio=ratio), assume_forest=True)
    expected = {tuple(sorted(e)) for e in edges}
    assert {tuple(sorted((int(u[1:]), int(v[1:])))) for u, v in result.to_networkx().edges} == expected
    assert result.is_forest
    assert np.isnan(result.dataset.pvalue).all()
    assert not result.dataset.orientation.values.any()
    matrix = np.zeros((3, 3))
    for (i, j), label in zip(((0, 1), (1, 2), (0, 2)), labels, strict=True):
        matrix[i, j] = matrix[j, i] = label
    oriented = orient_backbone(result, timescale_order=[0, 1, 2], labels=matrix, provenance="model")
    assert set(oriented.to_networkx(directed=True).edges) == {(f"x{s}", f"x{t}") for s, t in edges}
    assert not result.dataset.orientation.values.any()  # no in-place mutation
    candidates = oriented.candidate_sources()
    for s, t in edges:
        assert s in candidates[t] and t in candidates[s]  # structural arrows do not constrain TE directions


def test_disconnected_forest_and_relabeling():
    edges = [(1, 0), (0, 2)]
    perm = [3, 2, 0, 1]
    result = backbone_from_rates(_rates(edges, P=4, permutation=perm), assume_forest=True)
    assert result.candidate_sources()[0] == []
    expected = {(perm.index(s), perm.index(t)) for s, t in edges}
    matrix = np.full((4, 4), np.nan)
    for s, t in expected:
        label = 1 if perm[s] < perm[t] else 2
        matrix[s, t] = matrix[t, s] = label
    order = np.argsort(perm).tolist()
    oriented = orient_backbone(result, timescale_order=order, labels=matrix)
    assert set(oriented.to_networkx(directed=True).edges) == {(f"x{s}", f"x{t}") for s, t in expected}


def test_common_driver_collider_chain_and_ambiguous_mirmo():
    fork = backbone_from_rates(_rates([(1, 0), (1, 2)]), assume_forest=True)
    chain = backbone_from_rates(_rates([(1, 0), (0, 2)]), assume_forest=True)
    assert fork.rates.dataset.rate.values[0, 2] > 0
    assert fork.rates.dataset.conditional_rate.values[0, 2] < 1e-20
    assert chain.rates.dataset.conditional_rate.values[1, 2] < 1e-20
    assert chain.dataset.detected.values[0, 2] and not fork.dataset.detected.values[0, 2]
    collider = backbone_from_rates(_rates([(0, 2), (1, 2)]))
    assert collider.rates.dataset.rate.values[0, 1] == pytest.approx(0, abs=1e-20)
    assert collider.rates.dataset.conditional_rate.values[0, 1] > 0
    assert not collider.dataset.detected.values[0, 1]


def test_forest_acknowledgment_and_loop_guard():
    result = backbone_from_rates(_rates([(0, 1)]))
    with pytest.raises(ValueError, match="assume_forest"):
        result.candidate_sources()
    cyclic = backbone_from_rates(_rates([(0, 1), (1, 2), (0, 2)]), assume_forest=True)
    assert not cyclic.is_forest
    assert cyclic.to_networkx().number_of_edges() == 3  # no forced maximum spanning tree
    assert any("cycles" in note for note in cyclic.warnings)
    with pytest.raises(ValueError, match="cycle"):
        cyclic.candidate_sources()


def test_unknown_and_conflicting_labels_and_input_validation():
    b = backbone_from_rates(_rates([(0, 1)]), assume_forest=True)
    labels = np.full((3, 3), np.nan)
    result = orient_backbone(b, timescale_order=[0, 1, 2], labels=labels)
    assert not result.dataset.orientation.values.any()
    labels[0, 1] = labels[1, 0] = 0
    result = orient_backbone(b, timescale_order=[0, 1, 2], labels=labels)
    assert result.dataset.orientation_status.values[0, 1] == "conflict"
    assert not result.dataset.orientation.values.any()
    for kwargs in ({"timescale_order": [0, 0, 2]}, {"labels": np.ones((2, 2))}, {"provenance": "from_one_rate"}):
        options = dict(timescale_order=[0, 1, 2], labels=labels) | kwargs
        with pytest.raises(ValueError):
            orient_backbone(b, **options)
    labels[0, 1] = 2
    with pytest.raises(ValueError, match="symmetric"):
        orient_backbone(b, timescale_order=[0, 1, 2], labels=labels)


def test_conditional_null_preserves_context_and_removes_only_residual_coupling():
    r = _rates([(0, 1), (1, 2)])
    s = r.dataset.spectrum.values
    for i, j in ((0, 1), (1, 2), (0, 2)):
        null = _conditional_null(s, (i, j))
        for k in range(3):
            assert np.allclose(null[:, k, k], s[:, k, k])
            if k not in (i, j):
                assert np.allclose(null[:, k], s[:, k])
        assert np.max(conditional_coherence(null, [i, j])) < 1e-20
        assert np.linalg.eigvalsh(null).min() > 0


def test_fourier_synthesis_normalization_real_endpoints_and_covariance():
    fs, nfft = 5, 128
    f = np.fft.rfftfreq(nfft, 1 / fs)
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    s = np.broadcast_to(2 * covariance / fs, (len(f), 2, 2)).copy()
    s[[0, -1]] /= 2
    gen = _GaussianSpectrum(f, s, fs, nfft, (10000, 10001))
    a, b = gen.draw(np.random.default_rng(5))
    assert a.shape == (10000, 2) and b.shape == (10001, 2)
    assert np.isrealobj(a)
    assert np.allclose(np.cov(np.concatenate([a, b]).T), covariance, atol=0.05)
    assert abs(np.corrcoef(a[:-1, 0], a[1:, 0])[0, 1]) < 0.03


def test_inference_seeded_parallel_and_joint_pvalues():
    x = multiscale_ou(2048, coupling([(0, 1)], weight=1.4), [1, 2, 4], dt=0.5, seed=9)
    kwargs = dict(
        fs=2,
        spectral_settings=SpectralSettings(nperseg=128),
        settings=BackboneSettings(n_null=39, alpha=0.2),
        assume_forest=True,
        prng=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = infer_backbone(x, **kwargs)
        b = infer_backbone(x, map_fn=thread_map(2), **kwargs)
    assert a.dataset.equals(b.dataset)
    assert np.array_equal(
        a.dataset.pvalue, np.fmax(a.dataset.marginal_pvalue, a.dataset.conditional_pvalue), equal_nan=True
    )
    pairs = [(0, 1), (0, 2), (1, 2)]
    _, q = benjamini_hochberg([a.dataset.pvalue.values[e] for e in pairs], 0.2, dependent=True)
    assert np.allclose([a.dataset.qvalue.values[e] for e in pairs], q)
    assert not a.dataset.orientation.values.any()
    assert "experimental_gaussian_bootstrap" in a.summary()
    for i, j in pairs:
        null_m, null_c = a.nulls[i, j]
        for key, null in (("marginal_pvalue", null_m), ("conditional_pvalue", null_c)):
            value = a.rates.dataset["rate" if key == "marginal_pvalue" else "conditional_rate"].values[i, j]
            expected = (1 + np.count_nonzero(null >= value - 1e-12 * max(1, abs(value)))) / 40
            assert a.dataset[key].values[i, j] == expected


def test_underidentified_results_retain_uncertain_candidates():
    x = np.random.default_rng(0).normal(size=(128, 4))
    with pytest.warns(UserWarning, match="budget"):
        b = infer_backbone(
            x, spectral_settings=SpectralSettings(nperseg=128), settings=BackboneSettings(n_null=3), assume_forest=True
        )
    assert np.isnan(b.dataset.pvalue.values).all()
    assert b.candidate_sources() == {p: [k for k in range(4) if k != p] for p in range(4)}
    assert b.candidate_sources(include_unresolved=False) == {p: [] for p in range(4)}


def test_two_process_nulls_and_settings_validation():
    x = multiscale_ou(2000, np.zeros((2, 2)), [1, 2], dt=0.5, seed=4)
    b = infer_backbone(x, fs=2, spectral_settings=SpectralSettings(nperseg=128), settings=BackboneSettings(n_null=39))
    assert b.dataset.marginal_pvalue.values[0, 1] == b.dataset.conditional_pvalue.values[0, 1]
    assert not b.dataset.detected.values.any()
    for settings in (BackboneSettings(n_null=0), BackboneSettings(alpha=1)):
        with pytest.raises(ValueError):
            infer_backbone(x, settings=settings)


def test_scaling_is_not_automatic_orientation():
    ratios = [2, 5, 10, 20]
    results = [_rates([(0, 1)], P=2, ratio=r) for r in ratios]
    ds = rate_scaling(results, ratios, comparable=True)
    assert ds.log_slope.values[0, 1] > 0  # finite positive growth still converges (S23)
    assert np.isnan(ds.label.values).all()
    assert np.isnan(rate_scaling(results[:1], ratios[:1]).log_slope.values).all()
    with pytest.raises(ValueError, match="comparable"):
        rate_scaling(results, ratios)
    with pytest.raises(ValueError, match="hierarchy"):
        rate_scaling(results, [2, 4, 8, 16], comparable=True)


@pytest.mark.slow
def test_bounded_null_calibration():
    rejections = []
    for seed in range(40):
        x = multiscale_ou(4096, np.zeros((2, 2)), [1, 3], dt=0.5, seed=seed)
        b = infer_backbone(
            x,
            fs=2,
            spectral_settings=SpectralSettings(nperseg=128),
            settings=BackboneSettings(n_null=99, alpha=0.1),
            prng=seed,
        )
        rejections.append(bool(b.dataset.detected.values[0, 1]))
    assert sum(rejections) <= binom.ppf(0.99, len(rejections), 0.1)
