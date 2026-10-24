"""
Tests for spike-train input, spike-history encodings, and their automatic choice.
"""

import warnings
from math import factorial

import numpy as np
import pytest
from dit.inference import ordinal_patterns
from hypothesis import given, settings
from hypothesis import strategies as st

import infoflow
from infoflow.data import as_trials, realizations
from infoflow.datasets import spiking_glm
from infoflow.preprocess import (
    ExponentialBins,
    OrdinalIntervals,
    SpikeIntervals,
    discretize,
    preprocess,
    reencode,
)
from infoflow.preprocess.pipeline import _from_fixed, _score
from infoflow.selection import SkeletonSettings
from infoflow.spikes import SpikeTrains, bin_spikes

spike_series = st.lists(st.booleans(), min_size=20, max_size=120).map(lambda b: np.array(b, dtype=np.int64))


def test_bin_spikes_places_and_drops():
    spikes = SpikeTrains([[0.0, 0.012, 0.013, 0.049], [0.051, -0.01]], dt=0.005, t_start=0.0, t_stop=0.05)
    (counts,) = bin_spikes(spikes)
    assert counts.shape == (10, 2)
    assert counts[0, 0] == 1 and counts[2, 0] == 2 and counts[9, 0] == 1
    assert counts[:, 1].sum() == 0


def test_spike_trains_trials_and_defaults():
    trials = [[[0.001, 0.02], [0.03]], [[0.004], [0.0, 0.01, 0.011]]]
    spikes = SpikeTrains(trials, dt=0.005)
    assert spikes.n_trials == 2 and spikes.n_neurons == 2
    binned = as_trials(spikes)
    assert [b.shape for b in binned] == [(7, 2), (4, 2)]
    assert binned[1][:, 1].tolist() == [1, 0, 2, 0]
    single = SpikeTrains([[0.001], [0.002]], dt=0.005)
    assert single.n_trials == 1


def test_bin_spikes_round_trips_bin_centres():
    rng = np.random.default_rng(0)
    counts = (rng.random((500, 3)) < 0.1).astype(np.int64)
    times = [(np.flatnonzero(counts[:, i]) + 0.5) * 0.002 for i in range(3)]
    (binned,) = bin_spikes(SpikeTrains(times, dt=0.002, t_stop=500 * 0.002))
    assert np.array_equal(binned, counts)


def _exponential_brute(x, d, kappa, window):
    edges = ExponentialBins(d, kappa, window).edges()
    out = np.full(len(x), -1)
    for t in range(window - 1, len(x)):
        out[t] = sum(int(x[t - edges[j + 1] + 1 : t - edges[j] + 1].any()) << j for j in range(d))
    return out


@settings(max_examples=60, deadline=None)
@given(x=spike_series, d=st.integers(1, 4), kappa=st.sampled_from([0.0, 0.2, 0.5]), extra=st.integers(0, 12))
def test_exponential_bins_matches_brute_force(x, d, kappa, extra):
    window = d + extra
    enc = ExponentialBins(d, kappa, window).fit([x]).encode(x)
    assert np.array_equal(enc.past, _exponential_brute(x, d, kappa, window))
    assert np.array_equal(enc.present, x > 0)
    assert enc.past_offset == window - 1


@settings(max_examples=40, deadline=None)
@given(d=st.integers(1, 6), kappa=st.sampled_from([0.0, 0.1, 0.3, 0.5]), extra=st.integers(0, 200))
def test_exponential_bin_edges(d, kappa, extra):
    edges = ExponentialBins(d, kappa, d + extra).edges()
    widths = np.diff(edges)
    assert edges[0] == 0 and edges[-1] == d + extra and np.all(widths >= 1)
    if kappa == 0.0:
        assert widths.max() - widths.min() <= 1


def test_uniform_exponential_bins_are_binary_words():
    x = (np.random.default_rng(2).random(300) < 0.3).astype(np.int64)
    enc = ExponentialBins(4, 0.0, 4).fit([x]).encode(x)
    t = np.arange(3, 300)
    word = sum(x[t - j] << j for j in range(4))
    assert np.array_equal(enc.past[3:], word)


def _intervals_brute(x, k, edges_elapsed, edges_isi, bins, W):
    sp = np.flatnonzero(x)
    out = np.full(len(x), -1)
    for t in range(W - 1, len(x)):
        recent = sp[sp <= t]
        if len(recent) < k or t - recent[-k] >= W:
            out[t] = bins**k
            continue
        code = np.searchsorted(edges_elapsed, t - recent[-1], side="right")
        for i in range(1, k):
            code = code * bins + np.searchsorted(edges_isi, recent[-i] - recent[-i - 1], side="right")
        out[t] = code
    return out


@settings(max_examples=60, deadline=None)
@given(x=spike_series, k=st.integers(1, 3), bins=st.integers(2, 4), W=st.integers(4, 30))
def test_spike_intervals_matches_brute_force(x, k, bins, W):
    d = SpikeIntervals(k, bins, W).fit([x])
    enc = d.encode(x)
    assert d.past_alphabet == bins**k + 1
    expected = _intervals_brute(x, k, d.elapsed_edges, d.interval_edges, bins, W)
    assert np.array_equal(enc.past, expected)
    assert enc.past.max() < d.past_alphabet


def test_spike_intervals_elapsed_bins_are_balanced():
    x = (np.random.default_rng(3).random(20000) < 0.05).astype(np.int64)
    d = SpikeIntervals(1, 4, 200).fit([x])
    codes = d.encode(x).past
    counts = np.bincount(codes[(codes >= 0) & (codes < 4)], minlength=4)
    assert counts.min() > 0.15 * counts.sum()


@settings(max_examples=40, deadline=None)
@given(x=spike_series, order=st.integers(2, 4), W=st.integers(6, 40), c=st.integers(2, 4))
def test_ordinal_intervals_invariant_to_stretching_time(x, order, W, c):
    stretched = np.zeros(len(x) * c, dtype=np.int64)
    stretched[np.flatnonzero(x) * c] = 1
    a = OrdinalIntervals(order, W).fit([x]).encode(x).past
    b = OrdinalIntervals(order, W * c).fit([stretched]).encode(stretched).past
    t = np.arange(W, len(x))
    assert np.array_equal(a[t], b[t * c])


def test_ordinal_intervals_matches_dit_patterns():
    x = np.zeros(200, dtype=np.int64)
    x[[103, 110, 112, 120, 133, 135, 160, 164, 165, 190]] = 1
    enc = OrdinalIntervals(3, 100).fit([x]).encode(x)
    sp = np.flatnonzero(x)
    patterns = ordinal_patterns(np.diff(sp).astype(float), 3, 1)
    assert enc.past[199] == patterns[-1]
    assert enc.past[sp[3]] == patterns[0]
    assert enc.past[sp[3] - 1] == factorial(3)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10**6), lags=st.lists(st.integers(1, 12), min_size=1, max_size=3))
def test_spike_realizations_alignment(seed, lags):
    x = (np.random.default_rng(seed).random((150, 3)) < 0.2).astype(np.int64)
    d = discretize(x, [ExponentialBins(3, 0.3, 9), SpikeIntervals(2, 3, 12), OrdinalIntervals(2, 15)])
    variables = [(i % 3, lag) for i, lag in enumerate(lags)]
    r = realizations(d, 0, variables)
    for row in range(len(r)):
        t = r.time[row]
        assert r.present[row] == d.present[0][0, t] == x[t, 0]
        for col, (p, lag) in enumerate(variables):
            assert r.values[row, col] == d.past[0][p, t - lag] >= 0


def test_spike_encodings_reencode():
    x = (np.random.default_rng(4).random((400, 3)) < 0.1).astype(np.int64)
    d = discretize(x, [ExponentialBins(2, 0.0, 6), SpikeIntervals(1, 3, 20), OrdinalIntervals(3, 40)])
    again = reencode(d, [x])
    assert np.array_equal(again.past[0], d.past[0])
    assert np.array_equal(again.present[0], d.present[0])


def test_spike_score_is_history_dependence():
    x = np.zeros(6000, dtype=np.int64)
    x[::5] = 1
    chosen = _from_fixed(ExponentialBins(5, 0.0, 5), (1,))
    mean, *_ = _score(chosen, [x], (2,), 5, [x])
    assert mean > 0.95


def _renewal(n_spikes=1500, seed=0):
    t = np.cumsum(np.random.default_rng(seed).gamma(8, 5, size=n_spikes))
    return SpikeTrains([(np.floor(t) + 0.5) * 0.005], dt=0.005, t_stop=(int(t[-1]) + 1) * 0.005)


def test_renewal_neuron_gets_interval_encoding():
    result = preprocess(_renewal(), prng=0)
    assert result.report.nodes[0].chosen["kind"] == "spike_intervals"
    assert result.report.settings["spikes"]


def test_long_history_neuron_gets_long_window():
    spikes, _ = spiking_glm(n=40000, weights=np.zeros((1, 1)), baseline=0.1, adaptation=1.5, tau_adapt=60, seed=0)
    chosen = preprocess(spikes, prng=0).report.nodes[0].chosen
    window = chosen.get("window", chosen.get("max_interval"))
    assert window >= 30


def test_spike_discretizer_override():
    spikes, _ = spiking_glm(n=4000, seed=1)
    result = preprocess(spikes, discretizer=SpikeIntervals(1, 3, 20), prng=0)
    assert all(n.chosen["kind"] == "spike_intervals" and n.rule == "user" for n in result.report.nodes)
    assert all(isinstance(d, SpikeIntervals) for d in result.data.discretizers)


def test_spikes_flag_on_binned_counts():
    spikes, _ = spiking_glm(n=4000, seed=1)
    (counts,) = bin_spikes(spikes)
    result = preprocess(counts, spikes=True, spike_grid={"source_history": 4}, prng=0)
    assert result.report.settings["spikes"]
    assert all(e.max_lag >= 4 for e in result.embeddings)
    assert all(
        n.chosen["kind"] in ("exponential_bins", "spike_intervals", "ordinal_intervals") for n in result.report.nodes
    )


def test_spiking_glm_network_recovered():
    spikes, edges = spiking_glm(n=20000, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = infoflow.infer_multiplex(
            spikes,
            preprocess={"spike_grid": {"source_history": 8}},
            skeleton_settings=SkeletonSettings(null="strata"),
            interpret=False,
            n_boot=20,
            prng=1,
        )
    found = {(p, t, lag) for t, sk in net.skeleton.items() for p, lag in sk.sources}
    assert found == set(edges)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 2])
def test_spiking_glm_network_recovered_seeds(seed):
    spikes, edges = spiking_glm(n=20000, seed=seed)
    net = infoflow.infer_multiplex(
        spikes,
        preprocess={"spike_grid": {"source_history": 8}},
        skeleton_settings=SkeletonSettings(null="strata"),
        interpret=False,
        n_boot=20,
        prng=seed,
    )
    found = {(p, t, lag) for t, sk in net.skeleton.items() for p, lag in sk.sources}
    assert found == set(edges)
