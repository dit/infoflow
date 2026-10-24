"""
Tests for regime-dependent inference, context pooling, and the latent-confounding benchmark.
"""

import numpy as np
import pytest
from dit.inference import Trials

from infoflow.datasets import switching_coupling
from infoflow.regimes import add_context, compare_regimes, infer_regimes, infer_with_context, regime_data


def _truth(result, block):
    return np.array([(start // block) % 2 for _, start, _ in result.windows])


def test_infer_regimes_separates_reversed_couplings():
    block = 400
    data = switching_coupling(4800, block=block, seed=0)
    res = infer_regimes(data, n_regimes=2, window=block, prng=0)
    truth = _truth(res, block)
    agree = np.mean(res.labels == truth)
    assert max(agree, 1 - agree) == 1.0
    forward = int(res.labels[truth == 0][0])
    assert res.skeletons[forward][1].parents() == [0]
    assert res.skeletons[1 - forward][0].parents() == [1]
    assert regime_data(res, forward).n_trials == int(np.sum(truth == 0))


def test_infer_regimes_recovers_from_a_bad_start():
    block = 400
    data = switching_coupling(4800, block=block, seed=1)
    windows = len(range(0, 4800 - block + 1, block))
    start = np.array([0] * (windows // 2) + [1] * (windows - windows // 2))
    res = infer_regimes(data, n_regimes=2, window=block, labels=start, prng=1)
    agree = np.mean(res.labels == _truth(res, block))
    assert max(agree, 1 - agree) >= 0.9


def test_infer_regimes_validates_windows():
    with pytest.raises(ValueError):
        infer_regimes(switching_coupling(500, seed=0), n_regimes=3, window=400)


def test_compare_regimes_finds_reversal():
    block = 400
    res = infer_regimes(switching_coupling(4800, block=block, seed=0), window=block, prng=0)
    forward = int(res.labels[0])
    _, _, cmp = compare_regimes(res, a=forward, b=1 - forward, compare_kwargs={"n_perm": 50}, prng=0)
    diff = cmp.dataset["difference"].sel(layer="intrinsic").values
    assert diff[0, 1] > 0.3 and diff[1, 0] < -0.3


def _context_datasets(n=800, k=8, seed=0):
    # With 8 equal-length trials the selection null swaps whole trials, which breaks a
    # dataset-level association; with few trials it shuffles within trials and keeps it.
    rng = np.random.default_rng(seed)
    trials = []
    for c in range(k):
        level = c % 2  # an unobserved dataset-level offset shared by x and y
        x = level ^ (rng.random(n) < 0.3)
        y = level ^ (rng.random(n) < 0.3)
        trials.append(np.stack([x, y], axis=1).astype(np.int64))
    return Trials(trials)


def test_add_context_appends_constant_process():
    data, embeddings, c = add_context(_context_datasets(n=50, k=4), contexts=["a", "b", "a", "b"])
    assert c == 2 and data.names[-1] == "context" and len(embeddings) == 3
    assert data.past_alphabet[c] == 2
    assert [int(p[c, 0]) for p in data.past] == [0, 1, 0, 1]
    assert all(np.all(p[c] == p[c, 0]) for p in data.past)
    with pytest.raises(ValueError):
        add_context(_context_datasets(n=50, k=4), contexts=[0, 1])


def test_context_removes_dataset_confounding():
    from infoflow.network import infer_multiplex

    data = _context_datasets()
    pooled = infer_multiplex(data, interpret=False, node_layer=False, prng=0)
    with_context = infer_with_context(data, interpret=False, node_layer=False, prng=0)
    assert 0 in pooled.skeleton[1].parents()
    assert 0 not in with_context.skeleton[1].parents()
    assert (2, 1) in with_context.skeleton[1].conditionals


def test_latent_validation_baselines():
    pytest.importorskip("tigramite.pcmci")
    from infoflow.benchmarks import latent_validation

    ds = latent_validation(n_nodes=6, n_samples=1500, n_reps=2, methods=("var", "lpcmci"), prng=0)
    assert set(ds.data_vars) >= {"spurious", "flagged", "precision", "recall"}
    assert np.all(ds["n_confounded"].values >= 1)
    assert np.isnan(ds["flagged"].sel(algorithm="var")).all()
    assert np.all((ds["spurious"].values >= 0) & (ds["spurious"].values <= 1))
