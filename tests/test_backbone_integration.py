"""Candidate-guided discovery keeps lagged inference and shared-only layers intact."""

from dataclasses import replace

import numpy as np
import pytest

from infoflow import datasets
from infoflow.backbone import backbone_from_rates
from infoflow.checkpoint import Checkpoint
from infoflow.data import DiscreteData
from infoflow.embedding import Embedding
from infoflow.network import MultiplexNetwork, infer_multiplex
from infoflow.preprocess import EqualFrequency, discretize
from infoflow.preprocess.discretizers import reencode
from infoflow.selection import SkeletonSettings, infer_skeleton
from infoflow.spectral import _fingerprints

from .test_backbone import _rates

FAST = SkeletonSettings(
    n_perm_max_stat=39,
    n_perm_min_stat=39,
    n_perm_omnibus=49,
    n_perm_max_seq=49,
    n_perm_pairs=39,
    n_perm_tdmi=199,
    max_source_lag=2,
)


def _common_backbone(edges=((2, 0), (2, 1))):
    return backbone_from_rates(_rates(edges), assume_forest=True)


def _network(data, backbone=None, **kwargs):
    return infer_multiplex(
        data,
        backbone=backbone,
        max_lag=2,
        skeleton_settings=FAST,
        n_boot=0,
        n_null=99,
        adaptive_resamples=False,
        node_layer=False,
        prng=0,
        **kwargs,
    )


def test_all_source_map_matches_original_selection():
    data = DiscreteData.from_discrete(datasets.chain(1500, seed=1))
    ordinary = infer_skeleton(data, Embedding(2), settings=FAST, prng=0)
    full = {t: [s for s in range(3) if s != t] for t in range(3)}
    guided = infer_skeleton(data, Embedding(2), settings=FAST, prng=0, candidate_sources=full)
    for t in ordinary:
        a, b = ordinary[t], guided[t]
        for key in (
            "sources",
            "target_past",
            "conditionals",
            "omnibus_te",
            "omnibus_pvalue",
            "source_pvalues",
            "significant",
            "tdmi_candidates",
            "pairs",
            "n_samples",
        ):
            assert getattr(a, key) == getattr(b, key)
        assert b.candidate_sources == full[t]


def test_restricted_and_empty_families_do_not_admit_excluded_sources():
    data = DiscreteData.from_discrete(datasets.xor_synergy(2000, seed=3))
    for values in ([], [0]):
        skeleton = infer_skeleton(data, Embedding(1), targets=[2], settings=FAST, prng=0, candidate_sources={2: values})
        sk = skeleton[2]
        assert {p for p, _ in sk.sources} <= set(values)
        assert all(p != 1 for pair in sk.pairs for p, _ in pair if p != 2)
        assert sk.target_past is not None
        if not values:
            assert not sk.parents() and not sk.significant
    for mapping in ({}, {2: [2]}, {2: [-1]}, {2: [3]}, {2: [True]}, {3: []}):
        with pytest.raises(ValueError):
            infer_skeleton(data, targets=[2], settings=FAST, candidate_sources=mapping)


def test_faes_and_forced_conditionals_are_not_restricted_to_backbone_sources():
    data = DiscreteData.from_discrete(datasets.common_driver(1000, seed=4))
    settings = replace(FAST, faes=True, forced_conditionals=((1, 0),), tdmi_screen=False)
    sk = infer_skeleton(data, Embedding(1), targets=[0], settings=settings, prng=1, candidate_sources={0: []})[0]
    assert sk.conditionals == [(1, 0), (2, 0)]


def test_shared_only_nonneighbors_survive_and_layer_identities_hold():
    b = _common_backbone()
    net = _network(datasets.common_driver(3000, seed=1), b)
    ds = net.dataset
    assert net.backbone is b
    assert not ds.backbone_detected.values[0, 1]
    assert not ds.parent_candidate.values[0, 1]
    assert 1 in net.skeleton[0].tdmi_candidates and 0 in net.skeleton[1].tdmi_candidates
    assert ds.kind.values[0, 1] == "shared_candidate"
    assert ds.significant.sel(layer="shared").values[0, 1]
    assert not ds.significant.sel(layer="intrinsic").values[0, 1]
    assert ds.role.values[0, 1] == "confounded"  # actual explanation test still runs
    assert {2} == set(net.skeleton[0].parents()) == set(net.skeleton[1].parents())
    assert not (ds.significant.sel(layer="synergistic").values & (ds.kind.values != "parent")).any()
    for whole, first, second in (("te", "intrinsic", "synergistic"), ("tdmi", "intrinsic", "shared")):
        assert np.allclose(
            ds.weight.sel(layer=whole), ds.weight.sel(layer=first) + ds.weight.sel(layer=second), equal_nan=True
        )
    assert net.settings["discovery"]["candidate_sources"] == b.candidate_sources()
    assert "Spectral backbone" in net.summary()


def test_shared_screen_can_be_explicitly_disabled():
    net = _network(
        datasets.common_driver(1500, seed=0), _common_backbone(), include_shared_candidates=False, interpret=False
    )
    assert net.dataset.kind.values[0, 1] == ""
    assert np.isnan(net.dataset.weight.sel(layer="shared").values[0, 1])


def test_unconfirmed_backbone_edges_are_not_fabricated_parents():
    x = np.random.default_rng(0).integers(0, 2, size=(2000, 3))
    b = _common_backbone()
    net = _network(x, b)
    assert np.array_equal(net.dataset.kind.values == "parent", net.dataset.backbone_te_confirmed.values)
    assert net.dataset.backbone_unconfirmed.values[2, 0]
    assert not net.dataset.significant.sel(layer="intrinsic").values[2, 0]
    assert net.dataset.role.values[0, 1] == "not-tested"
    assert np.isnan(net.dataset.omnibus_pvalue.values[2])


def test_result_save_load_and_names_validation(tmp_path):
    net = _network(datasets.common_driver(1000, seed=0), _common_backbone(), interpret=False)
    net.save(tmp_path / "guided.pkl")
    loaded = MultiplexNetwork.load(tmp_path / "guided.pkl")
    assert loaded.dataset.equals(net.dataset)
    assert loaded.backbone.dataset.equals(net.backbone.dataset)
    assert loaded.settings["discovery"] == net.settings["discovery"]
    with pytest.raises(ValueError, match="names and order"):
        _network(datasets.common_driver(1000, seed=0), _common_backbone(), names=["a", "b", "c"])
    with pytest.raises(TypeError, match="BackboneResult"):
        _network(datasets.chain(100), np.eye(3))


def test_guided_checkpoint_reuse_and_changed_candidates(tmp_path, monkeypatch):
    from infoflow import network

    calls = []
    original = network.estimate_edge

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(network, "estimate_edge", counted)
    data = datasets.common_driver(1000, seed=0)
    b = _common_backbone()
    # A legacy key must never supply a guided skeleton.
    Checkpoint(tmp_path).put("skeleton-0", "not-a-skeleton")
    a = _network(data, b, checkpoint=tmp_path, interpret=False)
    n = len(calls)
    again = _network(data, b, checkpoint=tmp_path, interpret=False)
    assert len(calls) == n and again.dataset.equals(a.dataset)
    other = _common_backbone(edges=((2, 0),))
    _network(data, other, checkpoint=tmp_path, interpret=False)
    assert len(list(tmp_path.glob("guided-*"))) == 2
    assert len(calls) > n
    changed = data.copy()
    changed[0, 0] = 1 - changed[0, 0]
    _network(changed, b, checkpoint=tmp_path, interpret=False)
    assert len(list(tmp_path.glob("guided-*"))) == 3
    _network(data, b, checkpoint=tmp_path, interpret=False, holdout=0.5)
    assert len(list(tmp_path.glob("guided-*"))) == 4


def test_frozen_encoder_recipe_and_exact_reuse_warning():
    rng = np.random.default_rng(7)
    discovery = rng.normal(size=(1000, 3))
    analysis = rng.normal(size=(2000, 3))
    template = discretize(discovery, EqualFrequency(3))
    b = _common_backbone()
    b.rates.data_fingerprints = _fingerprints([discovery])
    symbols = reencode(template, [analysis])
    edges = [d.edges.copy() for d in template.discretizers]
    altered = analysis.copy()
    altered[1000:] += 100
    reencode(template, [altered])
    assert all(np.array_equal(d.edges, e) for d, e in zip(template.discretizers, edges, strict=True))
    assert (
        _network(symbols, b, interpret=False, holdout=0.5).settings["discovery"]["exact_discovery_trials_reused"]
        is False
    )
    with pytest.warns(UserWarning, match="reuses exact discovery"):
        _network(template, b, interpret=False, holdout=0.5)
