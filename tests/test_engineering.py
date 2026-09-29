"""
Tests for network comparison, parallel maps, checkpointing, persistence, IO, and plotting.
"""

import numpy as np
import pytest
from dit.inference import Trials

from infoflow import datasets
from infoflow.checkpoint import Checkpoint
from infoflow.compare import compare_networks
from infoflow.data import DiscreteData
from infoflow.io import export_brainnet, load, load_fieldtrip, load_mat, save
from infoflow.network import MultiplexNetwork, infer_multiplex
from infoflow.parallel import dask_map, thread_map
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=50, n_perm_min_stat=50, n_perm_omnibus=100, n_perm_max_seq=100, n_perm_pairs=50, n_perm_tdmi=100
)


def _coupled_trials(coupling, n_trials, length, seed):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        x = rng.integers(0, 2, length)
        y = np.where(rng.random(length) < coupling, np.roll(x, 1), rng.integers(0, 2, length))
        trials.append(np.stack([x, y], axis=1))
    return DiscreteData.from_discrete(trials)


def _net(data):
    return infer_multiplex(data, max_lag=1, skeleton_settings=FAST, n_boot=0, n_null=20, interpret=False, prng=0)


def test_compare_within_detects_stronger_coupling():
    a, b = _coupled_trials(0.9, 10, 300, seed=0), _coupled_trials(0.5, 10, 300, seed=1)
    result = compare_networks(_net(a), _net(b), a, b, design="within", n_perm=39, prng=0)
    diff = result.dataset["difference"].sel(layer="intrinsic").values[0, 1]
    assert diff > 0.2
    assert bool(result.dataset["significant"].sel(layer="intrinsic").values[0, 1])


def test_compare_within_no_difference():
    a, b = _coupled_trials(0.8, 10, 300, seed=2), _coupled_trials(0.8, 10, 300, seed=3)
    result = compare_networks(_net(a), _net(b), a, b, design="within", n_perm=39, prng=0)
    assert not bool(result.dataset["significant"].sel(layer="intrinsic").values[0, 1])
    assert np.nanmax(np.abs(result.dataset["difference"].values)) < 0.03


def test_compare_between_dependent():
    subjects_a = [_coupled_trials(0.9, 2, 400, seed=s) for s in range(6)]
    subjects_b = [_coupled_trials(0.4, 2, 400, seed=100 + s) for s in range(6)]
    net_a, net_b = _net(subjects_a[0]), _net(subjects_b[0])
    result = compare_networks(net_a, net_b, subjects_a, subjects_b, design="between", dependent=True, n_perm=63, prng=0)
    assert result.dataset["difference"].sel(layer="intrinsic").values[0, 1] > 0.2
    with pytest.raises(ValueError):
        compare_networks(net_a, net_b, subjects_a, subjects_b[:3], design="between", dependent=True)


@pytest.mark.parametrize("make_map", [thread_map, dask_map])
def test_parallel_matches_serial(make_map):
    if make_map is dask_map:
        pytest.importorskip("dask")
    data = datasets.chain(1500, seed=0)
    serial = infer_multiplex(data, max_lag=3, skeleton_settings=FAST, n_boot=20, n_null=20, prng=0)
    parallel = infer_multiplex(data, max_lag=3, skeleton_settings=FAST, n_boot=20, n_null=20, map_fn=make_map(), prng=0)
    assert np.array_equal(serial.dataset["significant"].values, parallel.dataset["significant"].values)
    assert np.allclose(serial.dataset["weight"].values, parallel.dataset["weight"].values, equal_nan=True)


def test_checkpoint_resume(tmp_path):
    data = datasets.chain(1500, seed=1)
    first = infer_multiplex(data, max_lag=3, skeleton_settings=FAST, n_boot=20, n_null=20, checkpoint=tmp_path, prng=0)
    stored = sorted(p.name for p in tmp_path.iterdir())
    assert any(n.startswith("skeleton-") for n in stored) and any(n.startswith("edge-") for n in stored)
    second = infer_multiplex(data, max_lag=3, skeleton_settings=FAST, n_boot=20, n_null=20, checkpoint=tmp_path, prng=0)
    assert np.allclose(first.dataset["weight"].values, second.dataset["weight"].values, equal_nan=True)
    cp = Checkpoint(tmp_path)
    assert "skeleton-0" in cp and cp.get("missing") is None


def test_save_load_and_provenance(tmp_path):
    net = infer_multiplex(datasets.chain(1000, seed=2), max_lag=2, skeleton_settings=FAST, n_boot=10, n_null=10, prng=0)
    path = tmp_path / "net.pkl"
    net.save(path)
    loaded = MultiplexNetwork.load(path)
    assert np.allclose(loaded.dataset["weight"].values, net.dataset["weight"].values, equal_nan=True)
    assert {"infoflow", "dit"} <= set(net.settings["versions"])
    save({"a": 1}, tmp_path / "x.pkl")
    assert load(tmp_path / "x.pkl") == {"a": 1}


def test_mat_and_fieldtrip_io(tmp_path):
    from scipy.io import savemat

    arr = np.random.default_rng(3).normal(size=(3, 50, 4))  # processes, samples, trials
    savemat(tmp_path / "d.mat", {"d": arr})
    trials = load_mat(tmp_path / "d.mat", "d", dim_order="psr")
    assert isinstance(trials, Trials) and len(trials) == 4 and trials[0].shape == (50, 3)
    cells = np.empty(2, dtype=object)
    cells[0], cells[1] = arr[:, :, 0], arr[:, :, 1]
    savemat(tmp_path / "ft.mat", {"data": {"trial": cells, "label": np.array(["a", "b", "c"], dtype=object)}})
    ft, labels = load_fieldtrip(tmp_path / "ft.mat")
    assert labels == ["a", "b", "c"] and ft[1].shape == (50, 3)


def test_brainnet_export(tmp_path):
    net = infer_multiplex(datasets.chain(1500, seed=4), max_lag=3, skeleton_settings=FAST, n_boot=20, n_null=20, prng=0)
    node, edge = export_brainnet(net, "intrinsic", np.zeros((3, 3)), tmp_path / "net")
    assert node.read_text().count("\n") == 3
    weights = np.loadtxt(edge)
    assert weights.shape == (3, 3) and weights[0, 1] > 0


def test_plotting():
    pytest.importorskip("holoviews")
    from infoflow.plot import plot_layer, plot_multiplex

    net = infer_multiplex(datasets.chain(1500, seed=5), max_lag=3, skeleton_settings=FAST, n_boot=20, n_null=20, prng=0)
    assert plot_layer(net, "intrinsic") is not None
    assert len(plot_multiplex(net)) == 3
