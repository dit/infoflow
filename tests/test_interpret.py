"""
Tests for the node layer, edge roles, and latent-confounding flags.
"""

import numpy as np

from infoflow import datasets
from infoflow.network import infer_multiplex
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)


def test_roles_confounded_and_mediated():
    driver = infer_multiplex(datasets.common_driver(3000, seed=0), max_lag=3, skeleton_settings=FAST, prng=0)
    assert driver.dataset["role"].values[0, 1] == "confounded"
    assert driver.dataset["explained_by"].values[0, 1] == "x2@2"
    assert driver.dataset["role"].values[2, 0] == "direct"
    assert "shared-dominant" not in driver.dataset["flags"].values[0, 1]
    chain = infer_multiplex(datasets.chain(3000, seed=0), max_lag=3, skeleton_settings=FAST, prng=0)
    assert chain.dataset["role"].values[0, 2] == "mediated"
    assert chain.dataset["explained_by"].values[0, 2] == "x1@2"


def test_hidden_common_driver_is_flagged():
    """
    A latent driver acting on both at the same lag shows up as zero-lag dependence.
    """
    rng = np.random.default_rng(1)
    n = 3000
    z = rng.integers(0, 2, n)
    x = np.roll(z, 1) ^ (rng.random(n) < 0.1)
    y = (np.roll(z, 1) ^ (rng.random(n) < 0.1)) | (np.roll(x, 1) & (rng.random(n) < 0.3))
    net = infer_multiplex(np.stack([x, y], axis=1), max_lag=2, skeleton_settings=FAST, prng=0)
    assert "zero-lag" in net.dataset["flags"].values[0, 1]


def test_node_layer_storage_and_predictability():
    rng = np.random.default_rng(2)
    n = 3000
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    noise = rng.normal(size=n)
    net = infer_multiplex(np.stack([ar, noise], axis=1), skeleton_settings=FAST, prng=0)
    ds = net.dataset
    assert bool(ds["ais_significant"].values[0]) and ds["ais"].values[0] > 0.2
    assert not bool(ds["ais_significant"].values[1])
    assert "near-unpredictable" in ds["node_flags"].values[1]
    assert ds["wpe"].values[1] > 0.95
    g = net.to_networkx()
    assert g.nodes["x0"]["ais"] > 0.2


def _persistent_drive(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=int)
    for t in range(1, n):
        x[t] = x[t - 1] if rng.random() < 0.85 else 1 - x[t - 1]
    y = np.roll(x, 1) ^ (rng.random(n) < 0.1)
    return np.stack([x, y], axis=1)


def test_reverse_dependence_is_explained_not_flagged():
    """
    A persistent source makes the target's past predict the source (shared flow y -> x);
    the target's own past explains it and x drives y, so it is ``reverse``, not latent.
    """
    net = infer_multiplex(_persistent_drive(), max_lag=2, skeleton_settings=FAST, prng=0)
    ds = net.dataset
    assert ds["role"].values[0, 1] == "direct"
    assert bool(ds["significant"].sel(layer="shared", source="x1", target="x0"))
    assert ds["role"].values[1, 0] == "reverse"
    assert ds["explained_by"].values[1, 0] == "x0 past"
    assert ds["flags"].values[1, 0] == ""
    assert "zero-lag" not in ds["flags"].values[0, 1]


def test_summary_hides_nonsignificant_estimates():
    net = infer_multiplex(_persistent_drive(), max_lag=2, skeleton_settings=FAST, prng=0)
    hidden, shown = net.summary(), net.summary(show_nonsignificant=True)
    row = next(line for line in hidden.splitlines() if line.strip().startswith("x1 -> x0"))
    assert row.split()[-3] == "–"  # intrinsic flow of the reverse edge is not significant
    assert "*" not in hidden and "*" in shown
