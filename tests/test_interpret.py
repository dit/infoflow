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


def test_orthogonal_features_of_one_driver_share_no_flow():
    """
    x1 reads the sign of an i.i.d. Gaussian x0 and x2 reads whether |x0| is large.
    Sign and magnitude are independent, so both edges from x0 are intrinsic and
    x1, x2 exchange no flow of any kind despite sharing a driver.
    """
    from infoflow.data import realizations
    from infoflow.layers import layer_statistics
    from infoflow.preprocess import preprocess

    raw = datasets.orthogonal_features(4000, seed=0)
    net = infer_multiplex(raw, max_lag=3, skeleton_settings=FAST, prng=0)
    ds = net.dataset
    # The driver has no dynamics of its own, so it keeps bins (sign and magnitude both survive).
    assert "unpredictable" in net.report.nodes[0].flags
    intrinsic = ds["significant"].sel(layer="intrinsic").values
    assert intrinsic[0, 1] and intrinsic[0, 2]
    assert not ds["significant"].values[:, [1, 2], [2, 1]].any()
    assert net.skeleton[0].sources == []

    # Estimate x1 -> x2 anyway, in x2's full context: no shared (or any) flow.
    data = preprocess(raw, max_lag=3, prng=0).data
    r = realizations(data, 2, [(1, 1), (2, 1), (0, 1)])
    stats = layer_statistics(r.present, r.columns([(1, 1)]), r.columns([(2, 1), (0, 1)]), n_boot=50, n_null=99, prng=0)
    assert stats.pvalue["shared"] > 0.05
    assert stats.flow.shared < 0.01 and stats.flow.tdmi < 0.01


def test_ordinal_present_does_not_leak_to_children():
    """
    A present symbol defined relative to past values would let x0's children (which saw
    those values) predict it; the ordinal present is the bin of x_t alone, so x0 has no parents.
    """
    from infoflow.preprocess import EqualFrequency, Ordinal

    net = infer_multiplex(
        datasets.orthogonal_features(4000, seed=0),
        max_lag=3,
        skeleton_settings=FAST,
        preprocess={"discretizer": [Ordinal(2, 2), EqualFrequency(6), EqualFrequency(6)]},
        prng=0,
    )
    assert net.skeleton[0].sources == []
