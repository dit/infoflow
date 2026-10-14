"""
Directed information per edge: transfer entropy plus instantaneous information exchange.
"""

import numpy as np
import pytest

from infoflow.contemporaneous import instantaneous_exchange
from infoflow.network import infer_multiplex
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=100, n_perm_min_stat=100, n_perm_omnibus=200, n_perm_max_seq=200, n_perm_pairs=50
)
QUICK = dict(max_lag=2, skeleton_settings=FAST, n_boot=50, n_null=50, node_layer=False, interpret=False, prng=0)


def _lagged_copy(n, instantaneous, seed=0):
    """
    ``y_t = x_{t-1} xor z_t``: ``z_t`` copies ``x_t`` with probability `instantaneous`,
    else is a 0.1-coin, so ``y_t`` depends on ``x_t`` within the same step.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    z = np.where(rng.random(n) < instantaneous, x, rng.random(n) < 0.1).astype(np.int64)
    y = np.roll(x, 1) ^ z
    return np.stack([x, y], axis=1)


def test_instantaneous_coupling_is_exchange_not_transfer():
    net = infer_multiplex(_lagged_copy(8000, 0.3), directed_information=True, **QUICK)
    ds = net.dataset
    te = float(ds["weight"].sel(layer="te", source="x0", target="x1"))
    iie = float(ds["iie"].sel(source="x0", target="x1"))
    assert te > 0.05
    assert iie > 0.02
    assert float(ds["iie_pvalue"].sel(source="x0", target="x1")) < 0.05
    assert float(ds["iie"].sel(source="x1", target="x0")) == pytest.approx(iie)  # symmetric
    assert float(ds["di"].sel(source="x0", target="x1")) == pytest.approx(te + iie)
    assert net.settings["directed_information"] is True


def test_without_instantaneous_coupling_di_is_te():
    net = infer_multiplex(_lagged_copy(8000, 0.0), directed_information=True, **QUICK)
    ds = net.dataset
    te = float(ds["weight"].sel(layer="te", source="x0", target="x1"))
    assert float(ds["iie"].sel(source="x0", target="x1")) < 0.005
    assert float(ds["di"].sel(source="x0", target="x1")) == pytest.approx(te, abs=0.005)
    assert float(ds["di"].sel(source="x1", target="x0")) < 0.01  # the reverse direction carries none
    assert np.isnan(float(ds["di"].sel(source="x0", target="x0")))  # never estimated


def test_exchange_matches_the_lag0_layer():
    net = infer_multiplex(_lagged_copy(4000, 0.3), directed_information=True, contemporaneous=True, **QUICK)
    assert np.allclose(net.dataset["iie"].values, net.dataset["contemporaneous_cmi"].values)
    from infoflow.data import DiscreteData

    values, pvalues = instantaneous_exchange(DiscreteData.from_discrete(_lagged_copy(4000, 0.3)), net.skeleton, prng=0)
    assert np.allclose(values, values.T) and np.all(np.diag(pvalues) == 1)


def test_model_di_is_te_for_a_lagged_channel():
    if not hasattr(pytest.importorskip("sofic.inference"), "Trials"):
        pytest.skip("needs sofic with multi-trial CSSR (sofic.inference.Trials)")
    from infoflow.machines import MachineSettings

    net = infer_multiplex(
        _lagged_copy(6000, 0.0), machines=True, machine_settings=MachineSettings(stability=False), **QUICK
    )
    edge = net.machines.dataset.sel(source="x0", target="x1")
    assert float(edge["model_di"]) == pytest.approx(float(edge["model_te"]), abs=1e-9)
