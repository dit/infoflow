"""
The machine layer: ε-machines per node, ε-transducers per edge and target, and the composed model.
"""

import pickle

import numpy as np
import pytest

sofic_inference = pytest.importorskip("sofic.inference")
if not hasattr(sofic_inference, "Trials"):
    pytest.skip("needs sofic with multi-trial CSSR (sofic.inference.Trials)", allow_module_level=True)

from sofic.examples.epsilon_machines import even_process, golden_mean  # noqa: E402
from sofic.generators.sampling import sample  # noqa: E402

import infoflow  # noqa: E402
from infoflow import datasets  # noqa: E402
from infoflow.data import DiscreteData  # noqa: E402
from infoflow.machines import MachineSettings, infer_machines  # noqa: E402
from infoflow.network import infer_multiplex  # noqa: E402
from infoflow.selection import SkeletonSettings  # noqa: E402

from ._generators import exact_flows, mixture, sample_pairs  # noqa: E402

FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)
QUICK = dict(skeleton_settings=FAST, n_boot=50, n_null=50, node_layer=False, interpret=False, prng=0)


def _h2(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _symbols(machine, n, seed):
    observations, _ = sample(machine, n, np.random.default_rng(seed))
    return np.array([int(s) for s in observations], dtype=np.int64)


def _delayed_copy(n, delay, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    y = np.roll(x, delay) ^ (rng.random(n) < noise)
    return np.stack([x, y], axis=1)


def test_known_processes_per_node():
    period = np.tile([0, 0, 1], 4000)[:12000]
    data = np.stack([_symbols(golden_mean(), 12000, 0), _symbols(even_process(), 12000, 1), period], axis=1)
    net = infer_multiplex(data, max_lag=3, machines=True, **QUICK)
    nodes = net.machines.nodes
    assert len(list(nodes["x0"].states())) == 2
    assert len(list(nodes["x1"].states())) == 2
    assert len(list(nodes["x2"].states())) == 3
    ds = net.machines.dataset
    assert float(ds["h_mu"].sel(node="x0")) == pytest.approx(2 / 3, abs=0.02)
    assert float(ds["h_mu"].sel(node="x2")) == pytest.approx(0.0, abs=1e-9)
    assert float(ds["C_mu"].sel(node="x2")) == pytest.approx(np.log2(3), abs=1e-6)
    assert np.all(ds["stability"].values == 1.0)


def test_trials_are_pooled_without_crossing_boundaries():
    seq = _symbols(even_process(), 12000, 2)
    whole = DiscreteData.from_discrete(np.stack([seq, seq[::-1]], axis=1))
    split = DiscreteData.from_discrete([whole.present[0][:, :6000].T, whole.present[0][:, 6000:].T])
    net = infer_multiplex(split, max_lag=3, **QUICK)
    layer = infer_machines(split, net)
    assert len(list(layer.nodes["x0"].states())) == 2
    assert int(layer.dataset["n_symbols"].sel(node="x0")) == 12000


def test_delay_channel_is_a_memoryless_transducer_on_the_shifted_input():
    """
    The input is the source shifted by the edge's delay, so a delayed noisy copy is memoryless.
    """
    net = infer_multiplex(_delayed_copy(8000, 2, 0.1, 0), max_lag=3, machines=True, **QUICK)
    assert int(net.dataset["delay"].sel(source="x0", target="x1")) == 2
    m = net.machines
    assert len(list(m.edges[("x0", "x1")].states())) == 1
    edge = m.dataset.sel(source="x0", target="x1")
    assert float(edge["te_model"]) == pytest.approx(1 - _h2(0.1), abs=0.02)
    assert abs(float(edge["te_gap"])) < 0.02
    assert edge["admitted_by"].item() == "plugin"
    assert m.model.roles == {"x0": "source", "x1": "driven"}
    assert float(m.dataset["model_te"].sel(source="x0", target="x1")) == pytest.approx(1 - _h2(0.1), abs=0.02)


def _trapdoor(n, seed):
    """
    The trapdoor channel driven by fair coins, with delay 1: x0 is the input, x1 the output.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    y = np.zeros(n, dtype=np.int64)
    ball = 0
    for t in range(1, n):
        y[t] = x[t - 1] if x[t - 1] == ball else rng.integers(0, 2)
        ball ^= x[t - 1] ^ y[t]
    return np.stack([x, y], axis=1)


def test_memoryless_edge_capacity_is_unchanged_by_feedback():
    """
    A one-state transducer is the edge's discrete memoryless channel, so both capacities equal the channel layer's.
    """
    data = DiscreteData.from_discrete(_delayed_copy(8000, 1, 0.1, 9))
    net = infer_multiplex(data, max_lag=2, channels=True, **QUICK)
    edge = infer_machines(data, net, MachineSettings(stability=False)).dataset.sel(source="x0", target="x1")
    dmc = float(net.channels.dataset["capacity"].sel(source="x0", target="x1"))
    assert dmc == pytest.approx(1 - _h2(0.1), abs=0.03)
    for name in ("capacity", "feedback_capacity"):
        assert float(edge[f"{name}_low"]) == pytest.approx(dmc, abs=1e-3)
        assert float(edge[f"{name}_high"]) == pytest.approx(dmc, abs=1e-3)


def test_trapdoor_edge_gains_capacity_from_feedback():
    """
    The trapdoor channel's feedback capacity is log2 of the golden ratio (Permuter et al. 2008); without
    feedback its capacity is below 0.6 (Huleihel et al. 2021).
    """
    data = DiscreteData.from_discrete(_trapdoor(20000, 0))
    net = infer_multiplex(data, max_lag=1, **QUICK)
    layer = infer_machines(data, net, MachineSettings(stability=False, exact_flows=False))
    assert len(list(layer.edges[("x0", "x1")].states())) == 2
    edge = layer.dataset.sel(source="x0", target="x1")
    golden = np.log2((1 + np.sqrt(5)) / 2)
    assert float(edge["feedback_capacity_low"]) == pytest.approx(golden, abs=0.01)
    assert float(edge["feedback_capacity_high"]) == pytest.approx(golden, abs=0.02)
    assert 0.55 < float(edge["capacity_low"]) <= float(edge["capacity_high"]) < float(edge["feedback_capacity_low"])
    assert "fb_capacity" in layer.summary()


def test_transducer_capacity_drops_transient_states():
    from sofic.examples.processes import trapdoor_et

    from infoflow.machines import transducer_capacity

    channel = trapdoor_et()
    padded = channel.copy()
    recurrent = sorted(channel.states(), key=repr)
    for x in channel.alphabets()[0]:
        for y in channel.alphabets()[1]:
            padded.add_transition("transient", recurrent[0], x, y, prob=0.5)
    assert len(list(padded.states())) == 3
    options = dict(max_order=1)
    assert transducer_capacity(padded, feedback=True, **options).upper == pytest.approx(
        transducer_capacity(channel, feedback=True, **options).upper, abs=1e-6
    )


def test_capacity_can_be_switched_off():
    data = DiscreteData.from_discrete(_delayed_copy(4000, 1, 0.1, 10))
    net = infer_multiplex(data, max_lag=2, **QUICK)
    layer = infer_machines(data, net, MachineSettings(stability=False, capacity=False))
    assert "capacity_low" not in layer.dataset
    assert "–" in layer.summary()


def test_xor_shows_only_in_the_target_transducer():
    net = infer_multiplex(datasets.xor_synergy(8000, noise=0.05, seed=1), max_lag=2, machines=True, **QUICK)
    m = net.machines
    assert {("x0", "x2"), ("x1", "x2")} <= set(m.edges)
    for src in ("x0", "x1"):
        assert float(m.dataset["te_model"].sel(source=src, target="x2")) < 0.02
    assert m.inputs["x2"]["parents"] == [("x0", 1), ("x1", 1)]
    assert float(m.dataset["target_te"].sel(node="x2")) == pytest.approx(1 - _h2(0.05), abs=0.03)
    loss = m.model.log_loss(DiscreteData.from_discrete(datasets.xor_synergy(4000, noise=0.05, seed=2)))
    assert float(loss["improvement"].sel(node="x2")) == pytest.approx(1 - _h2(0.05), abs=0.04)


@pytest.mark.parametrize("case", [dict(a_copy=0.8), dict(a_copy=0.4, a_xor=0.3, a_hold=0.2)])
def test_transducer_te_matches_exact_flows(case):
    generator = mixture(**case)
    exact = exact_flows(generator)
    data = DiscreteData.from_discrete(sample_pairs(generator, 20000, seed=0))
    net = infer_multiplex(data, max_lag=1, **QUICK)
    layer = infer_machines(data, net, MachineSettings(stability=False))
    assert float(layer.dataset["te_model"].sel(source="x0", target="x1")) == pytest.approx(exact["te"], abs=0.03)
    model_flows = layer.dataset.sel(source="x0", target="x1")
    assert float(model_flows["model_te"]) == pytest.approx(exact["te"], abs=0.03)
    assert float(model_flows["model_intrinsic"]) == pytest.approx(exact["intrinsic"], abs=0.05)


def test_simulation_reproduces_the_channel_and_passes_adequacy():
    data = DiscreteData.from_discrete(_delayed_copy(6000, 1, 0.1, 3))
    net = infer_multiplex(data, max_lag=2, **QUICK)
    model = infer_machines(data, net, MachineSettings(stability=False)).model
    sim = model.simulate(5000, seed=0)
    assert np.mean(sim[1:, 1] == sim[:-1, 0]) == pytest.approx(0.9, abs=0.02)
    held = DiscreteData.from_discrete(_delayed_copy(3000, 1, 0.1, 4))
    adequacy = model.adequacy(held, n_sim=39, seed=0)
    assert not adequacy["flagged"].any()
    with pytest.raises(ValueError, match="cannot reach alpha"):
        model.adequacy(held, n_sim=19, seed=0)


def test_to_hmm_guard():
    data = DiscreteData.from_discrete(_delayed_copy(4000, 1, 0.1, 5))
    net = infer_multiplex(data, max_lag=2, **QUICK)
    layer = infer_machines(data, net, MachineSettings(stability=False, max_composed_states=1))
    with pytest.raises(ValueError, match="max_composed_states"):
        layer.model.to_hmm()
    assert ("model", "exact_flows") in layer.skipped
    assert layer.model.simulate(10, seed=0).shape == (10, 2)


def test_joint_alphabet_guard_and_coarse_inputs():
    rng = np.random.default_rng(6)
    x = rng.integers(0, 4, (8000, 2))
    y = (np.roll(x[:, 0], 1) + np.roll(x[:, 1], 1) >= 4).astype(np.int64)
    data = np.column_stack([x, y])
    net = infer_multiplex(data, max_lag=1, **QUICK)
    small = infer_machines(data, net, MachineSettings(stability=False, max_joint_alphabet=8))
    assert ("target", "x2") in small.skipped
    assert small.model is None and ("model", "network") in small.skipped
    coarse = infer_machines(data, net, MachineSettings(stability=False, max_joint_alphabet=8, input_symbols="coarse"))
    assert "x2" in coarse.targets
    assert coarse.targets["x2"].alphabets()[0] <= {(a, b) for a in (0, 1) for b in (0, 1)}


def test_pickle_round_trip_rebuilds_the_model(tmp_path):
    net = infer_multiplex(_delayed_copy(4000, 1, 0.1, 7), max_lag=2, machines=True, **QUICK)
    path = tmp_path / "net.pkl"
    net.save(path)
    loaded = infoflow.MultiplexNetwork.load(path)
    assert set(loaded.machines.edges) == {("x0", "x1")}
    assert loaded.machines.model.roles == net.machines.model.roles
    assert pickle.loads(pickle.dumps(loaded.machines)).model is not None
    assert "states" in loaded.summary()


def test_infer_passes_machines_through_and_reports_them():
    res = infoflow.infer(_delayed_copy(4000, 1, 0.1, 8), preset="fast", machines=True, time_budget="1h", prng=0)
    assert res.network.machines is not None
    assert res.report.machines["edges"] == 1
    assert any("excludes the machine layer" in w for w in res.report.warnings)
    assert "machines:" in str(res.report)
