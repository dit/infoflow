"""
Coupled tent maps on the ring A -> B, B -> C, B -> D, C -> A, D -> A, seen through the
generating partition at 1/2.

The slopes are Misiurewicz points whose exact generating-partition ε-machines have
4, 5, 6, and 7 states; the slopes and the exact machines come from chaos_anatomy (a
local checkout: ``uv pip install -e ../chaos_anatomy``), and the module is skipped
without it. The five regimes were located by Lyapunov and periodicity sweeps of this
topology: uncoupled, weak (0.1), moderate (0.3), synchronized (0.5 with identical
slopes), and a period-12 window.
"""

import functools
import warnings

import numpy as np
import pytest

pytest.importorskip("chaos_anatomy")
sofic_inference = pytest.importorskip("sofic.inference")
if not hasattr(sofic_inference, "Trials"):
    pytest.skip("needs sofic with multi-trial CSSR (sofic.inference.Trials)", allow_module_level=True)

from chaos_anatomy.anatomy import forward_causal_graph  # noqa: E402
from chaos_anatomy.misiurewicz import misiurewicz_parameters  # noqa: E402
from chaos_anatomy.partitions import build_model  # noqa: E402
from sofic.generators.epsilon_machine import EpsilonMachine  # noqa: E402
from sofic.graph import ATTR_EMISSION, ATTR_PROB  # noqa: E402
from sofic.inference import topology_key  # noqa: E402

import infoflow  # noqa: E402
from infoflow.datasets import coupled_tent, tent_ring_coupling  # noqa: E402
from infoflow.machines import MachineSettings  # noqa: E402
from infoflow.network import infer_multiplex  # noqa: E402
from infoflow.selection import SkeletonSettings  # noqa: E402

pytestmark = pytest.mark.slow

NAMES = ["A", "B", "C", "D"]
# Roots of a^3 - 2a - 2, a^3 - a^2 - 2, a^4 - a^3 - 2a - 2, a^4 - a^3 - a^2 - 2.
PLANNED = {"A": 1.76929235, "B": 1.69562077, "C": 1.87370856, "D": 1.85356028}
EXACT_STATES = {"A": 4, "B": 5, "C": 6, "D": 7}
TRUE_EDGES = {("A", "B"), ("B", "C"), ("B", "D"), ("C", "A"), ("D", "A")}
N_TRAIN, N_HELD = 20000, 10000
FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)


@functools.cache
def slopes():
    """
    The chaos_anatomy Misiurewicz parameters nearest the planned slopes.
    """
    params = [a for a, _ in misiurewicz_parameters(160, a_min=np.sqrt(2), a_max=2.0)]
    found = {name: min(params, key=lambda a, t=target: abs(a - t)) for name, target in PLANNED.items()}
    assert all(abs(found[n] - PLANNED[n]) < 1e-7 for n in NAMES), found
    return found


@functools.cache
def exact_machine(a):
    """
    The exact generating-partition ε-machine topology at slope `a` (probabilities omitted).
    """
    model = build_model(a)
    states, alphabet, transitions = forward_causal_graph(model.matrix, model.labels([0.5]), model.pi)
    machine = EpsilonMachine(observation_alphabet=frozenset(int(s) for s in alphabet))
    for state in states:
        machine.graph.add_state(state)
    for (state, symbol), target in transitions.items():
        machine.graph.add_transition(state, target, **{ATTR_PROB: 1.0, ATTR_EMISSION: int(symbol)})
    return machine, model.measures([0.5])


REGIMES = {
    "uncoupled": dict(coupling=np.zeros((4, 4))),
    "weak": dict(coupling=tent_ring_coupling(0.1)),
    "moderate": dict(coupling=tent_ring_coupling(0.3)),
    "synchronized": dict(coupling=tent_ring_coupling(0.5), same_slopes=True),
    # Node C's symbols have period 12 and need more than 5 symbols of history to resolve.
    "periodic": dict(coupling=tent_ring_coupling(0.5, 0.22, 0.83, 0.57), period=12, max_history=12),
}


@functools.cache
def regime(name):
    """
    (training symbols, held-out continuation, network with machines) for one regime.
    """
    spec = REGIMES[name]
    s = slopes()
    values = [s["A"]] * 4 if spec.get("same_slopes") else [s[n] for n in NAMES]
    x = coupled_tent(N_TRAIN + N_HELD, values, spec["coupling"], seed=0, period=spec.get("period"))
    train, held = x[:N_TRAIN], x[N_TRAIN:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = infer_multiplex(
            train,
            names=NAMES,
            max_lag=5,
            skeleton_settings=FAST,
            n_boot=50,
            n_null=50,
            node_layer=False,
            interpret=False,
            include_shared_candidates=False,
            machines=True,
            machine_settings=MachineSettings(
                max_history=spec.get("max_history", 5), stability=name == "uncoupled", excess_entropy=False
            ),
            prng=0,
        )
    return train, held, net


def edges(net):
    kind = net.dataset["kind"].values
    return {(net.names[s], net.names[t]) for s, t in zip(*np.nonzero(kind == "parent"), strict=True)}


def test_exact_machines_have_the_planned_state_counts():
    for name, a in slopes().items():
        machine, _ = exact_machine(a)
        assert len(list(machine.states())) == EXACT_STATES[name]


def test_uncoupled_is_the_false_positive_control():
    _, held, net = regime("uncoupled")
    assert edges(net) == set()
    ds = net.machines.dataset
    for name, a in slopes().items():
        _, exact = exact_machine(a)
        h = float(ds["h_mu"].sel(node=name))
        assert h == pytest.approx(np.log2(a), abs=0.02)
        assert h == pytest.approx(exact["h_mu"], abs=0.02)
    assert set(net.machines.model.roles.values()) == {"source"}
    loss = net.machines.model.log_loss(held)
    assert np.allclose(loss["model_bits"], [np.log2(slopes()[n]) for n in NAMES], atol=0.02)


@pytest.mark.xfail(
    strict=False,
    reason="these processes have infinite Markov and cryptic order: no finite history synchronizes every "
    "past, so CSSR splits the unsynchronized histories into extra states at any max_history",
)
def test_uncoupled_cssr_recovers_the_exact_topologies():
    _, _, net = regime("uncoupled")
    for name, a in slopes().items():
        assert topology_key(net.machines.nodes[name]) == topology_key(exact_machine(a)[0])


@pytest.mark.parametrize("name", ["weak", "moderate"])
def test_coupled_regimes_recover_the_ring(name):
    """
    Every true edge is found. Extra edges are expected: the binary partition does not
    screen off the maps' continuous state, so indirect paths leak through.
    """
    _, _, net = regime(name)
    found = edges(net)
    assert found >= TRUE_EDGES
    m = net.machines
    for src, tgt in TRUE_EDGES:
        assert (src, tgt) in m.edges
        edge = m.dataset.sel(source=src, target=tgt)
        assert 1 <= float(edge["edge_n_states"]) < 50
        assert float(edge["te_model"]) > 0.001
        assert abs(float(edge["te_gap"])) < 0.08
        assert edge["admitted_by"].item() == "plugin"
    assert {p for p, _ in m.inputs["A"]["parents"]} >= {"C", "D"}
    assert set(m.model.roles.values()) == {"driven"}
    assert ("model", "exact_flows") in m.skipped  # the product exceeds max_composed_states


def test_moderate_model_beats_the_independent_baseline():
    _, held, net = regime("moderate")
    loss = net.machines.model.log_loss(held)
    improvement = loss["improvement"].values
    assert np.all(improvement > 0.05)
    assert np.all(improvement < loss["omnibus_te"].values + 0.05)


@pytest.mark.xfail(
    strict=False,
    reason="weak flows (~0.04 bits) on top of five steps of self-memory: transCSSR at joint history 2 "
    "forgets the self-memory, and at 3 it over-splits (hundreds of states) on 20000 samples",
)
def test_weak_model_beats_the_independent_baseline():
    _, held, net = regime("weak")
    assert np.all(net.machines.model.log_loss(held)["improvement"].values > 0)


def test_synchronization_hides_the_topology():
    train, held, net = regime("synchronized")
    assert np.all(train == train[:, [0]])
    assert edges(net) == set()
    keys = {topology_key(net.machines.nodes[n]) for n in NAMES}
    assert len(keys) == 1
    a = slopes()["A"]
    assert np.allclose(net.machines.dataset["h_mu"].values, np.log2(a), atol=0.02)
    assert set(net.machines.model.roles.values()) == {"source"}


def test_periodic_window_has_no_flow():
    _, held, net = regime("periodic")
    assert edges(net) == set()
    ds = net.machines.dataset
    assert np.allclose(ds["h_mu"].values, 0.0, atol=1e-9)
    assert np.all(ds["n_states"].values <= 12)
    assert np.all(ds["n_forbidden"].values == 0)
    loss = net.machines.model.log_loss(held)
    assert np.allclose(loss["model_bits"].values, 0.0, atol=1e-9)
    assert np.all(loss["n_forbidden"].values == 0)


def test_too_short_a_history_is_flagged():
    """
    At max_history 5, CSSR turns node C's period-12 symbols into a wrong deterministic
    cycle; the machine then forbids symbols it was fitted to, and the layer says so.
    """
    from infoflow.machines import infer_machines

    train, _, net = regime("periodic")
    with pytest.warns(UserWarning, match="forbid symbols"):
        layer = infer_machines(train, net, MachineSettings(max_history=5, stability=False, excess_entropy=False))
    assert int(layer.dataset["n_forbidden"].sel(node="C")) > 0
    assert int(layer.dataset["n_forbidden"].sel(node="A")) == 0


def test_infer_front_end_on_the_ring():
    train, _, _ = regime("weak")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = infoflow.infer(
            train,
            names=NAMES,
            preset="fast",
            max_lag=5,
            machines=True,
            machine_settings=MachineSettings(max_history=5, stability=False, excess_entropy=False),
            prng=0,
        )
    assert res.report.estimators == ("plugin",)
    assert set(res.report.admitted.values()) == {"plugin"}
    assert res.report.machines["nodes"] == 4
    assert edges(res.network) >= TRUE_EDGES
