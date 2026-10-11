"""
Noisy elementary cellular automata on a six-cell ring: the symbols are the full state,
so the factorization into machines is exact and every answer is known.

Rule 90 (``l xor r``) is pure synergy: each neighbour alone carries no transfer
entropy, and the joint-input transducer is the one-state XOR table. Rule 110 depends
on the cell itself, so its target transducers need two states: the cell's previous
value.
"""

import functools
import itertools
import warnings

import numpy as np
import pytest

sofic_inference = pytest.importorskip("sofic.inference")
if not hasattr(sofic_inference, "Trials"):
    pytest.skip("needs sofic with multi-trial CSSR (sofic.inference.Trials)", allow_module_level=True)

from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL  # noqa: E402

from infoflow.datasets import noisy_eca  # noqa: E402
from infoflow.machines import MachineSettings  # noqa: E402
from infoflow.network import infer_multiplex  # noqa: E402
from infoflow.selection import SkeletonSettings  # noqa: E402

pytestmark = pytest.mark.slow

N_CELLS, NOISE = 6, 0.05
NAMES = [f"c{i}" for i in range(N_CELLS)]
NEIGHBOURS = {NAMES[i]: (NAMES[(i - 1) % N_CELLS], NAMES[(i + 1) % N_CELLS]) for i in range(N_CELLS)}
TRUE_EDGES = {(nb, cell) for cell, pair in NEIGHBOURS.items() for nb in pair}
FAST = SkeletonSettings(
    n_perm_max_stat=100,
    n_perm_min_stat=100,
    n_perm_omnibus=200,
    n_perm_max_seq=200,
    n_perm_pairs=50,
    n_perm_tdmi=100,
)


def _h2(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def rule_bit(rule, left, centre, right):
    return (rule >> (4 * left + 2 * centre + right)) & 1


@functools.cache
def ring(rule):
    x = noisy_eca(30000, N_CELLS, rule, NOISE, seed=0)
    train, held = x[:20000], x[20000:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = infer_multiplex(
            train,
            names=NAMES,
            max_lag=3,
            skeleton_settings=FAST,
            n_boot=50,
            n_null=50,
            node_layer=False,
            interpret=False,
            include_shared_candidates=False,
            machines=True,
            machine_settings=MachineSettings(max_history=3, stability=False, excess_entropy=False),
            prng=0,
        )
    return held, net


def parents(net):
    kind = net.dataset["kind"].values
    return {(net.names[s], net.names[t]) for s, t in zip(*np.nonzero(kind == "parent"), strict=True)}


def reads_rule(layer, cell, rule):
    """
    Whether the target transducer of `cell` is the noisy rule table.

    Each state must stand for one value of the cell's previous symbol (a bijection
    found by search), and every row's P(output = 1) must be within 0.03 of the
    noisy rule's: ``1 - NOISE`` where the rule gives 1, ``NOISE`` where it gives 0.
    """
    transducer = layer.targets[cell]
    inputs = [name for name, _ in layer.inputs[cell]["parents"]]
    left, right = (inputs.index(nb) for nb in NEIGHBOURS[cell])
    rows = [
        (state, tr.data[ATTR_SYMBOL], float(tr.data[ATTR_PROB]))
        for state in transducer.states()
        for tr in transducer.graph.out_transitions(state)
        if tr.data[ATTR_OUTPUT] == 1
    ]
    states = sorted({state for state, _, _ in rows}, key=repr)
    for centres in itertools.product((0, 1), repeat=len(states)):
        centre = dict(zip(states, centres, strict=True))
        if all(
            abs(p1 - (1 - NOISE if rule_bit(rule, u[left], centre[state], u[right]) else NOISE)) < 0.03
            for state, u, p1 in rows
        ):
            return True
    return False


def test_rule_110_recovers_the_ring_and_the_rule():
    held, net = ring(110)
    assert parents(net) == TRUE_EDGES
    m = net.machines
    for cell in NAMES:
        assert int(m.dataset["target_n_states"].sel(node=cell)) == 2
        assert reads_rule(m, cell, 110)
    assert "model_te" in m.dataset  # the composed model is small enough for exact flows
    loss = m.model.log_loss(held)
    assert np.allclose(loss["model_bits"].values, _h2(NOISE), atol=0.02)
    assert not m.model.adequacy(held, n_sim=39, seed=0)["flagged"].any()


def test_rule_90_is_pure_synergy_with_one_state_xor_transducers():
    held, net = ring(90)
    found = parents(net)
    assert found >= TRUE_EDGES
    ds, m = net.dataset, net.machines
    for src, tgt in TRUE_EDGES:
        assert float(ds["weight"].sel(layer="synergistic", source=src, target=tgt)) > 0.3
        assert float(ds["weight"].sel(layer="intrinsic", source=src, target=tgt)) < 0.01
        assert float(m.dataset["te_model"].sel(source=src, target=tgt)) < 0.01
    for src, tgt in found - TRUE_EDGES:
        # Extra parents carry shared flow only: the layers already show no transfer.
        assert float(ds["weight"].sel(layer="te", source=src, target=tgt)) < 0.01
    for cell in NAMES:
        assert int(m.dataset["target_n_states"].sel(node=cell)) == 1
        assert reads_rule(m, cell, 90)
    loss = m.model.log_loss(held)
    assert np.allclose(loss["model_bits"].values, _h2(NOISE), atol=0.02)
    # At least the collective TE: more where CSSR's own-past baseline machine is weaker.
    assert np.all(loss["improvement"].values > loss["omnibus_te"].values - 0.03)
    assert not m.model.adequacy(held, n_sim=39, seed=0)["flagged"].any()
