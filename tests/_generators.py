"""
Tuple-emitting sofic generators with exactly known flows (test fixtures).
"""

import itertools

import numpy as np

sofic = __import__("pytest").importorskip("sofic")
from sofic.generators.mealy import MealyHMM  # noqa: E402


def mixture(a_copy=0.0, a_xor=0.0, a_hold=0.0, x_given_y=None):
    """
    y_t = x_{t-1} (prob a_copy), x_{t-1} xor y_{t-1} (a_xor), y_{t-1} (a_hold), else a fair coin.

    x_t is a fair coin, or, if `x_given_y` is set, equals y_{t-1} with that probability
    (a source that copies the target's history, which creates shared flow).
    State is (x_{t-1}, y_{t-1}).
    """
    rest = 1.0 - a_copy - a_xor - a_hold
    assert rest >= -1e-12
    hmm = MealyHMM(
        observation_alphabet=frozenset(itertools.product((0, 1), repeat=2)),
        initial_distribution=dict.fromkeys(itertools.product((0, 1), repeat=2), 0.25),
    )
    for state in itertools.product((0, 1), repeat=2):
        hmm.graph.add_state(state)
    for xp, yp in itertools.product((0, 1), repeat=2):
        for x in (0, 1):
            px = 0.5 if x_given_y is None else (x_given_y if x == yp else 1 - x_given_y)
            for y in (0, 1):
                py = a_copy * (y == xp) + a_xor * (y == (xp ^ yp)) + a_hold * (y == yp) + rest * 0.5
                if px * py > 0:
                    hmm.add_transition((xp, yp), (x, y), (x, y), px * py)
    hmm.validate()
    return hmm


def exact_flows(generator):
    from sofic.generators.directional_flow import (
        intrinsic_information_flow,
        shared_information_flow,
        synergistic_information_flow,
        transfer_entropy,
    )

    return {
        "te": transfer_entropy(generator, history=1),
        "intrinsic": intrinsic_information_flow(generator, history=1),
        "synergistic": synergistic_information_flow(generator, history=1),
        "shared": shared_information_flow(generator, history=1),
    }


def sample_pairs(generator, n, seed):
    observations, _ = generator.sample(n, np.random.default_rng(seed))
    return np.array(observations, dtype=np.int64)
