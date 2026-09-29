"""
Phase 1 milestone: sample estimates converge to sofic's exact flows.
"""

import pytest

from infoflow.crossfit import crossfit_flows
from infoflow.data import DiscreteData, realizations

from ._generators import exact_flows, mixture, sample_pairs

CASES = {
    "copy": dict(a_copy=0.8),
    "xor": dict(a_xor=0.9),
    "shared_history": dict(a_hold=0.8, x_given_y=0.9),
    "mixture": dict(a_copy=0.4, a_xor=0.3, a_hold=0.2),
    "copy_and_shared": dict(a_copy=0.5, a_hold=0.4, x_given_y=0.8),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_estimates_match_exact_flows(name):
    generator = mixture(**CASES[name])
    exact = exact_flows(generator)
    data = DiscreteData.from_discrete(sample_pairs(generator, 20000, seed=0))
    r = realizations(data, 1, [(0, 1), (1, 1)])
    flow = crossfit_flows(r.present, r.values[:, 0], r.values[:, 1], prng=0).as_dict()
    for layer in ("te", "intrinsic", "synergistic", "shared"):
        assert flow[layer] == pytest.approx(exact[layer], abs=0.02), layer
