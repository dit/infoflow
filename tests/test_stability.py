"""
Tests for stability selection with e-value FDR control.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from infoflow.datasets import chain
from infoflow.stability import e_bh, stability_selection


def test_e_bh_examples():
    assert e_bh([100, 1, 1, 1], 0.1).tolist() == [True, False, False, False]
    assert not e_bh([5, 5, 5, 5], 0.1).any()
    assert e_bh([20, 20, 20, 20], 0.2).all()  # 20 >= 4 / (0.2 * 4)
    assert e_bh([], 0.1).size == 0


@given(st.lists(st.floats(0, 1e3), min_size=1, max_size=30), st.floats(0.01, 0.5))
def test_e_bh_rejects_a_top_set_meeting_its_threshold(e, alpha):
    e = np.array(e)
    reject = e_bh(e, alpha)
    k = int(reject.sum())
    if k:
        assert e[reject].min() >= e[~reject].max(initial=-np.inf)
        assert e[reject].min() >= len(e) / (alpha * k) - 1e-9


def test_stability_selection_on_chain():
    res = stability_selection(chain(6000, seed=0), n_subsamples=6, alpha=0.1, prng=0)
    assert {(0, 1), (1, 2)} <= res.selected
    assert (0, 2) not in res.selected
    freq = res.dataset["frequency"].values
    assert freq[0, 1] >= 0.5 and freq[1, 2] >= 0.5
    assert np.isnan(freq[0, 0])


def test_stability_selection_is_quiet_on_noise():
    data = np.random.default_rng(1).integers(0, 2, size=(4000, 3))
    res = stability_selection(data, n_subsamples=5, prng=1)
    assert not res.selected
