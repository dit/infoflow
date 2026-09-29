"""
Transfer entropy from samples: estimates with flexible embeddings, surrogate
tests, and confidence intervals.

Transfer entropy :cite:`Schreiber2000` is estimated on aligned samples
(source past, target present, target and condition pasts) with any of dit's
entropy estimators. The alignment supports separate source and target history
lengths, a source lag, conditioning series, and pooled :class:`~dit.inference.Trials`.
"""

import numpy as np
from dit.inference import block_surrogates, shift_surrogates, stationary_bootstrap, whittle_surrogates
from dit.inference._symbols import as_generator, is_trials, standardize_trials
from dit.inference.estimators import _check_joint, _cmi_codes, _dense, _pair

from .stats import _interval, _result, _within_strata

__all__ = (
    "transfer_entropy",
    "transfer_entropy_ci",
    "transfer_entropy_test",
)


def _window_codes(codes, L, start, stop):
    """
    Rows ``codes[s:s + L]`` for ``s`` in ``range(start, stop)``.
    """
    if L == 0:
        return np.zeros((stop - start, 0), dtype=np.int64)
    return np.lib.stride_tricks.sliding_window_view(codes, L)[start:stop]


def _dense_rows(rows):
    if rows.shape[1] == 0:
        return np.zeros(len(rows), dtype=np.int64)
    return _dense(rows)


def _resolve_te(target, history_length, source_history, lag, max_history, prng):
    """
    Validate transfer entropy parameters, resolving ``history_length='auto'``.
    """
    if isinstance(history_length, str):
        if history_length != "auto":
            raise ValueError(f"Unknown history_length {history_length!r}.")
        from dit.inference import select_markov_order

        trials, alphabet = standardize_trials(target)
        if max_history is None:
            N = sum(len(t) for t in trials)
            M = max(len(alphabet), 2)
            max_history = max(int(np.log(max(N, 1) / 5) / np.log(M)) - 1, 1)
        history_length = select_markov_order(target, max_history, prng=prng)
    k = int(history_length)
    l = max(k, 1) if source_history is None else int(source_history)
    if k < 0:
        raise ValueError("`history_length` must be non-negative.")
    if l < 1:
        raise ValueError("`source_history` must be at least 1.")
    if lag < 1:
        raise ValueError("`lag` must be at least 1.")
    return k, l, int(lag)


def _te_codes(source, target, k, l=None, lag=1, conditions=()):
    """
    Aligned (source past, target present, context) codes, pooled over trials.

    For each time ``t`` the source past is ``X[t - lag - l + 1 : t - lag + 1]``,
    the target past ``Y[t - k : t]``, and the context joins the target past with
    each condition's past ``Z[t - m : t]``, ``m = max(k, 1)``.
    """
    l = max(k, 1) if l is None else l
    series = [source, target, *conditions]
    kinds = {is_trials(x) for x in series}
    if len(kinds) > 1:
        raise ValueError("`source`, `target` and `conditions` must all be Trials, or none.")
    coded = [standardize_trials(x)[0] for x in series]
    lengths = [[len(t) for t in c] for c in coded]
    if any(ls != lengths[0] for ls in lengths):
        raise ValueError("`source` and `target` (and `conditions`) must have the same length.")
    m = max(k, 1)
    t0 = max(k, l + lag - 1, m if conditions else 0)
    source_rows, target_rows, present, condition_rows = [], [], [], [[] for _ in conditions]
    for trial in range(len(coded[0])):
        N = lengths[0][trial]
        if t0 >= N:
            continue
        x, y = coded[0][trial], coded[1][trial]
        source_rows.append(_window_codes(x, l, t0 - lag - l + 1, N - lag - l + 1))
        target_rows.append(_window_codes(y, k, t0 - k, N - k))
        present.append(y[t0:N])
        for rows, z in zip(condition_rows, coded[2:], strict=True):
            rows.append(_window_codes(z[trial], m, t0 - m, N - m))
    if not present:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    source_past = _dense_rows(np.concatenate(source_rows))
    context = _dense_rows(np.concatenate(target_rows))
    for rows in condition_rows:
        context = _pair(context, _dense_rows(np.concatenate(rows)))
    return source_past, _dense(np.concatenate(present)), context


_TE_PARAMETERS = """source, target : array_like or Trials
        Equal-length series; rows of 2D arrays are joint symbols. Pass
        :class:`~dit.inference.Trials` of equal-length pairs to pool trials.
    history_length : int or 'auto'
        The target history length :math:`k \\geq 0`. ``'auto'`` uses the
        target's Markov order from :func:`~dit.inference.select_markov_order`
        (up to `max_history`).
    source_history : int, None
        The source history length :math:`l \\geq 1`; defaults to
        ``max(k, 1)``.
    lag : int
        The source past ends `lag` steps before the target's present.
    conditions : sequence of array_like, optional
        Further series whose pasts (of length ``max(k, 1)``) are conditioned on,
        giving the conditional transfer entropy."""


def transfer_entropy(
    source,
    target,
    history_length=1,
    estimator="plugin",
    source_history=None,
    lag=1,
    conditions=(),
    max_history=None,
    prng=None,
):
    """
    Estimate the transfer entropy from `source` to `target` in bits,

    .. math::

        T_{X \\to Y} = I[Y_t : X_{t-\\tau-l+1:t-\\tau+1} \\mid Y_{t-k:t}, Z_{t-m:t}]

    with :math:`k` = `history_length`, :math:`l` = `source_history`,
    :math:`\\tau` = `lag` and optional conditioning series :math:`Z`
    :cite:`Schreiber2000`. With the defaults this is
    :math:`I[Y_t : X_{t-k:t} \\mid Y_{t-k:t}]`.

    Parameters
    ----------
    {params}
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    max_history : int, None
        The largest history considered for ``history_length='auto'``.
    prng : None, int, Generator, RandomState
        Source of randomness for ``history_length='auto'``.

    Returns
    -------
    te : float
    """
    k, l, lag = _resolve_te(target, history_length, source_history, lag, max_history, prng)
    codes = _te_codes(source, target, k, l, lag, conditions)
    _check_joint(*codes)
    return _cmi_codes(*codes, estimator)


def transfer_entropy_test(
    source,
    target,
    history_length=1,
    null="conditional",
    n_surrogates=1000,
    estimator="plugin",
    surrogate_order=None,
    block_length=None,
    prng=None,
    source_history=None,
    lag=1,
    conditions=(),
    max_history=None,
):
    """
    Test whether the transfer entropy from `source` to `target` is zero.

    Parameters
    ----------
    {params}
    null : {'conditional', 'whittle', 'shift', 'block'}
        How surrogates are built:

        * ``'conditional'`` — permute the source past within strata of the target
          past (and the conditions' pasts). This targets exactly
          :math:`T_{X \\to Y} = 0`.
        * ``'whittle'`` — replace the source by :func:`~dit.inference.whittle_surrogates`
          of order `surrogate_order`, preserving its own Markov structure
          :cite:`Pethel2014`.
        * ``'shift'`` — circularly shift the source relative to the target.
        * ``'block'`` — shuffle blocks of `block_length` of the source.

        The last three test the stronger null that the source is independent of
        the target, while keeping the source's own memory. Unlike an i.i.d.
        shuffle, they do not inflate false positives when the source is
        autocorrelated.
    n_surrogates : int
        The number of surrogates.
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    surrogate_order : int, None
        The order preserved by ``'whittle'``; defaults to `source_history`.
    block_length : int, None
        The block length for ``'block'``; defaults to
        ``max(2 * (l + lag), round(sqrt(N)))``.
    prng : None, int, Generator, RandomState
        Source of randomness.
    max_history : int, None
        The largest history considered for ``history_length='auto'``.

    Returns
    -------
    result : SurrogateTest
    """
    rng = as_generator(prng)
    k, l, lag = _resolve_te(target, history_length, source_history, lag, max_history, rng)
    source_past, present, context = _te_codes(source, target, k, l, lag, conditions)
    _check_joint(source_past, present, context)
    value = _cmi_codes(source_past, present, context, estimator)
    n_samples = len(present)

    if null == "conditional":
        null_values = [
            _cmi_codes(_within_strata(source_past, context, rng), present, context, estimator)
            for _ in range(n_surrogates)
        ]
        return _result(value, np.array(null_values), n_samples)

    if not is_trials(source):
        source = np.asarray(source)
    if null == "whittle":
        order = l if surrogate_order is None else surrogate_order
        surrogates = whittle_surrogates(source, order, n=n_surrogates, prng=rng)
    elif null == "shift":
        surrogates = shift_surrogates(source, n=n_surrogates, prng=rng)
    elif null == "block":
        if block_length is None:
            N = min(len(t) for t in source) if is_trials(source) else len(source)
            block_length = max(2 * (l + lag), round(np.sqrt(N)))
        surrogates = block_surrogates(source, block_length, n=n_surrogates, prng=rng)
    else:
        raise ValueError(f"Unknown null {null!r}.")
    null_values = [_cmi_codes(*_te_codes(s, target, k, l, lag, conditions), estimator) for s in surrogates]
    return _result(value, np.array(null_values), n_samples)


def transfer_entropy_ci(
    source,
    target,
    history_length=1,
    n_boot=1000,
    confidence=0.95,
    mean_block_length=None,
    estimator="plugin",
    method="percentile",
    prng=None,
    source_history=None,
    lag=1,
    conditions=(),
    max_history=None,
):
    """
    A stationary-bootstrap confidence interval for the transfer entropy.

    The aligned samples (source past, target present, target and condition
    pasts) are resampled in blocks :cite:`Politis1994`, so block junctions never
    create windows that were not observed.

    Parameters
    ----------
    {params}
    n_boot : int
        The number of resamples.
    confidence : float
        The coverage of the interval.
    mean_block_length : float, None
        See :func:`stationary_bootstrap`.
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    method : {'percentile', 'basic'}
        See :func:`bootstrap_ci`.
    prng : None, int, Generator, RandomState
        Source of randomness.
    max_history : int, None
        The largest history considered for ``history_length='auto'``.

    Returns
    -------
    low, high : float
        The interval endpoints.
    """
    rng = as_generator(prng)
    k, l, lag = _resolve_te(target, history_length, source_history, lag, max_history, rng)
    source_past, present, context = _te_codes(source, target, k, l, lag, conditions)
    _check_joint(source_past, present, context)
    value = _cmi_codes(source_past, present, context, estimator)
    resamples = stationary_bootstrap(np.arange(len(present)), n_boot, mean_block_length, rng)
    samples = [_cmi_codes(source_past[i], present[i], context[i], estimator) for i in resamples]
    return _interval(samples, value, confidence, method)


for _function in (transfer_entropy, transfer_entropy_test, transfer_entropy_ci):
    _function.__doc__ = _function.__doc__.replace("{params}", _TE_PARAMETERS)
del _function
