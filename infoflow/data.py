"""
Discretized multivariate data and alignment of variables into realizations.

A *variable* is a pair ``(process, lag)``. For a target process ``j`` at time
``t`` its present is ``present[j][t]`` and a past variable ``(p, lag)`` takes the
value ``past[p][t - lag]``. Keeping separate past and present encodings lets
ordinal discretizations encode the present as the bin of the current value alone,
which shares no values with the patterns of the past :cite:`Staniek2008`.
"""

from dataclasses import dataclass, field

import numpy as np
from dit.inference import Trials

__all__ = (
    "DiscreteData",
    "Realizations",
    "as_trials",
    "realizations",
)

Variable = tuple[int, int]


def as_trials(data):
    """
    Normalize input to a list of 2D float or int arrays shaped (samples, processes).

    Accepts a 2D array (one trial), a 1D array (one process, one trial), or
    :class:`~dit.inference.Trials` / a list of such arrays.
    """
    if isinstance(data, (Trials, list, tuple)) and len(data) and np.ndim(data[0]) >= 1 and not np.isscalar(data[0]):
        arrays = [np.asarray(t) for t in data]
        if all(a.ndim == 1 for a in arrays) and not isinstance(data, Trials):
            # A list of equal-length 1D arrays is a (processes, samples) layout.
            lengths = {len(a) for a in arrays}
            if len(lengths) == 1:
                return [np.stack(arrays, axis=1)]
        return [a[:, None] if a.ndim == 1 else a for a in arrays]
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("data must be (samples, processes), or a list/Trials of such arrays.")
    return [arr]


@dataclass
class DiscreteData:
    """
    Discretized multivariate time series, one or more trials.

    Attributes
    ----------
    past, present : list of np.ndarray
        Per trial, integer arrays of shape (processes, samples). Entries before a
        process's offset are ``-1`` (undefined).
    past_offset, present_offset : np.ndarray
        Per process, the first sample index at which ``past`` / ``present`` is defined.
    past_alphabet, present_alphabet : np.ndarray
        Per process, the number of possible symbols.
    lag_step : np.ndarray
        Per process, the minimum spacing between stacked past lags that keeps their
        windows from overlapping (1 for binned data).
    names : list of str
        Process names.
    discretizers : list
        The discretizer used for each process (provenance).
    raw : list of np.ndarray or None
        The raw (pre-discretization) trials, used to re-encode surrogates.
    """

    past: list
    present: list
    past_offset: np.ndarray
    present_offset: np.ndarray
    past_alphabet: np.ndarray
    present_alphabet: np.ndarray
    lag_step: np.ndarray
    names: list = field(default_factory=list)
    discretizers: list = field(default_factory=list)
    raw: list | None = None

    @property
    def n_processes(self):
        return len(self.past_offset)

    @property
    def n_trials(self):
        return len(self.past)

    @property
    def n_samples(self):
        return sum(p.shape[1] for p in self.past)

    @classmethod
    def from_discrete(cls, data, names=None):
        """
        Wrap already-discrete data: each process's symbols are both its past and present.
        """
        trials = as_trials(data)
        P = trials[0].shape[1]
        uniques = [np.unique(np.concatenate([t[:, p] for t in trials])) for p in range(P)]
        alphabets = np.array([max(len(u), 1) for u in uniques], dtype=np.int64)
        codes = []
        for t in trials:
            enc = np.empty((P, t.shape[0]), dtype=np.int64)
            for p in range(P):
                enc[p] = np.searchsorted(uniques[p], t[:, p])
            codes.append(enc)
        zeros = np.zeros(P, dtype=np.int64)
        return cls(
            past=codes,
            present=[c.copy() for c in codes],
            past_offset=zeros.copy(),
            present_offset=zeros.copy(),
            past_alphabet=alphabets,
            present_alphabet=alphabets.copy(),
            lag_step=np.ones(P, dtype=np.int64),
            names=list(names) if names is not None else [f"x{p}" for p in range(P)],
            discretizers=["identity"] * P,
        )

    def split_time(self, fraction=0.5):
        """
        Split every trial in time: (first `fraction` of each trial, the rest).

        Offsets are kept, so the second part's first samples are undefined until
        its windows fill; use it with :func:`realizations` as usual.
        """

        def part(arrays, lo, hi):
            return [a[:, int(lo * a.shape[1]) : int(hi * a.shape[1])] for a in arrays]

        def make(lo, hi):
            return DiscreteData(
                past=part(self.past, lo, hi),
                present=part(self.present, lo, hi),
                past_offset=self.past_offset,
                present_offset=self.present_offset,
                past_alphabet=self.past_alphabet,
                present_alphabet=self.present_alphabet,
                lag_step=self.lag_step,
                names=self.names,
                discretizers=self.discretizers,
                raw=None if self.raw is None else [r[int(lo * len(r)) : int(hi * len(r))] for r in self.raw],
            )

        return make(0.0, fraction), make(fraction, 1.0)

    def subset_trials(self, indices):
        """
        A copy restricted to the given trials.
        """
        return DiscreteData(
            past=[self.past[i] for i in indices],
            present=[self.present[i] for i in indices],
            past_offset=self.past_offset,
            present_offset=self.present_offset,
            past_alphabet=self.past_alphabet,
            present_alphabet=self.present_alphabet,
            lag_step=self.lag_step,
            names=self.names,
            discretizers=self.discretizers,
            raw=None if self.raw is None else [self.raw[i] for i in indices],
        )


@dataclass
class Realizations:
    """
    Aligned realizations of a target's present and a list of variables.

    Attributes
    ----------
    present : np.ndarray
        The target's present, one entry per realization.
    values : np.ndarray
        Shape (realizations, variables): each variable's value.
    variables : list of (process, lag)
    trial : np.ndarray
        The trial each realization comes from.
    time : np.ndarray
        The time index of each realization within its trial.
    """

    present: np.ndarray
    values: np.ndarray
    variables: list
    trial: np.ndarray
    time: np.ndarray

    def __len__(self):
        return len(self.present)

    def columns(self, variables):
        """
        The value matrix for a subset of the variables, in the given order.
        """
        index = [self.variables.index(v) for v in variables]
        return self.values[:, index]


def realizations(data, target, variables, max_lag=None):
    """
    Align the target's present with past variables across all trials.

    Parameters
    ----------
    data : DiscreteData
    target : int
        The target process.
    variables : sequence of (process, lag)
        Lags are non-negative; lag 0 means the same time step as the target's
        present (used only for instantaneous-mixing compensation).
    max_lag : int, None
        If given, realizations start at a time valid for any variable with lag up
        to `max_lag`, so that realizations for different candidate sets align
        exactly (a common sample set, as in IDTxl).

    Returns
    -------
    Realizations
    """
    variables = [tuple(int(a) for a in v) for v in variables]
    start = int(data.present_offset[target])
    for p, lag in variables:
        start = max(start, int(data.past_offset[p]) + lag)
    if max_lag is not None:
        start = max(start, int(data.past_offset.max()) + int(max_lag), int(data.present_offset.max()))
    presents, values, trials, times = [], [], [], []
    for i, (past, present) in enumerate(zip(data.past, data.present, strict=True)):
        N = past.shape[1]
        if start >= N:
            continue
        t = np.arange(start, N)
        presents.append(present[target, t])
        cols = [past[p, t - lag] for p, lag in variables]
        values.append(np.stack(cols, axis=1) if cols else np.zeros((len(t), 0), dtype=np.int64))
        trials.append(np.full(len(t), i))
        times.append(t)
    if not presents:
        empty = np.zeros(0, dtype=np.int64)
        return Realizations(empty, np.zeros((0, len(variables)), dtype=np.int64), variables, empty, empty)
    return Realizations(
        np.concatenate(presents),
        np.concatenate(values),
        variables,
        np.concatenate(trials),
        np.concatenate(times),
    )
