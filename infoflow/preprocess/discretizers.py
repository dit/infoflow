"""
Discretizers turning continuous series into past and present symbols.

Every discretizer is a pure function of the raw series once fitted, so
surrogates built on raw data (shifts, blocks, trial permutations) are encoded
*after* generation with the same fitted discretizer. Rank-based discretizers
(equal-frequency bins and ordinal patterns) are invariant to monotone transforms
of each channel and are the only ones chosen automatically.
"""

from dataclasses import dataclass, field
from math import factorial

import numpy as np
from dit.inference import ordinal_patterns, relative_rank

from ..data import DiscreteData, as_trials

__all__ = (
    "Discretizer",
    "EqualFrequency",
    "EqualWidth",
    "Identity",
    "Ordinal",
    "Threshold",
    "discretize",
)


@dataclass
class Encoding:
    """
    One process's encoding of one trial.
    """

    past: np.ndarray
    present: np.ndarray
    past_offset: int
    present_offset: int


class Discretizer:
    """
    Base class. Subclasses implement :meth:`fit` and :meth:`encode`.

    Attributes
    ----------
    past_alphabet, present_alphabet : int
        Number of possible past and present symbols.
    lag_step : int
        Minimum spacing of stacked lags that keeps their windows disjoint.
    """

    rank_based = False
    past_alphabet = 0
    present_alphabet = 0
    lag_step = 1

    def fit(self, series):
        """
        Learn any data-dependent parameters from the pooled trials of one process.
        """
        return self

    def encode(self, x):
        """
        Encode one trial of one process; returns an :class:`Encoding`.
        """
        raise NotImplementedError

    def params(self):
        """
        A dict describing the discretizer, for provenance.
        """
        return {"name": type(self).__name__}

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items() if k != "name")
        return f"{type(self).__name__}({args})"


@dataclass(repr=False)
class Identity(Discretizer):
    """
    Already-discrete data: symbols are mapped to 0..K-1 and used as-is.
    """

    values: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, series):
        self.values = np.unique(np.concatenate([np.asarray(s) for s in series]))
        self.past_alphabet = self.present_alphabet = max(len(self.values), 1)
        return self

    def encode(self, x):
        codes = np.searchsorted(self.values, np.asarray(x)).astype(np.int64)
        return Encoding(codes, codes.copy(), 0, 0)

    def params(self):
        return {"name": "Identity", "alphabet": int(self.past_alphabet)}


@dataclass(repr=False)
class EqualFrequency(Discretizer):
    """
    Equal-frequency (maximum-entropy) bins, with edges at pooled quantiles.
    """

    bins: int = 4
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rank_based = True

    def fit(self, series):
        values = np.concatenate([np.asarray(s, dtype=float) for s in series])
        self.edges = np.quantile(values, np.linspace(0, 1, self.bins + 1)[1:-1])
        self.past_alphabet = self.present_alphabet = self.bins
        return self

    def encode(self, x):
        codes = np.searchsorted(self.edges, np.asarray(x, dtype=float), side="right").astype(np.int64)
        return Encoding(codes, codes.copy(), 0, 0)

    def params(self):
        return {"name": "EqualFrequency", "bins": self.bins}


@dataclass(repr=False)
class EqualWidth(Discretizer):
    """
    Equal-width bins over the pooled range. Not rank-based; never chosen automatically.
    """

    bins: int = 4
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, series):
        values = np.concatenate([np.asarray(s, dtype=float) for s in series])
        self.edges = np.linspace(values.min(), values.max(), self.bins + 1)[1:-1]
        self.past_alphabet = self.present_alphabet = self.bins
        return self

    def encode(self, x):
        codes = np.searchsorted(self.edges, np.asarray(x, dtype=float), side="right").astype(np.int64)
        return Encoding(codes, codes.copy(), 0, 0)

    def params(self):
        return {"name": "EqualWidth", "bins": self.bins}


@dataclass(repr=False)
class Threshold(Discretizer):
    """
    Binary events: 1 where the series exceeds a threshold (or a quantile of it).
    """

    threshold: float | None = None
    quantile: float = 0.9
    rank_based = True

    def fit(self, series):
        if self.threshold is None:
            values = np.concatenate([np.asarray(s, dtype=float) for s in series])
            self.threshold = float(np.quantile(values, self.quantile))
            self.rank_based = True
        self.past_alphabet = self.present_alphabet = 2
        return self

    def encode(self, x):
        codes = (np.asarray(x, dtype=float) > self.threshold).astype(np.int64)
        return Encoding(codes, codes.copy(), 0, 0)

    def params(self):
        return {"name": "Threshold", "threshold": self.threshold, "quantile": self.quantile}


@dataclass(repr=False)
class Ordinal(Discretizer):
    """
    Ordinal patterns :cite:`Bandt2002`.

    The past symbol at time ``t`` is the pattern of
    :math:`(x_{t-(m-1)\\tau}, \\ldots, x_t)` (``m!`` symbols). The *present* symbol
    is the rank of :math:`x_t` among the :math:`m` values
    :math:`x_{t-\\tau}, \\ldots, x_{t-m\\tau}` (``m + 1`` symbols). Encoding the
    present as a relative rank means it shares no values with the target's past
    patterns, which removes the leakage that biases symbolic transfer entropy
    :cite:`Staniek2008,Kugiumtzis2012`. Stacked lags of one process are spaced by
    ``lag_step = (m - 1) * tau + 1`` so that their windows are disjoint.
    """

    order: int = 3
    delay: int = 1
    ties: str = "first"
    rank_based = True

    def fit(self, series):
        self.past_alphabet = factorial(self.order)
        self.present_alphabet = self.order + 1
        self.lag_step = (self.order - 1) * self.delay + 1
        return self

    def encode(self, x):
        x = np.asarray(x, dtype=float)
        N = len(x)
        m, tau = self.order, self.delay
        past = np.full(N, -1, dtype=np.int64)
        present = np.full(N, -1, dtype=np.int64)
        past_offset = (m - 1) * tau
        present_offset = m * tau
        if past_offset < N:
            past[past_offset:] = ordinal_patterns(x, m, tau, self.ties)
        if present_offset < N:
            present[present_offset:] = relative_rank(x, m, tau, self.ties)
        return Encoding(past, present, past_offset, present_offset)

    def params(self):
        return {"name": "Ordinal", "order": self.order, "delay": self.delay, "ties": self.ties}


def discretize(data, discretizers, names=None):
    """
    Discretize raw multivariate data, one discretizer per process.

    Parameters
    ----------
    data : array_like or Trials
        Shape (samples, processes), or a list of such trials.
    discretizers : Discretizer or list of Discretizer
        One per process, or one shared (copied) for all processes. Each is fitted
        on that process's pooled trials.
    names : list of str, None
        Process names.

    Returns
    -------
    DiscreteData
    """
    import copy

    trials = as_trials(data)
    P = trials[0].shape[1]
    if isinstance(discretizers, Discretizer):
        discretizers = [copy.deepcopy(discretizers) for _ in range(P)]
    if len(discretizers) != P:
        raise ValueError(f"need {P} discretizers, got {len(discretizers)}")
    fitted = [d.fit([t[:, p] for t in trials]) for p, d in enumerate(discretizers)]
    past, present = [], []
    past_offset = np.zeros(P, dtype=np.int64)
    present_offset = np.zeros(P, dtype=np.int64)
    for t in trials:
        pa = np.empty((P, t.shape[0]), dtype=np.int64)
        pr = np.empty((P, t.shape[0]), dtype=np.int64)
        for p, d in enumerate(fitted):
            enc = d.encode(t[:, p])
            pa[p], pr[p] = enc.past, enc.present
            past_offset[p], present_offset[p] = enc.past_offset, enc.present_offset
        past.append(pa)
        present.append(pr)
    return DiscreteData(
        past=past,
        present=present,
        past_offset=past_offset,
        present_offset=present_offset,
        past_alphabet=np.array([d.past_alphabet for d in fitted], dtype=np.int64),
        present_alphabet=np.array([d.present_alphabet for d in fitted], dtype=np.int64),
        lag_step=np.array([d.lag_step for d in fitted], dtype=np.int64),
        names=list(names) if names is not None else [f"x{p}" for p in range(P)],
        discretizers=fitted,
        raw=[np.asarray(t, dtype=float) for t in trials],
    )


def reencode(data, raw_trials):
    """
    Encode new raw trials (e.g. surrogates built on the raw series) with the
    already-fitted discretizers of `data`.
    """
    template = DiscreteData(
        past=[],
        present=[],
        past_offset=data.past_offset,
        present_offset=data.present_offset,
        past_alphabet=data.past_alphabet,
        present_alphabet=data.present_alphabet,
        lag_step=data.lag_step,
        names=data.names,
        discretizers=data.discretizers,
        raw=[np.asarray(t, dtype=float) for t in raw_trials],
    )
    for t in raw_trials:
        P = t.shape[1]
        pa = np.empty((P, t.shape[0]), dtype=np.int64)
        pr = np.empty((P, t.shape[0]), dtype=np.int64)
        for p, d in enumerate(data.discretizers):
            enc = d.encode(t[:, p])
            pa[p], pr[p] = enc.past, enc.present
        template.past.append(pa)
        template.present.append(pr)
    return template
