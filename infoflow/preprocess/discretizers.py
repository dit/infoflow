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
from dit.inference import ordinal_patterns

from ..data import DiscreteData, as_trials

__all__ = (
    "Discretizer",
    "EqualFrequency",
    "EqualWidth",
    "ExponentialBins",
    "Identity",
    "Ordinal",
    "OrdinalIntervals",
    "SpikeIntervals",
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
    Equal-width bins over the pooled range. Not rank-based, so not invariant to
    monotone transforms; scored automatically only with ``preprocess(equal_width=True)``.
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
    is the equal-frequency bin of :math:`x_t` alone (``m + 1`` bins at pooled
    quantiles), so it shares no values with the target's past patterns, the leakage
    that biases symbolic transfer entropy :cite:`Staniek2008`. A present defined
    relative to past values, such as the rank of :math:`x_t` among them
    :cite:`Kugiumtzis2012`, would still carry those values' magnitudes, which the
    past patterns do not; any child of the process would then appear to predict it.
    Stacked lags of one process are spaced by ``lag_step = (m - 1) * tau + 1`` so
    that their windows are disjoint.
    """

    order: int = 3
    delay: int = 1
    ties: str = "first"
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rank_based = True

    def fit(self, series):
        values = np.concatenate([np.asarray(s, dtype=float) for s in series])
        self.edges = np.quantile(values, np.linspace(0, 1, self.order + 2)[1:-1])
        self.past_alphabet = factorial(self.order)
        self.present_alphabet = self.order + 1
        self.lag_step = (self.order - 1) * self.delay + 1
        return self

    def encode(self, x):
        x = np.asarray(x, dtype=float)
        N = len(x)
        m, tau = self.order, self.delay
        past = np.full(N, -1, dtype=np.int64)
        past_offset = (m - 1) * tau
        if past_offset < N:
            past[past_offset:] = ordinal_patterns(x, m, tau, self.ties)
        present = np.searchsorted(self.edges, x, side="right").astype(np.int64)
        return Encoding(past, present, past_offset, 0)

    def params(self):
        return {"name": "Ordinal", "order": self.order, "delay": self.delay, "ties": self.ties}


def _spike_indicator(x):
    return (np.asarray(x) > 0).astype(np.int64)


def _pooled_intervals(series):
    return np.concatenate(
        [np.diff(np.flatnonzero(_spike_indicator(s))) for s in series] + [np.zeros(0, dtype=np.int64)]
    )


@dataclass(repr=False)
class ExponentialBins(Discretizer):
    """
    Binary spike history in bins of exponentially growing width :cite:`Rudelt2021`.

    The past window of `window` bins ending at time ``t`` is split into `d` sub-bins
    whose widths grow as :math:`\\tau_1 10^{j \\kappa}` (:math:`j = 0` the most recent,
    :math:`\\tau_1` chosen so that the widths add up to `window`, edges rounded to
    whole bins); the past symbol records which sub-bins hold a spike (``2**d``
    symbols). With ``kappa = 0`` the sub-bins are uniform, the multi-bin messages of
    :cite:`Ito2011`. The present is the spike indicator of bin ``t``. Stacked lags
    are spaced by ``lag_step = window``.
    """

    d: int = 3
    kappa: float = 0.0
    window: int = 8
    rank_based = True
    present_alphabet = 2

    def edges(self):
        """
        Sub-bin boundaries in bins before ``t``: sub-bin ``j`` covers lags ``edges[j]`` to ``edges[j+1] - 1``.
        """
        if self.window < self.d:
            raise ValueError(f"window ({self.window}) must hold at least d ({self.d}) bins")
        widths = 10.0 ** (self.kappa * np.arange(self.d))
        edges = np.round(np.concatenate([[0.0], np.cumsum(widths)]) * self.window / widths.sum()).astype(np.int64)
        for j in range(1, self.d + 1):
            edges[j] = max(edges[j], edges[j - 1] + 1)
        edges[-1] = self.window
        for j in range(self.d - 1, 0, -1):
            edges[j] = min(edges[j], edges[j + 1] - 1)
        return edges

    def fit(self, series):
        self.edges()
        self.past_alphabet = 2**self.d
        self.lag_step = self.window
        return self

    def encode(self, x):
        s = _spike_indicator(x)
        N, offset = len(s), self.window - 1
        past = np.full(N, -1, dtype=np.int64)
        if offset < N:
            cum = np.concatenate([[0], np.cumsum(s)])
            t = np.arange(offset, N)
            edges = self.edges()
            code = np.zeros(len(t), dtype=np.int64)
            for j in range(self.d):
                spikes = cum[t - edges[j] + 1] - cum[t - edges[j + 1] + 1]
                code |= (spikes > 0).astype(np.int64) << j
            past[offset:] = code
        return Encoding(past, s, offset, 0)

    def params(self):
        return {"name": "ExponentialBins", "d": self.d, "kappa": self.kappa, "window": self.window}


@dataclass(repr=False)
class SpikeIntervals(Discretizer):
    """
    The time since the last spike and the preceding inter-spike intervals, binned.

    The past symbol at time ``t`` encodes the elapsed time since the most recent spike
    and the `k` - 1 intervals between the `k` most recent spikes, each in `bins`
    equal-frequency bins (edges at quantiles of the pooled elapsed times and of the
    pooled inter-spike intervals, respectively), plus one symbol for "fewer than `k`
    spikes in the last `max_interval` bins" (``bins**k + 1`` symbols). This is a
    binned form of the interval history of continuous-time transfer entropy
    :cite:`Shorten2021`. Equal-frequency bins of the intervals are those of their
    logarithm, which make interval entropies independent of the firing rate and less
    biased than linear interval bins :cite:`Dorval2008,Dorval2011`. The elapsed time
    gets its own edges because it ranges over the whole interval, where the hazard of
    a renewal neuron rises, and edges at interval quantiles would lump all of its
    early part into one bin. The present is the spike indicator of bin ``t``.
    """

    k: int = 1
    bins: int = 3
    max_interval: int = 50
    elapsed_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    interval_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rank_based = True
    present_alphabet = 2

    def _elapsed(self, x):
        spikes = np.flatnonzero(_spike_indicator(x))
        t = np.arange(len(x))
        n = np.searchsorted(spikes, t, side="right")
        elapsed = t[n > 0] - spikes[n[n > 0] - 1]
        return elapsed[elapsed < self.max_interval]

    def fit(self, series):
        probs = np.linspace(0, 1, self.bins + 1)[1:-1]

        def edges(values):
            if not len(values):
                return np.zeros(self.bins - 1)
            return np.quantile(values, probs, method="inverted_cdf").astype(float)

        self.elapsed_edges = edges(np.concatenate([self._elapsed(s) for s in series]))
        self.interval_edges = edges(_pooled_intervals(series))
        self.past_alphabet = self.bins**self.k + 1
        self.lag_step = self.max_interval
        return self

    def encode(self, x):
        s = _spike_indicator(x)
        spikes = np.flatnonzero(s)
        N, offset = len(s), self.max_interval - 1
        past = np.full(N, -1, dtype=np.int64)
        if offset < N:
            t = np.arange(offset, N)
            n = np.searchsorted(spikes, t, side="right")
            ok = n >= self.k
            idx = np.where(ok, n, self.k)
            if len(spikes):
                ok &= t - spikes[np.minimum(idx - self.k, len(spikes) - 1)] < self.max_interval
            code = np.zeros(len(t), dtype=np.int64)
            if ok.any():
                tt, nn = t[ok], n[ok]
                codes = [np.searchsorted(self.elapsed_edges, tt - spikes[nn - 1], side="right")] + [
                    np.searchsorted(self.interval_edges, spikes[nn - i] - spikes[nn - i - 1], side="right")
                    for i in range(1, self.k)
                ]
                for c in codes:
                    code[ok] = code[ok] * self.bins + c
            past[offset:] = np.where(ok, code, self.bins**self.k)
        return Encoding(past, s, offset, 0)

    def params(self):
        return {"name": "SpikeIntervals", "k": self.k, "bins": self.bins, "max_interval": self.max_interval}


@dataclass(repr=False)
class OrdinalIntervals(Discretizer):
    """
    Ordinal pattern of the most recent inter-spike intervals.

    The past symbol at time ``t`` is the ordinal pattern :cite:`Bandt2002` of the
    `order` intervals between the ``order + 1`` most recent spikes (oldest first),
    or one extra symbol if those spikes do not all fall in the last `max_interval`
    bins (``order! + 1`` symbols). Ordinal patterns of interval sequences carry the
    temporal structure of firing, invariant to any monotone rescaling of the
    intervals :cite:`Masoliver2020`. The present is the spike indicator of bin ``t``.
    """

    order: int = 3
    max_interval: int = 50
    ties: str = "first"
    rank_based = True
    present_alphabet = 2

    def fit(self, series):
        self.past_alphabet = factorial(self.order) + 1
        self.lag_step = self.max_interval
        return self

    def encode(self, x):
        s = _spike_indicator(x)
        spikes = np.flatnonzero(s)
        N, offset, m = len(s), self.max_interval - 1, self.order
        past = np.full(N, -1, dtype=np.int64)
        if offset < N:
            t = np.arange(offset, N)
            symbol = np.full(len(t), factorial(m), dtype=np.int64)
            intervals = np.diff(spikes)
            if len(intervals) >= m:
                patterns = np.asarray(ordinal_patterns(intervals.astype(float), m, 1, self.ties))
                n = np.searchsorted(spikes, t, side="right")
                ok = n >= m + 1
                oldest = spikes[np.where(ok, n - m - 1, 0)]
                ok &= t - oldest < self.max_interval
                symbol[ok] = patterns[n[ok] - 1 - m]
            past[offset:] = symbol
        return Encoding(past, s, offset, 0)

    def params(self):
        return {"name": "OrdinalIntervals", "order": self.order, "max_interval": self.max_interval, "ties": self.ties}


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
