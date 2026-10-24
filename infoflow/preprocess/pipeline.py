"""
The hands-off preprocessing pipeline.

For each process, from that process's series alone:

1. **Screen** for missing data (trials are split at gaps), ties, and nonstationarity.
2. **Propose delays** from the auto-mutual information, the mean curvature of the
   delay reconstruction, and the autocorrelation.
3. **Choose the discretizer and embedding jointly** among ordinal patterns
   ``(m, tau)`` and equal-frequency bins ``(B, d, tau)`` by the cross-validated
   multi-resolution score :math:`\\bar R` (:mod:`~infoflow.preprocess.scoring`),
   taking the smallest alphabet within one standard error of the best
   (:cite:`Garland2015`: low-dimensional reconstructions often suffice). If a chosen
   ordinal encoding loses clearly more predictability than its amplitude-weighted
   version, amplitude matters and the best binned candidate within one standard
   error is used instead :cite:`Garland2014`.
4. **Set the lag budget** at the Rudelt-style plateau onset of :math:`\\bar R(L)`
   as a function of history length :cite:`Rudelt2021`, cross-checked against the
   BIC Markov order, and capped where contexts become undersampled.
5. **Report** the entropy rate and excess entropy from the scaling region of the
   block entropy :cite:`Deshmukh2021`, and every score, choice, and warning.

Choosing each node's parameters without looking at other nodes cannot create
edges, so the later tests need no correction for this selection.

Spike trains (``spikes=True``, or :class:`~infoflow.spikes.SpikeTrains` input) skip
steps 2 and 3: each neuron's past is chosen among spike-history encodings by its
cross-validated history dependence at predicting its spike indicator (:doc:`spikes`).

With ``network_resolution``, a sixth step looks across nodes once: the per-node
choices optimize each node's prediction from its own past, which can leave the joint
contexts of parent selection (own past *and* several parents) far too sparse, as on
smoothly sampled flows such as Lorenz-96. :func:`~infoflow.preprocess.scoring.network_scores`
scores uniform resolutions :math:`(B, L)` by predicting each target from its own past
and its most correlated sources, and the best is adopted for every node (``'auto'``:
only if every node keeps at least ``network_retention`` of its own per-node score, so
nodes whose dynamics need fine symbols, such as a strongly oscillating driver, keep
them). This is one network-wide choice among a few resolutions, not an edge-wise
one; :doc:`benchmarks` checks that it does not raise false positives on empty
networks.
"""

from dataclasses import dataclass, field
from math import factorial

import numpy as np
from dit.inference import permutation_entropy, select_markov_order, weighted_permutation_entropy
from dit.inference._symbols import as_generator

from ..data import DiscreteData, as_trials
from ..embedding import Embedding
from .delays import delay_candidates
from .discretizers import (
    Discretizer,
    EqualFrequency,
    EqualWidth,
    ExponentialBins,
    Ordinal,
    OrdinalIntervals,
    SpikeIntervals,
    _spike_indicator,
    discretize,
)
from .scaling import block_entropy_scaling
from .scoring import context_codes, network_scores, score_contexts
from .screening import screen, split_missing

__all__ = ("SPIKE_GRID", "NodeReport", "PreprocessResult", "PreprocessingReport", "preprocess")

#: The candidate grid for spike trains: past windows (in bins; None for seven
#: log-spaced windows from 1 bin to the history limit), ExponentialBins depths and
#: exponents, SpikeIntervals interval counts and bins, OrdinalIntervals orders, and
#: the range of source delays (in bins) every spike node offers as a source.
SPIKE_GRID = {
    "windows": None,
    "source_history": 30,
    "depths": (1, 2, 3, 4, 5),
    "kappas": (0.0, 0.1, 0.2, 0.3, 0.5),
    "interval_counts": (1, 2, 3),
    "interval_bins": (2, 3, 4, 6),
    "interval_orders": (2, 3, 4),
}


@dataclass
class NodeReport:
    """
    Everything preprocessing decided for one process, and why.
    """

    name: str
    screen: object = None
    delays: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    chosen: dict = field(default_factory=dict)
    rule: str = ""
    history: dict = field(default_factory=dict)
    lag_budget: int = 1
    storage: float = 0.0
    markov_order: int | None = None
    permutation_entropy: float = float("nan")
    weighted_permutation_entropy: float = float("nan")
    entropy_rate: float = float("nan")
    excess_entropy: float = float("nan")
    flags: list = field(default_factory=list)


@dataclass
class PreprocessingReport:
    """
    The per-node reports and the settings used.
    """

    nodes: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    def summary(self):
        descs = [
            ", ".join(f"{k}={v}" for k, v in n.chosen.items() if k not in ("score", "se", "alphabet"))
            for n in self.nodes
        ]
        width = max([len("discretizer"), *map(len, descs)])
        lines = [f"{'node':<8}  {'discretizer':<{width}}  {'score':>6}  {'lag budget':>10}  {'storage':>7}  flags"]
        for n, desc in zip(self.nodes, descs, strict=True):
            lines.append(
                f"{n.name:<8}  {desc:<{width}}  {n.chosen.get('score', 0):6.3f}  {n.lag_budget:10d}  "
                f"{n.storage:7.3f}  {','.join(n.flags)}"
            )
        net = self.network
        if net:
            verdict = "adopted" if net["adopted"] else "not adopted"
            lines.append(
                f"network resolution: {net['bins']} bins, history {net['history']} ({verdict}; "
                f"smallest retained own score {min(net['retained']):.2f})"
            )
        return "\n".join(lines)


@dataclass
class PreprocessResult:
    """
    The discretized data, per-node embeddings, and the report.
    """

    data: DiscreteData
    embeddings: list
    report: PreprocessingReport


def _candidates(series, taus, N, orders, bins, dims, ties, max_alphabet_fraction, equal_width=False):
    cap = max(N * max_alphabet_fraction, 2)
    out = []
    for tau in taus:
        for m in orders:
            if factorial(m) <= cap:
                out.append({"kind": "ordinal", "order": m, "delay": tau, "alphabet": factorial(m), "ties": ties})
        for B in bins:
            for d in dims:
                if B**d <= cap and (d > 1 or tau == taus[0]):
                    out.append({"kind": "equal_frequency", "bins": B, "dims": d, "delay": tau, "alphabet": B**d})
                    if equal_width:
                        out.append({"kind": "equal_width", "bins": B, "dims": d, "delay": tau, "alphabet": B**d})
    return out


def _spike_windows(max_history, windows=None):
    if windows is not None:
        return tuple(int(w) for w in windows)
    return tuple(int(w) for w in np.unique(np.round(np.geomspace(1, max(max_history, 1), 7))))


def _spike_candidates(N, windows, grid, max_alphabet_fraction):
    cap = max(N * max_alphabet_fraction, 2)
    out, seen = [], set()
    full = max(grid["depths"])
    for W in windows:
        for d in grid["depths"]:
            for kappa in grid["kappas"] if d > 1 else (0.0,):
                # Windows short enough to resolve every bin do so: merged recent bins
                # would blur the timing a mediating source needs to screen off others.
                if d > W or 2**d > cap or (full >= W and d < W):
                    continue
                edges = tuple(ExponentialBins(d, kappa, W).edges())
                if edges not in seen:
                    seen.add(edges)
                    out.append({"kind": "exponential_bins", "d": d, "kappa": kappa, "window": W, "alphabet": 2**d})
        if W < 4:
            continue
        for k in grid["interval_counts"]:
            for B in grid["interval_bins"]:
                if B**k + 1 <= cap:
                    out.append({"kind": "spike_intervals", "k": k, "bins": B, "max_interval": W, "alphabet": B**k + 1})
        for m in grid["interval_orders"]:
            if factorial(m) + 1 <= cap:
                out.append({"kind": "ordinal_intervals", "order": m, "max_interval": W, "alphabet": factorial(m) + 1})
    return out


def _discretizer(c):
    if c["kind"] == "ordinal":
        return Ordinal(c["order"], c["delay"], c.get("ties", "first"))
    if c["kind"] == "equal_width":
        return EqualWidth(c["bins"])
    if c["kind"] == "exponential_bins":
        return ExponentialBins(c["d"], c["kappa"], c["window"])
    if c["kind"] == "spike_intervals":
        return SpikeIntervals(c["k"], c["bins"], c["max_interval"])
    if c["kind"] == "ordinal_intervals":
        return OrdinalIntervals(c["order"], c["max_interval"])
    return EqualFrequency(c["bins"])


def _window(c):
    """
    The number of samples one past symbol summarizes (None for single-sample bins).
    """
    if c["kind"] == "ordinal":
        return (c["order"] - 1) * c["delay"] + 1
    if c["kind"] == "exponential_bins":
        return c["window"]
    if c["kind"] in ("spike_intervals", "ordinal_intervals"):
        return c["max_interval"]
    return None


def _resolves_bins(c):
    return c["kind"] == "exponential_bins" and c["d"] == c["window"]


def _symbols(c, series):
    d = _discretizer(c).fit(series)
    return d, [d.encode(s).past for s in series]


def _context_lags(c, L=None):
    """
    Lags (relative to t) of the past symbols forming the context.
    """
    window = _window(c)
    if window is not None:
        step = window
        count = 1 if L is None else L
    else:
        step = c["delay"]
        count = c["dims"] if L is None else L
    return [1 + k * step for k in range(count)], step


def _score(c, series, resolutions, folds, targets=None):
    d, symbols = _symbols(c, series)
    lags, step = _context_lags(c)
    window = _window(c)
    gap = max(lags) + window - 1 if window is not None else max(lags)
    contexts = context_codes(symbols, lags, int(d.past_alphabet))
    return score_contexts(contexts, series, resolutions, folds, gap, targets=targets)


def _plateau(c, series, resolutions, folds, N, max_alphabet_fraction, max_history, rng, n_boot=500, targets=None):
    """
    Rudelt-style plateau onset of the score as a function of history length.
    """
    d, symbols = _symbols(c, series)
    alphabet = int(d.past_alphabet)
    curve, ses, folds_scores, lengths = [], [], [], []
    L = 1
    while True:
        lags, step = _context_lags(c, L)
        if alphabet**L > max(N * max_alphabet_fraction, 2) or max(lags) > max_history:
            break
        contexts = context_codes(symbols, lags, alphabet)
        mean, se, _, fs = score_contexts(contexts, series, resolutions, folds, max(lags) + step, targets=targets)
        curve.append(mean)
        ses.append(se)
        folds_scores.append(fs)
        lengths.append(L)
        L += 1
        if L > 12:
            break
    if not curve:
        return 1, {"lengths": [], "scores": [], "se": []}, 0.0, 1
    best = int(np.argmax(curve))
    fs = folds_scores[best]
    if len(fs) > 1:
        boots = [np.mean(rng.choice(fs, size=len(fs), replace=True)) for _ in range(n_boot)]
        lower = float(np.quantile(boots, 0.025))
    else:
        lower = curve[best]
    onset = next(i for i, v in enumerate(curve) if v >= lower)
    L_D = lengths[onset]
    lags, step = _context_lags(c, L_D)
    storage = float(np.mean(curve[onset:]))
    history = {"lengths": lengths, "scores": curve, "se": ses, "plateau_lower": lower, "onset": L_D}
    return max(lags), history, storage, step


def _node(p, name, series, screen_result, settings, rng):
    N = sum(len(s) for s in series)
    report = NodeReport(name=name, screen=screen_result)
    if screen_result.nonstationary:
        report.flags.append("nonstationary")
    ties = "noise" if screen_result.tie_fraction > 0.05 else "first"
    if screen_result.tie_fraction > 0.05:
        report.flags.append("ties")
    taus, details = delay_candidates(series, settings["max_delay"])
    report.delays = {k: v["delay"] for k, v in details.items()}

    if settings["discretizer"] is not None:
        fixed = settings["discretizer"]
        fixed = fixed[p] if isinstance(fixed, (list, tuple)) else fixed
        chosen = _from_fixed(fixed, taus)
        mean, se, per, _ = _score(chosen, series, settings["resolutions"], settings["folds"])
        chosen.update(score=mean, se=se)
        report.candidates = [dict(chosen)]
        report.rule = "user"
    else:
        cands = _candidates(
            series,
            taus,
            N,
            settings["orders"],
            settings["bins"],
            settings["dims"],
            ties,
            settings["max_alphabet_fraction"],
            settings["equal_width"],
        )
        for c in cands:
            mean, se, per, _ = _score(c, series, settings["resolutions"], settings["folds"])
            c.update(score=mean, se=se, per_resolution={B: v[0] for B, v in per.items()})
        report.candidates = cands
        best = max(cands, key=lambda c: c["score"])
        threshold = best["score"] - (best["se"] if np.isfinite(best["se"]) else 0.0)
        eligible = sorted((c for c in cands if c["score"] >= threshold), key=lambda c: (c["alphabet"], -c["score"]))
        chosen = eligible[0]
        report.rule = "one-standard-error"
        if threshold <= 0:
            # No candidate predicts the node's own future, so its own dynamics cannot
            # choose a symbolization; keep the middle reference resolution as bins.
            B = sorted(settings["resolutions"])[len(settings["resolutions"]) // 2]
            match = [c for c in cands if c["kind"] == "equal_frequency" and c["bins"] == B and c["dims"] == 1]
            chosen = (
                match[0]
                if match
                else {"kind": "equal_frequency", "bins": B, "dims": 1, "delay": taus[0], "alphabet": B}
            )
            report.rule = "unpredictable: middle reference resolution"
            report.flags.append("unpredictable")
        elif chosen["kind"] == "ordinal":
            pooled = series if len(series) > 1 else series[0]
            from dit.inference import Trials

            data = Trials(series) if len(series) > 1 else pooled
            pe = permutation_entropy(data, chosen["order"], chosen["delay"], normalize=True)
            wpe = weighted_permutation_entropy(data, chosen["order"], chosen["delay"], normalize=True)
            if pe - wpe > settings["wpe_margin"]:
                binned = [c for c in eligible if c["kind"] != "ordinal"]
                if binned:
                    chosen = binned[0]
                    report.rule = "one-standard-error; amplitude (WPE) favours bins"
    report.chosen = {k: v for k, v in chosen.items() if k != "per_resolution"}

    lag_budget, history, storage, step = _plateau(
        chosen,
        series,
        settings["resolutions"],
        settings["folds"],
        N,
        settings["max_alphabet_fraction"],
        settings["max_history"],
        rng,
    )
    report.lag_budget, report.history, report.storage = int(lag_budget), history, storage

    from dit.inference import Trials

    _symbolic_diagnostics(report, chosen, series, step, history)
    pe_data = Trials(series) if len(series) > 1 else series[0]
    report.permutation_entropy = permutation_entropy(pe_data, 3, 1, normalize=True)
    report.weighted_permutation_entropy = weighted_permutation_entropy(pe_data, 3, 1, normalize=True)
    embedding = Embedding(max_lag=int(lag_budget), tau=int(chosen["delay"]) if chosen["kind"] != "ordinal" else 1)
    return _discretizer(chosen), embedding, report


def _symbolic_diagnostics(report, chosen, series, step, history):
    """
    BIC Markov order and block-entropy scaling of the chosen past symbols.
    """
    from dit.inference import Trials

    d, symbols = _symbols(chosen, series)
    subsampled = [s[s >= 0][::step] if _window(chosen) is not None else s for s in symbols]
    sym_data = Trials(subsampled) if len(subsampled) > 1 else subsampled[0]
    L_max = max(history["lengths"]) if history["lengths"] else 1
    try:
        report.markov_order = int(select_markov_order(sym_data, L_max, method="bic"))
    except ValueError:
        report.markov_order = None
    if report.markov_order is not None and history.get("onset") is not None and report.markov_order != history["onset"]:
        report.flags.append("order-disagreement")
    try:
        region, _, _ = block_entropy_scaling(sym_data)
        report.entropy_rate, report.excess_entropy = region.slope, region.intercept
    except ValueError:
        pass


def _spike_node(p, name, series, screen_result, settings, rng):
    """
    Choose a spike-history encoding for one neuron by its cross-validated history
    dependence :math:`R` :cite:`Rudelt2021`: the score of every candidate predicts the
    spike indicator of the present bin.
    """
    N = sum(len(s) for s in series)
    report = NodeReport(name=name, screen=screen_result)
    if screen_result.nonstationary:
        report.flags.append("nonstationary")
    targets = [_spike_indicator(s) for s in series]
    if not any(t.any() for t in targets):
        report.flags.append("silent")
    resolutions, folds = settings["resolutions"], settings["folds"]
    windows = _spike_windows(settings["max_history"], settings["spike_grid"]["windows"])
    report.delays = {"windows": list(windows)}

    if settings["discretizer"] is not None:
        fixed = settings["discretizer"]
        fixed = fixed[p] if isinstance(fixed, (list, tuple)) else fixed
        chosen = _from_fixed(fixed, (1,))
        mean, se, _, _ = _score(chosen, series, resolutions, folds, targets)
        chosen.update(score=mean, se=se)
        report.candidates = [dict(chosen)]
        report.rule = "user"
    else:
        cands = _spike_candidates(N, windows, settings["spike_grid"], settings["max_alphabet_fraction"])
        for c in cands:
            mean, se, _, _ = _score(c, series, resolutions, folds, targets)
            c.update(score=mean, se=se)
        report.candidates = cands
        best = max(cands, key=lambda c: c["score"])
        threshold = best["score"] - (best["se"] if np.isfinite(best["se"]) else 0.0)
        # As a source the node's past symbols must resolve spike timing, which its own
        # score cannot see: within one standard error, words resolving every bin come
        # first, then the smallest alphabet and the shortest window.
        eligible = sorted(
            (c for c in cands if c["score"] >= threshold),
            key=lambda c: (not _resolves_bins(c), c["alphabet"], _window(c), -c["score"]),
        )
        chosen = eligible[0]
        report.rule = "one-standard-error"
        if threshold <= 0:
            # No history predicts the neuron's spiking; keep the previous bin's spike.
            chosen = {"kind": "exponential_bins", "d": 1, "kappa": 0.0, "window": 1, "alphabet": 2}
            report.rule = "unpredictable: previous bin"
            report.flags.append("unpredictable")
    report.chosen = dict(chosen)

    lag_budget, history, storage, step = _plateau(
        chosen,
        series,
        resolutions,
        folds,
        N,
        settings["max_alphabet_fraction"],
        settings["max_history"],
        rng,
        targets=targets,
    )
    report.lag_budget, report.history, report.storage = int(lag_budget), history, storage
    _symbolic_diagnostics(report, chosen, series, step, history)
    # Synaptic delays outlast most own-history plateaus, so the lag budget also spans
    # the source delays.
    budget = max(int(lag_budget), min(settings["spike_grid"]["source_history"], settings["max_history"]))
    report.lag_budget = budget
    return _discretizer(chosen), Embedding(max_lag=budget, tau=1), report


def _from_fixed(fixed, taus):
    if isinstance(fixed, ExponentialBins):
        return {
            "kind": "exponential_bins",
            "d": fixed.d,
            "kappa": fixed.kappa,
            "window": fixed.window,
            "alphabet": 2**fixed.d,
        }
    if isinstance(fixed, SpikeIntervals):
        return {
            "kind": "spike_intervals",
            "k": fixed.k,
            "bins": fixed.bins,
            "max_interval": fixed.max_interval,
            "alphabet": fixed.bins**fixed.k + 1,
        }
    if isinstance(fixed, OrdinalIntervals):
        return {
            "kind": "ordinal_intervals",
            "order": fixed.order,
            "max_interval": fixed.max_interval,
            "alphabet": factorial(fixed.order) + 1,
        }
    if isinstance(fixed, Ordinal):
        return {
            "kind": "ordinal",
            "order": fixed.order,
            "delay": fixed.delay,
            "alphabet": factorial(fixed.order),
            "ties": fixed.ties,
        }
    if isinstance(fixed, EqualFrequency):
        return {"kind": "equal_frequency", "bins": fixed.bins, "dims": 1, "delay": taus[0], "alphabet": fixed.bins}
    if isinstance(fixed, EqualWidth):
        return {"kind": "equal_width", "bins": fixed.bins, "dims": 1, "delay": taus[0], "alphabet": fixed.bins}
    if isinstance(fixed, str):
        if fixed == "ordinal":
            return {"kind": "ordinal", "order": 3, "delay": 1, "alphabet": 6, "ties": "first"}
        if fixed in ("equal_frequency", "bins"):
            return {"kind": "equal_frequency", "bins": 4, "dims": 1, "delay": 1, "alphabet": 4}
    if isinstance(fixed, Discretizer):
        raise ValueError(
            "only Ordinal, EqualFrequency, EqualWidth, ExponentialBins, SpikeIntervals, and OrdinalIntervals "
            "discretizers can be scored; discretize manually"
        )
    raise ValueError(f"unknown discretizer {fixed!r}")


def _network_resolution(trials, results, discretizers, embeddings, settings, mode, n_sources, retention):
    """
    The uniform network resolution, and whether it replaces the per-node choices.
    """
    histories = (1, 2) if settings["max_history"] >= 2 else (1,)
    N = sum(len(t) for t in trials)
    bins = tuple(B for B in settings["bins"] if B >= 2 and B ** (max(histories) + n_sources) <= N)
    scores = network_scores(trials, bins or (2,), histories, n_sources, settings["resolutions"], settings["folds"])
    B, L = max(scores, key=lambda k: scores[k])
    candidate = {"kind": "equal_frequency", "bins": B, "dims": L, "delay": 1, "alphabet": B**L}
    retained = []
    for p, (_, _, node) in enumerate(results):
        best = node.chosen.get("score", 0.0)
        own = _score(candidate, [t[:, p] for t in trials], settings["resolutions"], settings["folds"])[0]
        retained.append(float(own / best) if best > 0 else 1.0)
    adopted = mode is True or min(retained) >= retention
    network = {"bins": B, "history": L, "scores": scores, "retained": retained, "adopted": bool(adopted)}
    if adopted:
        discretizers = [EqualFrequency(B) for _ in discretizers]
        embeddings = [Embedding(max_lag=L) for _ in embeddings]
    return discretizers, embeddings, network


def preprocess(
    data,
    names=None,
    max_lag=None,
    discretizer=None,
    max_delay=20,
    resolutions=(2, 4, 8),
    folds=5,
    orders=(2, 3, 4, 5),
    bins=(2, 3, 4, 6, 8),
    dims=(1, 2, 3),
    max_alphabet_fraction=0.1,
    max_history=None,
    wpe_margin=0.05,
    equal_width=False,
    network_resolution=False,
    network_sources=2,
    network_retention=0.7,
    spikes=None,
    spike_grid=None,
    map_fn=map,
    prng=None,
):
    """
    Discretize raw series and choose per-node embeddings, hands-off.

    Parameters
    ----------
    data : array_like, Trials, or SpikeTrains
        Raw series, shape (samples, processes), or independent trials; NaNs split
        trials. :class:`~infoflow.spikes.SpikeTrains` are binned to spike counts.
    names : list of str, None
    max_lag : int, None
        Upper bound on any node's lag budget (default: 4 times the largest delay
        candidate, at most 50).
    discretizer : Discretizer, str, list, None
        Override the automatic choice: an :class:`Ordinal`, :class:`EqualFrequency`,
        :class:`EqualWidth`, :class:`ExponentialBins`, :class:`SpikeIntervals`, or
        :class:`OrdinalIntervals` instance, ``'ordinal'``, ``'equal_frequency'``, or
        one per process.
    max_delay : int
        Largest delay considered by the delay heuristics.
    resolutions : tuple of int
        Reference resolutions for the score.
    folds : int
        Blocked cross-validation folds.
    orders, bins, dims : tuples
        The candidate grid: ordinal orders, bin counts, and binned embedding dimensions.
    max_alphabet_fraction : float
        Contexts may have at most this fraction of the sample size in symbols.
    max_history : int, None
        Largest lag considered for the lag budget (default `max_lag`).
    wpe_margin : float
        Normalized-PE minus WPE above which amplitude is judged informative.
    equal_width : bool
        Also score equal-width bins. They keep amplitude, so they resolve heavy
        tails, where a parent's nonlinear effect (e.g. through :math:`x^2`) varies
        most, far better than equal-frequency bins; the price is that the pipeline
        is no longer invariant to monotone transforms of the data.
    network_resolution : {False, 'auto', True}
        Replace the per-node choices by one uniform resolution (equal-frequency bins
        and history length) scored for parent selection
        (:func:`~infoflow.preprocess.scoring.network_scores`): always (True), or only
        if every node keeps at least `network_retention` of its per-node score
        (``'auto'``). The decision is recorded in ``report.network``.
    network_sources : int
        Sources per target in the network score.
    network_retention : float
        Fraction of each node's own per-node score that the uniform resolution must
        keep under ``'auto'``.
    spikes : bool, None
        Treat every process as a binned spike train (bins with a spike are nonzero):
        each node's past is encoded by the best of :class:`ExponentialBins`,
        :class:`SpikeIntervals`, and :class:`OrdinalIntervals` at its predicted
        spiking, and its present is the spike indicator. None: only for
        :class:`~infoflow.spikes.SpikeTrains` input. The history limit then defaults
        to 200 bins (1 s at 5 ms), and network resolution is not used.
    spike_grid : dict, None
        Entries overriding :data:`SPIKE_GRID`.
    map_fn : callable
        ``map``-like function over processes.
    prng : None, int, Generator

    Returns
    -------
    PreprocessResult
    """
    from ..spikes import SpikeTrains

    rng = as_generator(prng)
    if spikes is None:
        spikes = isinstance(data, SpikeTrains)
    if names is None and isinstance(data, SpikeTrains):
        names = data.names
    if max_history is None and max_lag is None and spikes:
        max_history = 200
    trials = [np.asarray(t, dtype=float) for t in as_trials(data)]
    P = trials[0].shape[1]
    names = list(names) if names is not None else [f"x{p}" for p in range(P)]
    screens = screen(trials)
    clean = split_missing(trials)
    settings = {
        "spikes": bool(spikes),
        "spike_grid": {**SPIKE_GRID, **(spike_grid or {})},
        "max_delay": max_delay,
        "resolutions": tuple(resolutions),
        "folds": folds,
        "orders": tuple(orders),
        "bins": tuple(bins),
        "dims": tuple(dims),
        "max_alphabet_fraction": max_alphabet_fraction,
        "max_history": max_history if max_history is not None else (max_lag if max_lag is not None else 50),
        "wpe_margin": wpe_margin,
        "discretizer": discretizer,
        "equal_width": equal_width,
    }
    seeds = rng.integers(0, 2**32, size=P)

    def run(p):
        series = [t[:, p] for t in clean]
        node = _spike_node if spikes else _node
        return node(p, names[p], series, screens[p], settings, as_generator(int(seeds[p])))

    results = list(map_fn(run, range(P)))
    discretizers = [r[0] for r in results]
    embeddings = [r[1] for r in results]
    network = {}
    if network_resolution and settings["discretizer"] is None and P > 1 and not spikes:
        discretizers, embeddings, network = _network_resolution(
            clean, results, discretizers, embeddings, settings, network_resolution, network_sources, network_retention
        )
    if max_lag is not None:
        for e in embeddings:
            e.max_lag = min(e.max_lag, max_lag)
    settings.update(
        network_resolution=network_resolution, network_sources=network_sources, network_retention=network_retention
    )
    report = PreprocessingReport(nodes=[r[2] for r in results], settings=settings, network=network)
    discrete = discretize(clean, discretizers, names)
    return PreprocessResult(discrete, embeddings, report)
