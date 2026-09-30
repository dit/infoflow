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
"""

from dataclasses import dataclass, field
from math import factorial

import numpy as np
from dit.inference import permutation_entropy, select_markov_order, weighted_permutation_entropy
from dit.inference._symbols import as_generator

from ..data import DiscreteData, as_trials
from ..embedding import Embedding
from .delays import delay_candidates
from .discretizers import Discretizer, EqualFrequency, EqualWidth, Ordinal, discretize
from .scaling import block_entropy_scaling
from .scoring import context_codes, score_contexts
from .screening import screen, split_missing

__all__ = ("NodeReport", "PreprocessResult", "PreprocessingReport", "preprocess")


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


def _discretizer(c):
    if c["kind"] == "ordinal":
        return Ordinal(c["order"], c["delay"], c.get("ties", "first"))
    if c["kind"] == "equal_width":
        return EqualWidth(c["bins"])
    return EqualFrequency(c["bins"])


def _symbols(c, series):
    d = _discretizer(c).fit(series)
    return d, [d.encode(s).past for s in series]


def _context_lags(c, L=None):
    """
    Lags (relative to t) of the past symbols forming the context.
    """
    if c["kind"] == "ordinal":
        step = (c["order"] - 1) * c["delay"] + 1
        count = 1 if L is None else L
    else:
        step = c["delay"]
        count = c["dims"] if L is None else L
    return [1 + k * step for k in range(count)], step


def _score(c, series, resolutions, folds):
    d, symbols = _symbols(c, series)
    lags, step = _context_lags(c)
    gap = max(lags) + (c["order"] - 1) * c["delay"] if c["kind"] == "ordinal" else max(lags)
    contexts = context_codes(symbols, lags, int(d.past_alphabet))
    return score_contexts(contexts, series, resolutions, folds, gap)


def _plateau(c, series, resolutions, folds, N, max_alphabet_fraction, max_history, rng, n_boot=500):
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
        mean, se, _, fs = score_contexts(contexts, series, resolutions, folds, max(lags) + step)
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

    d, symbols = _symbols(chosen, series)
    subsampled = [s[s >= 0][::step] if chosen["kind"] == "ordinal" else s for s in symbols]
    from dit.inference import Trials

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
    pe_data = Trials(series) if len(series) > 1 else series[0]
    report.permutation_entropy = permutation_entropy(pe_data, 3, 1, normalize=True)
    report.weighted_permutation_entropy = weighted_permutation_entropy(pe_data, 3, 1, normalize=True)
    embedding = Embedding(max_lag=int(lag_budget), tau=int(chosen["delay"]) if chosen["kind"] != "ordinal" else 1)
    return _discretizer(chosen), embedding, report


def _from_fixed(fixed, taus):
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
        raise ValueError("only Ordinal, EqualFrequency, and EqualWidth discretizers can be scored; discretize manually")
    raise ValueError(f"unknown discretizer {fixed!r}")


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
    map_fn=map,
    prng=None,
):
    """
    Discretize raw series and choose per-node embeddings, hands-off.

    Parameters
    ----------
    data : array_like or Trials
        Raw series, shape (samples, processes), or independent trials; NaNs split trials.
    names : list of str, None
    max_lag : int, None
        Upper bound on any node's lag budget (default: 4 times the largest delay
        candidate, at most 50).
    discretizer : Discretizer, str, list, None
        Override the automatic choice: an :class:`Ordinal`, :class:`EqualFrequency`,
        or :class:`EqualWidth` instance, ``'ordinal'``, ``'equal_frequency'``, or one
        per process.
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
    map_fn : callable
        ``map``-like function over processes.
    prng : None, int, Generator

    Returns
    -------
    PreprocessResult
    """
    rng = as_generator(prng)
    trials = [np.asarray(t, dtype=float) for t in as_trials(data)]
    P = trials[0].shape[1]
    names = list(names) if names is not None else [f"x{p}" for p in range(P)]
    screens = screen(trials)
    clean = split_missing(trials)
    settings = {
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
        return _node(p, names[p], series, screens[p], settings, as_generator(int(seeds[p])))

    results = list(map_fn(run, range(P)))
    discretizers = [r[0] for r in results]
    embeddings = [r[1] for r in results]
    if max_lag is not None:
        for e in embeddings:
            e.max_lag = min(e.max_lag, max_lag)
    report = PreprocessingReport(nodes=[r[2] for r in results], settings=settings)
    discrete = discretize(clean, discretizers, names)
    return PreprocessResult(discrete, embeddings, report)
