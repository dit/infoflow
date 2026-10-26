"""
Hands-off network inference: :func:`infer` chooses the methods from the data and a budget.

The choices rest on the validation in :doc:`validation` and :doc:`auto`. The
linear-Gaussian estimator is by far the most powerful on linear couplings, but it
conditions linearly: when a target depends nonlinearly on what is conditioned on, the
residual dependence makes its tests anti-conservative (false links between the
children of a nonlinear driver). The plug-in CMI on the preprocessing symbols is far
less powerful but stays precise. :func:`infer` therefore always selects with the
plug-in and adds the Gaussian estimator, in one greedy search at a Bonferroni-split
level (:class:`~infoflow.selection.SkeletonSettings`), for targets whose dynamics
pass a test for neglected nonlinearity (:func:`linearity_pvalues`); a target that
fails it keeps the Gaussian estimator, under the local permutation null, only if the
parents the Gaussian estimator selects alone predict it at least as well as the
plug-in's, by nearest-neighbour regression (:func:`parent_prediction_error`). Every
stage is tested at the strict level 0.001. It times the selection on a
few sampled targets first (the pilot, whose skeletons are reused), predicts the cost
of the whole run, and, if a ``time_budget`` is given, drops the most expensive options
in a fixed order until the prediction fits. Every choice and its reason is recorded in
an :class:`AutoReport`.
"""

import dataclasses
import inspect
import os
import re
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
from dit.inference._symbols import as_generator

from .checkpoint import Checkpoint
from .data import DiscreteData, as_trials
from .embedding import Embedding
from .network import estimate_edge, infer_multiplex
from .parallel import thread_map
from .selection import SkeletonSettings, select_parents

__all__ = ("PRESETS", "AutoReport", "AutoResult", "infer", "linearity_pvalues", "parent_prediction_error")

#: Selection significance level: precision stayed near 1 for every estimator at this level.
ALPHA = 0.001

#: The options of each preset. ``continuous`` / ``discrete`` are the estimator ensembles.
PRESETS = {
    "fast": {
        "continuous": ("plugin", "gaussian"),
        "discrete": ("plugin",),
        "include_shared_candidates": False,
        "interpret": False,
        "hyperedges": False,
        "network_resolution": "auto",
    },
    "balanced": {
        "continuous": ("plugin", "gaussian"),
        "discrete": ("plugin",),
        "include_shared_candidates": False,
        "interpret": True,
        "hyperedges": False,
        "network_resolution": "auto",
    },
    "thorough": {
        "continuous": ("plugin", "gaussian", "ksg"),
        "discrete": ("plugin",),
        "include_shared_candidates": True,
        "interpret": True,
        "hyperedges": True,
        "network_resolution": "auto",
    },
}

#: Level of the neglected-nonlinearity test after which the Gaussian estimator must earn
#: its place by held-out prediction; lenient, since a false alarm only costs time.
LINEARITY_ALPHA = 0.05

# Heuristic cost factors for options the pilot does not time directly.
_PRESCREEN_FACTOR = 0.7
_HYPEREDGE_FACTOR = 0.25
# Pre-screen networks with more processes than this.
_PRESCREEN_ABOVE = 30
_PILOT_TARGETS = 3
_MAX_RESAMPLES = 2000
_LAYER_ALPHA = 0.05
# With a budget, checks and targets not started by these fractions of it are skipped or
# selected with the plug-in only: per-target selection times are heavy-tailed, so the
# pilot's prediction alone cannot keep a run within its budget.
_GATE_BUDGET_FRACTION = 0.5
_SELECTION_BUDGET_FRACTION = 0.8


@dataclass
class AutoReport:
    """
    What :func:`infer` decided, and why.

    Attributes
    ----------
    diagnostics : dict
        Properties of the data: kind (discrete or continuous), samples, processes,
        trials, and flagged nodes.
    preset : str
    estimators : tuple
        The selection ensemble finally used.
    options : dict
        The final optional stages and selection options.
    downgrades : list of str
        Options removed to fit the time budget, in order, with the reason.
    pilot : dict
        Pilot targets, seconds per target per estimator, and the measured layer cost.
    predicted_seconds, actual_seconds : float
    time_budget : float, None
        Seconds.
    admitted : dict
        ``(source, target) -> estimator`` that admitted each parent.
    machines : dict
        With ``machines=True``: the machine layer's seconds, counts, and skipped
        machines with reasons (:meth:`~infoflow.machines.MachineLayer.report`). The
        layer is pinned, so it runs outside the time budget.
    warnings : list of str
    """

    diagnostics: dict = field(default_factory=dict)
    preset: str = "balanced"
    estimators: tuple = ()
    options: dict = field(default_factory=dict)
    downgrades: list = field(default_factory=list)
    pilot: dict = field(default_factory=dict)
    predicted_seconds: float = float("nan")
    actual_seconds: float = float("nan")
    time_budget: float | None = None
    admitted: dict = field(default_factory=dict)
    machines: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __str__(self):
        def fmt(seconds):
            return "n/a" if seconds is None or not np.isfinite(seconds) else str(timedelta(seconds=round(seconds)))

        d = self.diagnostics
        lines = [
            f"data: {d.get('kind')} ({d.get('n_samples')} samples, {d.get('n_processes')} processes, "
            f"{d.get('n_trials')} trial(s))",
            f"preset: {self.preset}; selection estimators: {', '.join(self.estimators)} (alpha={ALPHA}, "
            f"split {len(self.estimators)} ways)",
            "options: " + ", ".join(f"{k}={v}" for k, v in self.options.items()),
            f"time: predicted {fmt(self.predicted_seconds)}, actual {fmt(self.actual_seconds)}, "
            f"budget {fmt(self.time_budget)}",
        ]
        if d.get("nonlinear_targets"):
            kept = d.get("gaussian_kept", [])
            dropped = [t for t in d["nonlinear_targets"] if t not in kept]
            lines.append(
                f"{len(d['nonlinear_targets'])} target(s) test nonlinear; Gaussian selection dropped for "
                f"{', '.join(map(str, dropped)) or 'none'} and kept with a local permutation null for "
                f"{', '.join(map(str, kept)) or 'none'} (by held-out prediction of their parents)"
            )
        if d.get("network_resolution"):
            n = d["network_resolution"]
            lines.append(
                f"network resolution {n['bins']} bins, history {n['history']}: "
                f"{'adopted' if n['adopted'] else 'not adopted'} (smallest retained own score {n['min_retained']:.2f})"
            )
        if d.get("flagged"):
            lines.append("flagged nodes: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in d["flagged"].items()))
        lines += [f"downgrade: {g}" for g in self.downgrades]
        if self.admitted:
            counts = {}
            for name in self.admitted.values():
                counts[name] = counts.get(name, 0) + 1
            lines.append("parents admitted by: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        if self.machines:
            m = self.machines
            lines.append(
                f"machines: {m['nodes']} node(s), {m['edges']} edge(s), {m['targets']} target(s), "
                f"composed model {'built' if m['model'] else 'not built'}, in {fmt(m['seconds'])}"
            )
            lines += [f"machines skipped {key}: {reason}" for key, reason in m["skipped"].items()]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class AutoResult:
    """
    The inferred network and the report of how it was inferred.
    """

    network: object
    report: AutoReport


def _parse_budget(budget):
    if budget is None:
        return None
    if isinstance(budget, timedelta):
        return budget.total_seconds()
    if isinstance(budget, (int, float)):
        return float(budget)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", str(budget))
    if not match:
        raise ValueError(f"time_budget {budget!r}: use seconds, a timedelta, or e.g. '90s', '30m', '2h', '1d'")
    value, unit = float(match.group(1)), match.group(2) or "s"
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _split_overrides(overrides):
    """
    Pinned :class:`SkeletonSettings` fields and :func:`infer_multiplex` parameters.

    ``estimator`` is the selection estimator; ``layer_estimator`` is passed on as
    :func:`infer_multiplex`'s ``estimator`` (the layers' estimator).
    """
    overrides = dict(overrides)
    layer = {"estimator": overrides.pop("layer_estimator")} if "layer_estimator" in overrides else {}
    skeleton_fields = {f.name for f in dataclasses.fields(SkeletonSettings)}
    infer_params = set(inspect.signature(infer_multiplex).parameters) - {"data", "skeleton_settings", "estimator"}
    unknown = set(overrides) - skeleton_fields - infer_params
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}: not SkeletonSettings or infer_multiplex parameters")
    return (
        {k: v for k, v in overrides.items() if k in skeleton_fields},
        {k: v for k, v in overrides.items() if k in infer_params} | layer,
    )


def _diagnose(trials, names, prng, max_lag=None, network_resolution=False):
    """
    The discrete data, embeddings, preprocessing report, and diagnostics of `trials`.
    """
    discrete_input = all(np.issubdtype(t.dtype, np.integer) for t in trials)
    n_samples = int(sum(len(t) for t in trials))
    diagnostics = {
        "kind": "discrete" if discrete_input else "continuous",
        "n_samples": n_samples,
        "n_processes": int(trials[0].shape[1]),
        "n_trials": len(trials),
        "flagged": {},
    }
    if discrete_input:
        data = DiscreteData.from_discrete(trials, names)
        # The rank-based estimators read the symbols as raw values.
        data.raw = [np.asarray(t, dtype=float) for t in trials]
        return data, Embedding(max_lag=max_lag or 3), None, diagnostics
    from .preprocess.pipeline import preprocess

    result = preprocess(trials, names=names, max_lag=max_lag, network_resolution=network_resolution, prng=prng)
    if result.report.network:
        net = result.report.network
        diagnostics["network_resolution"] = {k: net[k] for k in ("bins", "history", "adopted")} | {
            "min_retained": float(min(net["retained"]))
        }
    for node in result.report.nodes:
        flags = list(node.flags)
        if node.screen is not None and getattr(node.screen, "nonstationary", False) and "nonstationary" not in flags:
            flags.append("nonstationary")
        if flags:
            diagnostics["flagged"][node.name] = flags
    return result.data, result.embeddings, result.report, diagnostics


def parent_prediction_error(data, target, variables, folds=5, k_fraction=0.02):
    """
    Blocked cross-validated mean squared error of predicting `target`'s raw present from
    `variables` (raw values, standardized) by nearest-neighbour regression.

    A nonparametric referee for comparing parent sets chosen by different estimators:
    it assumes no functional form, so neither estimator's assumptions decide.
    """
    from scipy.spatial import KDTree

    from .data import realizations
    from .selection import _raw_columns

    variables = list(variables)
    r = realizations(data, target, variables, max_lag=max([lag for _, lag in variables] + [1]))
    y, X = _raw_columns(data, r, target)
    if X.shape[1]:
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    n = len(y)
    fold = np.minimum(folds - 1, (folds * np.arange(n) / max(n, 1)).astype(np.int64))
    error = 0.0
    for f in range(folds):
        train, test = fold != f, fold == f
        if not test.any() or train.sum() < 10:
            continue
        if X.shape[1] == 0:
            error += float(np.sum((y[test] - y[train].mean()) ** 2))
            continue
        k = max(5, int(k_fraction * train.sum()))
        neighbors = KDTree(X[train]).query(X[test], k)[1].reshape(int(test.sum()), -1)
        error += float(np.sum((y[test] - y[train][neighbors].mean(axis=1)) ** 2))
    return error / max(n, 1)


def _gaussian_ratio(data, target, embeddings, plugin_settings, gaussian_settings, seed):
    """
    The nearest-neighbour prediction error of the Gaussian estimator's parent set over the
    plug-in's, each selected alone, and the seconds the Gaussian selection took.
    """

    def chosen(settings):
        sk = select_parents(data, target, embeddings, None, settings, seed)
        return list(sk.target_past) + (list(sk.sources) if sk.significant else []), sk.timing

    plugin_vars, _ = chosen(plugin_settings)
    gaussian_vars, timing = chosen(gaussian_settings)
    plugin = parent_prediction_error(data, target, plugin_vars)
    gaussian = parent_prediction_error(data, target, gaussian_vars)
    return (gaussian / plugin if plugin > 0 else 1.0), float(timing.get("gaussian", 0.0))


def linearity_pvalues(data, targets=None, max_lag=5, max_regressors=None, bins=4):
    """
    Per-target p-values of a test that the target's present is linear in its past.

    The present is regressed (least squares, with an intercept) on its own lags and on
    the source lags most correlated with it. If the target is linear in its past, the
    residual is independent of every regressor; dependence that survives the linear
    fit is nonlinear. Each regressor is tested against the residual with the G-test
    of the plug-in mutual information in ``bins`` x ``bins`` equal-frequency cells
    (:math:`2 n \\ln 2 \\, \\hat I \\sim \\chi^2_{(b-1)^2}` under independence
    :cite:`Cover2006`), and the smallest p-value is Bonferroni-corrected over the
    regressors. Unlike a test of squared terms, this also sees nonlinearity in a
    combination of the past (e.g. a logistic map of a weighted sum of inputs). A small
    p-value means linear conditioning (the ``'gaussian'`` estimator) would leave
    residual dependence for this target.

    Parameters
    ----------
    data : DiscreteData
        With raw series.
    targets : list of int, None
    max_lag : int
    max_regressors : int, None
        Linear regressors per target (default: a twentieth of the samples, at least 10).
    bins : int

    Returns
    -------
    dict
        ``target -> p-value``.
    """
    from scipy.stats import chi2

    from .selection import _cmi, _equal_frequency, _ranks

    if data.raw is None:
        raise ValueError("the linearity test needs the raw series")
    P = data.n_processes
    targets = list(range(P)) if targets is None else list(targets)
    rows = [np.asarray(r, dtype=float) for r in data.raw if len(r) > max_lag + 1]
    lagged = np.vstack([np.stack([r[max_lag - lag : len(r) - lag] for lag in range(max_lag + 1)], 1) for r in rows])
    n = lagged.shape[0]
    present, past = lagged[:, 0, :], lagged[:, 1:, :]
    past = (past - past.mean(axis=0)) / np.where(past.std(axis=0) > 0, past.std(axis=0), 1.0)
    budget = max_regressors or max(10, n // 20)
    zero = np.zeros(n, dtype=np.int64)
    dof = (bins - 1) ** 2
    out = {}
    for t in targets:
        y = present[:, t]
        own = past[:, :, t]
        others = np.concatenate([past[:, :, p] for p in range(P) if p != t], axis=1) if P > 1 else own[:, :0]
        keep = max(0, budget - own.shape[1])
        if others.shape[1] > keep:
            yc = y - y.mean()
            corr = np.abs(others.T @ yc) / (np.linalg.norm(others, axis=0) * np.linalg.norm(yc) + 1e-300)
            others = others[:, np.argsort(-corr)[:keep]]
        X = np.column_stack([own, others])
        design = np.column_stack([np.ones(n), X])
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        r = _equal_frequency(_ranks(residual), bins)
        smallest = 1.0
        for k in range(X.shape[1]):
            v = _equal_frequency(_ranks(X[:, k]), bins)
            g = 2 * n * np.log(2) * _cmi(r, bins, v, bins, zero, 1)
            smallest = min(smallest, float(chi2.sf(g, dof)))
        out[t] = min(1.0, smallest * X.shape[1])
    return out


def _skeleton_settings(estimators, prescreen, threads, device, pinned, target_estimators=None):
    """
    Selection settings for an ensemble: every stage at :data:`ALPHA` split ``k`` ways,
    with enough permutations to reach it and curtailment (which never changes a decision).
    """
    k = len(estimators)
    n_perm = int(np.ceil(k / ALPHA))
    stages = ("max_stat", "min_stat", "omnibus", "max_seq", "pairs")
    kwargs: dict[str, Any] = {
        "estimator": tuple(estimators),
        "target_estimators": target_estimators or None,
        "curtail": True,
        "prescreen_alpha": 0.1 if prescreen else None,
        "n_perm_tdmi": max(200, int(np.ceil(k / 0.05))),
        "device": device,
        **{f"alpha_{s}": ALPHA for s in stages},
        **{f"n_perm_{s}": n_perm for s in stages},
    }
    if "ksg" in estimators:
        kwargs["ksg_threads"] = max(1, (os.cpu_count() or 1) // max(1, threads))
    kwargs.update(pinned)
    return SkeletonSettings(**kwargs)


def _rounds(items, threads):
    """
    Rounds of parallel work: `items` equal jobs on `threads` workers.
    """
    return float(np.ceil(items / threads)) if items > 0 else 0.0


def _predict(options, pilot, n_targets, threads):
    """
    Predicted seconds of a run with `options`, from the pilot's measurements.

    The pilot's per-target times were measured with its targets running together,
    each using the cores the others left free; once more targets run than the pilot
    had, they share the same cores, so selection takes the pilot's time for every
    ``concurrency`` targets. Edges (measured one at a time, and mostly single-threaded)
    run `threads` at a time, in as many rounds as they have full or partial batches.
    """
    usage = pilot.get("usage", {})
    per_target = sum(pilot["member_seconds"].get(name, 0.0) * usage.get(name, 1.0) for name in options["estimators"])
    if options["prescreen"] and not pilot["prescreen"]:
        per_target *= _PRESCREEN_FACTOR
    concurrency = max(1, pilot.get("concurrency", threads))
    selection = per_target * max(_rounds(n_targets, threads), n_targets / concurrency)
    edges = pilot["parents_per_target"] * n_targets
    if options["include_shared_candidates"]:
        edges += pilot["shared_per_target"] * n_targets
    layers = pilot["edge_seconds"] * _rounds(edges, threads)
    total = selection + layers
    if options["interpret"]:
        total += pilot["interpret_seconds"] * n_targets
    if options["hyperedges"]:
        total += _HYPEREDGE_FACTOR * layers
    return total


# Options dropped, in order, when a run is predicted to exceed its budget.
_DOWNGRADES = (
    ("ksg", "KSG selection"),
    ("include_shared_candidates", "layer estimates for shared-only candidates"),
    ("hyperedges", "hyperedge decomposition"),
    ("prescreen", "no source pre-screen (pre-screen turned on)"),
)


def _fit_budget(options, pilot, n_targets, threads, budget, pinned):
    """
    Apply :data:`_DOWNGRADES` in order until the predicted time fits `budget`.

    Returns the new options, the downgrades made (with reasons), and the prediction.
    Pinned options are never changed; the significance level never is.
    """
    options: dict[str, Any] = dict(options, estimators=tuple(options["estimators"]))
    predicted = _predict(options, pilot, n_targets, threads)
    downgrades = []
    if budget is None:
        return options, downgrades, predicted
    for key, label in _DOWNGRADES:
        if predicted <= budget:
            break
        if key == "ksg":
            if "estimator" in pinned or key not in options["estimators"] or len(options["estimators"]) == 1:
                continue
            options["estimators"] = tuple(e for e in options["estimators"] if e != key)
        elif key == "prescreen":
            if "prescreen_alpha" in pinned or options["prescreen"]:
                continue
            options["prescreen"] = True
        else:
            if key in pinned or not options[key]:
                continue
            options[key] = False
        before, predicted = predicted, _predict(options, pilot, n_targets, threads)
        downgrades.append(
            f"dropped {label}: predicted {timedelta(seconds=round(before))} > remaining budget "
            f"{timedelta(seconds=round(budget))}, now {timedelta(seconds=round(predicted))}"
        )
    return options, downgrades, predicted


def _ksg_seconds(data, embeddings, target, settings, cheap):
    """
    Predicted seconds of KSG selection for one target: one timed batch of KSG null
    estimates, scaled by the tests the cheap ensemble ran (each admitted variable needs
    every permutation over the whole candidate family; a rejected test stops after
    about two batches).
    """
    from .selection import _KsgColumns, _source_lags

    sources = [p for p in range(data.n_processes) if p != target]
    family = [(p, lag) for p in sources for lag in _source_lags(data, embeddings, p, settings.max_source_lag or 5)]
    variables = [(target, 1), *family[:4]]
    cols = _KsgColumns(data, target, variables, max(v[1] for v in variables), k=settings.ksg_k, threads=None)
    z, Kz = cols.context([(target, 1)])
    x, Kx = cols.column(variables[1])
    batch = settings.curtail_batch
    idx = [np.random.default_rng(i).permutation(cols.n) for i in range(batch)]
    start = time.perf_counter()
    cols.cmi_batch(x, Kx, z, Kz, idx)
    per_estimate = (time.perf_counter() - start) / batch
    admitted = len(cheap.sources) + len(cheap.target_past)
    evaluations = len(family) * (admitted * settings.n_perm_max_stat + 2 * batch * (1 + len(sources)))
    return per_estimate * evaluations


def _edge_seconds(data, skeletons, n_resamples):
    """
    Measured seconds per edge of layer estimation at `n_resamples`, from one pilot parent.
    """
    for sk in skeletons.values():
        for p in sk.parents():
            base = list(sk.conditionals) + list(sk.target_past)
            w = base + [v for v in sk.sources if v[0] != p]
            trial = 50
            start = time.perf_counter()
            estimate_edge(data, sk.target, sk.source_variables(p), w, "miller_madow", trial, trial, prng=0)
            return (time.perf_counter() - start) * n_resamples / trial
    return 0.0


def infer(
    data,
    preset="balanced",
    time_budget=None,
    names=None,
    targets=None,
    max_lag=None,
    threads=None,
    strict_budget=False,
    network_resolution=None,
    prng=None,
    **overrides,
):
    """
    Infer a multiplex information-flow network with automatically chosen methods.

    Parameters
    ----------
    data : array_like or Trials
        Raw series (samples, processes) or independent trials of them. Integer arrays
        are treated as symbols, float arrays are preprocessed
        (:func:`~infoflow.preprocess.preprocess`).
    preset : {'fast', 'balanced', 'thorough'}
        How much to compute (:data:`PRESETS`): the selection ensemble and the optional
        stages (interpretation, shared-only candidates, hyperedges).
    time_budget : float, str, timedelta, None
        Wall-clock budget, e.g. ``'2h'``. The run is sized to it from a pilot; options
        are dropped in a fixed order until the prediction fits, and the significance
        level is never loosened.
    names : list of str, None
    targets : list of int, None
    max_lag : int, None
        Upper bound on every node's lag budget (default: chosen by preprocessing; 3
        for discrete data).
    threads : int, None
        Targets (and edges) processed in parallel (default: all cores).
    strict_budget : bool
        Raise instead of warning when even the smallest configuration is predicted to
        exceed the budget.
    network_resolution : {None, False, 'auto', True}
        Preprocessing's uniform resolution for parent selection
        (:func:`~infoflow.preprocess.preprocess`); None takes the preset's (``'auto'``
        in every preset).
    prng : None, int, Generator
    **overrides
        Any :class:`~infoflow.selection.SkeletonSettings` field or
        :func:`~infoflow.infer_multiplex` parameter; pinned values are used as given
        and never changed by the budget. ``estimator`` pins the selection estimator(s);
        ``layer_estimator`` sets the layers' estimator (``infer_multiplex``'s
        ``estimator``).

    Returns
    -------
    AutoResult
        ``network`` (:class:`~infoflow.MultiplexNetwork`) and ``report``
        (:class:`AutoReport`).
    """
    started = time.perf_counter()
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    budget = _parse_budget(time_budget)
    pinned_skeleton, pinned_infer = _split_overrides(overrides)
    rng = as_generator(prng)
    pre_seed, run_seed = (int(s) for s in rng.integers(0, 2**32, size=2))
    threads = threads or os.cpu_count() or 1
    trials = as_trials(data)
    preset_options = PRESETS[preset]
    if network_resolution is None:
        network_resolution = preset_options["network_resolution"]
    discrete, embeddings, _, diagnostics = _diagnose(trials, names, pre_seed, max_lag, network_resolution)
    P = discrete.n_processes
    targets = list(range(P)) if targets is None else list(targets)
    chosen = pinned_skeleton.pop("estimator", preset_options[diagnostics["kind"]])
    estimators = (chosen,) if isinstance(chosen, str) else tuple(chosen)
    if "estimator" in overrides:
        pinned_skeleton["estimator"] = estimators
    nonlinear = []
    gate = diagnostics["kind"] == "continuous" and "gaussian" in estimators
    if gate and "estimator" not in overrides and "target_estimators" not in pinned_skeleton:
        pvalues = linearity_pvalues(discrete, targets, max_lag or 5)
        nonlinear = [t for t in targets if pvalues[t] < LINEARITY_ALPHA]
        diagnostics["linearity_pvalues"] = pvalues
        diagnostics["nonlinear_targets"] = nonlinear
    options: dict[str, Any] = {
        "estimators": estimators,
        "prescreen": P > _PRESCREEN_ABOVE,
        **{k: pinned_infer.get(k, preset_options[k]) for k in ("include_shared_candidates", "interpret", "hyperedges")},
    }
    if "prescreen_alpha" in pinned_skeleton:
        options["prescreen"] = pinned_skeleton["prescreen_alpha"] is not None
    try:
        import torch  # noqa: F401

        device = "auto"
    except ImportError:
        device = None
    device = pinned_infer.pop("device", device)
    report = AutoReport(diagnostics=diagnostics, preset=preset, time_budget=budget)

    # For targets that test nonlinear, select once with the plug-in alone and once with the
    # Gaussian estimator alone (local null), and keep the Gaussian estimator only where its
    # parents predict the target at least as well by a nonparametric referee.
    keep_gaussian, drop_gaussian = [], list(nonlinear)
    gate_downgrades, gaussian_seconds = [], []
    if nonlinear:
        pinned_base = {k: v for k, v in pinned_skeleton.items() if k != "estimator"}
        plugin_alone = _skeleton_settings(("plugin",), options["prescreen"], 1, device, pinned_base)
        gaussian_alone = _skeleton_settings(
            ("gaussian",), options["prescreen"], 1, device, {"gaussian_null": "local", **pinned_base}
        )
        choice_seeds = as_generator(run_seed + 2).integers(0, 2**32, size=len(nonlinear))

        def ratio(args):
            t, seed = args
            if budget is not None and time.perf_counter() - started > _GATE_BUDGET_FRACTION * budget:
                return t, None
            return t, _gaussian_ratio(discrete, t, embeddings, plugin_alone, gaussian_alone, int(seed))

        ratios, skipped = {}, []
        for t, out in thread_map(threads)(ratio, list(zip(nonlinear, choice_seeds, strict=True))):
            if out is None:
                skipped.append(t)
            else:
                ratios[t] = out[0]
                gaussian_seconds.append(out[1])
        if skipped:
            gate_downgrades.append(
                f"skipped the Gaussian check for {len(skipped)} nonlinear target(s) not started within "
                f"{_GATE_BUDGET_FRACTION:.0%} of the budget; they select with the plug-in only"
            )
        keep_gaussian = [t for t in nonlinear if ratios.get(t, 1.0) < 1.0]
        drop_gaussian = [t for t in nonlinear if t not in keep_gaussian]
        diagnostics["gaussian_ratio"] = ratios
        diagnostics["gaussian_kept"] = keep_gaussian

    def settings_for(opts, ksg_threads_divisor):
        pinned = {k: v for k, v in pinned_skeleton.items() if k != "estimator"}
        if keep_gaussian and "gaussian" in opts["estimators"]:
            pinned.setdefault("gaussian_null", "local")
        reduced = tuple(e for e in opts["estimators"] if e != "gaussian")
        per_target = dict.fromkeys(drop_gaussian, reduced) if "gaussian" in opts["estimators"] and reduced else {}
        return _skeleton_settings(
            opts["estimators"], opts["prescreen"], ksg_threads_divisor, device, pinned, per_target
        )

    def map_threads(opts):
        return max(1, threads // 4) if "ksg" in opts["estimators"] else threads

    # Pilot: the cheap members of the ensemble on a few sampled targets, with the seeds
    # infer_skeleton will use, so their skeletons are reused when the settings match.
    seeds = as_generator(run_seed).integers(0, 2**32, size=len(targets))
    seed_of = dict(zip(targets, (int(s) for s in seeds), strict=True))
    order = as_generator(run_seed + 1).permutation(len(targets))
    pilot_targets = [targets[i] for i in order[: min(_PILOT_TARGETS, len(targets))]]
    cheap: dict[str, Any] = dict(options, estimators=tuple(e for e in estimators if e != "ksg") or estimators)
    pilot_settings = settings_for(cheap, map_threads(cheap))

    def pilot_run(t):
        return t, select_parents(discrete, t, embeddings, None, pilot_settings, seed_of[t])

    pilot_skeletons = dict(thread_map(min(threads, len(pilot_targets)))(pilot_run, pilot_targets))
    member_seconds = {
        name: float(np.mean([sk.timing[name] for sk in pilot_skeletons.values() if name in sk.timing] or [0.0]))
        for name in cheap["estimators"]
    }
    if "gaussian" in member_seconds and member_seconds["gaussian"] == 0.0 and gaussian_seconds:
        # No pilot target used the Gaussian estimator; the gate timed it (local null) on others.
        member_seconds["gaussian"] = float(np.mean(gaussian_seconds))
    if "ksg" in estimators and "ksg" not in cheap["estimators"]:
        t0 = pilot_targets[0]
        member_seconds["ksg"] = _ksg_seconds(
            discrete, embeddings, t0, settings_for(options, map_threads(options)), pilot_skeletons[t0]
        )
    parents = float(np.mean([len(sk.parents()) for sk in pilot_skeletons.values()]))
    shared = float(np.mean([len(sk.tdmi_candidates) for sk in pilot_skeletons.values()])) if pilot_skeletons else 0.0
    expected_edges = max(1.0, parents * len(targets))
    n_resamples = min(_MAX_RESAMPLES, int(np.ceil(expected_edges / _LAYER_ALPHA)))
    from .interpret import edge_roles

    start = time.perf_counter()
    if options["interpret"]:
        edge_roles(discrete, pilot_skeletons, embeddings, pilot_settings, prng=0)
    interpret_seconds = (time.perf_counter() - start) / len(pilot_targets)
    pilot = {
        "concurrency": min(threads, len(pilot_targets)),
        "usage": {"gaussian": 1.0 - len(drop_gaussian) / len(targets)} if targets else {},
        "targets": pilot_targets,
        "interpret_seconds": interpret_seconds,
        "member_seconds": member_seconds,
        "prescreen": cheap["prescreen"],
        "parents_per_target": parents,
        "shared_per_target": shared,
        "edge_seconds": _edge_seconds(discrete, pilot_skeletons, n_resamples),
    }
    # Preprocessing, the linearity test, and the pilot are already spent.
    elapsed = time.perf_counter() - started
    remaining = None if budget is None else max(0.0, budget - elapsed)
    final, downgrades, predicted = _fit_budget(
        options, pilot, len(targets), map_threads(options), remaining, pinned_skeleton | pinned_infer
    )
    report.downgrades = gate_downgrades + downgrades
    if budget is not None and pinned_infer.get("machines"):
        report.warnings.append("the predicted time excludes the machine layer (machines=True runs outside the budget)")
    report.predicted_seconds = elapsed + predicted
    pilot["elapsed_before_run"] = elapsed
    report.pilot = pilot
    if budget is not None and report.predicted_seconds > budget:
        message = (
            f"predicted {timedelta(seconds=round(report.predicted_seconds))} exceeds the budget "
            f"{timedelta(seconds=round(budget))} with every option dropped"
        )
        if strict_budget:
            raise RuntimeError(message)
        report.warnings.append(message)
        warnings.warn(message, stacklevel=2)
    final_settings = settings_for(final, map_threads(final))
    report.estimators = final["estimators"]
    report.options = {k: final[k] for k in ("prescreen", "include_shared_candidates", "interpret", "hyperedges")} | {
        "device": device,
        "threads": map_threads(final),
    }

    with tempfile.TemporaryDirectory(prefix="infoflow-auto-") as directory:
        checkpoint = Checkpoint(directory)
        if final_settings == pilot_settings and pinned_infer.get("holdout") is None:
            for t, sk in pilot_skeletons.items():
                checkpoint.put(f"skeleton-{t}", sk)
        if budget is not None and pinned_infer.get("holdout") is None:
            # Select here, with the seeds infer_skeleton would use, so targets that start late
            # can fall back to the plug-in; infer_multiplex then reads them from the checkpoint.
            fallback = final_settings
            if "estimator" not in overrides:
                fallback = settings_for(dict(final, estimators=("plugin",)), map_threads(final))
            deadline = started + _SELECTION_BUDGET_FRACTION * budget
            late = []

            def select(t):
                if f"skeleton-{t}" in checkpoint:
                    return t, checkpoint.get(f"skeleton-{t}")
                settings = final_settings
                if time.perf_counter() > deadline:
                    settings = fallback
                    late.append(t)
                return t, select_parents(discrete, t, embeddings, None, settings, seed_of[t])

            for t, sk in thread_map(map_threads(final))(select, targets):
                checkpoint.put(f"skeleton-{t}", sk)
            if late and fallback != final_settings:
                report.downgrades.append(
                    f"selected {len(late)} target(s) with the plug-in only: started after "
                    f"{_SELECTION_BUDGET_FRACTION:.0%} of the budget ({', '.join(map(str, sorted(late)))})"
                )
        kwargs = {
            "embeddings": embeddings,
            "names": names,
            "targets": targets,
            "skeleton_settings": final_settings,
            "include_shared_candidates": final["include_shared_candidates"],
            "interpret": final["interpret"],
            "hyperedges": final["hyperedges"],
            "alpha": _LAYER_ALPHA,
            "max_resamples": _MAX_RESAMPLES,
            "map_fn": thread_map(map_threads(final)),
            "checkpoint": checkpoint,
            "prng": run_seed,
        }
        kwargs.update({k: v for k, v in pinned_infer.items() if k not in ("include_shared_candidates", "interpret")})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            network = infer_multiplex(discrete, **kwargs)
    for w in caught:
        if str(w.message) not in report.warnings:
            report.warnings.append(str(w.message))
            warnings.warn(w.message, w.category, stacklevel=2)
    report.admitted = {
        (v[0], t): name for t, sk in network.skeleton.items() for v, name in sk.admitted_by.items() if sk.significant
    }
    if network.machines is not None:
        report.machines = network.machines.report()
    report.actual_seconds = time.perf_counter() - started
    return AutoResult(network, report)
