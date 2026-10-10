"""
Hands-off network inference: :func:`infer` chooses the methods from the data and a budget.

The choices rest on the validation in :doc:`validation`. No single selection
estimator wins everywhere: the linear-Gaussian and stratified-trend statistics are the
most powerful on monotone couplings, coarse symbols on nonlinear ones, and the plug-in
CMI keeps full resolution on discrete data. :func:`infer` therefore runs an ensemble of
them in one greedy search at a Bonferroni-split level
(:class:`~infoflow.selection.SkeletonSettings`), at the strict significance level
0.001 at which every estimator kept its precision near 1. It times the selection on a
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

__all__ = ("PRESETS", "AutoReport", "AutoResult", "infer")

#: Selection significance level: precision stayed near 1 for every estimator at this level.
ALPHA = 0.001

#: The options of each preset. ``continuous`` / ``discrete`` are the estimator ensembles.
PRESETS = {
    "fast": {
        "continuous": ("gaussian", "coarse"),
        "discrete": ("coarse",),
        "include_shared_candidates": False,
        "interpret": False,
        "hyperedges": False,
    },
    "balanced": {
        "continuous": ("gaussian", "trend", "coarse"),
        "discrete": ("coarse", "plugin"),
        "include_shared_candidates": False,
        "interpret": True,
        "hyperedges": False,
    },
    "thorough": {
        "continuous": ("gaussian", "trend", "coarse", "ksg"),
        "discrete": ("coarse", "plugin"),
        "include_shared_candidates": True,
        "interpret": True,
        "hyperedges": True,
    },
}

# Heuristic cost factors for options the pilot does not time directly.
_PRESCREEN_FACTOR = 0.7
_HYPEREDGE_FACTOR = 0.25
# Pre-screen networks with more processes than this.
_PRESCREEN_ABOVE = 30
_PILOT_TARGETS = 3
_MAX_RESAMPLES = 2000
_LAYER_ALPHA = 0.05


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
        if d.get("flagged"):
            lines.append("flagged nodes: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in d["flagged"].items()))
        lines += [f"downgrade: {g}" for g in self.downgrades]
        if self.admitted:
            counts = {}
            for name in self.admitted.values():
                counts[name] = counts.get(name, 0) + 1
            lines.append("parents admitted by: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
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
    skeleton_fields = {f.name for f in dataclasses.fields(SkeletonSettings)}
    infer_params = set(inspect.signature(infer_multiplex).parameters) - {"data", "skeleton_settings"}
    unknown = set(overrides) - skeleton_fields - infer_params
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}: not SkeletonSettings or infer_multiplex parameters")
    return (
        {k: v for k, v in overrides.items() if k in skeleton_fields},
        {k: v for k, v in overrides.items() if k in infer_params},
    )


def _diagnose(trials, names, prng, max_lag=None):
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

    result = preprocess(trials, names=names, max_lag=max_lag, prng=prng)
    for node in result.report.nodes:
        flags = list(node.flags)
        if node.screen is not None and getattr(node.screen, "nonstationary", False) and "nonstationary" not in flags:
            flags.append("nonstationary")
        if flags:
            diagnostics["flagged"][node.name] = flags
    return result.data, result.embeddings, result.report, diagnostics


def _skeleton_settings(estimators, prescreen, threads, device, pinned):
    """
    Selection settings for an ensemble: every stage at :data:`ALPHA` split ``k`` ways,
    with enough permutations to reach it and curtailment (which never changes a decision).
    """
    k = len(estimators)
    n_perm = int(np.ceil(k / ALPHA))
    stages = ("max_stat", "min_stat", "omnibus", "max_seq", "pairs")
    kwargs: dict[str, Any] = {
        "estimator": tuple(estimators),
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

    Targets, and then edges, run `threads` at a time, so each phase takes as many
    rounds of its per-item time as it has full or partial batches of items.
    """
    per_target = sum(pilot["member_seconds"].get(name, 0.0) for name in options["estimators"])
    if options["prescreen"] and not pilot["prescreen"]:
        per_target *= _PRESCREEN_FACTOR
    selection = per_target * _rounds(n_targets, threads)
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
    ("trend", "trend selection"),
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
        if key in ("ksg", "trend"):
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
            f"dropped {label}: predicted {timedelta(seconds=round(before))} > budget "
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
    prng : None, int, Generator
    **overrides
        Any :class:`~infoflow.selection.SkeletonSettings` field or
        :func:`~infoflow.infer_multiplex` parameter; pinned values are used as given
        and never changed by the budget.

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
    discrete, embeddings, _, diagnostics = _diagnose(trials, names, pre_seed, max_lag)
    P = discrete.n_processes
    targets = list(range(P)) if targets is None else list(targets)
    preset_options = PRESETS[preset]
    chosen = pinned_skeleton.pop("estimator", preset_options[diagnostics["kind"]])
    estimators = (chosen,) if isinstance(chosen, str) else tuple(chosen)
    if "estimator" in overrides:
        pinned_skeleton["estimator"] = estimators
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

    def settings_for(opts, ksg_threads_divisor):
        pinned = {k: v for k, v in pinned_skeleton.items() if k != "estimator"}
        return _skeleton_settings(opts["estimators"], opts["prescreen"], ksg_threads_divisor, device, pinned)

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
        name: float(np.mean([sk.timing.get(name, 0.0) for sk in pilot_skeletons.values()]))
        for name in cheap["estimators"]
    }
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
        "targets": pilot_targets,
        "interpret_seconds": interpret_seconds,
        "member_seconds": member_seconds,
        "prescreen": cheap["prescreen"],
        "parents_per_target": parents,
        "shared_per_target": shared,
        "edge_seconds": _edge_seconds(discrete, pilot_skeletons, n_resamples),
    }
    final, report.downgrades, report.predicted_seconds = _fit_budget(
        options, pilot, len(targets), map_threads(options), budget, pinned_skeleton | pinned_infer
    )
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
    report.actual_seconds = time.perf_counter() - started
    return AutoResult(network, report)
