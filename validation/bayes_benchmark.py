"""
Benchmark the hierarchical Bayesian estimators of :mod:`infoflow.bayes`.

Parts:

* ``motifs``: layer RMSE on XOR (pure synergy) and common-driver (pure shared) edges
  whose context is padded with `d` independent nuisance bits, for cross-fitting, the
  single pooled law, and a four-component mixture prior.
* ``robust``: cross-fitting against the mixture prior on "graded" edges, where each
  of the ``2**d`` contexts has its own coupling strength (more distinct laws than
  mixture components; true layers from the exact joint), and on a sparser XOR edge,
  with seconds per edge.
* ``scaling``: :func:`infoflow.benchmarks.conditioning_scaling` RMSE with the Bayesian
  estimators next to Miller--Madow and the adaptive partition.
* ``zero``: rejection rates of ``TE = 0`` for the BDeu Bayes factor and the
  within-strata permutation test, under two nulls (independent; confounded, where
  TDMI > 0 but TE = 0) and two alternatives (copy; XOR, invisible to TDMI).
* ``timeseries``: the sofic test processes, serially dependent and with exact flows,
  with the target's past padded by persistent nuisance chains; layer RMSE and the
  coverage of 90% bootstrap (``layer_statistics``) and credible intervals.
* ``crossover``: cross-fitting against the mixture on graded edges over a grid of
  context counts and sample sizes.
* ``components``: the mixture with G = 1, 2, 4, 8, 16 components and with a tighter
  or looser prior on :math:`\\log \\alpha`.
* ``speed``: accuracy and seconds per edge for shorter chains, fewer posterior
  tables, nutpie, and ADVI.
* ``network``: :func:`~infoflow.network.infer_multiplex` end to end on motif and
  noisy-ECA networks, with the fast mixture (two components, one short chain)
  swapped in for the layer weights and intervals (significance unchanged). Each
  estimated edge is scored against the same variables on a 200000-sample run; also
  roles, flags, run time, and :func:`~infoflow.benchmarks.motif_validation` rates.

Writes ``validation/results/bayes/<part>.json``.

Example::

    python validation/bayes_benchmark.py                 # all parts
    python validation/bayes_benchmark.py --parts zero speed
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np

from infoflow.bayes import layer_posterior, log_bayes_factor
from infoflow.benchmarks import conditioning_scaling
from infoflow.crossfit import crossfit_flows
from infoflow.stats import conditional_mutual_information_test

LAYERS = ("te", "tdmi", "intrinsic", "synergistic", "shared")
OUT = Path(__file__).parent / "results" / "bayes"


def _h(p):
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def _flip(n, p, rng):
    return (rng.random(n) < p).astype(np.int64)


def motif(name, n, d, rng):
    nuisance = rng.integers(0, 2, (n, d))
    w0 = rng.integers(0, 2, n)
    if name == "xor":
        x = rng.integers(0, 2, n)
        y = x ^ w0 ^ _flip(n, 0.05, rng)
        t = 1 - _h(0.05)
        truth = {"te": t, "tdmi": 0.0, "intrinsic": 0.0, "synergistic": t, "shared": 0.0}
    else:
        x, y = w0 ^ _flip(n, 0.1, rng), w0 ^ _flip(n, 0.1, rng)
        t = 1 - _h(0.18)
        truth = {"te": 0.0, "tdmi": t, "intrinsic": 0.0, "synergistic": 0.0, "shared": t}
    return y, x, np.column_stack([w0, nuisance]), truth


def run_motifs(n=1000, dims=(0, 2, 5), reps=10, seed=0):
    rng = np.random.default_rng(seed)
    methods = {
        "crossfit": lambda y, x, w, s: crossfit_flows(y, x, w, prng=s).as_dict(),
        "pooled": lambda y, x, w, s: layer_posterior(y, x, w, 1, 100, 300, 300, prng=s).mean,
        "mixture": lambda y, x, w, s: layer_posterior(y, x, w, 4, 100, 300, 300, prng=s).mean,
    }
    out = {}
    for name in ("xor", "driver"):
        for d in dims:
            err = {m: [] for m in methods}
            for _ in range(reps):
                y, x, w, truth = motif(name, n, d, rng)
                s = int(rng.integers(2**31))
                for m, f in methods.items():
                    est = f(y, x, w, s)
                    err[m].append([est[k] - truth[k] for k in LAYERS])
            for m, e in err.items():
                e = np.asarray(e)
                out[f"{name}/d={d}/{m}"] = {
                    "rmse": dict(zip(LAYERS, np.sqrt((e**2).mean(0)).tolist(), strict=True)),
                    "bias": dict(zip(LAYERS, e.mean(0).tolist(), strict=True)),
                }
            print(f"motifs {name} d={d} done", flush=True)
    return out


def graded(n, d, rng, flips):
    """
    ``y = x XOR Bernoulli(flips[w])`` with every one of the ``2**d`` contexts relevant.
    """
    w = rng.integers(0, 2**d, n)
    x = rng.integers(0, 2, n)
    y = x ^ (rng.random(n) < flips[w]).astype(np.int64)
    return y, x, w


def graded_truth(d, flips):
    from infoflow.measures import cmi_from_joint, intrinsic_flow

    p = np.zeros((2, 2, 2**d))
    for x in (0, 1):
        p[x, x, :] = 0.5 * (1 - flips) / 2**d
        p[1 - x, x, :] = 0.5 * flips / 2**d
    te, tdmi = cmi_from_joint(p), cmi_from_joint(p.sum(axis=2, keepdims=True))
    iif = intrinsic_flow(p, restarts=20, prng=0)[0]
    return {"te": te, "tdmi": tdmi, "intrinsic": iif, "synergistic": te - iif, "shared": tdmi - iif}


def run_robust(reps=8, seed=1):
    import time

    rng = np.random.default_rng(seed)
    methods = {
        "crossfit": lambda y, x, w, s: crossfit_flows(y, x, w, prng=s).as_dict(),
        "mixture": lambda y, x, w, s: layer_posterior(y, x, w, 4, 100, 300, 300, prng=s).mean,
    }
    cases = [("graded", 1000, d) for d in (3, 5, 7)] + [("xor", 500, 6)]
    out = {}
    for name, n, d in cases:
        flips = rng.uniform(0.05, 0.45, 2**d) if name == "graded" else None
        err, secs = {m: [] for m in methods}, dict.fromkeys(methods, 0.0)
        for _ in range(reps):
            if name == "graded":
                y, x, w = graded(n, d, rng, flips)
                truth = graded_truth(d, flips)
            else:
                y, x, w, truth = motif("xor", n, d, rng)
            s = int(rng.integers(2**31))
            for m, f in methods.items():
                t0 = time.perf_counter()
                est = f(y, x, w, s)
                secs[m] += time.perf_counter() - t0
                err[m].append([est[k] - truth[k] for k in LAYERS])
        for m, e in err.items():
            e = np.asarray(e)
            out[f"{name}/n={n}/d={d}/{m}"] = {
                "rmse": dict(zip(LAYERS, np.sqrt((e**2).mean(0)).tolist(), strict=True)),
                "bias": dict(zip(LAYERS, e.mean(0).tolist(), strict=True)),
                "seconds_per_edge": secs[m] / reps,
            }
        print(f"robust {name} d={d} done", flush=True)
    return out


def run_scaling(seed=0):
    out = {}
    for coupled in (True, False):
        ds = conditioning_scaling(
            "discrete",
            ("miller_madow", "adaptive", "hierarchical", "mixture"),
            n_samples=(500, 2500),
            dims=(4, 8),
            n_reps=5,
            coupled=coupled,
            prng=seed,
        )
        rmse = np.sqrt(ds["bias"] ** 2 + ds["std"] ** 2)
        for e in ds.estimator.values:
            for n in ds.n.values:
                for d in ds.d.values:
                    out[f"{'coupled' if coupled else 'uncoupled'}/n={n}/d={d}/{e}"] = float(
                        rmse.sel(estimator=e, n=n, d=d)
                    )
        print(f"scaling coupled={coupled} done", flush=True)
    return out


def zero_case(kind, n, d, rng):
    nuisance = rng.integers(0, 2, (n, d))
    w0 = rng.integers(0, 2, n)
    x = rng.integers(0, 2, n)
    if kind == "independent":
        y = rng.integers(0, 2, n)
    elif kind == "confounded":
        x = w0 ^ _flip(n, 0.1, rng)
        y = w0 ^ _flip(n, 0.1, rng)
    elif kind == "copy":
        y = x ^ _flip(n, 0.35, rng)
    else:  # xor
        y = x ^ w0 ^ _flip(n, 0.3, rng)
    return y, x, np.column_stack([w0, nuisance])


def run_zero(n_samples=(250, 1000), dims=(0, 2, 4, 6), reps=100, n_surrogates=199, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("independent", "confounded", "copy", "xor"):
        for n in n_samples:
            for d in dims:
                hits = {"perm_p05": 0, "bf1_gt1": 0, "bf1_gt10": 0, "bf10_gt1": 0, "bf10_gt10": 0}
                for _ in range(reps):
                    y, x, w = zero_case(kind, n, d, rng)
                    test = conditional_mutual_information_test(x, y, w, n_surrogates=n_surrogates, prng=rng)
                    hits["perm_p05"] += test.pvalue <= 0.05
                    for ess in (1, 10):
                        lbf = log_bayes_factor(y, x, w, ess=ess)
                        hits[f"bf{ess}_gt1"] += lbf > 0
                        hits[f"bf{ess}_gt10"] += lbf > np.log(10)
                out[f"{kind}/n={n}/d={d}"] = {k: v / reps for k, v in hits.items()}
            print(f"zero {kind} done", flush=True)
    return out


# -- exploration parts: shared machinery --------------------------------------------


def _quiet():
    logging.getLogger("pymc").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def _pool(fn, jobs, workers):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers, initializer=_quiet) as ex:
        return list(ex.map(fn, jobs))


def _layers(method, y, x, w, seed, options=None):
    """
    ``(estimate, 90% interval or None, seconds)`` of one edge by one method.
    """
    import time

    from infoflow.layers import layer_statistics

    options = options or {}
    t0 = time.perf_counter()
    if method == "crossfit":
        est, ci = crossfit_flows(y, x, w, prng=seed).as_dict(), None
    elif method == "bootstrap":
        stats = layer_statistics(y, x, w, n_boot=100, n_null=0, confidence=0.9, prng=seed)
        est, ci = stats.flow.as_dict(), {k: stats.ci[k] for k in LAYERS}
    else:
        options = {"components": 4, "n_draws": 100, "draws": 300, "tune": 300, **options}
        post = layer_posterior(y, x, w, prng=seed, **options)
        est, ci = post.mean, post.interval(0.9)
    return est, ci, time.perf_counter() - t0


def _summary(records, truth):
    """
    RMSE, bias, interval coverage and width per layer from ``(estimate, ci, secs)`` records.
    """
    err = np.array([[r[0][k] - truth[k] for k in LAYERS] for r in records])
    out = {
        "rmse": dict(zip(LAYERS, np.sqrt((err**2).mean(0)).tolist(), strict=True)),
        "bias": dict(zip(LAYERS, err.mean(0).tolist(), strict=True)),
        "seconds_per_edge": float(np.mean([r[2] for r in records])),
    }
    if records[0][1] is not None:
        out["coverage"] = {
            k: float(np.mean([r[1][k][0] - 1e-9 <= truth[k] <= r[1][k][1] + 1e-9 for r in records])) for k in LAYERS
        }
        out["width"] = {k: float(np.mean([r[1][k][1] - r[1][k][0] for r in records])) for k in LAYERS}
    return out


# -- time series: serially dependent processes with exact flows ----------------------

TS_CASES = {
    "copy": {"a_copy": 0.8},
    "xor": {"a_xor": 0.9},
    "mixture": {"a_copy": 0.4, "a_xor": 0.3, "a_hold": 0.2},
    "copy_and_shared": {"a_copy": 0.5, "a_hold": 0.4, "x_given_y": 0.8},
}


def _ts_generators():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from tests import _generators

    return _generators


def _persistent_bits(n, d, rng, stay=0.9):
    """
    `d` independent binary Markov chains that keep their value with probability `stay`.
    """
    bits = np.zeros((n, d), dtype=np.int64)
    bits[0] = rng.integers(0, 2, d)
    flips = rng.random((n, d)) >= stay
    for t in range(1, n):
        bits[t] = bits[t - 1] ^ flips[t]
    return bits


def _ts_job(job):
    case, d, n, seed = job
    gen = _ts_generators()
    rng = np.random.default_rng(seed)
    obs = gen.sample_pairs(gen.mixture(**TS_CASES[case]), n + 1, seed)
    x, y = obs[:, 0], obs[:, 1]
    context = np.column_stack([y[:-1], _persistent_bits(n + 1, d, rng)[:-1]])
    return {m: _layers(m, y[1:], x[:-1], context, seed) for m in ("bootstrap", "mixture")}


def run_timeseries(dims=(0, 3, 6), n=2000, reps=20, workers=5):
    """
    The sofic test processes (state = previous pair), with the target's past padded by
    `d` independent persistent nuisance chains, so every sample is serially dependent
    and the exact flows are unchanged by the padding.
    """
    gen = _ts_generators()
    truths = {}
    for case, params in TS_CASES.items():
        t = gen.exact_flows(gen.mixture(**params))
        truths[case] = {**t, "tdmi": t["intrinsic"] + t["shared"]}
    jobs = [(case, d, n, 1000 * r + 7 * d) for case in TS_CASES for d in dims for r in range(reps)]
    results = _pool(_ts_job, jobs, workers)
    out = {"truth": truths}
    for case in TS_CASES:
        for d in dims:
            recs = [res for job, res in zip(jobs, results, strict=True) if job[0] == case and job[1] == d]
            for m in ("bootstrap", "mixture"):
                out[f"{case}/d={d}/{m}"] = _summary([r[m] for r in recs], truths[case])
    print("timeseries done", flush=True)
    return out


# -- crossover: samples per context ---------------------------------------------------


def _flips(d):
    return np.random.default_rng(100 + d).uniform(0.05, 0.45, 2**d)


def _crossover_job(job):
    d, n, seed = job
    y, x, w = graded(n, d, np.random.default_rng(seed), _flips(d))
    return {m: _layers(m, y, x, w, seed) for m in ("crossfit", "mixture")}


def run_crossover(dims=(2, 4, 6, 8), n_samples=(500, 1000, 2000, 4000, 8000), reps=4, workers=4):
    truths = {d: graded_truth(d, _flips(d)) for d in dims}
    jobs = [(d, n, 1000 * r + d + n) for d in dims for n in n_samples for r in range(reps)]
    results = _pool(_crossover_job, jobs, workers)
    out = {}
    for d in dims:
        for n in n_samples:
            recs = [res for job, res in zip(jobs, results, strict=True) if job[:2] == (d, n)]
            for m in ("crossfit", "mixture"):
                out[f"contexts={2**d}/n={n}/{m}"] = _summary([r[m] for r in recs], truths[d])
    print("crossover done", flush=True)
    return out


# -- components and the alpha prior -----------------------------------------------------

COMPONENT_CONFIGS = {
    "G=1": {"components": 1},
    "G=2": {"components": 2},
    "G=4": {"components": 4},
    "G=8": {"components": 8},
    "G=16": {"components": 16},
    "G=4,sd=0.5": {"components": 4, "prior_options": {"log_alpha_sd": 0.5}},
    "G=4,sd=5": {"components": 4, "prior_options": {"log_alpha_sd": 5.0}},
}


def _sparse_case(name, seed):
    rng = np.random.default_rng(seed)
    if name == "graded":
        y, x, w = graded(1000, 7, rng, _flips(7))
        return y, x, w, graded_truth(7, _flips(7))
    return motif(name, 1000, 5, rng)


def _components_job(job):
    name, config, seed = job
    y, x, w, _ = _sparse_case(name, seed)
    return _layers("mixture", y, x, w, seed, COMPONENT_CONFIGS[config])


def run_components(reps=6, workers=3):
    names = ("graded", "xor", "driver")
    truths = {name: _sparse_case(name, 0)[3] for name in names}
    jobs = [(name, c, 1000 * r + 11) for name in names for c in COMPONENT_CONFIGS for r in range(reps)]
    results = _pool(_components_job, jobs, workers)
    out = {}
    for name in names:
        for c in COMPONENT_CONFIGS:
            recs = [res for job, res in zip(jobs, results, strict=True) if job[:2] == (name, c)]
            out[f"{name}/{c}"] = _summary(recs, truths[name])
    print("components done", flush=True)
    return out


# -- speed ----------------------------------------------------------------------------

SPEED_CONFIGS = {
    "nuts 300/300 x2, 100 tables": {},
    "nuts 150/150 x1, 100 tables": {"draws": 150, "tune": 150, "chains": 1},
    "nuts 150/150 x1, 50 tables": {"draws": 150, "tune": 150, "chains": 1, "n_draws": 50},
    "nuts 100/100 x1, 50 tables": {"draws": 100, "tune": 100, "chains": 1, "n_draws": 50},
    "nutpie 300/300 x2, 100 tables": {"prior_options": {"nuts_sampler": "nutpie"}},
    "advi, 100 tables": {"chains": 1, "prior_options": {"method": "advi"}},
}


def run_speed(reps=6):
    """
    Sequential, configurations interleaved within each replicate so that any load from
    concurrent work affects them alike.
    """
    _quiet()
    names = ("graded", "xor")
    truths = {name: _sparse_case(name, 0)[3] for name in names}
    recs = {(name, c): [] for name in names for c in SPEED_CONFIGS}
    for r in range(reps):
        for name in names:
            seed = 1000 * r + 13
            y, x, w, _ = _sparse_case(name, seed)
            for c, options in SPEED_CONFIGS.items():
                recs[name, c].append(_layers("mixture", y, x, w, seed, options))
    out = {f"{name}/{c}": _summary(v, truths[name]) for (name, c), v in recs.items()}
    print("speed done", flush=True)
    return out


# -- end to end: full networks ----------------------------------------------------------

FAST = {"components": 2, "draws": 100, "tune": 100, "chains": 1, "n_draws": 50}


def _network_data(name, n, seed):
    from infoflow import datasets

    if name == "xor":
        return datasets.xor_synergy(n, noise=0.1, seed=seed)
    if name == "driver":
        return datasets.common_driver(n, seed=seed)
    if name == "chain":
        return datasets.chain(n, seed=seed)
    return datasets.noisy_eca(n, 6, int(name[3:]), 0.05, seed=seed)


def _edge_arrays(data, target, x_vars, w_vars):
    from infoflow.data import realizations

    variables = list(dict.fromkeys(list(x_vars) + list(w_vars)))
    r = realizations(data, target, variables, max_lag=max((v[1] for v in variables), default=0))
    return r.present, r.columns(list(x_vars)), (r.columns(list(w_vars)) if w_vars else None)


def _patch_layers(mode, records):
    """
    Wrap ``infoflow.network.estimate_edge``: always fit the fast mixture next to the
    usual estimate and record both; in mode ``'bayes'`` the network gets the mixture's
    weights and 95% intervals, while p-values (hence significance) are unchanged.
    """
    import infoflow.network as network
    from infoflow.measures import dense_codes

    original = network.estimate_edge

    def estimate_edge(data, target, x_vars, w_vars, *args, adaptive=None, **kwargs):
        stats, delay = original(data, target, x_vars, w_vars, *args, adaptive=adaptive, **kwargs)
        if adaptive is not None:
            return stats, delay
        y, x, w = _edge_arrays(data, target, x_vars, w_vars)
        post = layer_posterior(y, x, w, prng=len(records), **FAST)
        records.append(
            {
                "target": int(target),
                "x_vars": [list(map(int, v)) for v in x_vars],
                "w_vars": [list(map(int, v)) for v in w_vars],
                "n": len(y),
                "contexts": int(dense_codes(w)[1]) if w is not None else 1,
                "crossfit": stats.flow.as_dict(),
                "crossfit_ci": {k: list(stats.ci[k]) for k in LAYERS},
                "bayes": post.mean,
                "bayes_ci": {k: list(v) for k, v in post.interval(0.95).items()},
            }
        )
        if mode == "bayes":
            stats.flow, stats.ci = post.as_flow(), post.interval(0.95)
        return stats, delay

    network.estimate_edge = estimate_edge
    return original


def _network_job(job):
    import time

    import infoflow.network as network
    from infoflow.crossfit import crossfit_flows as reference_flows
    from infoflow.data import DiscreteData

    name, n, seed = job
    data = _network_data(name, n, seed)
    out = {}
    for mode in ("crossfit", "bayes"):
        records = []
        original = _patch_layers(mode, records) if mode == "bayes" else None
        t0 = time.perf_counter()
        ds = network.infer_multiplex(data, max_lag=3, prng=seed).dataset
        out[f"{mode}_seconds"] = time.perf_counter() - t0
        if original is not None:
            network.estimate_edge = original
        out[f"{mode}_roles"] = ds["role"].values.tolist()
        out[f"{mode}_flags"] = ds["flags"].values.tolist()
        out[f"{mode}_significant"] = ds["significant"].values.tolist()
    long = DiscreteData.from_discrete(_network_data(name, 200_000, seed + 99_991))
    for rec in records:
        y, x, w = _edge_arrays(
            long, rec["target"], [tuple(v) for v in rec["x_vars"]], [tuple(v) for v in rec["w_vars"]]
        )
        rec["reference"] = reference_flows(y, x, w, prng=0).as_dict()
    out["edges"] = records
    return out


def run_network(names=("xor", "driver", "chain", "eca90", "eca110"), n_samples=(500, 2000), reps=3, workers=6):
    from infoflow.benchmarks import motif_validation

    jobs = [(name, n, 1000 * r + 17) for name in names for n in n_samples for r in range(reps)]
    results = _pool(_network_job, jobs, workers)
    out = {"runs": {f"{j[0]}/n={j[1]}/seed={j[2]}": res for j, res in zip(jobs, results, strict=True)}}
    summary = {}
    for name in names:
        for n in n_samples:
            runs = [res for j, res in zip(jobs, results, strict=True) if j[:2] == (name, n)]
            edges = [e for r in runs for e in r["edges"]]
            s = {
                "edges": len(edges),
                "median_contexts": float(np.median([e["contexts"] for e in edges])),
                "min_samples_per_context": float(min(e["n"] / e["contexts"] for e in edges)),
                "same_significance": all(r["crossfit_significant"] == r["bayes_significant"] for r in runs),
                "role_agreement": float(
                    np.mean(
                        [
                            a == b
                            for r in runs
                            for ra, rb in zip(r["crossfit_roles"], r["bayes_roles"], strict=True)
                            for a, b in zip(ra, rb, strict=True)
                        ]
                    )
                ),
                "flag_agreement": float(
                    np.mean(
                        [
                            a == b
                            for r in runs
                            for ra, rb in zip(r["crossfit_flags"], r["bayes_flags"], strict=True)
                            for a, b in zip(ra, rb, strict=True)
                        ]
                    )
                ),
                "crossfit_seconds": float(np.mean([r["crossfit_seconds"] for r in runs])),
                "bayes_seconds": float(np.mean([r["bayes_seconds"] for r in runs])),
            }
            for m in ("crossfit", "bayes"):
                err = np.array([[e[m][k] - e["reference"][k] for k in LAYERS] for e in edges])
                s[f"{m}_rmse"] = dict(zip(LAYERS, np.sqrt((err**2).mean(0)).tolist(), strict=True))
                s[f"{m}_coverage_positive"] = {
                    k: float(np.mean(cov))
                    if (
                        cov := [
                            e[f"{m}_ci"][k][0] - 1e-9 <= e["reference"][k] <= e[f"{m}_ci"][k][1] + 1e-9
                            for e in edges
                            if e["reference"][k] > 0.005
                        ]
                    )
                    else None
                    for k in LAYERS
                }
            summary[f"{name}/n={n}"] = s
    out["summary"] = summary
    print("network runs done", flush=True)
    rates = {}
    for mode in ("crossfit", "bayes"):
        import infoflow.network as network

        original = _patch_layers(mode, []) if mode == "bayes" else None
        ds = motif_validation(n_samples=3000, n_reps=5, prng=0)
        if original is not None:
            network.estimate_edge = original
        rates[mode] = dict(zip(ds["check"].values.tolist(), ds["rate"].values.tolist(), strict=True))
    out["motif_validation"] = rates
    print("network done", flush=True)
    return out


PARTS = {
    "motifs": run_motifs,
    "robust": run_robust,
    "scaling": run_scaling,
    "zero": run_zero,
    "timeseries": run_timeseries,
    "crossover": run_crossover,
    "components": run_components,
    "speed": run_speed,
    "network": run_network,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="*", default=list(PARTS))
    args = parser.parse_args()
    _quiet()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        result = PARTS[part]()
        (OUT / f"{part}.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
