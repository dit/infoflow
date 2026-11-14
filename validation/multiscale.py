"""Bounded multiscale spectral validation, including failure regimes.

Run from the repo root, for example::

    python validation/multiscale.py --study all --reps 20 --n-null 99 \
        --output reports/multiscale_validation.json

This is an empirical Gaussian-bootstrap calibration study, not a proof of FDR
control for arbitrary stationary processes. Structural and predictive graphs are
reported separately. Every configuration, seed and allocated observation budget
is recorded; long runs are opt-in, never an import side effect.
"""

import argparse
import datetime
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.stats import binomtest

from infoflow import datasets
from infoflow.backbone import BackboneSettings, backbone_from_rates, infer_backbone
from infoflow.embedding import Embedding
from infoflow.network import infer_multiplex
from infoflow.preprocess import EqualFrequency, discretize
from infoflow.preprocess.discretizers import reencode
from infoflow.selection import SkeletonSettings
from infoflow.spectral import SpectralSettings, rates_from_spectrum

MOTIFS = (
    ((0, 1), (1, 2)),
    ((0, 1), (2, 0)),
    ((1, 0), (0, 2)),
    ((0, 2), (2, 1)),
    ((2, 1), (1, 0)),
    ((1, 2), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (1, 2)),
    ((1, 0), (2, 0)),
    ((0, 1), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 1)),
)


def _coupling(edges, P=3, weight=1.0):
    A = np.zeros((P, P))
    for s, t in edges:
        A[t, s] = weight
    return A


def _sampled_spectrum(f, A, tau, dt):
    tau = np.asarray(tau)
    F = (A - np.eye(len(A))) / tau[:, None]
    cov = solve_continuous_lyapunov(F, -np.diag(2 / tau))
    transition = expm(F * dt)
    noise = cov - transition @ cov @ transition.T
    B = np.eye(len(A))[None] - transition[None] * np.exp(-2j * np.pi * f * dt)[:, None, None]
    inv = np.linalg.inv(B)
    return dt * inv @ noise @ inv.conj().swapaxes(1, 2)


def _binomial_interval(count, n):
    ci = binomtest(count, n).proportion_ci(confidence_level=0.95)
    return [float(ci.low), float(ci.high)]


def _score(detected, edges):
    truth = {tuple(sorted(e)) for e in edges}
    found = {pair for pair in combinations(range(len(detected)), 2) if detected[pair]}
    tp, fp = len(found & truth), len(found - truth)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "truth_edges": len(truth),
        "detected_edges": len(found),
        "recall": tp / len(truth) if truth else None,
        "precision": tp / len(found) if found else None,
        "false_discovery_proportion": fp / max(len(found), 1),
    }


def _rss_bytes():
    if sys.platform == "win32":
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def _fit(x, tau, args, seed, nperseg=None, fs=2, shrinkage=0):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        b = infer_backbone(
            x,
            fs=fs,
            timescales=tau,
            spectral_settings=SpectralSettings(nperseg=nperseg or args.nperseg, shrinkage=shrinkage),
            settings=BackboneSettings(n_null=args.n_null, alpha=args.alpha),
            assume_forest=True,
            prng=seed,
        )
    return b, {
        "seed": seed,
        "n": len(x),
        "fs": fs,
        "timescales": list(map(float, tau)),
        "nperseg": nperseg or args.nperseg,
        "shrinkage": shrinkage,
        "seconds": time.perf_counter() - start,
        "process_peak_rss_bytes": _rss_bytes(),  # process high-water mark, not per-fit allocation
        "unresolved_pairs": int(np.triu(b.dataset.status.values == "unresolved", 1).sum()),
        "resolved_arrows": int(np.triu(b.dataset.orientation.values != 0, 1).sum()),
        "detected_graph_is_forest": b.is_forest,
        "warnings": [str(w.message) for w in caught] + list(b.warnings),
    }


def calibration(args):
    """Marginal and conditional nulls across independent Gaussian replications."""
    models = {
        "independent": ([], 2, (0, 1), "pvalue"),
        "common_driver": ([(0, 1), (0, 2)], 3, (1, 2), "conditional_pvalue"),
        "chain": ([(0, 1), (1, 2)], 3, (0, 2), "conditional_pvalue"),
        "collider": ([(0, 2), (1, 2)], 3, (0, 1), "marginal_pvalue"),
    }
    summary, records = {}, []
    for name, (edges, P, pair, component) in models.items():
        pvalues, false_families, fdps = [], [], []
        for seed in range(args.reps):
            tau = 2.0 ** np.arange(P)
            x = datasets.multiscale_ou(args.n, _coupling(edges, P), tau, dt=0.5, seed=seed)
            b, row = _fit(x, tau, args, seed)
            score = _score(b.dataset.detected.values, edges)
            pv = float(b.dataset[component].values[pair])
            pvalues.append(pv)
            false_families.append(score["false_positive"] > 0)
            fdps.append(score["false_discovery_proportion"])
            records.append(row | score | {"model": name, "null_component": component, "null_pvalue": pv})
        valid = np.isfinite(pvalues)
        count = int(np.count_nonzero(np.asarray(pvalues)[valid] <= args.alpha))
        summary[name] = {
            "replications": args.reps,
            "estimable_null_tests": int(valid.sum()),
            "null_component": component,
            "component_rejections": count,
            "component_rejection_interval_95": _binomial_interval(count, int(valid.sum())) if valid.any() else None,
            "families_with_false_edges": sum(false_families),
            "false_discovery_rate_estimate": float(np.mean(fdps)),
            "fdp_standard_error": float(np.std(fdps, ddof=1) / np.sqrt(args.reps)) if args.reps > 1 else None,
            "nominal_alpha": args.alpha,
        }
    return {"summary": summary, "records": records}


def motifs(args):
    """The complete three-node motif/ratio grid plus deliberately adverse windows."""
    records = []
    for ratio in args.ratios:
        tau = ratio ** np.arange(3, dtype=float)
        for k, edges in enumerate(MOTIFS):
            A = _coupling(edges)
            x = datasets.multiscale_ou(args.n, A, tau, dt=0.5, seed=100 + k)
            b, row = _fit(x, tau, args, 100 + k)
            f = b.rates.dataset.frequency.values
            reference = rates_from_spectrum(f, _sampled_spectrum(f, A, tau, 0.5))
            mask = ~np.eye(3, dtype=bool)
            row |= _score(b.dataset.detected.values, edges)
            row |= {
                "motif": k,
                "ratio": ratio,
                "edges": edges,
                "duration_over_slowest_tau": args.n * 0.5 / tau[-1],
                "segment_duration_over_slowest_tau": args.nperseg * 0.5 / tau[-1],
                "marginal_rate_mae_bits_per_time": float(
                    np.mean(np.abs(b.rates.dataset.rate.values[mask] - reference.dataset.rate.values[mask]))
                ),
                "conditional_rate_mae_bits_per_time": float(
                    np.mean(
                        np.abs(
                            b.rates.dataset.conditional_rate.values[mask]
                            - reference.dataset.conditional_rate.values[mask]
                        )
                    )
                ),
            }
            records.append(row)
    return records


def sensitivity(args):
    """One-factor changes, larger forests and too few spectral replicates."""
    records = []
    cases = [
        ("short", 2048, 128, 0.5, 1.0, 0, 3),
        ("long", max(args.n, 8192), 256, 0.5, 1.0, 0, 3),
        ("long_segments", args.n, 512, 0.5, 1.0, 0, 3),
        ("weak", args.n, 128, 0.5, 0.4, 0, 3),
        ("coarse_sampling", args.n, 128, 2.0, 1.0, 0, 3),
        ("fine_sampling", args.n, 128, 0.125, 1.0, 0, 3),
        ("regularized", args.n, 128, 0.5, 1.0, 0.1, 3),
        ("star_six", args.n, 256, 0.5, 1.0, 0, 6),
        ("chain_six", args.n, 256, 0.5, 1.0, 0, 6),
        ("underidentified", args.n, args.n // 2, 0.5, 1.0, 0, 8),
    ]
    for k, (name, n, segment, dt, weight, shrinkage, P) in enumerate(cases):
        edges = [(i, i + 1) for i in range(P - 1)] if "chain" in name else [(0, p) for p in range(1, P)]
        tau = 2.0 ** np.arange(P)
        x = datasets.multiscale_ou(n, _coupling(edges, P, weight), tau, dt=dt, seed=300 + k)
        b, row = _fit(x, tau, args, 300 + k, nperseg=segment, fs=1 / dt, shrinkage=shrinkage)
        records.append(
            row | _score(b.dataset.detected.values, edges) | {"case": name, "coupling": weight, "edges": edges}
        )
    return records


def _nonlinear(n, strength, kind, seed):
    """Euler reference used only for the explicitly non-Gaussian failure study."""
    tau = np.array([1.0, 2.0, 4.0])
    dt, stride, burn = 0.02, 25, 4000
    rng = np.random.default_rng(seed)
    x = np.zeros(3)
    out = []
    noise = rng.normal(size=(burn + n * stride, 3)) * np.sqrt(2 * dt / tau)
    for t, epsilon in enumerate(noise):
        f = np.tanh(strength * x) / strength if kind == "tanh" else x - strength * x**3
        drive = np.array([0, f[0], f[0]])
        x += dt * (-x + drive) / tau + epsilon
        if t >= burn and (t - burn) % stride == 0:
            out.append(x.copy())
    return np.asarray(out)


def failures(args):
    """Show model violations, including ones a spectral graph cannot diagnose."""
    records = []
    for kind, strength in (("tanh", 0.1), ("tanh", 2.0), ("cubic", 0.01), ("cubic", 0.2)):
        x = _nonlinear(args.n, strength, kind, seed=400)
        b, row = _fit(x, [1, 2, 4], args, 400)
        records.append(
            row
            | _score(b.dataset.detected.values, [(0, 1), (0, 2)])
            | {"case": kind, "strength": strength, "outside_theorem": "nonlinear dynamics; Euler reference at dt=.02"}
        )
    rng = np.random.default_rng(401)
    noise = rng.normal(size=(args.n, 2)) @ np.linalg.cholesky([[1, 0.8], [0.8, 1]]).T
    x = np.zeros_like(noise)
    for t in range(1, len(x)):
        x[t] = [0.7, 0.85] * x[t - 1] + noise[t]
    cases = [
        (
            "xor",
            datasets.xor_synergy(args.n, seed=402).astype(float),
            [1, 2, 4],
            [(0, 2), (1, 2)],
            "pure synergy invisible to Gaussian spectra",
        ),
        ("correlated_noise", x, [1, 2], [], "correlated innovations with no dynamical coupling"),
        (
            "reciprocal",
            datasets.multiscale_ou(args.n, [[0, 0.4], [0.4, 0]], [1, 2], dt=0.5, seed=403),
            [1, 2],
            [(0, 1), (1, 0)],
            "two directed interactions per undirected edge",
        ),
        (
            "triangle",
            datasets.multiscale_ou(args.n, _coupling([(0, 1), (1, 2), (0, 2)]), [1, 2, 4], dt=0.5, seed=404),
            [1, 2, 4],
            [(0, 1), (1, 2), (0, 2)],
            "undirected cycle, despite being a directed DAG",
        ),
    ]
    hidden = datasets.multiscale_ou(args.n, _coupling([(0, 1), (0, 2)], P=3), [1, 2, 4], dt=0.5, seed=405)
    cases.append(("hidden_driver", hidden[:, 1:], [2, 4], [], "unobserved common driver"))
    drift = rng.normal(size=(args.n, 3))
    drift += np.linspace(-3, 3, args.n)[:, None]
    cases.append(("nonstationary", drift, [1, 2, 4], [], "shared deterministic drift, nonstationary"))
    for k, (name, data, tau, edges, note) in enumerate(cases):
        b, row = _fit(data, tau, args, 410 + k)
        records.append(row | _score(b.dataset.detected.values, edges) | {"case": name, "outside_theorem": note})
    return records


def handoff(args):
    """Independent discovery, TE selection and layer evaluation with frozen encoders."""
    edges, tau = [(0, 1), (0, 2)], [1, 2, 4]
    A = _coupling(edges)
    d0, d1, d2 = [datasets.multiscale_ou(args.n, A, tau, dt=0.5, seed=500 + k) for k in range(3)]
    b, discovery = _fit(d0, tau, args, 500)
    # Preprocessing is learned only on discovery data. The concatenation boundary
    # is exactly the network's holdout boundary; no lagged window crosses it.
    template = discretize(d0, EqualFrequency(3))
    symbols = reencode(template, [np.concatenate([d1, d2])])
    # A population oracle is useful for identifying sample-estimation failures,
    # but is not substituted for the actual empirical filter below.
    f = b.rates.dataset.frequency.values
    population = backbone_from_rates(rates_from_spectrum(f, _sampled_spectrum(f, A, tau, 0.5)), assume_forest=True)
    rows = []
    for name, estimator, backbone in (
        ("plugin", "plugin", None),
        ("gaussian", "gaussian", None),
        ("guided_gaussian", "gaussian", b),
    ):
        settings = SkeletonSettings(
            estimator=estimator,
            max_source_lag=5,
            n_perm_max_stat=99,
            n_perm_min_stat=99,
            n_perm_omnibus=99,
            n_perm_max_seq=99,
            n_perm_pairs=99,
            n_perm_tdmi=199,
        )
        start = time.perf_counter()
        if backbone is not None and not backbone.is_forest:
            rows.append({"method": name, "skipped": "empirical forest conflict; no hard filtering"})
            continue
        net = infer_multiplex(
            symbols,
            embeddings=Embedding(5),
            backbone=backbone,
            holdout=0.5,
            skeleton_settings=settings,
            n_boot=0,
            n_null=99,
            adaptive_resamples=False,
            node_layer=False,
            interpret=False,
            prng=501,
        )
        seconds = time.perf_counter() - start
        selected = net.dataset.kind.values == "parent"
        layers = net.dataset.significant.sel(layer=["intrinsic", "synergistic"]).values.any(axis=0)
        whole = net.dataset.weight.sel(layer="te").values
        parts = net.dataset.weight.sel(layer="intrinsic").values + net.dataset.weight.sel(layer="synergistic").values
        rows.append(
            {
                "method": name,
                "allocated_samples": 3 * args.n,
                "discovery_samples": args.n,
                "selection_samples": args.n,
                "layer_samples": args.n,
                "seconds_without_discovery": seconds,
                "seconds_including_discovery": seconds + (discovery["seconds"] if backbone is not None else 0),
                "selected_directed_edges": np.argwhere(selected).tolist(),
                "layers_confirmed_directed_edges": np.argwhere(layers).tolist(),
                "structural_undirected_score": _score(selected | selected.T, edges),
                "te_decomposition_identity": bool(np.allclose(whole, parts, equal_nan=True)),
                "shared_only_edges": np.argwhere(net.dataset.kind.values == "shared_candidate").tolist(),
            }
        )
    return {
        "empirical_discovery": discovery | _score(b.dataset.detected.values, edges),
        "sampled_population_graph_is_forest": population.is_forest,
        "comparisons": rows,
    }


def run(args):
    functions = {
        "calibration": calibration,
        "motifs": motifs,
        "sensitivity": sensitivity,
        "failures": failures,
        "handoff": handoff,
    }
    chosen = functions if args.study == "all" else {args.study: functions[args.study]}
    result = {
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: version(p) for p in ("infoflow", "dit", "numpy", "scipy")},
        "configuration": {
            "n": args.n,
            "reps": args.reps,
            "ratios": args.ratios,
            "nperseg": args.nperseg,
            "backbone": asdict(BackboneSettings(args.n_null, args.alpha)),
        },
        "limitations": [
            "Fitted Gaussian tests are approximate and model/regime specific.",
            "Finite sampled predictive and continuous generating structural graphs differ.",
            "Rates are biased plug-in spectral estimates; nulls, not positivity, determine detections.",
            "RSS is the process-wide high-water mark, not per-case memory use.",
        ],
    }
    for name, function in chosen.items():
        print(f"Running {name}", file=sys.stderr, flush=True)
        result[name] = function(args)
    return result


def _json_ready(value):
    """Report unestimable numeric values as JSON null, never nonstandard NaN."""
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("all", "calibration", "motifs", "sensitivity", "failures", "handoff"), default="all"
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--n-null", type=int, default=99)
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ratios", type=float, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 1 or args.n < max(512, args.nperseg):
        parser.error("need reps >= 1 and n >= max(512, nperseg)")
    result = run(args)
    text = json.dumps(_json_ready(result), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"Saved {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
