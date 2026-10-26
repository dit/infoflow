"""
Subsequent-memory contrast of infoflow networks on catFR1 encoding epochs.

For each prepared session and signal:

1. Preprocess (discretize, choose embeddings) once on all of the session's
   trials, so every condition and every null split shares one symbol encoding.
2. Balance recalled and forgotten words by subsampling the larger class, select
   one network on both conditions together with ``infer_multiplex`` on the shared
   encoding (plug-in selection at auto's level), and test the per-layer
   differences on its edges with ``compare_networks(design="within")``. With
   ``--selection condition`` the tested edges are instead the union of one network
   per condition, which makes the test anti-conservative.
3. Calibration null: two disjoint same-condition draws from the larger class,
   each the size of the main contrast's arms (capped at half the class), pushed
   through the same steps ``--n-null`` times.
4. ``auto.infer`` on the balanced pooled trials, for its report.

    uv run --project ../.. --with netcdf4 python run.py --signal both
"""

import argparse
import json
import time

import numpy as np
import xarray as xr
from common import OUT

from infoflow import infer_multiplex
from infoflow.auto import _skeleton_settings, infer
from infoflow.compare import compare_networks
from infoflow.embedding import Embedding
from infoflow.layers import LAYERS
from infoflow.parallel import thread_map
from infoflow.preprocess import preprocess

DEFAULT_MAX_LAG = {"broadband": 40, "hfa": 15}


def condition_network(pre, idx, names, threads):
    settings = _skeleton_settings(("plugin",), False, 1, None, {})
    return infer_multiplex(
        pre.data.subset_trials(list(idx)),
        names=names,
        embeddings=pre.embeddings,
        skeleton_settings=settings,
        include_shared_candidates=False,
        interpret=False,
        map_fn=thread_map(threads),
        prng=0,
    )


def contrast(pre, idx_a, idx_b, names, n_perm, threads, seed, selection="condition", curtail=10):
    """
    ``selection='condition'`` tests the union of the per-condition skeletons, as
    ``compare_networks`` is designed; ``'pooled'`` tests the skeleton selected on both
    conditions together, which does not depend on the labels being permuted.
    """
    data = pre.data
    if selection == "pooled":
        net_a = net_b = condition_network(pre, np.concatenate([idx_a, idx_b]), names, threads)
    else:
        net_a = condition_network(pre, idx_a, names, threads)
        net_b = condition_network(pre, idx_b, names, threads)
    cmp = compare_networks(
        net_a,
        net_b,
        data.subset_trials(list(idx_a)),
        data.subset_trials(list(idx_b)),
        design="within",
        n_perm=n_perm,
        curtail=curtail,
        map_fn=thread_map(threads),
        prng=seed,
    )
    return net_a, net_b, cmp


def layer_counts(net):
    return {k: int(net.dataset["significant"].sel(layer=k).sum()) for k in LAYERS if k in net.dataset.layer}


def tested(cmp):
    return {k: int(np.isfinite(cmp.dataset["pvalue"].sel(layer=k)).sum()) for k in LAYERS}


def significant_edges(cmp, regions):
    names = list(cmp.dataset.source.values)
    out = []
    for k in LAYERS:
        sig = cmp.dataset["significant"].sel(layer=k).values
        for s, t in zip(*np.nonzero(sig)):
            out.append(
                {
                    "layer": k,
                    "source": names[s],
                    "target": names[t],
                    "regions": f"{regions[s]}->{regions[t]}",
                    "recalled": float(cmp.dataset["a"].sel(layer=k).values[s, t]),
                    "forgotten": float(cmp.dataset["b"].sel(layer=k).values[s, t]),
                    "qvalue": float(cmp.dataset["qvalue"].sel(layer=k).values[s, t]),
                }
            )
    return out


def run_one(path, args):
    ds = xr.open_dataset(path).load()
    signal = ds.attrs["signal"]
    stem = path.name.removesuffix(".nc")
    regions = [str(r) for r in ds.region.values]
    names = [f"{r}:{c}" for r, c in zip(regions, ds.channel.values)]
    trials = [ds.signal.values[i].astype(float) for i in range(ds.sizes["trial"])]
    recalled = ds.recalled.values.astype(bool)
    rng = np.random.default_rng(0)
    rec, forg = np.flatnonzero(recalled), np.flatnonzero(~recalled)
    n = min(len(rec), len(forg))
    idx_rec = np.sort(rng.choice(rec, n, replace=False))
    idx_forg = np.sort(rng.choice(forg, n, replace=False))
    max_lag = args.max_lag or DEFAULT_MAX_LAG[signal]
    summary = {
        "file": path.name,
        "signal": signal,
        "sfreq": ds.attrs["sfreq"],
        "n_per_condition": int(n),
        "n_recalled": int(len(rec)),
        "n_forgotten": int(len(forg)),
        "max_lag": max_lag,
        "names": names,
        "timing": {},
    }

    t0 = time.perf_counter()
    pre = preprocess(
        trials,
        names=names,
        max_lag=max_lag,
        network_resolution="auto",
        map_fn=thread_map(args.threads),
        prng=0,
    )
    summary["timing"]["preprocess"] = time.perf_counter() - t0
    summary["preprocessing"] = pre.report.summary()
    summary["lags"] = args.lags
    if args.lags == "long":
        pre.embeddings = [Embedding(max_lag=max_lag, log_spaced=True) for _ in names]
        stem += "_long"
    summary["selection"] = args.selection
    if args.selection == "pooled":
        stem += "_pooled"
    summary["candidate_lags"] = sorted({lag for e in pre.embeddings for lag in e.lags()})
    print(f"[{stem}] preprocessed in {summary['timing']['preprocess']:.0f}s\n{summary['preprocessing']}", flush=True)

    t0 = time.perf_counter()
    net_rec, net_forg, cmp = contrast(pre, idx_rec, idx_forg, names, args.n_perm, args.threads, 0, args.selection)
    summary["timing"]["contrast"] = time.perf_counter() - t0
    summary["edges"] = {"recalled": layer_counts(net_rec), "forgotten": layer_counts(net_forg)}
    summary["tested"] = tested(cmp)
    summary["sme_edges"] = significant_edges(cmp, regions)
    cmp.dataset.to_netcdf(OUT / f"{stem}_compare.nc")
    for label, net in (("recalled", net_rec), ("forgotten", net_forg)):
        net.dataset.drop_vars([v for v in net.dataset.data_vars if net.dataset[v].dtype == object]).to_netcdf(
            OUT / f"{stem}_net_{label}.nc"
        )
    print(
        f"[{stem}] contrast in {summary['timing']['contrast']:.0f}s: edges {summary['edges']}, "
        f"tested {summary['tested']}, significant SME {len(summary['sme_edges'])}",
        flush=True,
    )

    major = rec if len(rec) >= len(forg) else forg
    m = min(n, len(major) // 2)
    null = []
    t0 = time.perf_counter()
    for r in range(args.n_null):
        draw = np.random.default_rng(1000 + r).permutation(major)
        _, _, c = contrast(
            pre, np.sort(draw[:m]), np.sort(draw[m : 2 * m]), names, args.n_perm, args.threads, 1000 + r, args.selection
        )
        sig = {k: int(c.dataset["significant"].sel(layer=k).sum()) for k in LAYERS}
        null.append({"significant": sig, "tested": tested(c)})
        print(f"[{stem}] null {r}: {sig}", flush=True)
    summary["timing"]["null"] = time.perf_counter() - t0
    summary["null"] = {"class": "recalled" if major is rec else "forgotten", "n_per_split": int(m), "reps": null}

    path_out = OUT / f"{stem}_summary.json"
    path_out.write_text(json.dumps(summary, indent=2, default=str))

    if not args.skip_auto and args.lags == "auto":
        t0 = time.perf_counter()
        pooled = [trials[i] for i in np.concatenate([idx_rec, idx_forg])]
        try:
            result = infer(
                pooled,
                preset="balanced",
                time_budget=args.time_budget,
                names=names,
                max_lag=max_lag,
                threads=args.threads,
                prng=0,
            )
        except Exception as error:  # noqa: BLE001
            summary["auto_error"] = f"{type(error).__name__}: {error}"
            print(f"[{stem}] auto failed: {summary['auto_error']}", flush=True)
        else:
            summary["auto_report"] = str(result.report)
            summary["auto_edges"] = layer_counts(result.network)
            print(f"[{stem}] auto in {time.perf_counter() - t0:.0f}s\n{result.report}", flush=True)
        summary["timing"]["auto"] = time.perf_counter() - t0
        path_out.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=("broadband", "hfa", "both"), default="both")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--max-lag", type=int)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--time-budget", default="2h")
    parser.add_argument("--skip-auto", action="store_true")
    parser.add_argument(
        "--lags",
        choices=("auto", "long"),
        default="auto",
        help="auto: preprocessing's lag budgets; long: log-spaced lags up to --max-lag",
    )
    parser.add_argument(
        "--selection",
        choices=("condition", "pooled"),
        default="pooled",
        help="edges tested: union of per-condition skeletons, or the pooled skeleton",
    )
    args = parser.parse_args()
    signals = ("hfa", "broadband") if args.signal == "both" else (args.signal,)
    for signal in signals:
        for path in sorted(OUT.glob(f"sub-*_{signal}.nc")):
            if args.subjects and not any(s in path.name for s in args.subjects):
                continue
            run_one(path, args)


if __name__ == "__main__":
    main()
