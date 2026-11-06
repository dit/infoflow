"""
Calibration of ``compare_networks`` on random splits of one condition (catFR1).

All of one subject's recalled trials are split into two halves at random many times.
The tested network is selected once on all of them, so it is the same for every split,
and the split is one draw from the permutation distribution that the test compares it
with: for each edge and layer, ``P(p <= alpha) <= alpha`` exactly. The script records
every raw p-value, so their distribution can be checked directly.

    python validation/synergy_null.py --epochs ~/data/catfr_pilot_v3/sub-R1332M_ses-0_hfa.nc \\
        --splits 1000 --estimator miller_madow --out synergy_null_mm.json
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "catfr"))

STATE = {}


def _init(data, net):
    STATE["data"], STATE["net"] = data, net


def _split(args):
    from infoflow.compare import compare_networks
    from infoflow.layers import LAYERS

    seed, pool, half, n_perm, estimator = args
    draw = np.random.default_rng(seed).permutation(pool)
    data, net = STATE["data"], STATE["net"]
    cmp = compare_networks(
        net, net, data.subset_trials(list(np.sort(draw[:half]))), data.subset_trials(list(np.sort(draw[half:]))),
        n_perm=n_perm, estimator=estimator, prng=seed,
    )  # fmt: skip
    out = {}
    for k in LAYERS:
        p = cmp.dataset["pvalue"].sel(layer=k).values
        out[k] = [float(p[s, t]) for s, t in zip(*np.nonzero(np.isfinite(p)), strict=True)]
    return seed, out


def main():
    import xarray as xr
    from run import condition_network

    from infoflow.parallel import thread_map
    from infoflow.preprocess import preprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--splits", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--estimator", default="miller_madow")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ds = xr.open_dataset(args.epochs).load()
    names = [f"{r}:{c}" for r, c in zip(ds.region.values, ds.channel.values, strict=True)]
    trials = [ds.signal.values[i].astype(float) for i in range(ds.sizes["trial"])]
    pool = np.flatnonzero(ds.recalled.values)
    pre = preprocess(trials, names=names, max_lag=15, network_resolution="auto", map_fn=thread_map(8), prng=0)
    net = condition_network(pre, pool, names, 8)
    edges = [(names[p], names[t]) for t, sk in net.skeleton.items() if sk.significant for p in sk.parents()]
    print(f"{len(pool)} trials, edges {edges}", flush=True)
    jobs = [(1000 + r, pool, len(pool) // 2, args.n_perm, args.estimator) for r in range(args.splits)]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(pre.data, net)) as pool_:
        for seed, out in pool_.map(_split, jobs, chunksize=4):
            rows.append({"seed": seed, "pvalues": out})
    Path(args.out).write_text(json.dumps({"edges": edges, "estimator": args.estimator, "splits": rows}))
    for k in ("intrinsic", "synergistic", "shared"):
        p = np.array([v for r in rows for v in r["pvalues"][k]])
        print(f"{k:12s} n={len(p)} P(p<0.05)={np.mean(p < 0.05):.3f} P(p<0.01)={np.mean(p < 0.01):.3f}", flush=True)


if __name__ == "__main__":
    main()
