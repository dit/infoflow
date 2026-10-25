"""
Per-edge p-values of same-condition null splits, to check the calibration of
``compare_networks`` under pooled selection.

Each split draws two disjoint halves of the larger class exactly as ``run.py`` does
(seeds ``1000 + r``), selects one network on both halves together, and records every
tested edge's raw p-value per layer along with the FDR decisions. Under the null the
raw p-values should be (super-)uniform; a per-split excess with uniform p-values points
at the multiple-testing step instead.

    uv run --project ../.. --with netcdf4 python null_check.py --subject R1332M --start 0 --count 25
"""

import argparse
import json

import numpy as np
import xarray as xr
from common import OUT
from run import DEFAULT_MAX_LAG, contrast

from infoflow.layers import LAYERS
from infoflow.parallel import thread_map
from infoflow.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--signal", default="hfa")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    path = next(OUT.glob(f"sub-{args.subject}_*_{args.signal}.nc"))
    ds = xr.open_dataset(path).load()
    regions = [str(r) for r in ds.region.values]
    names = [f"{r}:{c}" for r, c in zip(regions, ds.channel.values, strict=True)]
    trials = [ds.signal.values[i].astype(float) for i in range(ds.sizes["trial"])]
    recalled = ds.recalled.values.astype(bool)
    rec, forg = np.flatnonzero(recalled), np.flatnonzero(~recalled)
    major = rec if len(rec) >= len(forg) else forg
    m = min(len(rec), len(forg), len(major) // 2)
    pre = preprocess(
        trials,
        names=names,
        max_lag=DEFAULT_MAX_LAG[args.signal],
        network_resolution="auto",
        map_fn=thread_map(args.threads),
        prng=0,
    )

    out = OUT / f"nullcheck_{path.stem}_{args.start:03d}.jsonl"
    with out.open("w") as f:
        for r in range(args.start, args.start + args.count):
            draw = np.random.default_rng(1000 + r).permutation(major)
            _, _, cmp = contrast(
                pre, np.sort(draw[:m]), np.sort(draw[m : 2 * m]), names, args.n_perm, args.threads, 1000 + r, "pooled"
            )
            record = {"split": r, "pvalues": {}, "significant": {}}
            for k in LAYERS:
                p = cmp.dataset["pvalue"].sel(layer=k).values
                s, t = np.nonzero(np.isfinite(p))
                record["pvalues"][k] = [[names[i], names[j], float(p[i, j])] for i, j in zip(s, t, strict=True)]
                record["significant"][k] = int(cmp.dataset["significant"].sel(layer=k).sum())
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"split {r}: {record['significant']}", flush=True)


if __name__ == "__main__":
    main()
