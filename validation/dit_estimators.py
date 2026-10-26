"""
Layer accuracy of dit's discrete entropy estimators on processes with exact flows.

The sofic generators of ``tests/_generators.py`` give the exact transfer entropy and
intrinsic, synergistic and shared flow of an edge (history 1). Each sample adds ``d``
independent ternary nuisance variables to the context, which leaves every true flow
unchanged while multiplying the number of contexts by ``3**d``, so the estimators'
bias shows up as the context grows. Both the single-joint estimate (``edge_flows``)
and the cross-fitted one ``infer_multiplex`` uses (``crossfit_flows``) are scored.

    python validation/dit_estimators.py --reps 20 --out layer_accuracy.nc
"""

import argparse
import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

ESTIMATORS = (
    "plugin", "miller_madow", "grassberger", "chao_shen", "nsb", "zhang", "chao_wang_jost", "shrink",
    "bayes_jeffreys", "bayes_laplace", "bayes_perks", "bayes_minimax", "ansb", "bonachela", "good_turing",
    "jackknife",
)  # fmt: skip
LAYERS = ("te", "intrinsic", "synergistic", "shared")
PROCESSES = {
    "copy": dict(a_copy=0.6),
    "xor": dict(a_xor=0.6),
    "shared": dict(a_hold=0.5, x_given_y=0.9),
    "mixed": dict(a_copy=0.3, a_xor=0.3, a_hold=0.2, x_given_y=0.8),
}


def truths():
    from _generators import exact_flows, mixture

    return {name: exact_flows(mixture(**kw)) for name, kw in PROCESSES.items()}


def one(args):
    from _generators import mixture, sample_pairs

    from infoflow.crossfit import crossfit_flows
    from infoflow.measures import edge_flows

    name, n, d, rep = args
    seed = zlib.crc32(f"{name}-{n}-{d}-{rep}".encode())
    pairs = sample_pairs(mixture(**PROCESSES[name]), n + 1, seed)
    x, y = pairs[:-1, 0], pairs[1:, 1]
    rng = np.random.default_rng(seed + 1)
    w = np.column_stack([pairs[:-1, 1], rng.integers(0, 3, size=(n, d))])
    out = np.full((2, len(ESTIMATORS), len(LAYERS)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for e, est in enumerate(ESTIMATORS):
            for m, fn in enumerate((edge_flows, crossfit_flows)):
                try:
                    flow = fn(y, x, w, estimator=est, prng=0)
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    continue
                values = flow.as_dict()
                out[m, e] = [values[k] for k in LAYERS]
    return args, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--n", type=int, nargs="+", default=[300, 1000, 3000])
    parser.add_argument("--d", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="layer_accuracy.nc")
    args = parser.parse_args()

    exact = truths()
    jobs = [(p, n, d, r) for p in PROCESSES for n in args.n for d in args.d for r in range(args.reps)]
    shape = (len(PROCESSES), len(args.n), len(args.d), args.reps, 2, len(ESTIMATORS), len(LAYERS))
    values = np.full(shape, np.nan)
    with ProcessPoolExecutor(args.workers) as pool:
        for (p, n, d, r), out in pool.map(one, jobs, chunksize=4):
            values[list(PROCESSES).index(p), args.n.index(n), args.d.index(d), r] = out
    truth = np.array([[exact[p][k] for k in LAYERS] for p in PROCESSES])
    ds = xr.Dataset(
        {
            "estimate": (("process", "n", "d", "rep", "path", "estimator", "layer"), values),
            "truth": (("process", "layer"), truth),
        },
        coords={
            "process": list(PROCESSES), "n": args.n, "d": args.d, "rep": range(args.reps),
            "path": ["single", "crossfit"], "estimator": list(ESTIMATORS), "layer": list(LAYERS),
        },
    )  # fmt: skip
    ds.to_netcdf(args.out)
    summarize(ds)


def summarize(ds):
    error = ds["estimate"] - ds["truth"]
    rmse = np.sqrt((error**2).mean("rep"))
    for path in ("single", "crossfit"):
        table = rmse.sel(path=path).mean(("process", "d", "layer"))
        print(f"\n{path}: RMSE (bits), mean over processes, context sizes and layers")
        print(f"{'estimator':16s}" + "".join(f"  n={n:<6d}" for n in ds.n.values))
        for est in ds.estimator.values:
            print(f"{est:16s}" + "".join(f"  {float(v):.4f}  " for v in table.sel(estimator=est).values))


if __name__ == "__main__":
    main()
