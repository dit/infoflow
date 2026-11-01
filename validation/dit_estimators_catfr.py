"""
dit's entropy estimators on the catFR1 data (no ground truth).

``bias``: for each cohort subject and directed region pair, the layers with the
cohort's fixed variables (``examples/catfr/cohort.py``) on the real realizations and
on realizations whose source is shuffled across all samples. The shuffled source is
independent of the target and context with its marginal intact, so every true layer
is zero there and the estimate is the estimator's bias.

``selection``: edges selected on the pilot's correctly aligned HFA epochs (all trials,
pooled) with each selection estimator, on the real data and on surrogates whose
channels' trials are permuted independently (inter-channel coupling within a trial
destroyed; each channel's dynamics and the stimulus-locked mean kept).

    python validation/dit_estimators_catfr.py bias --cohort ~/data/catfr_cohort
    python validation/dit_estimators_catfr.py selection --pilot ~/data/catfr_pilot_v3
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "catfr"))

LAYER_ESTIMATORS = (
    "plugin",
    "miller_madow",
    "grassberger",
    "nsb",
    "zhang",
    "chao_wang_jost",
    "bayes_minimax",
    "jackknife",
)
SELECTION_ESTIMATORS = ("plugin", "miller_madow", "grassberger", "zhang", "chao_wang_jost", "bayes_minimax")
LAYERS = ("te", "intrinsic", "synergistic", "shared")


def bias(cohort_dir, out, estimators=LAYER_ESTIMATORS):
    import os

    os.environ["CATFR_OUT"] = str(cohort_dir)
    from cohort import COHORT, fixed_network, region_trials
    from common import OUT, ROIS

    from infoflow.compare import _union_edges
    from infoflow.data import realizations
    from infoflow.measures import edge_flows
    from infoflow.preprocess import EqualFrequency, preprocess

    summary = json.loads((COHORT / "summary.json").read_text())
    edges = _union_edges(fixed_network(summary["source_lag"]), fixed_network(summary["source_lag"]))
    rng = np.random.default_rng(0)
    rows = []
    for subject in summary["subjects"]:
        trials, _ = region_trials(OUT / f"sub-{subject['subject']}_ses-{subject['session']}_hfa.nc")
        data = preprocess(trials, names=list(ROIS), max_lag=15, discretizer=EqualFrequency(4), prng=0).data
        for (s, t), (x_vars, w_vars) in edges.items():
            r = realizations(data, t, list(dict.fromkeys(x_vars + w_vars)), max_lag=max(v[1] for v in x_vars + w_vars))
            x, w = r.columns(list(x_vars)), r.columns(list(w_vars))
            shuffled = x[rng.permutation(len(x))]
            for est in estimators:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    real = edge_flows(r.present, x, w, estimator=est, prng=0).as_dict()
                    null = edge_flows(r.present, shuffled, w, estimator=est, prng=0).as_dict()
                rows.append(
                    {
                        "subject": subject["subject"],
                        "edge": f"{ROIS[s]}->{ROIS[t]}",
                        "estimator": est,
                        "n": len(x),
                        "real": real,
                        "shuffled": null,
                    }
                )
        print(subject["subject"], flush=True)
    Path(out).write_text(json.dumps(rows))
    print(f"\nmean over {len(summary['subjects'])} subjects x 6 edges (bits): shuffled source (= bias) | real")
    for est in estimators:
        sel = [r for r in rows if r["estimator"] == est]
        b = {k: np.mean([r["shuffled"][k] for r in sel]) for k in LAYERS}
        v = {k: np.mean([r["real"][k] for r in sel]) for k in LAYERS}
        print(
            f"{est:15s} "
            + " ".join(f"{k[:5]} {b[k]:.4f}" for k in LAYERS)
            + " | "
            + " ".join(f"{k[:5]} {v[k]:.4f}" for k in LAYERS)
        )


def selection(pilot_dir, out, estimators, pattern):
    import xarray as xr

    from infoflow.auto import _skeleton_settings
    from infoflow.parallel import thread_map
    from infoflow.preprocess import preprocess
    from infoflow.selection import infer_skeleton

    rows = []
    for path in sorted(Path(pilot_dir).glob(pattern)):
        ds = xr.open_dataset(path).load()
        names = [f"{r}:{c}" for r, c in zip(ds.region.values, ds.channel.values, strict=True)]
        x = ds.signal.values.astype(float)
        rng = np.random.default_rng(0)
        surrogate = np.stack([x[rng.permutation(len(x)), :, p] for p in range(x.shape[2])], axis=2)
        for label, values in (("real", x), ("surrogate", surrogate)):
            pre = preprocess(
                list(values), names=names, max_lag=15, network_resolution="auto", map_fn=thread_map(4), prng=0
            )
            for est in estimators:
                settings = _skeleton_settings(("plugin",), False, 1, None, {"entropy_estimator": est})
                start = time.perf_counter()
                skeleton = infer_skeleton(pre.data, pre.embeddings, settings=settings, prng=0, map_fn=thread_map(4))
                found = [(p, t) for t, sk in skeleton.items() if sk.significant for p in sk.parents()]
                rows.append(
                    {
                        "file": path.name,
                        "data": label,
                        "estimator": est,
                        "edges": len(found),
                        "seconds": time.perf_counter() - start,
                    }
                )
                print(rows[-1], flush=True)
    Path(out).write_text(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("bias", "selection"))
    parser.add_argument("--cohort", default=str(Path.home() / "data" / "catfr_cohort"))
    parser.add_argument("--pilot", default=str(Path.home() / "data" / "catfr_pilot_v3"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--estimators", nargs="+", default=list(SELECTION_ESTIMATORS))
    parser.add_argument("--files", default="sub-*_hfa.nc")
    parser.add_argument("--layer-estimators", nargs="+", default=list(LAYER_ESTIMATORS))
    args = parser.parse_args()
    if args.part == "bias":
        bias(args.cohort, args.out or "catfr_bias.json", args.layer_estimators)
    else:
        selection(args.pilot, args.out or "catfr_selection.json", args.estimators, args.files)


if __name__ == "__main__":
    main()
