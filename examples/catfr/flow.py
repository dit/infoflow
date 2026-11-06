"""
Pre-specified directed flow between MTL and PFC, across the catFR1 cohort.

Hypothesis (fixed before looking at the data): information flows between the medial
temporal lobe and prefrontal cortex in either direction, and that flow differs with
memory. For every eligible subject (``cohort.choose``), each session is downloaded,
epoched by ``prepare.prepare_phases`` and deleted again, and the regions' HFA is
averaged into three nodes (``cohort.region_trials``) and cut into 4 equal-frequency bins.

Tested edges: MTL -> PFC and PFC -> MTL with the source at lag 1, 2, 5 or 10 samples
(20, 40, 100, 200 ms), conditioned on the target's past (lags 1-2) and on LTC at the
source lag. Layers are estimated with Chao--Wang--Jost. Two questions per phase:

* *existence*: real data against the same data with the source's trials shuffled
  (within-trial coupling broken, each region's stimulus-locked time course kept);
* *contrast*: ``encoding``, recalled vs forgotten words; ``retrieval``, the 1.6 s before
  correct recalls vs deliberation windows.

Both are paired across subjects (``compare_networks(design="between", dependent=True)``),
with Benjamini--Hochberg over the eight direction x lag tests of each layer. The
primary layer is intrinsic flow. Subjects need ``--min-epochs`` epochs per condition.

    uv run --project ../.. --with mne --with netcdf4 --with "decorator<5.3" python flow.py prepare
    uv run --project ../.. --with netcdf4 python flow.py analyze
"""

import argparse
import json
import subprocess
from types import SimpleNamespace

import numpy as np
from common import BUCKET, OUT, ROIS, session_files

from infoflow.compare import _edge_values, compare_networks
from infoflow.data import DiscreteData
from infoflow.layers import LAYERS
from infoflow.preprocess import EqualFrequency, preprocess
from infoflow.selection import TargetSkeleton
from infoflow.stats import benjamini_hochberg

FLOW = OUT / "flow"
LAGS = (1, 2, 5, 10)
DIRECTIONS = ((0, 2), (2, 0))  # (source, target) indices into ROIS: MTL -> PFC, PFC -> MTL
ESTIMATOR = "chao_wang_jost"
PHASES = {"encoding": ("recalled", "forgotten"), "retrieval": ("pre-recall", "deliberation")}


def prepare(n_subjects, per_roi):
    from cohort import choose
    from prepare import prepare_phases

    FLOW.mkdir(parents=True, exist_ok=True)
    rows = choose(40, per_roi)[:n_subjects]
    for row in rows:
        stem = f"sub-{row['subject']}_ses-{row['session']}"
        if (FLOW / f"{stem}_retrieval.nc").exists():
            continue
        files = session_files(row["subject"], row["session"])
        downloaded = not files["edf"].exists()
        try:
            if downloaded:
                subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", BUCKET + files["key"],
                                str(files["edf"])], check=True)  # fmt: skip
            e, r = prepare_phases(row["subject"], row["session"], per_roi, FLOW)
            print(f"{stem}: encoding {int(e.recalled.sum())}/{int((~e.recalled).sum())}, "
                  f"retrieval {int(r.recalled.sum())}/{int((~r.recalled).sum())}", flush=True)  # fmt: skip
        except (ValueError, subprocess.CalledProcessError) as error:
            print(f"{stem}: skipped ({error})", flush=True)
        finally:
            if downloaded:
                files["edf"].unlink(missing_ok=True)


def edge_network(source, target, lag):
    skeleton = {
        target: TargetSkeleton(
            target=target, target_past=[(target, 1), (target, 2)], sources=[(source, lag)],
            conditionals=[(1, lag)], significant=True,
        )
    }  # fmt: skip
    return SimpleNamespace(names=list(ROIS), skeleton=skeleton)


def shuffle_source(data, source, rng):
    """`data` with the source process's trials permuted (other processes untouched)."""
    order = rng.permutation(data.n_trials)

    def swap(arrays):
        out = [a.copy() for a in arrays]
        for i, j in enumerate(order):
            out[i][source] = arrays[j][source]
        return out

    return DiscreteData(
        past=swap(data.past), present=swap(data.present), past_offset=data.past_offset,
        present_offset=data.present_offset, past_alphabet=data.past_alphabet,
        present_alphabet=data.present_alphabet, lag_step=data.lag_step, names=data.names,
        discretizers=data.discretizers,
    )  # fmt: skip


def retrieval_mask(path):
    """
    The retrieval epochs that :func:`common.retrieval_windows` keeps.

    Epochs written before their start times were stored came from overlapping
    deliberation windows (every 0.8 s, 1 s from vocalizations); their starts are
    rebuilt from the events and only the current, non-overlapping windows are kept.
    """
    import xarray as xr
    from common import retrieval_windows, sampling_rate

    ds = xr.open_dataset(path)
    files = session_files(ds.attrs["subject"], ds.attrs["session"])
    recall, deliberation = retrieval_windows(files["events"], sampling_rate(files))
    if "start" in ds.coords:
        starts = ds.start.values
    else:
        old = np.concatenate(retrieval_windows(files["events"], sampling_rate(files), step=0.8, after=1.0))
        if len(old) != ds.sizes["trial"]:
            raise ValueError(f"{path.name}: cannot match {ds.sizes['trial']} epochs to {len(old)} windows")
        starts = old
    keep = np.isin(np.round(starts, 6), np.round(np.concatenate([recall, deliberation]), 6))
    return keep


def subject_data(path, min_epochs, demean=False):
    from cohort import region_trials

    trials, label = region_trials(path)
    if path.name.endswith("_retrieval.nc"):
        keep = retrieval_mask(path)
        trials, label = [t for t, k in zip(trials, keep, strict=True) if k], label[keep]
    a, b = np.flatnonzero(label), np.flatnonzero(~label)
    n = min(len(a), len(b))
    if n < min_epochs:
        return None
    rng = np.random.default_rng(0)
    a, b = np.sort(rng.choice(a, n, replace=False)), np.sort(rng.choice(b, n, replace=False))
    amplitude = [
        float(np.stack(trials)[a][:, :, i].mean() - np.stack(trials)[b][:, :, i].mean()) for i in range(len(ROIS))
    ]
    if demean:
        # Remove each condition's mean time course per region: what is left cannot come
        # from a shared stimulus- or response-locked trajectory.
        x = np.stack(trials)
        for idx in (a, b):
            x[idx] -= x[idx].mean(axis=0)
        trials = list(x)
    data = preprocess(trials, names=list(ROIS), max_lag=12, discretizer=EqualFrequency(4), prng=0).data
    both = data.subset_trials(list(np.concatenate([a, b])))
    return {"a": data.subset_trials(list(a)), "b": data.subset_trials(list(b)), "both": both, "n": int(n),
            "amplitude": amplitude}  # fmt: skip


def analyze(min_epochs, n_perm, demean=False):
    results = {}
    for phase, (cond_a, cond_b) in PHASES.items():
        subjects = {}
        for path in sorted(FLOW.glob(f"sub-*_{phase}.nc")):
            d = subject_data(path, min_epochs, demean)
            if d is not None:
                subjects[path.name.split(f"_{phase}")[0]] = d
        names = sorted(subjects)
        units = [subjects[s] for s in names]
        amplitude = np.array([u["amplitude"] for u in units])
        out = {"subjects": names, "n_per_condition": [u["n"] for u in units],
               "amplitude": {r: {"mean": float(amplitude[:, i].mean()), "positive": int((amplitude[:, i] > 0).sum())}
                             for i, r in enumerate(ROIS)}, "tests": []}  # fmt: skip
        rng = np.random.default_rng(1)
        for s, t in DIRECTIONS:
            for lag in LAGS:
                net = edge_network(s, t, lag)
                x, w = [(s, lag)], [(1, lag), (t, 1), (t, 2)]
                real = [u["both"] for u in units]
                surrogate = [shuffle_source(u["both"], s, rng) for u in units]
                for question, (da, db) in {
                    "existence": (real, surrogate),
                    "contrast": ([u["a"] for u in units], [u["b"] for u in units]),
                }.items():
                    cmp = compare_networks(net, net, da, db, design="between", dependent=True, n_perm=n_perm,
                                           estimator=ESTIMATOR, prng=0)  # fmt: skip
                    values = np.array([[[_edge_values(d, t, x, w, estimator=ESTIMATOR)[k] for k in LAYERS]
                                        for d in pair] for pair in (da, db)])  # fmt: skip
                    out["tests"].append({
                        "question": question, "edge": f"{ROIS[s]}->{ROIS[t]}", "lag": lag,
                        **{k: {"a": float(cmp.dataset["a"].sel(layer=k).values[s, t]),
                               "b": float(cmp.dataset["b"].sel(layer=k).values[s, t]),
                               "pvalue": float(cmp.dataset["pvalue"].sel(layer=k).values[s, t]),
                               "positive": int((values[0, :, i] > values[1, :, i]).sum())}
                           for i, k in enumerate(LAYERS)},
                    })  # fmt: skip
                    print(phase, question, ROIS[s], "->", ROIS[t], lag,
                          {k: round(out["tests"][-1][k]["pvalue"], 3) for k in LAYERS}, flush=True)  # fmt: skip
        for question in ("existence", "contrast"):
            tests = [x for x in out["tests"] if x["question"] == question]
            for k in LAYERS:
                _, q = benjamini_hochberg([x[k]["pvalue"] for x in tests], 0.05)
                for x, qv in zip(tests, q, strict=True):
                    x[k]["qvalue"] = float(qv)
        out["conditions"] = [cond_a, cond_b]
        results[phase] = out
    (FLOW / f"results{'_demeaned' if demean else ''}.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "analyze"))
    parser.add_argument("--subjects", type=int, default=200)
    parser.add_argument("--per-roi", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--demean", action="store_true", help="remove each condition's mean time course first")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.subjects, args.per_roi)
    else:
        analyze(args.min_epochs, args.n_perm, args.demean)


if __name__ == "__main__":
    main()
