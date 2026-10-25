"""
Cohort follow-up: region-level subsequent-memory contrast across subjects.

1. Choose one session per subject with at least two non-overlapping bipolar pairs in
   each of MTL, LTC and PFC and at least ``--min-trials`` words in the smaller of the
   recalled and forgotten classes, ranked by that count.
2. For each session, download the bipolar EDF, write epochs with ``prepare.py``'s
   ``prepare`` (up to ``--per-roi`` pairs per region), and delete the EDF again unless
   it was already on disk.
3. Average each region's HFA (each pair z-scored over the session's epochs first) into
   one node, so every subject has the same three nodes.
4. Discretize every subject with the same 4 equal-frequency bins and test, across
   subjects, recalled minus forgotten flow on all six directed region pairs with
   ``compare_networks(design="between", dependent=True)``. The tested variables are
   fixed in advance (source at ``--source-lag``, target at lags 1-2, the third region
   at ``--source-lag``), so nothing is selected on the data.
5. Within each subject, the pilot's analysis on the three nodes: one network selected
   on both conditions pooled, and a within-subject comparison on its edges.

    uv run --project ../.. --with mne --with netcdf4 python cohort.py --subjects 25
"""

import argparse
import json
import subprocess
from collections import Counter
from types import SimpleNamespace

import numpy as np
import xarray as xr
from common import BUCKET, OUT, ROIS, ROOT, pair_regions, sampling_rate, session_files, word_events
from prepare import pick_pairs, prepare
from run import condition_network, layer_counts, significant_edges, tested

from infoflow.compare import compare_networks
from infoflow.layers import LAYERS
from infoflow.parallel import thread_map
from infoflow.preprocess import EqualFrequency, preprocess
from infoflow.selection import TargetSkeleton

COHORT = OUT / "cohort"


def choose(min_trials, per_roi):
    rows = []
    for events in sorted(ROOT.glob("sub-*/ses-*/ieeg/*_task-catFR1_events.tsv")):
        subject, session = events.parts[-4][4:], events.parts[-3][4:]
        files = session_files(subject, session)
        if files["electrodes"] is None or not files["channels"].exists():
            continue
        words = word_events(events, sampling_rate(files))
        n_rec = sum(w["recalled"] for w in words)
        regions = pair_regions(files["channels"], files["electrodes"])
        counts = Counter(regions[p][0] for p in pick_pairs(regions, per_roi))
        balanced = min(n_rec, len(words) - n_rec)
        if balanced >= min_trials and all(counts[r] >= 2 for r in ROIS):
            rows.append({"subject": subject, "session": session, "balanced": balanced, **{r: counts[r] for r in ROIS}})
    best = {}
    for r in sorted(rows, key=lambda r: -r["balanced"]):
        best.setdefault(r["subject"], r)
    return sorted(best.values(), key=lambda r: -r["balanced"])


def fetch_and_prepare(row, per_roi):
    stem = f"sub-{row['subject']}_ses-{row['session']}"
    if (OUT / f"{stem}_hfa.nc").exists():
        return
    files = session_files(row["subject"], row["session"])
    downloaded = not files["edf"].exists()
    if downloaded:
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", BUCKET + files["key"], str(files["edf"])],
            check=True,
        )
    try:
        prepare(row["subject"], row["session"], per_roi)
    finally:
        if downloaded:
            files["edf"].unlink(missing_ok=True)


def region_trials(path):
    """Trials of shape (time, 3): each region's z-scored HFA averaged over its pairs."""
    ds = xr.open_dataset(path).load()
    x = ds.signal.values
    x = (x - x.mean(axis=(0, 1))) / x.std(axis=(0, 1))
    region = ds.region.values
    nodes = np.stack([x[:, :, region == r].mean(axis=2) for r in ROIS], axis=2)
    return [nodes[i] for i in range(len(nodes))], ds.recalled.values.astype(bool)


def fixed_network(source_lag):
    skeleton = {}
    for t in range(len(ROIS)):
        sources = [(s, source_lag) for s in range(len(ROIS)) if s != t]
        skeleton[t] = TargetSkeleton(target=t, target_past=[(t, 1), (t, 2)], sources=sources, significant=True)
    return SimpleNamespace(names=list(ROIS), skeleton=skeleton)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=25)
    parser.add_argument("--min-trials", type=int, default=40)
    parser.add_argument("--per-roi", type=int, default=10)
    parser.add_argument("--source-lag", type=int, default=2)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    COHORT.mkdir(parents=True, exist_ok=True)

    rows = choose(args.min_trials, args.per_roi)
    print(f"{len(rows)} eligible subjects", flush=True)
    data_rec, data_forg, per_subject, skipped = [], [], [], {}
    for row in rows:
        if len(per_subject) == args.subjects:
            break
        stem = f"sub-{row['subject']}_ses-{row['session']}"
        try:
            fetch_and_prepare(row, args.per_roi)
        except ValueError as error:
            # e.g. no WORD onset falls inside the recording
            skipped[stem] = str(error)
            print(f"{stem}: skipped ({error})", flush=True)
            continue
        trials, recalled = region_trials(OUT / f"{stem}_hfa.nc")
        rng = np.random.default_rng(0)
        rec, forg = np.flatnonzero(recalled), np.flatnonzero(~recalled)
        n = min(len(rec), len(forg))
        idx_rec = np.sort(rng.choice(rec, n, replace=False))
        idx_forg = np.sort(rng.choice(forg, n, replace=False))
        pre = preprocess(trials, names=list(ROIS), max_lag=15, discretizer=EqualFrequency(4), prng=0)
        data_rec.append(pre.data.subset_trials(list(idx_rec)))
        data_forg.append(pre.data.subset_trials(list(idx_forg)))

        net = condition_network(pre, np.concatenate([idx_rec, idx_forg]), list(ROIS), args.threads)
        within = compare_networks(
            net,
            net,
            data_rec[-1],
            data_forg[-1],
            design="within",
            n_perm=500,
            curtail=10,
            map_fn=thread_map(args.threads),
            prng=0,
        )
        per_subject.append(
            {
                **row,
                "n_per_condition": int(n),
                "edges": layer_counts(net),
                "tested": tested(within),
                "sme_edges": significant_edges(within, list(ROIS)),
            }
        )
        print(
            f"{stem}: {per_subject[-1]['n_per_condition']} trials/condition, edges {per_subject[-1]['edges']}, "
            f"SME {per_subject[-1]['sme_edges']}",
            flush=True,
        )

    fixed = fixed_network(args.source_lag)
    between = compare_networks(
        fixed, fixed, data_rec, data_forg, design="between", dependent=True, n_perm=args.n_perm, prng=0
    )
    between.dataset.to_netcdf(COHORT / "between.nc")
    from infoflow.compare import _edge_values

    per_unit = {}
    for (s, t), (x_vars, w_vars) in between.edges.items():
        per_unit[f"{ROIS[s]}->{ROIS[t]}"] = {
            "recalled": [[_edge_values(d, t, x_vars, w_vars)[k] for k in LAYERS] for d in data_rec],
            "forgotten": [[_edge_values(d, t, x_vars, w_vars)[k] for k in LAYERS] for d in data_forg],
        }
    summary = {
        "subjects": per_subject,
        "skipped": skipped,
        "source_lag": args.source_lag,
        "n_perm": args.n_perm,
        "between": {
            f"{ROIS[s]}->{ROIS[t]}": {
                k: {
                    v: float(between.dataset[v].sel(layer=k).values[s, t])
                    for v in ("a", "b", "difference", "pvalue", "qvalue")
                }
                for k in LAYERS
            }
            for s, t in between.edges
        },
        "per_unit": per_unit,
    }
    (COHORT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["between"], indent=1), flush=True)


if __name__ == "__main__":
    main()
