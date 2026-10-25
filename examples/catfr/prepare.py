"""
Turn a catFR1 bipolar session into encoding epochs for infoflow.

For each chosen session (``sessions.json`` from ``fetch.py``) this picks up to
``--per-roi`` bipolar pairs in each of MTL, LTC and PFC (spread over atlas
labels), notch-filters line noise, and writes two signals of WORD epochs
(0 to 1.6 s after onset):

* ``broadband``: 1-80 Hz, resampled to 200 Hz;
* ``hfa``: log mean Morlet power over 8 log-spaced frequencies in 70-150 Hz,
  computed on epochs padded by 1 s each side and then cropped, averaged down to
  50 Hz.

Every non-practice word is kept with its ``recalled`` label; ``run.py`` balances.

    uv run --with mne python examples/catfr/prepare.py
"""

import argparse
import json
from collections import defaultdict
from itertools import zip_longest

import mne
import numpy as np
import xarray as xr
from common import OUT, ROIS, pair_regions, session_files, word_events

EPOCH = 1.6
PAD = 1.0
BROADBAND_SFREQ = 200.0
HFA_SFREQ = 50.0
HFA_FREQS = np.geomspace(70, 150, 8)


def pick_pairs(regions, per_roi):
    """
    Round-robin over atlas labels within each ROI so one electrode shaft doesn't
    dominate. Pairs sharing a contact share a signal term, so at most one is kept.
    """
    picks, used = [], set()
    for roi in ROIS:
        by_label = defaultdict(list)
        for name, (r, label) in sorted(regions.items()):
            if r == roi:
                by_label[label].append(name)
        ordered = [n for group in zip_longest(*by_label.values()) for n in group if n is not None]
        chosen = []
        for name in ordered:
            contacts = set(name.split("-"))
            if len(chosen) < per_roi and not contacts & used:
                chosen.append(name)
                used |= contacts
        picks += chosen
    return picks


def epochs_of(raw, onsets, tmin, tmax):
    sfreq = raw.info["sfreq"]
    starts = np.round((onsets + tmin) * sfreq).astype(int)
    n = int(round((tmax - tmin) * sfreq))
    data = raw.get_data()
    keep = (starts >= 0) & (starts + n <= data.shape[1])
    return np.stack([data[:, s : s + n] for s in starts[keep]]), keep


def to_dataset(values, sfreq, words, keep, picks, regions, attrs):
    kept = [w for w, k in zip(words, keep) if k]
    return xr.Dataset(
        {"signal": (("trial", "time", "channel"), values.transpose(0, 2, 1).astype(np.float32))},
        coords={
            "time": np.arange(values.shape[2]) / sfreq,
            "channel": picks,
            "region": ("channel", [regions[p][0] for p in picks]),
            "label": ("channel", [regions[p][1] for p in picks]),
            "recalled": ("trial", [w["recalled"] for w in kept]),
            "list": ("trial", [w["list"] for w in kept]),
            "serialpos": ("trial", [w["serialpos"] for w in kept]),
            "item": ("trial", [w["item"] for w in kept]),
        },
        attrs={**attrs, "sfreq": sfreq},
    )


def prepare(subject, session, per_roi):
    files = session_files(subject, session)
    regions = pair_regions(files["channels"], files["electrodes"])
    raw = mne.io.read_raw_edf(files["edf"], preload=False, verbose="error")
    words = word_events(files["events"], raw.info["sfreq"])
    onsets = np.array([w["onset"] for w in words])

    regions = {k: v for k, v in regions.items() if k in raw.ch_names}
    picks = pick_pairs(regions, per_roi)
    raw.pick(picks).load_data(verbose="error")
    sfreq = raw.info["sfreq"]
    raw.notch_filter(np.arange(60, sfreq / 2 - 1, 60), verbose="error")
    attrs = {"subject": subject, "session": session, "source_sfreq": sfreq}

    bb = raw.copy().filter(1, 80, verbose="error").resample(BROADBAND_SFREQ, verbose="error")
    values, keep = epochs_of(bb, onsets, 0.0, EPOCH)
    broadband = to_dataset(values, BROADBAND_SFREQ, words, keep, picks, regions, {**attrs, "signal": "broadband"})

    hf = raw.copy().resample(500.0, verbose="error")
    padded, keep = epochs_of(hf, onsets, -PAD, EPOCH + PAD)
    power = mne.time_frequency.tfr_array_morlet(padded, 500.0, HFA_FREQS, n_cycles=7, output="power", verbose="error")
    lo = int(PAD * 500)
    env = np.log10(power.mean(axis=2))[:, :, lo : lo + int(EPOCH * 500)]
    step = int(500 / HFA_SFREQ)
    env = env.reshape(*env.shape[:2], -1, step).mean(axis=-1)
    hfa = to_dataset(env, HFA_SFREQ, words, keep, picks, regions, {**attrs, "signal": "hfa"})

    stem = OUT / f"sub-{subject}_ses-{session}"
    broadband.to_netcdf(f"{stem}_broadband.nc")
    hfa.to_netcdf(f"{stem}_hfa.nc")
    return broadband, hfa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-roi", type=int, default=6)
    args = parser.parse_args()
    for s in json.loads((OUT / "sessions.json").read_text())["chosen"]:
        bb, hfa = prepare(s["subject"], s["session"], args.per_roi)
        rec = bb.recalled.values
        print(
            f"{s['subject']} ses-{s['session']}: {bb.sizes['channel']} pairs "
            f"{dict(zip(*np.unique(bb.region.values, return_counts=True)))}, "
            f"{rec.sum()} recalled / {(~rec).sum()} forgotten, broadband {bb.signal.shape}, hfa {hfa.signal.shape}"
        )


if __name__ == "__main__":
    main()
