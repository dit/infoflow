"""
Shared readers for the catFR1 pilot (OpenNeuro ds004809, Herrema & Kahana).

The BIDS events carry no ``recalled`` flag: a WORD is recalled when a REC_WORD
in the same list names the same item. Regions come from the hippocampal-subfield
atlases first (``das.region``, ``stein.region``), then the whole-brain
(``wb.region``) and Desikan-Killiany (``ind.region``) labels.
"""

import csv
import json
import os
import re
from pathlib import Path

DATASET = "ds004809"
BUCKET = f"s3://openneuro.org/{DATASET}/"
ROOT = Path(os.environ.get("CATFR_ROOT", Path.home() / "data" / "openneuro" / DATASET))
OUT = Path(os.environ.get("CATFR_OUT", Path.home() / "data" / "catfr_pilot"))

ROIS = ("MTL", "LTC", "PFC")

_MTL = re.compile(
    r"hippocamp|amygdala|\bamy\b|parahippocampal|\bphg\b|\bphc\b|entorhinal|\berc\b|\bec\b|"
    r"\bprc\b|\bba3[56]\b|\bca[123]\b|\bdg\b|\bsub\b",
    re.I,
)
_LTC = re.compile(
    r"superiortemporal|middletemporal|inferiortemporal|bankssts|temporalpole|"
    r"temporal gyrus|\b[smi]tg\b|\btmp\b",
    re.I,
)
_PFC = re.compile(
    r"middlefrontal|superiorfrontal|pars(triangularis|opercularis|orbitalis)|orbitofrontal|"
    r"frontalpole|frontal gyrus|\b[sm]fg\b|\bifg\b|dlpfc|\bofc\b",
    re.I,
)
_SKIP = re.compile(r"white matter|\bwm\b|^n/a$|^$|misc|sulcus", re.I)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def roi(labels):
    """Map an ordered list of atlas labels (most specific first) to MTL, LTC, PFC or None."""
    for label in labels:
        label = (label or "").strip()
        if _SKIP.search(label):
            continue
        for name, pattern in (("MTL", _MTL), ("LTC", _LTC), ("PFC", _PFC)):
            if pattern.search(label):
                return name
        return None
    return None


def contact_regions(electrodes_tsv):
    """``contact -> (roi, label)`` from an electrodes table."""
    out = {}
    for row in read_tsv(electrodes_tsv):
        labels = [row.get(k) for k in ("das.region", "stein.region", "wb.region", "ind.region")]
        named = next((l for l in labels if l and not _SKIP.search(l)), "n/a")
        out[row["name"]] = (roi(labels), named)
    return out


def pair_regions(channels_tsv, electrodes_tsv):
    """``bipolar pair -> (roi, label)`` for pairs whose contacts agree on one ROI."""
    contacts = contact_regions(electrodes_tsv)
    out = {}
    for row in read_tsv(channels_tsv):
        a, _, b = row["name"].partition("-")
        ra, la = contacts.get(a, (None, "n/a"))
        rb, _ = contacts.get(b, (None, "n/a"))
        if row.get("status", "good") == "bad":
            continue
        if ra is not None and ra == rb:
            out[row["name"]] = (ra, la)
    return out


def sampling_rate(files):
    return float(json.loads(files["sidecar"].read_text())["SamplingFrequency"])


def word_events(events_tsv, sfreq):
    """
    Non-practice WORD events as dicts with ``onset``, ``list``, ``item``, ``category``,
    ``serialpos``, ``recalled``.

    ``onset`` is ``sample / sfreq`` seconds from the start of the recording. The BIDS
    ``onset`` column of this dataset is not: it is offset from the recording by 1 s to
    15 min (drifting within a session), and only ``sample``-aligned epochs show the
    word-locked evoked response.
    """
    rows = read_tsv(events_tsv)
    recalled = {(r["list"], r["item_name"].upper()) for r in rows if r["trial_type"] == "REC_WORD"}
    words = []
    for r in rows:
        if r["trial_type"] != "WORD" or int(r["list"]) < 1 or int(r["sample"]) < 0:
            continue
        words.append(
            {
                "onset": int(r["sample"]) / sfreq,
                "list": int(r["list"]),
                "item": r["item_name"],
                "category": r["category"],
                "serialpos": int(r["serialpos"]),
                "recalled": (r["list"], r["item_name"].upper()) in recalled,
            }
        )
    return words


def session_files(subject, session, root=ROOT):
    base = root / f"sub-{subject}" / f"ses-{session}" / "ieeg"
    stem = f"sub-{subject}_ses-{session}_task-catFR1"
    electrodes = sorted(base.glob(f"{stem}_space-*_electrodes.tsv"))
    return {
        "events": base / f"{stem}_events.tsv",
        "channels": base / f"{stem}_acq-bipolar_channels.tsv",
        "electrodes": electrodes[0] if electrodes else None,
        "edf": base / f"{stem}_acq-bipolar_ieeg.edf",
        "sidecar": base / f"{stem}_acq-bipolar_ieeg.json",
        "key": f"sub-{subject}/ses-{session}/ieeg/{stem}_acq-bipolar_ieeg.edf",
    }
