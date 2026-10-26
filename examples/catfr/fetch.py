"""
Rank catFR1 sessions for the pilot and download their bipolar EDFs.

1. Sync only the small sidecars (events, bipolar channels, electrodes) from the
   public OpenNeuro bucket.
2. Score each session by bipolar-pair coverage of MTL, lateral temporal and
   prefrontal cortex, and by the number of recalled words.
3. Download the bipolar EDF of the best session of the top subjects.

    uv run --with mne python examples/catfr/fetch.py --top 3
"""

import argparse
import json
import subprocess
from collections import Counter

from common import BUCKET, OUT, ROIS, ROOT, pair_regions, sampling_rate, session_files, word_events


def sync_sidecars():
    includes = [
        "participants.tsv",
        "*_events.tsv",
        "*acq-bipolar_channels.tsv",
        "*_electrodes.tsv",
        "*acq-bipolar_ieeg.json",
    ]
    cmd = ["aws", "s3", "sync", "--no-sign-request", "--only-show-errors", BUCKET, str(ROOT), "--exclude", "*"]
    for pattern in includes:
        cmd += ["--include", pattern]
    subprocess.run(cmd, check=True)


def score_sessions(min_recalled=60, min_pairs=2):
    rows = []
    for events in sorted(ROOT.glob("sub-*/ses-*/ieeg/*_task-catFR1_events.tsv")):
        subject = events.parts[-4][4:]
        session = events.parts[-3][4:]
        files = session_files(subject, session)
        if files["electrodes"] is None or not files["channels"].exists():
            continue
        words = word_events(events, sampling_rate(files))
        n_recalled = sum(w["recalled"] for w in words)
        counts = Counter(r for r, _ in pair_regions(files["channels"], files["electrodes"]).values())
        rows.append(
            {
                "subject": subject,
                "session": session,
                "n_words": len(words),
                "n_recalled": n_recalled,
                **{roi: counts.get(roi, 0) for roi in ROIS},
            }
        )
    eligible = [r for r in rows if r["n_recalled"] >= min_recalled and all(r[k] >= min_pairs for k in ROIS)]
    eligible.sort(key=lambda r: (min(min(r[k] for k in ROIS), 6), r["n_recalled"]), reverse=True)
    return rows, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_sync:
        sync_sidecars()
    rows, eligible = score_sessions()
    chosen, seen = [], set()
    for r in eligible:
        if r["subject"] not in seen:
            chosen.append(r)
            seen.add(r["subject"])
        if len(chosen) == args.top:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sessions.json").write_text(
        json.dumps({"n_sessions": len(rows), "n_eligible": len(eligible), "chosen": chosen}, indent=2)
    )
    print(f"{len(rows)} sessions, {len(eligible)} eligible")
    for r in chosen:
        print(r)

    if args.no_download:
        return
    for r in chosen:
        files = session_files(r["subject"], r["session"])
        if not files["edf"].exists():
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--no-sign-request",
                    "--only-show-errors",
                    BUCKET + files["key"],
                    str(files["edf"]),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
