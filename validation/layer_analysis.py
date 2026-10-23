"""
Summarize the records of ``layer_report.py``: parent selection, the layer breakdown of
selected edges, and two ways of reporting layers.

* Edge-gated: an edge is reported when its transfer entropy passed parent selection
  (or its time-delayed mutual information passed the TDMI screen, for shared-only
  candidates), and its layers are given as point estimates without further tests.
  Its dominant layer is the larger of intrinsic and synergistic flow (parents) or
  shared flow (shared-only candidates).
* Per-layer: a layer is claimed only where its own permutation test survives the FDR
  correction within that layer.

    python validation/layer_analysis.py .bench_data/results/layers
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(directory):
    records = []
    for path in sorted(Path(directory).glob("*.p")):
        with open(path, "rb") as fh:
            records.append(pickle.load(fh))
    return records


def group_of(label):
    parts = label.split("/")
    return "/".join(parts[:2]) if parts[0] in ("motif", "synthetic", "netsim", "rivers") else parts[0]


def layer_index(rec, name):
    return rec["layers"].index(name)


def edge_layers(rec, s, t):
    w = {
        name: float(rec["weight"][layer_index(rec, name), s, t])
        for name in ("te", "tdmi", "intrinsic", "synergistic", "shared")
    }
    sig = {
        name: bool(rec["significant"][layer_index(rec, name), s, t]) for name in ("intrinsic", "synergistic", "shared")
    }
    return w, sig


def gated_label(rec, s, t):
    w, _ = edge_layers(rec, s, t)
    if (s, t) in set(map(tuple, rec["parents"])):
        return "intrinsic" if w["intrinsic"] >= w["synergistic"] else "synergistic"
    if (s, t) in set(map(tuple, rec["shared_candidates"])):
        return "shared"
    return None


def tested_labels(rec, s, t):
    _, sig = edge_layers(rec, s, t)
    return {name for name, ok in sig.items() if ok}


def summarize(records):
    groups = defaultdict(list)
    for rec in records:
        groups[group_of(rec["label"])].append(rec)
    rows = []
    for name, recs in sorted(groups.items()):
        tp = fp = fn = 0
        fractions = defaultdict(list)
        claims = defaultdict(int)
        n_edges = 0
        shared_only = 0
        shared_tested = 0
        shared_real = 0
        agree = 0
        seconds = []
        for rec in recs:
            truth = set(map(tuple, rec["truth"]))
            parents = set(map(tuple, rec["parents"]))
            tp += len(parents & truth)
            fp += len(parents - truth)
            fn += len(truth - parents)
            shared_only += len(rec["shared_candidates"])
            for s, t in map(tuple, rec["shared_candidates"]):
                shared_tested += edge_layers(rec, s, t)[1]["shared"]
                shared_real += (s, t) in truth
            seconds.append(rec["seconds"])
            for s, t in parents & truth:
                w, sig = edge_layers(rec, s, t)
                n_edges += 1
                total = w["intrinsic"] + w["synergistic"] + w["shared"]
                if total > 0:
                    fractions["intrinsic/total"].append(w["intrinsic"] / total)
                    fractions["synergistic/total"].append(w["synergistic"] / total)
                    fractions["shared/total"].append(w["shared"] / total)
                fractions["te"].append(w["te"])
                for layer, ok in sig.items():
                    claims[layer] += ok
                claims["none"] += not (sig["intrinsic"] or sig["synergistic"])
                claims["gated_synergistic"] += w["synergistic"] > w["intrinsic"]
                agree += sig["synergistic" if w["synergistic"] > w["intrinsic"] else "intrinsic"]
        rows.append(
            {
                "group": name,
                "networks": len(recs),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "true_edges_selected": n_edges,
                "te_bits": float(np.median(fractions["te"])) if fractions["te"] else np.nan,
                **{
                    f"{layer}_share": float(np.mean(fractions[f"{layer}/total"]))
                    if fractions[f"{layer}/total"]
                    else np.nan
                    for layer in ("intrinsic", "synergistic", "shared")
                },
                "gated_synergy_dominant": claims["gated_synergistic"] / max(n_edges, 1),
                "tested_intrinsic": claims["intrinsic"] / max(n_edges, 1),
                "tested_synergistic": claims["synergistic"] / max(n_edges, 1),
                "tested_shared": claims["shared"] / max(n_edges, 1),
                "tested_none": claims["none"] / max(n_edges, 1),
                "shared_only_edges": shared_only / len(recs),
                "shared_only_tested": shared_tested / len(recs),
                "shared_only_real": shared_real / max(shared_only, 1),
                "agreement": agree / max(n_edges, 1),
                "seconds": float(np.mean(seconds)),
            }
        )
    return rows


def motif_accuracy(records):
    """
    Per motif, how often each approach labels the edges of known composition correctly.

    Edge-gated is correct when the edge is reported and its dominant layer is the
    expected one. Per-layer is correct when the expected layer is significant and the
    competing layer (the other of intrinsic and synergistic for a parent; intrinsic
    for a shared-only edge) is not.
    """
    out = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if not rec["label"].startswith("motif/"):
            continue
        motif = rec["label"].split("/")[1]
        for (s, t), expected in rec["expected"].items():
            label = gated_label(rec, s, t)
            out[motif]["gated"].append(label == expected)
            tested = tested_labels(rec, s, t)
            competing = {"intrinsic": "synergistic", "synergistic": "intrinsic", "shared": "intrinsic"}[expected]
            out[motif]["tested"].append(expected in tested and competing not in tested)
            out[motif]["tested_silent"].append(not tested)
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in out.items()}


def markdown(rows, accuracy):
    lines = [
        "| Benchmark | Networks | Precision | Recall | Median TE (bits) | Intrinsic share | Synergistic share | Shared share | Synergy-dominant (gated) | Intrinsic sig. | Synergistic sig. | Shared sig. | No TE layer sig. | Dominant layer confirmed | Shared-only edges / net: gated, tested | Shared-only edges that are real links |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['networks']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['te_bits']:.3f} | "
            f"{r['intrinsic_share']:.2f} | {r['synergistic_share']:.2f} | {r['shared_share']:.2f} | "
            f"{r['gated_synergy_dominant']:.2f} | {r['tested_intrinsic']:.2f} | {r['tested_synergistic']:.2f} | "
            f"{r['tested_shared']:.2f} | {r['tested_none']:.2f} | {r['agreement']:.2f} | "
            f"{r['shared_only_edges']:.1f}, {r['shared_only_tested']:.1f} | {r['shared_only_real']:.2f} |"
        )
    lines += ["", "| Motif | Edge-gated correct | Per-layer correct | Per-layer silent |", "|---|---|---|---|"]
    for m, a in sorted(accuracy.items()):
        lines.append(f"| {m} | {a['gated']:.2f} | {a['tested']:.2f} | {a['tested_silent']:.2f} |")
    return "\n".join(lines)


def main():
    records = load(sys.argv[1])
    print(markdown(summarize(records), motif_accuracy(records)))


if __name__ == "__main__":
    main()
