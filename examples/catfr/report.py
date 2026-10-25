"""
Tables and figures for ``reports/catfr_pilot.md`` from ``run.py``'s outputs.

    uv run --project ../.. --with netcdf4 --with matplotlib python report.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from common import OUT
from scipy.stats import binomtest

from infoflow.layers import LAYERS

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
SIGNALS = ("hfa", "hfa-long", "hfa-pooled", "broadband", "broadband-long", "broadband-pooled")


def from_outputs(stem):
    """A summary for a run stopped during its null splits, from its saved contrast and networks."""
    cmp = xr.open_dataset(OUT / f"{stem}_compare.nc")
    names = [str(n) for n in cmp.source.values]
    signal = "hfa" if "_hfa" in stem else "broadband"
    epochs = xr.open_dataset(OUT / f"{stem.split(f'_{signal}')[0]}_{signal}.nc")
    rec = epochs.recalled.values.astype(bool)
    edges = {}
    for label in ("recalled", "forgotten"):
        net = xr.open_dataset(OUT / f"{stem}_net_{label}.nc")
        edges[label] = {k: int(net["significant"].sel(layer=k).sum()) for k in LAYERS}
    sme = []
    for k in LAYERS:
        for s, t in zip(*np.nonzero(cmp["significant"].sel(layer=k).values)):
            sme.append(
                {
                    "layer": k,
                    "source": names[s],
                    "target": names[t],
                    "regions": f"{names[s].split(':')[0]}->{names[t].split(':')[0]}",
                    "recalled": float(cmp["a"].sel(layer=k).values[s, t]),
                    "forgotten": float(cmp["b"].sel(layer=k).values[s, t]),
                    "qvalue": float(cmp["qvalue"].sel(layer=k).values[s, t]),
                }
            )
    return {
        "file": f"{stem.split(f'_{signal}')[0]}_{signal}.nc",
        "signal": signal,
        "n_per_condition": int(min(rec.sum(), (~rec).sum())),
        "edges": edges,
        "tested": {k: int(np.isfinite(cmp["pvalue"].sel(layer=k)).sum()) for k in LAYERS},
        "sme_edges": sme,
        "null": {"reps": []},
        "timing": {},
        "lags": "long" if "_long" in stem else "auto",
        "selection": "pooled" if "_pooled" in stem else "condition",
        "partial": True,
    }


def load():
    out = {}
    stems = {p.name.removesuffix("_summary.json") for p in OUT.glob("sub-*_summary.json")}
    partial = {p.name.removesuffix("_compare.nc") for p in OUT.glob("sub-*_compare.nc")} - stems
    for stem in sorted(stems | partial):
        s = from_outputs(stem) if stem in partial else json.loads((OUT / f"{stem}_summary.json").read_text())
        s["stem"] = stem
        variant = s["signal"] + ("-long" if s.get("lags") == "long" else "")
        variant += "-pooled" if s.get("selection") == "pooled" else ""
        out[(s["file"].split("_ses")[0][4:], variant)] = s
    return out


def runtime(seconds):
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def tables(summaries):
    lines = [
        "| Subject | Signal | Trials / condition | Edges recalled (I/S/Sh) | Edges forgotten (I/S/Sh) "
        "| Tested (I/S/Sh) | Significant SME (I/S/Sh) | Null splits with any significant (I/S/Sh) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (sub, sig), s in summaries.items():
        sme = Counter(e["layer"] for e in s["sme_edges"])
        reps = s["null"]["reps"]
        any_sig = [sum(r["significant"][k] > 0 for r in reps) for k in LAYERS]
        fmt = lambda d: "/".join(str(d.get(k, 0)) for k in LAYERS)  # noqa: E731
        lines.append(
            f"| {sub} | {sig} | {s['n_per_condition']} | {fmt(s['edges']['recalled'])} | "
            f"{fmt(s['edges']['forgotten'])} | {fmt(s['tested'])} | {fmt(sme)} | "
            f"{'/'.join(map(str, any_sig))} of {len(reps)} |"
        )

    lines += [
        "",
        "| Subject | Signal | Layer | Edge | Regions | Recalled (bits) | Forgotten (bits) | q |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (sub, sig), s in summaries.items():
        for e in s["sme_edges"]:
            lines.append(
                f"| {sub} | {sig} | {e['layer']} | {e['source']} -> {e['target']} | {e['regions']} | "
                f"{e['recalled']:.4f} | {e['forgotten']:.4f} | {e['qvalue']:.3f} |"
            )

    lines += [
        "",
        "| Subject | Signal | Null: significant / tested edges, pooled over splits (I/S/Sh) |",
        "|---|---|---|",
    ]
    for (sub, sig), s in summaries.items():
        reps = s["null"]["reps"]
        cells = [f"{sum(r['significant'][k] for r in reps)}/{sum(r['tested'][k] for r in reps)}" for k in LAYERS]
        lines.append(f"| {sub} | {sig} | {' '.join(cells)} |")

    lines += [
        "",
        "| Variant | Null splits | Any significant: intrinsic | synergistic | shared |",
        "|---|---|---|---|---|",
    ]
    for variant in SIGNALS:
        reps = [r for (_, sig), s in summaries.items() if sig == variant for r in s["null"]["reps"]]
        if not reps:
            continue
        cells = []
        for k in LAYERS:
            hits = sum(r["significant"][k] > 0 for r in reps)
            p = binomtest(hits, len(reps), 0.05, alternative="greater").pvalue
            cells.append(f"{hits} ({hits / len(reps):.0%}, p={p:.3f})")
        lines.append(f"| {variant} | {len(reps)} | {' | '.join(cells)} |")

    lines += [
        "",
        "| Subject | Signal | Preprocess | Contrast | Null (all splits) | auto.infer | auto edges (I/S/Sh) |",
        "|---|---|---|---|---|---|---|",
    ]
    for (sub, sig), s in summaries.items():
        t = s["timing"]
        if not t:
            continue
        auto = s.get("auto_edges", {})
        lines.append(
            f"| {sub} | {sig} | {runtime(t['preprocess'])} | {runtime(t['contrast'])} | "
            f"{runtime(t['null'])} | {runtime(t['auto']) if 'auto' in t else 'n/a'} | "
            f"{'/'.join(str(auto.get(k, 0)) for k in LAYERS) if auto else 'n/a'} |"
        )

    lines += ["", "Broadband vs HFA overlap of significant SME edges (layer, source, target):", ""]
    for suffix in ("", "-long", "-pooled"):
        for sub in sorted({k[0] for k in summaries}):
            keys = (f"hfa{suffix}", f"broadband{suffix}")
            if all((sub, k) in summaries for k in keys):
                hfa, bb = (
                    {(e["layer"], e["source"], e["target"]) for e in summaries[(sub, k)]["sme_edges"]} for k in keys
                )
                lines.append(f"- {sub}{suffix}: HFA {len(hfa)}, broadband {len(bb)}, shared {len(hfa & bb)}")
    return "\n".join(lines)


def figure(summaries):
    subjects = sorted({k[0] for k in summaries})
    for sig in SIGNALS:
        rows = [s for s in subjects if (s, sig) in summaries]
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), len(LAYERS), figsize=(4 * len(LAYERS), 3.8 * len(rows)), squeeze=False)
        for i, sub in enumerate(rows):
            ds = xr.open_dataset(OUT / f"{summaries[(sub, sig)]['stem']}_compare.nc")
            names = list(ds.source.values)
            for j, layer in enumerate(LAYERS):
                ax = axes[i, j]
                d = ds["difference"].sel(layer=layer).values
                lim = float(np.max(np.nan_to_num(np.abs(ds["difference"].values)))) or 1.0
                ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
                for s, t in zip(*np.nonzero(ds["significant"].sel(layer=layer).values)):
                    ax.text(t, s, "*", ha="center", va="center", fontsize=14)
                ax.set_xticks(range(len(names)), [n.split(":")[0] for n in names], rotation=90, fontsize=6)
                ax.set_yticks(range(len(names)), names, fontsize=6)
                ax.set_title(f"{sub} {sig}: {layer}", fontsize=9)
                ax.set_xlabel("target", fontsize=7)
                ax.set_ylabel("source", fontsize=7)
        fig.suptitle("Recalled minus forgotten flow (bits); * significant (FDR 0.05); blank: not tested", fontsize=10)
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / f"catfr_sme_{sig}.png", dpi=130)
        plt.close(fig)


def cohort_figure(summary_path):
    """Per-subject recalled minus forgotten flow for each region pair and layer (``cohort.py``)."""
    s = json.loads(Path(summary_path).read_text())
    edges = list(s["between"])
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(4.5 * len(LAYERS), 3.6), sharey=False)
    rng = np.random.default_rng(0)
    for i, (ax, layer) in enumerate(zip(axes, LAYERS, strict=True)):
        for j, e in enumerate(edges):
            d = np.array(s["per_unit"][e]["recalled"])[:, i] - np.array(s["per_unit"][e]["forgotten"])[:, i]
            ax.scatter(j + rng.uniform(-0.15, 0.15, len(d)), d, s=10, alpha=0.6)
            ax.plot([j - 0.3, j + 0.3], [np.nanmean(d)] * 2, color="k")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(range(len(edges)), edges, rotation=45, fontsize=7)
        q = [s["between"][e][layer]["qvalue"] for e in edges]
        ax.set_title(f"{layer} (smallest q = {min(q):.2f})", fontsize=9)
        ax.set_ylabel("recalled - forgotten (bits)", fontsize=7)
    fig.suptitle(f"Cohort: {len(s['subjects'])} subjects, region-averaged HFA; line: mean", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "catfr_cohort.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", help="cohort summary.json from cohort.py")
    args = parser.parse_args()
    summaries = load()
    print(tables(summaries))
    figure(summaries)
    if args.cohort:
        cohort_figure(args.cohort)


if __name__ == "__main__":
    main()
