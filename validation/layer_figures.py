"""
Figures of inferred networks from the records of ``layer_report.py``: the ground
truth, the layers of every link selected by its transfer entropy (edge-gated point
estimates), and only the layers whose own test is significant (per-layer).

    python validation/layer_figures.py .bench_data/results/layers reports/figures \\
        motif__xor__0 synthetic__lorenz96__0 ...
"""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from infoflow.draw import LAYER_COLORS, NODE_RADIUS, draw, node_positions

TITLES = {
    "motif/xor": "XOR motif",
    "motif/common_driver": "Common-driver motif",
    "mute": "MuTE network",
    "synthetic/var": "VAR network (10 nodes)",
    "synthetic/lorenz96": "Lorenz-96 ring (10 nodes)",
    "netsim/sim5": "NetSim, 5 nodes, 1200 TRs",
    "rivers/n5": "CausalRivers, 5 stations",
}
RINGS = ("synthetic/lorenz96",)


def gated_significance(rec):
    """
    What edge-level gating reports: every layer of a link whose transfer entropy passed
    parent selection, and the TDMI layers (intrinsic and shared) of a shared-only link.
    """
    layers = rec["layers"]
    sig = np.zeros_like(rec["significant"], dtype=bool)
    for s, t in rec["parents"]:
        sig[:, s, t] = True
    for s, t in rec["shared_candidates"]:
        for name in ("tdmi", "intrinsic", "shared"):
            sig[layers.index(name), s, t] = True
    return sig


def network_from_record(rec, significant=None):
    """
    A drawable stand-in for a MultiplexNetwork (dataset, names, edges) built from a record.
    """
    P = rec["weight"].shape[1]
    names = [f"x{i}" for i in range(P)]
    kind = np.full((P, P), "", dtype=object)
    for s, t in rec["parents"]:
        kind[s, t] = "parent"
    for s, t in rec["shared_candidates"]:
        kind[s, t] = "shared_candidate"
    dims = ("layer", "source", "target")
    ds = xr.Dataset(
        {
            "weight": (dims, rec["weight"]),
            "significant": (dims, rec["significant"] if significant is None else significant),
            "delay": (("source", "target"), np.ones((P, P), dtype=int)),
            "kind": (("source", "target"), kind.astype(str)),
        },
        coords={"layer": rec["layers"], "source": names, "target": names},
    )
    te = rec["weight"][rec["layers"].index("te")]
    edges = {(int(s), int(t)): None for s, t in zip(*np.nonzero(np.isfinite(te)), strict=True)}
    return SimpleNamespace(dataset=ds, names=names, edges=edges, hyperedges=None, machines=None)


def draw_truth(ax, names, truth, pos):
    radius = NODE_RADIUS
    for s, t in truth:
        p, q = np.array(pos[names[s]]), np.array(pos[names[t]])
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=14,
                color="#555555",
                lw=1.6,
                shrinkA=12,
                shrinkB=12,
                connectionstyle="arc3,rad=0.15",
            )
        )
    for n in names:
        ax.add_patch(plt.Circle(pos[n], radius, facecolor="#f4f4f4", edgecolor="#333333", zorder=3))
        ax.text(*pos[n], n, ha="center", va="center", fontsize=8, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")


def figure(rec, path):
    net = network_from_record(rec)
    group = "/".join(rec["label"].split("/")[:-1])
    layout = "circle" if group in RINGS or len(net.names) <= 3 else "spring"
    pos = node_positions(net, layout=layout, seed=1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    draw_truth(axes[0], net.names, rec["truth"], pos)
    axes[0].set_title("Ground truth")
    gated = network_from_record(rec, gated_significance(rec))
    draw(gated, backend="matplotlib", significant_only=True, positions=pos, ax=axes[1])
    axes[1].set_title("Edge-gated: point estimates of selected links")
    draw(net, backend="matplotlib", significant_only=True, positions=pos, ax=axes[2])
    axes[2].set_title("Per-layer: significant layers only")
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    axes[0].set_xlim(axes[1].get_xlim())
    axes[0].set_ylim(axes[1].get_ylim())
    handles = [Line2D([], [], color=c, lw=4, label=layer) for layer, c in LAYER_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(TITLES.get(group, group), fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    records, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for stem in sys.argv[3:]:
        with open(records / f"{stem}.p", "rb") as fh:
            rec = pickle.load(fh)
        figure(rec, out / f"{stem}.png")
        print(out / f"{stem}.png")


if __name__ == "__main__":
    main()
