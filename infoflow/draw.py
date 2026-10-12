"""
Drawing multiplex networks.

Five views of the same network:

* ``"glyphs"``: one arrow per source–target pair that encodes the decomposition's
  identities. Its core (width ∝ intrinsic flow) sits inside a body (width ∝
  transfer entropy = intrinsic + synergistic) and a halo (width ∝ time-delayed mutual
  information = intrinsic + shared), so the part of the body outside the core is
  synergy and the part of the halo outside it is shared flow.
* ``"arcs"``: one curved arc per layer and pair, coloured by layer, width ∝ weight.
* ``"stacked"``: one tilted plane per layer, nodes at the same positions on every
  plane, joined across planes by faint lines.
* ``"matrices"``: a source × target heatmap per layer; scales to large networks.

``hyperedges=True`` overlays the detected PID hyperedges (a hub joining each source
set to its target) on the glyph and arc views, and ``machines=True`` sizes nodes by
their ε-machines' statistical complexity and adds transducer details to the hover.

Each view first builds a backend-neutral :class:`Scene` (paths, polygons, points,
labels, with hover data), which is rendered either interactively with HoloViews and
bokeh (``backend="bokeh"``) or as a static matplotlib figure
(``backend="matplotlib"``), so the interactive and the exported figure show the same
thing.
"""

from dataclasses import dataclass, field

import numpy as np

from .layers import LAYERS

__all__ = ("LAYER_COLORS", "Scene", "draw", "node_positions", "save", "scene")

#: Layer colours (ColorBrewer Set1): red intrinsic, blue synergistic, green shared.
LAYER_COLORS = {"intrinsic": "#e41a1c", "synergistic": "#377eb8", "shared": "#4daf4a"}
LAYER_CMAPS = {"intrinsic": "Reds", "synergistic": "Blues", "shared": "Greens"}
_NODE = "#f4f4f4"
_INK = "#333333"
_FAINT = "#b0b0b0"
# Hyperedge labels in the colour of the layer they resemble.
_HYPER = {
    "synergistic": LAYER_COLORS["synergistic"],
    "redundant": LAYER_COLORS["shared"],
    "unique-dominated": LAYER_COLORS["intrinsic"],
    "mixed": "#888888",
}
NODE_RADIUS = 0.1


@dataclass
class Path:
    x: np.ndarray
    y: np.ndarray
    color: str
    width: float
    alpha: float = 1.0
    dash: str = "solid"
    group: str = "edges"
    hover: dict | None = None


@dataclass
class Polygon:
    x: np.ndarray
    y: np.ndarray
    color: str
    alpha: float = 1.0
    line_color: str | None = None
    group: str = "polygons"


@dataclass
class Point:
    x: float
    y: float
    size: float
    color: str
    line_color: str = _INK
    group: str = "nodes"
    hover: dict | None = None
    alpha: float = 1.0


@dataclass
class Label:
    x: float
    y: float
    text: str
    size: float = 10.0
    color: str = _INK
    align: str = "center"


@dataclass
class Scene:
    """
    Backend-neutral drawing: items in drawing order, a legend, and a title.
    """

    items: list = field(default_factory=list)
    legend: list = field(default_factory=list)
    title: str = ""

    def add(self, item):
        self.items.append(item)
        return item


def node_positions(network, positions=None, layout="spring", seed=0):
    """
    ``name -> (x, y)`` in roughly [-1, 1]², shared by every view and layer.

    `positions` (a dict by name or a (nodes, 2) array) is used as given, rescaled. Otherwise
    ``layout="spring"`` lays out the union of every layer's significant edges, and
    ``"circle"`` puts the nodes on a circle.
    """
    import networkx as nx

    names = network.names
    if positions is not None:
        pos = positions if isinstance(positions, dict) else dict(zip(names, np.asarray(positions, float), strict=True))
        xy = np.array([pos[n] for n in names], dtype=float)
    elif layout == "circle" or len(names) < 3:
        a = np.pi / 2 - 2 * np.pi * np.arange(len(names)) / len(names)
        xy = np.column_stack([np.cos(a), np.sin(a)])
    elif layout == "spring":
        g = nx.Graph()
        g.add_nodes_from(names)
        for e in _edges(network, True):
            w = sum(e["value"][k] for k in LAYERS)
            g.add_edge(
                e["source"], e["target"], weight=g.get_edge_data(e["source"], e["target"], {}).get("weight", 0) + w
            )
        pos = nx.spring_layout(g, seed=seed, weight=None)
        xy = np.array([pos[n] for n in names], dtype=float)
    else:
        raise ValueError("layout must be 'spring' or 'circle'")
    xy = xy - xy.mean(axis=0)
    span = np.abs(xy).max() or 1.0
    return {n: tuple(p) for n, p in zip(names, xy / span, strict=True)}


def _edges(network, significant_only):
    """
    One record per tested pair: names, delay, kind, and per-layer value and significance.
    """
    ds = network.dataset
    names = network.names
    out = []
    for s, t in sorted(network.edges):
        src, tgt = names[s], names[t]
        value, sig = {}, {}
        for layer in ("te", "tdmi", *LAYERS):
            v = float(ds["weight"].sel(layer=layer, source=src, target=tgt))
            sig[layer] = bool(ds["significant"].sel(layer=layer, source=src, target=tgt))
            value[layer] = 0.0 if not np.isfinite(v) or (significant_only and not sig[layer]) else max(v, 0.0)
        if significant_only and not any(sig[k] for k in LAYERS):
            continue
        out.append(
            dict(
                source=src,
                target=tgt,
                s=s,
                t=t,
                delay=int(ds["delay"].values[s, t]),
                kind=str(ds["kind"].values[s, t]),
                value=value,
                significant=sig,
            )
        )
    return out


def _bezier(p, q, bend, n=48):
    """
    A quadratic Bézier from `p` to `q` bulging to the right of travel by `bend` × length.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    normal = np.array([d[1], -d[0]])
    c = (p + q) / 2 + bend * normal
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * c + t**2 * q


def _trim(points, start, end):
    """
    The polyline with arc length `start` cut from its start and `end` from its end.
    """
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total <= start + end:
        return points
    keep = np.linspace(start, total - end, len(points))
    return np.column_stack([np.interp(keep, s, points[:, 0]), np.interp(keep, s, points[:, 1])])


HEAD_LENGTH, HEAD_WIDTH = 0.09, 0.075


def _arrow(points, length=HEAD_LENGTH, width=HEAD_WIDTH):
    """
    The polyline shortened to end at the base of an arrowhead, and the arrowhead
    (a triangle whose tip is the original end).
    """
    tip, back = points[-1], points[-4]
    d = tip - back
    d = d / (np.linalg.norm(d) or 1.0)
    n = np.array([-d[1], d[0]])
    base = tip - d * length
    tri = np.array([tip, base + n * width / 2, base - n * width / 2])
    return _trim(points, 0.0, length * 0.9), (tri[:, 0], tri[:, 1])


def _width_scale(values, max_width):
    vmax = max([v for v in values if v > 0], default=1.0)
    return lambda v: max_width * v / vmax if v > 0 else 0.0


def _machine_hover(network, machines):
    """
    Per-node and per-edge hover fields from the machine layer.
    """
    if not machines:
        return {}, {}
    m = network.machines
    if m is None:
        raise ValueError("machines=True needs a network inferred with machines=True")
    md = m.dataset
    nodes = {}
    for n in network.names:
        d = md.sel(node=n)
        nodes[n] = {
            "eps_states": float(d["n_states"]),
            "C_mu": float(d["C_mu"]),
            "h_mu": float(d["h_mu"]),
            "target_states": float(d["target_n_states"]),
        }
    edges = {}
    for src, tgt in m.edges:
        e = md.sel(source=src, target=tgt)
        edges[(src, tgt)] = {
            "transducer_states": float(e["edge_n_states"]),
            "te_model": float(e["te_model"]),
            "te_gap": float(e["te_gap"]),
        }
    return nodes, edges


def node_radii(network, machines, hover_nodes):
    """
    Node radius in layout units: fixed, or (with `machines`) growing with C_mu.
    """
    if not machines:
        return dict.fromkeys(network.names, NODE_RADIUS)
    cmu = {n: hover_nodes[n]["C_mu"] for n in network.names}
    cmax = max([c for c in cmu.values() if np.isfinite(c)], default=1.0) or 1.0
    return {n: 0.06 + 0.07 * np.sqrt(max(c, 0.0) / cmax) if np.isfinite(c) else 0.06 for n, c in cmu.items()}


def _add_nodes(sc, network, pos, radii, transform=lambda p: p, hover_nodes=None, squash=1.0):
    hover_nodes = hover_nodes or {}
    ais = network.dataset.get("ais", None)
    a = np.linspace(0, 2 * np.pi, 41)
    for i, n in enumerate(network.names):
        x, y = transform(pos[n])
        r = radii[n]
        sc.add(Polygon(x + r * np.cos(a), y + squash * r * np.sin(a), _NODE, line_color=_INK, group="nodes"))
        hover = {"node": n, **({"ais": float(ais.values[i])} if ais is not None else {}), **hover_nodes.get(n, {})}
        sc.add(Point(x, y, 24.0, _NODE, line_color=_NODE, hover=hover, alpha=0.0))  # hover target
        sc.add(Label(x, y, n, size=9))


def _glyph_scene(network, pos, radii, significant_only, max_width, hover_nodes, hover_edges):
    sc = Scene(title="Flow decomposition: core intrinsic, body TE, halo TDMI")
    edges = _edges(network, significant_only)
    width = _width_scale(
        [e["value"]["intrinsic"] + max(e["value"]["synergistic"], e["value"]["shared"]) for e in edges], max_width
    )
    for e in edges:
        v = e["value"]
        full = _trim(_bezier(pos[e["source"]], pos[e["target"]], 0.12), radii[e["source"]], radii[e["target"]] + 0.01)
        pts, (hx, hy) = _arrow(full)
        halo, body, core = v["intrinsic"] + v["shared"], v["intrinsic"] + v["synergistic"], v["intrinsic"]
        hover = {
            "edge": f"{e['source']} -> {e['target']}",
            "delay": e["delay"],
            "kind": e["kind"],
            "te": v["intrinsic"] + v["synergistic"],
            "tdmi": halo,
            **{k: v[k] for k in LAYERS},
            **hover_edges.get((e["source"], e["target"]), {}),
        }
        if halo > 0:
            sc.add(Path(pts[:, 0], pts[:, 1], LAYER_COLORS["shared"], width(halo) + 4, alpha=0.35, group="halo"))
        if body > 0:
            sc.add(Path(pts[:, 0], pts[:, 1], LAYER_COLORS["synergistic"], width(body) + 1, group="body", hover=hover))
        else:
            sc.add(Path(pts[:, 0], pts[:, 1], _FAINT, 1.0, dash="dashed", group="body", hover=hover))
        if core > 0:
            sc.add(Path(pts[:, 0], pts[:, 1], LAYER_COLORS["intrinsic"], width(core), group="core"))
        sc.add(Polygon(hx, hy, _INK if body > 0 else _FAINT, group="heads"))
    sc.legend = [
        ("intrinsic (core)", LAYER_COLORS["intrinsic"]),
        ("synergistic (body outside core)", LAYER_COLORS["synergistic"]),
        ("shared (halo outside core)", LAYER_COLORS["shared"]),
    ]
    return sc


def _arc_scene(network, pos, radii, significant_only, max_width, layers, hover_nodes, hover_edges):
    sc = Scene(title="Layers as parallel arcs")
    edges = _edges(network, significant_only)
    width = _width_scale([e["value"][k] for e in edges for k in layers], max_width)
    for e in edges:
        for k, layer in enumerate(layers):
            v = e["value"][layer]
            if v <= 0:
                continue
            full = _trim(
                _bezier(pos[e["source"]], pos[e["target"]], 0.06 + 0.1 * k),
                radii[e["source"]],
                radii[e["target"]] + 0.01,
            )
            pts, (hx, hy) = _arrow(full, 0.07, 0.06)
            hover = {
                "edge": f"{e['source']} -> {e['target']}",
                "layer": layer,
                "weight": v,
                "delay": e["delay"],
                **hover_edges.get((e["source"], e["target"]), {}),
            }
            sc.add(Path(pts[:, 0], pts[:, 1], LAYER_COLORS[layer], width(v) + 1, group=f"arc-{layer}", hover=hover))
            sc.add(Polygon(hx, hy, LAYER_COLORS[layer], group="heads"))
    sc.legend = [(layer, LAYER_COLORS[layer]) for layer in layers]
    return sc


def _stacked_scene(network, pos, radii, significant_only, max_width, layers, hover_nodes, hover_edges):
    sc = Scene(title="Layers as stacked planes")
    gap = 1.25

    def plane(k):
        return lambda p: (p[0] + 0.5 * p[1], 0.38 * p[1] + gap * k)

    edges = _edges(network, significant_only)
    width = _width_scale([e["value"][k] for e in edges for k in layers], max_width)
    corners = np.array([[-1.25, -1.25], [1.25, -1.25], [1.25, 1.25], [-1.25, 1.25]])
    for k, layer in enumerate(layers):
        t = plane(k)
        quad = np.array([t(c) for c in corners])
        sc.add(Polygon(quad[:, 0], quad[:, 1], LAYER_COLORS[layer], alpha=0.06, line_color=_FAINT, group="planes"))
        lx, ly = t((-1.25, 1.25))
        sc.add(Label(lx - 0.05, ly, layer, size=10, color=LAYER_COLORS[layer], align="right"))
    for name in network.names:
        xs, ys = zip(*(plane(k)(pos[name]) for k in range(len(layers))), strict=True)
        sc.add(Path(np.array(xs), np.array(ys), _FAINT, 0.8, dash="dotted", group="pillars"))
    for k, layer in enumerate(layers):
        t = plane(k)
        for e in edges:
            v = e["value"][layer]
            if v <= 0:
                continue
            raw = _trim(_bezier(pos[e["source"]], pos[e["target"]], 0.1), radii[e["source"]], radii[e["target"]] + 0.01)
            pts, (hx, hy) = _arrow(np.array([t(p) for p in raw]), 0.07, 0.06)
            hover = {"edge": f"{e['source']} -> {e['target']}", "layer": layer, "weight": v, "delay": e["delay"]}
            hover.update(hover_edges.get((e["source"], e["target"]), {}))
            sc.add(Path(pts[:, 0], pts[:, 1], LAYER_COLORS[layer], width(v) + 1, group=f"arc-{layer}", hover=hover))
            sc.add(Polygon(hx, hy, LAYER_COLORS[layer], group="heads"))
        _add_nodes(sc, network, pos, radii, t, hover_nodes, squash=0.75)
    sc.legend = [(layer, LAYER_COLORS[layer]) for layer in layers]
    return sc


def _add_hyperedges(sc, network, pos, radii):
    h = network.hyperedges
    if h is None:
        raise ValueError("hyperedges=True needs a network inferred with hyperedges=True")
    present = set()
    for i in range(h.sizes.get("hyperedge", 0)):
        row = h.isel(hyperedge=i)
        if not bool(row["significant"]):
            continue
        sources = str(row["sources"].item()).split("+")
        target = str(row["target"].item())
        members = np.array([pos[s] for s in sources])
        tp = np.asarray(pos[target])
        hub = 0.6 * members.mean(axis=0) + 0.4 * tp
        label = str(row["consensus"].item())
        color = _HYPER.get(label, _HYPER["mixed"])
        for s in sources:
            pts = _trim(np.linspace(pos[s], hub, 8), radii[s], 0.02)
            sc.add(Path(pts[:, 0], pts[:, 1], color, 1.5, dash="dashed", group="hyper"))
        pts, (hx, hy) = _arrow(_trim(np.linspace(hub, tp, 8), 0.02, radii[target] + 0.01), 0.06, 0.05)
        sc.add(Path(pts[:, 0], pts[:, 1], color, 2.0, group="hyper"))
        sc.add(Polygon(hx, hy, color, group="heads"))
        present.add(label)
        synergy = float(np.nanmean(row["synergy"].values))
        sc.add(
            Point(
                float(hub[0]),
                float(hub[1]),
                10.0,
                color,
                line_color=color,
                group="hubs",
                hover={"hyperedge": f"{'+'.join(sources)} -> {target}", "label": label, "synergy": synergy},
            )
        )
    sc.legend += [(f"hyperedge: {k}", c) for k, c in _HYPER.items() if k in present]


def scene(
    network,
    view="glyphs",
    significant_only=True,
    layers=LAYERS,
    positions=None,
    layout="spring",
    hyperedges=False,
    machines=False,
    max_width=9.0,
    seed=0,
):
    """
    The backend-neutral :class:`Scene` of a view (see :func:`draw`); not for ``"matrices"``.
    """
    pos = node_positions(network, positions, layout, seed)
    hover_nodes, hover_edges = _machine_hover(network, machines)
    radii = node_radii(network, machines, hover_nodes)
    if view == "glyphs":
        sc = _glyph_scene(network, pos, radii, significant_only, max_width, hover_nodes, hover_edges)
    elif view == "arcs":
        sc = _arc_scene(network, pos, radii, significant_only, max_width, layers, hover_nodes, hover_edges)
    elif view == "stacked":
        if hyperedges:
            raise ValueError("the hyperedge overlay is drawn on the 'glyphs' and 'arcs' views")
        return _stacked_scene(network, pos, radii, significant_only, max_width, layers, hover_nodes, hover_edges)
    else:
        raise ValueError("view must be 'glyphs', 'arcs', 'stacked', or 'matrices'")
    if hyperedges:
        _add_hyperedges(sc, network, pos, radii)
    _add_nodes(sc, network, pos, radii, hover_nodes=hover_nodes)
    return sc


def _matrices(network, layers, significant_only):
    ds = network.dataset
    out = {}
    for layer in layers:
        w = ds["weight"].sel(layer=layer).values.astype(float)
        if significant_only:
            w = np.where(ds["significant"].sel(layer=layer).values, w, np.nan)
        out[layer] = w
    return out


# -- renderers ---------------------------------------------------------------


def _render_bokeh(sc, width=560):
    import holoviews as hv

    hv.extension("bokeh", logo=False)
    layers = []
    groups = {}
    for item in sc.items:
        groups.setdefault((type(item).__name__, item.group if hasattr(item, "group") else "labels"), []).append(item)
    order = []
    for item in sc.items:
        key = (type(item).__name__, item.group if hasattr(item, "group") else "labels")
        if key not in order:
            order.append(key)
    for key in order:
        kind, group = key
        items = groups[key]
        # Literal colours are style options, not data: HoloViews would colour-map a column.
        if kind == "Path":
            hover_keys = sorted({k for p in items if p.hover for k in p.hover})
            for dash, color, alpha in dict.fromkeys((p.dash, p.color, p.alpha) for p in items):
                sub = [p for p in items if (p.dash, p.color, p.alpha) == (dash, color, alpha)]
                data = [
                    {"x": p.x, "y": p.y, "width": p.width, **{k: (p.hover or {}).get(k, "") for k in hover_keys}}
                    for p in sub
                ]
                el = hv.Path(data, vdims=["width", *hover_keys]).opts(
                    color=color,
                    line_width="width",
                    alpha=alpha,
                    line_dash=dash,
                    line_cap="butt",
                    tools=["hover"] if hover_keys else [],
                )
                layers.append(el)
        elif kind == "Polygon":
            for color, alpha, line in dict.fromkeys((p.color, p.alpha, p.line_color) for p in items):
                sub = [p for p in items if (p.color, p.alpha, p.line_color) == (color, alpha, line)]
                layers.append(
                    hv.Polygons([{"x": p.x, "y": p.y} for p in sub]).opts(
                        color=color, alpha=alpha, line_color=line or color, line_width=1 if line else 0
                    )
                )
        elif kind == "Point":
            hover_keys = sorted({k for p in items if p.hover for k in p.hover})
            for color, line, alpha in dict.fromkeys((p.color, p.line_color, p.alpha) for p in items):
                sub = [p for p in items if (p.color, p.line_color, p.alpha) == (color, line, alpha)]
                rows = [(p.x, p.y, p.size, *[(p.hover or {}).get(k, "") for k in hover_keys]) for p in sub]
                layers.append(
                    hv.Points(rows, vdims=["size", *hover_keys]).opts(
                        size="size", color=color, line_color=line, alpha=alpha, tools=["hover"]
                    )
                )
        else:
            for align in sorted({lb.align for lb in items}):
                sub = [lb for lb in items if lb.align == align]
                for color in dict.fromkeys(lb.color for lb in sub):
                    rows = [(lb.x, lb.y, lb.text) for lb in sub if lb.color == color]
                    layers.append(
                        hv.Labels(rows, vdims=["text"]).opts(
                            text_color=color, text_font_size=f"{sub[0].size:.0f}pt", text_align=align
                        )
                    )
    for name, color in sc.legend:
        layers.append(hv.Curve([(np.nan, np.nan)], label=name).opts(color=color, line_width=6))
    return hv.Overlay(layers).opts(
        title=sc.title,
        frame_width=width,
        xaxis=None,
        yaxis=None,
        data_aspect=1,
        legend_position="right",
        show_grid=False,
        toolbar="above",
    )


def _render_matplotlib(sc, ax=None, figsize=(8.5, 6)):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pt = 0.75  # bokeh pixels -> matplotlib points
    dashes = {"solid": "-", "dashed": "--", "dotted": ":"}
    z = 1
    for item in sc.items:
        z += 0.001
        if isinstance(item, Path):
            ax.plot(
                item.x,
                item.y,
                color=item.color,
                lw=item.width * pt,
                alpha=item.alpha,
                ls=dashes[item.dash],
                solid_capstyle="butt",
                zorder=z,
            )
        elif isinstance(item, Polygon):
            ax.fill(item.x, item.y, color=item.color, alpha=item.alpha, ec=item.line_color or "none", zorder=z)
        elif isinstance(item, Point):
            if item.alpha == 0:
                continue
            ax.scatter(
                [item.x], [item.y], s=(item.size * pt) ** 2, c=item.color, edgecolors=item.line_color, zorder=z + 5
            )
        else:
            ha = {"center": "center", "right": "right", "left": "left"}[item.align]
            ax.text(item.x, item.y, item.text, ha=ha, va="center", fontsize=item.size, color=item.color, zorder=z + 6)
    if sc.legend:
        ax.legend(
            [Line2D([0], [0], color=c, lw=5) for _, c in sc.legend],
            [n for n, _ in sc.legend],
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=8,
            frameon=False,
        )
    ax.set_title(sc.title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig if fig is not None else ax.figure


def _render_matrices(network, layers, significant_only, backend, common_scale=False):
    mats = _matrices(network, layers, significant_only)
    names = network.names
    vmax = max((np.nanmax(w) for w in mats.values() if np.isfinite(w).any()), default=1.0) if common_scale else None
    if backend == "bokeh":
        import holoviews as hv

        hv.extension("bokeh", logo=False)
        panels = []
        for layer, w in mats.items():
            rows = [(names[s], names[t], w[s, t]) for s in range(len(names)) for t in range(len(names))]
            panels.append(
                hv.HeatMap(rows, kdims=["source", "target"], vdims=["bits"]).opts(
                    title=layer,
                    cmap=LAYER_CMAPS.get(layer, "Greys"),
                    colorbar=True,
                    tools=["hover"],
                    width=300,
                    height=280,
                    xrotation=90,
                    invert_yaxis=True,
                    clipping_colors={"NaN": "#f7f7f7"},
                    **({"clim": (0, vmax)} if vmax else {}),
                )
            )
        return hv.Layout(panels).cols(len(panels))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mats), figsize=(3.4 * len(mats), 3.2), squeeze=False)
    for a, (layer, w) in zip(axes[0], mats.items(), strict=True):
        cmap = LAYER_CMAPS.get(layer, "Greys")
        im = a.imshow(np.ma.masked_invalid(w.T), cmap=cmap, vmin=0, vmax=vmax)
        a.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        a.set_yticks(range(len(names)), names, fontsize=7)
        a.set_xlabel("source")
        a.set_ylabel("target")
        a.set_title(layer, fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, label="bits")
    fig.tight_layout()
    return fig


def draw(
    network,
    view="glyphs",
    backend="bokeh",
    significant_only=True,
    layers=LAYERS,
    positions=None,
    layout="spring",
    hyperedges=False,
    machines=False,
    max_width=9.0,
    seed=0,
    ax=None,
    common_scale=False,
):
    """
    Draw a multiplex network.

    Parameters
    ----------
    network : MultiplexNetwork
    view : {'glyphs', 'arcs', 'stacked', 'matrices'}
        See the module docstring.
    backend : {'bokeh', 'matplotlib'}
        Interactive HoloViews element, or a static matplotlib Figure
        (``pip install 'infoflow[plot]'`` for either).
    significant_only : bool
        Draw only significant layer values (others count as zero).
    layers : sequence of str
        Layers of the arc, stacked, and matrix views.
    positions : dict or array, None
        Node positions; by default from `layout`, shared by all layers.
    layout : {'spring', 'circle'}
    hyperedges : bool
        Overlay the network's significant PID hyperedges (glyph and arc views).
    machines : bool
        Size nodes by C_mu and add ε-machine and transducer details to the hover.
    max_width : float
        Line width (pixels) of the largest drawn flow.
    ax : matplotlib Axes, None
        Draw into existing axes (matplotlib, not 'matrices').
    common_scale : bool
        One colour range for every matrix (the layers are in the same units), instead
        of one per layer.

    Returns
    -------
    holoviews element or matplotlib Figure
    """
    if backend not in ("bokeh", "matplotlib"):
        raise ValueError("backend must be 'bokeh' or 'matplotlib'")
    if view == "matrices":
        return _render_matrices(network, layers, significant_only, backend, common_scale)
    sc = scene(network, view, significant_only, layers, positions, layout, hyperedges, machines, max_width, seed)
    if backend == "bokeh":
        return _render_bokeh(sc)
    return _render_matplotlib(sc, ax)


def save(figure, path):
    """
    Save a drawing: HoloViews elements to ``.html`` (or ``.png`` with selenium), matplotlib
    figures to any format matplotlib writes (``.pdf``, ``.svg``, ``.png``).
    """
    if hasattr(figure, "savefig"):
        figure.savefig(path, bbox_inches="tight")
        return path
    import holoviews as hv

    hv.save(figure, path, backend="bokeh")
    return path
