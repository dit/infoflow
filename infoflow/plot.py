"""
Interactive plots of multiplex networks with HoloViews (bokeh backend).
"""

import numpy as np

from .layers import LAYERS

__all__ = ("plot_layer", "plot_multiplex")


def _hv():
    try:
        import holoviews as hv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("plotting needs holoviews and bokeh (pip install infoflow[plot])") from error
    hv.extension("bokeh", logo=False)
    return hv


def plot_layer(network, layer="intrinsic", significant_only=True):
    """
    One layer as a directed ``hv.Graph`` on a circular layout, edges coloured by weight.
    """
    hv = _hv()
    names = network.names
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    nodes = hv.Nodes(
        (np.cos(angles), np.sin(angles), names, network.dataset.get("ais", np.zeros(len(names)))),
        vdims=["name", "ais"],
    )
    g = network.layer(layer, significant_only)
    edges = [(names.index(u), names.index(v), d["weight"], d["delay"]) for u, v, d in g.edges(data=True)]
    graph = hv.Graph((edges, nodes), vdims=["weight", "delay"]).opts(
        directed=True,
        arrowhead_length=0.04,
        edge_color="weight",
        edge_cmap="viridis",
        edge_line_width=3,
        node_size=18,
        node_color="ais",
        cmap="Blues",
        title=layer,
        tools=["hover"],
        xaxis=None,
        yaxis=None,
    )
    labels = hv.Labels((np.cos(angles) * 1.15, np.sin(angles) * 1.15, names))
    return graph * labels


def plot_multiplex(network, significant_only=True):
    """
    The intrinsic, synergistic, and shared layers side by side.
    """
    hv = _hv()
    return hv.Layout([plot_layer(network, layer, significant_only) for layer in LAYERS]).cols(3)
