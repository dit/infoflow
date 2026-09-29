"""
Per-layer network comparison between two conditions.

Following IDTxl's network comparison, the units of observation are either
*replications* (trials) within one subject or *subjects* (independent datasets),
and the two conditions are either *dependent* (paired units, e.g. baseline vs
task in the same trials or matched patients and controls) or *independent*. The
test runs on the union of both conditions' edges: each edge's source and context
variables come from both skeletons, its layers are estimated in each condition,
and the difference is compared with the distribution obtained by permuting
condition labels (swapping within pairs for dependent designs, shuffling across
pooled units otherwise). P-values are two-sided and corrected per layer.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .data import DiscreteData, realizations
from .layers import LAYERS, adjust_layer_pvalues
from .measures import edge_flows

__all__ = ("NetworkComparison", "compare_networks")


@dataclass
class NetworkComparison:
    """
    Differences in each layer between conditions A and B.

    Attributes
    ----------
    dataset : xarray.Dataset
        ``a``, ``b``, ``difference``, ``pvalue``, ``qvalue``, ``significant`` over
        (layer, source, target).
    edges : dict
        ``(source, target) -> (source variables, context variables)`` of the union.
    """

    dataset: xr.Dataset
    edges: dict


def _edge_values(data, target, x_vars, w_vars, restarts=1):
    variables = list(dict.fromkeys(list(x_vars) + list(w_vars)))
    r = realizations(data, target, variables, max_lag=max(v[1] for v in variables))
    if len(r) < 10:
        return dict.fromkeys(LAYERS, np.nan)
    w = r.columns(list(w_vars)) if w_vars else None
    flow = edge_flows(r.present, r.columns(list(x_vars)), w, estimator="miller_madow", restarts=restarts, prng=0)
    return {k: flow.as_dict()[k] for k in LAYERS}


def _union_edges(net_a, net_b):
    edges = {}
    for net in (net_a, net_b):
        for t, sk in net.skeleton.items():
            base = list(sk.conditionals) + list(sk.target_past)
            for p in sk.parents():
                x = sk.source_variables(p)
                w = base + [v for v in sk.sources if v[0] != p]
                ex, ew = edges.get((p, t), ([], []))
                edges[p, t] = (list(dict.fromkeys(ex + x)), list(dict.fromkeys(ew + w)))
    return edges


def _pool(units):
    """
    Pool a list of DiscreteData units into one (their trials concatenated).
    """
    first = units[0]
    return DiscreteData(
        past=[p for u in units for p in u.past],
        present=[p for u in units for p in u.present],
        past_offset=first.past_offset,
        present_offset=first.present_offset,
        past_alphabet=np.max([u.past_alphabet for u in units], axis=0),
        present_alphabet=np.max([u.present_alphabet for u in units], axis=0),
        lag_step=first.lag_step,
        names=first.names,
        discretizers=first.discretizers,
    )


def compare_networks(
    net_a,
    net_b,
    data_a,
    data_b,
    design="within",
    dependent=False,
    n_perm=200,
    alpha=0.05,
    prng=None,
):
    """
    Compare the layers of two inferred networks.

    Parameters
    ----------
    net_a, net_b : MultiplexNetwork
        Networks inferred under conditions A and B (their skeletons define the union).
    data_a, data_b : DiscreteData or list of DiscreteData
        For ``design='within'``, one DiscreteData per condition whose trials are the
        units. For ``design='between'``, a list of per-subject DiscreteData.
    design : {'within', 'between'}
    dependent : bool
        Paired units (requires equal numbers of units per condition).
    n_perm : int
    alpha : float
        FDR level per layer.
    prng : None, int, Generator

    Returns
    -------
    NetworkComparison
    """
    rng = as_generator(prng)
    edges = _union_edges(net_a, net_b)
    names = net_a.names
    P = len(names)
    if design == "within":
        units_a = [data_a.subset_trials([i]) for i in range(data_a.n_trials)]
        units_b = [data_b.subset_trials([i]) for i in range(data_b.n_trials)]
    elif design == "between":
        units_a, units_b = list(data_a), list(data_b)
    else:
        raise ValueError("design must be 'within' or 'between'")
    if dependent and len(units_a) != len(units_b):
        raise ValueError("dependent designs need equal numbers of units per condition")
    units = units_a + units_b
    labels = np.array([0] * len(units_a) + [1] * len(units_b))

    if design == "between":
        # Per-subject values are fixed; permutations reassign subjects to conditions.
        per_unit = {e: np.array([[_edge_values(u, e[1], *edges[e])[k] for k in LAYERS] for u in units]) for e in edges}

        def statistic(lab, e):
            v = per_unit[e]
            return np.nanmean(v[lab == 0], axis=0), np.nanmean(v[lab == 1], axis=0)

    else:

        def statistic(lab, e):
            a = _edge_values(_pool([u for u, g in zip(units, lab, strict=True) if g == 0]), e[1], *edges[e])
            b = _edge_values(_pool([u for u, g in zip(units, lab, strict=True) if g == 1]), e[1], *edges[e])
            return np.array([a[k] for k in LAYERS]), np.array([b[k] for k in LAYERS])

    def permuted():
        if dependent:
            n = len(units_a)
            swap = rng.random(n) < 0.5
            lab = labels.copy()
            lab[:n][swap], lab[n:][swap] = 1, 0
            return lab
        return rng.permutation(labels)

    shape = (len(LAYERS), P, P)
    va, vb, diff, pvalue = (np.full(shape, np.nan) for _ in range(4))
    for e in edges:
        a, b = statistic(labels, e)
        observed = a - b
        null = np.array([np.subtract(*statistic(permuted(), e)) for _ in range(n_perm)])
        s, t = e
        va[:, s, t], vb[:, s, t], diff[:, s, t] = a, b, observed
        pvalue[:, s, t] = (1 + np.sum(np.abs(null) >= np.abs(observed) - 1e-12, axis=0)) / (1 + n_perm)
    significant, adjusted = adjust_layer_pvalues({k: pvalue[i] for i, k in enumerate(LAYERS)}, alpha)
    ds = xr.Dataset(
        {
            "a": (("layer", "source", "target"), va),
            "b": (("layer", "source", "target"), vb),
            "difference": (("layer", "source", "target"), diff),
            "pvalue": (("layer", "source", "target"), pvalue),
            "qvalue": (("layer", "source", "target"), np.stack([adjusted[k] for k in LAYERS])),
            "significant": (("layer", "source", "target"), np.stack([significant[k] for k in LAYERS])),
        },
        coords={"layer": list(LAYERS), "source": names, "target": names},
        attrs={"design": design, "dependent": int(dependent), "n_perm": n_perm},
    )
    return NetworkComparison(ds, edges)
