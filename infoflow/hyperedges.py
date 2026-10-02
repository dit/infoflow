"""
Hyperedges: partial information decompositions of a target's sets of parents.

For each target, every set of up to ``max_sources`` parents (and every pair the
synergy-aware parent search selected jointly) is decomposed with a partial
information decomposition :cite:`Williams2010` of :math:`I[Y_t : (S_1, \\ldots, S_k)]`
into redundant, unique, synergistic, and higher-order atoms, using any of dit's PID
measures (:data:`PID_MEASURES`; default ``'ccs'`` :cite:`Ince2017`, which is fast
and reliable for three or more sources; ``'broja'`` :cite:`Bertschinger2014` is limited
to pairs on sample distributions).

*Detection.* A set's synergy (the top atom) is tested against a null that permutes
one of its sources within strata of the target. Each source keeps its relation to the
target, so redundancy and unique information survive, while the joint dependence of
the sources given the target, which synergy needs, is destroyed. P-values are
corrected per target with Benjamini--Hochberg.

*Classification.* Each set is labelled ``synergistic``, ``redundant``,
``unique-dominated``, or ``mixed`` by the share of :math:`I[Y_t : S]` in its top,
bottom, and unique atoms. As a measure-free cross-check the sign of the
O-information :cite:`Rosas2019` of :math:`(S_1, \\ldots, S_k, Y_t)` is reported
(negative: synergy-dominated). With several measures, the share of measures agreeing
with the majority label is reported.

The decompositions are of :math:`I[Y_t : S]`, not conditioned on the target's past;
they complement the synergistic edge layer, which is synergy *with the context*.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import dit
import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .data import realizations
from .measures import dense_codes

__all__ = ("PID_MEASURES", "PIDMeasure", "hypergraph", "multivariate_hyperedges", "pid_hyperedges", "pid_measure")


@dataclass(frozen=True)
class PIDMeasure:
    """
    A dit PID measure with the capabilities infoflow relies on.

    Attributes
    ----------
    name : str
        Registry key.
    cls_name : str
        The ``dit.pid`` class.
    max_sources : int, None
        Largest number of sources it handles in reasonable time (None: unknown).
    nonnegative : bool, None
        Whether its atoms are non-negative on the textbook distributions.
    cost : {'fast', 'medium', 'slow', 'unknown'}
        Relative time for two- and three-source decompositions.
    requires : tuple of str
        Extra packages it needs.
    fails_on_data : bool
        Fails on sample (non-textbook) distributions.
    """

    name: str
    cls_name: str
    max_sources: int | None = 3
    nonnegative: bool | None = True
    cost: str = "fast"
    requires: tuple = ()
    fails_on_data: bool = False

    @property
    def cls(self):
        import dit.pid

        return getattr(dit.pid, self.cls_name)


def _m(name, cls_name, **kwargs):
    return name, PIDMeasure(name, cls_name, **kwargs)


# Capabilities measured on dit's textbook distributions (redundant, XOR, unique, AND,
# three-source XOR) and on sample distributions under the synergy null (2 and 3 sources
# of 4000 samples): max_sources is the largest that gave finite atoms without errors,
# cost the time per decomposition (fast < 0.2 s, medium < 1.5 s, slow beyond).
PID_MEASURES = dict(
    [
        _m("broja", "PID_BROJA", max_sources=2),
        _m("ccs", "PID_CCS"),
        _m("ct", "PID_CT"),
        _m("deg", "PID_Deg", max_sources=2, cost="medium"),
        _m("delta", "PID_Delta", max_sources=2, nonnegative=False),
        _m("deltalambda", "PID_DeltaLambda", max_sources=2),
        _m("dep", "PID_dep", cost="slow"),
        _m("do", "PID_Do"),
        _m("gh", "PID_GH", max_sources=2, cost="slow"),
        _m("gk", "PID_GK"),
        _m("ig", "PID_IG", max_sources=2),
        _m("ipid", "PID_IPID", max_sources=2),
        _m("mc", "PID_MC", max_sources=2, cost="medium"),
        _m("mes", "PID_MES"),
        _m("mmi", "PID_MMI"),
        _m("pm", "PID_PM", nonnegative=False),
        _m("prec", "PID_Prec", max_sources=None, nonnegative=None, cost="unknown", requires=("pypoman",)),
        _m("proj", "PID_Proj", max_sources=None, fails_on_data=True),
        _m("ra", "PID_RA", nonnegative=False, cost="slow"),
        _m("rav", "PID_RAV", max_sources=2, cost="slow"),
        _m("rdr", "PID_RDR"),
        _m("rr", "PID_RR"),
        _m("skar_nw", "PID_SKAR_nw", max_sources=2),
        _m("skar_owa", "PID_SKAR_owa", max_sources=2, cost="medium"),
        _m("skar_owb", "PID_SKAR_owb", cost="slow"),
        _m("skar_tw", "PID_SKAR_tw", max_sources=2, cost="slow"),
        _m("sx", "PID_SX", nonnegative=False, cost="medium"),
        _m("wb", "PID_WB"),
    ]
)


def pid_measure(spec):
    """
    A :class:`PIDMeasure` from a registry name, a :class:`PIDMeasure`, or a dit PID class.

    Measures that fail on sample distributions are refused.
    """
    if isinstance(spec, PIDMeasure):
        return spec
    if isinstance(spec, str):
        try:
            measure = PID_MEASURES[spec.lower()]
        except KeyError:
            raise ValueError(f"unknown PID measure {spec!r}; choose from {sorted(PID_MEASURES)}") from None
        if measure.fails_on_data:
            raise ValueError(f"PID measure {spec!r} fails on sample distributions")
        return measure
    if isinstance(spec, type):
        for m in PID_MEASURES.values():
            if m.cls_name == spec.__name__:
                return m
        return PIDMeasure(spec.__name__, spec.__name__, max_sources=None, nonnegative=None, cost="unknown")
    raise TypeError(f"cannot interpret {spec!r} as a PID measure")


def _resolve(measure):
    measure = "broja" if measure is None else measure
    return [pid_measure(m) for m in (measure if isinstance(measure, (list, tuple)) else [measure])]


def _decompose(measure, d, k):
    cls = measure.cls if isinstance(measure, PIDMeasure) else measure
    pid = cls(d, [[i] for i in range(k)], [k])
    atoms = {node: float(pid.get_pi(node)) for node in pid._lattice}
    top = tuple(range(k))
    redundancy = atoms[tuple((i,) for i in range(k))]
    synergy = atoms[(top,)]
    unique = [atoms[((i,),)] for i in range(k)]
    total = sum(atoms.values())
    return {"redundancy": redundancy, "synergy": synergy, "unique": unique, "total": total, "atoms": atoms}


def _label(parts, threshold=0.5):
    total = parts["total"]
    if total <= 1e-9:
        return "none"
    shares = {
        "synergistic": parts["synergy"] / total,
        "redundant": parts["redundancy"] / total,
        "unique-dominated": sum(parts["unique"]) / total,
    }
    best = max(shares, key=lambda k: shares[k])
    return best if shares[best] > threshold else "mixed"


def _distribution(columns, max_outcomes):
    codes = np.stack([dense_codes(c)[0] for c in columns], axis=1)
    keys, counts = np.unique(codes, axis=0, return_counts=True)
    if len(keys) > max_outcomes:
        return None
    return dit.Distribution([tuple(int(v) for v in k) for k in keys], counts / counts.sum())


def _within_target(x, y, rng):
    grouped = np.argsort(y, kind="stable")
    shuffled = np.lexsort((rng.random(len(y)), y))
    out = np.empty_like(x)
    out[grouped] = x[shuffled]
    return out


def _source_sets(sk, max_sources):
    parents = sk.parents()
    sets = [s for k in range(2, max_sources + 1) for s in combinations(parents, k)]
    for pair in sk.pairs:
        procs = tuple(sorted({v[0] for v in pair if v[0] != sk.target}))
        if len(procs) >= 2 and procs not in sets:
            sets.append(procs)
    return sets


def multivariate_hyperedges(
    data, skeleton, measure="ccs", max_sources=3, n_null=None, alpha=0.05, max_outcomes=2048, prng=None
):
    """
    Detect and classify hyperedges: PIDs of every target's sets of 2..`max_sources` parents.

    Parameters
    ----------
    data : DiscreteData
    skeleton : dict
        ``target -> TargetSkeleton``.
    measure : str, PIDMeasure, dit PID class, or a list of them
        The first measure drives detection; with several, every measure's atoms and
        label are reported along with their agreement.
    max_sources : int
        Largest source set (capped by each measure's ``max_sources``).
    n_null : int, None
        Within-target permutations for the synergy test (0 skips testing). None uses
        ``max(99, ceil(m / alpha))`` for a target with ``m`` source sets, so that the
        smallest attainable p-value can survive the per-target correction.
    alpha : float
        FDR level per target.
    max_outcomes : int
        Skip sets whose joint distribution has more outcomes.
    prng : None, int, Generator

    Returns
    -------
    xarray.Dataset
        Over ``hyperedge`` (and ``measure``): ``target``, ``sources`` (process names
        joined by ``+``), ``order``, ``redundancy``, ``synergy``, ``unique``,
        ``total``, ``label`` (per measure); ``pvalue``, ``qvalue``, ``significant``,
        ``o_information``, ``o_label``, ``consensus``, ``agreement``.
    """
    from dit.multivariate import o_information

    from .stats import benjamini_hochberg

    rng = as_generator(prng)
    measures = _resolve(measure)
    lead = measures[0]
    names = data.names
    rows = []
    for t, sk in skeleton.items():
        cap = min(max_sources, *(m.max_sources or max_sources for m in measures))
        target_rows = []
        sets = _source_sets(sk, cap)
        resamples = n_null if n_null is not None else max(99, int(np.ceil(len(sets) / alpha)))
        for procs in sets:
            variables = [sk.source_variables(p) for p in procs]
            flat = list(dict.fromkeys(v for vs in variables for v in vs))
            if not flat:
                continue
            r = realizations(data, t, flat, max_lag=max(v[1] for v in flat))
            sources = [dense_codes(r.columns(vs))[0] for vs in variables]
            y = dense_codes(r.present)[0]
            d = _distribution([*sources, y], max_outcomes)
            if d is None:
                continue
            k = len(procs)
            parts = [_decompose(m, d, k) for m in measures]
            pvalue = np.nan
            if resamples:
                observed = parts[0]["synergy"]
                null = []
                for _ in range(resamples):
                    i = int(rng.integers(k))
                    permuted = [_within_target(s, y, rng) if j == i else s for j, s in enumerate(sources)]
                    dn = _distribution([*permuted, y], max_outcomes)
                    null.append(_decompose(lead, dn, k)["synergy"] if dn is not None else 0.0)
                pvalue = (1 + np.sum(np.asarray(null) >= observed - 1e-12)) / (1 + resamples)
            labels = [_label(p) for p in parts]
            consensus, votes = Counter(labels).most_common(1)[0]
            omega = float(o_information(d))
            target_rows.append(
                {
                    "target": names[t],
                    "sources": "+".join(names[p] for p in procs),
                    "order": k,
                    "redundancy": [p["redundancy"] for p in parts],
                    "synergy": [p["synergy"] for p in parts],
                    "unique": [sum(p["unique"]) for p in parts],
                    "total": [p["total"] for p in parts],
                    "label": labels,
                    "consensus": consensus,
                    "agreement": votes / len(labels),
                    "o_information": omega,
                    "o_label": "synergy-dominated" if omega < 0 else "redundancy-dominated",
                    "pvalue": pvalue,
                }
            )
        if target_rows and resamples:
            reject, q = benjamini_hochberg([row["pvalue"] for row in target_rows], alpha)
            for row, rj, qv in zip(target_rows, reject, q, strict=True):
                row["qvalue"], row["significant"] = float(qv), bool(rj)
        else:
            for row in target_rows:
                row["qvalue"], row["significant"] = np.nan, False
        rows.extend(target_rows)
    per_hyperedge = ("target", "sources", "order", "consensus", "agreement", "o_information", "o_label")
    per_hyperedge += ("pvalue", "qvalue", "significant")
    per_measure = ("redundancy", "synergy", "unique", "total", "label")
    data_vars: dict = {k: (("hyperedge",), [row[k] for row in rows]) for k in per_hyperedge}
    for k in per_measure:
        data_vars[k] = (("hyperedge", "measure"), np.array([row[k] for row in rows]).reshape(len(rows), len(measures)))
    return xr.Dataset(
        data_vars,
        coords={"hyperedge": np.arange(len(rows)), "measure": [m.name for m in measures]},
        attrs={"max_sources": max_sources, "n_null": n_null, "alpha": alpha, "lead_measure": lead.name},
    )


def hypergraph(hyperedges, significant_only=True):
    """
    A bipartite networkx graph: process nodes, one node per hyperedge, and edges from
    each source to its hyperedge and from the hyperedge to the target.
    """
    import networkx as nx

    g = nx.DiGraph()
    for i in range(hyperedges.sizes.get("hyperedge", 0)):
        h = hyperedges.isel(hyperedge=i)
        if significant_only and not bool(h["significant"]):
            continue
        node = f"h{i}:{h['sources'].item()}->{h['target'].item()}"
        g.add_node(node, kind="hyperedge", label=str(h["consensus"].item()), order=int(h["order"]))
        for s in str(h["sources"].item()).split("+"):
            g.add_edge(s, node)
        g.add_edge(node, str(h["target"].item()))
    for n in g.nodes:
        g.nodes[n].setdefault("kind", "process")
    return g


def pid_hyperedges(data, skeleton, measure=None, max_outcomes=512):
    """
    Two-source PIDs for every pair of parents of every target, as a list of dicts.

    ``target``, ``sources`` (process pair), ``redundancy``, ``unique`` (pair), ``synergy``
    in bits. See :func:`multivariate_hyperedges` for larger sets, testing, and labels.
    """
    lead = _resolve(measure)[0]
    out = []
    for t, sk in skeleton.items():
        for a, b in combinations(sk.parents(), 2):
            va, vb = sk.source_variables(a), sk.source_variables(b)
            variables = list(dict.fromkeys(va + vb))
            r = realizations(data, t, variables, max_lag=max(v[1] for v in variables))
            d = _distribution([r.columns(va), r.columns(vb), r.present], max_outcomes)
            if d is None:
                continue
            parts = _decompose(lead, d, 2)
            out.append(
                {
                    "target": t,
                    "sources": (a, b),
                    "redundancy": parts["redundancy"],
                    "unique": tuple(parts["unique"]),
                    "synergy": parts["synergy"],
                }
            )
    return out
