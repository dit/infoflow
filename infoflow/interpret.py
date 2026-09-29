"""
Interpreting edges: direct, confounded, and mediated roles, and latent-confounding flags.

Roles compare the bivariate picture (transfer entropy given only the target's own
past) with the multivariate skeleton. If a bivariately significant edge
``X -> Y`` disappears once some variable ``Z`` joins the context, ``Z`` explains it
away, and the edge is labelled

* ``confounded`` if ``Z``'s process drives both ``X`` and ``Y``,
* ``mediated`` if ``X`` drives ``Z``'s process, which drives ``Y``,
* ``explained`` otherwise,

while edges that survive in the skeleton are ``direct``. Unobserved drivers cannot
be conditioned away, so they are *flagged*, with the evidence, not asserted:

* ``shared-dominant``: significant shared flow at least twice the intrinsic flow,
  with no significant intrinsic flow, not explained by an observed confounder or
  mediator;
* ``unexplained-tdmi``: a shared-only candidate with no observed confounder or mediator;
* ``bidirectional``: significant flow in both directions;
* ``zero-lag``: the contemporaneous dependence exceeds the lagged dependence;
* ``history-collider``: transfer entropy grows when more target history is
  conditioned on while the intrinsic flow stays near zero. Conditioning on a
  target past driven by a latent input opens a path from the source to that
  input (collider bias through the target's past).

These labels assume stationarity and faithfulness; they interpret a functional
network and are not a causal discovery.
"""

import numpy as np
from dit.inference._symbols import as_generator

from .embedding import candidate_lags
from .selection import _cmi, _Columns, _max_statistic, _permuter

__all__ = ("edge_roles", "latent_flags")


def edge_roles(data, skeleton, embeddings, settings, prng=None):
    """
    Bivariate-versus-multivariate roles for every ordered pair of processes.

    Returns
    -------
    roles : np.ndarray of str
        (source, target): ``direct``, ``confounded``, ``mediated``, ``explained``, or ``""``.
    explained_by : np.ndarray of str
        (source, target): the explaining variable as ``"process@lag"``.
    """
    rng = as_generator(prng)
    P = data.n_processes
    roles = np.full((P, P), "", dtype=object)
    explained_by = np.full((P, P), "", dtype=object)
    parents = {t: set(sk.parents()) for t, sk in skeleton.items()}
    for t, sk in skeleton.items():
        base = list(sk.conditionals) + list(sk.target_past)
        others = [p for p in range(P) if p != t]
        cands = {p: [(p, lag) for lag in candidate_lags(data, embeddings, p)] for p in others}
        everything = list(dict.fromkeys(base + [v for c in cands.values() for v in c] + list(sk.sources)))
        max_lag = max((v[1] for v in everything), default=0)
        cols = _Columns(data, t, everything, max_lag)
        if cols.n < 10:
            continue
        perm = _permuter(cols, settings, rng)
        for p in others:
            if p in parents[t]:
                roles[p, t] = "direct"
                continue
            best, value, pv = _max_statistic(cols, cands[p], base, settings.n_perm_max_stat, perm)
            if pv > settings.alpha_max_stat:
                continue
            # Which selected variable explains the bivariate dependence away?
            reductions = []
            for z in sk.sources:
                if z[0] == p:
                    continue
                cond = base + [z]
                zc, Kz = cols.joint(cond)
                c, a = cols.column(best)
                reductions.append((value - _cmi(cols.y, cols.Ky, c, a, zc, Kz), z))
            if not reductions:
                roles[p, t] = "explained"
                continue
            _, z = max(reductions)
            explained_by[p, t] = f"{data.names[z[0]]}@{z[1]}"
            if z[0] in parents.get(p, set()):
                roles[p, t] = "confounded"
            elif p in parents.get(z[0], set()):
                roles[p, t] = "mediated"
            else:
                roles[p, t] = "explained"
    return roles.astype(str), explained_by.astype(str)


def _zero_lag_mi(data, s, t):
    xs = np.concatenate([past[s] for past in data.past])
    ys = np.concatenate([past[t] for past in data.past])
    ok = (xs >= 0) & (ys >= 0)
    from dit.inference import conditional_mutual_information

    return conditional_mutual_information(xs[ok], ys[ok])


def latent_flags(data, skeleton, edges, dataset, embeddings, roles, margin=0.01):
    """
    Evidence-carrying flags for possible latent confounding, per edge.

    Returns
    -------
    np.ndarray of str
        (source, target): comma-separated flags.
    """
    P = data.n_processes
    flags = np.full((P, P), "", dtype=object)
    sig = dataset["significant"]
    weight = dataset["weight"]
    names = list(dataset.coords["source"].values)

    def w(layer, s, t):
        v = float(weight.sel(layer=layer, source=names[s], target=names[t]))
        return 0.0 if not np.isfinite(v) else v

    def g(layer, s, t):
        return bool(sig.sel(layer=layer, source=names[s], target=names[t]))

    for s, t in edges:
        f = []
        observed = roles[s, t] in ("confounded", "mediated")
        if (
            not observed
            and g("shared", s, t)
            and not g("intrinsic", s, t)
            and w("shared", s, t) >= 2 * w("intrinsic", s, t)
        ):
            f.append("shared-dominant")
        kind = str(dataset["kind"].values[s, t])
        if kind == "shared_candidate" and roles[s, t] not in ("confounded", "mediated"):
            f.append("unexplained-tdmi")
        forward = g("intrinsic", s, t) or g("synergistic", s, t)
        backward = (t, s) in edges and (g("intrinsic", t, s) or g("synergistic", t, s))
        if forward and backward:
            f.append("bidirectional")
        if _zero_lag_mi(data, s, t) > w("tdmi", s, t) + margin:
            f.append("zero-lag")
        sk = skeleton[t]
        extra = [(t, lag) for lag in candidate_lags(data, embeddings, t) if (t, lag) not in sk.target_past][:2]
        if extra and w("intrinsic", s, t) < margin:
            x_vars = sk.source_variables(s) or [sk.tdmi_candidates.get(s, {}).get("variable", (s, 1))]
            base = list(sk.conditionals) + list(sk.target_past) + [v for v in sk.sources if v[0] != s]
            everything = list(dict.fromkeys(x_vars + base + extra))
            cols = _Columns(data, t, everything, max(v[1] for v in everything))
            xc, Kx = cols.joint(x_vars)
            z0, K0 = cols.joint(base)
            z1, K1 = cols.joint(base + extra)
            te0 = _cmi(cols.y, cols.Ky, xc, Kx, z0, K0)
            te1 = _cmi(cols.y, cols.Ky, xc, Kx, z1, K1)
            if te1 > te0 + margin:
                f.append("history-collider")
        flags[s, t] = ",".join(f)
    return flags.astype(str)
