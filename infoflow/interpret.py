"""
Interpreting edges: direct, confounded, mediated, and reverse roles, and latent-confounding flags.

Roles compare the bivariate picture (transfer entropy given only the target's own
past) with the multivariate skeleton. If a bivariately significant edge
``X -> Y`` disappears once some variable ``Z`` joins the context, ``Z`` explains it
away, and the edge is labelled

* ``confounded`` if ``Z``'s process drives both ``X`` and ``Y``,
* ``mediated`` if ``X`` drives ``Z``'s process, which drives ``Y``,
* ``explained`` otherwise,

while edges that survive in the skeleton are ``direct``.

Shared flow is explained separately (:func:`explain_shared`): an edge whose
time-delayed mutual information is mostly removed by an observed variable is
``confounded``, ``mediated``, or ``explained`` as above, and one whose dependence
is removed by the target's own past while the target drives the source is
``reverse`` (the source's past carries the target's past).

Unobserved drivers cannot be conditioned away, so they are *flagged*, with the
evidence, not asserted:

* ``shared-dominant``: significant shared flow at least twice the intrinsic flow,
  with no significant intrinsic flow, and no observed variable or reverse coupling
  that explains it;
* ``unexplained-tdmi``: a shared-only candidate that nothing observed explains;
* ``layers-unconfirmed``: a selected parent whose intrinsic and synergistic layers are
  both non-significant, e.g. a weak edge that a continuous (KSG) selection resolves
  but the symbol-based layer estimates cannot;
* ``bidirectional``: significant flow in both directions;
* ``zero-lag``: the source's present is significantly informative about the
  target's present given the lagged context, by more than the transfer entropy;
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
from .selection import _cmi, _Columns, _densify, _max_statistic, _permuter
from .stats import _within_strata

__all__ = ("edge_roles", "explain_shared", "latent_flags")


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


def _cmi_test(y, Ky, x, Kx, z, Kz, rng, n_perm):
    """
    Plug-in :math:`I[y : x \\mid z]`, its bias-corrected value (minus the mean of a
    within-strata permutation null), and the permutation p-value.
    """
    value = _cmi(y, Ky, x, Kx, z, Kz)
    null = np.array([_cmi(y, Ky, _within_strata(x, z, rng), Kx, z, Kz) for _ in range(n_perm)])
    return value, value - null.mean(), (1 + np.sum(null >= value)) / (1 + n_perm)


def _ancestors(parents, node):
    seen, stack = set(), list(parents.get(node, ()))
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(parents.get(p, ()))
    return seen


def _relation(parents, z, s, t):
    """
    How process `z` relates the source `s` to the target `t` in the skeleton.
    """
    anc_s, anc_t = _ancestors(parents, s), _ancestors(parents, t)
    if z in anc_s and z in anc_t:
        return "confounded"
    if s in _ancestors(parents, z) and z in anc_t:
        return "mediated"
    return "explained"


def _source_variables(sk, s, delay):
    return sk.source_variables(s) or [sk.tdmi_candidates.get(s, {}).get("variable", (s, max(int(delay), 1)))]


def explain_shared(data, skeleton, edges, dataset, fraction=0.5, alpha=0.05, n_perm=49, prng=None):
    """
    What removes each edge's time-delayed dependence: an observed variable, or the target's past.

    For every edge with significant shared flow and no significant intrinsic flow,
    the time-delayed mutual information :math:`I[Y_t : X]` is compared with
    :math:`I[Y_t : X \\mid E]` for explainers :math:`E`, tried in order: each observed
    variable (the target's other parents, and the source's parents shifted to the
    target's time frame), all of them jointly, then the target's own past. An
    explainer accounts for the dependence if the residual is not significant (a
    within-strata permutation test at level `alpha`) or its bias-corrected value is
    at most `fraction` of the bias-corrected time-delayed mutual information; plug-in
    estimates alone would not do, as conditioning inflates them.

    Returns
    -------
    kind : np.ndarray of str
        (source, target): ``confounded``, ``mediated``, ``explained``, ``reverse``,
        ``target-history`` (only the target's past explains it, and the target does
        not drive the source), ``unexplained``, or ``""`` (not tested).
    by : np.ndarray of str
        (source, target): the explaining variable(s).
    """
    rng = as_generator(prng)
    P = data.n_processes
    kind = np.full((P, P), "", dtype=object)
    by = np.full((P, P), "", dtype=object)
    names = data.names
    sig = dataset["significant"]
    parents = {t: set(sk.parents()) | set(sk.tdmi_candidates) for t, sk in skeleton.items()}
    direct = {t: set(sk.parents()) for t, sk in skeleton.items()}
    for s, t in edges:
        if t not in skeleton:
            continue
        shared = bool(sig.sel(layer="shared", source=names[s], target=names[t]))
        intrinsic = bool(sig.sel(layer="intrinsic", source=names[s], target=names[t]))
        if not shared or intrinsic:
            continue
        sk = skeleton[t]
        x_vars = _source_variables(sk, s, int(dataset["delay"].values[s, t]))
        d = min(v[1] for v in x_vars)
        observed = [v for v in sk.sources if v[0] not in (s, t)]
        if s in skeleton:
            observed += [(z, d + lag) for z, lag in skeleton[s].sources if z not in (s, t)]
        observed = list(dict.fromkeys(observed))
        history = list(sk.target_past) or [(t, 1)]
        everything = list(dict.fromkeys(x_vars + observed + history))
        cols = _Columns(data, t, everything, max(v[1] for v in everything))
        if cols.n < 10:
            continue
        xc, Kx = cols.joint(x_vars)
        empty = np.zeros(cols.n, dtype=np.int64)
        _, tdmi, _ = _cmi_test(cols.y, cols.Ky, xc, Kx, empty, 1, rng, n_perm)
        if tdmi <= 0:
            continue

        def residual(cond, cols=cols, xc=xc, Kx=Kx, tdmi=tdmi):
            zc, Kz = cols.joint(cond)
            _, debiased, pvalue = _cmi_test(cols.y, cols.Ky, xc, Kx, zc, Kz, rng, n_perm)
            return max(debiased, 0.0) / tdmi, pvalue

        def explains(r):
            return r[0] <= fraction or r[1] > alpha

        singles = sorted((residual([v]), v) for v in observed)
        singles = [(r, v) for r, v in singles if explains(r)]
        if singles:
            z = singles[0][1]
            kind[s, t], by[s, t] = _relation(parents, z[0], s, t), f"{names[z[0]]}@{z[1]}"
        elif observed and explains(residual(observed)):
            kind[s, t] = "explained"
            by[s, t] = "+".join(f"{names[v[0]]}@{v[1]}" for v in observed)
        elif explains(residual(history)) or explains(residual(history + observed)):
            reverse = t in direct.get(s, set())
            kind[s, t], by[s, t] = ("reverse" if reverse else "target-history"), f"{names[t]} past"
        else:
            kind[s, t] = "unexplained"
    return kind.astype(str), by.astype(str)


def _instantaneous(data, sk, source_parents, s, t, delay, n_perm, rng):
    """
    Bias-corrected :math:`I[Y_t : X_t \\mid Y_{\\text{past}}, X_{\\text{lagged}}, \\text{parents}]` on
    present symbols, and its permutation p-value. Conditioning on the observed
    parents of both processes leaves only same-lag dependence that nothing observed
    explains.
    """
    x_vars = _source_variables(sk, s, delay)
    parents = [v for v in sk.sources if v[0] != s] + [v for v in source_parents if v[0] != t]
    cond = list(dict.fromkeys(list(sk.target_past) + x_vars + parents))
    cols = _Columns(data, t, cond, max(v[1] for v in cond))
    if cols.n < 10:
        return 0.0, 1.0
    xs = np.array([data.present[i][s, j] for i, j in zip(cols.r.trial, cols.r.time, strict=True)])
    ok = xs >= 0
    x0, K0 = _densify(xs[ok])
    zc, Kz = cols.joint(cond)
    zc, _ = _densify(zc[ok])
    y = cols.y[ok]
    _, debiased, pvalue = _cmi_test(y, cols.Ky, x0, K0, zc, int(zc.max()) + 1, rng, n_perm)
    return debiased, pvalue


def latent_flags(
    data, skeleton, edges, dataset, embeddings, shared_kind, margin=0.01, n_perm=99, alpha=0.01, prng=None
):
    """
    Evidence-carrying flags for possible latent confounding, per edge.

    `shared_kind` is the first output of :func:`explain_shared`.

    Returns
    -------
    np.ndarray of str
        (source, target): comma-separated flags.
    """
    rng = as_generator(prng)
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
        unexplained = shared_kind[s, t] in ("unexplained", "target-history")
        if (
            unexplained
            and g("shared", s, t)
            and not g("intrinsic", s, t)
            and w("shared", s, t) >= 2 * w("intrinsic", s, t)
        ):
            f.append("shared-dominant")
        kind = str(dataset["kind"].values[s, t])
        if kind == "shared_candidate" and unexplained:
            f.append("unexplained-tdmi")
        if kind == "parent" and not (g("intrinsic", s, t) or g("synergistic", s, t)):
            f.append("layers-unconfirmed")
        forward = g("intrinsic", s, t) or g("synergistic", s, t)
        backward = (t, s) in edges and (g("intrinsic", t, s) or g("synergistic", t, s))
        if forward and backward:
            f.append("bidirectional")
        sk = skeleton[t]
        source_parents = list(skeleton[s].sources) if s in skeleton else []
        value, pvalue = _instantaneous(data, sk, source_parents, s, t, int(dataset["delay"].values[s, t]), n_perm, rng)
        if pvalue <= alpha and value > w("te", s, t) + margin:
            f.append("zero-lag")
        extra = [(t, lag) for lag in candidate_lags(data, embeddings, t) if (t, lag) not in sk.target_past][:2]
        if extra and w("intrinsic", s, t) < margin:
            x_vars = sk.source_variables(s) or [sk.tdmi_candidates.get(s, {}).get("variable", (s, 1))]
            base = list(sk.conditionals) + list(sk.target_past) + [v for v in sk.sources if v[0] != s]
            everything = list(dict.fromkeys(x_vars + base + extra))
            cols = _Columns(data, t, everything, max(v[1] for v in everything))
            xc, Kx = cols.joint(x_vars)
            z0, K0 = cols.joint(base)
            z1, K1 = cols.joint(base + extra)
            _, te0, _ = _cmi_test(cols.y, cols.Ky, xc, Kx, z0, K0, rng, n_perm // 4)
            _, te1, p1 = _cmi_test(cols.y, cols.Ky, xc, Kx, z1, K1, rng, n_perm // 4)
            if te1 > te0 + margin and p1 <= alpha:
                f.append("history-collider")
        flags[s, t] = ",".join(f)
    return flags.astype(str)
