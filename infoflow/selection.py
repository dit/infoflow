"""
Skeleton inference: which variables are parents of each target.

For each target this follows the hierarchical procedure of IDTxl's multivariate
transfer entropy :cite:`Lizier2012,Novelli2019`, with two additions:

1. Greedily add the target's own past variables while the best candidate passes a
   *maximum statistic* test against surrogates.
2. Greedily add source variables the same way, then re-test the unselected target
   lags given the selected sources (a target lag can be informative only jointly
   with a source, e.g. ``y_t = x_{t-1} xor y_{t-1}``). If single-variable additions
   stop, a *synergy-aware pair search* tests pairs of remaining variables jointly,
   which recovers parents that act only together (e.g. ``Y = X1 xor X2``, where
   each alone carries no information).
3. Prune source variables with a *minimum statistic* test.
4. Test all selected sources jointly with an *omnibus* test, and give each source
   variable a p-value with the *sequential maximum statistic*.
5. Correct the omnibus p-values across targets (network FDR, Genovese constants
   :cite:`Benjamini1995,Benjamini2001`).

A separate *TDMI screen* records sources with significant time-delayed mutual
information that were not selected as parents: they are candidates for purely
shared flow. Forced conditionals and Faes compensation :cite:`Faes2013` add
variables to every conditioning set. All CMIs of one target use a common set of
realizations, so candidate sets are compared on the same samples.
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

import numpy as np
from dit.inference._symbols import as_generator

from .data import realizations
from .embedding import Embedding, candidate_lags

__all__ = (
    "SkeletonSettings",
    "TargetSkeleton",
    "check_n_perm",
    "infer_skeleton",
    "select_parents",
)


def check_n_perm(n_perm, alpha):
    """
    Refuse permutation budgets whose smallest p-value, ``1 / (n_perm + 1)``, cannot reach `alpha`.
    """
    if 1.0 / (n_perm + 1) > alpha:
        raise ValueError(f"{n_perm} permutations cannot reach alpha={alpha}; use at least {int(np.ceil(1 / alpha))}.")


@dataclass
class SkeletonSettings:
    """
    Settings for skeleton inference (defaults follow IDTxl where it has one).

    ``estimator='ksg'`` runs parent selection on the raw continuous values with the
    KSG conditional mutual information (``ksg_k`` neighbours) :cite:`Runge2018`, as
    IDTxl does, instead of the plug-in estimate on symbols. Conditioning on a parent
    continuously removes its influence exactly, where bins leave a residue inside
    each bin that can hide weak edges; the layers are still estimated on symbols.
    It is much slower.
    """

    n_perm_max_stat: int = 200
    n_perm_min_stat: int = 200
    n_perm_omnibus: int = 500
    n_perm_max_seq: int = 500
    alpha_max_stat: float = 0.05
    alpha_min_stat: float = 0.05
    alpha_omnibus: float = 0.05
    alpha_max_seq: float = 0.05
    permutation: str = "auto"
    block_size: int | None = None
    max_shift: int | None = None
    perm_range: int | None = None
    forced_conditionals: tuple = ()
    faes: bool = False
    synergy_search: bool = True
    max_pair_candidates: int = 20
    max_source_lag: int | None = None
    n_perm_pairs: int = 100
    alpha_pairs: float = 0.05
    tdmi_screen: bool = True
    n_perm_tdmi: int = 200
    alpha_tdmi: float = 0.05
    fdr: bool = True
    fdr_constant: int = 1
    estimator: str = "plugin"
    ksg_k: int = 4

    def check(self):
        for name in ("max_stat", "min_stat", "omnibus", "max_seq"):
            check_n_perm(getattr(self, f"n_perm_{name}"), getattr(self, f"alpha_{name}"))
        if self.synergy_search:
            check_n_perm(self.n_perm_pairs, self.alpha_pairs)
        if self.tdmi_screen:
            check_n_perm(self.n_perm_tdmi, self.alpha_tdmi)
        if self.estimator not in ("plugin", "ksg"):
            raise ValueError("estimator must be 'plugin' or 'ksg'")
        if self.fdr_constant not in (1, 2):
            raise ValueError("fdr_constant must be 1 (Benjamini-Hochberg) or 2 (Benjamini-Yekutieli)")


@dataclass
class TargetSkeleton:
    """
    Selected variables and test results for one target.
    """

    target: int
    target_past: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    source_pvalues: dict = field(default_factory=dict)
    conditionals: list = field(default_factory=list)
    omnibus_te: float = 0.0
    omnibus_pvalue: float = 1.0
    significant: bool = False
    pairs: list = field(default_factory=list)
    tdmi_candidates: dict = field(default_factory=dict)
    n_samples: int = 0

    def parents(self):
        """
        Source processes with at least one significant variable.
        """
        return sorted({p for p, _ in self.sources}) if self.significant else []

    def source_variables(self, process):
        return [v for v in self.sources if v[0] == process]


# -- fast plug-in CMI on integer codes -------------------------------------------------


def _densify(code):
    _, inverse = np.unique(code, return_inverse=True)
    inverse = inverse.ravel().astype(np.int64)
    return inverse, int(inverse.max()) + 1 if len(inverse) else 1


def _combine(a, Ka, b, Kb):
    if Ka * Kb > 10_000_000:
        return _densify(a * Kb + b)
    return a * Kb + b, Ka * Kb


def _entropy(code, K):
    counts = np.bincount(code, minlength=0) if K <= 10_000_000 else np.unique(code, return_counts=True)[1]
    counts = counts[counts > 0].astype(float)
    n = counts.sum()
    return float(np.log(n) - np.sum(counts * np.log(counts)) / n)


def _cmi(y, Ky, c, Kc, z, Kz):
    """
    Plug-in I[y : c | z] in bits from integer codes.
    """
    cz, Kcz = _combine(c, Kc, z, Kz)
    yz, Kyz = _combine(y, Ky, z, Kz)
    ycz, Kycz = _combine(y, Ky, cz, Kcz)
    value = _entropy(cz, Kcz) + _entropy(yz, Kyz) - _entropy(ycz, Kycz) - _entropy(z, Kz)
    return max(value, 0.0) / np.log(2)


class _Columns:
    """
    Integer columns of one target's realizations, with joint encodings.
    """

    def __init__(self, data, target, variables, max_lag):
        self.r = realizations(data, target, variables, max_lag=max_lag)
        self.variables = list(self.r.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        self.alphabet = {v: int(data.past_alphabet[v[0]]) for v in self.variables}
        self.y, self.Ky = _densify(self.r.present)
        self.n = len(self.y)

    def column(self, v):
        return self.r.values[:, self.index[v]], self.alphabet[v]

    def joint(self, variables):
        code = np.zeros(self.n, dtype=np.int64)
        K = 1
        for v in variables:
            c, a = self.column(v)
            code, K = _combine(code, K, c, a)
        return code, K

    def cmi(self, c, Kc, z, Kz):
        """
        Plug-in :math:`I[Y : c \\mid z]` in bits.
        """
        return _cmi(self.y, self.Ky, c, Kc, z, Kz)


class _KsgColumns:
    """
    The raw (continuous) values of one target's realizations, for KSG estimates.

    Same interface as :class:`_Columns`: columns are raw values (each process
    z-scored, since the KSG max-norm compares dimensions), joints are stacked
    columns, and :meth:`cmi` is the Frenzel--Pompe/KSG estimate :cite:`Runge2018`.
    """

    def __init__(self, data, target, variables, max_lag, k=4, prng=None):
        if data.raw is None:
            raise ValueError("estimator='ksg' needs the raw series; pass continuous data (not DiscreteData).")
        self.r = realizations(data, target, variables, max_lag=max_lag)
        self.variables = list(self.r.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        self.alphabet = dict.fromkeys(self.variables, 1)
        pooled = np.concatenate(data.raw)
        scale = pooled.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        raw = [(t - pooled.mean(axis=0)) / scale for t in data.raw]
        trial, time = self.r.trial, self.r.time
        self.y = np.array([raw[i][j, target] for i, j in zip(trial, time, strict=True)])
        self.Ky = 1
        self.n = len(self.y)
        self._values = np.array(
            [[raw[i][j - lag, p] for p, lag in self.variables] for i, j in zip(trial, time, strict=True)]
        ).reshape(self.n, len(self.variables))
        self.k = k
        self.rng = as_generator(prng)

    def column(self, v):
        return self._values[:, self.index[v]], 1

    def joint(self, variables):
        cols = [self.index[v] for v in variables]
        return self._values[:, cols], len(cols)

    def cmi(self, c, Kc, z, Kz):
        """
        KSG :math:`I[Y : c \\mid z]` in bits.
        """
        from dit.inference import total_correlation_ksg

        c = np.asarray(c, dtype=float).reshape(self.n, -1)
        z = np.asarray(z, dtype=float).reshape(self.n, -1)
        dc, dz = c.shape[1], z.shape[1]
        stacked = np.column_stack([c, self.y, z])
        crvs = list(range(dc + 1, dc + 1 + dz)) or None
        value = total_correlation_ksg(stacked, [list(range(dc)), [dc]], crvs, k=self.k, prng=self.rng)
        return max(float(value), 0.0)


def _permuter(columns, settings, rng):
    """
    A function returning a permutation of realization indices, per the surrogate scheme.

    ``'trials'`` swaps whole trials (time order within trials kept); ``'random'``,
    ``'circular'``, ``'block'``, and ``'local'`` permute realizations in time
    within each trial (as IDTxl's ``permute_in_time`` options).
    """
    trial = columns.r.trial
    trials = np.unique(trial)
    scheme = settings.permutation
    if scheme == "auto":
        sizes = {int(np.sum(trial == t)) for t in trials}
        enough = len(trials) >= 2 and factorial(min(len(trials), 20)) >= max(settings.n_perm_max_stat, 100)
        scheme = "trials" if enough and len(sizes) == 1 else "random"
    groups = [np.flatnonzero(trial == t) for t in trials]
    n = columns.n

    if scheme == "trials":
        if len({len(g) for g in groups}) != 1:
            raise ValueError("trial permutation needs equal-length trials")

        def perm():
            order = rng.permutation(len(groups))
            return np.concatenate([groups[k] for k in order])

        return perm

    def within(fn):
        def perm():
            out = np.empty(n, dtype=np.int64)
            for g in groups:
                out[g] = g[fn(len(g))]
            return out

        return perm

    if scheme == "random":
        return within(rng.permutation)
    if scheme == "circular":

        def circ(m):
            max_shift = settings.max_shift or m // 2
            return np.roll(np.arange(m), int(rng.integers(1, max(max_shift, 2))))

        return within(circ)
    if scheme == "block":

        def block(m):
            size = settings.block_size or max(int(np.sqrt(m)), 2)
            blocks = [np.arange(i, min(i + size, m)) for i in range(0, m, size)]
            return np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])

        return within(block)
    if scheme == "local":

        def local(m):
            width = settings.perm_range or 10
            idx = np.arange(m)
            for start in range(0, m, width):
                seg = idx[start : start + width]
                idx[start : start + width] = seg[rng.permutation(len(seg))]
            return idx

        return within(local)
    raise ValueError(f"Unknown permutation scheme {scheme!r}")


def _pvalue(null, observed, tol=1e-12):
    null = np.asarray(null)
    return float((1 + np.sum(null >= observed - tol)) / (1 + len(null)))


def _max_statistic(cols, candidates, cond, n_perm, perm):
    """
    The best candidate, its CMI, and the max-statistic p-value.
    """
    z, Kz = cols.joint(cond)
    obs = []
    for v in candidates:
        c, a = cols.column(v)
        obs.append(cols.cmi(c, a, z, Kz))
    best = int(np.argmax(obs))
    null = np.empty(n_perm)
    for s in range(n_perm):
        idx = perm()
        null[s] = max(cols.cmi(cols.column(v)[0][idx], cols.alphabet[v], z, Kz) for v in candidates)
    return candidates[best], obs[best], _pvalue(null, obs[best])


def _greedy(cols, candidates, cond, n_perm, alpha, perm):
    selected = []
    candidates = list(candidates)
    while candidates:
        best, _, p = _max_statistic(cols, candidates, cond + selected, n_perm, perm)
        if p > alpha:
            break
        selected.append(best)
        candidates.remove(best)
    return selected


def _pair_search(cols, candidates, cond, settings, perm):
    """
    The best pair of candidates (by joint CMI) if it passes a max statistic over pairs.
    """
    pool = sorted(candidates, key=lambda v: (v[1], v[0]))[: settings.max_pair_candidates]
    pairs = [(a, b) for a, b in combinations(pool, 2) if a[0] != b[0] or a[1] != b[1]]
    if not pairs:
        return None
    z, Kz = cols.joint(cond)
    obs = []
    codes = []
    for a, b in pairs:
        c, K = cols.joint([a, b])
        codes.append((c, K))
        obs.append(cols.cmi(c, K, z, Kz))
    best = int(np.argmax(obs))
    null = np.empty(settings.n_perm_pairs)
    for s in range(settings.n_perm_pairs):
        idx = perm()
        null[s] = max(cols.cmi(c[idx], K, z, Kz) for c, K in codes)
    if _pvalue(null, obs[best]) <= settings.alpha_pairs:
        return pairs[best]
    return None


def _individual(cols, variables, cond, v, idx=None):
    others = [u for u in variables if u != v]
    z, Kz = cols.joint(cond + others)
    c, a = cols.column(v)
    if idx is not None:
        c = c[idx]
    return cols.cmi(c, a, z, Kz)


def _prune(cols, sources, cond, n_perm, alpha, perm):
    sources = list(sources)
    while sources:
        obs = [_individual(cols, sources, cond, v) for v in sources]
        k = int(np.argmin(obs))
        null = np.empty(n_perm)
        for s in range(n_perm):
            idx = perm()
            null[s] = min(_individual(cols, sources, cond, v, idx) for v in sources)
        if _pvalue(null, obs[k]) <= alpha:
            break
        sources.pop(k)
    return sources


def _sequential(cols, sources, cond, n_perm, alpha, perm):
    obs = np.array([_individual(cols, sources, cond, v) for v in sources])
    order = np.argsort(-obs)
    null = np.empty((n_perm, len(sources)))
    for s in range(n_perm):
        idx = perm()
        null[s] = np.sort([_individual(cols, sources, cond, v, idx) for v in sources])[::-1]
    pvalues = {}
    failed = False
    for rank, k in enumerate(order):
        p = 1.0 if failed else _pvalue(null[:, rank], obs[k])
        if p > alpha:
            failed = True
        pvalues[sources[k]] = p
    return pvalues


def _max_budget(embeddings, P):
    if isinstance(embeddings, (list, tuple)):
        return max(int(e.max_lag) for e in embeddings)
    return int(embeddings.max_lag)


def _source_lags(data, embeddings, process, max_lag):
    from dataclasses import replace

    embedding = embeddings[process] if isinstance(embeddings, (list, tuple)) else embeddings
    return replace(embedding, max_lag=max(max_lag, embedding.max_lag)).lags(int(data.lag_step[process]))


def select_parents(data, target, embeddings=None, sources=None, settings=None, prng=None):
    """
    Select the target's past and its source parents.

    Parameters
    ----------
    data : DiscreteData
    target : int
    embeddings : Embedding or list of Embedding, None
        Per-process candidate grids (default: lags 1..3).
    sources : list of int, None
        Candidate source processes (default: all others).
    settings : SkeletonSettings, None
    prng : None, int, Generator

    Returns
    -------
    TargetSkeleton
    """
    settings = settings or SkeletonSettings()
    settings.check()
    rng = as_generator(prng)
    embeddings = embeddings or Embedding()
    P = data.n_processes
    sources = [p for p in range(P) if p != target] if sources is None else list(sources)
    target_cands = [(target, lag) for lag in candidate_lags(data, embeddings, target)]
    # Coupling delays can exceed a source's own memory, so source lags extend to
    # max_source_lag (IDTxl's max_lag_sources; default the larger of 5 and the largest
    # lag budget in the network), on each source's own grid.
    source_max = settings.max_source_lag or max(5, _max_budget(embeddings, P))
    source_cands = [(p, lag) for p in sources for lag in _source_lags(data, embeddings, p, source_max)]
    conditionals = [tuple(v) for v in settings.forced_conditionals]
    if settings.faes:
        conditionals += [(p, 0) for p in sources if (p, 0) not in conditionals]
    everything = list(dict.fromkeys(target_cands + source_cands + conditionals))
    max_lag = max(v[1] for v in everything) if everything else 0
    if settings.estimator == "ksg":
        cols = _KsgColumns(data, target, everything, max_lag, k=settings.ksg_k, prng=rng)
    else:
        cols = _Columns(data, target, everything, max_lag)
    result = TargetSkeleton(target=target, conditionals=conditionals, n_samples=cols.n)
    if cols.n < 10:
        return result
    perm = _permuter(cols, settings, rng)

    past = _greedy(cols, target_cands, conditionals, settings.n_perm_max_stat, settings.alpha_max_stat, perm)
    selected = []
    remaining = list(source_cands)
    for _ in range(4):
        new = _greedy(
            cols, remaining, conditionals + past + selected, settings.n_perm_max_stat, settings.alpha_max_stat, perm
        )
        selected += new
        remaining = [v for v in remaining if v not in selected]
        # The target's own past can act only jointly with a source (e.g.
        # y_t = x_{t-1} xor y_{t-1}): re-test unselected target lags given the sources.
        more = _greedy(
            cols,
            [v for v in target_cands if v not in past],
            conditionals + past + selected,
            settings.n_perm_max_stat,
            settings.alpha_max_stat,
            perm,
        )
        past += more
        if more:
            continue
        if not settings.synergy_search:
            break
        pool = remaining + [v for v in target_cands if v not in past]
        if len(pool) < 2:
            break
        pair = _pair_search(cols, pool, conditionals + past + selected, settings, perm)
        if pair is None:
            break
        for v in pair:
            (past if v[0] == target else selected).append(v)
        result.pairs.append(pair)
        remaining = [v for v in remaining if v not in pair]
    base = conditionals + past

    if selected:
        selected = _prune(cols, selected, base, settings.n_perm_min_stat, settings.alpha_min_stat, perm)
    result.target_past = past
    result.sources = selected
    if selected:
        s, Ks = cols.joint(selected)
        z, Kz = cols.joint(base)
        result.omnibus_te = cols.cmi(s, Ks, z, Kz)
        null = np.empty(settings.n_perm_omnibus)
        for k in range(settings.n_perm_omnibus):
            null[k] = cols.cmi(s[perm()], Ks, z, Kz)
        result.omnibus_pvalue = _pvalue(null, result.omnibus_te)
        result.source_pvalues = _sequential(cols, selected, base, settings.n_perm_max_seq, settings.alpha_max_seq, perm)
        result.sources = [v for v in selected if result.source_pvalues[v] <= settings.alpha_max_seq]
        result.significant = result.omnibus_pvalue <= settings.alpha_omnibus and bool(result.sources)

    if settings.tdmi_screen:
        parents = {p for p, _ in result.sources}
        for p in sources:
            if p in parents:
                continue
            cands = [v for v in source_cands if v[0] == p]
            best, value, pv = _max_statistic(cols, cands, [], settings.n_perm_tdmi, perm)
            result.tdmi_candidates[p] = {"variable": best, "tdmi": value, "pvalue": pv}
    return result


def infer_skeleton(data, embeddings=None, targets=None, settings=None, prng=None, map_fn=map, checkpoint=None):
    """
    Skeletons for all targets, with network-level FDR on the omnibus tests.

    Parameters
    ----------
    data : DiscreteData
    embeddings : Embedding or list of Embedding, None
    targets : list of int, None
        Targets to analyse (default: all processes).
    settings : SkeletonSettings, None
    prng : None, int, Generator
    map_fn : callable
        A ``map``-like function over targets (e.g. a dask or process-pool map).
    checkpoint : Checkpoint, None
        Reuse and store per-target results.

    Returns
    -------
    dict
        ``target -> TargetSkeleton``.
    """
    from .stats import benjamini_hochberg

    settings = settings or SkeletonSettings()
    rng = as_generator(prng)
    targets = list(range(data.n_processes)) if targets is None else list(targets)
    seeds = rng.integers(0, 2**32, size=len(targets))

    def run(args):
        target, seed = args

        def compute():
            return select_parents(data, target, embeddings, None, settings, int(seed))

        return compute() if checkpoint is None else checkpoint.cached(f"skeleton-{target}", compute)

    results = dict(zip(targets, map_fn(run, zip(targets, seeds)), strict=True))
    tested = [t for t in targets if results[t].sources]
    if settings.fdr and tested:
        reject, _ = benjamini_hochberg(
            [results[t].omnibus_pvalue for t in tested], settings.alpha_omnibus, dependent=settings.fdr_constant == 2
        )
        for t, ok in zip(tested, reject, strict=True):
            results[t].significant = bool(ok) and bool(results[t].sources)
    # TDMI candidates are corrected across all screened pairs.
    screened = [(t, p) for t in targets for p in results[t].tdmi_candidates]
    if screened:
        reject, _ = benjamini_hochberg(
            [results[t].tdmi_candidates[p]["pvalue"] for t, p in screened], settings.alpha_tdmi, dependent=False
        )
        for (t, p), ok in zip(screened, reject, strict=True):
            results[t].tdmi_candidates[p]["significant"] = bool(ok)
    return results
