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

import os
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
    KSG conditional mutual information (``ksg_k`` neighbours, ``ksg_threads``
    threads, default all cores) :cite:`Kraskov2004,Frenzel2007`, as IDTxl does,
    instead of the plug-in estimate on symbols. The target is then offered its own
    past up to ``max_source_lag`` (IDTxl's ``max_lag_target``), since KSG resolves
    memory beyond the symbolic lag budget. With ``ksg_null='local'`` (the default) the inclusion,
    pair, and omnibus nulls use Runge's local permutation in the conditioning set
    :cite:`Runge2018` instead of a free permutation: a candidate keeps its dependence
    on what is conditioned on, so a redundant proxy for the target's own past (such
    as a child of the target) is not mistaken for a parent.

    ``estimator='gaussian'`` selects parents with the linear-Gaussian conditional mutual
    information on the raw values (log-ratio of regression residual variances, equal to
    Granger causality for Gaussian processes :cite:`Barnett2009`), as IDTxl does for
    vector-autoregressive data :cite:`Novelli2019`: far more powerful than symbols or
    KSG when couplings are linear and weak, blind to nonlinear ones.

    ``estimator='trend'`` keeps symbols (``trend_bins`` equal-frequency bins) but tests
    a stratified ordinal trend: one degree of freedom per candidate, so weak monotone
    couplings are found with far fewer samples than the plug-in CMI, at the price of
    missing non-monotone ones. ``estimator='coarse'`` keeps the plug-in CMI on
    binary candidates and context bins that shrink with the conditioning set (about
    ``coarse_min_cell`` samples per cell), trading resolution for degrees of freedom.

    ``curtail`` (default: on for KSG and Gaussian) evaluates permutations ``curtail_batch`` at a time
    and stops a test as soon as its p-value must exceed the stage's level, which leaves
    every decision unchanged and saves most permutations in the many tests that fail.
    ``prescreen_alpha`` first tests each source process on its own (its best lag's
    transfer entropy given the selected target past, ``n_perm_prescreen`` permutations)
    and keeps only those passing for the multivariate search, at the risk of missing a
    source that is informative only jointly with another. The screened-out sources still
    enter the max-statistic nulls, which keeps the selection's error rate; the saving is
    in the search, not the nulls. Conditioning on a parent
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
    null: str = "free"
    statistic: str = "raw"
    adaptive_target_bins: int = 6
    adaptive_candidate_bins: int = 4
    trend_bins: int = 4
    coarse_min_cell: int = 10
    ksg_k: int = 4
    ksg_threads: int | None = None
    ksg_null: str = "local"
    curtail: bool | None = None
    curtail_batch: int = 20
    prescreen_alpha: float | None = None
    n_perm_prescreen: int = 100
    device: str | None = None

    def check(self):
        for name in ("max_stat", "min_stat", "omnibus", "max_seq"):
            check_n_perm(getattr(self, f"n_perm_{name}"), getattr(self, f"alpha_{name}"))
        if self.synergy_search:
            check_n_perm(self.n_perm_pairs, self.alpha_pairs)
        if self.tdmi_screen:
            check_n_perm(self.n_perm_tdmi, self.alpha_tdmi)
        if self.ksg_null not in ("permutation", "local"):
            raise ValueError("ksg_null must be 'permutation' or 'local'")
        if self.estimator not in ("plugin", "ksg", "adaptive", "gaussian", "trend", "coarse"):
            raise ValueError("estimator must be 'plugin', 'ksg', 'adaptive', 'gaussian', 'trend', or 'coarse'")
        if self.null not in ("free", "strata"):
            raise ValueError("null must be 'free' or 'strata'")
        if self.statistic not in ("raw", "debiased"):
            raise ValueError("statistic must be 'raw' or 'debiased'")
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
    prescreened: list | None = None
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
        self.map = map
        self.device = None

    def column(self, v):
        return self.r.values[:, self.index[v]], self.alphabet[v]

    def joint(self, variables):
        code = np.zeros(self.n, dtype=np.int64)
        K = 1
        for v in variables:
            c, a = self.column(v)
            code, K = _combine(code, K, c, a)
        return code, K

    def context(self, variables):
        """
        The joint code of a conditioning set (the same as :meth:`joint` here).
        """
        return self.joint(variables)

    def cmi(self, c, Kc, z, Kz):
        """
        Plug-in :math:`I[Y : c \\mid z]` in bits.
        """
        return _cmi(self.y, self.Ky, c, Kc, z, Kz)

    def cmi_batch(self, c, Kc, z, Kz, idx):
        """
        :meth:`cmi` of ``c[i]`` for every permutation ``i`` in `idx`, batched on :attr:`device` if set.
        """
        if self.device is not None:
            from .backend import plugin_cmi_batch

            return plugin_cmi_batch(self.y, self.Ky, np.asarray(c)[np.asarray(idx)], Kc, z, Kz, self.device)
        return np.array([self.cmi(c[i], Kc, z, Kz) for i in idx])


def _raw_columns(data, r, target):
    """
    Raw values of a realization set: the target's present and one column per variable.
    """
    if data.raw is None:
        raise ValueError("this estimator needs the raw series; pass continuous data (not DiscreteData).")
    trial, time = r.trial, r.time
    y = np.array([data.raw[i][j, target] for i, j in zip(trial, time, strict=True)], dtype=float)
    values = np.array(
        [[data.raw[i][j - lag, p] for p, lag in r.variables] for i, j in zip(trial, time, strict=True)], dtype=float
    ).reshape(len(y), len(r.variables))
    return y, values


def _ranks(v):
    from scipy.stats import rankdata

    return (rankdata(v) - 0.5) / len(v)


def _equal_frequency(ranks, bins):
    return np.minimum((ranks * bins).astype(np.int64), bins - 1)


def _residual_bits(E, ey, vy):
    """
    :math:`\\tfrac12 \\log_2` of the target's residual variance `vy` over what remains after
    regressing residuals `ey` on each residualized candidate block ``E`` (B, n, dx).
    """
    if E.shape[2] == 1:
        e = E[:, :, 0]
        num = e @ ey
        den = np.einsum("bn,bn->b", e, e)
        explained = np.where(den > 0, num**2 / np.where(den > 0, den, 1), 0.0)
    else:
        G = np.einsum("bni,bnj->bij", E, E)
        g = np.einsum("bni,n->bi", E, ey)
        ridge = 1e-12 * np.eye(G.shape[1])
        sol = np.linalg.solve(G + ridge, g[..., None])[..., 0]
        explained = np.einsum("bi,bi->b", g, sol)
    residual = np.maximum(vy - explained, 1e-300)
    return 0.5 * np.log2(vy / residual)


class _GaussianColumns:
    """
    The raw values of one target's realizations, for linear-Gaussian estimates.

    :math:`I[Y : X \\mid Z] = \\tfrac12 \\log_2 \\sigma^2(Y \\mid Z) / \\sigma^2(Y \\mid X, Z)`, the
    log-ratio of least-squares residual variances (with an intercept); for Gaussian
    processes transfer entropy equals Granger causality :cite:`Barnett2009`, and this
    is the estimator :cite:`Novelli2019` used for vector-autoregressive networks. Each
    conditioning set's orthonormal basis is cached, and a batch of permuted candidates
    is residualized with one matrix product.
    """

    def __init__(self, data, target, variables, max_lag):
        self.r = realizations(data, target, variables, max_lag=max_lag)
        self.variables = list(self.r.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        self.alphabet = dict.fromkeys(self.variables, 1)
        y, values = _raw_columns(data, self.r, target)
        self.y = (y - y.mean()) / (y.std() or 1.0)
        scale = values.std(axis=0)
        self._values = (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
        self.Ky = 1
        self.n = len(self.y)
        self.map = map
        self.device = None
        self._bases = {}

    def column(self, v):
        return self._values[:, self.index[v]], 1

    def joint(self, variables):
        cols = [self.index[v] for v in variables]
        return self._values[:, cols], len(cols)

    def context(self, variables):
        return self.joint(variables)

    def _basis(self, z):
        z = np.asarray(z, dtype=float).reshape(self.n, -1)
        key = (z.shape, hash(z.tobytes()))
        if key not in self._bases:
            q, _ = np.linalg.qr(np.column_stack([np.ones(self.n), z]))
            ey = self.y - q @ (q.T @ self.y)
            self._bases[key] = (q, ey, float(ey @ ey))
            if len(self._bases) > 64:
                self._bases.pop(next(iter(self._bases)))
        return self._bases[key]

    def _values_for(self, X, q, ey, vy):
        """
        CMI (bits) for candidate blocks ``X`` of shape (B, n, dx).
        """
        E = X - np.einsum("nk,bk...->bn...", q, np.einsum("nk,bn...->bk...", q, X))
        return _residual_bits(E, ey, vy)

    def cmi(self, c, Kc, z, Kz):
        """
        Linear-Gaussian :math:`I[Y : c \\mid z]` in bits.
        """
        q, ey, vy = self._basis(z)
        X = np.asarray(c, dtype=float).reshape(1, self.n, -1)
        return float(self._values_for(X, q, ey, vy)[0])

    def cmi_batch(self, c, Kc, z, Kz, idx):
        q, ey, vy = self._basis(z)
        c = np.asarray(c, dtype=float).reshape(self.n, -1)
        return self._values_for(c[np.asarray(idx)], q, ey, vy)


class _AdaptiveColumns(_Columns):
    """
    Symbols for adaptive conditioning :cite:`Kontkanen2007,Marx2021`.

    The target's present and every candidate are cut into fixed equal-frequency bins of
    their ranks; a conditioning set is cut by a greedy MDL joint histogram learned
    together with the (fixed) target bins (:func:`~infoflow.adaptive.joint_histogram`),
    so it is resolved finely exactly where it changes the target. No partition depends
    on how a candidate is paired with the target, so permuting candidates gives a valid
    null. Partitions are cached per conditioning set.
    """

    def __init__(self, data, target, variables, max_lag, target_bins=6, candidate_bins=4):
        self.r = realizations(data, target, variables, max_lag=max_lag)
        self.variables = list(self.r.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        y, values = _raw_columns(data, self.r, target)
        self._y_ranks = _ranks(y)
        self._ranked = (
            np.column_stack([_ranks(values[:, i]) for i in range(values.shape[1])]) if values.size else values
        )
        self.y, self.Ky = _equal_frequency(self._y_ranks, target_bins), target_bins
        self._codes = _equal_frequency(self._ranked, candidate_bins) if values.size else values.astype(np.int64)
        self.alphabet = dict.fromkeys(self.variables, candidate_bins)
        self.target_bins = target_bins
        self.n = len(self.y)
        self.map = map
        self.device = None
        self._contexts = {}
        self._cuts = {}

    def column(self, v):
        return self._codes[:, self.index[v]], self.alphabet[v]

    def context(self, variables):
        """
        The adaptive joint code of a conditioning set, warm-started from the cached
        partition sharing the most variables (or built up one variable at a time).
        """
        from .adaptive import codes, joint_histogram
        from .measures import dense_codes

        if not variables:
            return np.zeros(self.n, dtype=np.int64), 1
        key = tuple(variables)
        if key not in self._contexts:
            known = {}
            best = max(self._cuts, key=lambda k: len(set(k) & set(key)), default=None)
            if best is not None and set(best) & set(key):
                known = self._cuts[best]
            elif len(key) > 2:
                self.context(list(key[:-1]))
                known = self._cuts[key[:-1]]
            columns = [self._ranked[:, self.index[v]] for v in variables]
            initial = [None] + [known.get(v) for v in variables] if known else None
            cuts = joint_histogram([self._y_ranks, *columns], fixed={0: self.target_bins}, initial=initial)
            self._cuts[key] = dict(zip(variables, cuts[1:], strict=True))
            binned = np.stack([codes(c, k) for c, k in zip(columns, cuts[1:], strict=True)], axis=1)
            self._contexts[key] = dense_codes(binned)
        return self._contexts[key]


class _RankColumns(_Columns):
    """
    Equal-frequency bins of the ranks of the raw values (shared by the trend and
    coarse estimators): `target_bins` for the target's present, `source_bins` for
    every candidate.
    """

    def __init__(self, data, target, variables, max_lag, target_bins, source_bins):
        self.r = realizations(data, target, variables, max_lag=max_lag)
        self.variables = list(self.r.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        y, values = _raw_columns(data, self.r, target)
        self._ranked = (
            np.column_stack([_ranks(values[:, i]) for i in range(values.shape[1])]) if values.size else values
        )
        self.y, self.Ky = _equal_frequency(_ranks(y), target_bins), target_bins
        self._codes = _equal_frequency(self._ranked, source_bins) if values.size else values.astype(np.int64)
        self.alphabet = dict.fromkeys(self.variables, source_bins)
        self.n = len(self.y)
        self.map = map
        self.device = None
        self._contexts = {}

    def column(self, v):
        return self._codes[:, self.index[v]], self.alphabet[v]

    def _binned_context(self, variables, bins):
        key = (tuple(variables), bins)
        if key not in self._contexts:
            code, K = np.zeros(self.n, dtype=np.int64), 1
            for v in variables:
                code, K = _combine(code, K, _equal_frequency(self._ranked[:, self.index[v]], bins), bins)
            self._contexts[key] = _densify(code)
        return self._contexts[key]


class _CoarseColumns(_RankColumns):
    """
    Plug-in CMI on coarse symbols: binary (median-split) candidates, and context bins
    per variable that shrink as the conditioning set grows, so that a cell of
    (target, candidate, context) keeps about `min_cell` expected samples.

    A plug-in test has :math:`(K_x - 1)(K_y - 1) K_z` degrees of freedom; splitting a
    linearly coupled Gaussian source at its median keeps :math:`2/\\pi` of its squared
    correlation with the target while dividing that by :math:`K_x - 1`. The bins depend
    only on the sample size and the size of the conditioning set, never on the data,
    so permutation nulls stay valid.
    """

    def __init__(self, data, target, variables, max_lag, target_bins=4, source_bins=2, min_cell=10, max_bins=4):
        super().__init__(data, target, variables, max_lag, target_bins, source_bins)
        self.min_cell = min_cell
        self.max_bins = max_bins
        self._source_bins = source_bins

    def context_bins(self, d):
        budget = self.n / (self.min_cell * self.Ky * self._source_bins)
        return int(np.clip(np.floor(budget ** (1 / d)), 2, self.max_bins)) if d else 1

    def context(self, variables):
        if not variables:
            return np.zeros(self.n, dtype=np.int64), 1
        return self._binned_context(variables, self.context_bins(len(variables)))


class _TrendColumns(_RankColumns):
    """
    A stratified ordinal-trend (linear-by-linear association) statistic on symbols.

    The target's present and the candidates are scored by their equal-frequency bin
    (`bins` levels); the conditioning set is cut into the same bins and used as strata.
    Within every stratum the scores are centered, and the statistic is the
    linear-Gaussian CMI of the centered scores,
    :math:`\\tfrac12 \\log_2 \\sigma^2(Y \\mid Z) / \\sigma^2(Y \\mid X, Z)`, i.e. a regression of
    the target's score on the candidate's with one intercept per stratum. A single
    candidate costs one degree of freedom instead of the plug-in's
    :math:`(K_x - 1)(K_y - 1) K_z`, so monotone couplings are detected with far fewer
    samples; non-monotone ones (e.g. XOR) are invisible to it. Generalizes the
    Cochran-Mantel-Haenszel test for ordered tables :cite:`Agresti2013`.
    """

    def __init__(self, data, target, variables, max_lag, bins=4):
        super().__init__(data, target, variables, max_lag, bins, bins)
        self.bins = bins
        self._yscore = self.y.astype(float)
        self._scores = self._codes.astype(float)
        self._strata = {}

    def column(self, v):
        return self._scores[:, self.index[v]], 1

    def joint(self, variables):
        cols = [self.index[v] for v in variables]
        return self._scores[:, cols], len(cols)

    def context(self, variables):
        if not variables:
            return np.zeros(self.n, dtype=np.int64), 1
        return self._binned_context(variables, self.bins)

    def _stratum(self, z, Kz):
        from scipy.sparse import csr_matrix

        z = np.asarray(z, dtype=np.int64)
        key = (Kz, hash(z.tobytes()))
        if key not in self._strata:
            S = csr_matrix((np.ones(self.n), (np.arange(self.n), z)), shape=(self.n, Kz))
            counts = np.maximum(np.bincount(z, minlength=Kz), 1).astype(float)
            ey = self._yscore - (S @ ((S.T @ self._yscore) / counts))
            self._strata[key] = (S, counts, ey, float(ey @ ey))
            if len(self._strata) > 64:
                self._strata.pop(next(iter(self._strata)))
        return self._strata[key]

    def _values(self, X, z, Kz):
        S, counts, ey, vy = self._stratum(z, Kz)
        B, n, d = X.shape
        flat = X.transpose(1, 0, 2).reshape(n, B * d)
        centered = flat - S @ ((S.T @ flat) / counts[:, None])
        return _residual_bits(centered.reshape(n, B, d).transpose(1, 0, 2), ey, vy)

    def cmi(self, c, Kc, z, Kz):
        return float(self._values(np.asarray(c, dtype=float).reshape(1, self.n, -1), z, Kz)[0])

    def cmi_batch(self, c, Kc, z, Kz, idx):
        c = np.asarray(c, dtype=float).reshape(self.n, -1)
        return self._values(c[np.asarray(idx)], z, Kz)


class _KsgColumns:
    """
    The raw (continuous) values of one target's realizations, for KSG estimates.

    Same interface as :class:`_Columns`: columns are raw values (each process
    z-scored, since the max-norm compares dimensions), joints are stacked columns,
    and :meth:`cmi` is the Frenzel--Pompe estimate :cite:`Kraskov2004,Frenzel2007`
    with neighbours counted strictly inside each point's radius. The trees over
    (target, context) and over the context are built once per conditioning set and
    reused for every candidate and permutation, and :attr:`map` evaluates candidates
    and permutations on a thread pool (the tree queries release the GIL).
    """

    def __init__(self, data, target, variables, max_lag, k=4, threads=None, prng=None):
        if data.raw is None:
            raise ValueError("estimator='ksg' needs the raw series; pass continuous data (not DiscreteData).")
        self.r = realizations(data, target, variables, max_lag=max_lag)
        self.variables = list(self.r.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        self.alphabet = dict.fromkeys(self.variables, 1)
        pooled = np.concatenate(data.raw)
        scale = pooled.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        rng = as_generator(prng)
        raw = [(t - pooled.mean(axis=0)) / scale for t in data.raw]
        raw = [t + 1e-10 * rng.normal(size=t.shape) for t in raw]  # break ties (KSG assumes none)
        trial, time = self.r.trial, self.r.time
        self.y = np.array([raw[i][j, target] for i, j in zip(trial, time, strict=True)])
        self.Ky = 1
        self.n = len(self.y)
        self._values = np.array(
            [[raw[i][j - lag, p] for p, lag in self.variables] for i, j in zip(trial, time, strict=True)]
        ).reshape(self.n, len(self.variables))
        self.k = k
        self.device = None
        self._contexts = {}
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(threads or os.cpu_count() or 1)
        self.map = self._pool.map

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    def column(self, v):
        return self._values[:, self.index[v]], 1

    def cmi_batch(self, c, Kc, z, Kz, idx):
        """
        :meth:`cmi` of ``c[i]`` for every permutation ``i`` in `idx`: on :attr:`device`
        by brute force if set, otherwise with the cached trees on the thread pool.
        """
        if self.device is not None:
            from .backend import ksg_cmi_batch

            c = np.asarray(c, dtype=float).reshape(self.n, -1)
            z = np.asarray(z, dtype=float).reshape(self.n, -1)
            return ksg_cmi_batch(c[np.asarray(idx)], self.y, z, self.k, self.device)
        return np.array(list(self.map(lambda i: self.cmi(c[i], Kc, z, Kz), idx)))

    def local_permutation(self, z, rng, k_perm=5):
        """
        Indices of Runge's local permutation in `z` :cite:`Runge2018`: each sample takes
        a distinct sample among its `k_perm` nearest neighbours in the conditioning set,
        so a candidate keeps its dependence on the conditioning set under the null.
        """
        from scipy.spatial import KDTree

        z = np.asarray(z, dtype=float).reshape(self.n, -1)
        if z.shape[1] == 0:
            return rng.permutation(self.n)
        key = ("neighbors", z.shape, hash(z.tobytes()))
        if key not in self._contexts:
            self._contexts[key] = KDTree(z).query(z, k_perm, p=np.inf, workers=-1)[1].reshape(self.n, -1)
        neighbors = self._contexts[key]
        n, k = neighbors.shape
        # Each sample proposes its neighbours in a random order; a neighbour proposed by
        # several samples goes to the one with the highest random priority, and samples
        # left without a free neighbour after k rounds reuse their first proposal.
        order = np.argsort(rng.random((n, k)), axis=1)
        shuffled = np.take_along_axis(neighbors, order, axis=1)
        priority = rng.random(n)
        choice = np.full(n, -1, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        for r in range(k):
            pending = np.flatnonzero(choice < 0)
            if not len(pending):
                break
            proposal = shuffled[pending, r]
            free = ~used[proposal]
            pending, proposal = pending[free], proposal[free]
            ranked = np.lexsort((-priority[pending], proposal))
            first = np.ones(len(ranked), dtype=bool)
            first[1:] = proposal[ranked][1:] != proposal[ranked][:-1]
            winners = ranked[first]
            choice[pending[winners]] = proposal[winners]
            used[proposal[winners]] = True
        left = choice < 0
        choice[left] = shuffled[left, 0]
        return choice

    def joint(self, variables):
        cols = [self.index[v] for v in variables]
        return self._values[:, cols], len(cols)

    def context(self, variables):
        return self.joint(variables)

    def _context(self, z):
        from scipy.spatial import KDTree

        key = (z.shape, hash(z.tobytes()))
        if key not in self._contexts:
            yz = np.column_stack([self.y, z])
            self._contexts[key] = (yz, KDTree(yz), KDTree(z) if z.shape[1] else None)
            if len(self._contexts) > 64:
                self._contexts.pop(next(iter(self._contexts)))
        return self._contexts[key]

    def cmi(self, c, Kc, z, Kz):
        """
        KSG :math:`I[Y : c \\mid z]` in bits.
        """
        from scipy.spatial import KDTree
        from scipy.special import digamma

        c = np.asarray(c, dtype=float).reshape(self.n, -1)
        z = np.asarray(z, dtype=float).reshape(self.n, -1)
        yz, yz_tree, z_tree = self._context(z)
        k = self.k
        xyz = np.column_stack([c, yz])
        radius = np.nextafter(KDTree(xyz).query(xyz, k + 1, p=np.inf)[0][:, -1], 0)

        def count(tree, points):
            return tree.query_ball_point(points, radius, p=np.inf, return_length=True)

        xz = np.column_stack([c, z])
        n_xz, n_yz = count(KDTree(xz), xz), count(yz_tree, yz)
        if z_tree is None:
            value = digamma(k) + digamma(self.n) - np.mean(digamma(n_xz) + digamma(n_yz))
        else:
            value = digamma(k) - np.mean(digamma(n_xz) + digamma(n_yz) - digamma(count(z_tree, z)))
        # Not clipped at zero: KSG is biased downward in high dimensions, and clipping
        # would tie weak true effects with the many null values clipped to zero.
        return float(value) / np.log(2)


def _base_permuter(columns, settings, rng):
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


def _permuter(columns, settings, rng):
    """
    A permuter that accepts (and ignores) the conditioning set, like the local one.
    """
    base = _base_permuter(columns, settings, rng)

    def perm(z=None):
        return base()

    return perm


def _centers(nulls, debiased):
    """
    Per-candidate null means (for bias-corrected statistics), or zeros.
    """
    return nulls.mean(axis=1) if debiased else np.zeros(len(nulls))


def _null_statistics(n_perm, draw, evaluate, observed, alpha=None, batch=None):
    """
    Null values of a statistic, with optional curtailment.

    `evaluate` maps a list of permutation index arrays to one null value per
    permutation; `draw` returns the next permutation. With `batch` (and `alpha`), the
    permutations are evaluated `batch` at a time and stopped as soon as enough null
    values reach `observed` that the p-value over all `n_perm` permutations must exceed
    `alpha` whatever the rest show: the decision is the same as with every permutation
    (a curtailed test), and the p-value reported from the evaluated ones is conservative.
    """
    if batch is None or alpha is None:
        return np.asarray(evaluate([draw() for _ in range(n_perm)]))
    limit = int(np.floor(alpha * (n_perm + 1) - 1)) + 1
    values, exceed = [], 0
    while len(values) < n_perm:
        chunk = np.asarray(evaluate([draw() for _ in range(min(batch, n_perm - len(values)))]))
        values.extend(chunk)
        exceed += int(np.sum(chunk >= observed - 1e-12))
        if exceed >= limit:
            break
    return np.asarray(values)


def _max_statistic(cols, candidates, cond, n_perm, perm, debiased=False, alpha=None, batch=None, screened=()):
    """
    The best candidate, its CMI, and the max-statistic p-value.

    `screened` candidates are not searched but still enter the null maximum: when a
    data-driven screen removed them, a null over the survivors alone ignores that they
    survived for having large statistics, and the test is anti-conservative.

    With `debiased`, candidates are ranked by their CMI minus the mean of their own
    null, and the null maximum is taken over equally centered values: plug-in biases
    differ between candidates (they depend on how each relates to the conditioning
    set), so the raw maximum can favour the most biased candidate over the most
    informative one. With `batch`, raw tests are curtailed at `alpha`
    (:func:`_null_statistics`).
    """
    z, Kz = cols.context(cond)
    family = list(candidates) + [v for v in screened if v not in candidates]
    obs = np.array(list(cols.map(lambda v: cols.cmi(cols.column(v)[0], cols.alphabet[v], z, Kz), candidates)))
    if debiased:
        idx = [perm(z) for _ in range(n_perm)]
        nulls = np.array([cols.cmi_batch(cols.column(v)[0], cols.alphabet[v], z, Kz, idx) for v in family])
        shift = _centers(nulls, True)
        best = int(np.argmax(obs - shift[: len(obs)]))
        null = (nulls - shift[:, None]).max(axis=0)
        return candidates[best], float(obs[best]), _pvalue(null, obs[best] - shift[best])
    best = int(np.argmax(obs))

    def evaluate(idx):
        return np.max([cols.cmi_batch(cols.column(v)[0], cols.alphabet[v], z, Kz, idx) for v in family], axis=0)

    null = _null_statistics(n_perm, lambda: perm(z), evaluate, obs[best], alpha, batch)
    return candidates[best], float(obs[best]), _pvalue(null, obs[best])


def _greedy(cols, candidates, cond, n_perm, alpha, perm, debiased=False, batch=None, screened=()):
    selected = []
    candidates = list(candidates)
    while candidates:
        best, _, p = _max_statistic(cols, candidates, cond + selected, n_perm, perm, debiased, alpha, batch, screened)
        if p > alpha:
            break
        selected.append(best)
        candidates.remove(best)
    return selected


def _pair_search(cols, candidates, cond, settings, perm, batch=None):
    """
    The best pair of candidates (by joint CMI) if it passes a max statistic over pairs.
    """
    debiased = settings.statistic == "debiased"
    pool = sorted(candidates, key=lambda v: (v[1], v[0]))[: settings.max_pair_candidates]
    pairs = [(a, b) for a, b in combinations(pool, 2) if a[0] != b[0] or a[1] != b[1]]
    if not pairs:
        return None
    z, Kz = cols.context(cond)
    codes = [cols.joint([a, b]) for a, b in pairs]
    obs = np.array(list(cols.map(lambda code: cols.cmi(code[0], code[1], z, Kz), codes)))
    if debiased:
        idx = [perm(z) for _ in range(settings.n_perm_pairs)]
        nulls = np.array([cols.cmi_batch(c, K, z, Kz, idx) for c, K in codes])
        shift = _centers(nulls, True)
        best = int(np.argmax(obs - shift))
        null = (nulls - shift[:, None]).max(axis=0)
        observed = obs[best] - shift[best]
    else:
        best = int(np.argmax(obs))
        observed = obs[best]

        def evaluate(idx):
            return np.max([cols.cmi_batch(c, K, z, Kz, idx) for c, K in codes], axis=0)

        null = _null_statistics(settings.n_perm_pairs, lambda: perm(z), evaluate, observed, settings.alpha_pairs, batch)
    if _pvalue(null, observed) <= settings.alpha_pairs:
        return pairs[best]
    return None


def _individual(cols, variables, cond, v, idx=None, perm=None, n_perm=0):
    """
    :math:`I[Y : v \\mid \\text{cond}, \\text{others}]`, or its values under permutations:
    the shared indices `idx`, or `n_perm` fresh draws of `perm` given this conditioning set.
    """
    others = [u for u in variables if u != v]
    z, Kz = cols.context(cond + others)
    c, a = cols.column(v)
    if perm is not None:
        idx = [perm(z) for _ in range(n_perm)]
    if idx is not None:
        return cols.cmi_batch(c, a, z, Kz, idx)
    return cols.cmi(c, a, z, Kz)


def _per_source_nulls(cols, sources, cond, n_perm, perm):
    """
    Null CMIs ``(n_perm, len(sources))``: one permutation shared by all sources, or, for
    conditioning-aware permutations, draws given each source's own conditioning set.
    """
    if getattr(perm, "uses_condition", False):
        return np.stack([_individual(cols, sources, cond, v, perm=perm, n_perm=n_perm) for v in sources], axis=1)
    idx = [perm() for _ in range(n_perm)]
    return np.stack([_individual(cols, sources, cond, v, idx) for v in sources], axis=1)


def _can_curtail(perm, debiased, batch):
    return batch is not None and not debiased and not getattr(perm, "uses_condition", False)


def _prune(cols, sources, cond, n_perm, alpha, perm, debiased=False, batch=None):
    sources = list(sources)
    while sources:
        obs = np.array(list(cols.map(lambda v, sources=sources: _individual(cols, sources, cond, v), sources)))
        if _can_curtail(perm, debiased, batch):
            k = int(np.argmin(obs))

            def evaluate(idx, sources=sources):
                return np.min([_individual(cols, sources, cond, v, idx) for v in sources], axis=0)

            null = _null_statistics(n_perm, perm, evaluate, obs[k], alpha, batch)
            observed = obs[k]
        else:
            nulls = _per_source_nulls(cols, sources, cond, n_perm, perm)
            shift = _centers(nulls.T, debiased)
            k = int(np.argmin(obs - shift))
            null = (nulls - shift).min(axis=1)
            observed = obs[k] - shift[k]
        if _pvalue(null, observed) <= alpha:
            break
        sources.pop(k)
    return sources


def _sequential(cols, sources, cond, n_perm, alpha, perm, debiased=False, batch=None):
    obs = np.array(list(cols.map(lambda v: _individual(cols, sources, cond, v), sources)))
    if _can_curtail(perm, debiased, batch):
        # Curtail on the first (largest) rank: if it fails, every source fails.
        rows = []

        def evaluate(idx):
            block = np.stack([_individual(cols, sources, cond, v, idx) for v in sources], axis=1)
            rows.append(block)
            return block.max(axis=1)

        top = _null_statistics(n_perm, perm, evaluate, obs.max(), alpha, batch)
        nulls = np.vstack(rows)
        if _pvalue(top, obs.max()) > alpha:
            return {v: (_pvalue(top, obs.max()) if i == int(np.argmax(obs)) else 1.0) for i, v in enumerate(sources)}
        shift = np.zeros(len(sources))
    else:
        nulls = _per_source_nulls(cols, sources, cond, n_perm, perm)
        shift = _centers(nulls.T, debiased)
    obs = obs - shift
    order = np.argsort(-obs)
    null = np.sort(nulls - shift, axis=1)[:, ::-1]
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
    # Coupling delays can exceed a source's own memory, so source lags extend to
    # max_source_lag (IDTxl's max_lag_sources; default the larger of 5 and the largest
    # lag budget in the network), on each source's own grid.
    source_max = settings.max_source_lag or max(5, _max_budget(embeddings, P))
    if settings.estimator in ("ksg", "gaussian"):
        # KSG resolves the target's own past finely enough that memory beyond the symbolic
        # lag budget matters; offer the target the source lag range (IDTxl's max_lag_target).
        target_cands = [(target, lag) for lag in _source_lags(data, embeddings, target, source_max)]
    else:
        target_cands = [(target, lag) for lag in candidate_lags(data, embeddings, target)]
    source_cands = [(p, lag) for p in sources for lag in _source_lags(data, embeddings, p, source_max)]
    conditionals = [tuple(v) for v in settings.forced_conditionals]
    if settings.faes:
        conditionals += [(p, 0) for p in sources if (p, 0) not in conditionals]
    everything = list(dict.fromkeys(target_cands + source_cands + conditionals))
    max_lag = max(v[1] for v in everything) if everything else 0
    if settings.estimator == "ksg":
        cols = _KsgColumns(data, target, everything, max_lag, k=settings.ksg_k, threads=settings.ksg_threads, prng=rng)
    elif settings.estimator == "gaussian":
        cols = _GaussianColumns(data, target, everything, max_lag)
    elif settings.estimator == "adaptive":
        cols = _AdaptiveColumns(
            data, target, everything, max_lag, settings.adaptive_target_bins, settings.adaptive_candidate_bins
        )
    elif settings.estimator == "trend":
        cols = _TrendColumns(data, target, everything, max_lag, settings.trend_bins)
    elif settings.estimator == "coarse":
        cols = _CoarseColumns(data, target, everything, max_lag, min_cell=settings.coarse_min_cell)
    else:
        cols = _Columns(data, target, everything, max_lag)
    if settings.device is not None:
        from .backend import resolve_device

        cols.device = resolve_device(settings.device)
    result = TargetSkeleton(target=target, conditionals=conditionals, n_samples=cols.n)
    if cols.n < 10:
        return result
    perm = _permuter(cols, settings, rng)
    if isinstance(cols, _KsgColumns) and settings.ksg_null == "local":
        free, ksg = perm, cols

        def perm(z=None):
            return ksg.local_permutation(z, rng) if z is not None else free()

    elif not isinstance(cols, _KsgColumns) and settings.null == "strata":
        from .stats import _within_strata

        free_perm = perm

        def perm(z=None):
            return _within_strata(np.arange(cols.n), z, rng) if z is not None else free_perm()

        setattr(perm, "uses_condition", True)  # noqa: B010

    deb = settings.statistic == "debiased"
    curtail = settings.curtail if settings.curtail is not None else settings.estimator in ("ksg", "gaussian")
    bt = settings.curtail_batch if curtail else None
    past = _greedy(cols, target_cands, conditionals, settings.n_perm_max_stat, settings.alpha_max_stat, perm, deb, bt)
    all_source_cands = list(source_cands)
    if settings.prescreen_alpha is not None:
        kept = []
        for p in sources:
            lags = [v for v in source_cands if v[0] == p]
            _, _, pv = _max_statistic(
                cols,
                lags,
                conditionals + past,
                settings.n_perm_prescreen,
                perm,
                deb,
                settings.prescreen_alpha,
                settings.curtail_batch,
            )
            if pv <= settings.prescreen_alpha:
                kept.append(p)
        source_cands = [v for v in source_cands if v[0] in kept]
        result.prescreened = kept
    screened = [v for v in all_source_cands if v not in source_cands]
    selected = []
    remaining = list(source_cands)
    for _ in range(4):
        new = _greedy(
            cols,
            remaining,
            conditionals + past + selected,
            settings.n_perm_max_stat,
            settings.alpha_max_stat,
            perm,
            deb,
            bt,
            screened,
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
            deb,
            bt,
        )
        past += more
        if more:
            continue
        if not settings.synergy_search:
            break
        pool = remaining + screened + [v for v in target_cands if v not in past]
        if len(pool) < 2:
            break
        pair = _pair_search(cols, pool, conditionals + past + selected, settings, perm, bt)
        if pair is None:
            break
        for v in pair:
            (past if v[0] == target else selected).append(v)
        result.pairs.append(pair)
        remaining = [v for v in remaining if v not in pair]
        screened = [v for v in screened if v not in pair]
    base = conditionals + past

    if selected:
        selected = _prune(cols, selected, base, settings.n_perm_min_stat, settings.alpha_min_stat, perm, deb, bt)
    result.target_past = past
    result.sources = selected
    if selected:
        s, Ks = cols.joint(selected)
        z, Kz = cols.context(base)
        result.omnibus_te = cols.cmi(s, Ks, z, Kz)
        null = _null_statistics(
            settings.n_perm_omnibus,
            lambda: perm(z),
            lambda idx: cols.cmi_batch(s, Ks, z, Kz, idx),
            result.omnibus_te,
            settings.alpha_omnibus,
            bt,
        )
        result.omnibus_pvalue = _pvalue(null, result.omnibus_te)
        result.source_pvalues = _sequential(
            cols, selected, base, settings.n_perm_max_seq, settings.alpha_max_seq, perm, deb, bt
        )
        result.sources = [v for v in selected if result.source_pvalues[v] <= settings.alpha_max_seq]
        result.significant = result.omnibus_pvalue <= settings.alpha_omnibus and bool(result.sources)

    if settings.tdmi_screen:
        parents = {p for p, _ in result.sources}
        for p in sources:
            if p in parents:
                continue
            cands = [v for v in all_source_cands if v[0] == p]
            best, value, pv = _max_statistic(cols, cands, [], settings.n_perm_tdmi, perm, deb, settings.alpha_tdmi, bt)
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
