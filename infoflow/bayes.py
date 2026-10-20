"""
Hierarchical Bayesian estimation of the flow layers of one edge.

Each context value's conditional law :math:`\\theta_w = p(y, x \\mid w)` gets a shared
hierarchical Dirichlet prior :cite:`MacKay1995,Gelman2013`,

.. math::

    \\theta_w \\sim \\mathrm{Dirichlet}(\\alpha m), \\qquad
    m \\sim \\mathrm{Dirichlet}(1), \\qquad
    \\log \\alpha \\sim N(\\log K, 2^2),

with :math:`K = |Y||X|`. Integrating out :math:`\\theta` leaves a
Dirichlet-multinomial likelihood for each context's counts, so PyMC samples only
:math:`(\\alpha, m)`; given those, :math:`\\theta_w` is conjugate,
:math:`\\mathrm{Dirichlet}(\\alpha m + C_{\\cdot\\cdot w})`, and is drawn in NumPy along
with :math:`p(w) \\sim \\mathrm{Dirichlet}(C_{\\cdot\\cdot w} + 1/2)`. Sparse contexts are
shrunk toward the pooled law :math:`m` (toward "the context does not matter") by an
amount the data choose through :math:`\\alpha`: this is a soft version of
:func:`~infoflow.context.merge_contexts_statistical`.

Every posterior joint is decomposed by
:func:`~infoflow.measures.cmi_from_joint` and
:func:`~infoflow.measures.intrinsic_flow`, so
``te = intrinsic + synergistic`` and ``tdmi = intrinsic + shared`` hold on every draw.

The likelihood treats the aligned :math:`(y, x)` given :math:`w` as independent
across time, the same approximation as the multinomial intrinsic null of
:mod:`infoflow.layers`. Posterior draws of a mutual information are biased upward
like the plug-in estimate :cite:`Archer2013`, so posterior mass near zero is not a
test of a layer; use the nulls of :func:`~infoflow.layers.layer_statistics` for that.

Install with ``pip install infoflow[bayes]``.
"""

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator
from scipy.special import gammaln

from .measures import EdgeFlow, cmi_from_joint, dense_codes, intrinsic_flow, joint_table, layers_from_values

__all__ = (
    "ContextPrior",
    "LayerPosterior",
    "fit_context_prior",
    "layer_posterior",
    "log_bayes_factor",
    "posterior_tables",
)

QUANTITIES = ("te", "tdmi", "intrinsic", "synergistic", "shared")


def _pymc():
    try:
        import pymc
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Bayesian layers need the optional dependency pymc (pip install infoflow[bayes])") from error
    return pymc


@dataclass(frozen=True)
class ContextPrior:
    """
    Posterior samples of the hierarchical prior, from :func:`fit_context_prior`.

    Attributes
    ----------
    alpha : np.ndarray
        Concentrations, shape (S, G) for G mixture components.
    m : np.ndarray
        Pooled laws over the flattened ``(y, x)`` cells, shape (S, G, Ky * Kx).
    weights : np.ndarray
        Mixture weights, shape (S, G).
    """

    alpha: np.ndarray
    m: np.ndarray
    weights: np.ndarray


def _dm_logp(columns, a, gammaln):
    """
    Dirichlet-multinomial log-likelihood (without the multinomial coefficient) of
    each row of `columns` (W, K) under each row of `a` (G, K): shape (W, G).
    """
    n = columns.sum(axis=-1)[:, None]
    A = a.sum(axis=-1)[None, :]
    return (
        gammaln(A)
        - gammaln(n + A)
        + gammaln(columns[:, None, :] + a[None, :, :]).sum(axis=-1)
        - gammaln(a).sum(axis=-1)[None, :]
    )


def fit_context_prior(
    counts,
    components=1,
    draws=500,
    tune=500,
    chains=2,
    log_alpha_sd=2.0,
    method="nuts",
    nuts_sampler="pymc",
    prng=None,
):
    """
    Sample the posterior of the hierarchical prior given a count table.

    With ``components = G > 1`` the contexts' laws are a mixture: context ``w`` belongs
    to component ``g`` with probability ``weights[g]`` and then
    :math:`\\theta_w \\sim \\mathrm{Dirichlet}(\\alpha_g m_g)`, so contexts are pooled
    within groups rather than toward one law (a soft version of merging into several
    clusters). The assignments are summed out of the likelihood. The weights get a
    :math:`\\mathrm{Dirichlet}(1)` prior, which lets superfluous components empty
    :cite:`Rousseau2011`; component labels are not identified, which is harmless
    because every downstream quantity is label-invariant.

    Parameters
    ----------
    counts : np.ndarray
        Count table ``C[y, x, w]``.
    components : int
        Mixture components G (capped at the number of observed contexts).
    draws, tune, chains : int
        Passed to :func:`pymc.sample` (NUTS, chains run sequentially). With ADVI,
        ``draws * chains`` samples are drawn from the fitted approximation.
    log_alpha_sd : float
        Standard deviation of the normal prior on :math:`\\log \\alpha_g`.
    method : {'nuts', 'advi'}
        Exact posterior sampling, or a mean-field variational approximation
        (:func:`pymc.fit`, 20000 steps).
    nuts_sampler : {'pymc', 'nutpie', 'numpyro', 'blackjax'}
        Passed to :func:`pymc.sample`; alternatives to ``'pymc'`` need their package.
    prng : None, int, Generator

    Returns
    -------
    ContextPrior
    """
    pm = _pymc()
    seed = int(as_generator(prng).integers(2**31))
    counts = np.asarray(counts, dtype=float)
    Ky, Kx, Kw = counts.shape
    K = Ky * Kx
    columns = counts.reshape(K, Kw).T
    columns = columns[columns.sum(axis=1) > 0].astype(np.int64)
    G = max(1, min(int(components), len(columns)))
    with pm.Model():
        m = pm.Dirichlet("m", a=np.ones((G, K)))
        log_alpha = pm.Normal("log_alpha", mu=np.log(K), sigma=log_alpha_sd, shape=G)
        alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
        if G == 1:
            pm.DirichletMultinomial(
                "obs", n=columns.sum(axis=1), a=alpha[0] * m[0], shape=columns.shape, observed=columns
            )
        else:
            weights = pm.Dirichlet("weights", a=np.ones(G))
            logp = _dm_logp(columns, alpha[:, None] * m, pm.math.gammaln)
            pm.Potential("obs", pm.math.logsumexp(pm.math.log(weights)[None, :] + logp, axis=1).sum())
        if method == "advi":
            approx = pm.fit(20_000, method="advi", random_seed=seed, progressbar=False)
            idata = approx.sample(draws * chains, random_seed=seed)
        elif method == "nuts":
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                nuts_sampler=nuts_sampler,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        else:
            raise ValueError(f"Unknown method {method!r}; use 'nuts' or 'advi'.")
    post = idata.posterior
    alpha = np.asarray(post["alpha"]).reshape(-1, G)
    weights = np.asarray(post["weights"]).reshape(-1, G) if G > 1 else np.ones_like(alpha)
    return ContextPrior(alpha=alpha, m=np.asarray(post["m"]).reshape(-1, G, K), weights=weights)


def _dirichlet_rows(a, rng):
    """
    One Dirichlet draw per row of `a`; rows whose gammas all underflow get a point mass.
    """
    g = rng.standard_gamma(a)
    totals = g.sum(axis=-1, keepdims=True)
    empty = totals[..., 0] <= 0
    if np.any(empty):
        g[empty] = 0.0
        g[empty, np.argmax(a[empty], axis=-1)] = 1.0
        totals = g.sum(axis=-1, keepdims=True)
    return g / totals


def posterior_tables(counts, prior, n_draws=200, prng=None):
    """
    Posterior joints ``p[y, x, w]`` under the hierarchical Dirichlet model.

    With several components, each context's component is drawn from its posterior
    responsibilities before its law is drawn.

    Parameters
    ----------
    counts : np.ndarray
        Count table ``C[y, x, w]``.
    prior : ContextPrior
        From :func:`fit_context_prior` on the same `counts`.
    n_draws : int
        Joints to draw; each uses a random sample of the prior.
    prng : None, int, Generator

    Yields
    ------
    np.ndarray
        A joint of the same shape as `counts`, summing to one.
    """
    rng = as_generator(prng)
    counts = np.asarray(counts, dtype=float)
    Ky, Kx, Kw = counts.shape
    columns = counts.reshape(Ky * Kx, Kw).T
    totals = columns.sum(axis=1)
    for s in rng.integers(0, len(prior.alpha), size=n_draws):
        a = prior.alpha[s][:, None] * prior.m[s]
        if len(a) > 1:
            logr = np.log(prior.weights[s])[None, :] + _dm_logp(columns, a, gammaln)
            z = np.argmax(logr + rng.gumbel(size=logr.shape), axis=1)
        else:
            z = np.zeros(Kw, dtype=np.int64)
        theta = _dirichlet_rows(a[z] + columns, rng)
        p_w = _dirichlet_rows(totals + 0.5, rng)
        yield (theta * p_w[:, None]).T.reshape(Ky, Kx, Kw)


@dataclass
class LayerPosterior:
    """
    Posterior samples of one edge's layers, from :func:`layer_posterior`.

    Attributes
    ----------
    samples : xarray.Dataset
        One variable per quantity (te, tdmi, intrinsic, synergistic, shared) over
        dimension ``draw``, in bits.
    alpha : np.ndarray
        Posterior samples of the pooling concentration.
    channel : np.ndarray
        The full-data minimizing channel used to warm-start every draw.
    n_samples, n_contexts : int
    raw : dict
        The plug-in values on all data, for comparison.
    """

    samples: xr.Dataset
    alpha: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    channel: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    n_samples: int = 0
    n_contexts: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def mean(self):
        """``quantity -> posterior mean``."""
        return {k: float(self.samples[k].mean()) for k in QUANTITIES}

    def interval(self, confidence=0.95):
        """
        ``quantity -> (low, high)``, the equal-tailed credible intervals.
        """
        tail = (1 - confidence) / 2
        return {k: tuple(float(v) for v in np.quantile(self.samples[k], [tail, 1 - tail])) for k in QUANTITIES}

    def as_flow(self):
        """
        An :class:`~infoflow.measures.EdgeFlow` of posterior means.
        """
        mean = self.mean
        return EdgeFlow(
            mean["te"],
            mean["tdmi"],
            mean["intrinsic"],
            mean["synergistic"],
            mean["shared"],
            self.channel,
            self.n_samples,
            self.n_contexts,
            False,
            self.raw,
        )


def _decompose(p, Q, rng):
    """
    ``(te, tdmi, intrinsic, synergistic, shared)`` of a joint, warm-started from `Q`.
    """
    te = cmi_from_joint(p)
    tdmi = cmi_from_joint(p.sum(axis=2, keepdims=True))
    iif, _ = intrinsic_flow(p, initial=Q, prng=rng)
    return layers_from_values(te, tdmi, iif)[:5]


def layer_posterior(
    y, x, w, components=1, n_draws=200, draws=500, tune=500, chains=2, restarts=3, prior_options=None, prng=None
):
    """
    Posterior samples of the intrinsic, synergistic, and shared flow of one edge.

    Parameters
    ----------
    y, x, w : array_like
        Aligned target present, source past, and context (w may be None).
    components : int
        Mixture components of the context prior (see :func:`fit_context_prior`).
    n_draws : int
        Posterior joints decomposed into layers.
    draws, tune, chains : int
        NUTS settings for the hierarchical prior (see :func:`fit_context_prior`).
    restarts : int
        Restarts of the full-data channel optimization; each draw is warm-started
        from that channel.
    prior_options : dict, None
        Further keywords for :func:`fit_context_prior` (``log_alpha_sd``, ``method``,
        ``nuts_sampler``).
    prng : None, int, Generator

    Returns
    -------
    LayerPosterior
    """
    rng = as_generator(prng)
    y, Ky = dense_codes(y)
    x, Kx = dense_codes(x)
    w, Kw = dense_codes(np.zeros(len(y), dtype=np.int64) if w is None else w)
    full = joint_table(y, x, w, (Ky, Kx, Kw))
    raw_iif, Q = intrinsic_flow(full, restarts=restarts, prng=rng)
    prior = fit_context_prior(
        full, components, draws=draws, tune=tune, chains=chains, prng=rng, **(prior_options or {})
    )
    values = np.zeros((len(QUANTITIES), n_draws))
    for i, p in enumerate(posterior_tables(full, prior, n_draws, rng)):
        values[:, i] = _decompose(p, Q, rng)
    samples = xr.Dataset({k: ("draw", v) for k, v in zip(QUANTITIES, values, strict=True)})
    raw = {"te": cmi_from_joint(full), "tdmi": cmi_from_joint(full.sum(axis=2, keepdims=True)), "intrinsic": raw_iif}
    return LayerPosterior(samples, prior.alpha, Q, int(full.sum()), Kw, raw)


def _bdeu(rows, ess):
    """
    BDeu log marginal likelihood of count `rows` (parent configurations x outcomes).
    """
    q, r = rows.shape
    a = ess / (q * r)
    return float(
        np.sum(gammaln(ess / q) - gammaln(ess / q + rows.sum(axis=1))) + np.sum(gammaln(a + rows) - gammaln(a))
    )


def log_bayes_factor(y, x, w=None, ess=1.0):
    """
    Log Bayes factor (nats) for :math:`Y \\not\\perp X \\mid W` against :math:`Y \\perp X \\mid W`.

    The two models predict :math:`Y` from :math:`(X, W)` or from :math:`W` alone, each
    next-symbol distribution with a BDeu prior of equivalent sample size `ess`
    :cite:`Heckerman1995`, so the marginal likelihoods are closed-form. Positive values
    favour a nonzero transfer entropy (or, with ``w=None``, a nonzero time-delayed
    mutual information). Unlike posterior mass near zero, this weighs evidence for the
    zero itself. It is sensitive to `ess` :cite:`Silander2007`.

    Parameters
    ----------
    y, x, w : array_like
        Aligned target present, source past, and context (w may be None).
    ess : float
        Equivalent sample size of the BDeu prior.

    Returns
    -------
    float
    """
    y, Ky = dense_codes(y)
    x, Kx = dense_codes(x)
    w, Kw = dense_codes(np.zeros(len(y), dtype=np.int64) if w is None else w)
    table = joint_table(y, x, w, (Ky, Kx, Kw))
    dependent = table.reshape(Ky, Kx * Kw).T
    independent = table.sum(axis=1).T
    return _bdeu(dependent, ess) - _bdeu(independent, ess)
