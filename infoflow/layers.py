"""
Uncertainty and significance for the flow layers of one edge.

* Confidence intervals come from a stationary bootstrap of the aligned
  (target present, source past, context) tuples :cite:`Politis1994`, so block
  junctions never create unobserved windows. The context merge is fixed from the
  full data and each resample's channel is warm-started from the full-data
  channel. Intervals are percentile intervals of the resampled bias-corrected
  estimates, recentred on the cross-fitted point estimate.
* The intrinsic flow is tested with a parametric null: the joint is projected onto
  :math:`p_0(y, x, w) = \\sum_{\\bar w} p(w) Q(\\bar w \\mid w) q(y \\mid \\bar w) q(x \\mid \\bar w)`,
  in which :math:`X \\perp Y \\mid \\bar W` under the minimizing channel :math:`Q` (so
  its intrinsic flow is zero) while the (X, W̄) and (Y, W̄) marginals are kept.
  Counts are resampled from :math:`p_0` and re-estimated.
* The synergistic layer is tested against surrogates in which the context is
  permuted within strata of its hard clustering under the minimizing channel:
  (Y, X) then depends on W only through W̄, which keeps the intrinsic flow at the
  observed magnitude and removes synergy.
* The shared layer is tested against unconditional permutations of the context,
  which keep the time-delayed mutual information but make W independent of (Y, X),
  removing shared flow.

Every null uses the same bias-corrected estimator as the observed statistic, and
the bootstrap is used only for intervals (a bootstrap cannot test a layer that
sits on its zero boundary).
"""

from dataclasses import dataclass, field

import numpy as np
from dit.inference import stationary_bootstrap
from dit.inference._symbols import as_generator

from .context import apply_merge, merge_contexts_statistical
from .crossfit import crossfit_flows
from .measures import EdgeFlow, _degrade, cmi_from_joint, dense_codes, intrinsic_flow, joint_table, layers_from_values

__all__ = (
    "LAYERS",
    "LayerStatistics",
    "adjust_layer_pvalues",
    "layer_statistics",
)

LAYERS = ("intrinsic", "synergistic", "shared")
_EPS = 1e-6


@dataclass
class LayerStatistics:
    """
    Point estimates, confidence intervals, and p-values for one edge's layers.

    Attributes
    ----------
    flow : EdgeFlow
        The cross-fitted point estimates.
    ci : dict
        ``layer -> (low, high)`` for each of te, tdmi, intrinsic, synergistic, shared.
    pvalue : dict
        ``layer -> p`` for the intrinsic, synergistic, and shared layers being positive.
    n_boot, n_null : int
    """

    flow: EdgeFlow
    ci: dict = field(default_factory=dict)
    pvalue: dict = field(default_factory=dict)
    n_boot: int = 0
    n_null: int = 0


def _layers_on_table(table, mapping, n_clusters, estimator, initial, prng):
    """
    Bias-corrected single-joint layers with a fixed merge and a warm-started channel.
    """
    merged = apply_merge(table, mapping, n_clusters)
    n = merged.sum()
    _, Q = intrinsic_flow(merged, initial=initial, prng=prng)
    te = cmi_from_joint(table, estimator)
    tdmi = cmi_from_joint(table.sum(axis=2, keepdims=True), estimator)
    iif = min(cmi_from_joint(_degrade(merged, Q), estimator, n=n), te, tdmi)
    te, tdmi, iif, syn, shared, _ = layers_from_values(te, tdmi, iif)
    return {"te": te, "tdmi": tdmi, "intrinsic": iif, "synergistic": syn, "shared": shared}


def _upper_pvalue(null, value):
    null = np.asarray(null)
    return float((1 + np.sum(null >= value - 1e-12)) / (1 + len(null)))


def _within_strata(x, strata, rng):
    grouped = np.argsort(strata, kind="stable")
    shuffled = np.lexsort((rng.random(len(strata)), strata))
    out = np.empty_like(x)
    out[grouped] = x[shuffled]
    return out


def _null_projection(merged, Q):
    """
    The closest joint with zero intrinsic flow under channel Q (see module docs).
    """
    p = merged / merged.sum()
    q = _degrade(p, Q)  # q[y, x, z]
    q_z = q.sum(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        q_y_given_z = np.where(q_z > 0, q.sum(axis=1) / np.where(q_z > 0, q_z, 1), 0)  # (y, z)
        q_x_given_z = np.where(q_z > 0, q.sum(axis=0) / np.where(q_z > 0, q_z, 1), 0)  # (x, z)
    p_w = p.sum(axis=(0, 1))
    # p0[y, x, w] = sum_z p(w) Q[w, z] q(y|z) q(x|z)
    p0 = np.einsum("w,wz,yz,xz->yxw", p_w, Q, q_y_given_z, q_x_given_z)
    return p0 / p0.sum()


def layer_statistics(
    y,
    x,
    w,
    n_boot=100,
    n_null=100,
    confidence=0.95,
    mean_block_length=None,
    estimator="miller_madow",
    merge=True,
    merge_alpha=0.01,
    gap=0,
    prng=None,
):
    """
    Cross-fitted layers with bootstrap intervals and per-layer p-values.

    Parameters
    ----------
    y, x, w : array_like
        Aligned target present, source past, and context (w may be None).
    n_boot : int
        Stationary-bootstrap resamples of the aligned tuples.
    n_null : int
        Null resamples per layer (parametric for intrinsic, permutations for the others).
    confidence : float
        Coverage of the intervals.
    mean_block_length : float, None
        Mean bootstrap block length (default ``N ** (1/3)``).
    estimator, merge, merge_alpha, gap
        As in :func:`~infoflow.crossfit.crossfit_flows`.
    prng : None, int, Generator

    Returns
    -------
    LayerStatistics
    """
    rng = as_generator(prng)
    flow = crossfit_flows(y, x, w, estimator=estimator, merge=merge, merge_alpha=merge_alpha, gap=gap, prng=rng)
    y, Ky = dense_codes(y)
    x, Kx = dense_codes(x)
    w, Kw = dense_codes(np.zeros(len(y), dtype=np.int64) if w is None else w)
    shape = (Ky, Kx, Kw)
    full = joint_table(y, x, w, shape)
    mapping = merge_contexts_statistical(full, merge_alpha) if merge else np.arange(Kw)
    n_clusters = int(mapping.max()) + 1
    merged = apply_merge(full, mapping, n_clusters)
    _, Q = intrinsic_flow(merged, prng=rng)
    observed = _layers_on_table(full, mapping, n_clusters, estimator, Q, rng)

    stats = LayerStatistics(flow, n_boot=n_boot, n_null=n_null)
    names = ("te", "tdmi", *LAYERS)
    if n_boot:
        samples = {k: [] for k in names}
        for idx in stationary_bootstrap(np.arange(len(y)), n_boot, mean_block_length, rng):
            table = joint_table(y[idx], x[idx], w[idx], shape)
            values = _layers_on_table(table, mapping, n_clusters, estimator, Q, rng)
            for k in names:
                samples[k].append(values[k])
        tail = (1 - confidence) / 2
        point = flow.as_dict()
        for k in names:
            s = np.asarray(samples[k])
            shift = point[k] - observed[k]
            low, high = np.quantile(s, [tail, 1 - tail]) + shift
            stats.ci[k] = (max(float(low), 0.0), max(float(high), 0.0))
    if n_null:
        # Intrinsic: parametric projection onto zero intrinsic flow.
        p0 = _null_projection(merged, Q)
        null = []
        for _ in range(n_null):
            table = rng.multinomial(int(merged.sum()), p0.ravel()).reshape(merged.shape).astype(float)
            null.append(_layers_on_table(table, np.arange(n_clusters), n_clusters, estimator, Q, rng)["intrinsic"])
        stats.pvalue["intrinsic"] = _upper_pvalue(null, observed["intrinsic"])
        # Synergistic: permute the context within strata of its hard clustering under
        # the minimizing channel. (Y, X) then depends on W only through that clustering,
        # so the flow routed through W-bar (the intrinsic flow) is kept and the synergy
        # carried by W's finer detail is removed.
        hard = np.argmax(Q, axis=1)[mapping[w]]
        # Shared: an unconditional permutation of the context makes W independent of
        # (Y, X), keeping the TDMI while removing shared (and synergistic) flow.
        for layer, permute in (
            ("synergistic", lambda: _within_strata(w, hard, rng)),
            ("shared", lambda: w[rng.permutation(len(w))]),
        ):
            null = []
            for _ in range(n_null):
                table = joint_table(y, x, permute(), shape)
                null.append(_layers_on_table(table, mapping, n_clusters, estimator, Q, rng)[layer])
            stats.pvalue[layer] = _upper_pvalue(null, observed[layer])
    return stats


def adjust_layer_pvalues(pvalues, alpha=0.05, dependent=False):
    """
    False-discovery-rate control within each layer across edges.

    Parameters
    ----------
    pvalues : dict
        ``layer -> array`` of per-edge p-values (NaN for untested edges).
    alpha : float
    dependent : bool
        Benjamini–Yekutieli instead of Benjamini–Hochberg :cite:`Benjamini1995,Benjamini2001`.

    Returns
    -------
    significant, adjusted : dict
        ``layer -> bool array`` and ``layer -> adjusted p-value array`` (NaN where untested).
    """
    from .stats import benjamini_hochberg

    significant, adjusted = {}, {}
    for layer, p in pvalues.items():
        p = np.asarray(p, dtype=float)
        mask = np.isfinite(p)
        sig = np.zeros(p.shape, dtype=bool)
        adj = np.full(p.shape, np.nan)
        if mask.any():
            reject, q = benjamini_hochberg(p[mask], alpha, dependent)
            sig[mask], adj[mask] = reject, q
        significant[layer], adjusted[layer] = sig, adj
    return significant, adjusted
