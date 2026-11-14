"""
Multiscale Gaussian spectral backbone evidence :cite:`Raiola2026`.

Finite-timescale ordinary AND full-conditioning dependence supports an undirected
forest edge. MIRMO directions require separate asymptotic label evidence. Fitted
Gaussian spectral bootstrap p-values are experimental/model-based, not exact
permutation tests or distribution-free causal guarantees.
"""

import warnings
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations

import networkx as nx
import numpy as np
import xarray as xr
from dit.inference._symbols import as_generator

from .spectral import SpectralResult, SpectralSettings, spectral_rates
from .stats import benjamini_hochberg

__all__ = (
    "BackboneResult",
    "BackboneSettings",
    "backbone_from_rates",
    "infer_backbone",
    "orient_backbone",
    "rate_scaling",
)


@dataclass(frozen=True)
class BackboneSettings:
    """Gaussian null simulation count, edge FDR level, and BY (default) versus BH."""

    n_null: int = 199
    alpha: float = 0.05
    dependent: bool = True

    def check(self):
        if (
            not isinstance(self.n_null, (int, np.integer))
            or isinstance(self.n_null, (bool, np.bool_))
            or self.n_null < 1
        ):
            raise ValueError("n_null must be a positive integer")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if not isinstance(self.dependent, (bool, np.bool_)):
            raise ValueError("dependent must be a boolean")


@dataclass
class BackboneResult:
    """
    Separate structural evidence, never a fabricated lagged/TE skeleton.

    ``dataset`` stores symmetric rate-test p-values, edge q-values, ``detected``
    and ``status`` (detected/not_detected/unresolved), plus MIRMO ``label`` (NaN
    unknown) and ``orientation`` (1 row->column, -1 reverse, 0 unresolved).
    Unresolved includes numerical failures and a Monte Carlo floor that cannot
    survive correction (``budget_unresolved`` on empirical results).
    ``settings`` records the forest acknowledgment and population versus empirical
    provenance. Non-rejection is not evidence of absence. ``nulls`` retains the
    marginal/conditional null statistics by unordered index pair.
    """

    rates: SpectralResult
    dataset: xr.Dataset
    settings: dict = field(default_factory=dict)
    warnings: tuple = ()
    nulls: dict = field(default_factory=dict)

    @property
    def names(self):
        return self.rates.names

    def to_networkx(self, directed=False):
        """
        Undirected detected graph, or only resolved arrows when ``directed=True``.

        All nodes are retained. The directed export intentionally omits unresolved
        edges; inspect the undirected graph/dataset as well, not as a complete DAG.
        """
        g = nx.DiGraph() if directed else nx.Graph()
        g.add_nodes_from(self.names)
        for i, j in combinations(range(len(self.names)), 2):
            if not self.dataset.detected.values[i, j]:
                continue
            direction = int(self.dataset.orientation.values[i, j])
            if directed and not direction:
                continue
            a, b = (j, i) if directed and direction < 0 else (i, j)
            g.add_edge(
                self.names[a],
                self.names[b],
                rate=float(self.rates.dataset.rate.values[i, j]),
                conditional_rate=float(self.rates.dataset.conditional_rate.values[i, j]),
                pvalue=float(self.dataset.pvalue.values[i, j]),
                qvalue=float(self.dataset.qvalue.values[i, j]),
                orientation_status=str(self.dataset.orientation_status.values[i, j]),
            )
        return g

    @property
    def is_forest(self):
        return nx.is_forest(self.to_networkx())

    def candidate_sources(self, include_unresolved=True):
        """
        Candidate indices per target, ignoring structural arrow directions.

        Requires explicit acknowledgment of the forest model and refuses an
        incompatible detected cycle. Numerically/budget-unresolved pairs are retained by
        default; non-detections are screened out, with the attendant loss-of-power
        risk. This is a model-based screen, not a complete predictive network.
        """
        if not self.settings.get("assume_forest", False):
            raise ValueError("candidate filtering requires explicit assume_forest=True")
        if not self.is_forest:
            raise ValueError("detected graph contains an undirected cycle; cannot apply the forest backbone filter")
        mask = self.dataset.detected.values.copy()
        if include_unresolved:
            mask |= self.dataset.status.values == "unresolved"
        np.fill_diagonal(mask, False)
        return {t: np.flatnonzero(mask[:, t]).tolist() for t in range(len(self.names))}

    def summary(self):
        P = len(self.names)
        detected = int(np.triu(self.dataset.detected.values, 1).sum())
        unknown = int(np.triu(self.dataset.status.values == "unresolved", 1).sum())
        oriented = int(np.triu(self.dataset.orientation.values != 0, 1).sum())
        return (
            f"Spectral backbone ({self.settings.get('method', 'unknown')}): {P} nodes, "
            f"{detected} detected undirected edges, {unknown} unresolved pairs, "
            f"{oriented} resolved arrows; forest={self.is_forest}. "
            "Non-detection is not independence."
        )


def _assemble(rates, detected, status, settings, p_marginal=None, p_conditional=None, pvalue=None, qvalue=None):
    P = len(rates.names)
    shape = (P, P)
    empty = np.full(shape, np.nan)
    dims = ("source", "target")
    np.fill_diagonal(detected, False)
    np.fill_diagonal(status, "self")
    ds = xr.Dataset(
        {
            "detected": (dims, detected),
            "status": (dims, status),
            "marginal_pvalue": (dims, empty.copy() if p_marginal is None else p_marginal),
            "conditional_pvalue": (dims, empty.copy() if p_conditional is None else p_conditional),
            "pvalue": (dims, empty.copy() if pvalue is None else pvalue),
            "qvalue": (dims, empty.copy() if qvalue is None else qvalue),
            "label": (dims, empty.copy()),
            "orientation": (dims, np.zeros(shape, dtype=np.int8)),
            "orientation_status": (dims, np.full(shape, "unresolved", dtype="<U24")),
        },
        coords={"source": rates.names, "target": rates.names},
        attrs={"description": "Structural spectral evidence, not transfer-entropy/layer significance"},
    )
    result = BackboneResult(rates, ds, settings)
    notes = ["Forest, Gaussian, stationarity, full-observation and independent-noise assumptions are not verified."]
    if not result.is_forest:
        notes.append("Detected undirected cycles conflict with the forest reconstruction theorem; no tree is forced.")
    result.warnings = tuple(notes)
    return result


def backbone_from_rates(rates, *, assume_forest=False, atol=1e-10):
    """
    Population/analytic oracle using the two-positive-rate criterion (SM S56).

    This is NOT a sample significance test: ``atol`` only handles numerical zero
    in known population spectra. It deliberately returns no statistical p-values.
    Use :func:`infer_backbone` for empirical trajectories.
    """
    if rates.trial_lengths:
        raise ValueError("sample rate estimates are not population zeros; use infer_backbone for trajectories")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    m, c = rates.dataset.rate.values, rates.dataset.conditional_rate.values
    valid = np.isfinite(m) & np.isfinite(c)
    detected = valid & (m > atol) & (c > atol)
    status = np.where(valid, np.where(detected, "detected", "not_detected"), "unresolved")
    return _assemble(rates, detected, status, {"method": "population", "assume_forest": assume_forest, "atol": atol})


def _conditional_null(spectrum, pair):
    """Zero only pair residual coherence; preserve S_AC, S_CC and both autospectra."""
    s = np.asarray(spectrum, dtype=complex)
    a = list(pair)
    c = [k for k in range(s.shape[1]) if k not in a]
    out = s.copy()
    if c:
        ac, cc, ca = s[:, a][:, :, c], s[:, c][:, :, c], s[:, c][:, :, a]
        mediated = ac @ np.linalg.solve(cc, ca)
        out[:, a[0], a[1]] = mediated[:, 0, 1]
        out[:, a[1], a[0]] = mediated[:, 1, 0]
    else:
        out[:, a[0], a[1]] = out[:, a[1], a[0]] = 0
    return out


class _GaussianSpectrum:
    """Reusable Fourier synthesis factors for Gaussian spectral parametric nulls."""

    def __init__(self, frequency, spectrum, fs, nfft, lengths):
        # Convert one-sided Welch density to two-sided density before interpolation.
        two = spectrum.copy() / 2
        two[0] *= 2
        if nfft % 2 == 0:
            two[-1] *= 2
        self.factors = {}
        self.lengths = lengths
        for n in set(lengths):
            f = np.fft.rfftfreq(n, 1 / fs)
            index = np.searchsorted(frequency, f, side="right") - 1
            index = np.clip(index, 0, len(frequency) - 2)
            weight = np.clip((f - frequency[index]) / (frequency[index + 1] - frequency[index]), 0, 1)
            s = (1 - weight[:, None, None]) * two[index] + weight[:, None, None] * two[index + 1]
            # Real-valued time series have real DC and Nyquist coefficients.
            s[0] = s[0].real
            if n % 2 == 0:
                s[-1] = s[-1].real
            eig, vec = np.linalg.eigh(s)
            scale = np.maximum(np.max(np.abs(eig), axis=1, keepdims=True), np.finfo(float).tiny)
            if np.any(eig < -1e-8 * scale):
                raise ValueError("fitted spectral null is not positive semidefinite")
            self.factors[n] = vec * np.sqrt(np.maximum(eig, 0) * (fs * n))[:, None, :]

    def draw(self, rng):
        trials = []
        for n in self.lengths:
            root = self.factors[n]
            z = (rng.normal(size=root.shape[:2]) + 1j * rng.normal(size=root.shape[:2])) / np.sqrt(2)
            z[0] = rng.normal(size=root.shape[1])
            if n % 2 == 0:
                z[-1] = rng.normal(size=root.shape[1])
            ft = np.einsum("fij,fj->fi", root, z, optimize=True)
            trials.append(np.fft.irfft(ft, n=n, axis=0))
        return trials


def _pvalue(value, null):
    if not np.isfinite(value) or not np.all(np.isfinite(null)):
        return np.nan
    tolerance = 1e-12 * max(1, abs(value))
    return float((1 + np.count_nonzero(null >= value - tolerance)) / (len(null) + 1))


def infer_backbone(
    data,
    *,
    fs=1.0,
    names=None,
    timescales=None,
    spectral_settings=None,
    settings=None,
    assume_forest=False,
    prng=None,
    map_fn=map,
):
    """
    Experimental Gaussian spectral null tests and undirected backbone evidence.

    Pair adjacency requires both marginal and conditional dependence, so its union
    null is tested with ``max(p_marginal, p_conditional)``. Correct across the fixed
    unordered-pair family using BY (default) or BH :cite:`Benjamini2001,Benjamini1995`.
    The fitted Gaussian bootstrap is an additional approximation, not an exact
    finite-sample theorem of :cite:`Raiola2026`; inspect calibration for your setting.
    Unestimable pairs remain unresolved and are retained in candidate maps.
    ``map_fn`` maps over pairs with deterministic independent seeds.
    """
    settings = settings or BackboneSettings()
    settings.check()
    spectral_settings = spectral_settings or SpectralSettings()
    rates = spectral_rates(data, fs=fs, names=names, timescales=timescales, settings=spectral_settings)
    P = len(rates.names)
    pairs = list(combinations(range(P), 2))
    m, c = rates.dataset.rate.values, rates.dataset.conditional_rate.values
    s = rates.dataset.spectrum.values
    f = rates.dataset.frequency.values
    seeds = as_generator(prng).integers(0, 2**32, size=len(pairs))
    correction = sum(1 / k for k in range(1, len(pairs) + 1)) if settings.dependent else 1
    limited = bool(pairs and 1 / (settings.n_null + 1) > settings.alpha / (len(pairs) * correction))
    if limited:
        warnings.warn(
            "spectral null budget cannot resolve an isolated discovery at the first FDR threshold", stacklevel=2
        )

    def run(job):
        (i, j), seed = job
        if not np.isfinite(m[i, j]) or not np.isfinite(c[i, j]):
            return np.nan, np.nan, np.full(settings.n_null, np.nan), np.full(settings.n_null, np.nan)
        rng = as_generator(int(seed))
        marginal = np.zeros((len(f), 2, 2), dtype=complex)
        marginal[:, 0, 0], marginal[:, 1, 1] = s[:, i, i], s[:, j, j]
        gen_m = _GaussianSpectrum(f, marginal, fs, rates.settings["nfft"], rates.trial_lengths)
        gen_c = (
            None
            if P == 2
            else _GaussianSpectrum(f, _conditional_null(s, (i, j)), fs, rates.settings["nfft"], rates.trial_lengths)
        )
        null_m, null_c = np.empty(settings.n_null), np.empty(settings.n_null)
        for k in range(settings.n_null):
            trial_m = gen_m.draw(rng)
            null_m[k] = spectral_rates(trial_m, fs=fs, settings=spectral_settings).dataset.rate.values[0, 1]
            if gen_c is None:
                null_c[k] = null_m[k]
            else:
                trial_c = gen_c.draw(rng)
                null_c[k] = spectral_rates(trial_c, fs=fs, settings=spectral_settings).dataset.conditional_rate.values[
                    i, j
                ]
        return _pvalue(m[i, j], null_m), _pvalue(c[i, j], null_c), null_m, null_c

    results = list(map_fn(run, zip(pairs, seeds)))
    pm = np.full((P, P), np.nan)
    pc, p, q = pm.copy(), pm.copy(), pm.copy()
    for (i, j), (a, b, _, _) in zip(pairs, results, strict=True):
        pm[i, j] = pm[j, i] = a
        pc[i, j] = pc[j, i] = b
        p[i, j] = p[j, i] = max(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
    # Unestimable hypotheses still belong to the family; p=1 for correction only.
    family = np.array([p[i, j] if np.isfinite(p[i, j]) else 1 for i, j in pairs])
    _, adjusted = benjamini_hochberg(family, settings.alpha, dependent=settings.dependent)
    for (i, j), value in zip(pairs, adjusted, strict=True):
        if np.isfinite(p[i, j]):
            q[i, j] = q[j, i] = value
    detected = q <= settings.alpha
    # A floor p-value that cannot survive correction is budget-unresolved, not
    # evidence to remove a candidate. Keep it conservatively in the handoff.
    budget_unresolved = np.isfinite(p) & (p == 1 / (settings.n_null + 1)) & ~detected
    resolved = np.isfinite(p) & ~budget_unresolved
    status = np.where(resolved, np.where(detected, "detected", "not_detected"), "unresolved")
    provenance = asdict(settings) | {
        "method": "experimental_gaussian_bootstrap",
        "assume_forest": assume_forest,
        "resolution_limited": limited,
        "pair_seeds": [int(seed) for seed in seeds],
        "spectral_settings": dict(rates.settings),
    }
    result = _assemble(rates, detected, status, provenance, pm, pc, p, q)
    result.dataset["budget_unresolved"] = (("source", "target"), budget_unresolved)
    result.nulls = {pair: (record[2], record[3]) for pair, record in zip(pairs, results, strict=True)}
    result.warnings += (
        "Fitted Gaussian spectral-bootstrap p-values are approximate and require regime-specific calibration.",
    )
    return result


def orient_backbone(backbone, *, timescale_order, labels, provenance="provided"):
    """
    Orient supported links using explicitly supplied/model MIRMO labels.

    ``timescale_order`` is a permutation of process indices, fastest first.
    Symmetric labels are 0/1/2, with NaN or -1 for unknown. On an existing link,
    1 means fast->slow and 2 slow->fast. A supplied 0 conflicts with that link.
    A single observed rate, or merely supplying timescales, never assigns labels.
    Returns a copy and leaves unresolved/conflicting arrows unset.
    """
    P = len(backbone.names)
    order = list(timescale_order)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) for k in order) or sorted(
        order
    ) != list(range(P)):
        raise ValueError("timescale_order must be a strict permutation of process indices, fastest first")
    if "timescale" in backbone.rates.dataset:
        tau = backbone.rates.dataset.timescale.values
        if np.any(np.diff(np.sort(tau)) <= 0) or not np.array_equal(np.argsort(tau), order):
            raise ValueError("timescale_order must agree with the supplied, strictly ordered numeric timescales")
    if provenance not in ("provided", "model"):
        raise ValueError("label provenance must be 'provided' or 'model'; empirical scaling is diagnostic only")
    lab = np.array(labels, dtype=float, copy=True)
    if lab.shape != (P, P) or not np.all(np.isnan(lab) | np.isin(lab, [-1, 0, 1, 2])):
        raise ValueError("labels must be a symmetric P x P matrix of 0/1/2 or unknown NaN/-1")
    lab[lab == -1] = np.nan
    if not np.array_equal(lab, lab.T, equal_nan=True):
        raise ValueError("MIRMO labels must be symmetric")
    np.fill_diagonal(lab, np.nan)
    ds = backbone.dataset.copy(deep=True)
    ds["label"][:] = lab
    ds["orientation"][:] = 0
    ds["orientation_status"][:] = "unresolved"
    ranks = np.argsort(order)
    conflicts = []
    for i, j in combinations(range(P), 2):
        if not ds.detected.values[i, j] or np.isnan(lab[i, j]):
            continue
        if lab[i, j] == 0 or not backbone.is_forest or not backbone.settings.get("assume_forest"):
            ds.orientation_status.values[i, j] = ds.orientation_status.values[j, i] = "conflict"
            conflicts.append((i, j))
            continue
        forward = (ranks[i] < ranks[j]) == (lab[i, j] == 1)
        ds.orientation.values[i, j], ds.orientation.values[j, i] = (1, -1) if forward else (-1, 1)
        ds.orientation_status.values[i, j] = ds.orientation_status.values[j, i] = provenance
    notes = backbone.warnings + (
        (f"Unresolved orientation/model conflicts at pairs {conflicts}.",) if conflicts else ()
    )
    return replace(
        backbone,
        dataset=ds,
        settings=backbone.settings | {"label_provenance": provenance, "timescale_order": order},
        warnings=notes,
    )


def rate_scaling(results, ratios, *, comparable=False):
    """
    Experimental rate curves and log-log slopes across known changing scale ratios.

    Callers must acknowledge unchanged couplings/noise and a common scale hierarchy
    with ``comparable=True``. All results need supplied numeric timescales and matching
    names. Regression standard errors describe the curve fit, not measurement error.
    A growing finite curve can still converge: labels deliberately remain NaN, even
    for a precise positive slope. One recording returns no slope or direction.
    """
    from scipy.stats import linregress

    results = list(results)
    r = np.asarray(ratios, dtype=float)
    if (
        not results
        or r.shape != (len(results),)
        or not np.all(np.isfinite(r))
        or np.any(r <= 1)
        or np.any(np.diff(r) <= 0)
    ):
        raise ValueError("need matching results and strictly increasing finite timescale ratios > 1")
    if len(results) > 1 and not comparable:
        raise ValueError("acknowledge unchanged coupling/noise and scale hierarchy with comparable=True")
    names = results[0].names
    for result, ratio in zip(results, r, strict=True):
        if result.names != names or "rescaled_rate" not in result.dataset:
            raise ValueError("results must have matching process names and supplied numeric timescales")
        tau = np.sort(result.dataset.timescale.values)
        if not np.allclose(tau / tau[0], ratio ** np.arange(len(tau))):
            raise ValueError("supplied timescales must follow the declared geometric ratio hierarchy")
        if not np.array_equal(
            np.argsort(result.dataset.timescale.values), np.argsort(results[0].dataset.timescale.values)
        ):
            raise ValueError("timescale order must be unchanged across conditions")
    values = np.stack([v.dataset.rescaled_rate.values for v in results])
    slope = np.full((len(names), len(names)), np.nan)
    error = slope.copy()
    if len(results) >= 3:
        for i, j in combinations(range(len(names)), 2):
            y = values[:, i, j]
            if np.all(np.isfinite(y)) and np.all(y > 0):
                fit = linregress(np.log(r), np.log(y))
                slope[i, j] = slope[j, i] = fit.slope
                error[i, j] = error[j, i] = fit.stderr
    dims = ("source", "target")
    return xr.Dataset(
        {
            "rescaled_rate": (("ratio", *dims), values),
            "log_slope": (dims, slope),
            "slope_stderr": (dims, error),
            "label": (dims, np.full_like(slope, np.nan)),
        },
        coords={"ratio": r, "source": names, "target": names},
        attrs={
            "description": "Experimental finite-ratio diagnostic; no asymptotic labels are inferred",
            "comparable_acknowledged": comparable,
        },
    )
