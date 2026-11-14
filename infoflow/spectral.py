"""
Gaussian spectral mutual-information rates :cite:`Raiola2026,Komaee2020`.

These symmetric, whole-trajectory rates are not transfer entropy or a decomposition
into intrinsic/shared/synergistic flow. Frequencies are in Hz and rates in bits per
unit of time. Independent trials share a stationary spectrum; windows never cross
trial boundaries.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.signal import get_window

from .data import DiscreteData, as_trials

__all__ = (
    "SpectralResult",
    "SpectralSettings",
    "conditional_coherence",
    "rates_from_spectrum",
    "spectral_rates",
)


@dataclass(frozen=True)
class SpectralSettings:
    """
    Welch estimation and explicit diagonal-target regularization.

    ``noverlap=None`` uses half a segment; ``nfft=None`` uses ``nperseg``.
    ``shrinkage`` replaces S by ``(1-shrinkage)*S + shrinkage*diag(S)`` before
    either rate is evaluated. It is never selected automatically. ``condition_limit``
    applies to the scale-normalized spectral matrix, not the measurement units.
    """

    nperseg: int = 256
    noverlap: int | None = None
    nfft: int | None = None
    window: str | tuple = "hann"
    detrend: str | bool = "constant"
    shrinkage: float = 0.0
    condition_limit: float = 1e12

    def resolved(self):
        """Validate settings and return the actual overlap and FFT size."""
        overlap = self.nperseg // 2 if self.noverlap is None else self.noverlap
        nfft = self.nperseg if self.nfft is None else self.nfft
        for name, value in (("nperseg", self.nperseg), ("noverlap", overlap), ("nfft", nfft)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.nperseg < 4 or not 0 <= overlap < self.nperseg or nfft < self.nperseg:
            raise ValueError("need nperseg >= 4, 0 <= noverlap < nperseg, and nfft >= nperseg")
        if self.detrend not in (False, "constant"):
            raise ValueError("detrend must be False or 'constant'")
        if not np.isfinite(self.shrinkage) or not 0 <= self.shrinkage <= 1:
            raise ValueError("shrinkage must be between zero and one")
        if not np.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError("condition_limit must be finite and greater than one")
        get_window(self.window, self.nperseg)
        return int(overlap), int(nfft)


@dataclass
class SpectralResult:
    """
    Spectra, Gaussian rates and diagnostics, separate from the flow layers.

    ``dataset`` contains ``spectrum`` (the matrix actually used), ``raw_spectrum``,
    ``coherence``, ``partial_coherence`` over (frequency, source, target), rates
    over (source, target), and frequency-wise rank/conditioning/validity. Missing
    conditional frequencies make the conditional rate NaN, not zero. ``settings``
    and ``trial_lengths`` describe estimation; ``data_fingerprints`` can detect exact
    reuse of discovery trials but cannot establish that arbitrary windows are disjoint.
    """

    dataset: xr.Dataset
    settings: dict = field(default_factory=dict)
    trial_lengths: tuple = ()
    data_fingerprints: tuple = ()

    @property
    def names(self):
        return list(self.dataset.coords["source"].values)

    def summary(self):
        ds = self.dataset
        invalid = float(ds.attrs["conditional_invalid_fraction"])
        return (
            f"Gaussian spectral rates: {len(self.names)} processes, "
            f"{len(ds.frequency)} frequencies; bits/time; "
            f"{invalid:.1%} conditional frequencies unestimable; "
            f"shrinkage={self.settings.get('shrinkage', 0):g}"
        )


def _trials(data):
    if isinstance(data, DiscreteData):
        raise ValueError("spectral rates need raw continuous observations, not DiscreteData symbol codes")
    trials = as_trials(data)
    if not trials or trials[0].ndim != 2 or trials[0].shape[1] < 1:
        raise ValueError("need nonempty (samples, processes) continuous trials")
    P = trials[0].shape[1]
    out = []
    for t in trials:
        if t.ndim != 2 or t.shape[1] != P or not len(t) or not np.isrealobj(t):
            raise ValueError("trials must be real, nonempty and have the same process count")
        a = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(a)):
            raise ValueError(
                "split missing/nonfinite multivariate intervals into complete trials before spectral analysis"
            )
        out.append(a)
    return out


def _fingerprints(trials):
    return tuple(
        sha256(str(t.shape).encode() + np.ascontiguousarray(t, dtype=float).tobytes()).hexdigest() for t in trials
    )


def _names(names, P):
    result = [f"x{p}" for p in range(P)] if names is None else list(names)
    if len(result) != P or any(not isinstance(n, str) for n in result) or len(set(result)) != P:
        raise ValueError("names must be unique strings, one per process")
    return result


def _coherence(matrix):
    power = np.real(np.diagonal(matrix, axis1=-2, axis2=-1))
    root = np.sqrt(np.maximum(power, 0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi = np.abs(matrix / (root[..., :, None] * root[..., None, :])) ** 2
    if np.any(phi > 1 + 1e-9):
        raise ValueError("coherence materially exceeds one; the spectral matrix is invalid")
    phi = np.minimum(phi, 1.0)
    phi = np.where((power[..., :, None] > 0) & (power[..., None, :] > 0), phi, np.nan)
    idx = np.arange(matrix.shape[-1])
    phi[..., idx, idx] = np.nan
    return phi


def _integrate(phi, frequency):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -trapezoid(np.log1p(-phi), x=frequency, axis=0) / np.log(2)


def conditional_coherence(spectrum, pair, conditioning=None):
    """
    Coherence of a pair after conditioning on a subset of whole trajectories.

    Implements the Schur complement :cite:`Raiola2026` (SM S57--S58). ``None``
    conditions on all other processes; an empty sequence gives ordinary coherence.
    The input uses frequency as its first axis and the convention ``S_ij = X_i X_j*``.
    Singular conditioning blocks raise rather than using a pseudoinverse.
    """
    s = np.asarray(spectrum, dtype=complex)
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise ValueError("spectrum must have shape (frequencies, processes, processes)")
    P = s.shape[1]
    pair = list(pair)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("pair must contain two distinct process indices")
    c = [p for p in range(P) if p not in pair] if conditioning is None else list(conditioning)
    if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or not 0 <= p < P for p in pair + c):
        raise ValueError("pair and conditioning must contain valid process indices")
    if len(set(c)) != len(c) or set(pair) & set(c):
        raise ValueError("conditioning must be distinct and disjoint from the pair")
    v = s[:, pair][:, :, pair]
    if c:
        ac = s[:, pair][:, :, c]
        cc = s[:, c][:, :, c]
        ca = s[:, c][:, :, pair]
        v = v - ac @ np.linalg.solve(cc, ca)
    return _coherence(v)[:, 0, 1]


def rates_from_spectrum(frequency, spectrum, *, names=None, timescales=None, settings=None):
    """
    Evaluate Gaussian rates from an explicit nonnegative-Hz spectral matrix.

    This numerical entry point also supports analytic spectra and nonuniform grids.
    The integral is over the supplied bandwidth only. It does not infer sampling
    frequency, statistical significance, or MIRMO labels. Natural-log reference
    rates must be divided by ln(2) for comparison :cite:`Raiola2026,Komaee2020`.
    """
    settings = settings or SpectralSettings()
    settings.resolved()
    f = np.asarray(frequency, dtype=float)
    s = np.asarray(spectrum, dtype=complex)
    if f.ndim != 1 or len(f) < 2 or not np.all(np.isfinite(f)) or f[0] < 0 or np.any(np.diff(f) <= 0):
        raise ValueError("frequency must be a finite, strictly increasing nonnegative Hz grid")
    if s.ndim != 3 or s.shape != (len(f), s.shape[1], s.shape[1]) or not s.shape[1] or not np.all(np.isfinite(s)):
        raise ValueError("spectrum must be finite with shape (frequencies, processes, processes)")
    P = s.shape[1]
    names = _names(names, P)
    power = np.real(np.diagonal(s, axis1=1, axis2=2))
    if np.any(power < 0):
        raise ValueError("spectral powers must be nonnegative")
    root = np.sqrt(power)
    scale = root[:, :, None] * root[:, None, :]
    if np.any(np.abs(s - s.conj().swapaxes(1, 2)) > 1e-9 * np.maximum(scale, np.finfo(float).tiny)):
        raise ValueError("spectral matrices must be Hermitian")
    raw = s.copy()
    good_power = np.all(power > 0, axis=1)
    raw_corr = np.zeros_like(s)
    raw_corr[good_power] = s[good_power] / scale[good_power]
    raw_eig = np.linalg.eigvalsh(raw_corr)
    if np.any(raw_eig < -1e-9 * max(P, 1)):
        raise ValueError("spectral matrices must be positive semidefinite, before regularization")
    s = (1 - settings.shrinkage) * s
    idx = np.arange(P)
    s[:, idx, idx] = power
    corr = np.zeros_like(s)
    corr[good_power] = s[good_power] / scale[good_power]
    eig = np.linalg.eigvalsh(corr)
    if np.any(eig < -1e-9 * max(P, 1)):
        raise ValueError("spectral matrices must be positive semidefinite")
    rank = np.sum(eig > np.maximum(eig[:, -1:], 1) * P * np.finfo(float).eps, axis=1)
    condition = np.full(len(f), np.inf)
    pos = good_power & (eig[:, 0] > 0)
    condition[pos] = eig[pos, -1] / eig[pos, 0]
    valid = good_power & (condition <= settings.condition_limit)
    phi = _coherence(s)
    partial = np.full_like(phi, np.nan)
    if np.any(valid):
        precision = np.linalg.solve(corr[valid], np.broadcast_to(np.eye(P), corr[valid].shape))
        partial[valid] = _coherence(precision)
    dims = ("frequency", "source", "target")
    ds = xr.Dataset(
        {
            "raw_spectrum": (dims, raw),
            "spectrum": (dims, s),
            "coherence": (dims, phi),
            "partial_coherence": (dims, partial),
            "rate": (("source", "target"), _integrate(phi, f)),
            "conditional_rate": (("source", "target"), _integrate(partial, f)),
            "spectral_rank": (("frequency",), rank),
            "raw_spectral_rank": (
                ("frequency",),
                np.sum(raw_eig > np.maximum(raw_eig[:, -1:], 1) * P * np.finfo(float).eps, axis=1),
            ),
            "condition_number": (("frequency",), condition),
            "conditional_valid": (("frequency",), valid),
        },
        coords={"frequency": f, "source": names, "target": names},
        attrs={
            "description": "Gaussian whole-trajectory spectral information rates; not transfer entropy",
            "frequency_units": "Hz",
            "rate_units": "bits/time",
            "conditional_invalid_fraction": float(1 - valid.mean()),
            "shrinkage_coherence_change": float(np.nanmax(np.abs(phi - _coherence(raw)))) if P > 1 else 0.0,
        },
    )
    for key in ("rate", "conditional_rate"):
        ds[key].attrs["units"] = "bits/time"
    if timescales is not None:
        tau = np.asarray(timescales, dtype=float)
        if tau.shape != (P,) or not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError("timescales must contain a finite positive time per process")
        ds["timescale"] = (("source",), tau)
        reference = np.maximum(tau[:, None], tau[None, :])
        for key in ("rate", "conditional_rate"):
            ds[f"rescaled_{key}"] = ds[key] * reference
            ds[f"rescaled_{key}"].attrs["units"] = "bits/reference-time"
    return SpectralResult(ds, asdict(settings))


def spectral_rates(data, *, fs=1.0, names=None, timescales=None, settings=None):
    """
    Welch ordinary and all-other-process-conditioned Gaussian information rates.

    ``data`` is a (samples, processes) array or independent trials of such arrays.
    All series must be finite, real, synchronized observations with common ``fs``.
    Every trial must contain at least ``nperseg`` samples. Trials contribute in
    proportion to their segment counts. Constants are rejected; rank-deficient
    conditioning is reported as unestimable unless regularization was requested.
    ``timescales`` optionally rescales the rates but never assigns directions.
    """
    settings = settings or SpectralSettings()
    overlap, nfft = settings.resolved()
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    trials = _trials(data)
    if any(len(t) < settings.nperseg for t in trials):
        raise ValueError("each trial must contain at least nperseg samples; use shorter segments explicitly")
    P = trials[0].shape[1]
    if np.any(np.var(np.concatenate(trials), axis=0) == 0):
        raise ValueError("constant processes have undefined spectral coherence")
    window = get_window(settings.window, settings.nperseg)
    step = settings.nperseg - overlap
    spectrum = np.zeros((nfft // 2 + 1, P, P), dtype=complex)
    counts = []
    for t in trials:
        segments = np.lib.stride_tricks.sliding_window_view(t, settings.nperseg, axis=0)[::step]
        if settings.detrend == "constant":
            segments = segments - segments.mean(axis=-1, keepdims=True)
        ft = np.fft.rfft(segments * window, n=nfft, axis=-1)
        spectrum += np.einsum("spf,sqf->fpq", ft, ft.conj(), optimize=True)
        counts.append(len(segments))
    spectrum /= sum(counts) * fs * np.sum(window**2)
    spectrum[1 : -1 if nfft % 2 == 0 else None] *= 2
    result = rates_from_spectrum(
        np.fft.rfftfreq(nfft, 1 / fs), spectrum, names=names, timescales=timescales, settings=settings
    )
    result.settings.update(fs=float(fs), noverlap=overlap, nfft=nfft, n_segments=sum(counts))
    result.trial_lengths = tuple(len(t) for t in trials)
    result.data_fingerprints = _fingerprints(trials)
    result.dataset["segment_count"] = (("trial",), counts)
    result.dataset.attrs.update(fs=float(fs), n_segments=sum(counts))
    return result
