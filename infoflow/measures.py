"""
Edge measures: transfer entropy, time-delayed mutual information, and the
intrinsic / synergistic / shared decomposition of information flow.

For a target present :math:`Y`, a source past :math:`X`, and a context :math:`W`
(the target's past, other parents' pasts, and forced conditionals),

.. math::

    \\mathrm{TE} &= I[Y : X \\mid W], \\qquad \\mathrm{TDMI} = I[Y : X], \\\\
    \\mathrm{intrinsic} &= I[Y : X \\downarrow W]
        = \\min_{p(\\bar w \\mid w)} I[Y : X \\mid \\bar W], \\\\
    \\mathrm{synergistic} &= \\mathrm{TE} - \\mathrm{intrinsic}, \\qquad
    \\mathrm{shared} = \\mathrm{TDMI} - \\mathrm{intrinsic}

:cite:`James2016,Maurer1999`. The identity channel gives TE and a constant
channel gives TDMI, so both remainders are non-negative, and all three layers are
computed from one joint distribution so the identities hold exactly.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

__all__ = (
    "EdgeFlow",
    "cmi_from_joint",
    "edge_flows",
    "intrinsic_flow",
    "joint_table",
)

_LOG2 = np.log(2)


def dense_codes(values):
    """
    Integer codes 0..K-1 for a 1D array, or for the rows of a 2D array.
    """
    values = np.asarray(values)
    if values.ndim == 2:
        if values.shape[1] == 0:
            return np.zeros(len(values), dtype=np.int64), 1
        _, inverse = np.unique(values, axis=0, return_inverse=True)
    else:
        _, inverse = np.unique(values, return_inverse=True)
    inverse = inverse.ravel().astype(np.int64)
    return inverse, int(inverse.max()) + 1 if len(inverse) else 1


def joint_table(y, x, w, shape=None):
    """
    The count table ``C[y, x, w]`` of dense codes.
    """
    if shape is None:
        shape = (int(y.max()) + 1, int(x.max()) + 1, int(w.max()) + 1)
    table = np.zeros(shape)
    np.add.at(table, (y, x, w), 1.0)
    return table


def _entropy_terms(q):
    """
    Plug-in entropy (nats) of an array of probabilities (zeros ignored).
    """
    q = q[q > 0]
    return float(-np.sum(q * np.log(q)))


def cmi_from_joint(p, estimator="plugin", n=None):
    """
    :math:`I[Y : X \\mid W]` in bits from a table ``p[y, x, w]`` (counts or probabilities).

    Parameters
    ----------
    p : np.ndarray
        Joint table over (y, x, w). With a single context value this is
        :math:`I[Y : X]`.
    estimator : {'plugin', 'miller_madow'}
        ``'miller_madow'`` adds each entropy term's :math:`(K - 1) / 2N` bias
        correction :cite:`Miller1955`, with :math:`K` the number of occupied
        cells. It needs the sample size `n` (taken from ``p.sum()`` for counts).
    n : float, None
        The number of samples behind a probability table.
    """
    total = p.sum()
    if total <= 0:
        return 0.0
    n = total if n is None else n
    q = p / total
    q_xw, q_yw, q_w = q.sum(axis=0), q.sum(axis=1), q.sum(axis=(0, 1))
    value = _entropy_terms(q_xw) + _entropy_terms(q_yw) - _entropy_terms(q) - _entropy_terms(q_w)
    if estimator == "miller_madow":
        k = lambda a: np.count_nonzero(a > 1e-12)  # noqa: E731
        value += ((k(q_xw) - 1) + (k(q_yw) - 1) - (k(q) - 1) - (k(q_w) - 1)) / (2 * n)
    elif estimator != "plugin":
        raise ValueError(f"Unknown estimator {estimator!r}; use 'plugin' or 'miller_madow'.")
    return value / _LOG2


def _degrade(p, Q):
    """
    ``q[y, x, z] = sum_w p[y, x, w] Q[w, z]``.
    """
    return np.tensordot(p, Q, axes=([2], [0]))


def _softmax_rows(theta):
    theta = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(theta)
    return e / e.sum(axis=1, keepdims=True)


def _objective(theta_flat, p, Kw, Kz):
    theta = theta_flat.reshape(Kw, Kz)
    Q = _softmax_rows(theta)
    q = _degrade(p, Q)
    eps = 1e-300
    q_xz, q_yz, q_z = q.sum(axis=0), q.sum(axis=1), q.sum(axis=(0, 1))
    ratio = (q + eps) * (q_z + eps)[None, None, :] / ((q_xz + eps)[None, :, :] * (q_yz + eps)[:, None, :])
    G = np.log(ratio)
    value = float(np.sum(q * G))
    grad_Q = np.tensordot(p, G, axes=([0, 1], [0, 1]))  # (Kw, Kz)
    grad_theta = Q * (grad_Q - np.sum(Q * grad_Q, axis=1, keepdims=True))
    return value, grad_theta.ravel()


def intrinsic_flow(p, bound=None, restarts=5, initial=None, prng=None, maxiter=500, max_parameters=1_000_000):
    """
    Minimize :math:`I[Y : X \\mid \\bar W]` over channels :math:`p(\\bar w \\mid w)`.

    Parameters
    ----------
    p : np.ndarray
        Joint table ``p[y, x, w]`` (counts or probabilities).
    bound : int, None
        The size of :math:`\\bar W`; defaults to :math:`|W|`, which suffices.
    restarts : int
        Random restarts, in addition to starts near the identity and constant channels.
        The problem is non-convex, so the result is a local minimum; restarts make
        it the global one with high probability for small contexts.
    initial : np.ndarray, None
        A channel (``|W| x bound``) to warm-start from; if given, only it is used.
    prng : None, int, Generator
    maxiter : int
        Iterations per L-BFGS run.
    max_parameters : int
        Largest channel (``|W| * bound`` entries) to optimize. Beyond it the bound is
        reduced to fit, or, if even a binary :math:`\\bar W` does not fit, only the
        identity and constant channels are evaluated; either way the result is an
        upper bound on the intrinsic flow and a warning is issued.

    Returns
    -------
    value : float
        The intrinsic flow in bits, no larger than :math:`I[Y : X \\mid W]` or
        :math:`I[Y : X]` (both are evaluated exactly and the minimum is kept).
    channel : np.ndarray
        The minimizing channel, rows ``p(\\bar w \\mid w)``.
    """
    from dit.inference._symbols import as_generator

    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    Kw = p.shape[2]
    Kz = Kw if bound is None else int(bound)
    optimize = Kw > 1
    if Kw * Kz > max_parameters:
        import warnings

        if 2 * Kw <= max_parameters:
            Kz = max_parameters // Kw
            warnings.warn(
                f"{Kw} context values: channel bound reduced to {Kz}; the intrinsic flow is an upper bound.",
                stacklevel=2,
            )
        else:
            optimize = False
            warnings.warn(
                f"{Kw} context values: too many to optimize; the intrinsic flow is min(TE, TDMI), an upper bound.",
                stacklevel=2,
            )
    identity = np.eye(Kw, Kz)
    if Kz < Kw:
        identity[Kz:, 0] = 1.0
    constant = np.zeros((Kw, Kz))
    constant[:, 0] = 1.0
    candidates = [(cmi_from_joint(_degrade(p, identity)), identity), (cmi_from_joint(_degrade(p, constant)), constant)]
    if optimize:
        rng = as_generator(prng)
        if initial is not None:
            starts = [np.log(np.clip(initial, 1e-6, None))]
        else:
            starts = [6.0 * identity + 0.01 * rng.normal(size=(Kw, Kz)), 0.01 * rng.normal(size=(Kw, Kz))]
            starts += [rng.normal(scale=2.0, size=(Kw, Kz)) for _ in range(restarts)]
        for theta0 in starts:
            res = minimize(
                _objective, theta0.ravel(), args=(p, Kw, Kz), jac=True, method="L-BFGS-B", options={"maxiter": maxiter}
            )
            Q = _softmax_rows(res.x.reshape(Kw, Kz))
            candidates.append((cmi_from_joint(_degrade(p, Q)), Q))
    value, channel = min(candidates, key=lambda c: c[0])
    return max(value, 0.0), channel


@dataclass
class EdgeFlow:
    """
    The flow decomposition of one edge.

    Attributes
    ----------
    te, tdmi, intrinsic, synergistic, shared : float
        In bits; ``te = intrinsic + synergistic`` and ``tdmi = intrinsic + shared``.
    channel : np.ndarray
        The minimizing channel for the intrinsic flow (on the merged context).
    n_samples : int
        The number of aligned realizations.
    n_contexts : int
        The number of distinct context values used (after merging, if any).
    clipped : bool
        Whether a bias-corrected remainder was negative and clipped to zero.
    raw : dict
        The plug-in values on all data, for comparison.
    """

    te: float
    tdmi: float
    intrinsic: float
    synergistic: float
    shared: float
    channel: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    n_samples: int = 0
    n_contexts: int = 0
    clipped: bool = False
    raw: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "te": self.te,
            "tdmi": self.tdmi,
            "intrinsic": self.intrinsic,
            "synergistic": self.synergistic,
            "shared": self.shared,
        }


def layers_from_values(te, tdmi, intrinsic):
    """
    Enforce ``0 <= intrinsic <= min(te, tdmi)`` and return the three layers and a clip flag.
    """
    clipped = intrinsic < 0 or intrinsic > min(te, tdmi) or te < 0 or tdmi < 0
    te, tdmi = max(te, 0.0), max(tdmi, 0.0)
    intrinsic = min(max(intrinsic, 0.0), te, tdmi)
    return te, tdmi, intrinsic, te - intrinsic, tdmi - intrinsic, clipped


def edge_flows(y, x, w, estimator="plugin", bound=None, restarts=3, prng=None):
    """
    The flow decomposition of one edge from aligned samples, on a single joint.

    Parameters
    ----------
    y : array_like
        The target's present, one per realization.
    x : array_like
        The source's past (1D codes or a 2D array whose rows are joint values).
    w : array_like or None
        The context (1D codes or 2D rows); None for an empty context.
    estimator : {'plugin', 'miller_madow'}
        Applied to TE, TDMI, and the intrinsic flow at the minimizing channel.
    bound, restarts, prng
        Passed to :func:`intrinsic_flow`.

    Returns
    -------
    EdgeFlow
    """
    y, _ = dense_codes(y)
    x, _ = dense_codes(x)
    w, Kw = dense_codes(np.zeros(len(y), dtype=np.int64) if w is None else w)
    table = joint_table(y, x, w)
    n = table.sum()
    iif_plugin, Q = intrinsic_flow(table, bound=bound, restarts=restarts, prng=prng)
    te = cmi_from_joint(table, estimator)
    tdmi = cmi_from_joint(table.sum(axis=2, keepdims=True), estimator)
    iif = cmi_from_joint(_degrade(table, Q), estimator, n=n)
    te, tdmi, iif, syn, shared, clipped = layers_from_values(te, tdmi, iif)
    raw = {
        "te": cmi_from_joint(table),
        "tdmi": cmi_from_joint(table.sum(axis=2, keepdims=True)),
        "intrinsic": iif_plugin,
    }
    return EdgeFlow(te, tdmi, iif, syn, shared, Q, int(n), Kw, clipped, raw)
