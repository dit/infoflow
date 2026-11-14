"""Analytic multiscale OU oracles shared by the spectral/backbone tests."""

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

# Paper Table S2, with node indices shifted down by one. Pair order: 01, 12, 02.
MOTIFS = (
    (((0, 1), (1, 2)), (1, 1, 0)),
    (((0, 1), (2, 0)), (1, 2, 2)),
    (((1, 0), (0, 2)), (2, 1, 1)),
    (((0, 2), (2, 1)), (0, 2, 1)),
    (((2, 1), (1, 0)), (2, 2, 2)),
    (((1, 2), (2, 0)), (0, 1, 2)),
    (((0, 1), (2, 1)), (1, 2, 0)),
    (((0, 2), (1, 2)), (0, 1, 1)),
    (((1, 0), (2, 0)), (2, 0, 2)),
    (((0, 1), (0, 2)), (1, 0, 1)),
    (((1, 0), (1, 2)), (2, 1, 1)),
    (((2, 0), (2, 1)), (0, 2, 2)),
)


def coupling(edges, P=3, weight=1.0):
    A = np.zeros((P, P))
    for s, t in edges:
        A[t, s] = weight
    return A


def ou_spectrum(frequency, A, tau, diffusion=None):
    tau = np.asarray(tau)
    D = np.ones(len(tau)) if diffusion is None else np.asarray(diffusion)
    B = np.eye(len(tau))[None] * (1 + 2j * np.pi * np.asarray(frequency)[:, None] * tau)[..., None, :]
    B = B - A
    inv = np.linalg.inv(B)
    return (inv * (2 * D * tau)[None, None, :]) @ inv.conj().swapaxes(1, 2)


def sampled_spectrum(frequency, A, tau, dt, diffusion=None):
    tau = np.asarray(tau)
    D = np.ones(len(tau)) if diffusion is None else np.asarray(diffusion)
    F = (A - np.eye(len(tau))) / tau[:, None]
    cov = solve_continuous_lyapunov(F, -np.diag(2 * D / tau))
    transition = expm(F * dt)
    noise = cov - transition @ cov @ transition.T
    B = np.eye(len(tau))[None] - transition[None] * np.exp(-2j * np.pi * frequency * dt)[:, None, None]
    inv = np.linalg.inv(B)
    return dt * inv @ noise @ inv.conj().swapaxes(1, 2)
