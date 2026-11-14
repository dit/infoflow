"""Independent multiscale discovery -> lag selection -> existing flow layers.

Run from the repository root: ``python examples/multiscale_backbone.py``.
No asymptotic direction is inferred from a single recording. The three stationary
OU recordings are generated independently; discovery also trains the frozen bins.
"""

import argparse

import numpy as np

import infoflow
from infoflow.preprocess import EqualFrequency, discretize
from infoflow.preprocess.discretizers import reencode


def demo(n=4096, n_null=199):
    tau, dt = [1.0, 2.0, 4.0], 0.5
    # A[target, source]: fast process 0 drives processes 1 and 2.
    A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    discovery, selection, evaluation = [
        infoflow.datasets.multiscale_ou(n, A, tau, dt=dt, seed=500 + k) for k in range(3)
    ]
    backbone = infoflow.infer_backbone(
        discovery,
        fs=1 / dt,
        timescales=tau,
        assume_forest=True,
        spectral_settings=infoflow.SpectralSettings(nperseg=128),
        settings=infoflow.BackboneSettings(n_null=n_null),
        prng=500,
    )
    print(backbone.summary())
    for note in backbone.warnings:
        print(f"Note: {note}")
    if not backbone.is_forest:
        print("Forest conflict: keep the spectral evidence, but do not hard-filter the predictive search.")
        return backbone, None
    template = discretize(discovery, EqualFrequency(3))
    analysis = reencode(template, [np.concatenate([selection, evaluation])])
    settings = infoflow.SkeletonSettings(
        estimator="gaussian",
        max_source_lag=5,
        n_perm_max_stat=99,
        n_perm_min_stat=99,
        n_perm_omnibus=199,
        n_perm_max_seq=199,
        n_perm_pairs=99,
        n_perm_tdmi=199,
    )
    net = infoflow.infer_multiplex(
        analysis,
        embeddings=infoflow.Embedding(5),
        backbone=backbone,
        holdout=0.5,
        skeleton_settings=settings,
        n_boot=20,
        n_null=99,
        adaptive_resamples=False,
        node_layer=False,
        prng=501,
    )
    print(net.summary())
    print("Spectral rate units:", backbone.rates.dataset.rate.attrs["units"])
    print("Flow layers remain per-symbol bits, not a decomposition of those spectral rates.")
    return backbone, net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--n-null", type=int, default=199)
    args = parser.parse_args()
    demo(args.n, args.n_null)
