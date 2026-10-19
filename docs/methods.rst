Further methods
===============

These additions adapt recent network-inference techniques (surveyed in
:doc:`literature`) to infoflow's information-theoretic pipeline.

Stability selection
-------------------

:func:`~infoflow.stability.stability_selection` repeats parent selection on
contiguous subsamples and keeps links selected stably, with false-discovery-rate
control that tolerates the subsamples' overlap. Following time-series knockoffs
:cite:`Chi2025` and derandomized knockoffs :cite:`Ren2024`, each subsample selects on
one half and permutation-tests every selected parent on the other (so its p-value is
valid despite the selection), Benjamini--Hochberg at ``alpha / 2`` gives per-subsample
e-values, and e-BH on their average controls the FDR at ``alpha``
:cite:`Vovk2021,Wang2022`. A link is then kept roughly when half the subsamples
select it, stability selection :cite:`Meinshausen2010` with a threshold set by the FDR
level, and the selection ``frequency`` is a continuous edge score.

.. code-block:: python

    from infoflow.stability import stability_selection

    res = stability_selection(data, n_subsamples=20, alpha=0.1, prng=0)
    res.selected                     # {(source, target), ...}
    res.dataset["frequency"]         # (source, target)

Regimes
-------

:func:`~infoflow.regimes.infer_regimes` learns regime-dependent skeletons as
Regime-PCMCI does :cite:`Saggioro2020`: time is cut into windows (the unit of
persistence), initialised by k-means on per-window lagged mutual information, and
the method alternates between inferring a skeleton per regime and reassigning each
window to the regime whose skeleton predicts it best (plug-in predictive
log-likelihood, the window's own counts left out of its regime's model).
:func:`~infoflow.regimes.compare_regimes` then infers a full network per regime and
tests their layer differences with :func:`~infoflow.compare.compare_networks`, the
windows being the units. On :func:`~infoflow.datasets.switching_coupling`, whose
coupling reverses every block, the regimes are recovered exactly, from k-means or from
a deliberately wrong start.

.. code-block:: python

    from infoflow.regimes import compare_regimes, infer_regimes

    res = infer_regimes(data, n_regimes=2, window=400, prng=0)
    res.labels                       # regime per window
    net_a, net_b, diff = compare_regimes(res, a=0, b=1)

Multiple datasets
-----------------

J-PCMCI+ :cite:`Gunther2023` pools datasets of one system with dataset-index
variables that absorb unobserved differences between them.
:func:`~infoflow.regimes.infer_with_context` does the same: one trial per dataset, a
``context`` process constant within each (:func:`~infoflow.regimes.add_context`;
labels can group datasets that share a context), forced into every conditioning set.
Whether it matters depends on the selection null: within-trial permutations (the
default for few or unequal trials) keep a dataset-level association in the null, so it
biases the estimated layers but not the tests, while whole-trial permutations (the
default for many equal-length trials) report it as links.

Mixed continuous and categorical data
-------------------------------------

``SkeletonSettings(estimator="mixed")`` extends the KSG estimator to series that mix
continuous and categorical processes. Categorical processes (declared in
``mixed_categorical``, or detected as raw series with few integer values) only ever
match within a category, so they act as conditions rather than numbers, which
:cite:`Popescu2025` found the most robust of the mixed-data nearest-neighbour CMI
estimators; tied neighbours are counted as in :cite:`Mesner2021`, and the local
permutation null runs within categorical strata. On purely categorical data the
estimate agrees with the plug-in value, and independent continuous variables give
values near zero.

Predictive transfer entropy
---------------------------

:func:`~infoflow.predictive.predictive_transfer_entropy` estimates transfer entropy as
the held-out difference in log-likelihood between forecasters of the target with and
without the source's past, as AGM-TE does :cite:`Kornai2025`. Built-in forecasters are
linear-Gaussian and smoothed categorical tables; any callable
``model(X_train, y_train) -> logpdf(X_test, y_test)`` plugs in, which is how neural
forecasters would be used. Overfitting lowers held-out estimates rather than inflating
them, so it stays near zero for independent series however many conditions are given.

Higher-order structure
----------------------

:mod:`infoflow.oinfo` computes the O-information rate of a set of processes
:cite:`Faes2022` (positive: redundancy-dominated; negative: synergy-dominated) as the
difference of two static O-informations :cite:`Rosas2019` of block variables
(present and past, and past alone) computed with dit, and its first-order (node) and
second-order (link) gradients :cite:`Scagliarini2023,Mijatovic2025`.

The OIR describes undirected, whole-system structure and is no substitute for the
directed synergistic layer. On the XOR motif both agree (negative OIR, synergistic
edges). On a noisy rule-90 cellular automaton, where each cell is the XOR of its
neighbours, infoflow marks every neighbour edge synergistic (about 0.86 bits), but
the ring is nearly deterministic, its joint entropy rate is small compared with its
parts', and the OIR and every link gradient are positive (redundancy-dominated),
neighbours or not. For a common driver acting at lag 2, the OIR is zero with a
one-step history and positive with two: the history must cover the coupling lags.
:func:`~infoflow.oinfo.compare_with_layers` puts the link gradients next to the
synergistic and shared layers.
