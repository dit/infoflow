Automatic inference
===================

:func:`infoflow.infer` infers a network without tuning: it diagnoses the data,
chooses the selection estimators and options, sizes the run to an optional
wall-clock budget, and reports every choice.

.. code-block:: python

    import infoflow

    result = infoflow.infer(data)                       # (samples, processes) array
    result = infoflow.infer(data, time_budget="2h")      # fit the run to two hours
    result = infoflow.infer(data, preset="thorough")     # 'fast', 'balanced', 'thorough'
    print(result.report)                                 # what was chosen, and why
    net = result.network                                 # a MultiplexNetwork

Any :class:`~infoflow.selection.SkeletonSettings` field or
:func:`~infoflow.infer_multiplex` parameter can be pinned as a keyword argument
(``estimator`` pins the selection estimator; ``layer_estimator`` the layers'
estimator); pinned values are used as given and never changed by the budget.

Pipeline
--------

1. **Diagnose.** Integer arrays are symbols; float arrays are preprocessed
   (:func:`~infoflow.preprocess.preprocess`: screening, discretizer and lag budget
   per node). The report lists the sample size, processes, trials, and flagged
   nodes.
2. **Choose the estimators.** Selection always uses the plug-in CMI on the
   preprocessing symbols. For continuous data it adds the linear-Gaussian estimator,
   in one greedy search at a Bonferroni-split level, for every target whose present
   is linear in its past (:func:`~infoflow.auto.linearity_pvalues`, below). Every
   selection stage is tested at 0.001, with enough permutations for the split level
   and curtailment (which never changes a decision).
3. **Pilot.** The selection is run on three sampled targets, with the seeds the full
   run will use, so their skeletons are reused. The pilot measures the time per
   target of each estimator, the time per edge of the layers (at the resample count
   the run will use), and of interpretation.
4. **Fit the budget.** The cost of the whole run is predicted from the pilot
   (targets, then edges, run ``threads`` at a time). If it exceeds ``time_budget``,
   options are dropped in a fixed order, KSG selection, layer estimates for
   shared-only candidates, hyperedges, then the source pre-screen is turned on,
   until it fits; the significance level is never loosened. With
   ``strict_budget=True`` an unattainable budget raises instead of warning.
5. **Run** :func:`~infoflow.infer_multiplex` and return an
   :class:`~infoflow.auto.AutoResult` with the network and an
   :class:`~infoflow.auto.AutoReport`: diagnostics, gated targets, the final
   estimators and options, downgrades and their reasons, predicted and actual time,
   which estimator admitted each parent, and warnings.

Presets
-------

=====================  =========================  =========================  ==============================
Option                 fast                       balanced (default)         thorough
=====================  =========================  =========================  ==============================
Selection, continuous  plug-in + gated Gaussian   plug-in + gated Gaussian   plug-in + gated Gaussian + KSG
Selection, discrete    plug-in                    plug-in                    plug-in
Interpretation         off                        on                         on
Shared-only layers     off                        off                        on
Hyperedges             off                        off                        on
Source pre-screen      above 30 processes         above 30 processes         above 30 processes
=====================  =========================  =========================  ==============================

The batched GPU kernels are used when PyTorch is installed.

Why the Gaussian estimator is gated
-----------------------------------

The linear-Gaussian estimator is by far the most powerful on linear couplings
(:doc:`validation`), but it conditions linearly. When a target depends
nonlinearly on what is conditioned on, linear residualization leaves part of that
dependence behind, and any other variable that carries it looks informative: on the
MuTE network :cite:`Montalto2014`, whose root drives two children nonlinearly, the
Gaussian estimator links the two children to each other. The plug-in estimator,
conditioning on the full symbols, does not.

:func:`~infoflow.auto.linearity_pvalues` regresses each target's present on its own
lags and its most correlated source lags and tests whether the residual still
depends on any regressor (G-tests in equal-frequency cells, Bonferroni-corrected).
If the target is linear in its past, the residual is independent of every
regressor; surviving dependence is nonlinear, including nonlinearity in a weighted
combination of inputs, which a test of squared terms misses. The test runs at the
lenient level 0.05, since a false alarm only costs the Gaussian estimator's power
for that target. On 100-node linear networks it flagged 2-4% of targets; it flagged
every target of the coupled logistic maps and exactly the two nonlinear targets of
MuTE.

The stratified-trend and coarse-symbol estimators are not in the presets: they
condition on coarse cells, and the information left inside a cell creates false
links on the same MuTE network whether or not the target is linear (precision 0.38
and 0.36). They remain available as ``estimator="trend"`` / ``"coarse"``.
