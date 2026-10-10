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

Validation
----------

Every arm runs through :func:`~infoflow.infer` with the balanced preset (so all share
the 0.001 level, pre-screen, lag bound 5, and layer settings); ``infer`` is the
ensemble, the others pin one estimator. The 100-node networks are those of
:doc:`validation`, scored on 20 sampled targets; interpretation and the node layer
are off (they do not change these levels). ``validation/run_auto.sh``, results in
``validation/results/auto/``. Precision / recall of the selected parents, and of the
intrinsic layer:

========================  ===========  =========  ======  =====  ===========  =======
Scenario                  Arm          Precision  Recall  Links  Intrinsic    Runtime
========================  ===========  =========  ======  =====  ===========  =======
VAR, T = 3000             **infer**    1.00       0.80    53     1.00 / 0.73  15 min
\                         plug-in      1.00       0.06    4      1.00 / 0.06  3 min
\                         Gaussian     1.00       0.86    57     1.00 / 0.77  8 min
VAR, T = 10000            **infer**    1.00       1.00    60     1.00 / 1.00  60 min
\                         plug-in      1.00       0.35    21     1.00 / 0.30  13 min
\                         Gaussian     1.00       1.00    60     1.00 / 1.00  22 min
logistic maps, T = 10000  **infer**    1.00       0.80    33     1.00 / 0.78  41 min
\                         plug-in      1.00       0.80    33     1.00 / 0.78  24 min
\                         Gaussian     1.00       0.78    32     1.00 / 0.76  7 min
empty, T = 10000          all arms     no links                               1-3 min
MuTE, T = 3000            **infer**    1.00       0.80    4      1.00 / 0.80  1 min
\                         plug-in      1.00       0.80    4      1.00 / 0.80  < 1 min
\                         Gaussian     0.60       0.60    5      0.75 / 0.60  2 min
\                         trend        0.38       0.60    8      0.00 / 0.00  1 min
\                         coarse       0.36       0.80    11     0.44 / 0.80  4 min
orthogonal features       **infer**    1.00       1.00    2      1.00 / 1.00  < 1 min
\                         plug-in      1.00       1.00    2      1.00 / 1.00  < 1 min
\                         Gaussian     1.00       0.50    1      1.00 / 0.50  < 1 min
\                         trend        1.00       0.50    1      1.00 / 0.50  < 1 min
\                         coarse       1.00       0.50    1      1.00 / 0.50  < 1 min
XOR, common driver,       **infer**    1.00       1.00    2      (see note)   < 1 min
chain (discrete)
========================  ===========  =========  ======  =====  ===========  =======

``infer`` kept precision 1.00 in every scenario and selected nothing on the empty
network, while recovering at least 93% of the best single estimator's recall
everywhere: the Gaussian estimator's on the linear networks (13 times the plug-in's
recall at 3000 samples), the plug-in's on the nonlinear ones, where the gate turned
the Gaussian estimator off for every logistic-map target and for MuTE's two
nonlinear targets. The small loss at 3000 samples (0.80 against 0.86) is the price of
splitting the level two ways and of the few linear targets the gate turns off by
mistake. On the orthogonal-features process only the plug-in sees ``x2``'s
dependence on whether ``|x0|`` exceeds a quartile, which linear, trend, and
median-split statistics all miss. (XOR's edges are purely synergistic, so they have
no intrinsic flow.) Each scenario is one network, so small differences are noise.

The pilot's prediction of the run time, which counts preprocessing and the pilot
itself, came within 20% of the actual time on the 3000-sample VAR network (12.7
against 15.5 minutes); it is a guide for sizing runs, not a guarantee.
