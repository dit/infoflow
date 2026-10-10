Validation
==========

:func:`infoflow.benchmarks.network_validation` reproduces the design of
:cite:`Novelli2019`: directed Erdős–Rényi networks with link probability
:math:`3/N` (expected in-degree 3), one random coupling lag in :math:`1..5` per
link, self-coupling 0.5 and incoming couplings summing to 0.4, simulated either as
a vector autoregression or as coupled logistic maps with Gaussian noise of standard
deviation 0.1. Inferred networks are scored by precision, recall, specificity, and
the lag error normalized by its chance value (0 is perfect, 1 is chance), both for
the selected parents (``skeleton``) and the intrinsic layer.
:func:`infoflow.benchmarks.motif_validation` checks layers, roles, and flags on
motifs with known answers. The scripts in ``validation/`` run both.

Results (plug-in selection on the automatic preprocessing, significance 0.01 at
every stage, 2 networks per configuration; ``validation/results/``):

=========================  =====  =====  =========  ======  ===========
Network                    N      T      Precision  Recall  Specificity
=========================  =====  =====  =========  ======  ===========
VAR                        10     1000   1.00       0.24    1.000
VAR                        10     3000   0.92       0.45    0.986
VAR                        30     3000   0.84       0.16    0.997
logistic maps              10     1000   1.00       0.39    1.000
logistic maps              10     3000   0.96       0.39    0.992
logistic maps              30     3000   0.96       0.38    0.998
=========================  =====  =====  =========  ======  ===========

On empty networks (every link false) the false-positive rate was 0.4% (10 nodes)
and 0.1% (30 nodes). Lags of recalled links were almost always exact. On the motifs,
every check passed in 10 of 10 replications except the chain's ``mediated`` role
(9 of 10).

Precision and specificity are close to those of :cite:`Novelli2019`, whose recall is
much higher because they used continuous estimators (Gaussian for the VAR, Kraskov
for the logistic maps). Part of infoflow's recall gap comes from the selection null:
a free permutation of the candidate also breaks its dependence on the conditioning
set, which spreads it over more joint cells and inflates its plug-in estimate, so the
null sits above the observed value for weak but real edges. Two options address it
(``SkeletonSettings(null="strata", statistic="debiased")``): permuting the candidate
within strata of the conditioning set, and ranking candidates by their estimate minus
their own null mean, since plug-in biases differ between candidates. On the same
10-node networks:

=========================  =====  ==========================  ==========================
Network                    T      Precision (free -> strata)  Recall (free -> strata)
=========================  =====  ==========================  ==========================
VAR                        1000   1.00 -> 0.85                0.24 -> 0.52
VAR                        3000   0.92 -> 0.85                0.45 -> 0.48
logistic maps              1000   1.00 -> 0.94                0.39 -> 0.71
logistic maps              3000   0.96 -> 0.97                0.39 -> 0.73
logistic maps (intrinsic)  3000   1.00 -> 1.00                0.21 -> 0.65
=========================  =====  ==========================  ==========================

The false-positive rate on empty networks rose from 0.4% to 1.5% (10 nodes) and from
0.1% to 0.3% (30 nodes). A
within-strata permutation also breaks the candidate's own autocorrelation, so it is
somewhat anti-conservative for strongly autocorrelated processes such as these VAR
series, while weakly autocorrelated ones (the logistic maps) keep their precision.
The options are therefore not the default.

Large networks
--------------

At the scale of :cite:`Novelli2019` (100 nodes, up to 10 000 samples; one network
per configuration; significance 0.001 at every selection stage, as in
:cite:`Novelli2019`; source pre-screening at 0.1 except for the Gaussian estimator;
``validation/run_large.sh``, ``validation/run_large_gaussian.sh``,
``validation/run_plugin_options.sh``; results in ``validation/results/alpha001/``).
The last numeric column is precision / recall of the intrinsic layer:

===============================  =====  =========  ======  ===========  ===========  =======
Configuration                    T      Precision  Recall  Specificity  Intrinsic    Runtime
===============================  =====  =========  ======  ===========  ===========  =======
VAR, plug-in                     1000   0.93       0.05    0.9999       0.93 / 0.04  9 min
VAR, plug-in                     3000   1.00       0.11    1.0000       1.00 / 0.11  17 min
VAR, plug-in                     10000  1.00       0.28    1.0000       1.00 / 0.23  47 min
VAR, plug-in, strata + debiased  10000  0.98       0.41    0.9998       1.00 / 0.32  62 min
VAR, trend                       1000   1.00       0.34    1.0000       1.00 / 0.31  17 min
VAR, trend                       3000   0.98       0.73    0.9996       0.99 / 0.59  62 min
VAR, trend                       10000  0.99       0.98    0.9998       0.99 / 0.95  9.6 h
VAR, coarse                      1000   1.00       0.10    1.0000       1.00 / 0.10  10 min
VAR, coarse                      3000   1.00       0.31    1.0000       1.00 / 0.26  25 min
VAR, coarse                      10000  1.00       0.63    1.0000       1.00 / 0.47  62 min
VAR, Gaussian                    1000   0.99       0.44    0.9999       0.99 / 0.35  15 min
VAR, Gaussian                    3000   1.00       0.85    1.0000       1.00 / 0.75  53 min
VAR, Gaussian                    10000  1.00       1.00    1.0000       1.00 / 0.99  3.3 h
logistic maps, plug-in           1000   1.00       0.11    1.0000       1.00 / 0.10  10 min
logistic maps, plug-in           3000   1.00       0.38    1.0000       1.00 / 0.23  28 min
logistic maps, plug-in           10000  1.00       0.57    1.0000       1.00 / 0.51  96 min
logistic maps, trend             10000  0.99       0.35    0.9999       1.00 / 0.30  110 min
logistic maps, coarse            10000  1.00       0.96    0.9999       1.00 / 0.90  2.2 h
logistic maps, Gaussian          10000  0.98       0.54    0.9997       0.99 / 0.53  61 min
===============================  =====  =========  ======  ===========  ===========  =======

On empty 100-node networks (10 000 samples) no estimator selected a single false
parent. Precision is at or near 1 everywhere, as :cite:`Novelli2019` report for this
level; what separates the estimators is recall, i.e. statistical power.

The plug-in CMI on symbols is weak on these networks because its test has
:math:`(K_x - 1)(K_y - 1) K_z` degrees of freedom: with four symbols per variable and
a one-variable context that is 36, against 1 for the linear-Gaussian estimator, while
each VAR parent carries only about 0.02 bits. Each selected parent multiplies
:math:`K_z` again. Two symbol estimators spend fewer degrees of freedom:

* ``estimator="trend"`` tests a stratified ordinal trend (one degree of freedom per
  candidate). On the VAR networks it comes close to the Gaussian estimator (recall
  0.98 at 10 000 samples, 0.73 at 3000) without leaving symbols, but it cannot see
  non-monotone couplings and recovers only a third of the logistic-map links.
* ``estimator="coarse"`` keeps the plug-in CMI on median-split candidates, with
  context bins that shrink as the conditioning set grows. It improves on the
  plug-in everywhere and is the strongest estimator here on the nonlinear logistic
  maps (recall 0.96 against 0.57 for the plug-in and 0.54 for the Gaussian); a
  median split cannot, however, see dependence confined to the tails of a source.

These networks reward the two estimators: couplings are weak, linear or (for the
logistic maps) through a weighted sum, and no node is a strong nonlinear driver of
several others. On the MuTE network, whose root drives two children nonlinearly,
both condition too coarsely and link the children to each other and to the root
(precision 0.38 for trend and 0.36 for coarse; see :doc:`auto`), so
:func:`~infoflow.infer` does not use them.

The strata null with the debiased statistic helps the plug-in more modestly (recall
0.28 to 0.41). The linear-Gaussian estimator, the one :cite:`Novelli2019` used for
VAR networks, recovers every true link at 10 000 samples, with exact lags, and
misses half the logistic-map links, as a linear estimator should.

The trend, coarse, and plug-in runs also estimate layers for shared-only candidates
(the Gaussian runs skip them with ``--no-shared``); these are not scored, but they
dominate some runtimes (the 9.6 h trend run), so runtimes compare only roughly.

KSG at this level was not run to completion. Sources removed by the pre-screen still
enter every max-statistic null (otherwise the screen's selection makes the test
anti-conservative), so each admitted parent needs 2000 permutations of KSG
estimates over nearly all of the 495 candidate variables, more than 13 hours per
target on one machine. An earlier run at significance 0.01, before that correction
and with the pair search at 0.05, recovered every one of the 41 true links into 20
sampled logistic-map targets with one false link (precision 0.98, recall 1.00) in
202 minutes.
