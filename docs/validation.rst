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
per configuration, significance 0.01, source pre-screening at 0.1;
``validation/run_large.sh``):

===================================  =====  =========  ======  ===========  =======
Configuration                        T      Precision  Recall  Specificity  Runtime
===================================  =====  =========  ======  ===========  =======
VAR, plug-in                         1000   0.62       0.10    0.998        4 min
VAR, plug-in                         3000   0.88       0.20    0.999        7 min
VAR, plug-in                         10000  0.95       0.30    0.9995       16 min
VAR, plug-in, strata + debiased      10000  0.89       0.50    0.998        39 min
logistic maps, plug-in               1000   0.87       0.20    0.999        4 min
logistic maps, plug-in               3000   0.99       0.43    0.9999       11 min
logistic maps, plug-in               10000  0.95       0.59    0.999        31 min
logistic maps, KSG (20 targets)      10000  0.98       1.00    0.9995       202 min
VAR, Gaussian, no pre-screen         1000   0.93       0.56    0.999        11 min
VAR, Gaussian, no pre-screen         3000   0.96       0.85    0.999        29 min
VAR, Gaussian, no pre-screen         10000  0.94       1.00    0.998        122 min
logistic maps, Gaussian, no pre-sc.  10000  0.93       0.64    0.999        25 min
===================================  =====  =========  ======  ===========  =======

The plug-in and KSG rows were run before the pre-screened-out sources were added to
the max-statistic nulls, so their selections were somewhat anti-conservative (on the
empty network, 37 false parents with plug-in). The Gaussian rows use the corrected
selection without pre-screening, which for this estimator is also faster.

The intrinsic layer is more precise still (0.92–1.00) at somewhat lower recall. On
an empty 100-node network with 10 000 samples the false-positive rate was 0.4% for
selected parents and 0.06% in the intrinsic layer; with the Gaussian estimator and
no pre-screen it was 0.11% (11 links) and 0.03%.

With the KSG estimator on coupled logistic maps, every one of the 41 true links into
the 20 sampled targets was recovered with one false link, matching the
precision, recall, and specificity above 98% that :cite:`Novelli2019` report at this
size and length; it costs about ten minutes per target. On the linear VAR networks
the plug-in path stays conservative (recall 0.30, rising to 0.50 with the strata null
at a small cost in precision), and KSG is not the remedy there: each parent explains
only about 2% of its target's variance, which needs the efficiency of a parametric
(linear-Gaussian) estimator, the one :cite:`Novelli2019` used for these networks.
With ``estimator="gaussian"`` every true VAR link is recovered at 10 000 samples, with
exact lags and precision 0.94 (0.98 in the intrinsic layer). :cite:`Novelli2019`
report precision, recall, and specificity above 98% at this size and length, at the
stricter level 0.001 (here 0.01, which trades some precision for recall). On the
nonlinear logistic maps it misses about a third of the links, as a linear estimator
should; KSG is the estimator for those.
