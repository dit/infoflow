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
