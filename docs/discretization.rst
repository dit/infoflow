Adaptive discretization
=======================

Weak edges into strongly driven targets are where fixed bins fail. In the MuTE
network :cite:`Montalto2014`, :math:`x_3` receives :math:`-0.5\,x_0^2` (variance
59) and a weak :math:`0.35\,x_4` (variance 2.3); eight equal-frequency bins of
:math:`x_0` leave a residual variance of about 20 from the :math:`x_0^2` term
inside the bins, which buries :math:`4 \to 3`. The literature offers several
alternatives to fixed bins that stay discrete:

* MDL-optimal adaptive histograms for conditional mutual information
  :cite:`Marx2021`, built on one-dimensional MDL histograms :cite:`Kontkanen2007`;
* NML-optimal, test-specific cut points found by dynamic programming
  :cite:`Cabeli2020`;
* recursive partitioning until local independence :cite:`Darbellay1999`;
* unsupervised interaction-preserving discretization :cite:`Nguyen2014`;
* variable-depth contexts by context-tree weighting for directed information
  :cite:`Jiao2013`.

Adaptive conditioning
---------------------

:mod:`infoflow.adaptive` implements the MDL histograms of :cite:`Kontkanen2007` and
the greedy joint histograms of :cite:`Marx2021`, on ranks (so that bins arise from
dependence, not from the shape of a margin), starting from a few equal-frequency
bins per column (with uniform margins a greedy search from one bin per column never
starts) and warm-started across conditioning sets that share variables.
``SkeletonSettings(estimator="adaptive")`` uses them for parent selection with one
change from :cite:`Marx2021`: the joint histogram is learned over the target and
the conditioning set only, and the target and candidates keep fixed equal-frequency
bins. A partition that also adapted to the candidate would fit the very dependence
being tested, and a permutation test against it would be anti-conservative.

On the conditioning-scaling benchmark (:doc:`benchmarks`; 2500 samples, true value
0.322 bits for the Gaussian, 0.456 for the binary channel, 0 when uncoupled), irrelevant conditioning variables
collapse to a single bin, so the adaptive estimate does not drift with their
number, where the plug-in estimate first inflates and then collapses:

========================  =======  =======  =======  =======  =======
Estimator                 d = 0    2        4        8        12
========================  =======  =======  =======  =======  =======
*Gaussian, coupled*
plug-in (4 bins)          0.231    0.289    0.793    0.020    0.000
adaptive                  0.231    0.244    0.239    0.244    0.225
KSG                       0.321    0.328    0.296    0.216    0.163
*Gaussian, uncoupled*
plug-in (4 bins)          0.002    0.044    0.667    0.024    0.000
adaptive                  0.002    0.002    0.002    0.001    0.003
*Binary, coupled*
plug-in                   0.450    0.450    0.445    0.525    0.177
adaptive                  0.450    0.450    0.442    0.461    0.459
========================  =======  =======  =======  =======  =======

The adaptive estimate's only loss is resolution (fixed target and candidate bins:
0.24 against 0.32 bits).

Weak edges in selection
-----------------------

On MuTE target :math:`x_3` (3 seeds, 10 000 samples), two further changes were needed
before any discrete estimator selected :math:`4 \to 3`:

* the free permutation null also breaks the candidate's dependence on the
  conditioning set and inflates its plug-in estimate, so the null sat above the
  observed value for every candidate (p = 1); ``null="strata"`` permutes within
  strata of the conditioning set instead;
* plug-in biases differ between candidates, and the raw maximum statistic picked the
  most biased candidates (lags 4–5 of :math:`x_4` and :math:`x_1`) rather than the
  most informative; ``statistic="debiased"`` ranks candidates by their estimate minus
  their own null mean.

=============================  =================  ===============================
Selection on :math:`x_3`       :math:`4 \to 3`    False parents
=============================  =================  ===============================
plug-in (default)              0 of 3 seeds       none
plug-in, strata, debiased      0 of 3 seeds       :math:`x_1` (1 seed)
adaptive, strata, debiased     3 of 3 seeds       :math:`x_1` at lags 1–5
KSG (local null)               3 of 3 seeds       none
=============================  =================  ===============================

Adaptive conditioning finds the weak edge without nearest-neighbour estimates, but
also selects lags of :math:`x_1` (which carries :math:`x_0^2`) that the partitioned
context does not fully absorb, and it is slow (15–50 minutes per target). Estimating
the layers on the adaptive partition (``infer_multiplex(layer_symbols="adaptive")``)
did not confirm :math:`4 \to 3` either: the layer estimates merge contexts before
optimizing, which removes most of the resolution the partition provides. For weak
edges into strongly driven targets, KSG selection remains the most reliable option,
and such edges carry the ``layers-unconfirmed`` flag.

Not implemented
---------------

* :cite:`Cabeli2020` chooses cut points per test to maximize the information minus
  its complexity, so the partition depends on the candidate; inside infoflow's
  permutation tests it would have to be re-learned for every permutation.
* :cite:`Darbellay1999` estimates mutual information only, and its conditional
  version is a difference of two separately partitioned estimates.
* :cite:`Nguyen2014` preserves interactions among all variables without reference to
  the target, which the target-aware partition above already does more directly.
* :cite:`Jiao2013` (context-tree weighting) addresses sparse symbol contexts rather
  than discretization, and remains a candidate for future work.
