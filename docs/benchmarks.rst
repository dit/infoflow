Estimator scaling
=================

Whether a weak edge can be found depends on how the conditional mutual information
estimate behaves as the conditioning set grows. :mod:`infoflow.benchmarks` measures
this with the design of :cite:`Young2021`: :math:`Y` depends on :math:`X` with a
known mutual information and the conditioning set holds :math:`d` independent
nuisance variables, so the true :math:`I[X : Y \mid Z]` does not change with
:math:`d`.

.. code-block:: python

    from infoflow.benchmarks import conditioning_scaling

    ds = conditioning_scaling("gaussian", n_samples=(500, 2500), dims=(0, 1, 2, 4, 8, 12), prng=0)
    ds["bias"].sel(n=2500)

Mean estimates in bits, 2500 samples, 5 datasets each (Gaussian variables are cut
into 4 equal-frequency bins for the symbol-based estimators):

==============================  ======  ======  ======  ======  ======  ======
Estimator, data                 d = 0   1       2       4       8       12
==============================  ======  ======  ======  ======  ======  ======
*Gaussian, true value 0.322*
plug-in                         0.228   0.252   0.282   0.796   0.022   0.000
Miller--Madow                   0.226   0.242   0.243   0.754   0.029   0.000
KSG                             0.311   0.340   0.326   0.312   0.228   0.173
*Gaussian, true value 0*
plug-in                         0.003   0.009   0.046   0.658   0.021   0.000
KSG                             -0.001  0.004   0.000   0.001   -0.003  -0.001
*Binary, true value 0.456*
plug-in                         0.459   0.456   0.461   0.464   0.525   0.173
Miller--Madow                   0.459   0.456   0.460   0.459   0.543   0.228
==============================  ======  ======  ======  ======  ======  ======

Symbol-based estimates first inflate as the joint cells thin out (the permutation
tests absorb this), then collapse towards zero once nearly every context value is
unique, and with them the power to detect anything. That is why a finely resolved
parent cannot be conditioned away with bins, and why a weak edge into the same
target can be missed. The KSG estimate on raw values never manufactures dependence
and decays only gradually and downward, as :cite:`Young2021` report for their
classifier-based estimator; it is available for parent selection as
``SkeletonSettings(estimator="ksg")``.

Rank metrics
------------

Thresholded precision and recall depend on each method's significance level. Recent
benchmarks therefore rank every candidate link by a continuous score and report the
AUROC and AUPRC :cite:`Cheng2024,Stein2025`. With
``SkeletonSettings(edge_scores=True)`` every candidate source gets a score, the
conditional transfer entropy of its best lag given the target's selected past and the
other selected sources, as a z-score against permutations (comparable across targets,
unlike raw estimates); the network's ``score``
variable holds them and :func:`~infoflow.benchmarks.score_network` reports ``auroc``
and ``auprc`` (:func:`~infoflow.benchmarks.rank_metrics`) beside the thresholded
metrics. :func:`~infoflow.benchmarks.network_validation` turns the scores on by default.

Baselines
---------

On real river networks a plain vector autoregression is the most robust method
:cite:`Stein2025`, so new methods should be compared with it.
:mod:`infoflow.baselines` provides it (:func:`~infoflow.baselines.var_granger`,
pairwise-conditional linear Granger causality) and Tigramite's PCMCI+ and LPCMCI with
partial correlation (:func:`~infoflow.baselines.tigramite_baseline`, needs
``infoflow[causal]``). ``network_validation(methods=("infoflow", "var", "pcmciplus"))``
runs every method on the same graphs and simulations; the results gain an
``algorithm`` dimension. ``kind="lorenz96"`` uses the Lorenz-96 ring
(:func:`~infoflow.datasets.lorenz96`, :func:`~infoflow.benchmarks.lorenz96_graph`)
:cite:`Lorenz1996,Tank2022`. ``validation/novelli.py`` exposes both as ``--methods``
and ``--kind lorenz96``.

Random networks :cite:`Novelli2019` and Lorenz-96 rings with 10 and 20 nodes, 1000
samples, five simulations each, :func:`infoflow.infer` (fast preset, with edge scores)
against the baselines at level 0.01
(``network_validation(kind, n_nodes=(10, 20), n_reps=5, methods=("auto", "var", "pcmciplus"), baseline_alpha=0.01, prng=0)``;
lag limit 5, or 2 for Lorenz-96). AUROC is the mean over simulations (standard
deviations 0.00-0.06):

=============  =====  =========  =========  ======  =====  =====
Network        Nodes  Method     Precision  Recall  AUROC  AUPRC
=============  =====  =========  =========  ======  =====  =====
VAR            10     infer      0.99       0.79    0.99   0.99
VAR            10     VAR        1.00       0.77    0.98   0.98
VAR            10     PCMCI+     0.98       0.90    0.97   0.97
VAR            20     infer      1.00       0.57    0.97   0.92
VAR            20     VAR        0.99       0.49    0.95   0.88
VAR            20     PCMCI+     0.96       0.78    0.93   0.87
logistic maps  10     infer      1.00       0.33    0.93   0.90
logistic maps  10     VAR        1.00       0.09    0.76   0.65
logistic maps  10     PCMCI+     0.79       0.33    0.76   0.66
logistic maps  20     infer      1.00       0.22    0.92   0.80
logistic maps  20     VAR        1.00       0.05    0.72   0.41
logistic maps  20     PCMCI+     0.53       0.27    0.73   0.43
Lorenz-96      10     infer      1.00       0.93    1.00   1.00
Lorenz-96      10     VAR        0.87       0.99    1.00   0.99
Lorenz-96      10     PCMCI+     1.00       0.41    0.74   0.71
Lorenz-96      20     infer      1.00       0.95    1.00   1.00
Lorenz-96      20     VAR        0.87       0.98    1.00   0.99
Lorenz-96      20     PCMCI+     0.99       0.42    0.80   0.65
=============  =====  =========  =========  ======  =====  =====

On these synthetic systems ``infer`` ranks links as well as or better than both
baselines everywhere, clearly so on the nonlinear logistic maps, and never loses
precision; at its strict selection level its thresholded recall trails PCMCI+ on the
linear networks. Lorenz-96 needed the guarded network resolution of
:doc:`preprocessing`: with per-node symbols (6 to 8 bins over two steps) and three
parents in the conditioning set, the plug-in tests are starved, and infoflow's recall
was 0.28 (AUROC 0.69). The VAR's extra links on Lorenz-96 join nodes two steps apart
on the ring, whose influence arrives within a sampling step; exact conditioning on
symbols screens them off. ``infer`` takes 1-15 minutes per network here against
seconds for the baselines, mostly in layer estimation.

External datasets
-----------------

:mod:`infoflow.io` reads the ground-truth benchmarks of the literature (download the
data from their sites; nothing is shipped):

* :func:`~infoflow.io.load_netsim`: one subject of a NetSim fMRI simulation
  :cite:`Smith2011`;
* :func:`~infoflow.io.load_causaltime`: a CausalTime dataset, keeping the observed
  :math:`N \times N` block of its :math:`2N \times 2N` graph :cite:`Cheng2024`;
* :func:`~infoflow.io.load_causalrivers` with :func:`~infoflow.io.sample_subgraph`:
  river-discharge subgraphs of CausalRivers :cite:`Stein2025`;
* :func:`~infoflow.io.to_causeme`: CauseMe's upload format (row-major ``scores``,
  ``pvalues``, ``lags`` matrices per dataset) :cite:`Runge2019nc`.

Every loader returns the data and a ``networkx.DiGraph``, which
:func:`~infoflow.benchmarks.score_edges` scores against.

Results on external benchmarks
------------------------------

``validation/external.py`` runs :func:`infoflow.infer` (fast preset, with edge scores)
against VAR and PCMCI+ (partial correlation, level 0.01) on downloaded benchmark data.
CausalTime's generated datasets could not be obtained (its download site was
unreachable; the code repository holds only the generator's inputs), and CauseMe
scores uploads on its own platform behind an account, so neither is included.

*NetSim* :cite:`Smith2011`, ten subjects per simulation, lag limit 2; AUROC for
directed links and, as in Smith et al.'s main analysis, for links in either direction:

==========  =====  ======  ===============  ===============  ===============
Simulation  Nodes  TRs     infer            VAR              PCMCI+
==========  =====  ======  ===============  ===============  ===============
1           5      200     0.62 / 0.67      0.53 / 0.54      0.56 / 0.61
2           10     200     0.62 / 0.66      0.65 / 0.62      0.70 / 0.68
3           15     200     0.61 / 0.65      0.59 / 0.65      0.68 / 0.73
5           5      1200    0.68 / 0.81      0.68 / 0.93      0.46 / 0.52
6           10     1200    0.77 / 0.80      0.80 / 0.90      0.58 / 0.63
==========  =====  ======  ===============  ===============  ===============

(directed / undirected AUROC.) Directions are hard for every method, as Smith et al.
found for lag-based methods on BOLD signals; with the long sessions the linear VAR
detects connections best, and ``infer`` is at the VAR's level for directions. At the
strict selection level ``infer`` declares few links on these short series (recall
0.00-0.13).

*CausalRivers* :cite:`Stein2025`, forty random connected subgraphs of the East German
network per size, resampled to 6 hours (24 samples) and 1 day (96), lag limit 3; mean
AUROC per graph, self-links ignored, as in the paper:

========================  =====  =====  ======
Subgraphs, resolution     infer  VAR    PCMCI+
========================  =====  =====  ======
3 nodes, 6 h              0.64   0.67   0.67
3 nodes, 1 day            0.58   0.60   0.55
5 nodes, 6 h              0.67   0.74   0.68
5 nodes, 1 day            0.56   0.62   0.58
========================  =====  =====  ======

The paper's best-tuned VAR reaches 0.82 and 0.80 on its full Random-3 and Random-5
sets, choosing resolution, lag, normalization, and coefficient handling with the
labels; with one fixed configuration our VAR is lower, and ``infer`` trails it by
0.03-0.07, at PCMCI+'s level. Its precision here is 0.24-0.41: discharge is smooth,
strongly autocorrelated, and driven by shared weather, so coarse symbols leave most
of each station's memory unconditioned and stations appear to predict each other.
Pinning the Gaussian estimator or turning network resolution off did not help on a
sample of these graphs. On this real-world benchmark, unlike the synthetic ones,
``infer`` is not competitive with the linear baseline.

Latent confounding
------------------

:func:`~infoflow.benchmarks.latent_validation` hides drivers with at least two
children from a random network and asks, for the observed pairs that share a hidden
driver but no link, how often each method reports a link (``spurious``) and how often
it flags the confounding (``flagged``: infoflow's latent flags, LPCMCI's ``<->``
:cite:`Gerhardus2020`). Eight-node VAR networks, one hidden driver, 3000 samples,
three networks (``latent_validation(methods=("infoflow", "var", "pcmciplus", "lpcmci"), prng=0)``):

==========  =========  ======  ========  =======
Method      Precision  Recall  Spurious  Flagged
==========  =========  ======  ========  =======
infoflow    0.83       0.59    0.14      0.11
VAR         0.90       1.00    0.14      --
PCMCI+      0.94       1.00    0.14      --
LPCMCI      0.87       0.95    0.19      0.11
==========  =========  ======  ========  =======

On linear data the linear baselines have the higher recall, as expected. No method
avoids the confounded pairs reliably, and both infoflow's flags and LPCMCI's marks
catch only a small fraction of them: hidden drivers remain the main failure mode.
