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
other selected sources, minus its permutation-null mean; the network's ``score``
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

Ten-node random networks :cite:`Novelli2019` and a ten-variable Lorenz-96 ring
(lag limit 2), 1000 samples, two simulations each, default infoflow settings,
baselines at level 0.01
(``network_validation(kind, n_nodes=(10,), methods=("infoflow", "var", "pcmciplus"), baseline_alpha=0.01, prng=0)``):

=============  ==========  =========  ======  ===========  =====  =====
Network        Method      Precision  Recall  Specificity  AUROC  AUPRC
=============  ==========  =========  ======  ===========  =====  =====
VAR            infoflow    0.96       0.48    0.993        0.92   0.82
VAR            VAR         1.00       0.92    1.000        1.00   1.00
VAR            PCMCI+      0.95       0.96    0.986        0.99   0.98
logistic maps  infoflow    0.94       0.53    0.993        0.93   0.87
logistic maps  VAR         1.00       0.16    1.000        0.81   0.67
logistic maps  PCMCI+      0.88       0.50    0.978        0.83   0.72
Lorenz-96      infoflow    0.89       0.28    0.983        0.69   0.68
Lorenz-96      infoflow\*  0.98       0.98    0.992        0.99   0.99
Lorenz-96      VAR         0.86       1.00    0.917        0.99   0.99
Lorenz-96      PCMCI+      1.00       0.37    1.000        0.74   0.71
=============  ==========  =========  ======  ===========  =====  =====

On linear dynamics the linear baselines are near perfect, and infoflow's symbol-based
selection trails in recall (``estimator="gaussian"`` closes that gap, see
:doc:`validation`). On the nonlinear logistic maps infoflow ranks links best
(AUROC 0.93 against 0.81 and 0.83) while the linear methods lose recall; this is the
regime its nonparametric estimates are for. Lorenz-96 was its weak case with the default per-node symbols: sampled smoothly, each
node is best predicted from its own past by fine symbols over two steps, and with
three parents in the conditioning set those contexts hold too few samples for the
plug-in tests. With ``preprocess={"network_resolution": "auto"}`` (marked \*;
:doc:`preprocessing`), which chooses one coarser resolution for parent selection,
infoflow recovers the ring (precision 0.98, recall 0.98) and ranks links as well as
the VAR, with far fewer false links: the VAR's and the continuous estimators' extra
links join nodes two steps apart on the ring, whose influence arrives within a
sampling step, while exact conditioning on symbols screens them off. On the VAR
networks the option trades precision for recall (0.86 / 0.72 against 0.96 / 0.48,
AUROC 0.91 against 0.92); on the logistic maps its guard keeps the per-node symbols,
so the results are unchanged. Runtime on Lorenz-96 (about 11 minutes per network) is
almost all layer estimation for the many shared-flow candidates and bootstrap
resamples; parent selection takes one to two seconds per target.

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
