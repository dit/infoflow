Preprocessing
=============

:func:`infoflow.preprocess.preprocess` turns continuous series into symbols with
as few free parameters as possible. For each process it

1. **screens** the series: splits at missing values, measures ties, and tests
   stationarity with a G-test on ordinal-pattern frequencies across segments;
2. proposes **delays** on the rank (copula) transform: the first minimum of the
   auto mutual information :cite:`Garland2014`, the first minimum of the smoothed
   mean Menger curvature of the delay reconstruction :cite:`Deshmukh2021`, and the
   autocorrelation decay time;
3. scores **discretizers** by cross-validated predictability of a multi-resolution
   reference target (equal-frequency 2/4/8-bin codes of the next value, each
   normalized as in :cite:`Rudelt2021`) and keeps the smallest alphabet within one
   standard error of the best, a forecast-driven choice in the spirit of
   :cite:`Garland2015,Garland2016`. Candidates are equal-frequency and equal-width
   bins, a median threshold, and ordinal patterns :cite:`Bandt2002` (with the
   present encoded as the bin of the current value alone, so it carries nothing
   about past values); weighted permutation entropy
   :cite:`Fadlallah2013` breaks ties between ordinal candidates;
   A node whose own future no candidate predicts gives no basis for a choice; it
   keeps equal-frequency bins at the middle reference resolution and is flagged
   ``unpredictable``;
4. chooses a **lag budget** with the bootstrap plateau rule of :cite:`Rudelt2021`,
   cross-checked against a BIC Markov order.

Surrogates are generated on the raw series and re-encoded
(:func:`~infoflow.preprocess.discretizers.reencode`), so permutation tests see the
same symbolization as the data :cite:`Staniek2008`.

:mod:`infoflow.preprocess.scaling` finds scaling regions in block-entropy curves
:cite:`Deshmukh2020` for entropy rate and excess entropy summaries; delay
reconstruction for nonlinear systems follows :cite:`Bradley2015`.

The result's ``report.summary()`` records every candidate and the reason each
choice was made.

Network resolution
------------------

Each node's symbols are chosen to predict that node's own future, but parent
selection conditions a target's past *and* several parents on the same samples. On
smoothly sampled flows the per-node rule picks fine symbols with long histories
(6 to 8 bins over two steps on Lorenz-96 :cite:`Lorenz1996`), and the joint contexts
of the multivariate tests then hold too few samples per cell for the plug-in
estimate to detect a parent. ``preprocess(network_resolution="auto")`` (or
``infer_multiplex(data, preprocess={"network_resolution": "auto"})``) scores uniform
resolutions for parent selection instead
(:func:`~infoflow.preprocess.scoring.network_scores`: each target predicted from its
own past and its two most correlated sources, every variable in the same
equal-frequency bins) and gives every node the best one, provided every node keeps
at least ``network_retention`` (default 0.7) of its own per-node score.

The guard protects nodes whose own dynamics need fine symbols. On the MuTE network
the oscillating driver keeps only 30-60% of its score at the network resolution;
coarsening it would let its children appear to predict it and each other. The
smallest retention over ten simulations of each system:

=============================  ===========================
System                         Smallest retention per node
=============================  ===========================
Lorenz-96, 1000 samples        0.73 - 0.81
Lorenz-96, 3000 samples        0.81 - 0.84
VAR random networks            0.76 - 0.89
MuTE, 3000 samples             0.29 - 0.63
MuTE, 1000 samples             0.28 - 0.74
coupled logistic maps          -0.43 - 0.76
=============================  ===========================

The threshold separates these cases except for one of ten short MuTE runs, so the
guard is a heuristic; ``network_resolution=True`` forces the uniform resolution and
``report.network`` records the scores, the retentions, and the decision. Unlike the
per-node choices this step looks across nodes, but it picks one of about ten
network-wide resolutions rather than anything edge-specific, and it did not raise
false positives on empty networks (0.4% of pairs against 1.1% with per-node
symbols).
