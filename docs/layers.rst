Flow layers
===========

Estimation
----------

For a target :math:`Y`, candidate source past :math:`X`, and context :math:`W`
(the target's own selected past, the other selected parents, and any forced
conditionals), the intrinsic flow is

.. math::

   I[Y : X \downarrow W] = \min_{p(\bar w \mid w)} I[Y : X \mid \bar W],

the intrinsic conditional mutual information :cite:`Maurer1999` specialized to
flows :cite:`James2016`. It is minimized over softmax-parametrized channels with
L-BFGS and random restarts (:func:`infoflow.measures.intrinsic_flow`); the
identity channel recovers the transfer entropy and a constant channel recovers
the time-delayed mutual information, so both bound the minimum.

Plug-in estimates are biased upward, and the minimization overfits the channel to
sampling noise. :func:`infoflow.crossfit.crossfit_flows` therefore separates the
two in the style of double machine learning :cite:`Chernozhukov2018`: contexts are
merged (exactly when their predictive distributions coincide, statistically
otherwise, as CSSR merges histories :cite:`Shalizi2004`), the channel is fit on
one fold and evaluated with the Miller--Madow estimator :cite:`Miller1955` on the
other, and the folds are swapped and averaged.

Statistics
----------

:func:`infoflow.layers.layer_statistics` gives each layer

* a stationary-bootstrap confidence interval :cite:`Politis1994`;
* a layer-specific null: for the intrinsic layer, a parametric null that makes the
  source independent of the target given the fitted coarse context; for the
  synergistic layer, a permutation of the context within the strata of the fitted
  channel's hard clustering (keeping what the coarse context explains); for the
  shared layer, an unconditional permutation of the context;
* p-values adjusted per layer with Benjamini--Hochberg :cite:`Benjamini1995` or,
  for dependent tests, Benjamini--Yekutieli :cite:`Benjamini2001`.

With ``adaptive_resamples=True`` the number of null resamples grows with the number
of tests so that the smallest attainable p-value can survive the correction.

Intrinsic and synergistic flow are each at most the transfer entropy, so on edges
whose transfer entropy parent selection found non-significant (shared-only
candidates) neither layer is declared significant, whatever its own test says; the
estimates and p-values are still reported. Conversely, a selected parent whose
intrinsic and synergistic layers are both non-significant is flagged
``layers-unconfirmed``.

Hierarchical Bayesian layers
----------------------------

:func:`infoflow.bayes.layer_posterior` (``pip install infoflow[bayes]``) is an
experimental alternative to cross-fitting. Each context value's conditional law
:math:`p(y, x \mid w)` gets a shared prior :math:`\mathrm{Dirichlet}(\alpha m)`, the
hierarchical Dirichlet model of :cite:`MacKay1995` (see also :cite:`Gelman2013`),
so sparse contexts are shrunk toward the pooled law :math:`m` by an amount the data
choose through :math:`\alpha`: a soft version of context merging. PyMC samples
:math:`(\alpha, m)` from the Dirichlet-multinomial marginal likelihood; each
context's law is then conjugate and drawn in NumPy, and every posterior joint is
decomposed into layers, so the layer identities hold on every draw.

It gives credible intervals, not tests: posterior draws of a mutual information are
biased upward like the plug-in estimate :cite:`Archer2013`, so the nulls of
:func:`~infoflow.layers.layer_statistics` remain the way to decide significance.

With ``components=G > 1`` the contexts' laws are a mixture of G pooled laws
(context assignments summed out, :math:`\mathrm{Dirichlet}(1)` weights so that
unneeded components empty :cite:`Rousseau2011`): contexts are pooled within
groups, a soft version of merging into several clusters.

:func:`infoflow.bayes.log_bayes_factor` gives the closed-form BDeu Bayes factor
:cite:`Heckerman1995` for :math:`Y \not\perp X \mid W` (a nonzero transfer entropy),
which weighs evidence for the zero itself.

``validation/bayes_benchmark.py`` compares these with the existing estimators
(results in ``validation/results/bayes/results.json``):

* A single pooled law (``components=1``) loses to cross-fitting. It shrinks every
  context toward "the context does not matter", erasing the context dependence
  that synergy is made of: with 64 contexts and 1000 samples, XOR transfer entropy
  was biased down by 0.13 bits (cross-fitting: 0.08 bits up).
* A four-component mixture matches cross-fitting with few contexts and is much more
  accurate with many. Transfer-entropy RMSE, mixture against cross-fitting: XOR with
  64 contexts, 0.035 against 0.088 bits; XOR with 128 contexts and 500 samples,
  0.037 against 0.091; a "graded" edge where each of 128 contexts has its own
  coupling strength, 0.020 against 0.160 (cross-fitting overestimates the synergy
  by 0.16 bits). Shared flow on a common-driver edge is slightly worse (0.030
  against 0.025). It costs 4--12 s per edge against under 0.2 s.
* On :func:`~infoflow.benchmarks.conditioning_scaling` (``'hierarchical'``,
  ``'mixture'``), where the nuisance contexts are irrelevant, both remove most of
  the plug-in and Miller--Madow error at 256 contexts but are no better than the
  ``'adaptive'`` MDL partition.
* The Bayes factor is not a usable replacement for the permutation test of
  :func:`~infoflow.stats.conditional_mutual_information_test`. With the context
  large relative to the data it fails in both directions: 89% of independent edges
  favoured dependence at 250 samples and 128 contexts, and no copy coupling was
  detected at 1000 samples and 16 or more contexts (the permutation test: 2--8%
  false positives, 99--100% power). This is the BDeu prior's known sensitivity
  :cite:`Silander2007`; it also rules the Bayes factor out as a parent-selection
  score.

Further exploration of the mixture (same script, one JSON file per part):

* *Serially dependent data.* On the processes of the sofic validation tests
  (exact flows; target past padded with persistent nuisance chains, 2000 samples),
  the iid likelihood did not hurt. With 128 contexts, transfer-entropy RMSE was
  0.013--0.021 bits against 0.055--0.094 for cross-fitting; with 16 or fewer
  contexts the two agree.
* *Intervals.* For layers whose true value is positive, 90% credible intervals
  covered it in 70--100% of replicates (mostly 85--100%), and kept doing so with
  128 contexts, where the bootstrap intervals of
  :func:`~infoflow.layers.layer_statistics` inherit cross-fitting's bias and cover
  in 0% of replicates. Credible intervals essentially never contain an exact zero
  (posterior draws of an information are positive), so zero layers must still be
  decided by the nulls.
* *Crossover.* Over 4--256 contexts and 500--8000 samples, the mixture is clearly
  better below roughly 60 samples per context (transfer-entropy RMSE 0.02--0.03
  against 0.09--0.17) and ties above roughly 100.
* *Components and prior.* Two components capture the gain; 4, 8, or 16 add cost
  but no accuracy. The prior spread of :math:`\log \alpha` (0.5, 2, or 5) barely
  matters.
* *Speed.* One short chain (100 tuning and 100 draws, 50 posterior tables) is as
  accurate as two chains of 300 and 3.5--4 times faster; nutpie saved about 15%, and
  ADVI was less accurate without being faster. Two components with the short chain
  take 0.7--2.4 s per edge (128 contexts at the top end; 256 contexts with the
  default settings took 38 s).

* *End to end.* With the fast mixture swapped into
  :func:`~infoflow.network.infer_multiplex` for the layer weights and intervals
  (p-values unchanged) on XOR, common-driver, chain, and noisy ECA (rules 90 and
  110) networks, each edge scored against the same variables on a 200000-sample
  run: the contexts the skeleton selects are small (median 1--32 values), so the
  two agree almost everywhere. The exception is rule 110 at 500 samples (down to 8
  samples per context), where the mixture lowered the RMSE (transfer entropy 0.030
  against 0.039, intrinsic 0.010 against 0.014, synergistic 0.028 against 0.035)
  and raised the 95% coverage of positive layers (intrinsic 0.77 against 0.54,
  synergistic 0.89 against 0.72). Roles agreed on every edge, flags on 98--100%,
  and the :func:`~infoflow.benchmarks.motif_validation` checks all passed either
  way. Whole networks took 3--8 times longer.

Hence the Bayesian layers are opt-in and selective.
``infer_multiplex(layer_method='bayes')`` (or
:func:`~infoflow.layers.layer_statistics` with the same argument) replaces every
edge's cross-fitted weights and bootstrap intervals with posterior means and credible
intervals of the two-component mixture fitted with one short chain
(:data:`~infoflow.layers.BAYES_OPTIONS`); ``layer_method='auto'`` does so only for
edges with fewer than ``bayes_threshold`` (default 50) samples per observed context
value, which keeps the gains on sparse contexts at little cost elsewhere. The
p-values come from the nulls above in every case, and the method used for each edge
is reported in the dataset's ``layer_method`` variable.

Directed information
--------------------

Directed information :cite:`Massey1990`, with Kramer's causal conditioning
:cite:`Kramer1998`, is the information-theoretic form of Granger causality. In the
bivariate case it decomposes as :cite:`Amblard2013`

.. math::

   I[X \to Y] = T_{X \to Y} + I[X_t : Y_t \mid X_{\text{past}}, Y_{\text{past}}],

transfer entropy (the dynamical part, which the layers above decompose further)
plus the *instantaneous information exchange* (the contemporaneous part).
``infer_multiplex(directed_information=True)`` adds, over ``(source, target)``:

* ``iie``: the exchange of each pair, conditioned on both processes' selected pasts
  and lagged parents (:func:`infoflow.contemporaneous.instantaneous_exchange`, the
  statistic of the lag-0 layer), with a permutation p-value ``iie_pvalue``;
* ``di = weight[te] + iie`` for each estimated edge.

Because the transfer entropy conditions on the target's context and the exchange on
both processes' lagged parents, ``di`` is a causally conditioned directed
information. With side information the decomposition is only approximate:
instantaneous coupling through a third process blurs it :cite:`Amblard2013`. On
coarsely sampled data, where processes interact within a sample, ``iie`` carries
influence that the lagged layers cannot see. With the machine layer the composed
model's exact one-step directed information is added as ``model_di``.

Validation
----------

The test suite checks the estimates against exact flows of known processes
(computed with ``sofic``), and checks the qualitative benchmarks: an XOR target is
purely synergistic, a common driver produces shared flow, a chain produces a
mediated edge, and uncoupled processes produce no edges.
