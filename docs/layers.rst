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

Validation
----------

The test suite checks the estimates against exact flows of known processes
(computed with ``sofic``), and checks the qualitative benchmarks: an XOR target is
purely synergistic, a common driver produces shared flow, a chain produces a
mediated edge, and uncoupled processes produce no edges.
