Multiscale spectral backbones
=============================

The optional :mod:`infoflow.backbone` module adapts :cite:`Raiola2026` to discover
structural-backbone candidates before the existing predictive lag selection and
intrinsic/shared/synergistic layers. It also works independently. Default network
inference and ``infoflow.infer`` presets are unchanged.

.. warning::

   The population reconstruction theorem assumes fully observed stationary linear
   Gaussian scalar processes, independent positive noises, fixed nonzero couplings,
   and one directed coupling per undirected forest edge. Finite-data spectral
   bootstrap tests are an **experimental, model-based addition**, not an exact
   finite-sample guarantee in the paper. Nonlinear/latent/correlated-noise systems
   can give misleading results even when the detected graph looks like a forest.

Finite-ratio adjacency, asymptotic direction
----------------------------------------------

For the paper's model, Table 1 and SM Eq. (S56) identify an undirected edge at finite
timescale ratio precisely when both its ordinary and all-other-process-conditioned
rates are positive. A common-ancestor pair can have positive ordinary rate and
zero partial rate; collider co-parents can have zero ordinary rate and positive
partial rate. Neither matrix alone suffices.

This population criterion does **not** justify thresholding positive sample rate
estimates. :func:`~infoflow.backbone.backbone_from_rates` is only for known
population/analytic spectra; it returns no statistical p-values and refuses results
from empirical ``spectral_rates``. Use ``infer_backbone`` on observations.

MIRMO labels describe the rescaled rate as the *system's timescale separation*
grows: 0 vanishes, 1 stays finite and positive, 2 diverges. On a supported forest
link, label 1 means faster to slower and label 2 slower to faster. A single finite
estimate cannot identify these limits. Increasing recording length at fixed
parameters, changing sampling frequency, or changing time units is not such an
experiment. Unknown labels are **not label 0**.

``infer_backbone`` therefore returns undirected evidence. ``orient_backbone``
accepts explicit, validated labels with ``provided`` or ``model`` provenance and a
strict process order (fastest first); unsupported/conflicting directions remain
unset. ``rate_scaling`` reports finite-ratio curves and log-log regression slopes
across acknowledged comparable conditions, but deliberately leaves labels unknown:
a growing curve may still converge to a constant. Its regression standard error
is not measurement uncertainty. The graph's directed export contains resolved
arrows only, so inspect the undirected graph as well.

Experimental spectral tests
---------------------------

``infer_backbone`` computes Welch spectra once, then fits two Gaussian nulls per
unordered pair and re-estimates each surrogate with the same settings:

1. A marginal null synthesizes independent processes with the fitted autospectra.
2. A conditional null preserves the fitted spectrum of the conditioning set and
   each pair member's dependence on it. In the Schur residual spectrum, only the
   residual cross-spectrum is removed. This is not a free time-row permutation,
   a circular shift, or a phase rotation of the estimated cross-spectrum.

For adjacency, the union null is marginal **or** conditional independence. Its
intersection-union p-value is ``max(p_marginal, p_conditional)``. BY is applied once
across the fixed unordered-pair family by default :cite:`Benjamini2001`;
``dependent=False`` requests BH with its usual dependence assumptions
:cite:`Benjamini1995`. This combination is valid if its component p-values are
valid; it does not turn fitted Gaussian-bootstrap p-values into exact tests.
Unestimable pairs stay in the correction family and retain unresolved status.
A p-value at the Monte Carlo floor that cannot survive correction is also
``budget_unresolved`` and retained in the candidate handoff, not treated as an
absent link.

``BackboneSettings(n_null=199, alpha=0.05)`` controls the budget and level. The
Monte Carlo floor is ``1/(n_null+1)``; a warning reports budgets that cannot resolve
an isolated first-threshold discovery. Result settings retain pair seeds,
regularization and the statistical-method label. ``nulls`` preserves each pair's
null statistics. Non-rejection is reported as ``not_detected``, not proof of
independence. See the bounded validation in ``reports/multiscale_validation.json``
and ``reports/multiscale_validation.md``; validate the chosen window, sample count,
dimension and dynamics for each new application.

Independent discovery, selection and layer estimation
------------------------------------------------------

.. code-block:: python

    import numpy as np
    import infoflow
    from infoflow.preprocess import EqualFrequency, discretize
    from infoflow.preprocess.discretizers import reencode

    # d0, d1, d2 are independent synchronized recordings of the same system,
    # with identical process order and length for d1 and d2.
    backbone = infoflow.infer_backbone(
        d0, fs=fs, timescales=tau, assume_forest=True, prng=0,
    )
    template = discretize(d0, EqualFrequency(3))
    # Encoders are frozen. The concatenation boundary is exactly the holdout
    # boundary: neither selected lag windows nor evaluation windows cross it.
    analysis = reencode(template, [np.concatenate([d1, d2])])
    net = infoflow.infer_multiplex(
        analysis, embeddings=infoflow.Embedding(max_lag=5),
        backbone=backbone, holdout=0.5, prng=1,
    )
    net.backbone                     # separate structural evidence and rates
    net.dataset["parent_candidate"]  # source family, not a significance decision
    net.dataset["backbone_unconfirmed"]
    net.layer("intrinsic")           # ordinary, separately tested flow layers

A complete runnable example is ``examples/multiscale_backbone.py``. The exact
OU simulator ``datasets.multiscale_ou`` initializes stationarily and uses the
matrix-exponential sampled transition, rather than a stiff Euler approximation.

For one long recording, use a documented mixing/gap strategy rather than assuming
adjacent pieces independent. ``holdout`` separates TE selection from layer
estimation *within its own input*; it cannot undo earlier spectral discovery on
those evaluation observations. Raw ``infer_multiplex`` also preprocesses before
its holdout split. The recipe above avoids refitting encoders on evaluation data;
learn automatic embeddings only on discovery/training data too. Exact reuse of a
discovery trial is warned about, but absence of an exact fingerprint match does
not prove that windows are disjoint. Same-data exploratory results must be labeled
as such; their ordinary downstream p-values do not account for the adaptive screen.

Handoff semantics
-----------------

``backbone.candidate_sources()`` returns candidate source indices per target.
Both directions of a structural neighbor are offered, even when explicit structural
arrows are available. Numerically unresolved pairs are retained conservatively.
``infer_skeleton(candidate_sources=...)`` also accepts a complete per-target map;
``None`` means all sources and an empty list means target history only.

``infer_multiplex(backbone=...)`` validates process names/order and the explicit
forest-model acknowledgment. A detected undirected cycle prevents hard filtering;
no maximum spanning tree is used to force a forest. A graph looking like a forest
does not validate the assumptions or rule out missing weak links.

The source mask narrows parent discovery, not the conditioning model or output
meaning:

* Target history, source lags, synergy pair search, pruning, sequential/omnibus TE
  tests and layer-specific nulls still run. Forced/Faes conditionals are preserved.
* TE-selected parents are not manufactured from spectral edges. The spectral graph
  remains attached when predictive selection or symbol-based layers cannot confirm
  its links. ``backbone_te_confirmed``, ``backbone_unconfirmed`` and
  ``backbone_direction_disagreement`` keep those distinctions visible.
* TDMI screening still considers **all nonself processes**, preserving shared-only
  dependence between two children of a common driver even though they are not
  structural neighbors. ``include_shared_candidates=False`` or
  ``SkeletonSettings(tdmi_screen=False)`` explicitly opts out.
* An excluded parent candidate is marked not tested, not rejected or explained
  away by multivariate TE. Shared explanations can still be supplied by the
  separate explanation tests that actually run.
* Candidate-guided checkpoint namespaces include data, splits, candidates, settings
  and RNG state; edge keys also include source/context variables. Old unrestricted
  cache entries are not silently reused.

The identities ``TE = intrinsic + synergistic`` and
``TDMI = intrinsic + shared`` remain unchanged and refer to one finite-history
symbol distribution. They are not identities for spectral MIR. A continuous-time
structural graph and a sampled predictive graph need not agree, and weak spectral
links can remain unresolved by discretized layers.

Validation and limits
---------------------

``validation/multiscale.py`` runs the 12 motifs, timescale ratios 2/5/10/20,
window/sample/sampling/coupling sensitivity, larger forests, conditional and marginal
null calibration, independent-data handoff comparisons, and nonlinear/latent/noise/
loop failures. Reported runtime includes spectral discovery and surrogate fitting,
not just the reduced parent search. Python tests cover analytic closed forms,
Schur/precision equivalence, real Fourier synthesis, all motifs, independent data,
cache invalidation, shared-only edges and existing layer identities.

Hard filtering trades recall for a smaller search. In particular, pure XOR-like
couplings can be invisible to Gaussian spectra while the unrestricted pair search
finds them. Correlated innovations, hidden drivers and nonstationarity can generate
spurious structural edges without an obvious graph conflict. Coarse sampling and
finite bandwidth can change precision-spectrum zeros. No automatic nonlinearity
or causal-validity certificate is provided; keep unrestricted inference available.
