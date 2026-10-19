Literature and roadmap
======================

This page surveys recent work (mostly 2020--2026) on inferring functional and
effective networks from multivariate time series, sorted into benchmarks and
method families, and records which ideas ``infoflow`` has adopted and which are
still open.

Benchmarks
----------

**Synthetic dynamics.** The neural Granger-causality literature evaluates on
vector autoregressions and the Lorenz-96 system :cite:`Lorenz1996,Tank2022`, in which
each variable is driven by its two lower neighbours and its upper neighbour, so the
true graph is known and the forcing :math:`F` tunes the dynamics from periodic to
chaotic. Random lagged networks simulated as VAR processes or coupled logistic maps
:cite:`Novelli2019` are the standard test for multivariate transfer entropy.

**Quasi-real simulations.** NetSim :cite:`Smith2011` simulates BOLD fMRI from
known networks of 5--50 regions over 28 configurations, and DREAM-3/4
:cite:`Marbach2010` simulate gene-regulatory dynamics. CausalTime :cite:`Cheng2024`
fits deep generative models to real air-quality, traffic, and medical recordings,
derives a ground-truth graph from the fitted model, and regenerates data from it; of
the 13 methods it benchmarks, almost none reach an AUROC of 0.9.

**Real systems with known structure.** CauseMe :cite:`Runge2019nc` is a leaderboard
that scores uploaded adjacency-score matrices on datasets with delays,
autocorrelation, nonlinearity, chaos, extreme events, and measurement error.
CausalRivers :cite:`Stein2025` provides river-discharge records for 666 and 494
gauging stations with flow-network ground truth and samples thousands of subgraphs
from them; its headline result is that a plain VAR is the most robust method and
that sophisticated methods rarely beat simple baselines. Engineered systems give
ground truth by construction: the European Spallation Source cooling system
:cite:`Mogensen2024` and the wind- and light-tunnel causal chambers
:cite:`Gamella2025`.

**Neural data.** :cite:`Donner2024` benchmark spike-train connectivity inference
(including transfer entropy) on leaky integrate-and-fire networks and on cultures
whose synaptic connections were labelled by simultaneous patch clamp; performance
depends strongly on the dynamical regime and recording length. Continuous-time
transfer entropy :cite:`Shorten2021` used inside the greedy multivariate network
inference recovers spiking-network ground truth better than pairwise TE, CNN, and
GLM baselines, particularly under synchrony :cite:`Shorten2025`.

**Functional connectivity without ground truth.** pyspi :cite:`Cliff2023` collects
over 200 statistics of pairwise interactions and compares them on 1053 multivariate
time series. :cite:`Liu2025` use it to benchmark 239 fMRI connectivity statistics on
structure--function coupling, fingerprinting, and behaviour prediction; precision
(partial correlation) and covariance score best. Such benchmarks measure usefulness
rather than correctness and suit evaluating ``infoflow``'s layers as features.

**Metrics.** Rank-based scores (AUROC and AUPRC over a continuous edge score) are now
standard :cite:`Cheng2024,Stein2025` and allow comparing methods without matching
significance thresholds. Thresholded precision, recall, and specificity remain the
norm for TE network inference :cite:`Novelli2019`.

Method families
---------------

**Latent confounding, regimes, and multiple datasets.** LPCMCI
:cite:`Gerhardus2020` returns a time-series PAG that marks links that may be due to
hidden common causes. Regime-PCMCI :cite:`Saggioro2020` alternates between assigning
time points to a small number of persistent regimes and learning one graph per
regime. J-PCMCI+ :cite:`Gunther2023` pools datasets of the same system, adding
dataset-index dummy variables to absorb unobserved context differences.

**Estimators.** Mixed continuous/categorical conditional independence tests use
nearest-neighbour CMI estimators :cite:`Mesner2021` with a local permutation null
:cite:`Runge2018`; :cite:`Popescu2025` find that treating categorical variables
as conditions only, rather than one-hot coordinates, is the most robust variant.
AGM-TE :cite:`Kornai2025` estimates transfer entropy as the held-out difference in
log-likelihood between two probabilistic forecasters, with and without the source,
and handles conditioning sets naturally; TREET :cite:`Luxembourg2024` uses a
transformer with a Donsker--Varadhan bound.

**Error control.** Time-series knockoffs :cite:`Chi2025,Zuo2024` handle serial
dependence by running selection on subsamples and aggregating with e-values
:cite:`Vovk2021`; e-values combine across dependent runs by averaging and admit FDR
control through the e-BH procedure :cite:`Wang2022,Ren2024`. Stability selection
:cite:`Meinshausen2010` reports how often each variable is selected over subsamples.

**Higher-order structure.** The O-information rate :cite:`Faes2022` extends the
O-information :cite:`Rosas2019` to processes; its local version (for a link) and
gradients (for a node) :cite:`Scagliarini2023` give a network representation of
redundancy- versus synergy-dominated interactions :cite:`Mijatovic2025`.

Gap analysis
------------

==================================  ======================================================
Idea                                Status in ``infoflow``
==================================  ======================================================
Random VAR / logistic networks      :func:`~infoflow.benchmarks.network_validation`
AUROC / AUPRC                       :func:`~infoflow.benchmarks.rank_metrics` over
                                    ``SkeletonSettings(edge_scores=True)``
Lorenz-96                           :func:`~infoflow.datasets.lorenz96`,
                                    ``network_validation(kind="lorenz96")``
NetSim, CausalTime, CausalRivers    :func:`~infoflow.io.load_netsim`,
                                    :func:`~infoflow.io.load_causaltime`,
                                    :func:`~infoflow.io.load_causalrivers`
CauseMe                             :func:`~infoflow.io.to_causeme`
VAR, PCMCI+, LPCMCI baselines       :mod:`infoflow.baselines`,
                                    ``network_validation(methods=...)``
Latent confounding                  :func:`~infoflow.interpret.latent_flags`, compared
                                    with LPCMCI by
                                    :func:`~infoflow.benchmarks.latent_validation`
Regimes                             :func:`~infoflow.regimes.infer_regimes`,
                                    :func:`~infoflow.regimes.compare_regimes`
Multiple datasets                   :func:`~infoflow.regimes.infer_with_context`
Stability selection / e-values      :func:`~infoflow.stability.stability_selection`
Mixed-type CMI                      ``SkeletonSettings(estimator="mixed")``
Predictive (AGM-TE style) TE        :func:`~infoflow.predictive.predictive_transfer_entropy`
O-information rate                  :mod:`infoflow.oinfo`
Continuous-time spike TE            open: needs an event-time data model
                                    :cite:`Shorten2021`
Neural forecasters (AGM-TE, TREET)  open: the predictive estimator accepts any forecaster
                                    as a callable
pyspi-style FC benchmarks           open: no ground truth; would evaluate layers as
                                    features :cite:`Liu2025`
==================================  ======================================================

Roadmap
-------

The implementation order followed dependencies: rank metrics first, because the
external benchmarks and stability selection report through them; then the
simulators and loaders; then baselines in the validation harness; then the method
additions (stability selection, regimes and contexts, estimators, higher-order
diagnostics, :doc:`methods`). Each item cites its source in its docstring. What the
first runs showed (:doc:`benchmarks`):

* Against VAR and PCMCI+ baselines, infoflow ranks links best on nonlinear (logistic
  map) networks but trails the linear methods on linear ones and, markedly, on the
  smoothly sampled Lorenz-96 ring (AUROC 0.69 against 0.99 for VAR), in line with the
  CausalRivers finding that linear baselines are hard to beat :cite:`Stein2025`.
  Continuous estimators in selection (``estimator="gaussian"``, ``"ksg"``, or an
  ensemble) are the first thing to try there.
* Hidden drivers remain the main failure mode for every method: infoflow's latent
  flags and LPCMCI's bidirected marks each catch only a small fraction of confounded
  pairs.
* The O-information rate is not a proxy for the synergistic layer: on a rule-90
  cellular automaton it reports redundancy where every edge is synergistic.

Open next steps: run the CausalTime, CausalRivers, and NetSim loaders on the released
data and report the tables; an event-time data model for continuous-time spike TE
:cite:`Shorten2021,Shorten2025`; neural forecasters for the predictive estimator
:cite:`Kornai2025`; and pyspi-style usefulness benchmarks of the layers as features
:cite:`Cliff2023,Liu2025`.
