# Multiscale spectral backbone validation

## Reproduction and scope

```sh
uv sync --extra dev --extra docs --python 3.14
uv run python validation/multiscale.py --study all --reps 20 --n 4096 \
    --n-null 99 --ratios 2 5 10 20 --output reports/multiscale_validation.json
```

Raw observations, seeds, window choices, warnings, p-value calibration summaries,
structural scores and timings are in [`multiscale_validation.json`](multiscale_validation.json).
This report describes the run created **2026-11-14T17:24:28Z**, on Python 3.14.7,
NumPy 2.5.3, SciPy 1.18.1 and dit 2.4.dev54. The implementation was uncommitted on
Infoflow base `4558d2f`; its installed package version still identifies that base,
not a separately published release.

The default grid used 4096 observations, `fs=2`, 128-sample Hann segments with
half overlap, no shrinkage, 99 null simulations and BY at **alpha=0.1**. Diffusion
coefficients are one. There were **148 spectral fits**, plus an independent-data
handoff comparison of three full predictive/layer pipelines. Spectral-fit time
summed to about **115 s**; these are single local timings, not repeated performance
benchmarks. Reported RSS is a process high-water mark, not per-case memory use.

**Conclusion: retain the feature as experimental/model-based and opt-in.** The
analytic implementation and integration have strong regression checks, but this
bounded experiment does not establish universal finite-sample FDR control,
orientation identifiability, or a reliable structural filter outside its model.

## Analytic and engineering checks

- The test suite checks all 12 directed three-node trees at finite ratios, including
  signed couplings and unequal diffusion, plus disconnected forests, node
  permutations, collider co-parents and the chain/fork sharing MIRMO labels.
- Two-node rates agree with paper S23/S36 after nats-to-bits conversion. White-noise
  normalization, one-sided endpoints, SciPy Welch agreement, trial weighting,
  Schur/precision equivalence and extreme measurement-unit invariance are covered.
- Rank deficiency is never reported as conditional independence; explicit
  shrinkage preserves original-rank diagnostics. The population-zero oracle refuses
  empirical rate estimates. Labels and directions remain unresolved for a single
  recording.
- A Monte Carlo floor that cannot survive correction is `budget_unresolved` and
  retained in candidate maps. It is not silently classified as an absent edge.
- Integration preserves genuine TE selection, all-pair shared-only screening,
  forced/Faes conditionals, layer identities, holdout boundaries, separate spectral
  provenance and guided checkpoint invalidation.

## Bounded null calibration

Each model had **20 independent replications**, all estimable. At nominal 0.1,
component-null rejection counts and exact 95% binomial intervals were:

- Independent processes: **3/20**, interval **[0.032, 0.379]**; estimated edge FDR
  0.15, FDP standard error 0.082.
- Common-driver children (conditional null): **2/20**, interval **[0.012, 0.317]**;
  one family had a spurious edge, mean FDP 0.0167.
- Chain endpoints (conditional null): **1/20**, interval **[0.0013, 0.249]**;
  no spurious adjacency after correction.
- Collider co-parents (marginal null): **0/20**, interval **[0, 0.168]**;
  no spurious adjacency after correction.

All intervals contain the nominal rate, but they are **wide**. In particular,
3/20 under the independent null is not a claim that the true false-positive rate
is below 0.1. These checks neither falsify the tested regime decisively nor certify
it. More replications, larger null budgets and application-specific validation are
needed for precise error-rate claims. The separate slow regression checks a fixed
40-replicate independent-null ensemble against a prespecified binomial bound.

The composite adjacency test is the maximum of marginal and conditional p-values,
followed by one correction over unordered pairs. Its correctness still depends on
the fitted Gaussian component tests being sufficiently calibrated; BY cannot repair
a misspecified or anti-conservative component null.

## Three-node motifs and timescale separation

All 12 motifs were simulated at each ratio. Across their 24 true links:

- Ratio 2: **24 recovered**, no false links.
- Ratio 5: **21 recovered**, no false links.
- Ratio 10: **16 recovered**, no false links.
- Ratio 20: **8 recovered**, no false links.

These are one recording per motif/ratio, not confidence intervals for recall.
The same observation count does not mean the same sampling quality: total duration
spanned 512, 81.92, 20.48 and 5.12 slowest relaxation times respectively. The fixed
Welch segment covers only a fraction of a slow timescale at the largest ratios.
Thus the loss of recall is not a refutation of the population theorem; it shows why
the theorem does not remove finite observation/window requirements. No orientation
was inferred from these single-condition recordings.

## Window, dimension and budget sensitivity

Short/long trajectories, longer segments, weaker couplings and coarse sampling
recovered the two links in the chosen three-node fork. The finer-sampling case
recovered both but also a spurious edge. These isolated cases do not define a
universal safe window or sampling interval.

The six-node star and chain had no FDR-confirmed edges with 99 nulls. With 15
unordered pairs, the Monte Carlo resolution and BY penalty are restrictive;
**3 and 4 pairs**, respectively, were budget-unresolved and remain in candidate
maps. It would be incorrect to claim their generating networks are empty.
An eight-node case with too few spectral replicates left all **28 pairs**
numerically unresolved. Zero-padding would not fix that rank deficit.

The single regularized three-node case recovered both links, but this is not a
calibration study of regularization. Shrinkage changes spectral dependence and
must be reported and validated, not treated as a free numerical repair.

## Failure regimes

- Weak tanh/cubic perturbations retained both links; stronger versions lost one.
  These are Euler-reference nonlinear experiments, explicitly outside the Gaussian
  theorem, not a quantitative universal nonlinearity threshold.
- Pure XOR recovered neither true parent. Gaussian spectral screening is not a
  replacement for Infoflow's unrestricted synergy-aware pair search.
- Correlated innovations and a hidden common driver each generated a spurious
  structural edge **without** an obvious forest conflict. A forest-shaped result
  is not an assumption certificate.
- The triangle retained all three links and was flagged as an undirected cycle;
  candidate hard filtering is refused rather than forcing a tree.
- A reciprocal system has one undirected adjacency but two generating arrows;
  the one-arrow-per-edge theorem does not apply. No direction was manufactured.
- The particular nonstationary drift case had no detections, which is not evidence
  of general robustness to nonstationarity.

## Independent-data handoff and timing

Three independent OU recordings of 4096 samples each were allocated to discovery,
TE selection and layer evaluation. Bins were fitted on discovery only; the
selection/evaluation concatenation boundary was exactly `holdout=0.5`.
The empirical spectral screen found the fork's two links.

Unrestricted plug-in, unrestricted Gaussian and candidate-guided Gaussian pipelines
all selected `0 -> 1` and `0 -> 2`, confirmed their layers, preserved four
shared-only candidate directions, and satisfied `TE = intrinsic + synergistic`.
The shared-only directions include the nonadjacent children.

Illustrative total inference times were about **4.54 s**, **3.45 s**, and **3.70 s**,
respectively; the guided time includes its spectral discovery cost. Thus this run
**does not demonstrate a total speedup over unrestricted Gaussian inference**.
The guided downstream search was smaller, but the discovery cost matters.

The analytic *sampled-process* population precision graph for this example was
not a forest, despite the continuous generator being a fork. This is an important
sampling distinction: whole continuous trajectories, finite-band sampled
observations, and finite-lag predictive parents need not have the same independence
structure. The finite-sample screen's successful two-edge result is not a proof
that arbitrary sampled data inherit the continuous-time reconstruction theorem.

## Release posture

Keep default discovery and automatic presets unchanged. Expose the standalone
spectral analysis, clearly experimental null evidence, explicit-label orientation,
and optional candidate handoff with their warnings and unresolved states intact.
Do not advertise general nonlinear/latent/loop reconstruction, calibrated causal
orientation from one recording, a decomposition of spectral MIR into flow layers,
or a universal speed/recall improvement.

## Local release checks

- Python 3.14.7: full non-slow suite **368 passed, 13 optional-dependency skips**;
  focused new tests including bounded slow calibration **94 passed**.
- Python 3.11.15: focused compatibility tests **93 passed**, one slow test deselected.
- Ruff lint/format, type checking and Sphinx warnings-as-errors passed; the runnable
  independent-data example completed. Cross-platform CI was not run locally.
- The study provides rate error and calibration intervals, not independently
  validated per-rate confidence-interval coverage.
