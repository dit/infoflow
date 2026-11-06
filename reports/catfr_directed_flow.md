# Directed flow between MTL and PFC in catFR1

*Branch `catfr-directed-flow`, November 2026. Scripts: `examples/catfr/flow.py` (prepare,
analyze), `validation/synergy_null.py`; follows `reports/catfr_pilot.md` and
`reports/dit_estimators.md`.*

## Summary

- **No directed MTL-PFC flow that differs with memory.** Across 88 subjects (encoding) and
  56 (retrieval), the pre-specified test of intrinsic flow MTL -> PFC and PFC -> MTL at
  20-200 ms found no difference between recalled and forgotten words or between the
  1.6 s before correct recalls and deliberation, in any layer. Nor did intrinsic flow
  exceed a shuffled-source surrogate in the pre-specified analysis.
- **Shared flow is there throughout**, in both directions, at every lag and in both phases
  (real > shuffled source, q < 0.01): MTL and PFC co-fluctuate within trials, beyond what
  each region's time course explains, but not in a way the target's past and LTC do not
  already carry or that is directed.
- **Two artifacts found and removed along the way.** Overlapping retrieval windows made
  bias-corrected estimates look like strong synergy (and a strong recall effect); a shared-flow
  recall effect at 200 ms came from both regions following the same pre-recall trajectory.
- **Exploratory:** with each condition's mean time course removed, intrinsic flow exceeded
  the surrogate in 3 of 8 encoding tests (MTL -> PFC at 20 and 40 ms, PFC -> MTL at 200 ms,
  q = 0.012-0.031), by about 0.0001 bits. This was not pre-specified and is not a finding.
- **Open issues closed:** `compare_networks(estimator=...)` (the 25-subject memory contrast
  rerun with Chao-Wang-Jost: still null); the synergistic null excess traced to chance;
  dit's broken `prng` alias fixed upstream (dit #324).

## 1. Design (fixed before the analysis)

- **Subjects:** every catFR1 session that `cohort.choose` accepts (one per subject, two or
  more non-overlapping bipolar pairs in each of MTL, LTC and PFC, 40 or more trials per
  encoding condition): 91 subjects, 88 prepared (3 have no usable epochs). Each EDF is
  downloaded, epoched and deleted (`flow.py prepare`). A subject enters a phase with 30 or
  more epochs per condition after balancing: 88 for encoding, 56 for retrieval (median 71 and
  43 epochs per condition).
- **Nodes:** each region's HFA (70-150 Hz log power, 50 Hz), every pair z-scored and then
  averaged, cut into 4 equal-frequency bins per subject.
- **Edges:** MTL -> PFC and PFC -> MTL, source at lag 1, 2, 5, 10 samples (20, 40, 100,
  200 ms), conditioned on the target's past (lags 1-2) and LTC at the source lag; layers
  estimated with Chao-Wang-Jost.
- **Phases:** *encoding*, 0-1.6 s after each word (recalled vs forgotten); *retrieval*, the
  1.6 s before correct first recalls with no vocalization in the preceding 1.8 s, against
  non-overlapping deliberation windows of the recall periods with no vocalization from 1 s
  before to 1.6 s after (`common.retrieval_windows`).
- **Questions:** *existence*, real data against the source's trials shuffled (within-trial
  coupling broken, stimulus-locked time courses kept); *contrast*, condition A against B.
  Both paired across subjects (`compare_networks(design="between", dependent=True)`,
  10 000 sign flips), Benjamini-Hochberg over the eight direction x lag tests per layer.
  Primary layer: intrinsic.

## 2. Results

Means over subjects, bits; q per layer over the 8 tests.

**Encoding (88 subjects).**

| Question | Intrinsic | Synergistic | Shared |
|---|---|---|---|
| Existence (real vs shuffled) | 0.0001-0.0002 vs 0.0001; smallest q 0.23 | 0.0012-0.0021 vs 0.0014-0.0023; smallest q 0.16 | 0.0005-0.0017 vs 0.0001; q <= 0.002 in all 8 |
| Recalled vs forgotten | smallest q 0.93 | smallest q 0.48 | smallest q 0.92 |

**Retrieval (56 subjects), non-overlapping windows.**

| Question | Intrinsic | Synergistic | Shared |
|---|---|---|---|
| Existence | smallest q 0.81 | smallest q 0.77 | 0.0008-0.0016 vs 0.0001-0.0003; q <= 0.006 in all 8 |
| Pre-recall vs deliberation | smallest q 0.52 | smallest q 0.55 | PFC -> MTL at 200 ms: 0.0019 vs 0.0008, q = 0.019; others q >= 0.54 |

The single surviving retrieval contrast disappears once each condition's mean time course
is removed from each region (section 3), so it reflects the shared pre-recall rise (PFC HFA
is higher before recalls in 46 of 56 subjects) rather than coupling.

**Amplitude, for comparison.** Encoding: PFC HFA higher for recalled words in 67 of 88
subjects, MTL in 54. Retrieval: PFC higher before recalls in 46 of 56.

## 3. Checks and artifacts

- **Overlapping deliberation windows.** The first retrieval run tiled deliberation windows
  every 0.8 s (1.6 s long), so neighbours shared half their samples. Duplicated samples
  turn singletons into doubletons, which bias-corrected estimators such as Chao-Wang-Jost
  read as a better-sampled distribution, inflating transfer entropy and synergy. That run
  reported synergy existing at q < 0.001 in 6 of 8 tests and deliberation synergy twice
  pre-recall synergy (0.022-0.026 against 0.010-0.015 bits, q < 0.001 in 7 of 8 and 0.021
  in the eighth). With
  non-overlapping windows (the same epochs, subselected) both vanish: deliberation synergy
  drops to 0.009-0.012 bits and every synergistic q is >= 0.55. `retrieval_windows` now
  produces non-overlapping windows and stores each epoch's start.
- **Condition-locked trajectories.** Removing each condition's mean time course per region
  before discretization (`flow.py analyze --demean`) leaves no contrast in any layer in
  either phase, and keeps shared-flow existence (encoding 8 of 8, retrieval 6 of 8).
- **Exploratory intrinsic flow.** In the de-meaned encoding data, intrinsic flow exceeded the
  surrogate for MTL -> PFC at 20 ms (0.00018 vs 0.00005, q = 0.012) and 40 ms (q = 0.031),
  and PFC -> MTL at 200 ms (0.00021 vs 0.00008, q = 0.012), with real above shuffled in only
  25-32 of 88 subjects (most others tie at zero, where the estimator clips). Not
  pre-specified, inconsistent across lags, and about a ten-thousandth of a bit.

## 4. Open issues closed

- **`compare_networks(estimator=...)`.** The layers' estimator is now an option (default
  Miller-Madow, unchanged). Rerunning the 25-subject region-level memory contrast with
  Chao-Wang-Jost lowers the baseline synergy per condition about fourfold (0.0013-0.0025
  bits, against 0.0056-0.0075 with Miller-Madow) and still finds nothing: every
  between-subject q >= 0.31, and no within-subject edge (Miller-Madow had one).
- **Synergistic null excess.** `validation/synergy_null.py` split R1332M's 170 recalled
  trials 1000 times, with the network selected once on all of them. Miller-Madow synergy
  had 7.0% of p-values below 0.05 (intrinsic 4.6%, shared 4.7%). An independent exact test
  on 5000 fresh random splits reproduced the p-values (correlation 0.998, 6.5% below 0.05),
  the statistic is deterministic and order-independent, and the observed splits' statistics
  follow the permutation distribution (KS p = 0.58): the test is exact and the excess is
  chance (binomial p of about 0.02 across six rates examined). With Chao-Wang-Jost,
  synergy is nominal (4.6%) but 56% of its p-values are exactly 1: at about 85 trials per
  group its synergy is usually clipped to zero, so it has little power for synergy at this
  sample size.
- **dit's `prng` alias.** dit #324 adds the missing `@renamed_kwarg("prng", "rng")` to both
  implementations behind `total_correlation_ksg`, with a regression test.

## 5. Where this leaves directed flow

With region-averaged HFA, 50 Hz, four bins and 43-71 epochs per condition, any memory-related
change in directed MTL-PFC information flow is below what this design can see; the shared
co-fluctuation is robust but undirected. Directions that might still change that:

- finer spatial resolution (hippocampal and prefrontal pairs with the strongest amplitude
  effects, chosen on a held-out session), since region averaging mixes signals;
- broadband phase-based measures at theta/gamma, where MTL-PFC coupling is usually reported,
  instead of HFA envelopes;
- pooling sessions within subjects to raise epochs per condition.

## Reproducing

```bash
cd examples/catfr
export CATFR_OUT=~/data/catfr_cohort
uv run --project ../.. --with mne --with netcdf4 --with "decorator<5.3" python flow.py prepare
uv run --project ../.. --with mne --with netcdf4 --with "decorator<5.3" python flow.py analyze
uv run --project ../.. --with mne --with netcdf4 --with "decorator<5.3" python flow.py analyze --demean
uv run --project ../.. --with mne --with netcdf4 --with "decorator<5.3" python cohort.py --estimator chao_wang_jost --tag cwj
uv run --project ../.. --with netcdf4 python ../../validation/synergy_null.py \
    --epochs ~/data/catfr_pilot_v3/sub-R1332M_ses-0_hfa.nc --splits 1000 --estimator miller_madow --out mm.json
```
