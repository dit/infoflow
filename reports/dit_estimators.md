# dit's new entropy estimators in infoflow

*Branch `dit-estimators`, October 2026. dit `e3611818` (bias-corrected estimators, dit
PR #295). Scripts: `validation/dit_estimators.py` (layers, exact flows),
`validation/dit_estimators_selection.py` (selection, known graphs),
`validation/dit_estimators_catfr.py` (catFR1).*

## Summary

- **Layers: yes.** Chao-Wang-Jost (CWJ) cut the cross-fitted layer RMSE by 25-30% against
  the current Miller-Madow default at 300-1000 samples, mostly by removing synergistic
  bias in large contexts; on catFR it cut spurious synergy 2.5x. NSB and jackknife are
  as good or better but too slow for routine use.
- **Selection: modestly.** Miller-Madow in the selection statistic (already in dit, now
  selectable) raised recall by 0.05 at no cost on both synthetic network types. CWJ
  raised recall most (0.06 VAR, 0.13 logistic maps) at 2-9x the cost, with a few more
  false positives on logistic maps and a slightly worse edge-score ranking. On catFR the
  selection estimator made almost no difference.
- **Nothing changes by default.** Both are opt-in: `infer_multiplex(estimator="chao_wang_jost")`
  and `SkeletonSettings(entropy_estimator="miller_madow")`.

## What was added

- `measures.cmi_from_joint` and everything above it (`edge_flows`, `crossfit_flows`,
  layer nulls, `infer_multiplex(estimator=...)`) accept any of
  `dit.inference.estimators.ENTROPY_ESTIMATORS`, applied to each entropy term of the
  count table with the product of its axes' sizes as the alphabet. The intrinsic flow
  is evaluated at the channel hardened to its most likely output per context
  (`measures.evaluation_table`), since these estimators need integer counts;
  plug-in and Miller-Madow keep the soft channel.
- `SkeletonSettings(entropy_estimator=...)` swaps the entropies inside the plug-in
  selection statistic. The permutation tests stay valid for any statistic.
- `uv.lock` moves dit to `e3611818`. Tests: each estimator's layer CMI equals dit's
  `conditional_mutual_information`, layer identities hold, selection recovers a chain.

## 1. Layer accuracy (exact flows)

Sofic copy, XOR, shared and mixed edges (history 1) with 0-3 independent ternary
nuisance variables added to the context (2-54 contexts; true flows unchanged), 20
replicates. RMSE in bits, averaged over processes, context sizes and layers:

| Estimator | Cross-fit n=300 | 1000 | 3000 | Single joint n=300 | 1000 | 3000 | Cost per entropy |
|---|---|---|---|---|---|---|---|
| plugin | 0.044 | 0.024 | 0.010 | 0.043 | 0.018 | 0.008 | 1x |
| miller_madow (default) | 0.047 | 0.019 | 0.007 | 0.052 | 0.019 | 0.008 | 1x |
| grassberger | 0.044 | 0.015 | 0.007 | 0.040 | 0.015 | 0.007 | 2x |
| zhang | 0.046 | 0.018 | 0.007 | 0.040 | 0.014 | 0.007 | 18x |
| **chao_wang_jost** | **0.035** | **0.014** | **0.007** | 0.035 | 0.014 | 0.007 | 13x |
| nsb | 0.030 | 0.013 | 0.007 | 0.033 | 0.014 | 0.007 | ~2500x |
| jackknife | 0.044 | 0.014 | 0.007 | 0.038 | 0.014 | 0.007 | 180x |
| bayes_minimax | 0.031 | 0.018 | 0.009 | 0.030 | 0.014 | 0.008 | 1x |
| shrink, bayes_laplace, bonachela, ansb | 0.05-0.09 | | | | | | |

The gain is in the synergistic layer: at 54 contexts, CWJ's synergy bias was 0.033 bits
at 300 samples against Miller-Madow's 0.081 (cross-fit), and its synergy RMSE over all
settings 0.024 against 0.038. Intrinsic and shared flow are about equally accurate
under the better estimators.

## 2. Selection accuracy (known graphs)

`network_validation`, 10-node random networks simulated as a VAR or coupled logistic
maps, 500 and 2000 samples, five networks each, identical networks for every estimator;
the layer estimator stays Miller-Madow. Paired differences against the plug-in over the
10 networks per kind (better / worse count in parentheses):

| Estimator | VAR recall | VAR precision | logistic recall | logistic precision | logistic edge-score AUROC | Runtime vs plug-in |
|---|---|---|---|---|---|---|
| miller_madow | +0.053 (7/0) | +0.168 (6/0) | +0.053 (7/0) | +0.037 (5/2) | -0.001 | 1.0-1.1x |
| zhang | +0.053 (7/0) | +0.168 (6/0) | +0.060 (7/0) | +0.038 (5/2) | -0.002 | 2.2-7.5x |
| chao_wang_jost | +0.064 (9/0) | +0.150 (7/1) | +0.125 (9/1) | +0.002 (4/3) | -0.037 (2/8) | 1.4-9.4x |
| grassberger | +0.007 (2/1) | +0.055 (6/3) | +0.082 (7/1) | +0.031 (3/5) | -0.041 (1/8) | 1.0-1.8x |
| bayes_minimax | +0.025 (5/0) | +0.089 (4/1) | +0.013 (3/0) | -0.017 (1/1) | -0.002 | 1.0-1.5x |

At 2000 samples on logistic maps, recall went from 0.49 (plug-in) to 0.56 (Miller-Madow)
and 0.69 (CWJ) at precision 0.94-0.95; CWJ also found 0.4 more false edges per network.
NSB and jackknife were not run: at 100 ms and 7 ms per CMI they are too slow for
thousands of permutations per candidate.

## 3. catFR1 (no ground truth)

**Bias on real data.** For the 25 cohort subjects and 6 region pairs (region-averaged
HFA, 4 bins, the cohort's fixed variables, all trials), the layers with the source
shuffled across samples (every true layer zero, so the estimate is bias) against the real
data, mean bits:

| Estimator | Shuffled: TE / intrinsic / synergistic / shared | Real: TE / intrinsic / synergistic / shared |
|---|---|---|
| plugin | 0.0182 / 0.0003 / 0.0179 / 0.0000 | 0.0191 / 0.0013 / 0.0178 / 0.0015 |
| miller_madow | 0.0011 / 0.0000 / 0.0010 / 0.0000 | 0.0019 / 0.0001 / 0.0018 / 0.0023 |
| chao_wang_jost | 0.0005 / 0.0000 / 0.0004 / 0.0000 | 0.0011 / 0.0004 / 0.0008 / 0.0021 |
| nsb | 0.0002 / 0.0000 / 0.0002 / 0.0001 | 0.0007 / 0.0003 / 0.0004 / 0.0021 |
| jackknife | 0.0002 / 0.0000 / 0.0002 / 0.0000 | 0.0008 / 0.0003 / 0.0005 / 0.0022 |
| bayes_minimax | 0.0176 / 0.0003 / 0.0173 / 0.0000 | 0.0184 / 0.0019 / 0.0165 / 0.0008 |

Half of the synergy Miller-Madow reports here is bias; with CWJ, NSB or jackknife it
shrinks to a few ten-thousandths of a bit, while shared flow (about 0.002 bits) is the
same under every corrected estimator. Bayes-minimax, good on the synthetic processes,
is nearly as biased as the plug-in here (large sparse contexts).

**Selection.** Edges selected on the pilot's HFA epochs (18 channels, all 300 trials):
R1065J 6 / 6 / 6, R1332M 2 / 2 / 2, R1364C 6 / 8 / 6 (plug-in / Miller-Madow / CWJ);
none with any estimator on surrogates whose channels' trials are permuted independently.
CWJ took 1.3-2.6x as long.

## 4. Both changes together on other benchmarks

`validation/new_defaults.py`: *current* (plug-in selection, Miller-Madow layers) against
*proposed* (Miller-Madow selection, CWJ layers), same seeds and data. Paired over
networks or instances (better / worse counts in parentheses).

| Benchmark | Metric | Current | Proposed |
|---|---|---|---|
| VAR, 10 nodes, 500 / 2000 samples | edge recall | 0.12 / 0.33 | 0.17 / 0.39 (4/0, 3/0) |
| | edge precision | 0.59 / 0.84 | 0.89 / 0.87 |
| | intrinsic-layer recall (2000) | 0.29 | 0.39 (4/0), precision 0.98 -> 0.97 |
| Logistic maps, 500 / 2000 | edge recall | 0.20 / 0.49 | 0.24 / 0.56 (3/0, 4/0) |
| | intrinsic-layer recall (2000) | 0.35 | 0.54 (5/0), precision 1.00 -> 1.00 |
| Lorenz-96, 10 nodes, 1000 | edge recall / precision | 0.28 / 0.91 | 0.33 / 0.98 (4/0) |
| | intrinsic-layer recall | 0.04 | 0.33 (5/0), precision 0.98 |
| Motifs, 20 replications | six layer and role checks | 1.00, 1.00, 1.00, 1.00, 1.00, 0.95 | 1.00, 1.00, 1.00, 1.00, 0.95, 0.90 (chain checks) |
| NetSim, sims 1-4 x 5 subjects, `auto` | AUROC | 0.625 | 0.627 (9/11) |
| CausalRivers, 20 three-station graphs, `auto` | AUROC / recall | 0.581 / 0.50 | 0.525 / 0.45 (2/6) |

Synthetic networks run 50% longer. On CausalRivers the whole loss comes from the
selection change: Miller-Madow selection alone gives the same 0.525, CWJ layers alone
give exactly the current results. NetSim is unchanged because `auto` finds almost no
edges there either way (recall 0.001) and its scores barely move.

## 5. dit's second batch (dit PR #305)

Six more discrete estimators (Panzeri-Treves, quadratic extrapolation, Wu-Yang, BUB,
unseen, PYM) and a Gaussian-copula mutual information, against the current defaults
(CWJ layers, plug-in selection). None is a clear improvement.

**Cost per CMI** (2000 samples, 1024 cells): Panzeri-Treves 1.1 ms, Wu-Yang 2.3,
quadratic extrapolation 3.5, BUB 7.6, unseen 20, PYM 42 (CWJ 0.5, plug-in 0.03).

**Layers** (section 1's exact-flow screen, cross-fitted RMSE in bits at 300 / 1000 / 3000
samples; "better cells": of the 48 process x size x context cells, how many beat CWJ):

| Estimator | RMSE | Better cells | catFR synergy bias (section 3) |
|---|---|---|---|
| chao_wang_jost (default) | 0.035 / 0.014 / 0.007 | | 0.0004 |
| panzeri_treves | 0.028 / 0.013 / 0.007 | 19 / 48 | 0.0006 |
| nsb | 0.030 / 0.013 / 0.007 | 22 / 48 | 0.0002 |
| quadratic_extrapolation | 0.048 / 0.015 / 0.007 | 18 / 48 | 0.0002 |
| wu_yang | 0.039 / 0.015 / 0.007 | | 0.0012 |
| pym | 0.042 / 0.016 / 0.008 | 12 / 48 | 0.0000, but real TE 0.0002 |
| bub | 0.052 / 0.018 / 0.007 | | 0.0010 |
| unseen | 0.079 / 0.046 / 0.036 | | 0.0000, but real TE 0.0000 |

Panzeri-Treves has the lowest mean RMSE, from the smallest, most context-heavy settings
(its synergy bias at 54 contexts and 300 samples is 0.019 bits against CWJ's 0.057), but
it is worse than CWJ in most cells and on the catFR bias probe. Quadratic extrapolation
matches NSB on catFR but is worse than CWJ on the synthetic processes. Unseen and PYM
remove the bias by suppressing the flow itself: on the catFR data their transfer entropy
is 0.0000 and 0.0002 bits against CWJ's 0.0011, and they add bias to shared flow.

**Selection** (section 2's networks, paired against the plug-in over 10 networks):

| Estimator | VAR recall | VAR precision | VAR AUROC | Logistic recall | Logistic AUROC | Runtime vs plug-in |
|---|---|---|---|---|---|---|
| panzeri_treves | 0.23 -> 0.32 (10/0) | 0.71 -> 0.86 | 0.79 -> 0.77 (3/7) | 0.34 -> 0.47 (9/0) | 0.87 -> 0.83 (2/8) | 12x, 120x |
| wu_yang | 0.23 -> 0.29 (7/0) | 0.71 -> 0.76 | 0.79 -> 0.73 (2/8) | 0.34 -> 0.43 (7/1) | 0.87 -> 0.81 (2/8) | 1.1x, 2.2x |

Both find more edges with a worse edge-score ranking, the trade-off CWJ selection showed;
Panzeri-Treves is the strongest selection statistic tested but far too slow, and neither
was run on CausalRivers, where a recall-raising selection statistic (Miller-Madow) cost
0.06 AUROC.

**Gaussian copula.** Copula-normalizing the data before `auto.infer` turns its
linear-Gaussian selection estimator into the Gaussian-copula one :cite:`Ince2017`:

| Benchmark | AUROC raw -> copula |
|---|---|
| VAR, 500 / 2000 | 0.946 -> 0.948 (3/2), 0.995 -> 0.995 (2/2) |
| Logistic maps, 500 / 2000 | 0.712 -> 0.648 (0/5), 0.970 -> 0.972 (3/2) |
| Lorenz-96, 1000 | 0.936 -> 0.936 (no change) |
| NetSim, 20 subjects | 0.625 -> 0.618 (6/13) |
| CausalRivers, 20 graphs | 0.581 -> 0.606 (8/4) |

It helps CausalRivers and hurts NetSim and short logistic series, so it is not a default
either; `copula_normalize` before `auto.infer` stays a user option for river-like data.

`dit` also renamed `prng` to `rng` (PR #306); `total_correlation_ksg` no longer accepts
the old name despite the deprecation alias, so infoflow now passes `rng`.

## Recommendation

CWJ layers are a safe default: they improved every synthetic benchmark (most of all the
intrinsic layer's recall) and changed nothing on NetSim or CausalRivers. Miller-Madow
selection helps the synthetic networks but costs CausalRivers 0.06 AUROC, the one
real-data benchmark with known structure, so it stays an option.

**Adopted:** the layer estimator now defaults to CWJ
(`measures.DEFAULT_LAYER_ESTIMATOR`, used by `infer_multiplex`, `estimate_edge`,
`layer_statistics`, `crossfit_flows`, the channel layer and `auto`'s cost pilot); the
selection statistic stays plug-in, and `compare_networks` still evaluates with
Miller-Madow. On 40 pairs of independent AR(1) processes (23 edges tested), the false
layers were 3 intrinsic, 5 synergistic and 5 shared with CWJ against 1, 5 and 13 with
Miller-Madow, and 4 runs in each had a false intrinsic or synergistic layer; the
uncoupled test now counts runs with any false flow (at most one of eight), since on one
seed CWJ marks both layers of the edge that Miller-Madow marks once.
