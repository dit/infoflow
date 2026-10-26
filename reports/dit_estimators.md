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
