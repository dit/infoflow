Parent selection
================

:func:`infoflow.selection.infer_skeleton` runs, for each target, the greedy
multivariate transfer-entropy search of IDTxl
:cite:`Lizier2012,Novelli2019,Wollstadt2019` over a non-uniform embedding
(:class:`~infoflow.embedding.Embedding`):

1. add the target's own past lags while they are significant (maximum statistic);
2. add source lags conditioned on everything selected so far;
3. revisit the target's past conditioned on the selected sources;
4. search pairs of candidates that are only jointly informative, as in an XOR
   (``synergy_search``);
5. prune with the minimum statistic, run an omnibus test of all sources jointly,
   and a sequential maximum-statistic test per source;
6. control the false discovery rate across targets on the omnibus p-values.

Candidates with significant time-delayed mutual information but no significant
transfer entropy are kept for the shared layer (``tdmi_screen``). Lag-0 source
values can be added as conditionals to compensate instantaneous effects
:cite:`Faes2011,Faes2013`. Surrogates respect the data's structure: trial
shuffles when there are trials, otherwise circular, block, or local permutations
(:class:`~infoflow.selection.SkeletonSettings`).

Selection and estimation can use disjoint halves of the data (``holdout``) to
avoid post-selection bias in the layer estimates.

For continuous data, ``SkeletonSettings(estimator="ksg")`` selects parents with the
KSG conditional mutual information on the raw values :cite:`Kraskov2004,Frenzel2007`,
as IDTxl does. Conditioning on a parent continuously removes its influence where bins
leave a residue, so weak edges into strongly driven targets can be found (see
:doc:`benchmarks`); the layers are still estimated on symbols. Permutations run on
all cores (``ksg_threads``) and reuse the trees of each conditioning set. By default
(``ksg_null="local"``) the nulls are Runge's local permutations :cite:`Runge2018`, which keep
a candidate's dependence on the conditioning set and so keep children of a target
out of its parent set.
