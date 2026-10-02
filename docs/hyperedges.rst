Hyperedges and PID measures
===========================

``infer_multiplex(hyperedges=True)`` (or :func:`infoflow.hyperedges.multivariate_hyperedges`)
decomposes, for every target, each set of up to ``max_sources`` parents (default 3)
and each pair the synergy-aware parent search selected jointly, with a partial
information decomposition :cite:`Williams2010` of :math:`I[Y_t : (S_1, \ldots, S_k)]`.

* **Detection.** The synergy (top) atom is tested against a null that permutes one
  source within strata of the target: each source keeps its relation to the target,
  so redundancy and unique information survive, while the joint dependence that
  synergy needs is destroyed. The number of permutations grows with the number of
  source sets so that the per-target Benjamini--Hochberg correction can reject.
* **Classification.** Each set is labelled ``synergistic``, ``redundant``,
  ``unique-dominated``, or ``mixed`` by the share of :math:`I[Y_t : S]` in its top,
  bottom, and unique atoms, with the sign of the O-information :cite:`Rosas2019` as a
  measure-free cross-check (negative: synergy-dominated).
* **Ensembles.** ``hyperedge_measure=["ccs", "mmi", ...]`` decomposes with every
  measure and reports the majority label and the share of measures agreeing.

The result is an :class:`xarray.Dataset` over ``hyperedge`` (and ``measure``);
:func:`infoflow.hyperedges.hypergraph` turns it into a bipartite networkx graph.

PID measures
------------

Any of dit's PID measures can be used by name, or any dit PID class. The registry
records what was measured on dit's textbook distributions and on sample
distributions under the synergy null (2 and 3 sources, 4000 samples): the largest
number of sources that gave finite atoms without errors, whether atoms stayed
non-negative, and the time per decomposition (fast < 0.2 s, medium < 1.5 s). The
default is ``ccs`` :cite:`Ince2017`; ``broja`` :cite:`Bertschinger2014`, the usual
two-source choice, returns non-finite atoms for three sources on sample data and is
limited to pairs.

===========  ===================  =======  ============  =======  =============================
Name         dit class            Sources  Non-negative  Cost     Note                         
===========  ===================  =======  ============  =======  =============================
broja        ``PID_BROJA``        2        yes           fast
ccs          ``PID_CCS``          3        yes           fast
ct           ``PID_CT``           3        yes           fast
deg          ``PID_Deg``          2        yes           medium
delta        ``PID_Delta``        2        no            fast
deltalambda  ``PID_DeltaLambda``  2        yes           fast
dep          ``PID_dep``          3        yes           slow
do           ``PID_Do``           3        yes           fast
gh           ``PID_GH``           2        yes           slow
gk           ``PID_GK``           3        yes           fast
ig           ``PID_IG``           2        yes           fast
ipid         ``PID_IPID``         2        yes           fast
mc           ``PID_MC``           2        yes           medium
mes          ``PID_MES``          3        yes           fast
mmi          ``PID_MMI``          3        yes           fast
pm           ``PID_PM``           3        no            fast
prec         ``PID_Prec``         ?        ?             unknown  needs pypoman
proj         ``PID_Proj``         ?        yes           fast     fails on sample distributions
ra           ``PID_RA``           3        no            slow
rav          ``PID_RAV``          2        yes           slow
rdr          ``PID_RDR``          3        yes           fast
rr           ``PID_RR``           3        yes           fast
skar_nw      ``PID_SKAR_nw``      2        yes           fast
skar_owa     ``PID_SKAR_owa``     2        yes           medium
skar_owb     ``PID_SKAR_owb``     3        yes           slow
skar_tw      ``PID_SKAR_tw``      2        yes           slow
sx           ``PID_SX``           3        no            medium
wb           ``PID_WB``           3        yes           fast
===========  ===================  =======  ============  =======  =============================
