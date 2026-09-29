Interpretation
==============

Each edge is labeled with a **role** by comparing its pairwise and conditioned
flows (:func:`infoflow.interpret.edge_roles`): *direct*, *confounded* (explained by
a common parent), *mediated* (explained by an intermediate), or *explained*.

**Latent flags** (:func:`infoflow.interpret.latent_flags`) mark edges that point to
something unobserved: *shared-dominant* (shared flow exceeds intrinsic flow and no
observed process explains it), *unexplained-tdmi*, *bidirectional*, *zero-lag*,
and *history-collider* (conditioning on a common child's past creates
dependence).

Node attributes (:func:`infoflow.nodes.node_layer`) are active information storage
with a permutation test :cite:`Lizier2012storage` and weighted permutation entropy
:cite:`Fadlallah2013`, which flags nearly unpredictable nodes.

Optional layers:

* a lag-0 layer with collider orientation in the style of PCMCI+
  :cite:`Runge2020` (:mod:`infoflow.contemporaneous`), with an adapter to Tigramite
  :cite:`Runge2018`;
* two-source partial-information-decomposition hyperedges
  :cite:`Williams2010,Bertschinger2014` (:mod:`infoflow.hyperedges`).

A five-process benchmark from the MuTE toolbox :cite:`Montalto2014` is available
as :func:`infoflow.datasets.mute_network`.
