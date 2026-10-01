Interpretation
==============

Each edge is labeled with a **role** by comparing its pairwise and conditioned
flows (:func:`infoflow.interpret.edge_roles`): *direct*, *confounded* (explained by
a common parent), *mediated* (explained by an intermediate), or *explained*.

Edges that carry only shared flow are explained separately
(:func:`infoflow.interpret.explain_shared`): the time-delayed mutual information is
conditioned on each observed variable (the target's other parents and the
source's parents), on all of them, and on the target's own past. An observed
explainer gives *confounded*, *mediated*, or *explained*; the target's own past,
when the target drives the source, gives *reverse*. Every comparison is
bias-corrected with a within-strata permutation null, because conditioning on
more variables inflates plug-in estimates by about as much as a weak shared flow.

**Latent flags** (:func:`infoflow.interpret.latent_flags`) mark edges that point to
something unobserved or unconfirmed: *shared-dominant* and *unexplained-tdmi*
(shared flow that no observed variable or reverse coupling explains),
*layers-unconfirmed* (a selected parent whose intrinsic and synergistic layers are
both non-significant, e.g. a weak edge found by KSG selection that the symbol-based
layer estimates cannot resolve), *bidirectional*, *zero-lag* (the source's present
informs the target's present beyond the lagged context and both processes' observed
parents, by more than the transfer entropy), and *history-collider* (conditioning on
more target history raises the bias-corrected transfer entropy while the intrinsic
flow stays near zero).

``net.summary()`` shows non-significant layer estimates as ``–``; pass
``show_nonsignificant=True`` to print them all.

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
