infoflow
========

**Multiplex information-flow network inference from time series**, built on
`dit <https://github.com/dit/dit>`_.

``infoflow`` infers a directed network from multivariate time series and splits
each edge's transfer entropy into three layers (James, Barnett & Crutchfield 2016):

* **intrinsic** flow, ``I[Y_t : X_past ↓ W]``, which remains after any processing of
  the target's context ``W``;
* **synergistic** flow, the part of transfer entropy that needs the context
  (``TE − intrinsic``);
* **shared** flow, the part of time-delayed mutual information carried by common
  history or drivers (``TDMI − intrinsic``).

Each node also carries its active information storage. The pipeline is mostly
hands-off and nonparametric:

1. **Preprocess.** Rank-based discretization (equal-frequency bins or ordinal
   patterns) and embedding are chosen per node by cross-validated predictive
   information, and each node gets a history budget.
2. **Skeleton.** Parents are selected by greedy multivariate transfer entropy
   with non-uniform embedding and hierarchical maximum, minimum, omnibus, and
   sequential statistics with FDR, plus a synergy-aware pair search and a TDMI
   screen.
3. **Layers.** Each edge's flows are estimated from one joint distribution, with
   context merging, cross-fitting, bootstrap intervals, and per-layer tests.
4. **Interpretation.** Edges are labelled confounded, mediated, or direct, and
   possible latent confounding is flagged.

.. code-block:: python

   import infoflow
   from infoflow.datasets import common_driver

   data = common_driver(n=3000, seed=0)       # (samples, processes) array
   net = infoflow.infer_multiplex(data, prng=0)
   net.dataset                                # xarray Dataset (layer, source, target)
   net.to_networkx()                          # networkx MultiDiGraph with a layer attribute

Installation
------------

``infoflow`` currently needs the development version of dit
(``dit.inference`` Markov-order, ordinal, and surrogate tools)::

   pip install "infoflow @ git+https://github.com/dit/infoflow"
