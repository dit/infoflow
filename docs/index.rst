infoflow
========

``infoflow`` infers *multiplex information-flow networks* from multivariate time
series. Each directed edge's transfer entropy :cite:`Schreiber2000` is split into
three layers :cite:`James2016`:

* **intrinsic** flow, :math:`I[Y_t : X_{\text{past}} \downarrow W]`, the part of the
  source's influence that no channel applied to the context can explain away;
* **synergistic** flow, the transfer entropy minus the intrinsic flow, information
  about the target that the source carries only jointly with the context;
* **shared** flow, the time-delayed mutual information minus the intrinsic flow,
  information the source and the context carry redundantly.

The identities :math:`TE = \text{intrinsic} + \text{synergistic}` and
:math:`TDMI = \text{intrinsic} + \text{shared}` hold exactly because all three
quantities are computed from one joint distribution. Nodes carry a fourth layer,
active information storage :cite:`Lizier2012storage`.

The pipeline is non-parametric and discrete. Continuous data are rank-discretized
by an automatic preprocessing step; parents are chosen by an IDTxl-style greedy
multivariate transfer-entropy search :cite:`Lizier2012,Novelli2019,Wollstadt2019`;
each layer gets bootstrap confidence intervals, a layer-specific permutation null,
and false-discovery-rate control.

.. toctree::
   :maxdepth: 2

   quickstart
   layers
   preprocessing
   skeleton
   interpretation
   workflow
   gpu
   benchmarks
   api
   zreferences
