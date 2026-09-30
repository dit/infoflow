Estimator scaling
=================

Whether a weak edge can be found depends on how the conditional mutual information
estimate behaves as the conditioning set grows. :mod:`infoflow.benchmarks` measures
this with the design of :cite:`Young2021`: :math:`Y` depends on :math:`X` with a
known mutual information and the conditioning set holds :math:`d` independent
nuisance variables, so the true :math:`I[X : Y \mid Z]` does not change with
:math:`d`.

.. code-block:: python

    from infoflow.benchmarks import conditioning_scaling

    ds = conditioning_scaling("gaussian", n_samples=(500, 2500), dims=(0, 1, 2, 4, 8, 12), prng=0)
    ds["bias"].sel(n=2500)

Mean estimates in bits, 2500 samples, 5 datasets each (Gaussian variables are cut
into 4 equal-frequency bins for the symbol-based estimators):

==============================  ======  ======  ======  ======  ======  ======
Estimator, data                 d = 0   1       2       4       8       12
==============================  ======  ======  ======  ======  ======  ======
*Gaussian, true value 0.322*
plug-in                         0.228   0.252   0.282   0.796   0.022   0.000
Miller--Madow                   0.226   0.242   0.243   0.754   0.029   0.000
KSG                             0.311   0.340   0.326   0.312   0.228   0.173
*Gaussian, true value 0*
plug-in                         0.003   0.009   0.046   0.658   0.021   0.000
KSG                             -0.001  0.004   0.000   0.001   -0.003  -0.001
*Binary, true value 0.456*
plug-in                         0.459   0.456   0.461   0.464   0.525   0.173
Miller--Madow                   0.459   0.456   0.460   0.459   0.543   0.228
==============================  ======  ======  ======  ======  ======  ======

Symbol-based estimates first inflate as the joint cells thin out (the permutation
tests absorb this), then collapse towards zero once nearly every context value is
unique, and with them the power to detect anything. That is why a finely resolved
parent cannot be conditioned away with bins, and why a weak edge into the same
target can be missed. The KSG estimate on raw values never manufactures dependence
and decays only gradually and downward, as :cite:`Young2021` report for their
classifier-based estimator; it is available for parent selection as
``SkeletonSettings(estimator="ksg")``.
