Quickstart
==========

Install from the repository (``infoflow`` builds on the inference tools in
:mod:`dit.inference`)::

    pip install "infoflow @ git+https://github.com/dit/infoflow"

Infer a network from a (time, process) array, or a list of such arrays (trials):

.. code-block:: python

    import infoflow
    from infoflow import datasets

    data = datasets.common_driver(4000, seed=0)  # Z drives both X and Y
    net = infoflow.infer_multiplex(data, names=["Z", "X", "Y"], prng=0)
    print(net.summary())

    net.layer("intrinsic")   # xarray.DataArray (source, target) of significant weights
    net.dataset              # every estimate, CI, p- and q-value, role, and flag
    graph = net.to_networkx()

Continuous input goes through :func:`infoflow.preprocess.preprocess` first
(``preprocess="auto"``); already-discrete input is used as is. Pass
``preprocess=False`` with a :class:`~infoflow.data.DiscreteData` to skip it, or a
dict of :mod:`~infoflow.preprocess.discretizers` to fix the symbolization.

Transfer entropy on its own (moved here from ``dit``):

.. code-block:: python

    from infoflow import transfer_entropy, transfer_entropy_test

    te = transfer_entropy(source, target, k=2, l=1)
    test = transfer_entropy_test(source, target, k=2, l=1, n_perm=500)
