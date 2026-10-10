Quickstart
==========

Install from the repository (``infoflow`` builds on the inference tools in
:mod:`dit.inference`)::

    pip install "infoflow @ git+https://github.com/dit/infoflow"

Infer a network from a (time, process) array, or a list of such arrays (trials),
with every method chosen automatically (:doc:`auto`):

.. code-block:: python

    import infoflow
    from infoflow import datasets

    data = datasets.common_driver(4000, seed=0)  # columns (x, y, z); z drives x and y
    result = infoflow.infer(data, names=["X", "Y", "Z"], prng=0)
    print(result.report)  # the estimators, options, and timings it chose, and why
    net = result.network

To set every option yourself, call :func:`infoflow.infer_multiplex` directly:

.. code-block:: python

    net = infoflow.infer_multiplex(data, names=["X", "Y", "Z"], prng=0)
    print(net.summary())
    # Z -> X and Z -> Y are intrinsic; X -> Y appears only as shared flow.

    net.layer("intrinsic")   # networkx.DiGraph of the significant intrinsic edges
    net.dataset              # every estimate, CI, p- and q-value, role, and flag
    graph = net.to_networkx()

Continuous input goes through :func:`infoflow.preprocess.preprocess` first
(``preprocess="auto"``); already-discrete input is used as is. Pass
``preprocess=False`` with a :class:`~infoflow.data.DiscreteData` to skip it, or a
dict of :mod:`~infoflow.preprocess.discretizers` to fix the symbolization.

Transfer entropy on its own (moved here from ``dit``):

.. code-block:: python

    from infoflow import transfer_entropy, transfer_entropy_test

    te = transfer_entropy(source, target, history_length=2, source_history=1)
    test = transfer_entropy_test(source, target, history_length=2, null="whittle", n_surrogates=499)
