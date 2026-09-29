Workflow tools
==============

Comparing conditions
--------------------

:func:`infoflow.compare.compare_networks` tests layer differences between two
conditions on the union of their edges, for trials within a subject
(``design="within"``) or for subjects (``design="between"``), with paired
(``dependent=True``) or independent units. Condition labels are permuted (swapped
within pairs when dependent) and p-values are corrected per layer.

Parallelism
-----------

``infer_multiplex`` takes a ``map_fn`` used over targets and edges; seeds are
drawn before mapping, so results do not depend on the map:

.. code-block:: python

    from infoflow.parallel import dask_map, thread_map

    net = infoflow.infer_multiplex(data, map_fn=thread_map(8), prng=0)
    net = infoflow.infer_multiplex(data, map_fn=dask_map(), prng=0)

Checkpointing and provenance
----------------------------

``checkpoint="run/"`` stores each target's skeleton and each edge's estimates as
they finish; rerunning with the same directory and seed resumes. ``net.settings``
records every option and the package versions; ``net.save(path)`` and
:meth:`MultiplexNetwork.load <infoflow.network.MultiplexNetwork.load>` persist
results.

Input and output
----------------

:func:`infoflow.io.load_mat` and :func:`infoflow.io.load_fieldtrip` read MATLAB
data; :func:`infoflow.io.export_brainnet` writes BrainNet Viewer ``.node``/``.edge``
files. :func:`infoflow.plot.plot_layer` and :func:`infoflow.plot.plot_multiplex`
draw layers with HoloViews (``pip install infoflow[plot]``).
