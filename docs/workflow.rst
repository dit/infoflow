Workflow tools
==============

Comparing conditions
--------------------

:func:`infoflow.compare.compare_networks` tests layer differences between two
conditions on the union of their edges, for trials within a subject
(``design="within"``) or for subjects (``design="between"``), with paired
(``dependent=True``) or independent units. Condition labels are permuted (swapped
within pairs when dependent) and p-values are corrected per layer.

The permutation test is exact only if the tested edges do not depend on the labels.
Edges selected separately in each condition are partly selected *because* of
chance differences between the conditions, which the permutations then destroy: on
random halves of one condition of an iEEG dataset, 13-15% of splits gave a
significant intrinsic or synergistic difference instead of 5%. For a calibrated
test, select one network on both conditions pooled and pass it as both networks:

.. code-block:: python

    pooled = infoflow.infer_multiplex(data.subset_trials(idx_a + idx_b), prng=0)
    cmp = compare_networks(pooled, pooled, data.subset_trials(idx_a), data.subset_trials(idx_b))

Every permutation re-estimates every tested edge, which is slow for finely binned
contexts. ``curtail=10`` stops an edge's permutations once each layer's permuted
difference has reached the observed one ten times (sequential Monte Carlo p-values,
:cite:`Besag1991`), which leaves small p-values unchanged and stops edges with large
ones early; ``map_fn`` runs edges in parallel.

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
files. :func:`infoflow.draw.draw` draws networks as decomposition glyphs, parallel
arcs, stacked planes, or layer matrices, interactively or as static figures
(:doc:`drawing`); :func:`infoflow.plot.plot_layer` and
:func:`infoflow.plot.plot_multiplex` draw single layers with HoloViews
(``pip install infoflow[plot]``).
