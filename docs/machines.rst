Machines: ε-machines, ε-transducers, and the composed model
============================================================

``infer_multiplex(machines=True)`` (or :func:`infoflow.machines.infer_machines` on an
inferred network) adds a computational-mechanics layer to the network:

* an **ε-machine** for every node: the minimal unifilar predictor of the node's
  symbols, reconstructed by Causal-State Splitting Reconstruction (CSSR)
  :cite:`Shalizi2004`;
* an **ε-transducer** for every discovered edge :cite:`Barnett2015`: the minimal
  unifilar model of the target as a channel driven by the source, reconstructed by
  transCSSR;
* a **joint-input ε-transducer** for every target with parents, whose input symbol
  is the tuple of all its parents;
* the **composed model** (:class:`~infoflow.network_model.NetworkModel`): one
  generative model of the whole network built from these pieces.

The layer needs sofic (``pip install 'infoflow[machines]'``).

.. code-block:: python

   import infoflow

   net = infoflow.infer_multiplex(data, machines=True, prng=0)
   net.machines.nodes["x0"]            # EpsilonMachine
   net.machines.edges[("x0", "x1")]    # EpsilonTransducer of the edge
   net.machines.targets["x1"]          # EpsilonTransducer reading all of x1's parents
   net.machines.dataset                # node, target, and edge measures
   print(net.summary())                # the layers, then the machine table

Inputs and delays
-----------------

Machines are reconstructed from the nodes' preprocessing *present* symbols
(``MachineSettings(symbols="past")`` uses the past codes; ordinal past codes are
overlapping windows, not a clean symbol process). The trials are pooled, but no
history ever spans the boundary between two trials.

A transducer is Mealy: on reading the input symbol at ``t`` it emits the target's
symbol at ``t``. The input at ``t`` is the source at ``t - d``, for the edge's delay
``d`` (the network's ``delay`` variable), so a transducer reads only the source's
past: a channel that copies the source with delay ``d`` is a one-state transducer.
A target transducer reads each parent at its own delay.

A target's joint input alphabet is the product of its parents' alphabets. Targets
whose alphabet exceeds ``max_joint_alphabet`` are skipped, and the reason is recorded
in ``machines.skipped``. ``MachineSettings(input_symbols="coarse")`` reads a median
split of each parent instead (of its raw values, or of its symbols for discrete
data), which shrinks the alphabet to :math:`2^k` for :math:`k` parents.

History lengths default to each node's lag budget for the node machines and to at
most 2 for the transducers, whose joint history space grows as
:math:`(|X||Y|)^L`. The split tests run at ``alpha=0.001`` with a Bonferroni
correction, which favors fewer, robust states. With ``stability=True`` (the default)
every machine is refit on the two time halves of the data (``stability``: the
fraction of halves that reproduce the topology), and every node machine at
``max_history + 1``. A node that gains states there is marked ``unsynchronized`` with
a warning: its history is too short to synchronize, so its state count is a lower
bound, not the process's ε-machine. Every node machine is also replayed on the data
it was fitted to; ``n_forbidden`` counts the symbols it assigns probability zero once
synchronized, with a warning when any. A too-short history can turn a long-period
process into a confidently wrong deterministic cycle, and this is how it shows.

Two limits are worth knowing. Processes whose Markov order is infinite (no finite
history synchronizes every past, as for the tent map's generating partition at
Misiurewicz parameters) have no finite-history reconstruction: CSSR splits the
unsynchronized histories into extra, low-probability states, while the entropy rate
still converges. And transCSSR's cost grows with :math:`(|X||Y|)^L`: for a target
with three binary parents a joint history of 3 already takes minutes and over-splits
on tens of thousands of samples, so ``transducer_max_history`` defaults to at most 2.
``excess_entropy=False`` skips the excess entropy and crypticity, which need the
reverse ε-machine and can take tens of seconds per node when it is infinite.

Measures
--------

The dataset carries, per node, the statistical complexity ``C_mu``, entropy rate
``h_mu``, excess entropy ``E``, crypticity, and the number of states. Channel
quantities are defined relative to a driving input process :cite:`Barnett2015`; the
driver of each transducer is the ε-machine of its own input sequence, so the channel
complexity ``channel_C``, the driven entropy rate ``driven_h``, and the transfer
entropy ``te_model`` describe the channel driven by an independent copy of its
input. In a feedback loop that is not the observed joint process; the composed
model below is.

``te_model`` is :math:`I[Y_t : U_{t-h..t} \mid Y_{t-h..t-1}]`
(:func:`~infoflow.machines.driven_transfer_entropy`), which includes the current
input because it is already in the source's past. ``te_gap`` is ``te_model`` minus
the network's estimated transfer entropy: a model-adequacy check. For a target with
several parents the network's estimate conditions on the other parents and the
pairwise transducer does not, so the target transducer's collective transfer entropy
``target_te`` is compared with the skeleton's omnibus transfer entropy instead
(``target_te_gap``).

``admitted_by`` records which selection estimator admitted each parent. The machines
always work on symbols, so a parent admitted only by the ``gaussian`` or ``trend``
estimator may carry little dependence at the symbol level: a near-trivial transducer
and a large ``te_gap`` are then expected, not a failure of the model.

The composed model
------------------

Each node without parents is a *source*, generated by its ε-machine; every other node
is *driven*, generated by its target transducer. Every delay is at least one step,
so each step updates every component from the previous symbols and feedback loops
are well defined (a delayed parent is read from a shift register).

.. code-block:: python

   model = net.machines.model
   model.roles                        # {"x0": "source", "x1": "driven", ...}
   model.simulate(5000, seed=0)       # (samples, processes) symbol codes
   model.log_loss(heldout)            # bits per symbol: model vs each node's eps-machine
   model.adequacy(heldout, seed=0)    # parametric-bootstrap check, per node
   model.to_hmm()                     # explicit product HMM, up to max_composed_states

* :meth:`~infoflow.network_model.NetworkModel.log_loss` scores held-out data. The
  machines are unifilar, so once the data synchronize a component its state is known
  and the likelihood is a sum of per-node terms. The improvement over a node's own
  ε-machine estimates the collective transfer entropy from its parents and is
  reported beside the skeleton's omnibus estimate.
* :meth:`~infoflow.network_model.NetworkModel.adequacy` compares each node's held-out
  loss with its loss on simulations from the model. The product of machines is exact
  only if a node's present is independent of the other nodes' presents given its own
  past and its parents' pasts; coarse-grained deterministic systems (coupled maps
  seen through a partition) violate this, and their nodes are flagged.
* When the explicit model has at most ``max_composed_states`` states, its exact flows
  per edge (``model_te``, ``model_intrinsic``, ``model_synergistic``, ``model_shared``,
  ``model_tdmi``) are added beside the estimated layers. Larger models still simulate
  and score, and the reason is recorded in ``machines.skipped``.
* :meth:`~infoflow.network_model.NetworkModel.parametric_check` simulates surrogate
  trials, re-infers the network, and compares it with the original
  (:func:`~infoflow.compare_networks`).

Pickling a network (:meth:`~infoflow.MultiplexNetwork.save`) stores the machines
through sofic's serialization and rebuilds the composed model on load.

Validation on coupled tent maps
-------------------------------

``tests/test_tent_ring.py`` (``pytest -m slow``)
couples four tent maps on the ring A → B, B → C, B → D, C → A, D → A, each at a
Misiurewicz slope whose exact generating-partition ε-machine has 4, 5, 6, or 7
states, and observes them through the partition at 1/2 (20000 samples, lag budget 5):

* **Uncoupled**: no edges; every node's entropy rate is within 0.02 bits of
  :math:`\log_2 s`. The exact topologies are not recovered (infinite Markov order).
* **Weak (0.1) and moderate (0.3) coupling**: every true edge is found, with extra
  edges (2 and 7): the binary partition does not screen off the maps' continuous
  state, so indirect paths leak through. In the moderate regime the composed model
  beats each node's own ε-machine by 0.12--0.14 bits per symbol on held-out data; in
  the weak regime (flows of about 0.04 bits) it does not.
* **Synchronized (0.5, identical slopes)**: identical series, no edges, identical
  node machines -- synchronization hides the topology.
* **Period-12 window**: no edges, zero entropy rates, cycles of at most 12 states
  (with ``max_history=12``; at 5 node C is flagged by ``n_forbidden``).

With a lag budget of 2 instead of 5, the synchronized ring shows spurious edges:
source candidates reach lags beyond the target's own past budget, and in identical
series a source at such a lag is the target's own past.

With ``infer()``
----------------

The machine layer is off in every :func:`~infoflow.infer` preset. Pass
``infer(data, machines=True, machine_settings=...)``: like any pinned option it is
used as given, so it runs outside the time budget (the report warns when a budget is
set). ``result.report.machines`` records its time, counts, and skipped machines.
