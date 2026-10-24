Channels: mutual information and capacity per edge
===================================================

``infer_multiplex(channels=True)`` (or :func:`infoflow.channels.infer_channels` on an
inferred network) reads every discovered edge ``s -> t`` with delay ``d`` as a
discrete memoryless channel :cite:`Cover2006` from the source's symbol at ``t - d``
to the target's symbol at ``t``. Its transition matrix :math:`p(y \mid x)` is
estimated from the nodes' present symbols, pooled over trials without letting a pair
span two of them (as for the machine layer's transducers).

.. code-block:: python

   import infoflow

   net = infoflow.infer_multiplex(data, channels=True, prng=0)
   net.channels.channels[("x0", "x1")]        # p(y | x) over (input, output)
   net.channels.optimal_inputs[("x0", "x1")]  # capacity-achieving p(x)
   net.channels.dataset                       # mi, capacity, n_inputs, n_outputs, n_samples
   print(net.summary())

Measures
--------

* ``mi`` is :math:`I[X : Y]` at the observed input distribution: the information
  the edge carries as the source actually drives it. It uses the network's estimator
  (Miller--Madow by default :cite:`Miller1955`), or
  ``ChannelSettings(estimator=...)``.
* ``capacity`` is :math:`C = \max_{p(x)} I[X : Y]` of the estimated channel,
  computed by the Blahut--Arimoto algorithm :cite:`Blahut1972,Arimoto1972`
  (:func:`dit.algorithms.channel_capacity`): the most the edge could carry if the
  source were free to choose its input distribution.

``mi`` never exceeds ``capacity``, and the gap measures how far the source's
statistics are from the ones the channel is best at. A binary symmetric channel with
crossover 0.1 has capacity :math:`1 - h(0.1) \approx 0.531` bits, but a source that
emits 1 one time in ten drives only :math:`h(0.18) - h(0.1) \approx 0.211` bits
through it.

``mi`` is the time-delayed mutual information between the source's present symbol at
one lag and the target's present symbol. The network's ``tdmi`` layer instead uses
the source's whole past embedding, so the two can differ. The capacity is the plug-in
capacity of the estimated channel. Like any plug-in information estimate it is biased
upward when there are few samples per input symbol.

Memory and feedback
-------------------

The memoryless channel averages the target's dependence on its own past and on
earlier inputs into :math:`p(y \mid x)`. Feedback cannot increase the capacity of a
memoryless channel :cite:`Cover2006`, so a single capacity is reported. To keep the
memory, use the machine layer (:doc:`machines`): each edge's ε-transducer is a
finite-state channel, and its capacity is bracketed both without and with feedback.
For a memoryless edge, the transducer has one state and all three capacities agree.
