Spike trains
============

Information measures on spike trains are usually estimated after cutting time into
bins of width :math:`\Delta t` and recording which bins hold a spike
:cite:`Strong1998`. Fine bins keep spike timing, and transfer entropy over several
delays at 1 ms resolution recovers cortical connectivity far better than one
delay at coarse resolution :cite:`Ito2011`; but a neuron's own history can reach
back hundreds of milliseconds, far more bins than any context can hold.
``infoflow`` therefore encodes each neuron's past in one of three spike-history
encodings, chosen per neuron by its cross-validated history dependence.

.. code-block:: python

   import infoflow
   from infoflow.datasets import spiking_glm
   from infoflow.selection import SkeletonSettings
   from infoflow.spikes import SpikeTrains

   spikes, true_edges = spiking_glm(n=20000, seed=1)   # SpikeTrains, [(source, target, delay)]
   # or: spikes = SpikeTrains([times_neuron_0, times_neuron_1, ...], dt=0.005)
   net = infoflow.infer_multiplex(spikes, skeleton_settings=SkeletonSettings(null="strata"))

Input
-----

:class:`~infoflow.spikes.SpikeTrains` holds per-neuron spike times, in one or more
trials, and a bin width ``dt`` (default 5 ms for times in seconds, as in
:cite:`Rudelt2021`). The bin width is the user's choice: it sets the finest timing
any measure can see. Already-binned counts are accepted with
``preprocess(counts, spikes=True)`` (or ``infer_multiplex(counts,
preprocess={"spikes": True})``).

Encodings
---------

In every encoding the *present* of a neuron at bin :math:`t` is its spike indicator,
and the *past* symbol at :math:`t` summarizes a window of bins ending at :math:`t`;
stacked lags of one neuron are spaced by the window, so their windows are disjoint.

:class:`~infoflow.preprocess.ExponentialBins` ``(d, kappa, window)``
   The window is split into :math:`d` sub-bins whose widths grow as
   :math:`\tau_1 10^{j\kappa}` from the present backwards, and the symbol records
   which sub-bins hold a spike (:math:`2^d` symbols) :cite:`Rudelt2021`. With
   :math:`d` equal to the window every bin is resolved: the binary words of
   :cite:`Ito2011`.

:class:`~infoflow.preprocess.SpikeIntervals` ``(k, bins, max_interval)``
   The time since the last spike and the :math:`k-1` preceding inter-spike intervals,
   each in equal-frequency bins (equivalently, bins of their logarithm
   :cite:`Dorval2008,Dorval2011`), plus one symbol for fewer than :math:`k` spikes
   in the window. This is a binned form of the interval history used by
   continuous-time transfer entropy :cite:`Shorten2021`, and suits renewal-like
   neurons whose firing depends on the time since their last spike.

:class:`~infoflow.preprocess.OrdinalIntervals` ``(order, max_interval)``
   The ordinal pattern :cite:`Bandt2002` of the most recent ``order`` inter-spike
   intervals, plus one symbol for too few spikes in the window; invariant to any
   monotone rescaling of the intervals :cite:`Masoliver2020`.

Choice
------

Each candidate (grid in :data:`~infoflow.preprocess.pipeline.SPIKE_GRID`; windows
log-spaced from one bin to ``max_history``, 200 bins by default) is scored by the
held-out fraction of the spike indicator's uncertainty explained by the past, the
history dependence :math:`R = I[\text{past} : \text{present}] / H(\text{present})`
of :cite:`Rudelt2021` estimated by blocked cross-validation (Rudelt et al.'s
cross-validated variant; no bias criterion is needed). Among candidates within one
standard error of the best, encodings that resolve every bin come first, then the
smallest alphabet, then the shortest window.

The first preference matters for networks. A neuron's own score cannot see how well
its past symbols resolve timing *as a source*, yet a mediating neuron whose recent
bins are merged cannot screen off the upstream neuron it relays: on the three-neuron
chain of :func:`~infoflow.datasets.spiking_glm`, merged bins let the indirect edge
:math:`0 \to 2` through the mediator survive conditioning, while resolved bins
recover exactly the true edges and delays. Coarse windows and interval encodings are
chosen only when they predict clearly better, as for a renewal neuron (interval
encoding) or a neuron with slow adaptation (a window of tens of bins).

Each neuron's lag budget spans at least ``SPIKE_GRID["source_history"]`` bins (30 by
default, the delay range of :cite:`Ito2011`), since synaptic delays outlast most
own-history plateaus. Network resolution is not used for spike data.

Significance
------------

Shifting a source spike train in time does not produce surrogates under the null
hypothesis of conditional independence, and gives high false-positive rates when the
target is driven by other neurons :cite:`Shorten2021`. Use the stratified null,
``SkeletonSettings(null="strata")``, which permutes the source within strata of the
conditioning set, the discrete analogue of Shorten et al.'s local permutation.

Not implemented
---------------

* Continuous-time k-nearest-neighbour transfer entropy :cite:`Shorten2021` avoids
  binning altogether; it is an estimator, not a discretization.
* A neuron's past serves both as its own history and as a source; with a coarse own
  history (a long window or an interval encoding) its timing as a source is limited
  to that encoding. Separate source and history encodings would remove the tradeoff.
