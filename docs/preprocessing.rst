Preprocessing
=============

:func:`infoflow.preprocess.preprocess` turns continuous series into symbols with
as few free parameters as possible. For each process it

1. **screens** the series: splits at missing values, measures ties, and tests
   stationarity with a G-test on ordinal-pattern frequencies across segments;
2. proposes **delays** on the rank (copula) transform: the first minimum of the
   auto mutual information :cite:`Garland2014`, the first minimum of the smoothed
   mean Menger curvature of the delay reconstruction :cite:`Deshmukh2021`, and the
   autocorrelation decay time;
3. scores **discretizers** by cross-validated predictability of a multi-resolution
   reference target (equal-frequency 2/4/8-bin codes of the next value, each
   normalized as in :cite:`Rudelt2021`) and keeps the smallest alphabet within one
   standard error of the best, a forecast-driven choice in the spirit of
   :cite:`Garland2015,Garland2016`. Candidates are equal-frequency and equal-width
   bins, a median threshold, and ordinal patterns :cite:`Bandt2002` (with the
   present encoded as the bin of the current value alone, so it carries nothing
   about past values); weighted permutation entropy
   :cite:`Fadlallah2013` breaks ties between ordinal candidates;
   A node whose own future no candidate predicts gives no basis for a choice; it
   keeps equal-frequency bins at the middle reference resolution and is flagged
   ``unpredictable``;
4. chooses a **lag budget** with the bootstrap plateau rule of :cite:`Rudelt2021`,
   cross-checked against a BIC Markov order.

Surrogates are generated on the raw series and re-encoded
(:func:`~infoflow.preprocess.discretizers.reencode`), so permutation tests see the
same symbolization as the data :cite:`Staniek2008`.

:mod:`infoflow.preprocess.scaling` finds scaling regions in block-entropy curves
:cite:`Deshmukh2020` for entropy rate and excess entropy summaries; delay
reconstruction for nonlinear systems follows :cite:`Bradley2015`.

The result's ``report.summary()`` records every candidate and the reason each
choice was made.
