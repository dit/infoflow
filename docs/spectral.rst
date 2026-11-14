Gaussian spectral information rates
===================================

:mod:`infoflow.spectral` estimates symmetric, whole-trajectory dependence for
stationary, jointly Gaussian processes :cite:`Komaee2020,Raiola2026`. It works
independently of the intrinsic/shared/synergistic layers. On non-Gaussian data it
is a Gaussian spectral approximation, not a general mutual-information estimator.

Rates and units
---------------

The mutual information rate is the long-observation-time limit of the information
between two trajectories divided by their duration. If :math:`S(f)` is their
spectral matrix and :math:`\phi_{ij}(f)=|S_{ij}|^2/(S_{ii}S_{jj})` their coherence,
its Gaussian expression on a nonnegative Hz grid is

.. math::

   R_{ij}=-\int_0^{f_{\max}}\log_2(1-\phi_{ij}(f))\,df.

For sampled observations, :math:`f_{\max}` is the available Nyquist bandwidth
(up to grid/quadrature resolution); the continuous-time formula integrates to
infinity. A nonuniform explicit grid integrates only the supplied bandwidth.
Physical rates have units **bits per unit of time**, with time defined by ``fs``.
Divide by ``fs`` for bits per sample. An angular-frequency integral instead has
prefactor :math:`1/(2\pi\ln 2)` on its nonnegative half. Do not double-count the
one-sided PSD scaling. The paper uses natural logarithms: divide its nats by
:math:`\ln 2` to compare with these bit-valued results.

Conditioning on the whole trajectories of all other processes replaces coherence
by partial coherence, :math:`|P_{ij}|^2/(P_{ii}P_{jj})`, where
:math:`P(f)=S(f)^{-1}` :cite:`Dahlhaus2000`. A subset can instead be conditioned
out by the Schur complement (:func:`~infoflow.spectral.conditional_coherence`).
This conditions on whole trajectories, not merely a prediction target's past.
It is **not transfer entropy, intrinsic flow, or a causal orientation**.

With supplied numeric relaxation times, ``rescaled_rate`` multiplies each rate
by :math:`\max(\tau_i,\tau_j)` :cite:`Raiola2026`. It is measured in bits per
reference-time unit. This coordinate change does not resample/filter the data.
An embedding delay or marginal ACF decay is not automatically an intrinsic
relaxation time in a coupled system.

Analysis alone
--------------

.. code-block:: python

    import numpy as np
    from infoflow import datasets, spectral_rates, SpectralSettings

    # coupling[target, source]; one fast-to-slow link.
    A = np.array([[0., 0.], [1., 0.]])
    x = datasets.multiscale_ou(10000, A, [1., 5.], dt=0.1, seed=0)
    rates = spectral_rates(
        x, fs=10., timescales=[1., 5.],
        settings=SpectralSettings(nperseg=512),
    )
    print(rates.summary())
    rates.dataset["rate"]               # bits/time; symmetric, masked diagonal
    rates.dataset["conditional_rate"]   # conditional on every other process
    rates.dataset["rescaled_rate"]      # optional supplied-timescale normalization

``spectral_rates`` accepts (samples, processes) arrays and independent trials with
common process order and sampling frequency. Trials must share a stationary
spectral law. Missing values must first be split into complete multivariate
intervals; no Welch window crosses a trial boundary. ``DiscreteData`` is rejected
so symbol codes are not silently treated as continuous observations.

:func:`~infoflow.spectral.rates_from_spectrum` evaluates an analytic/external
Hermitian positive-semidefinite matrix directly. Its convention is
:math:`S_{ij}=X_iX_j^*`. It can test the paper's population identities, but it
does not attach sampling significance or infer MIRMO labels.

Numerical and statistical diagnostics
----------------------------------------

The Welch matrix :cite:`Welch1967` uses common windows for every pair, weighted by
the number of segments in each trial. ``SpectralSettings`` controls segment length,
overlap, window, FFT size and constant detrending (or no detrending).

* ``raw_spectral_rank`` reports the original rank; ``spectral_rank`` and
  ``condition_number`` describe the scale-normalized matrix actually evaluated.
* ``conditional_valid`` and ``conditional_invalid_fraction`` expose failures of
  full conditioning. An unestimable conditional integral is NaN, not zero.
* Full conditioning needs enough spectral replicates relative to process count.
  Overlapping segments are not independent and zero-padding adds no replicates.
* ``shrinkage`` explicitly requests a diagonal-target regularization. It is never
  guessed automatically; ``raw_spectrum``, ``spectrum`` and
  ``shrinkage_coherence_change`` preserve its effect. It changes the estimand and
  may change which small dependencies are detectable.
* Perfect coherence may give an infinite rate. Materially invalid spectra are
  rejected, not repaired by silently capping coherence or using a pseudoinverse.
* Sample coherence is biased positive. Its integrated rate can be appreciably
  biased even for independent series, especially with few segments and large
  bandwidth. Neither positivity nor a positive percentile-bootstrap interval is
  an independence test. See :doc:`backbone` for the separate experimental nulls.

The implementation is a NumPy/SciPy spectral route, not an alternative continuous
backend for the existing discrete flow-layer estimator. Spectral MIR and the
per-symbol TE/TDMI used by those layers must not be added or substituted for one
another.
