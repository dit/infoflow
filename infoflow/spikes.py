"""
Spike trains: event times and their binning onto a regular time grid.

Information measures on spike trains are almost always estimated after cutting time
into bins of width :math:`\\Delta t` and recording the spikes in each bin
:cite:`Strong1998`. Fine bins keep spike timing (1 ms bins with several delays
recover cortical connectivity best in :cite:`Ito2011`), but leave a long history
spread over many bins; the encodings in :mod:`infoflow.preprocess.discretizers`
(:class:`~infoflow.preprocess.ExponentialBins`,
:class:`~infoflow.preprocess.SpikeIntervals`,
:class:`~infoflow.preprocess.OrdinalIntervals`) summarize that history in few
symbols.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ("SpikeTrains", "bin_spikes")


def _per_trial(value, n_trials):
    if value is None or np.isscalar(value):
        return [value] * n_trials
    value = list(value)
    if len(value) != n_trials:
        raise ValueError(f"need one value per trial ({n_trials}), got {len(value)}")
    return value


@dataclass
class SpikeTrains:
    """
    Spike times of several neurons, in one or more independent trials.

    Parameters
    ----------
    times : list
        One trial as a list of per-neuron arrays of spike times, or a list of such
        trials.
    dt : float
        Bin width, in the units of `times` (default 5 ms for times in seconds, the
        resolution of :cite:`Rudelt2021`).
    t_start, t_stop : float, list of float, None
        Recording window per trial (scalars apply to every trial). Defaults: 0 and
        just past the last spike of the trial.
    names : list of str, None
        Neuron names.
    """

    times: list
    dt: float = 0.005
    t_start: object = 0.0
    t_stop: object = None
    names: list | None = None

    def __post_init__(self):
        times = list(self.times)
        single = not times or all(np.isscalar(t) for t in times[0])
        trials = [times] if single else times
        self.times = [[np.sort(np.asarray(n, dtype=float)) for n in trial] for trial in trials]
        if len({len(trial) for trial in self.times}) > 1:
            raise ValueError("every trial needs the same neurons")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def n_trials(self):
        return len(self.times)

    @property
    def n_neurons(self):
        return len(self.times[0])


def bin_spikes(spikes, dt=None):
    """
    Spike counts per bin.

    Bin ``i`` of a trial covers :math:`[t_0 + i \\Delta t, t_0 + (i + 1) \\Delta t)`;
    spikes outside :math:`[t_0, t_1)` are dropped.

    Parameters
    ----------
    spikes : SpikeTrains
    dt : float, None
        Bin width (default ``spikes.dt``).

    Returns
    -------
    list of np.ndarray
        Per trial, integer counts of shape (bins, neurons).
    """
    dt = spikes.dt if dt is None else dt
    starts = _per_trial(spikes.t_start, spikes.n_trials)
    stops = _per_trial(spikes.t_stop, spikes.n_trials)
    out = []
    for trial, t0, t1 in zip(spikes.times, starts, stops, strict=True):
        t0 = 0.0 if t0 is None else float(t0)
        if t1 is None:
            last = max((n[-1] for n in trial if len(n)), default=t0)
            t1 = last + dt
        n_bins = max(int(np.ceil((float(t1) - t0) / dt - 1e-9)), 1)
        counts = np.zeros((n_bins, len(trial)), dtype=np.int64)
        for j, times in enumerate(trial):
            idx = np.floor((times - t0) / dt + 1e-9).astype(np.int64)
            idx = idx[(idx >= 0) & (idx < n_bins)]
            np.add.at(counts[:, j], idx, 1)
        out.append(counts)
    return out
