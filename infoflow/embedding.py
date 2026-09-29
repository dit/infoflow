"""
Candidate grids for non-uniform embedding.

Rather than a contiguous block of history, IDTxl-style non-uniform embedding
selects individual past variables ``(process, lag)`` from a candidate grid
:cite:`Faes2011,Novelli2019`. Each process has its own grid, set by its lag
budget, its delay :math:`\\tau`, and (for ordinal symbols) a lag step that keeps
the windows of any two candidates disjoint. Log-spaced grids cover a distant past
with few variables, following the exponentially growing past bins of Rudelt et al.
:cite:`Rudelt2021`.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ("Embedding", "candidate_lags")


@dataclass
class Embedding:
    """
    One process's embedding parameters.

    Attributes
    ----------
    max_lag : int
        The lag budget: the largest lag considered.
    min_lag : int
        The smallest lag considered (1: the most recent past).
    tau : int
        Spacing between candidate lags (the delay).
    log_spaced : bool
        Use roughly geometric lag spacing up to `max_lag` instead of a uniform grid.
    """

    max_lag: int = 3
    min_lag: int = 1
    tau: int = 1
    log_spaced: bool = False

    def lags(self, lag_step=1):
        """
        The candidate lags, spaced at least ``max(tau, lag_step)`` apart.
        """
        step = max(int(self.tau), int(lag_step), 1)
        if self.log_spaced:
            raw = np.unique(np.round(np.geomspace(self.min_lag, max(self.max_lag, self.min_lag), num=16)).astype(int))
            lags, last = [], None
            for lag in raw:
                if last is None or lag - last >= step:
                    lags.append(int(lag))
                    last = lag
            return lags
        return list(range(int(self.min_lag), int(self.max_lag) + 1, step))


def candidate_lags(data, embeddings, process):
    """
    The candidate lags of `process` under its embedding and lag step.
    """
    embedding = embeddings[process] if isinstance(embeddings, (list, tuple)) else embeddings
    return embedding.lags(int(data.lag_step[process]))
