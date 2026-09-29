"""
``map``-like functions for parallel inference over targets and edges.

Targets (and edges) are independent, so :func:`~infoflow.infer_multiplex` and
:func:`~infoflow.preprocess.preprocess` accept any ``map_fn``. Threads work well
because the heavy lifting is in NumPy and SciPy; dask adds its schedulers,
including distributed clusters.
"""

from concurrent.futures import ThreadPoolExecutor

__all__ = ("dask_map", "thread_map")


def thread_map(n_jobs=None):
    """
    A ``map`` over a thread pool, preserving order.
    """

    def map_fn(fn, iterable):
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            return list(pool.map(fn, iterable))

    return map_fn


def dask_map(scheduler="threads", **compute_kwargs):
    """
    A ``map`` through :func:`dask.delayed` with the given scheduler.
    """
    try:
        import dask
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("dask_map needs the optional dependency dask (pip install infoflow[parallel])") from error

    def map_fn(fn, iterable):
        tasks = [dask.delayed(fn)(item) for item in iterable]
        return list(dask.compute(*tasks, scheduler=scheduler, **compute_kwargs))

    return map_fn
