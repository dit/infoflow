"""
Input and output: MATLAB / FieldTrip import, BrainNet Viewer export, save and load.
"""

import pickle
from pathlib import Path

import numpy as np
from dit.inference import Trials

__all__ = ("export_brainnet", "load", "load_fieldtrip", "load_mat", "save")


def load_mat(path, variable, dim_order="sp"):
    """
    Load an array from a MATLAB file as (samples, processes) or trials of them.

    Parameters
    ----------
    path : str or Path
    variable : str
        The variable name in the file.
    dim_order : str
        The order of the array's axes, using ``s`` (samples), ``p`` (processes),
        and ``r`` (replications/trials), e.g. ``'ps'`` or ``'psr'``.
    """
    from scipy.io import loadmat

    arr = np.asarray(loadmat(path)[variable], dtype=float)
    order = [dim_order.index(c) for c in ("s", "p", "r") if c in dim_order]
    arr = np.transpose(arr, order)
    if arr.ndim == 3:
        return Trials(arr[:, :, r] for r in range(arr.shape[2]))
    return arr


def load_fieldtrip(path, variable="data"):
    """
    Load a FieldTrip raw-data structure (``data.trial`` cell array of channels x samples).

    Returns
    -------
    trials : Trials
        One (samples, channels) array per trial.
    labels : list of str
        Channel labels.
    """
    from scipy.io import loadmat

    struct = loadmat(path, squeeze_me=True, struct_as_record=False)[variable]
    trials = np.atleast_1d(struct.trial)
    labels = [str(label) for label in np.atleast_1d(struct.label)]
    return Trials(np.asarray(t, dtype=float).T for t in trials), labels


def export_brainnet(network, layer, coordinates, path, significant_only=True):
    """
    Write BrainNet Viewer ``.node`` and ``.edge`` files for one layer.

    Parameters
    ----------
    network : MultiplexNetwork
    layer : str
    coordinates : array_like
        (nodes, 3) MNI coordinates.
    path : str or Path
        Output path without extension.
    """
    path = Path(path)
    names = network.names
    coordinates = np.asarray(coordinates, dtype=float)
    ds = network.dataset.sel(layer=layer)
    weights = np.nan_to_num(ds["weight"].values)
    if significant_only:
        weights = np.where(ds["significant"].values, weights, 0.0)
    with open(path.with_suffix(".node"), "w") as fh:
        for (x, y, z), name in zip(coordinates, names, strict=True):
            fh.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t1\t1\t{name.replace(' ', '_')}\n")
    np.savetxt(path.with_suffix(".edge"), weights, delimiter="\t", fmt="%.6f")
    return path.with_suffix(".node"), path.with_suffix(".edge")


def save(obj, path):
    """
    Pickle a network, comparison, or preprocessing result.
    """
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load(path):
    """
    Load an object saved with :func:`save`.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)
