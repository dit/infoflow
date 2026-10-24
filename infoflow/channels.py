"""
The channel layer: a discrete memoryless channel for every edge.

Each discovered edge ``s -> t`` with delay ``d`` is read as a discrete memoryless
channel :cite:`Cover2006` from the source's symbol at ``t - d`` to the target's symbol
at ``t``, with the transition matrix :math:`p(y \\mid x)` estimated from the aligned
present symbols, pooled over trials. Two quantities are reported per edge:

* the mutual information :math:`I[X : Y]` at the observed input distribution: what
  the edge carries as driven;
* the capacity :math:`C = \\max_{p(x)} I[X : Y]` of the estimated channel, computed by
  the Blahut--Arimoto algorithm :cite:`Blahut1972,Arimoto1972`: what it could carry
  if the source were free to choose its input distribution.

The channel ignores memory: dependence of the target on its own past and on earlier
inputs is averaged into :math:`p(y \\mid x)`. The ε-transducers of the machine layer
(:mod:`infoflow.machines`) keep that memory, and bracket the capacity of the stateful
channel with and without feedback. Feedback cannot increase the capacity of a
memoryless channel :cite:`Cover2006`, so a single capacity is reported here.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import xarray as xr

from .machines import _aligned, _as_data
from .measures import cmi_from_joint

__all__ = ("ChannelLayer", "ChannelSettings", "edge_channel", "infer_channels")


@dataclass
class ChannelSettings:
    """
    Options of the channel layer.

    Attributes
    ----------
    estimator : {'miller_madow', 'plugin'}, None
        Estimator of the mutual information (None: the network's). The capacity is
        always the plug-in capacity of the estimated channel.
    include_shared_candidates : bool
        Also estimate channels for shared-only candidate edges.
    """

    estimator: str | None = None
    include_shared_candidates: bool = False


@dataclass
class ChannelLayer:
    """
    The channel layer of a network.

    Attributes
    ----------
    dataset : xarray.Dataset
        Edge variables over ``(source, target)``: ``mi`` (mutual information at the
        observed input distribution), ``capacity`` (capacity of the estimated
        channel), ``n_inputs`` and ``n_outputs`` (observed input and output symbols),
        and ``n_samples``. All in bits; NaN where no channel was estimated.
    channels : dict
        ``(source name, target name) -> DataArray`` of :math:`p(y \\mid x)` over
        ``(input, output)``.
    optimal_inputs : dict
        ``(source name, target name) -> DataArray`` over ``(input,)``: the
        capacity-achieving input distribution.
    settings : dict
        The :class:`ChannelSettings` used.
    seconds : float
        Wall-clock time of the layer.
    """

    dataset: xr.Dataset
    channels: dict = field(default_factory=dict)
    optimal_inputs: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    seconds: float = float("nan")

    def summary(self):
        """
        A text table of the edge channels.
        """
        lines = ["source -> target  inputs  outputs      mi  capacity"]
        for src, tgt in sorted(self.channels):
            e = self.dataset.sel(source=src, target=tgt)
            lines.append(
                f"{src:>6} -> {tgt:<6}  {int(e['n_inputs']):6d}  {int(e['n_outputs']):7d}  "
                f"{float(e['mi']):6.3f}  {float(e['capacity']):8.3f}"
            )
        return "\n".join(lines)


def edge_channel(x, y, estimator="miller_madow"):
    """
    The discrete memoryless channel from aligned inputs `x` to outputs `y`.

    Parameters
    ----------
    x, y : array_like
        Aligned input and output symbols.
    estimator : {'miller_madow', 'plugin'}
        Estimator of the mutual information.

    Returns
    -------
    channel : xarray.DataArray
        :math:`p(y \\mid x)` over ``(input, output)``, for the observed symbols.
    optimal_input : xarray.DataArray
        The capacity-achieving input distribution over ``(input,)``.
    mi : float
        :math:`I[X : Y]` at the observed input distribution, in bits.
    capacity : float
        The capacity of `channel` (Blahut--Arimoto :cite:`Blahut1972,Arimoto1972`), in bits.
    """
    from dit.algorithms import channel_capacity

    xs, xc = np.unique(np.asarray(x), return_inverse=True)
    ys, yc = np.unique(np.asarray(y), return_inverse=True)
    counts = np.zeros((len(xs), len(ys)))
    np.add.at(counts, (xc.ravel(), yc.ravel()), 1.0)
    matrix = counts / counts.sum(axis=1, keepdims=True)
    mi = max(0.0, cmi_from_joint(counts.T[:, :, None], estimator))
    # Blahut-Arimoto approaches C from below; dit's default tolerance can stop short of
    # the observed input's plug-in mutual information.
    capacity, p_opt = channel_capacity(matrix, rtol=1e-10, atol=1e-12)
    channel = xr.DataArray(matrix, dims=("input", "output"), coords={"input": xs, "output": ys})
    optimal = xr.DataArray(np.asarray(p_opt), dims=("input",), coords={"input": xs})
    return channel, optimal, mi, max(0.0, float(capacity))


def infer_channels(data, network, settings=None):
    """
    Estimate the channel layer of an inferred network.

    Parameters
    ----------
    data : DiscreteData or integer array_like / Trials
        The symbols the network was inferred on.
    network : MultiplexNetwork
    settings : ChannelSettings or dict, None

    Returns
    -------
    ChannelLayer
    """
    started = time.perf_counter()
    if settings is None:
        settings = ChannelSettings()
    elif isinstance(settings, dict):
        settings = ChannelSettings(**settings)
    estimator = settings.estimator or network.settings.get("estimator", "miller_madow")
    names = network.names
    data = _as_data(data, names)
    P = data.n_processes
    ds_net = network.dataset
    delay = ds_net["delay"].values
    kind = ds_net["kind"].values
    edge_kinds = ("parent", "shared_candidate") if settings.include_shared_candidates else ("parent",)

    layer = ChannelLayer(dataset=xr.Dataset(), settings=asdict(settings))
    edge_vars = {k: np.full((P, P), np.nan) for k in ("mi", "capacity", "n_inputs", "n_outputs", "n_samples")}
    for src, tgt in sorted(network.edges):
        if kind[src, tgt] not in edge_kinds:
            continue
        xs, ys = _aligned(data, tgt, [(src, int(delay[src, tgt]))], data.present, data.present_offset, False)
        x, y = np.concatenate([np.asarray(v, dtype=np.int64) for v in xs]), np.concatenate(ys).astype(np.int64)
        channel, optimal, mi, capacity = edge_channel(x, y, estimator)
        key = (names[src], names[tgt])
        layer.channels[key], layer.optimal_inputs[key] = channel, optimal
        values = {
            "mi": mi,
            "capacity": capacity,
            "n_inputs": channel.sizes["input"],
            "n_outputs": channel.sizes["output"],
            "n_samples": len(y),
        }
        for k, v in values.items():
            edge_vars[k][src, tgt] = v
    layer.dataset = xr.Dataset(
        {k: (("source", "target"), v) for k, v in edge_vars.items()},
        coords={"source": names, "target": names},
        attrs={"description": "Channel layer: discrete memoryless channel per edge (bits)", "estimator": estimator},
    )
    layer.seconds = time.perf_counter() - started
    return layer
