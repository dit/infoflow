"""
The machine layer: ε-machines for nodes and ε-transducers for edges and targets.

Each node's symbol process is reconstructed as an ε-machine by Causal-State
Splitting Reconstruction :cite:`Shalizi2004`. Each discovered edge gets a pairwise
ε-transducer :cite:`Barnett2015`, reconstructed by transCSSR from the source shifted
by the edge's delay (input) and the target (output); each target with parents gets
one joint-input ε-transducer whose input is the tuple of all its delay-shifted
parents. The transducers are Mealy: the input symbol at ``t`` is the source at
``t - delay``, already in the source's past, so a transducer never reads the
future. Every learner pools the trials without letting a history span two of them.

Channel quantities are defined relative to a driving input process
:cite:`Barnett2015`; here the driver is the inferred ε-machine of the transducer's
own input sequence. They describe the channel as if driven by an independent copy
of its input, so in feedback loops they are not quantities of the observed joint
process (use :class:`~infoflow.network_model.NetworkModel` for that). Each edge
transducer's capacity, without and with feedback, needs no driver: it is bracketed
for the transducer as a finite-state channel (:func:`transducer_capacity`).
"""

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .data import DiscreteData, as_trials

if TYPE_CHECKING:
    from .network_model import NetworkModel

__all__ = ("MachineLayer", "MachineSettings", "infer_machines", "transducer_capacity")


def _require_sofic():
    try:
        from sofic.inference import Trials  # noqa: F401
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the machine layer needs sofic with multi-trial CSSR (sofic.inference.Trials): "
            "pip install 'infoflow[machines]'"
        ) from err


@dataclass
class MachineSettings:
    """
    Options of the machine layer.

    Attributes
    ----------
    symbols : {'present', 'past'}
        Node symbols: the preprocessing present symbols, or the past codes. Ordinal
        past codes are overlapping windows, not a clean symbol process.
    input_symbols : {'present', 'coarse'}
        Transducer inputs: the source's node symbols, or a median split of its raw
        values (of its symbols, for discrete data), which shrinks a target's joint
        input alphabet to ``2 ** n_parents``.
    max_history : int, 'auto', None
        CSSR history length of the node machines. None uses each node's lag budget
        (its embedding's ``max_lag``); ``'auto'`` the Markov order estimated by
        :func:`sofic.inference.suggest_max_history`.
    transducer_max_history : int, None
        transCSSR joint-history length. None uses ``min(node max_history, 2)`` of the
        target: the joint history space grows as ``(|X| |Y|) ** L``.
    alpha : float
        Significance level of the causal-state split tests (Bonferroni-corrected
        within each reconstruction). Small values favor fewer, robust states.
    test : {'g', 'chi2', 'exact'}
        The split test.
    min_count : int
        Histories seen fewer times are not tested.
    max_joint_alphabet : int
        A target transducer whose joint input alphabet exceeds this is skipped, and
        the reason recorded.
    include_shared_candidates : bool
        Also fit pairwise transducers for shared-only candidate edges.
    te_history : int
        Steps of the target's past (and of the input window before its current
        symbol) in the transducer transfer entropy.
    stability : bool
        Refit every machine on the two time halves of the data (topology stability)
        and every node machine at ``max_history + 1`` (state growth).
    excess_entropy : bool
        Compute each node's excess entropy and crypticity. They need the reverse
        ε-machine, which is infinite for many processes; sofic then falls back to
        block entropies, which can take tens of seconds per node.
    max_composed_states : int
        Largest explicit composed HMM built by :class:`~infoflow.network_model.NetworkModel`.
    exact_flows : bool
        Compute exact per-edge flows of the composed model, when it fits under
        `max_composed_states`.
    capacity : bool
        Bracket each edge transducer's capacity without and with feedback
        (:func:`transducer_capacity`). The no-feedback bracket takes seconds to tens of
        seconds per edge with memory.
    capacity_options, feedback_capacity_options : dict
        Keyword arguments of :func:`sofic.generators.capacity.capacity` and
        :func:`sofic.generators.capacity.feedback_capacity`.
    """

    symbols: str = "present"
    input_symbols: str = "present"
    max_history: int | str | None = None
    transducer_max_history: int | None = None
    alpha: float = 0.001
    test: str = "g"
    min_count: int = 5
    max_joint_alphabet: int = 64
    include_shared_candidates: bool = False
    te_history: int = 1
    stability: bool = True
    excess_entropy: bool = True
    max_composed_states: int = 4096
    exact_flows: bool = True
    capacity: bool = True
    capacity_options: dict = field(default_factory=dict)
    feedback_capacity_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.symbols not in ("present", "past"):
            raise ValueError("symbols must be 'present' or 'past'")
        if self.input_symbols not in ("present", "coarse"):
            raise ValueError("input_symbols must be 'present' or 'coarse'")


@dataclass
class MachineLayer:
    """
    The machine layer of a network.

    Attributes
    ----------
    dataset : xarray.Dataset
        Node variables over ``(node,)``: ``C_mu``, ``h_mu``, ``E``, ``crypticity``,
        ``n_states``, ``n_symbols`` (symbols counted), ``n_forbidden`` (fitted symbols
        the machine assigns probability zero once synchronized, with a warning when
        any), ``max_history``, ``stability``
        (fraction of time halves reproducing the topology) and ``unsynchronized``
        (more states at ``max_history + 1``). Target variables over ``(node,)``,
        prefixed ``target_``, for the joint-input transducers: ``target_C``
        (channel statistical complexity), ``target_h`` (driven entropy rate),
        ``target_te`` (collective transfer entropy from all parents),
        ``target_te_gap`` (``target_te`` minus the skeleton's omnibus TE),
        ``target_n_states``, ``target_n_symbols``, ``target_stability``. Edge variables
        over ``(source, target)`` for the pairwise transducers: ``edge_n_states``,
        ``channel_C`` (channel statistical complexity), ``driven_h`` (driven entropy
        rate, from entropy-rate bounds converged to 0.001 bits), ``te_model`` (:func:`driven_transfer_entropy`), ``te_gap`` (``te_model``
        minus the network's ``weight[te]``), ``edge_n_symbols``, ``edge_stability``,
        ``admitted_by`` (the selection estimators that admitted the source), and, with
        ``capacity``, the brackets ``capacity_low`` / ``capacity_high`` and
        ``feedback_capacity_low`` / ``feedback_capacity_high`` of the transducer's
        capacity without and with feedback (:func:`transducer_capacity`). With
        the composed model, ``model_te``, ``model_tdmi``, ``model_intrinsic``,
        ``model_synergistic``, ``model_shared``, ``model_di``: the exact flows (and
        one-step directed information) of :class:`~infoflow.network_model.NetworkModel`.
    nodes : dict
        ``name -> EpsilonMachine``.
    edges : dict
        ``(source name, target name) -> EpsilonTransducer``.
    targets : dict
        ``name -> EpsilonTransducer`` with joint input ``tuple`` symbols.
    inputs : dict
        ``name -> dict(parents=[(parent name, delay), ...], machine=EpsilonMachine)``:
        a target's parents in input order and the ε-machine of its joint input.
    skipped : dict
        ``(kind, key) -> reason`` for every machine that was not fitted.
    settings : dict
        The :class:`MachineSettings` used.
    seconds : float
        Wall-clock time of the layer.
    model : NetworkModel, None
        The composed generative model (:mod:`infoflow.network_model`), or None (with
        the reason in `skipped`) when it cannot be built.
    """

    dataset: xr.Dataset
    nodes: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    seconds: float = float("nan")
    model: "NetworkModel | None" = None

    def __getstate__(self):
        from sofic.serialization import model_to_dict

        state = dict(self.__dict__)
        state["nodes"] = {k: model_to_dict(m) for k, m in self.nodes.items()}
        state["edges"] = {k: model_to_dict(m) for k, m in self.edges.items()}
        state["targets"] = {k: model_to_dict(m) for k, m in self.targets.items()}
        state["inputs"] = {k: {**v, "machine": model_to_dict(v["machine"])} for k, v in self.inputs.items()}
        state["model"] = self.model is not None
        return state

    def __setstate__(self, state):
        from sofic.serialization import model_from_dict

        state = dict(state)
        state["nodes"] = {k: model_from_dict(m) for k, m in state["nodes"].items()}
        state["edges"] = {k: model_from_dict(m) for k, m in state["edges"].items()}
        state["targets"] = {k: model_from_dict(m) for k, m in state["targets"].items()}
        state["inputs"] = {k: {**v, "machine": model_from_dict(v["machine"])} for k, v in state["inputs"].items()}
        rebuild = state.pop("model")
        self.__dict__.update(state, model=None)
        if rebuild:
            from .network_model import NetworkModel

            self.model = NetworkModel.from_layer(self)

    def report(self):
        """
        Timing, counts, and skips, as recorded in :class:`~infoflow.auto.AutoReport`.
        """
        return {
            "seconds": self.seconds,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "targets": len(self.targets),
            "model": self.model is not None,
            "skipped": {f"{kind} {key}": reason for (kind, key), reason in self.skipped.items()},
        }

    def summary(self):
        """
        A text table of the node machines and the edge transducers.
        """
        ds = self.dataset
        lines = ["node      states    C_mu    h_mu       E  stability   target: states  C_X    TE"]
        for name in ds.coords["node"].values:
            n = ds.sel(node=name)
            row = (
                f"{name!s:<8}  {_fmt_int(n['n_states'])}  {float(n['C_mu']):6.3f}  {float(n['h_mu']):6.3f}  "
                f"{float(n['E']):6.3f}  {float(n['stability']):9.2f}"
            )
            if np.isfinite(float(n["target_n_states"])):
                row += f"   {_fmt_int(n['target_n_states'])}  {float(n['target_C']):6.3f}  {float(n['target_te']):6.3f}"
            lines.append(row)
        if self.edges:
            lines.append("source -> target  states  channel_C  te_model  te_gap     capacity  fb_capacity  admitted_by")
            for src, tgt in sorted(self.edges):
                e = ds.sel(source=src, target=tgt)
                lines.append(
                    f"{src:>6} -> {tgt:<6}  {_fmt_int(e['edge_n_states'])}  {float(e['channel_C']):9.3f}  "
                    f"{float(e['te_model']):8.3f}  {float(e['te_gap']):6.3f}  "
                    f"{_fmt_bracket(e, 'capacity')}  {_fmt_bracket(e, 'feedback_capacity')}  "
                    f"{e['admitted_by'].item()}"
                )
        lines += [f"skipped {kind} {key}: {reason}" for (kind, key), reason in self.skipped.items()]
        return "\n".join(lines)


def _fmt_int(value):
    value = float(value)
    return f"{int(value):6d}" if np.isfinite(value) else f"{'–':>6}"


def _fmt_bracket(edge, name):
    if f"{name}_low" not in edge:
        return f"{'–':>11}"
    low, high = float(edge[f"{name}_low"]), float(edge[f"{name}_high"])
    return f"{low:.3f}-{high:.3f}" if np.isfinite(low) else f"{'–':>11}"


def _codes(data, symbols):
    if symbols == "present":
        return data.present, data.present_offset
    return data.past, data.past_offset


def _coarse_codes(data):
    """
    Per trial, a (processes, samples) array of median splits of each process's ranks.

    Ranks are taken over every defined sample of every trial, from the raw values
    when available and from the present symbols otherwise; undefined samples are -1.
    """
    from .selection import rank_bins

    P = data.n_processes
    values = (
        [np.asarray(r, dtype=float).T for r in data.raw]
        if data.raw is not None
        else [p.astype(float) for p in data.present]
    )
    out = [np.full(p.shape, -1, dtype=np.int64) for p in data.present]
    for p in range(P):
        off = int(data.present_offset[p])
        pieces = [v[p, off : out[i].shape[1]] for i, v in enumerate(values)]
        codes = rank_bins(np.concatenate(pieces), 2) if sum(map(len, pieces)) else np.zeros(0, dtype=np.int64)
        start = 0
        for i, piece in enumerate(pieces):
            out[i][p, off : off + len(piece)] = codes[start : start + len(piece)]
            start += len(piece)
    return out


def _node_trials(data, p, symbols):
    arrays, offsets = _codes(data, symbols)
    off = int(offsets[p])
    return [a[p, off:].tolist() for a in arrays if a.shape[1] > off]


def _aligned(data, target, parents, inputs, input_offsets, as_tuple):
    """
    Per trial, the delay-shifted inputs and the target's present, aligned in time.

    `parents` is a list of ``(process, delay)``; the input at ``t`` is each parent at
    ``t - delay``. Single inputs are scalars unless `as_tuple`.
    """
    start = max([int(data.present_offset[target])] + [int(input_offsets[p]) + int(d) for p, d in parents])
    xs, ys = [], []
    for i, present in enumerate(data.present):
        N = present.shape[1]
        if start >= N:
            continue
        t = np.arange(start, N)
        cols = np.stack([inputs[i][p, t - d] for p, d in parents], axis=1)
        xs.append([tuple(row) for row in cols.tolist()] if as_tuple else cols[:, 0].tolist())
        ys.append(present[target, t].tolist())
    return xs, ys


def _fit_machine(trials, max_history, s):
    from sofic.inference import Trials, learn_epsilon_machine_cssr

    return learn_epsilon_machine_cssr(
        Trials(trials), max_history=max_history, alpha=s.alpha, test=s.test, min_count=s.min_count
    )


def _fit_transducer(xs, ys, max_history, s):
    from sofic.inference import Trials, learn_epsilon_transducer_cssr

    return learn_epsilon_transducer_cssr(
        Trials(xs), Trials(ys), max_history=max_history, alpha=s.alpha, test=s.test, min_count=s.min_count
    )


def _node_measures(machine, excess_entropy=True):
    from sofic.exceptions import SoficError

    out = {
        "C_mu": max(0.0, float(machine.statistical_complexity())),
        "h_mu": float(machine.entropy_rate()),
        "n_states": len(list(machine.states())),
    }
    if not excess_entropy:
        return out
    # E needs the reverse ε-machine, which is infinite for many processes (sofic then
    # falls back to block entropies, which can fail); the node machine stays valid.
    try:
        out["E"] = float(machine.excess_entropy())
        out["crypticity"] = out["C_mu"] - out["E"]
    except (RuntimeError, ValueError, SoficError):
        out["E"] = out["crypticity"] = np.nan
    return out


def filled_transducer(transducer, inputs=None):
    """
    A copy of `transducer` that reads every input symbol in every state.

    A reconstructed state never saw some inputs; driving it with an input process
    would otherwise send those inputs to sofic's absorbing reject state. A missing
    input takes the state's row averaged over the inputs it saw (as in
    :class:`~infoflow.network_model.NetworkModel`).
    """
    from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL

    inputs = frozenset(transducer.alphabets()[0]) | frozenset(inputs or ())
    result = transducer.copy()
    for state in list(result.states()):
        rows = {}
        for tr in result.graph.out_transitions(state):
            rows.setdefault(tr.data[ATTR_SYMBOL], []).append(
                (tr.target, tr.data[ATTR_OUTPUT], float(tr.data[ATTR_PROB]))
            )
        missing = inputs - set(rows)
        if not missing or not rows:
            continue
        pooled = {}
        for options in rows.values():
            for target, output, prob in options:
                pooled[(target, output)] = pooled.get((target, output), 0.0) + prob / len(rows)
        for symbol in missing:
            for (target, output), prob in pooled.items():
                result.add_transition(state, target, symbol, output, prob=prob)
    return result


def driven_transfer_entropy(transducer, input_process, history=1):
    """
    Transfer entropy ``I[Y_t : U_{t-h..t} | Y_{t-h..t-1}]`` of a driven Mealy transducer.

    The input ``U_t`` (the source at ``t - delay``) is included: it is already in the
    source's past, so this matches the transfer entropy of the edge, unlike
    :func:`sofic.generators.channel_measures.transfer_entropy`, which conditions on the
    input's strict past only.
    """
    from dit.multivariate import total_correlation
    from sofic.generators.channel_measures import driven_joint_generator
    from sofic.generators.directional_flow import _window_distribution

    joint = driven_joint_generator(filled_transducer(transducer, input_process.observation_alphabet), input_process)
    dist = _window_distribution(joint, history=history)
    inputs = [2 * k for k in range(history + 1)]
    past = [2 * k + 1 for k in range(history)]
    return float(total_correlation(dist, [[2 * history + 1], inputs], past))


def _channel_measures(transducer, input_process, history):
    from sofic.automata.transducer_operations import transduce_generator
    from sofic.generators.channel_measures import channel_statistical_complexity
    from sofic.generators.measures import entropy_rate

    filled = filled_transducer(transducer, input_process.observation_alphabet)
    # The driven output is a non-unifilar HMM: its mixed states rarely close, so the
    # exact rate can take minutes, while the Cover & Thomas bounds converge fast.
    driven_h = entropy_rate(transduce_generator(filled, input_process), "bounds", tol=1e-3, max_length=12)
    return {
        "n_states": len(list(transducer.states())),
        "channel_C": max(0.0, float(channel_statistical_complexity(filled, input_process))),
        "driven_h": float(driven_h),
        "te_model": driven_transfer_entropy(filled, input_process, history),
    }


def _recurrent_tensors(transducer):
    """
    The channel tensors of `transducer` restricted to its single recurrent component.

    Reconstructed transducers can keep transient states, while the capacity bounds
    need a strongly connected channel. With one recurrent component the capacity does
    not depend on the initial state, so the transient states can be dropped.
    """
    from dataclasses import replace

    from scipy.sparse.csgraph import connected_components
    from sofic.generators.capacity import channel_tensors

    tensors = channel_tensors(transducer)
    n = tensors.n_states
    live = tensors.P > 0
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[np.nonzero(live)[0], tensors.F[live]] = True
    n_components, labels = connected_components(adjacency, directed=True, connection="strong")
    closed = [c for c in range(n_components) if not adjacency[labels == c][:, labels != c].any()]
    if len(closed) != 1:
        raise ValueError(
            f"the transducer has {len(closed)} recurrent components, so its capacity depends on the initial state"
        )
    keep = np.flatnonzero(labels == closed[0])
    if len(keep) == n:
        return tensors
    index = np.full(n, -1)
    index[keep] = np.arange(len(keep))
    F = tensors.F[keep]
    return replace(
        tensors,
        states=tuple(tensors.states[i] for i in keep),
        P=tensors.P[keep],
        F=np.where(F >= 0, index[F], -1),
        allowed=tensors.allowed[keep],
        initial=np.full(len(keep), 1.0 / len(keep)),
    )


def transducer_capacity(transducer, inputs=None, feedback=False, **options):
    """
    Bracket the capacity of an ε-transducer as a finite-state channel.

    An ε-transducer is a unifilar finite-state channel, so its capacity is defined
    without reference to a driving input. Without feedback the encoder sees only its
    own messages; the bracket is an optimized Markov-input rate
    :cite:`Vontobel2008` below and a Q-graph duality bound :cite:`Huleihel2021`
    above. With feedback the encoder also sees past outputs, hence the channel state;
    the bracket comes from Q-graph bounds :cite:`Sabag2017,Permuter2008`. Neither
    capacity is computable exactly in general :cite:`Grigorescu2024`. Feedback can
    only increase the capacity, and does not for a memoryless channel
    :cite:`Cover2006`.

    Every state is first made to read every input (:func:`filled_transducer`) and
    transient states are dropped; a transducer with several recurrent components
    raises ``ValueError``.

    Parameters
    ----------
    transducer : EpsilonTransducer
    inputs : iterable, None
        The input alphabet (default: the transducer's).
    feedback : bool
        Bracket the feedback capacity instead of the capacity without feedback.
    **options
        Passed to :func:`sofic.generators.capacity.feedback_capacity` or
        :func:`sofic.generators.capacity.capacity`.

    Returns
    -------
    sofic.generators.capacity.CapacityBounds
        ``lower <= C <= upper`` in bits per channel use.
    """
    from sofic.generators.capacity import capacity, feedback_capacity

    tensors = _recurrent_tensors(filled_transducer(transducer, inputs))
    return (feedback_capacity if feedback else capacity)(tensors, **options)


def _capacity_measures(transducer, inputs, s):
    out, errors = {}, []
    for name, feedback, options in (
        ("capacity", False, s.capacity_options),
        ("feedback_capacity", True, s.feedback_capacity_options),
    ):
        try:
            bounds = transducer_capacity(transducer, inputs, feedback, **options)
            out[f"{name}_low"], out[f"{name}_high"] = float(bounds.lower), float(bounds.upper)
        except Exception as err:  # noqa: BLE001 - recorded per edge, the layer continues
            errors.append(f"{name}: {type(err).__name__}: {err}")
    if errors:
        out["capacity_error"] = "; ".join(errors)
    return out


def _topology(machine):
    from sofic.inference import topology_key

    return topology_key(machine)


def _node_job(data, halves, p, max_history, s):
    trials = _node_trials(data, p, s.symbols)
    machine = _fit_machine(trials, max_history, s)
    from .network_model import count_forbidden

    out = {
        "machine": machine,
        **_node_measures(machine, s.excess_entropy),
        "n_symbols": sum(map(len, trials)),
        "n_forbidden": count_forbidden(machine, trials),
    }
    if s.stability:
        key, agree = _topology(machine), []
        for half in halves:
            try:
                agree.append(_topology(_fit_machine(_node_trials(half, p, s.symbols), max_history, s)) == key)
            except Exception:  # noqa: BLE001 - a failed refit counts as unstable
                agree.append(False)
        out["stability"] = float(np.mean(agree))
        if isinstance(max_history, int):
            grown = _fit_machine(trials, max_history + 1, s)
            out["unsynchronized"] = len(list(grown.states())) > out["n_states"]
    return out


def _transducer_job(data, halves, target, parents, inputs, halves_inputs, offsets, max_history, as_tuple, s):
    xs, ys = _aligned(data, target, parents, inputs, offsets, as_tuple)
    transducer = _fit_transducer(xs, ys, max_history, s)
    input_machine = _fit_machine(xs, max_history, s)
    out = {
        "machine": transducer,
        "input_machine": input_machine,
        **_channel_measures(transducer, input_machine, s.te_history),
        "n_symbols": sum(map(len, ys)),
    }
    if s.stability:
        agree = []
        for half, half_inputs in zip(halves, halves_inputs, strict=True):
            try:
                hx, hy = _aligned(half, target, parents, half_inputs, offsets, as_tuple)
                agree.append(len(list(_fit_transducer(hx, hy, max_history, s).states())) == out["n_states"])
            except Exception:  # noqa: BLE001 - a failed refit counts as unstable
                agree.append(False)
        out["stability"] = float(np.mean(agree))
    return out


def _as_data(data, names):
    if isinstance(data, DiscreteData):
        return data
    trials = as_trials(data)
    if not all(np.issubdtype(t.dtype, np.integer) for t in trials):
        raise ValueError(
            "the machine and channel layers need the DiscreteData the network was inferred on (or integer "
            "symbols); for float data use infer_multiplex(..., machines=True) or channels=True."
        )
    return DiscreteData.from_discrete(trials, names)


def infer_machines(data, network, settings=None, embeddings=None, map_fn=map):
    """
    Reconstruct the machine layer of an inferred network.

    Parameters
    ----------
    data : DiscreteData or integer array_like / Trials
        The symbols the network was inferred on.
    network : MultiplexNetwork
    settings : MachineSettings or dict, None
    embeddings : Embedding or list of Embedding, None
        The lag budgets used for the default history lengths; by default the
        network's.
    map_fn : callable
        ``map``-like function used over the reconstructions.

    Returns
    -------
    MachineLayer
    """
    _require_sofic()
    started = time.perf_counter()
    if settings is None:
        settings = MachineSettings()
    elif isinstance(settings, dict):
        settings = MachineSettings(**settings)
    s = settings
    names = network.names
    data = _as_data(data, names)
    P = data.n_processes
    if embeddings is None:
        embeddings = network.settings.get("embeddings")
    if embeddings is None:
        budgets = [3] * P
    else:
        budgets = [
            int((embeddings[p] if isinstance(embeddings, list | tuple) else embeddings).max_lag) for p in range(P)
        ]
    node_history = [s.max_history if s.max_history is not None else budgets[p] for p in range(P)]

    def transducer_history(t):
        if s.transducer_max_history is not None:
            return s.transducer_max_history
        h = node_history[t]
        return min(h, 2) if isinstance(h, int) else 2

    halves = list(data.split_time(0.5)) if s.stability else []
    if s.input_symbols == "coarse":
        inputs, offsets = _coarse_codes(data), data.present_offset
        halves_inputs = [_coarse_codes(h) for h in halves]
    else:
        inputs, offsets = _codes(data, "present")
        halves_inputs = [h.present for h in halves]

    ds_net = network.dataset
    delay = ds_net["delay"].values
    kind = ds_net["kind"].values
    jobs: list[tuple] = [("node", p, None) for p in range(P)]
    edge_kinds = ("parent", "shared_candidate") if s.include_shared_candidates else ("parent",)
    for src, tgt in sorted(network.edges):
        if kind[src, tgt] in edge_kinds:
            jobs.append(("edge", (src, tgt), [(src, int(delay[src, tgt]))]))
    parent_sets = {}
    for t, sk in sorted(network.skeleton.items()):
        parents = [(p, int(delay[p, t])) for p in sk.parents()]
        if not parents:
            continue
        parent_sets[t] = parents
        alphabet = 1
        for p, _ in parents:
            alphabet *= 2 if s.input_symbols == "coarse" else int(data.present_alphabet[p])
        if alphabet > s.max_joint_alphabet:
            jobs.append(("skip", t, f"joint input alphabet {alphabet} > max_joint_alphabet {s.max_joint_alphabet}"))
        else:
            jobs.append(("target", t, parents))

    def run(job):
        what, key, parents = job
        if what == "skip":
            return job, None, parents
        try:
            if what == "node":
                return job, _node_job(data, halves, key, node_history[key], s), None
            target = key[1] if what == "edge" else key
            result = _transducer_job(
                data,
                halves,
                target,
                parents,
                inputs,
                halves_inputs,
                offsets,
                transducer_history(target),
                what == "target",
                s,
            )
            if what == "edge" and s.capacity:
                inputs_seen = result["input_machine"].observation_alphabet
                result.update(_capacity_measures(result["machine"], inputs_seen, s))
            return job, result, None
        except Exception as err:  # noqa: BLE001 - recorded per machine, the layer continues
            return job, None, f"{type(err).__name__}: {err}"

    results = list(map_fn(run, jobs))

    layer = MachineLayer(dataset=xr.Dataset(), settings=asdict(s))
    nan_node = np.full(P, np.nan)
    node_vars = {
        k: nan_node.copy()
        for k in ("C_mu", "h_mu", "E", "crypticity", "n_states", "n_symbols", "n_forbidden", "stability", "max_history")
    }
    node_vars["unsynchronized"] = np.zeros(P, dtype=bool)
    target_vars = {
        f"target_{k}": nan_node.copy()
        for k in ("C", "h", "te", "te_gap", "n_states", "n_symbols", "stability", "omnibus_te")
    }
    target_inputs = np.full(P, "", dtype=object)
    edge_keys = {
        "n_states": "edge_n_states",
        "n_symbols": "edge_n_symbols",
        "stability": "edge_stability",
        **{k: k for k in ("channel_C", "driven_h", "te_model")},
        **(
            {f"{c}_{b}": f"{c}_{b}" for c in ("capacity", "feedback_capacity") for b in ("low", "high")}
            if s.capacity
            else {}
        ),
    }
    edge_vars = {name: np.full((P, P), np.nan) for name in [*edge_keys.values(), "te_gap"]}
    admitted = np.full((P, P), "", dtype=object)
    te = ds_net["weight"].sel(layer="te").values
    for (what, key, parents), result, reason in results:
        if what == "skip":
            layer.skipped[("target", names[key])] = reason
            continue
        if result is None:
            label = names[key] if what != "edge" else (names[key[0]], names[key[1]])
            layer.skipped[(what, label)] = reason
            continue
        if what == "node":
            layer.nodes[names[key]] = result["machine"]
            for k in node_vars:
                if k in result:
                    node_vars[k][key] = result[k]
            if isinstance(node_history[key], int):
                node_vars["max_history"][key] = node_history[key]
        elif what == "edge":
            src, tgt = key
            layer.edges[(names[src], names[tgt])] = result["machine"]
            for k, name in edge_keys.items():
                if k in result:
                    edge_vars[name][src, tgt] = result[k]
            edge_vars["te_gap"][src, tgt] = result["te_model"] - te[src, tgt]
            if "capacity_error" in result:
                layer.skipped[("capacity", (names[src], names[tgt]))] = result["capacity_error"]
            by = network.skeleton[tgt].admitted_by if tgt in network.skeleton else {}
            admitted[src, tgt] = ",".join(sorted({str(name) for v, name in by.items() if v[0] == src}))
        else:
            t = key
            layer.targets[names[t]] = result["machine"]
            layer.inputs[names[t]] = {
                "parents": [(names[p], d) for p, d in parents],
                "machine": result["input_machine"],
            }
            omnibus = float(network.skeleton[t].omnibus_te)
            values = {
                "C": result["channel_C"],
                "h": result["driven_h"],
                "te": result["te_model"],
                "te_gap": result["te_model"] - omnibus,
                "n_states": result["n_states"],
                "n_symbols": result["n_symbols"],
                "stability": result.get("stability", np.nan),
                "omnibus_te": omnibus,
            }
            for k, v in values.items():
                target_vars[f"target_{k}"][t] = v
            target_inputs[t] = ",".join(f"{names[p]}@{d}" for p, d in parents)
    coords = {"node": names, "source": names, "target": names}
    layer.dataset = xr.Dataset(
        {
            **{k: (("node",), v) for k, v in node_vars.items()},
            **{k: (("node",), v) for k, v in target_vars.items()},
            "target_inputs": (("node",), target_inputs.astype(str)),
            **{k: (("source", "target"), v) for k, v in edge_vars.items()},
            "admitted_by": (("source", "target"), admitted.astype(str)),
        },
        coords=coords,
        attrs={"description": "Machine layer: epsilon-machines and epsilon-transducers (bits)"},
    )
    import warnings

    if any(node_vars["unsynchronized"]):
        grown = [names[p] for p in range(P) if node_vars["unsynchronized"][p]]
        warnings.warn(
            f"node machine(s) {', '.join(grown)} gain states at max_history + 1: the history may be too "
            "short to synchronize, so their state counts are lower bounds, not the processes' ε-machines.",
            stacklevel=2,
        )
    forbidding = [names[p] for p in range(P) if node_vars["n_forbidden"][p] > 0]
    if forbidding:
        warnings.warn(
            f"node machine(s) {', '.join(forbidding)} forbid symbols observed in the data they were fitted "
            "to: max_history is too short to resolve the process (see n_forbidden).",
            stacklevel=2,
        )
    from .network_model import NetworkModel

    try:
        layer.model = NetworkModel.from_layer(layer)
    except ValueError as err:
        layer.skipped[("model", "network")] = str(err)
    if layer.model is not None and s.exact_flows:
        try:
            reason = layer.model.add_exact_flows(layer.dataset, network, s.te_history)
        except Exception as err:  # noqa: BLE001 - the flows are optional diagnostics
            reason = f"{type(err).__name__}: {err}"
        if reason:
            layer.skipped[("model", "exact_flows")] = reason
    layer.seconds = time.perf_counter() - started
    return layer
