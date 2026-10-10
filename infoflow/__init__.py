"""
infoflow: multiplex information-flow network inference from time series.

Networks are inferred from discrete (or rank-discretized) multivariate time
series. Each directed edge's transfer entropy is decomposed into intrinsic,
synergistic, and shared flow :cite:`James2016`, and each node carries its active
information storage. Estimation, testing, and interpretation build on the
estimators and null generators of :mod:`dit.inference`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import benchmarks, datasets, preprocess
from .auto import AutoReport, AutoResult, infer
from .checkpoint import Checkpoint
from .compare import NetworkComparison, compare_networks
from .data import DiscreteData, realizations
from .embedding import Embedding
from .layers import LAYERS, layer_statistics
from .measures import edge_flows, intrinsic_flow
from .network import MultiplexNetwork, infer_multiplex
from .parallel import dask_map, thread_map
from .preprocess import preprocess as run_preprocess
from .selection import SkeletonSettings, infer_skeleton, select_parents
from .stats import (
    SurrogateTest,
    benjamini_hochberg,
    bootstrap_ci,
    conditional_mutual_information_test,
    conditional_mutual_information_test_knn,
)
from .te import transfer_entropy, transfer_entropy_ci, transfer_entropy_test

try:
    __version__ = _version("infoflow")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
