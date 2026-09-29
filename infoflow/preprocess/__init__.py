"""
Preprocessing: discretization, delay and embedding selection, and diagnostics.
"""

from .delays import auto_mi_delay, autocorrelation_delay, curvature_delay, delay_candidates
from .discretizers import (
    Discretizer,
    EqualFrequency,
    EqualWidth,
    Identity,
    Ordinal,
    Threshold,
    discretize,
    reencode,
)
from .pipeline import NodeReport, PreprocessingReport, PreprocessResult, preprocess
from .scaling import ScalingRegion, block_entropy_scaling, scaling_region
from .screening import ScreenResult, screen, split_missing
