"""
Preprocessing: discretization, delay and embedding selection, and diagnostics.
"""

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
