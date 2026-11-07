"""Candidate-wise trajectory geometry for Trail-MOTIP V3."""

from .geometry_features import (
    DEFAULT_GEOMETRY_LAGS,
    GEOMETRY_FEATURE_DIM,
    build_g4_features,
    build_pairwise_g4_features,
)
from .geometry_branch import GeometryCompatibilityBranch

__all__ = [
    "DEFAULT_GEOMETRY_LAGS",
    "GEOMETRY_FEATURE_DIM",
    "build_g4_features",
    "build_pairwise_g4_features",
    "GeometryCompatibilityBranch",
]
