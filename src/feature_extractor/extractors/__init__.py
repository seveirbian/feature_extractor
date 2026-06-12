"""Feature extractors: DINO, Depth, Pose."""

from .dino import DINOExtractor
from .depth import DepthExtractor

__all__ = ["DINOExtractor", "DepthExtractor"]
