"""Feature extractors: DINO, Depth, Pose."""

from .dino import DINOExtractor
from .depth import DepthExtractor
from .pose import PoseExtractor

__all__ = ["DINOExtractor", "DepthExtractor", "PoseExtractor"]
