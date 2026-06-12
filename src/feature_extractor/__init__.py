"""feature_extractor: standalone DINO/Depth/Pose feature extraction."""

from feature_extractor.extractors import DINOExtractor, DepthExtractor, PoseExtractor
from feature_extractor.storage import FeatureStore

__version__ = "0.1.0"

__all__ = ["DINOExtractor", "DepthExtractor", "PoseExtractor", "FeatureStore"]
