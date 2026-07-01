import numpy as np

from feature_extractor.cli import should_stream, branches_to_resume_skip
from feature_extractor.storage import FeatureStore


def test_should_stream_threshold():
    assert should_stream(n_frames=2500, threshold=2000) is True
    assert should_stream(n_frames=120, threshold=2000) is False


def test_resume_skips_only_completed(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "clip"
    store.write_dino_chunk(vid, np.zeros((1, 2, 2), np.float32), np.array([0]), reset=True)
    # dino written but NOT marked complete -> must NOT be skipped
    assert branches_to_resume_skip(store, vid, ["dino"]) is False
    store.mark_branch_complete(vid, "dino")
    assert branches_to_resume_skip(store, vid, ["dino"]) is True


import inspect
from feature_extractor.cli import extract_single_video


def test_extract_single_video_accepts_depth_overlap():
    sig = inspect.signature(extract_single_video)
    assert "depth_overlap" in sig.parameters
    assert sig.parameters["depth_overlap"].default == 96
