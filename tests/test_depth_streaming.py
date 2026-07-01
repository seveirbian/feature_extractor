import numpy as np
import pytest

from feature_extractor.extractors.depth_streaming import plan_segment_keyframes


def test_keyframes_first_segment_starts_at_row0_and_forces_last():
    # no carried keyframe; interval 2; source indices 0..4
    rows = plan_segment_keyframes([0, 1, 2, 3, 4], last_kf_source_idx=None, keyframe_interval=2)
    assert rows[0] == 0            # first segment must anchor its own row 0
    assert rows[-1] == 4           # last row always forced (bounds interpolation)
    assert 2 in rows               # 2 - 0 >= interval


def test_keyframes_respect_global_spacing_with_carried_anchor():
    # previous keyframe was at global source idx 10; this segment covers 11..15
    rows = plan_segment_keyframes([11, 12, 13, 14, 15], last_kf_source_idx=10, keyframe_interval=3)
    # 11-10=1 <interval, 12-10=2 <interval, 13-10=3 -> keyframe (row 2)
    assert 0 not in rows           # row 0 (src 11) interpolated from carried anchor, not a keyframe
    assert 2 in rows
    assert rows[-1] == 4           # last row forced


def test_keyframes_carried_anchor_dominates_still_forces_last():
    # every segment frame within interval of the carried keyframe -> only last row forced
    rows = plan_segment_keyframes([4, 5, 6, 7], last_kf_source_idx=6, keyframe_interval=2)
    assert rows == [3]


def test_keyframes_single_frame_segment():
    assert plan_segment_keyframes([7], last_kf_source_idx=None, keyframe_interval=30) == [0]


def test_keyframes_empty():
    assert plan_segment_keyframes([], last_kf_source_idx=None, keyframe_interval=30) == []


from feature_extractor.extractors.depth_streaming import interpolate_segment_params


def test_interpolate_linear_between_two_anchors():
    # anchors: (source_idx, scale, shift)
    anchors = [(0, 2.0, 1.0), (4, 6.0, 5.0)]
    scales, shifts = interpolate_segment_params([0, 1, 2, 3, 4], anchors)
    np.testing.assert_allclose(scales, [2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(shifts, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_interpolate_uses_carried_anchor_before_segment():
    # carried anchor at src 10 (before segment), in-segment keyframe at src 14
    anchors = [(10, 1.0, 0.0), (14, 5.0, 8.0)]
    scales, shifts = interpolate_segment_params([11, 12, 13, 14], anchors)
    np.testing.assert_allclose(scales, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(shifts, [2.0, 4.0, 6.0, 8.0])


def test_interpolate_constant_single_anchor():
    scales, shifts = interpolate_segment_params([3, 4], [(3, 2.5, 1.5)])
    np.testing.assert_allclose(scales, [2.5, 2.5])
    np.testing.assert_allclose(shifts, [1.5, 1.5])


import torch

from feature_extractor.storage import FeatureStore
from feature_extractor.validation.synthetic import make_ramp_video
from feature_extractor.extractors.depth import DepthExtractor


class _StubDepth:
    """Drives extract_video_depth_streaming with fake VDA + Depth Pro models."""

    extract_video_depth_streaming = DepthExtractor.extract_video_depth_streaming
    _fit_depth_affine = staticmethod(DepthExtractor._fit_depth_affine)
    _to_inverse_depth = DepthExtractor._to_inverse_depth

    def __init__(self):
        self.vda_metric = False
        self.keyframe_interval = 2
        self.z_min = 0.1
        self.z_max = 100.0
        self.vda_input_size = 224
        self.device = torch.device("cpu")
        self.model = self  # so self.model.infer_video_depth resolves here

    def infer_video_depth(self, frames, target_fps=30, input_size=224, device="cpu", fp32=True):
        # deterministic raw depth: each frame -> constant plane = mean/255 + 1
        b = frames.shape[0]
        raw = np.stack(
            [np.full(frames.shape[1:3], float(frames[j].mean()) / 255.0 + 1.0, np.float32)
             for j in range(b)]
        )
        return raw, target_fps

    def _extract_depth_pro(self, frame):
        # metric = 3 * raw(frame); fit should recover it, giving valid metric
        val = float(frame.mean()) / 255.0 + 1.0
        return np.full(frame.shape[:2], val * 3.0, np.float32)


def test_depth_streaming_writes_every_frame(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=10, width=64, height=48, step=20)
    store = FeatureStore(str(tmp_path / "store"))

    _StubDepth().extract_video_depth_streaming(
        video_path=path,
        frame_indices=list(range(10)),
        store=store,
        video_id="clip",
        block_size=4,
        overlap=2,
    )

    depth = store.read_depth("clip")
    assert depth.shape == (10, 48, 64, 1)
    assert np.isfinite(depth).all() and depth.min() >= 0.0 and depth.max() <= 1.0
    np.testing.assert_array_equal(store.read_frame_indices("clip", "depth"), list(range(10)))
    assert store.is_branch_complete("clip", "depth") is True


def test_depth_streaming_boundary_matches_single_segment(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=8, width=64, height=48, step=20)

    store_a = FeatureStore(str(tmp_path / "a"))
    _StubDepth().extract_video_depth_streaming(
        path, list(range(8)), store_a, "clip", block_size=8, overlap=0)  # single segment
    single = store_a.read_depth("clip")

    store_b = FeatureStore(str(tmp_path / "b"))
    _StubDepth().extract_video_depth_streaming(
        path, list(range(8)), store_b, "clip", block_size=5, overlap=3)  # multi segment
    stitched = store_b.read_depth("clip")

    assert single.shape == stitched.shape == (8, 48, 64, 1)
    # boundary frames (around row 5) should agree closely with the single-segment result
    np.testing.assert_allclose(stitched[4:6], single[4:6], atol=2e-3)
