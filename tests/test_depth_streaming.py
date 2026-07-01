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
