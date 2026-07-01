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
