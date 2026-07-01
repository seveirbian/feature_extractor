import numpy as np
import pytest

from feature_extractor.extractors.pose_streaming import umeyama_similarity


def _rot(axis, ang):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))
    R_true = _rot([0.2, 1.0, -0.3], 0.7)
    s_true, t_true = 2.5, np.array([1.0, -2.0, 3.0])
    dst = s_true * (src @ R_true.T) + t_true

    s, R, t = umeyama_similarity(src, dst)
    assert abs(s - s_true) < 1e-6
    np.testing.assert_allclose(R, R_true, atol=1e-6)
    np.testing.assert_allclose(t, t_true, atol=1e-6)
    # transform maps src onto dst
    np.testing.assert_allclose(s * (src @ R.T) + t, dst, atol=1e-6)


def test_umeyama_identity_when_equal():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    s, R, t = umeyama_similarity(pts, pts)
    assert abs(s - 1.0) < 1e-9
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(t, np.zeros(3), atol=1e-9)
