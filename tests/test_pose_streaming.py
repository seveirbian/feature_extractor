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


from feature_extractor.extractors.pose_streaming import apply_similarity_to_poses


def _c2w(R, C):
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = C
    return G


def test_apply_similarity_transforms_center_and_rotation():
    R_pose = _rot([0, 0, 1], 0.4)
    G = _c2w(R_pose, np.array([2.0, 0.0, 1.0]))[None]  # (1,4,4)

    R_s = _rot([0, 1, 0], 0.9)
    s, t = 3.0, np.array([1.0, 1.0, 1.0])
    out = apply_similarity_to_poses(G, s, R_s, t)

    np.testing.assert_allclose(out[0, :3, 3], s * (R_s @ G[0, :3, 3]) + t, atol=1e-9)
    np.testing.assert_allclose(out[0, :3, :3], R_s @ G[0, :3, :3], atol=1e-9)
    assert out[0, 3, 3] == 1.0


def test_apply_identity_similarity_noop():
    G = _c2w(_rot([1, 0, 0], 0.3), np.array([1.0, 2.0, 3.0]))[None]
    out = apply_similarity_to_poses(G, 1.0, np.eye(3), np.zeros(3))
    np.testing.assert_allclose(out, G, atol=1e-9)


from feature_extractor.extractors.pose_streaming import robust_similarity


def test_robust_similarity_matches_umeyama_for_normal_motion():
    src = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    R_true = _rot([0.3, 0.2, 1.0], 0.5)
    s_true, t_true = 1.0, np.array([2.0, 0.0, -1.0])
    dst = s_true * (src @ R_true.T) + t_true
    src_poses = np.stack([_c2w(np.eye(3), c) for c in src])
    dst_poses = np.stack([_c2w(R_true, c) for c in dst])

    s, R, t = robust_similarity(src_poses, dst_poses)
    np.testing.assert_allclose(s * (src @ R.T) + t, dst, atol=1e-6)


def test_robust_similarity_stationary_falls_back_without_crash():
    # all centers identical (camera stationary): Umeyama on centers is degenerate
    C = np.array([1.0, 2.0, 3.0])
    R_rel = _rot([0, 0, 1], 0.5)
    src_poses = np.stack([_c2w(_rot([0, 0, 1], a), C) for a in (0.0, 0.1, 0.2)])
    dst_poses = np.stack([_c2w(R_rel @ p[:3, :3], C) for p in src_poses])

    s, R, t = robust_similarity(src_poses, dst_poses)
    assert np.isfinite(s) and np.isfinite(R).all() and np.isfinite(t).all()
    assert abs(s - 1.0) < 1e-6                       # fallback keeps unit scale
    np.testing.assert_allclose(R, R_rel, atol=1e-6)  # recovered from orientations


from feature_extractor.extractors.pose import PoseExtractor


def test_relative_se3_row0_is_identity_pose():
    # two world-to-camera extrinsics; relative to row 0 must give identity at row 0
    E = np.stack([np.eye(4), _c2w(_rot([0, 0, 1], 0.5), np.array([1.0, 0, 0]))])
    rel = PoseExtractor._relative_se3_from_extrinsics(E)
    assert rel.shape == (2, 9)
    # identity pose: translation 0, rot6d = first two columns of I = [1,0,0, 0,1,0]
    np.testing.assert_allclose(rel[0], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)


def test_relative_se3_explicit_reference():
    E = np.stack([np.eye(4), _c2w(_rot([0, 0, 1], 0.5), np.array([1.0, 0, 0]))])
    # reference = row 1 -> row 1 becomes identity
    rel = PoseExtractor._relative_se3_from_extrinsics(E, ref_extrinsic=E[1])
    np.testing.assert_allclose(rel[1], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
