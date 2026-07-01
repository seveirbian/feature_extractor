"""Pure sim(3) helpers for streaming VGGT pose stitching.

No torch/VGGT imports: estimate and apply the similarity transform that maps one
window's local camera trajectory onto the running global trajectory, using the
overlap frames' camera centers (Umeyama) with a rotation-only fallback for
degenerate (near-stationary) motion.
"""

from __future__ import annotations

import numpy as np


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``(s, R, t)`` with ``dst ≈ s * (src @ R.T) + t``.

    ``src``/``dst`` are ``(N, 3)`` point sets (camera centers). Uses the Umeyama
    (1991) closed form. Assumes ``src`` has non-degenerate spread; callers guard
    degeneracy via ``robust_similarity``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    sigma = (xd.T @ xs) / n
    u, d, vt = np.linalg.svd(sigma)
    s_corr = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_corr[-1, -1] = -1.0
    R = u @ s_corr @ vt
    var_s = (xs ** 2).sum() / n
    scale = float(np.trace(np.diag(d) @ s_corr) / var_s)
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def apply_similarity_to_poses(
    poses_c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Apply world similarity ``(s, R, t)`` to camera-to-world poses ``(N,4,4)``.

    ``center' = s*R@center + t``; ``rotation' = R @ rotation`` (scale does not
    affect orientation). Returns a new ``(N,4,4)`` array.
    """
    poses_c2w = np.asarray(poses_c2w, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    out = np.tile(np.eye(4), (poses_c2w.shape[0], 1, 1))
    out[:, :3, :3] = R[None] @ poses_c2w[:, :3, :3]
    centers = poses_c2w[:, :3, 3]
    out[:, :3, 3] = s * (centers @ R.T) + t[None]
    return out


def _orthonormalize(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def robust_similarity(
    src_poses_c2w: np.ndarray, dst_poses_c2w: np.ndarray, min_spread: float = 1e-4
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate ``(s, R, t)`` mapping ``src`` camera poses onto ``dst``.

    Uses Umeyama on camera centers when they have enough spread; otherwise (near
    stationary) falls back to a rotation-only fit from the pose orientations with
    unit scale.
    """
    src_poses_c2w = np.asarray(src_poses_c2w, dtype=np.float64)
    dst_poses_c2w = np.asarray(dst_poses_c2w, dtype=np.float64)
    src_c = src_poses_c2w[:, :3, 3]
    dst_c = dst_poses_c2w[:, :3, 3]
    spread = float(np.sqrt(((src_c - src_c.mean(0)) ** 2).sum(axis=1)).mean())

    if spread >= min_spread and src_c.shape[0] >= 3:
        s, R, t = umeyama_similarity(src_c, dst_c)
        if np.isfinite(s) and s > 0 and np.isfinite(R).all() and np.isfinite(t).all():
            return s, R, t

    # Rotation-only fallback: R aligns src orientations to dst orientations.
    m = np.zeros((3, 3))
    for i in range(src_poses_c2w.shape[0]):
        m += dst_poses_c2w[i, :3, :3] @ src_poses_c2w[i, :3, :3].T
    R = _orthonormalize(m)
    t = dst_c.mean(0) - (R @ src_c.mean(0))
    return 1.0, R, t
