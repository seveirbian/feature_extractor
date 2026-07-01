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
