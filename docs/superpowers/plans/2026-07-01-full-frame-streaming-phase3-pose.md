# Full-Frame Streaming Extraction — Phase 3 (Pose) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Pose (VGGT) branch extract every frame within GPU limits by running VGGT on overlapping windows and stitching each window into one global, frame-0-relative trajectory with a sim(3) (rotation+translation+scale) transform estimated from the overlap frames.

**Architecture:** Pure, model-free helpers (`umeyama_similarity`, `apply_similarity_to_poses`, `robust_similarity`) live in a new `pose_streaming.py` and do the sim(3) math; a pure `_relative_se3_from_extrinsics` produces the 9D output. `PoseExtractor.extract_video_pose_streaming` orchestrates windowed VGGT inference (behind a `_window_extrinsics` seam) → per-window sim(3) alignment to the running global trajectory → incremental `write_pose_chunk`. Tests stub `_window_extrinsics` so the entire pipeline — including a trajectory-recovery regression — runs without VGGT weights.

**Tech Stack:** Python, NumPy, h5py, pytest, VGGT (stubbed in tests), `feature_extractor.chunking`, synthetic videos via `make_ramp_video`.

**Spec:** `docs/superpowers/specs/2026-06-30-full-frame-streaming-extraction-design.md` (Component 4).

**Depends on:** Phases 1–2 (merged) — `iter_frame_blocks`, `store.write_pose_chunk`, `store.mark_branch_complete`, CLI `stream` plumbing.

---

## Background (read before starting)

Existing pose output ([pose.py](../../../src/feature_extractor/extractors/pose.py)):
- `_infer_sequence_pose_enc(images) -> (pose_enc, image_hw)` runs VGGT jointly on a frame list.
- `_pose_enc_sequence_to_relative_se3(pose_enc, image_hw)` (pose.py:228) converts pose_enc → world-to-camera extrinsics `(T,3,4)` via `pose_encoding_to_extri_intri`, makes them homogeneous `(T,4,4)`, then `rel = E_i @ inv(E_0)` (frame-0-relative), emitting 9D `pose_to_se3(translation, rot6d)` per frame.
- Module helpers `rotation_to_6d` (pose.py:29) and `pose_to_se3` (pose.py:38) build the 9D vector.
- Translation is in VGGT's per-inference **normalized scene scale** (not metric). Each window has its own world frame and scale ⟹ stitching is **sim(3)**.

Key convention: `E` = world-to-camera. Camera-to-world `G = inv(E)`; camera center `C = G[:3,3]`. A world similarity `(s,R,t)` maps a camera-to-world pose to `center' = s·R·C + t`, `rotation' = R·G[:3,:3]` (scale does not rotate orientation).

Single-window streaming must reduce to the existing output: with one window the global frame *is* the window frame, so `rel = E_i @ inv(E_0)` is unchanged.

---

## File Structure

- **Create** `src/feature_extractor/extractors/pose_streaming.py` — pure helpers `umeyama_similarity`, `apply_similarity_to_poses`, `robust_similarity`. No torch/VGGT imports.
- **Modify** `src/feature_extractor/extractors/pose.py` — refactor a pure `_relative_se3_from_extrinsics` and a `_pose_enc_to_extrinsics` / `_window_extrinsics` seam out of `_pose_enc_sequence_to_relative_se3`; add `extract_video_pose_streaming`.
- **Modify** `src/feature_extractor/cli.py` — route the pose branch to streaming when `stream=True`; thread `pose_window`/`pose_overlap`.
- **Create** `tests/test_pose_streaming.py` — helper unit tests + stubbed orchestration + trajectory-recovery regression.
- **Modify** `tests/test_cli_streaming.py` — assert pose plumbing.
- **Modify** `README.md` — pose now streams.

---

## Task 1: Umeyama similarity (`umeyama_similarity`)

**Files:**
- Create: `src/feature_extractor/extractors/pose_streaming.py`
- Test: `tests/test_pose_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_streaming.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pose_streaming.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'feature_extractor.extractors.pose_streaming'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/extractors/pose_streaming.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_streaming.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/pose_streaming.py tests/test_pose_streaming.py
git commit -m "feat(pose): umeyama_similarity for sim(3) alignment"
```

---

## Task 2: Apply similarity to camera poses (`apply_similarity_to_poses`)

**Files:**
- Modify: `src/feature_extractor/extractors/pose_streaming.py`
- Test: `tests/test_pose_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_streaming.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pose_streaming.py -k apply -q`
Expected: FAIL with `ImportError: cannot import name 'apply_similarity_to_poses'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/extractors/pose_streaming.py  (append)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_streaming.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/pose_streaming.py tests/test_pose_streaming.py
git commit -m "feat(pose): apply_similarity_to_poses"
```

---

## Task 3: Robust similarity with degenerate fallback (`robust_similarity`)

**Files:**
- Modify: `src/feature_extractor/extractors/pose_streaming.py`
- Test: `tests/test_pose_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_streaming.py  (append)
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
    assert abs(s - 1.0) < 1e-6                      # fallback keeps unit scale
    np.testing.assert_allclose(R, R_rel, atol=1e-6)  # recovered from orientations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pose_streaming.py -k robust -q`
Expected: FAIL with `ImportError: cannot import name 'robust_similarity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/extractors/pose_streaming.py  (append)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_streaming.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/pose_streaming.py tests/test_pose_streaming.py
git commit -m "feat(pose): robust_similarity with stationary-motion fallback"
```

---

## Task 4: Pure relative-SE3 conversion refactor (`_relative_se3_from_extrinsics`)

**Files:**
- Modify: `src/feature_extractor/extractors/pose.py` (`_pose_enc_sequence_to_relative_se3`, pose.py:228-259)
- Test: `tests/test_pose_streaming.py`

Refactors the frame-0-relative math into a pure, testable method that also accepts an explicit reference extrinsic (needed so every streaming window is relative to the *video's* frame 0). Behavior of the existing path is unchanged (default ref = row 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_streaming.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pose_streaming.py -k relative_se3 -q`
Expected: FAIL with `AttributeError: type object 'PoseExtractor' has no attribute '_relative_se3_from_extrinsics'`

- [ ] **Step 3: Write minimal implementation**

Add the pure static method to `PoseExtractor` and refactor the existing converter to use it. Replace the body of `_pose_enc_sequence_to_relative_se3` (pose.py:228-259) so it extracts extrinsics then delegates:

```python
    @staticmethod
    def _relative_se3_from_extrinsics(
        extrinsics_h: np.ndarray, ref_extrinsic: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Frame-relative 9D poses from world-to-camera extrinsics ``(T,4,4)``.

        ``rel = E_i @ inv(ref)``; ``ref`` defaults to ``extrinsics_h[0]``.
        """
        extrinsics_h = np.asarray(extrinsics_h, dtype=np.float32)
        ref = extrinsics_h[0] if ref_extrinsic is None else np.asarray(ref_extrinsic, dtype=np.float32)
        ref_inv = np.linalg.inv(ref)
        rel_extrinsics = extrinsics_h @ ref_inv[None, :, :]
        rel_se3 = [
            pose_to_se3(extr[:3, 3].astype(np.float32), extr[:3, :3].astype(np.float32))
            for extr in rel_extrinsics
        ]
        return np.stack(rel_se3, axis=0).astype(np.float32)

    def _pose_enc_to_extrinsics(self, pose_enc: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
        """VGGT pose encodings -> homogeneous world-to-camera extrinsics ``(T,4,4)``."""
        import sys

        vggt_repo, _checkpoint_path = _resolve_local_vggt_paths(self.assets_root)
        if str(vggt_repo) not in sys.path:
            sys.path.insert(0, str(vggt_repo))
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pose_tensor = torch.from_numpy(np.asarray(pose_enc, dtype=np.float32)).to(self.device).unsqueeze(0)
        extrinsics, _intrinsics = pose_encoding_to_extri_intri(
            pose_tensor, image_size_hw=image_hw, build_intrinsics=False
        )
        extrinsics_np = extrinsics.squeeze(0).float().cpu().numpy()
        extrinsics_h = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (extrinsics_np.shape[0], 1, 1))
        extrinsics_h[:, :3, :4] = extrinsics_np
        return extrinsics_h

    def _pose_enc_sequence_to_relative_se3(
        self, pose_enc: np.ndarray, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Convert VGGT sequence pose encodings into frame-0-relative extrinsics."""
        return self._relative_se3_from_extrinsics(self._pose_enc_to_extrinsics(pose_enc, image_hw))

    def _window_extrinsics(self, frames) -> np.ndarray:
        """Run VGGT on a window of frames and return world-to-camera ``(T,4,4)``."""
        pose_enc, image_hw = self._infer_sequence_pose_enc(list(frames))
        return self._pose_enc_to_extrinsics(pose_enc, image_hw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_streaming.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/pose.py tests/test_pose_streaming.py
git commit -m "refactor(pose): pure _relative_se3_from_extrinsics + _window_extrinsics seam"
```

---

## Task 5: Streaming pose orchestration (`extract_video_pose_streaming`)

**Files:**
- Modify: `src/feature_extractor/extractors/pose.py` (add method after `extract_video`, ~pose.py:539)
- Test: `tests/test_pose_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_streaming.py  (append)
from feature_extractor.storage import FeatureStore
from feature_extractor.validation.synthetic import make_ramp_video


def _ramp_index(frame):
    return int(round(float(frame.mean()) / 20.0))


def _gt_c2w(i):
    """Deterministic ground-truth camera-to-world for frame i (helix + yaw)."""
    ang = 0.15 * i
    R = _rot([0, 0, 1], ang)
    C = np.array([np.cos(ang), np.sin(ang), 0.1 * i])
    return _c2w(R, C)


class _StubPose:
    extract_video_pose_streaming = PoseExtractor.extract_video_pose_streaming
    _relative_se3_from_extrinsics = staticmethod(PoseExtractor._relative_se3_from_extrinsics)

    def _window_extrinsics(self, frames):
        # recover frame indices from ramp content, apply a per-window similarity
        idx = [_ramp_index(frames[j]) for j in range(len(frames))]
        i0 = idx[0]
        if i0 == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)        # window 0 defines global frame
        else:
            s, R, t = 1.0, _rot([0, 1, 0], 0.3 + 0.01 * i0), np.array([0.5 * i0, -0.2, 1.0])
        c2w_local = apply_similarity_to_poses(np.stack([_gt_c2w(i) for i in idx]), s, R, t)
        return np.linalg.inv(c2w_local).astype(np.float32)  # world-to-camera


def test_pose_streaming_writes_every_frame(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=10, width=64, height=48, step=20)
    store = FeatureStore(str(tmp_path / "store"))

    _StubPose().extract_video_pose_streaming(
        path, list(range(10)), store, "clip", window=4, overlap=2)

    pose = store.read_pose("clip")
    assert pose.shape == (10, 9)
    np.testing.assert_array_equal(store.read_frame_indices("clip", "pose"), list(range(10)))
    assert store.is_branch_complete("clip", "pose") is True
    # frame 0 is identity pose
    np.testing.assert_allclose(pose[0], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pose_streaming.py -k pose_streaming_writes -q`
Expected: FAIL with `AttributeError: type object 'PoseExtractor' has no attribute 'extract_video_pose_streaming'`

- [ ] **Step 3: Write minimal implementation**

Add to `PoseExtractor` in `src/feature_extractor/extractors/pose.py` (after `extract_video`):

```python
    def extract_video_pose_streaming(
        self,
        video_path: str,
        frame_indices: list[int],
        store,
        video_id: str,
        window: int = 600,
        overlap: int = 120,
    ) -> None:
        """Stream frame-0-relative pose to ``store`` over overlapping VGGT windows.

        Each window's world-to-camera extrinsics are aligned into one global
        camera-to-world trajectory (anchored at video frame 0) via a sim(3)
        transform estimated from the overlap frames, then the non-overlap tail is
        written incrementally as frame-0-relative 9D pose.
        """
        from ..chunking import iter_frame_blocks
        from .pose_streaming import apply_similarity_to_poses, robust_similarity

        prev_global_overlap = None   # (overlap,4,4) global c2w of previous window tail
        frame0_w2c = None            # global world-to-camera of the video's frame 0
        first = True

        pbar = tqdm(total=len(frame_indices),
                    desc=f"Pose stream [{Path(video_path).name}]", unit="f")
        for block_idx, frames, write_offset in iter_frame_blocks(
            video_path, frame_indices, window, overlap
        ):
            w2c = np.asarray(self._window_extrinsics(frames), dtype=np.float64)  # (b,4,4)
            c2w_local = np.linalg.inv(w2c)

            if prev_global_overlap is None:
                c2w_global = c2w_local                       # window 0 defines global frame
            else:
                s, R, t = robust_similarity(c2w_local[:write_offset], prev_global_overlap)
                c2w_global = apply_similarity_to_poses(c2w_local, s, R, t)

            if frame0_w2c is None:
                frame0_w2c = np.linalg.inv(c2w_global[0])

            w2c_global = np.linalg.inv(c2w_global)
            rel = self._relative_se3_from_extrinsics(
                w2c_global[write_offset:].astype(np.float32), ref_extrinsic=frame0_w2c
            )
            store.write_pose_chunk(video_id, rel, block_idx[write_offset:], reset=first)
            first = False
            if overlap > 0:
                prev_global_overlap = c2w_global[-overlap:].copy()
            pbar.update(len(block_idx) - write_offset)

        pbar.close()
        store.mark_branch_complete(video_id, "pose")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_streaming.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/pose.py tests/test_pose_streaming.py
git commit -m "feat(pose): extract_video_pose_streaming windowed sim(3) stitching"
```

---

## Task 6: Trajectory-recovery regression

**Files:**
- Test: `tests/test_pose_streaming.py`

Proves the sim(3) stitching reconstructs a globally consistent trajectory: multi-window streaming must match the ground-truth frame-0-relative poses (the stub applies a per-window similarity that stitching must undo).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_streaming.py  (append)
def test_pose_streaming_recovers_global_trajectory(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    n = 12
    make_ramp_video(path, codec="libx264", n_frames=n, width=64, height=48, step=20)
    store = FeatureStore(str(tmp_path / "store"))

    _StubPose().extract_video_pose_streaming(
        path, list(range(n)), store, "clip", window=6, overlap=3)  # multi-window
    got = store.read_pose("clip")

    # ground-truth frame-0-relative poses from the same GT trajectory
    E_gt = np.linalg.inv(np.stack([_gt_c2w(i) for i in range(n)]))  # world-to-camera
    want = PoseExtractor._relative_se3_from_extrinsics(E_gt.astype(np.float32))

    assert got.shape == want.shape == (n, 9)
    np.testing.assert_allclose(got, want, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_streaming.py -k recovers_global -q`
Expected: PASS. If it FAILS, the stitching/alignment wiring is wrong — fix Task 5 (do not loosen the tolerance blindly).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pose_streaming.py
git commit -m "test(pose): trajectory-recovery regression for sim(3) stitching"
```

---

## Task 7: CLI routing, README, full regression

**Files:**
- Modify: `src/feature_extractor/cli.py`
- Modify: `tests/test_cli_streaming.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_streaming.py  (append)
def test_extract_single_video_accepts_pose_params():
    sig = inspect.signature(extract_single_video)
    assert sig.parameters["pose_window"].default == 600
    assert sig.parameters["pose_overlap"].default == 120
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_streaming.py -k pose_params -q`
Expected: FAIL with `KeyError: 'pose_window'`

- [ ] **Step 3: Write minimal implementation**

In `src/feature_extractor/cli.py`, add `pose_window: int = 600` and `pose_overlap: int = 120` to `extract_single_video`'s signature (after `depth_overlap: int = 96,`) and replace the pose branch:

```python
        # Extract Pose
        if extractor_pose is not None:
            if stream:
                extractor_pose.extract_video_pose_streaming(
                    video_path, frame_indices, store, video_id,
                    window=pose_window, overlap=pose_overlap,
                )
            else:
                pose_se3 = extractor_pose.extract_video(video_path, frame_indices=frame_indices)
                store.write_pose(video_id, pose_se3, frame_indices=frame_indices)
                store.mark_branch_complete(video_id, "pose")
```

Then in the `main` processing loop pass the args in the `extract_single_video(...)` call (after `depth_overlap=args.depth_overlap,`):

```python
            pose_window=args.pose_window,
            pose_overlap=args.pose_overlap,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_streaming.py -q`
Expected: PASS.

- [ ] **Step 5: Update README and run full suite**

In `README.md` §6.1, replace the warning that says only `dino`/`depth` stream with:

```
**`dino` / `depth` / `pose` 三个分支均支持全帧流式提取。** Pose 为 VGGT 滑窗 +
sim(3) 拼接:每个窗口(`--pose_window`,默认 600,受显存约束)推理后,用重叠帧
(`--pose_overlap`,默认 120)估计相似变换拼接成全局轨迹。窗口越大漂移越小。
```

And update the §10 limitation bullet: remove "Pose 待 Phase 3", noting pose streaming has slow cross-window drift on very long videos (larger `--pose_overlap` reduces it).

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/feature_extractor/cli.py tests/test_cli_streaming.py README.md
git commit -m "feat(cli): route pose branch to streaming; docs for pose full-frame"
```

---

## Self-Review notes

- **Spec coverage (Component 4):** windowed VGGT via `iter_frame_blocks(window, overlap)` + `_window_extrinsics` (Task 5); global frame anchored at video frame 0 (Task 5, `frame0_w2c`); sim(3) via Umeyama on overlap camera centers (Tasks 1,3,5); apply to non-overlap poses `center'=sRC+t`, `rot'=R·rot` (Task 2,5); frame-0-relative 9D output via pure `_relative_se3_from_extrinsics` (Task 4); incremental `write_pose_chunk` (Task 5); degenerate-motion rotation-only fallback (Task 3); drift verified bounded by trajectory-recovery test (Task 6); CLI `--pose_window`/`--pose_overlap` (Task 7). Single-window reduces to existing output (Task 4 identity + Task 5 window-0 branch).
- **Placeholder scan:** none — all code steps complete.
- **Type consistency:** `umeyama_similarity(src,dst)->(s,R,t)`, `apply_similarity_to_poses(poses_c2w,s,R,t)->(N,4,4)`, `robust_similarity(src_poses_c2w,dst_poses_c2w,min_spread=1e-4)->(s,R,t)`, `_relative_se3_from_extrinsics(extrinsics_h,ref_extrinsic=None)->(T,9)`, `_window_extrinsics(frames)->(T,4,4)`, `extract_video_pose_streaming(video_path,frame_indices,store,video_id,window=600,overlap=120)`, `extract_single_video(...,pose_window=600,pose_overlap=120)` used consistently. All poses are camera-to-world `(N,4,4)` in the stitching helpers; extrinsics are world-to-camera. Conversion `c2w = inv(w2c)` applied at the seam.
- **Note:** absolute translation scale is window-0's normalized scale (as in the existing non-metric pose output); streaming vs single-inference agree only up to a global scale — expected and documented.
