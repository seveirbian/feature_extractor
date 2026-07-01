# Full-Frame Streaming Extraction — Phase 2 (Depth) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Depth (Video-Depth-Anything) branch extract every frame within bounded memory by processing the video in overlapping segments, aligning each segment's raw depth to the running scale, and metricizing with globally-spaced Depth Pro keyframes carried across segment boundaries.

**Architecture:** Two pure helpers (keyframe planning + affine-parameter interpolation) do the segment-boundary bookkeeping and are unit-tested with synthetic arrays. A new `DepthExtractor.extract_video_depth_streaming` orchestrates: `iter_frame_blocks` (with overlap) → per-segment `infer_video_depth` → raw-domain affine alignment to the previous segment's overlap (reusing `_fit_depth_affine`) → per-keyframe Depth Pro metricization with a carried anchor → per-frame inverse depth → `write_depth_chunk`. The orchestration is integration-tested with stubbed models (the Phase-1 `_StubDINO` pattern).

**Tech Stack:** Python, NumPy, h5py, pytest, Video-Depth-Anything + Depth Pro (stubbed in tests), `feature_extractor.chunking`, synthetic videos via `make_ramp_video`.

**Spec:** `docs/superpowers/specs/2026-06-30-full-frame-streaming-extraction-design.md` (Component 3).

**Depends on:** Phase 1 (merged) — `iter_frame_blocks`, `store.write_depth_chunk`, `store.mark_branch_complete`, CLI `stream`/`--depth_overlap` plumbing.

---

## Background (read before starting)

Current in-memory VDA path in [depth.py](../../../src/feature_extractor/extractors/depth.py) `extract_video` (lines ~496-519):

```python
depths, _ = self.model.infer_video_depth(np.stack(frames), target_fps=30,
    input_size=self.vda_input_size, device=self.device.type, fp32=self.device.type == "cpu")
depths = np.asarray(depths[:len(valid_indices)], dtype=np.float32)   # (T,H,W) raw
if self.vda_metric:
    metric_depths = np.clip(depths, self.z_min, self.z_max)
else:
    metric_depths = self._metricize_vda_depth_sequence(depths, frames, valid_indices)
inv_depths = [self._to_inverse_depth(depth)[..., None] for depth in metric_depths]
```

Relevant existing pieces (do NOT reimplement):
- `DepthExtractor._fit_depth_affine(src, tgt) -> (scale, shift)` (static, depth.py:212) — least-squares affine `src→tgt`; handles the degenerate constant-input case by returning `scale=1, shift=max(0, tgt_mean-src_mean)`.
- `DepthExtractor._to_inverse_depth(depth_map) -> (H,W)` (depth.py:328) — per-frame clip to `[z_min,z_max]` then normalized inverse depth in `[0,1]`.
- `DepthExtractor._extract_depth_pro(image) -> (H,W)` metric depth (depth.py:448).
- `self.model.infer_video_depth(frames, target_fps, input_size, device, fp32) -> (raw (T,H,W), fps)`.
- `self.keyframe_interval` (default 30), `self.vda_metric`, `self.z_min/z_max`, `self.vda_input_size`, `self.device`.

Why both raw-alignment AND carried keyframes: each `infer_video_depth` call is internally scale-consistent, but different segments have independent raw scales. Raw-domain affine alignment puts all segments into one running raw frame so a carried keyframe `(scale, shift)` anchor stays valid across the boundary; Depth Pro keyframes then re-anchor to true metric.

---

## File Structure

- **Create** `src/feature_extractor/extractors/depth_streaming.py` — pure helpers `plan_segment_keyframes`, `interpolate_segment_params`. No torch/model imports.
- **Modify** `src/feature_extractor/extractors/depth.py` — add `extract_video_depth_streaming`.
- **Modify** `src/feature_extractor/cli.py` — route the depth branch to streaming when `stream=True`; thread `depth_overlap` through `extract_single_video`.
- **Create** `tests/test_depth_streaming.py` — helper unit tests + stubbed orchestration test.
- **Modify** `tests/test_cli_streaming.py` — assert depth-overlap plumbing.

---

## Task 1: Keyframe planning helper (`plan_segment_keyframes`)

**Files:**
- Create: `src/feature_extractor/extractors/depth_streaming.py`
- Test: `tests/test_depth_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_streaming.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_depth_streaming.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'feature_extractor.extractors.depth_streaming'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/extractors/depth_streaming.py
"""Pure segment-boundary helpers for streaming VDA depth metricization.

No torch/model imports: given a segment's source frame indices and carried
keyframe state, decide which rows are Depth Pro keyframes and interpolate the
per-frame affine (scale, shift) that maps VDA raw depth to metric depth.
"""

from __future__ import annotations

import numpy as np


def plan_segment_keyframes(
    source_indices: list[int],
    last_kf_source_idx: int | None,
    keyframe_interval: int,
) -> list[int]:
    """Return row indices (into this segment) that should be Depth Pro keyframes.

    Keyframes are spaced by ``keyframe_interval`` over the *global* source frame
    index. The first-ever segment (``last_kf_source_idx is None``) anchors its
    own row 0. The last row is always forced so interpolation is bounded and the
    next segment inherits a valid anchor.
    """
    n = len(source_indices)
    if n == 0:
        return []

    rows: list[int] = []
    if last_kf_source_idx is None:
        rows.append(0)
        last = source_indices[0]
    else:
        last = last_kf_source_idx

    for r in range(n):
        if r in rows:
            continue
        if source_indices[r] - last >= keyframe_interval:
            rows.append(r)
            last = source_indices[r]

    if rows[-1] != n - 1:
        rows.append(n - 1)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_depth_streaming.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/depth_streaming.py tests/test_depth_streaming.py
git commit -m "feat(depth): plan_segment_keyframes streaming keyframe planner"
```

---

## Task 2: Affine-parameter interpolation helper (`interpolate_segment_params`)

**Files:**
- Modify: `src/feature_extractor/extractors/depth_streaming.py`
- Test: `tests/test_depth_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_streaming.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_depth_streaming.py -k interpolate -q`
Expected: FAIL with `ImportError: cannot import name 'interpolate_segment_params'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/extractors/depth_streaming.py  (append)
def interpolate_segment_params(
    source_indices: list[int],
    anchors: list[tuple[int, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate (scale, shift) across a segment's source indices.

    ``anchors`` are ``(source_idx, scale, shift)`` sorted ascending by
    source_idx (a carried anchor from the previous segment, if any, comes
    first). Uses ``np.interp``; queries at or beyond the anchor range clamp to
    the nearest anchor.
    """
    if not anchors:
        raise ValueError("interpolate_segment_params requires at least one anchor")
    xs = np.asarray([a[0] for a in anchors], dtype=np.float64)
    scales = np.asarray([a[1] for a in anchors], dtype=np.float64)
    shifts = np.asarray([a[2] for a in anchors], dtype=np.float64)
    q = np.asarray(source_indices, dtype=np.float64)
    return (
        np.interp(q, xs, scales).astype(np.float32),
        np.interp(q, xs, shifts).astype(np.float32),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_depth_streaming.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/depth_streaming.py tests/test_depth_streaming.py
git commit -m "feat(depth): interpolate_segment_params affine interpolation"
```

---

## Task 3: Streaming depth orchestration (`extract_video_depth_streaming`)

**Files:**
- Modify: `src/feature_extractor/extractors/depth.py` (add method after `extract_video`, ~line 529)
- Test: `tests/test_depth_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_streaming.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_depth_streaming.py -k depth_streaming_writes -q`
Expected: FAIL with `AttributeError: type object 'DepthExtractor' has no attribute 'extract_video_depth_streaming'`

- [ ] **Step 3: Write minimal implementation**

Add to `DepthExtractor` in `src/feature_extractor/extractors/depth.py` (import `tqdm` and `Path` already present at top of file; verify and add if missing):

```python
    def extract_video_depth_streaming(
        self,
        video_path: str,
        frame_indices: list[int],
        store,
        video_id: str,
        block_size: int = 1024,
        overlap: int = 96,
    ) -> None:
        """Stream VDA inverse depth to ``store`` in overlapping segments.

        Each segment is VDA-inferred, affine-aligned in the raw domain to the
        previous segment's overlap, metricized with globally-spaced Depth Pro
        keyframes (anchor carried across the boundary), converted to inverse
        depth per frame, and its non-overlap tail written incrementally.
        """
        from ..chunking import iter_frame_blocks
        from .depth_streaming import plan_segment_keyframes, interpolate_segment_params

        prev_aligned_overlap = None      # (overlap,H,W) aligned raw of previous seg tail
        last_kf = None                   # (source_idx, scale, shift) carried anchor
        last_kf_source_idx = None
        carry = overlap > 0              # carrying only valid when raw is aligned
        first = True

        pbar = tqdm(total=len(frame_indices),
                    desc=f"Depth stream [{Path(video_path).name}]", unit="f")
        for block_idx, frames, write_offset in iter_frame_blocks(
            video_path, frame_indices, block_size, overlap
        ):
            seg_src = [int(x) for x in block_idx]

            raw, _ = self.model.infer_video_depth(
                frames, target_fps=30, input_size=self.vda_input_size,
                device=self.device.type, fp32=self.device.type == "cpu",
            )
            raw = np.asarray(raw[: len(seg_src)], dtype=np.float32)  # (b,H,W)

            # Raw-domain affine alignment to the running scale
            if prev_aligned_overlap is not None and write_offset > 0:
                scale, shift = self._fit_depth_affine(raw[:write_offset], prev_aligned_overlap)
                raw = raw * scale + shift

            # Metricize
            if self.vda_metric:
                metric = np.clip(raw, self.z_min, self.z_max)
            else:
                kf_rows = plan_segment_keyframes(
                    seg_src, last_kf_source_idx if carry else None, self.keyframe_interval
                )
                anchors: list[tuple[int, float, float]] = []
                if carry and last_kf is not None:
                    anchors.append(last_kf)
                for r in kf_rows:
                    m = self._extract_depth_pro(frames[r])
                    sc, sh = self._fit_depth_affine(raw[r], m)
                    anchors.append((seg_src[r], float(sc), float(sh)))
                scales, shifts = interpolate_segment_params(seg_src, anchors)
                metric = np.clip(
                    raw * scales[:, None, None] + shifts[:, None, None],
                    self.z_min, self.z_max,
                )
                last_kf = anchors[-1]
                last_kf_source_idx = anchors[-1][0]

            inv = np.stack(
                [self._to_inverse_depth(metric[r])[..., None] for r in range(len(seg_src))],
                axis=0,
            ).astype(np.float32)

            store.write_depth_chunk(
                video_id, inv[write_offset:], block_idx[write_offset:], reset=first
            )
            first = False
            if overlap > 0:
                prev_aligned_overlap = raw[-overlap:].copy()
            pbar.update(len(block_idx) - write_offset)

        pbar.close()
        store.mark_branch_complete(video_id, "depth")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_depth_streaming.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/depth.py tests/test_depth_streaming.py
git commit -m "feat(depth): extract_video_depth_streaming segmented overlap-aligned extraction"
```

---

## Task 4: Segment-boundary continuity regression test

**Files:**
- Test: `tests/test_depth_streaming.py`

Proves the overlap alignment + carried anchor keep depth continuous across a segment boundary: splitting a clip into two overlapping segments must match a single-segment run on the same frames (same stub math), within tolerance.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_streaming.py  (append)
def test_depth_streaming_boundary_matches_single_segment(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=8, width=64, height=48, step=20)

    store_a = FeatureStore(str(tmp_path / "a"))
    _StubDepth().extract_video_depth_streaming(
        path, list(range(8)), store_a, "clip", block_size=8, overlap=0)  # single segment
    single = store_a.read_depth("clip")

    store_b = FeatureStore(str(tmp_path / "b"))
    _StubDepth().extract_video_depth_streaming(
        path, list(range(8)), store_b, "clip", block_size=5, overlap=3)  # two segments
    stitched = store_b.read_depth("clip")

    assert single.shape == stitched.shape == (8, 48, 64, 1)
    # boundary frames (around row 5) should agree closely with the single-segment result
    np.testing.assert_allclose(stitched[4:6], single[4:6], atol=2e-3)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_depth_streaming.py -k boundary -q`
Expected: PASS (the stub metric is deterministic per-frame, and alignment/interp preserve it). If it FAILS, the alignment or interpolation wiring is wrong — fix Task 3 before continuing, do not loosen the tolerance blindly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_depth_streaming.py
git commit -m "test(depth): segment-boundary continuity regression"
```

---

## Task 5: CLI — route depth branch to streaming

**Files:**
- Modify: `src/feature_extractor/cli.py` (`extract_single_video` and the `main` loop call)
- Test: `tests/test_cli_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_streaming.py  (append)
import inspect
from feature_extractor.cli import extract_single_video


def test_extract_single_video_accepts_depth_overlap():
    sig = inspect.signature(extract_single_video)
    assert "depth_overlap" in sig.parameters
    assert sig.parameters["depth_overlap"].default == 96
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_streaming.py -k depth_overlap -q`
Expected: FAIL with `AssertionError` (no `depth_overlap` parameter yet)

- [ ] **Step 3: Write minimal implementation**

In `src/feature_extractor/cli.py`, add `depth_overlap: int = 96` to `extract_single_video`'s signature (after `block_size: int = 1024,`) and replace the depth branch body:

```python
        # Extract Depth
        if extractor_depth is not None:
            if stream:
                extractor_depth.extract_video_depth_streaming(
                    video_path, frame_indices, store, video_id,
                    block_size=block_size, overlap=depth_overlap,
                )
            else:
                depth_inv = extractor_depth.extract_video(video_path, frame_indices=frame_indices)
                store.write_depth(video_id, depth_inv, frame_indices=frame_indices)
                store.mark_branch_complete(video_id, "depth")
```

Then in the `main` processing loop, pass `depth_overlap=args.depth_overlap` in the `extract_single_video(...)` call (alongside `block_size=args.block_size`):

```python
            stream=stream,
            block_size=args.block_size,
            depth_overlap=args.depth_overlap,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_streaming.py -q`
Expected: PASS (all cli-streaming tests)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/cli.py tests/test_cli_streaming.py
git commit -m "feat(cli): route depth branch to streaming with --depth_overlap"
```

---

## Task 6: Full-suite regression + docs

**Files:**
- Modify: `README.md` (6.1 note)

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -q`
Expected: all pass (existing 46 + new depth-streaming + cli tests).

- [ ] **Step 2: Update README §6.1 streaming-branch coverage note**

In `README.md`, change the warning that says only `dino` streams. Replace the sentence
"**当前仅 `dino` 分支支持全帧流式**" with:

```
**当前 `dino` 与 `depth` 分支支持全帧流式**;`pose` 仍走内存路径(Phase 3 处理)。
`--pose_window` / `--pose_overlap` 参数已预留,将在 Phase 3(Pose 滑窗拼接)启用。
```

And update the "全帧流式仅 DINO" limitation bullet in §10 to read "全帧流式支持 DINO/Depth;Pose 待 Phase 3"。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): depth branch now supports full-frame streaming"
```

---

## Self-Review notes

- **Spec coverage (Component 3):** overlapping segments + `iter_frame_blocks(overlap)` (Task 3); raw-domain affine alignment via `_fit_depth_affine` (Task 3); streaming keyframe metricization with global spacing + carried anchor (Tasks 1–3); `vda_metric=True` per-frame clip branch (Task 3); per-frame `_to_inverse_depth` + `write_depth_chunk` non-overlap write (Task 3); memory bounded by one segment (Task 3, buffered); CLI routing + `--depth_overlap` (Task 5); boundary continuity verified (Task 4).
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `plan_segment_keyframes(source_indices, last_kf_source_idx, keyframe_interval) -> list[int]`, `interpolate_segment_params(source_indices, anchors) -> (np.ndarray, np.ndarray)`, `extract_video_depth_streaming(video_path, frame_indices, store, video_id, block_size=1024, overlap=96)`, `extract_single_video(..., depth_overlap=96)` used consistently across tasks. Anchors are `(source_idx, scale, shift)` throughout.
- **Note on carried anchor:** only used when `overlap > 0` (raw is aligned); with `overlap == 0` each segment metricizes independently (Task 3 `carry` gate), which Task 4's single-segment `overlap=0` baseline relies on.
