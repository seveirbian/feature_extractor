# Full-Frame Streaming Extraction — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared streaming/chunking infrastructure, make the DINO branch extract every frame within bounded memory, and make the batch loop resilient so one bad video (or an OOM) never abandons the rest of the directory.

**Architecture:** A new `chunking` module yields fixed-size frame blocks from a video (bounded RAM). The `FeatureStore` gains append-mode HDF5 writers with per-branch `complete` markers (crash-safe resume). `DINOExtractor` gets a streaming method that writes block-by-block. The CLI routes large frame counts to the streaming path, samples frame indices *inside* the per-video try/except, and resumes on completion markers rather than file existence.

**Tech Stack:** Python, NumPy, h5py, pytest, PyAV/decord (`video_io.VideoReader`), synthetic test videos via `feature_extractor.validation.synthetic.make_ramp_video`.

**Spec:** `docs/superpowers/specs/2026-06-30-full-frame-streaming-extraction-design.md` (Components 1, 2, 5).

**Out of scope (later plans):** Depth (VDA) segmented extraction = Phase 2; Pose (VGGT) windowed sim(3) stitching = Phase 3. This plan still adds the `write_depth_chunk` / `write_pose_chunk` writers (shared infra) so Phases 2–3 only add the extraction logic.

---

## File Structure

- **Create** `src/feature_extractor/chunking.py` — `plan_blocks` (block boundary math) and `iter_frame_blocks` (streaming reader). One responsibility: turn a frame-index list + block/overlap into bounded blocks.
- **Modify** `src/feature_extractor/storage.py` — add `write_dino_chunk`, `write_depth_chunk`, `write_pose_chunk`, `mark_branch_complete`, `is_branch_complete`, `is_video_complete`. Append-mode counterparts to the existing one-shot writers.
- **Modify** `src/feature_extractor/extractors/dino.py` — add `extract_video_streaming`.
- **Modify** `src/feature_extractor/cli.py` — new args, streaming routing, move `sample_frame_indices` inside the try, completion-based resume.
- **Create** `tests/test_chunking.py`, `tests/test_storage_chunked.py`, `tests/test_dino_streaming.py`, `tests/test_cli_streaming.py`.

---

## Task 1: Block planning math (`plan_blocks`)

**Files:**
- Create: `src/feature_extractor/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
import numpy as np
import pytest

from feature_extractor.chunking import plan_blocks


def _covered(blocks):
    """Concatenate the WRITE ranges and return the list of written positions."""
    written = []
    for read_start, read_end, write_start in blocks:
        written.extend(range(write_start, read_end))
    return written


def test_plan_blocks_no_overlap_tiles_exactly():
    blocks = plan_blocks(total=10, block_size=4, overlap=0)
    assert blocks == [(0, 4, 0), (4, 8, 4), (8, 10, 8)]
    assert _covered(blocks) == list(range(10))


def test_plan_blocks_overlap_writes_each_frame_once():
    blocks = plan_blocks(total=10, block_size=4, overlap=2)
    # step = 2; reads overlap by 2, writes the non-overlap tail
    assert blocks[0] == (0, 4, 0)
    assert blocks[1] == (2, 6, 4)
    assert _covered(blocks) == list(range(10))


def test_plan_blocks_single_block_when_total_below_block_size():
    assert plan_blocks(total=3, block_size=4, overlap=2) == [(0, 3, 0)]


def test_plan_blocks_empty_for_zero_total():
    assert plan_blocks(total=0, block_size=4, overlap=0) == []


def test_plan_blocks_rejects_overlap_ge_block_size():
    with pytest.raises(ValueError):
        plan_blocks(total=10, block_size=4, overlap=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feature_extractor.chunking'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/chunking.py
"""Streaming block planning and reading for full-frame extraction.

A "block" is a contiguous slice of the selected frame-index list. Blocks may
overlap (so downstream branches can align across boundaries), but every frame is
*written* exactly once: block 0 writes its whole range, later blocks write only
the non-overlap tail.
"""

from __future__ import annotations

import numpy as np


def plan_blocks(total: int, block_size: int, overlap: int = 0) -> list[tuple[int, int, int]]:
    """Return ``(read_start, read_end, write_start)`` triples over ``range(total)``.

    ``read_start:read_end`` is the slice to load (for inference context).
    ``write_start:read_end`` is the slice to persist; overlap frames before
    ``write_start`` are alignment context only. Coverage of all write ranges is
    exactly ``range(total)`` with no gaps or duplicates.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if overlap < 0 or overlap >= block_size:
        raise ValueError(f"overlap must be in [0, block_size), got {overlap}")
    if total <= 0:
        return []

    step = block_size - overlap
    blocks: list[tuple[int, int, int]] = []
    read_start = 0
    first = True
    while True:
        read_end = min(read_start + block_size, total)
        write_start = 0 if first else read_start + overlap
        if write_start < read_end:
            blocks.append((read_start, read_end, write_start))
        if read_end >= total:
            break
        read_start += step
        first = False
    return blocks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): plan_blocks block boundary math"
```

---

## Task 2: Streaming block reader (`iter_frame_blocks`)

**Files:**
- Modify: `src/feature_extractor/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py  (append)
from feature_extractor.validation.synthetic import make_ramp_video
from feature_extractor.chunking import iter_frame_blocks


def _frame_index(frame):
    # ramp video: frame i is filled with value i*STEP; recover i from the mean
    return int(round(float(frame.mean()) / 20.0))


def test_iter_frame_blocks_streams_full_coverage(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=10, width=64, height=48, step=20)

    seen_written = []
    n_blocks = 0
    for block_idx, frames, write_offset in iter_frame_blocks(
        path, frame_indices=list(range(10)), block_size=4, overlap=2
    ):
        n_blocks += 1
        assert frames.shape[0] == len(block_idx)
        assert frames.ndim == 4 and frames.shape[-1] == 3
        # frames must correspond to their declared indices
        recovered = [_frame_index(frames[j]) for j in range(len(frames))]
        assert recovered == list(block_idx)
        # written tail
        seen_written.extend(int(i) for i in block_idx[write_offset:])

    assert n_blocks == 4
    assert seen_written == list(range(10))


def test_iter_frame_blocks_respects_frame_indices_subset(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=10, width=64, height=48, step=20)
    blocks = list(iter_frame_blocks(path, frame_indices=[0, 2, 4, 6], block_size=2, overlap=0))
    flat = [int(i) for blk, _f, off in blocks for i in blk[off:]]
    assert flat == [0, 2, 4, 6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -k iter_frame_blocks -v`
Expected: FAIL with `ImportError: cannot import name 'iter_frame_blocks'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/feature_extractor/chunking.py  (append)
from typing import Iterator, Sequence


def iter_frame_blocks(
    video_path: str,
    frame_indices: Sequence[int],
    block_size: int,
    overlap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield ``(block_frame_indices, frames, write_offset)`` blocks.

    ``frames`` has shape ``(b, H, W, 3)`` uint8 and contains *all* frames in the
    block (overlap included) for inference context. ``block_frame_indices`` are
    the absolute source frame indices. Persist only
    ``frames[write_offset:]`` / ``block_frame_indices[write_offset:]``.

    Reads are issued in monotonically increasing order, which the PyAV backend
    decodes in a single forward pass. Memory is bounded by one block.
    """
    from .video_io import VideoReader, cpu

    idx = [int(i) for i in frame_indices]
    vr = VideoReader(video_path, ctx=cpu(0))
    for read_start, read_end, write_start in plan_blocks(len(idx), block_size, overlap):
        block_positions = idx[read_start:read_end]
        frames = np.stack([vr[i].asnumpy() for i in block_positions], axis=0)
        write_offset = write_start - read_start
        yield np.asarray(block_positions, dtype=np.int64), frames, write_offset
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): iter_frame_blocks streaming reader"
```

---

## Task 3: Incremental DINO writer + completion markers

**Files:**
- Modify: `src/feature_extractor/storage.py`
- Test: `tests/test_storage_chunked.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage_chunked.py
import numpy as np

from feature_extractor.storage import FeatureStore


def test_write_dino_chunk_appends_and_reads_back_in_order(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "clip0"
    chunk0 = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    chunk1 = (chunk0 + 100).copy()

    store.write_dino_chunk(vid, chunk0, np.array([0, 1]), reset=True)
    store.write_dino_chunk(vid, chunk1, np.array([2, 3]), reset=False)

    feats = store.read_dino(vid)
    assert feats.shape == (4, 3, 5)
    np.testing.assert_allclose(feats[:2], chunk0)
    np.testing.assert_allclose(feats[2:], chunk1)
    np.testing.assert_array_equal(store.read_frame_indices(vid, "dino"), [0, 1, 2, 3])


def test_branch_completion_markers(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "clip1"
    store.write_dino_chunk(vid, np.zeros((1, 2, 2), np.float32), np.array([0]), reset=True)
    assert store.is_branch_complete(vid, "dino") is False
    store.mark_branch_complete(vid, "dino")
    assert store.is_branch_complete(vid, "dino") is True
    assert store.is_video_complete(vid, ["dino"]) is True
    assert store.is_video_complete(vid, ["dino", "depth"]) is False


def test_is_video_complete_false_when_missing(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    assert store.is_video_complete("nope", ["dino"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage_chunked.py -v`
Expected: FAIL with `AttributeError: 'FeatureStore' object has no attribute 'write_dino_chunk'`

- [ ] **Step 3: Write minimal implementation**

Add these methods to `FeatureStore` in `src/feature_extractor/storage.py` (place them after `write_dino`):

```python
    def _append_or_create(
        self,
        grp: "h5py.Group",
        name: str,
        data: np.ndarray,
        *,
        reset: bool,
        chunks: tuple[int, ...],
    ) -> None:
        if reset:
            if name in grp:
                del grp[name]
            maxshape = (None,) + data.shape[1:]
            grp.create_dataset(
                name,
                data=data,
                maxshape=maxshape,
                chunks=chunks,
                dtype=data.dtype,
                **self._compression_kwargs(),
            )
        else:
            dset = grp[name]
            n = dset.shape[0]
            dset.resize(n + data.shape[0], axis=0)
            dset[n:] = data

    def write_dino_chunk(
        self,
        video_id: str,
        features: np.ndarray,
        frame_indices: np.ndarray,
        *,
        reset: bool,
    ) -> None:
        """Append a block of DINO features to a resizable dataset."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim not in (2, 3):
            raise ValueError(f"DINO features must be (T,D) or (T,N,D), got {features.shape}")
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        chunks = (1,) + features.shape[1:]
        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("dino")
            self._append_or_create(grp, "features", features, reset=reset, chunks=chunks)
            self._append_or_create(grp, "frame_indices", frame_indices, reset=reset, chunks=(min(1024, len(frame_indices) or 1),))
            if reset:
                grp.attrs["complete"] = False
                grp.attrs["representation"] = "patch_tokens" if features.ndim == 3 else "global_descriptor"

    def mark_branch_complete(self, video_id: str, branch: "FeatureBranch") -> None:
        with self._get_file(video_id, mode="a") as f:
            f.require_group(branch).attrs["complete"] = True

    def is_branch_complete(self, video_id: str, branch: "FeatureBranch") -> bool:
        if not self.exists(video_id):
            return False
        with self._get_file(video_id, mode="r") as f:
            if branch not in f:
                return False
            return bool(f[branch].attrs.get("complete", False))

    def is_video_complete(self, video_id: str, branches: list["FeatureBranch"]) -> bool:
        return all(self.is_branch_complete(video_id, b) for b in branches)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage_chunked.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/storage.py tests/test_storage_chunked.py
git commit -m "feat(storage): incremental DINO writer + completion markers"
```

---

## Task 4: Incremental depth & pose writers (shared infra for Phases 2–3)

**Files:**
- Modify: `src/feature_extractor/storage.py`
- Test: `tests/test_storage_chunked.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage_chunked.py  (append)
def test_write_depth_chunk_roundtrips_uint16(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "d0"
    d0 = np.full((2, 4, 4, 1), 0.25, np.float32)
    d1 = np.full((3, 4, 4, 1), 0.75, np.float32)
    store.write_depth_chunk(vid, d0, np.array([0, 1]), reset=True)
    store.write_depth_chunk(vid, d1, np.array([2, 3, 4]), reset=False)
    out = store.read_depth(vid)
    assert out.shape == (5, 4, 4, 1)
    np.testing.assert_allclose(out[:2], 0.25, atol=1e-4)
    np.testing.assert_allclose(out[2:], 0.75, atol=1e-4)


def test_write_pose_chunk_appends(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "p0"
    p0 = np.zeros((2, 9), np.float32)
    p1 = np.ones((1, 9), np.float32)
    store.write_pose_chunk(vid, p0, np.array([0, 1]), reset=True)
    store.write_pose_chunk(vid, p1, np.array([2]), reset=False)
    out = store.read_pose(vid)
    assert out.shape == (3, 9)
    np.testing.assert_array_equal(out[2], np.ones(9, np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage_chunked.py -k "depth_chunk or pose_chunk" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'write_depth_chunk'`

- [ ] **Step 3: Write minimal implementation**

Add to `FeatureStore` (after `write_depth` / `write_pose` respectively, or grouped with the chunk writers):

```python
    def write_depth_chunk(
        self,
        video_id: str,
        inv_depth: np.ndarray,
        frame_indices: np.ndarray,
        *,
        reset: bool,
    ) -> None:
        """Append a block of normalized inverse depth as uint16."""
        inv_depth = self._normalize_depth(inv_depth)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        u16 = (np.clip(inv_depth, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
        chunks = (1, u16.shape[1], u16.shape[2], 1)
        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("depth")
            self._append_or_create(grp, "inv_depth", u16, reset=reset, chunks=chunks)
            self._append_or_create(grp, "frame_indices", frame_indices, reset=reset, chunks=(min(1024, len(frame_indices) or 1),))
            if reset:
                grp.attrs["complete"] = False
                grp.attrs["scale"] = 65535.0
                grp.attrs["representation"] = "normalized_inverse_depth"

    def write_pose_chunk(
        self,
        video_id: str,
        se3_trajectory: np.ndarray,
        frame_indices: np.ndarray,
        *,
        reset: bool,
        representation: str | None = None,
    ) -> None:
        """Append a block of pose targets (T, P)."""
        pose = self._normalize_pose(se3_trajectory)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if representation is None:
            representation = "se3_log" if pose.shape[1] == 6 else "translation_rot6d"
        chunks = (1, pose.shape[1])
        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("pose")
            self._append_or_create(grp, "se3_trajectory", pose, reset=reset, chunks=chunks)
            self._append_or_create(grp, "frame_indices", frame_indices, reset=reset, chunks=(min(1024, len(frame_indices) or 1),))
            if reset:
                grp.attrs["complete"] = False
                grp.attrs["pose_dim"] = pose.shape[1]
                grp.attrs["representation"] = representation
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage_chunked.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/storage.py tests/test_storage_chunked.py
git commit -m "feat(storage): incremental depth and pose chunk writers"
```

---

## Task 5: DINO streaming extraction

**Files:**
- Modify: `src/feature_extractor/extractors/dino.py` (add method after `extract_video`, ~line 233)
- Test: `tests/test_dino_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dino_streaming.py
import numpy as np

from feature_extractor.storage import FeatureStore
from feature_extractor.validation.synthetic import make_ramp_video


class _StubDINO:
    """Exercises extract_video_streaming without loading a real model."""

    embed_dim = 4

    # bind the real method under test
    from feature_extractor.extractors.dino import DINOExtractor
    extract_video_streaming = DINOExtractor.extract_video_streaming

    def extract_frame(self, frame):
        # deterministic per-frame "feature": mean of the frame, broadcast
        return np.full((1, self.embed_dim), float(frame.mean()), dtype=np.float32)


def test_dino_streaming_writes_every_frame(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=10, width=64, height=48, step=20)
    store = FeatureStore(str(tmp_path / "store"))

    _StubDINO().extract_video_streaming(
        video_path=path,
        frame_indices=list(range(10)),
        store=store,
        video_id="clip",
        block_size=4,
    )

    feats = store.read_dino("clip")
    assert feats.shape == (10, 1, 4)
    np.testing.assert_array_equal(store.read_frame_indices("clip", "dino"), list(range(10)))
    assert store.is_branch_complete("clip", "dino") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dino_streaming.py -v`
Expected: FAIL with `AttributeError: type object 'DINOExtractor' has no attribute 'extract_video_streaming'`

- [ ] **Step 3: Write minimal implementation**

Add to `DINOExtractor` in `src/feature_extractor/extractors/dino.py`:

```python
    def extract_video_streaming(
        self,
        video_path: str,
        frame_indices: list[int],
        store,
        video_id: str,
        block_size: int = 1024,
    ) -> None:
        """Stream DINO features to ``store`` block-by-block (bounded memory).

        Writes directly via ``store.write_dino_chunk`` instead of returning one
        large array, so host RAM stays bounded by a single block.
        """
        from ..chunking import iter_frame_blocks

        first = True
        for block_idx, frames, write_offset in iter_frame_blocks(
            video_path, frame_indices, block_size, overlap=0
        ):
            feats = np.stack(
                [self.extract_frame(frames[j]) for j in range(frames.shape[0])], axis=0
            ).astype(np.float32)
            feats = feats[write_offset:]
            out_idx = block_idx[write_offset:]
            if len(feats) == 0:
                continue
            store.write_dino_chunk(video_id, feats, out_idx, reset=first)
            first = False
            del feats
        store.mark_branch_complete(video_id, "dino")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dino_streaming.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/dino.py tests/test_dino_streaming.py
git commit -m "feat(dino): streaming full-frame extraction with incremental writes"
```

---

## Task 6: CLI — new args, streaming routing, resilient loop, completion-based resume

**Files:**
- Modify: `src/feature_extractor/cli.py` (`extract_single_video` ~108-152, `main` arg parsing ~155-186, processing loop ~247-282)
- Test: `tests/test_cli_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_streaming.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_streaming.py -v`
Expected: FAIL with `ImportError: cannot import name 'should_stream'`

- [ ] **Step 3: Write minimal implementation**

In `src/feature_extractor/cli.py` add two small pure helpers (near `sample_frame_indices`):

```python
def should_stream(n_frames: int, threshold: int) -> bool:
    """Route to the streaming path when the selected frame count is large."""
    return n_frames > threshold


def branches_to_resume_skip(store, video_id: str, branches: list[str]) -> bool:
    """Resume skips a video only when every requested branch is complete."""
    return store.is_video_complete(video_id, branches)
```

Add the new arguments in `main` (after the existing `--frames_per_video` argument):

```python
    parser.add_argument("--block_size", type=int, default=1024,
                        help="DINO/Depth streaming block length (frames)")
    parser.add_argument("--pose_window", type=int, default=600,
                        help="VGGT pose window length (frames); GPU-bound")
    parser.add_argument("--depth_overlap", type=int, default=96,
                        help="Overlap frames between depth segments")
    parser.add_argument("--pose_overlap", type=int, default=120,
                        help="Overlap frames between pose windows")
    parser.add_argument("--stream_threshold", type=int, default=2000,
                        help="Route to streaming path above this selected frame count")
```

Replace the body of `extract_single_video` so DINO uses streaming when requested. Change the signature to accept `branches`, `stream`, and `block_size`, and pass `store`/`video_id` through (it already receives them). New `extract_single_video`:

```python
def extract_single_video(
    video_path: str,
    video_id: str,
    extractor_dino: DINOExtractor | None,
    extractor_depth: DepthExtractor | None,
    extractor_pose: PoseExtractor | None,
    store: FeatureStore,
    frame_indices: list[int],
    branches: list[str],
    future_horizon: int = 4,
    resume: bool = False,
    stream: bool = False,
    block_size: int = 1024,
) -> bool:
    """Extract all features for one video. Returns True if successful."""
    if resume and branches_to_resume_skip(store, video_id, branches):
        print(f"  [SKIP] {video_id} already extracted")
        return True

    try:
        dino_feats = None
        if extractor_dino is not None:
            if stream:
                extractor_dino.extract_video_streaming(
                    video_path, frame_indices, store, video_id, block_size=block_size
                )
            else:
                dino_feats = extractor_dino.extract_video(video_path, frame_indices=frame_indices)
                store.write_dino(video_id, dino_feats, frame_indices=frame_indices)
                store.mark_branch_complete(video_id, "dino")

        depth_inv = None
        if extractor_depth is not None:
            depth_inv = extractor_depth.extract_video(video_path, frame_indices=frame_indices)
            store.write_depth(video_id, depth_inv, frame_indices=frame_indices)
            store.mark_branch_complete(video_id, "depth")

        pose_se3 = None
        if extractor_pose is not None:
            pose_se3 = extractor_pose.extract_video(video_path, frame_indices=frame_indices)
            store.write_pose(video_id, pose_se3, frame_indices=frame_indices)
            store.mark_branch_complete(video_id, "pose")

        print(f"  [OK] {video_id}")
        return True

    except Exception as e:
        print(f"  [ERROR] {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
```

Update the processing loop in `main` so index sampling is **inside** the per-video error boundary and the new args are passed. Replace the loop body (`cli.py:251-275`) with:

```python
    for video_path in tqdm(videos, desc="Extracting features"):
        video_id = video_id_from_path(video_path, stem_only=args.id_from_stem)
        try:
            frame_indices = sample_frame_indices(video_path, args.frames_per_video)
        except Exception as e:
            print(f"  [ERROR] {video_id}: failed to read/sample frames: {e}")
            failures += 1
            continue

        stream = should_stream(len(frame_indices), args.stream_threshold)
        ok = extract_single_video(
            video_path=video_path,
            video_id=video_id,
            extractor_dino=extractor_dino,
            extractor_depth=extractor_depth,
            extractor_pose=extractor_pose,
            store=store,
            frame_indices=frame_indices,
            branches=branches,
            future_horizon=args.future_horizon,
            resume=args.resume,
            stream=stream,
            block_size=args.block_size,
        )

        if ok:
            successes += 1
            ann_path = os.path.join(ann_dir, f"{video_id}.json")
            if not os.path.exists(ann_path):
                ann = create_annotation(video_path, video_id)
                with open(ann_path, "w") as f:
                    json.dump(ann, f, indent=2)
        else:
            failures += 1
```

> Note: in this phase only DINO streams. When `stream` is True and `depth`/`pose`
> are selected, those branches still use the in-memory path (Phases 2–3 add their
> streaming). The `branches` list is `[b.strip() for b in args.branches.split(",")]`,
> already computed at `cli.py:194`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_streaming.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/cli.py tests/test_cli_streaming.py
git commit -m "feat(cli): streaming routing, in-try frame sampling, completion-based resume"
```

---

## Task 7: Full-suite regression + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -q`
Expected: all tests pass (existing + new `test_chunking`, `test_storage_chunked`, `test_dino_streaming`, `test_cli_streaming`).

- [ ] **Step 2: Manual smoke for the original bug (corrupt video does not abort batch)**

```bash
python - <<'PY'
import os, tempfile, numpy as np
from feature_extractor.validation.synthetic import make_ramp_video
d = tempfile.mkdtemp()
make_ramp_video(os.path.join(d, "good1.mp4"), n_frames=8, width=64, height=48, step=20)
open(os.path.join(d, "bad.mp4"), "wb").write(b"not a video")
make_ramp_video(os.path.join(d, "good2.mp4"), n_frames=8, width=64, height=48, step=20)
print("data dir:", d)
PY
# Then run DINO-only extraction over that dir (CPU) and confirm BOTH good videos
# are processed and the run exits with the bad one counted as a failure, not a crash:
python -m feature_extractor.cli --data_root <data dir> --output_root /tmp/out \
    --branches dino --device cpu --frames_per_video 0 --stream_threshold 4 --block_size 4
```

Expected: console shows two `[OK]` lines and `Failures: 1`; the process does not abort early.

- [ ] **Step 3: Commit any fixups** (only if Step 1/2 surfaced issues)

```bash
git add -A
git commit -m "test: phase-1 streaming regression fixups"
```

---

## Self-Review notes

- **Spec coverage:** §1.1 `iter_frame_blocks` (Task 2), §1.2 `plan_blocks` (Task 1), §1.3 chunk writers (Tasks 3–4), §1.4 completion markers (Task 3), §2 DINO streaming (Task 5), §5 routing + in-try sampling + resume completeness + new args (Task 6). Depth (§3) and Pose (§4) streaming are explicitly deferred to Phase 2/3 plans; their *writers* land here as shared infra.
- **Type consistency:** `write_*_chunk(..., frame_indices, *, reset)`, `mark_branch_complete`, `is_branch_complete`, `is_video_complete`, `extract_video_streaming(video_path, frame_indices, store, video_id, block_size=...)`, `should_stream(n_frames, threshold)`, `branches_to_resume_skip(store, video_id, branches)` are used identically across tasks.
- **Default consistency with spec:** `block_size=1024`, `pose_window=600`, `depth_overlap=96`, `pose_overlap=120`, `stream_threshold=2000` match the spec's defaults table. (`pose_window`/`pose_overlap`/`depth_overlap` are wired as CLI args now; consumed in Phases 2–3.)
