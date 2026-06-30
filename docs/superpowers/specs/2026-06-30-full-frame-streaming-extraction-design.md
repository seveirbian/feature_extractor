# Full-Frame Streaming Extraction — Design

**Date:** 2026-06-30
**Status:** Approved (pending spec review)

## Problem

`--frames_per_video <= 0` (extract all frames) is effectively unusable today and
silently aborts batch runs.

### Root cause

The per-video loop in [cli.py](../../../src/feature_extractor/cli.py) catches
`Exception` inside `extract_single_video` and continues to the next video, so a
*normal* error never aborts the batch. The reported "mid-process exit that leaves
remaining videos unprocessed" therefore comes from a **hard crash Python cannot
catch** — an OS OOM-kill (SIGKILL).

The OOM is specific to the all-frames path:

- `sample_frame_indices` returns every frame index with **no cap** when
  `frames_per_video <= 0` (`cli.py:97-100`), and the CLI always passes this
  explicit list to the extractors. The extractors' own internal 1000-frame safety
  cap only triggers when `frame_indices is None`, so it is dead code on the CLI
  path.
- Each branch then holds the **entire video in host RAM several times over**:
  - Depth (VDA): raw frame stack + VDA's internal `depth_list` *and*
    `depth_list_aligned` (two full-resolution depth copies) + downstream
    `metric_depths`/`inv_depths` copies.
  - DINO: all per-frame features accumulated in a list, then `np.stack`.
- A secondary fragility: `sample_frame_indices` is called **outside** the
  per-video `try/except` (`cli.py:253`), so a single corrupt video also aborts the
  whole batch.

GPU memory is **not** the primary problem: VDA already does internal 32-frame
sliding-window inference and DINO/Depth-Pro run per-frame. The blowup is host RAM.

### Constraints (from requirements gathering)

- Videos up to **10 min @ 30 Hz = ~18,000 frames**.
- **GPU 32 GB, host RAM 32 GB.** Raw RGB for 18k frames alone is ~16 GB — the
  whole video cannot live in RAM at once.
- **All three branches** (DINO, Depth, Pose) must produce **true full-frame**
  output.
- Cross-chunk consistency: **overlap-based stitching/alignment is acceptable**
  (option 1), with **larger overlap windows** preferred. Perfect global
  consistency is not required.
- Per-video on-disk size is large (depth ~11 GB uint16, DINO several GB to tens of
  GB). Acknowledged and accepted by the user; out of scope to optimize here.

## Goals

1. True full-frame extraction for DINO, Depth, and Pose within 32 GB RAM / 32 GB
   GPU on videos up to ~18k frames.
2. A single corrupt/failing video never aborts the batch.
3. No regression for the existing sampled path (small `frames_per_video`).

## Non-goals

- Reducing on-disk output size (compression/quantization beyond what exists).
- Metric accuracy improvements beyond preserving current behavior.
- Bundle-adjustment / global pose optimization (option 2 was declined).

## Overall approach

**In-process streaming + chunked inference + incremental HDF5 writes** (Approach
A). Models load once and are reused across videos; each video is processed in
blocks, and each block is written to disk and freed before the next. Memory is
bounded by one block, not the whole video.

A per-video subprocess-isolation mode (Approach B) is included only as an optional
fallback flag, off by default, because bounded memory already removes the OOM-kill
that motivated isolation.

Delivered in three independently verifiable phases:

- **Phase 1** — shared infrastructure + DINO streaming + CLI routing + batch
  resilience. Fixes the original bug and proves the pipeline end-to-end.
- **Phase 2** — Depth (VDA) segmented extraction with overlap alignment.
- **Phase 3** — Pose (VGGT) windowed extraction with sim(3) stitching.

---

## Component 1 — Shared infrastructure

### 1.1 Streaming block reader

`iter_frame_blocks(video_path, frame_indices, block_size, overlap=0)`
(new module `chunking.py`, or alongside `video_io.py`).

- Yields `(block_indices, frames_uint8)` where `frames_uint8` has shape
  `(b, H, W, 3)`, `b <= block_size`.
- Slides with `step = block_size - overlap`; consecutive blocks share `overlap`
  frames (used by Depth/Pose for boundary alignment).
- Reads sequentially, exploiting the PyAV backend's monotonic-forward-access
  optimization (`video_io.py:103-112`). Memory per block ≈ `block_size × frame
  bytes` (e.g. 1000 × ~0.9 MB ≈ 0.9 GB).

### 1.2 Block plan

`plan_blocks(total, block_size, overlap) -> list[(read_start, read_end, write_start)]`

- Block 0: read `[0, B)`, write `[0, B)`.
- Block k>0: read `[k·step, k·step+B)`, write only the non-overlap tail
  `[k·step+overlap, …)`; the leading `overlap` frames serve only as alignment
  context against already-written frames.
- Guarantees every frame is written **exactly once**. DINO uses `overlap=0`.

### 1.3 Incremental HDF5 writes (extend `storage.py`, backward compatible)

Add chunked writers; leave the existing one-shot `write_dino/write_depth/
write_pose` untouched for the sampled path:

```python
def write_dino_chunk(self, video_id, features, frame_indices, *, reset): ...
def write_depth_chunk(self, video_id, inv_depth, frame_indices, *, reset): ...
def write_pose_chunk(self,  video_id, se3,       frame_indices, *, reset): ...
```

- `reset=True` (first block): delete any existing dataset, create with
  `maxshape=(None, …)`, write block.
- `reset=False`: `resize` axis 0 by the block length, write the new slice, append
  `frame_indices`.
- Each block flushes on close → output memory bounded per block; a crash leaves
  already-written blocks intact on disk.
- Dataset layout, dtype, chunking, and `frame_indices` semantics are identical to
  the one-shot writers, so `read_dino/read_depth/read_pose` are unchanged and
  downstream readers are unaffected.

### 1.4 Completion marker (crash-safe resume)

After the final block of a branch is written, set a `complete = True` attribute on
that branch group. `--resume` skips a video only when **all requested branches**
are marked complete; a half-written `.h5` from a crash is re-run and overwritten.

---

## Component 2 — DINO streaming (Phase 1)

New `DINOExtractor.extract_video_streaming(video_path, frame_indices, store,
video_id, block_size=512)`; existing `extract_video` is left in place.

```
for bi, (blk_idx, frames) in enumerate(
        iter_frame_blocks(video_path, frame_indices, block_size, overlap=0)):
    feats = np.stack([self.extract_frame(f) for f in frames]).astype(np.float32)
    store.write_dino_chunk(video_id, feats, blk_idx, reset=(bi == 0))
    del feats
```

- Frames are independent — no cross-block coupling.
- GPU memory bounded (per-frame inference, as today). Host RAM holds one block of
  frames + one block of features at a time.
- Output shape / dtype / `frame_indices` semantics identical to `write_dino`.

---

## Component 3 — Depth (VDA) segmented (Phase 2)

### Key facts that simplify this

- `_metricize_vda_depth_sequence` (`depth.py:264-299`) is **not** a single global
  fit: it calibrates VDA relative depth to metric using **Depth Pro keyframes**
  spaced by `keyframe_interval`, then interpolates per-frame scale/shift. Each
  segment with its own keyframes is thus re-anchored to true metric → cross-segment
  consistency is largely free in the default `vda_metric=False` path.
- `_to_inverse_depth` (`depth.py:328-337`) is **fully per-frame** (fixed
  z_min/z_max normalization), no sequence coupling.

### Design

New `DepthExtractor.extract_video_streaming(video_path, frame_indices, store,
video_id, block_size, overlap)`. Per segment (read `block_size`, overlap `overlap`
with previous):

1. `raw = self.model.infer_video_depth(segment)` — VDA's internal 32-frame
   windowing keeps GPU bounded and aligns within the segment.
2. **Raw-domain cross-segment affine alignment**: using the overlap frames, run
   VDA's own `compute_scale_and_shift` to align this segment's raw depth to the
   previous segment's already-aligned overlap frames. The whole video's VDA raw
   depth thus stays in one consistent affine frame (temporal scale continuity).
   Segment 0 is the reference (no alignment).
3. **Metricization (streaming refactor):**
   - `vda_metric=True`: per-frame `clip(z_min, z_max)`, stateless.
   - `vda_metric=False`: refactor `_metricize_vda_depth_sequence` into a
     streaming-friendly form that **carries keyframe state across segments** —
     keyframes are selected over the **global** frame sequence (not reset per
     segment), and the previous segment's last keyframe `(scale, shift)` seeds the
     interpolation so the boundary does not jump.
4. `_to_inverse_depth` per frame → write only the **non-overlap new frames** via
   `write_depth_chunk(reset=(segment == 0))`.
5. Memory holds: current segment raw depth (~0.7 GB for 600 frames) + previous
   segment's last `overlap` aligned-raw frames. Freed after each segment.

**Main work:** the keyframe-calibration streaming refactor (step 3). Everything
else reuses existing helpers.

**Trade-off:** segment boundaries may show a tiny residual depth jump, suppressed
by both raw-domain overlap alignment and Depth Pro keyframe metric anchoring —
within the accepted option-1 tolerance.

---

## Component 4 — Pose (VGGT) windowed sim(3) stitching (Phase 3)

### Key facts

- VGGT (`_infer_sequence_pose_enc` → `pose_encoding_to_extri_intri`) is O(W²)
  global attention; window `W` is bounded by GPU (current `max_frames=600`).
- Output is world-to-camera extrinsics made **frame-0-relative** (`E_i @ E_0^{-1}`),
  stored as 9-dim (translation + rot6d). Translation is in VGGT's **per-inference
  normalized scene scale** — each window has its own world frame *and* its own
  scale ⟹ stitching must be **sim(3)** (rotation + translation + scale).

### Design

New `PoseExtractor.extract_video_streaming(video_path, frame_indices, store,
video_id, window=max_frames, overlap)`.

- Global frame is anchored at the **video's frame-0 camera** (matches the existing
  frame-0-relative output convention exactly).
- Per window (read `W` frames, overlap `O` with previous):
  1. `_infer_sequence_pose_enc` → per-frame extrinsics in the window-local world
     frame.
  2. **Window 0**: defines the global frame (frame 0 = identity). **Window k>0**:
     estimate a similarity transform `(s, R, t)` via **Umeyama** on the `O` overlap
     frames' camera centers (window-local vs. already-established global).
  3. Apply the sim(3) to the window's non-overlap poses → global frame:
     center `C' = sRC + t`, orientation `R_c' = R·R_c` (scale does not affect
     rotation).
  4. Each new frame's global pose is its frame-0-relative pose → 9-dim
     (translation + rot6d) → `write_pose_chunk(reset=(window == 0))`.
  5. Keep the last `O` global poses in memory for the next window's alignment
     (poses are tiny).
- New helper `umeyama_similarity(src, dst)` (pure-numpy SVD).

### Risks and mitigations

- **Drift**: ~18k frames / step ≈ tens of windows; each handoff accumulates small
  error. Larger `O` reduces it (user chose larger overlap).
- **Degenerate motion** (near-pure rotation / negligible translation) makes
  center-based Umeyama ill-conditioned → fall back to rotation-only alignment with
  a median-scale estimate; estimate robustly over the overlap frames.
- **Verification**: split a short clip that fits in one window into two overlapping
  windows, run streaming, and confirm the stitched trajectory ≈ the single-pass
  result (regression test for stitching correctness).

---

## Component 5 — CLI wiring, routing, and resilience

### Routing

In `extract_single_video`, route by selected frame count:
`len(frame_indices) > --stream_threshold` (default 2000) → `extract_video_streaming`
(branch writes incrementally); otherwise the existing in-memory path
(returns arrays, `store.write_*`). Full-frame on long videos naturally exceeds the
threshold; short videos / small samples keep the simple path. An explicit override
flag is also provided.

### New CLI arguments

- `--block_size` (DINO/Depth segment length, default 512)
- `--pose_window` (default = pose `max_frames`, 600)
- `--depth_overlap` (default 64), `--pose_overlap` (default 96) — intentionally
  large per the consistency requirement
- `--stream_threshold` (default 2000)
- `--isolate_subprocess` (optional, default off)

### Batch resilience (fixes the original bug)

1. Move `sample_frame_indices` **inside** the per-video `try/except` (currently at
   `cli.py:253`) so a corrupt video counts as one failure instead of aborting the
   batch.
2. Block-bounded memory removes the OOM-kill that Python cannot catch.
3. **Resume completeness**: rely on the §1.4 `complete=True` markers — `--resume`
   skips only videos whose requested branches are all complete; half-written files
   are re-run.
4. **Optional `--isolate_subprocess`**: run each video's extraction in a child
   process so an unexpected hard crash only loses that video. Off by default
   (model reload cost); bounded memory makes it unnecessary in normal operation.

---

## Testing strategy

- **Unit**: `plan_blocks` boundaries/coverage (every frame once, overlap correct);
  `iter_frame_blocks` block shapes and overlap; `umeyama_similarity` against known
  transforms; chunked writers produce datasets bit-identical in layout to one-shot
  writers.
- **Integration**: a short synthetic/real clip extracted via the streaming path
  vs. the in-memory path produces equal DINO features and equivalent depth/pose
  (within stitching tolerance for split windows).
- **Resilience**: a deliberately corrupt video in a directory does not abort the
  batch; resume skips only completed videos and re-runs a truncated `.h5`.
- **Memory**: full-frame run on a long clip stays within RAM budget (observed peak
  RSS bounded by block size, not video length).

## Phasing summary

| Phase | Scope | Verifies |
|-------|-------|----------|
| 1 | §1 infra + §2 DINO + §5 routing/resilience + resume markers | Original bug fixed; pipeline end-to-end |
| 2 | §3 Depth segmented alignment | Full-frame depth within memory budget |
| 3 | §4 Pose windowed sim(3) stitching | Full-frame pose within memory budget |
