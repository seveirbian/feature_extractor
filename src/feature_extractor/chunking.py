"""Streaming block planning and reading for full-frame extraction.

A "block" is a contiguous slice of the selected frame-index list. Blocks may
overlap (so downstream branches can align across boundaries), but every frame is
*written* exactly once: block 0 writes its whole range, later blocks write only
the non-overlap tail.
"""

from __future__ import annotations

from typing import Iterator, Sequence

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
