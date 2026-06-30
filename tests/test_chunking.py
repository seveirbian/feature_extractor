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
