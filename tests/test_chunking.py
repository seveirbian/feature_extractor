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
