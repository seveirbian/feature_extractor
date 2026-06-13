"""Tests for the decord/PyAV video reader shim.

Fixtures are synthesized at runtime so the suite needs no large data files. Each
clip is a grayscale ramp: frame ``i`` is filled with value ``i * STEP``. After
lossy decoding the mean luminance still maps unambiguously back to ``i``, which
lets us assert the *exact* index->frame mapping (and so catch off-by-one bugs),
not merely the frame shape.
"""

import numpy as np
import pytest

from feature_extractor.validation.synthetic import make_ramp_video
from feature_extractor.video_io import VideoReader, cpu

N_FRAMES = 12
STEP = 20  # grayscale value per frame index; well above lossy-decode error
WIDTH, HEIGHT = 128, 96


def _make_video(path, codec):
    make_ramp_video(path, codec=codec, n_frames=N_FRAMES, width=WIDTH, height=HEIGHT, step=STEP)


@pytest.fixture(scope="module")
def av1_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video_io") / "ramp_av1.mp4"
    _make_video(path, "libsvtav1")
    return str(path)


@pytest.fixture(scope="module")
def h264_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video_io") / "ramp_h264.mp4"
    _make_video(path, "libx264")
    return str(path)


def _frame_index(array: np.ndarray) -> int:
    """Recover the source frame index from a decoded ramp frame."""
    return int(round(float(array.mean()) / STEP))


# -- root cause: decord cannot decode AV1 -------------------------------------


def test_raw_decord_fails_on_av1(av1_video):
    """Documents why this shim exists: bare decord can't open AV1 clips."""
    decord = pytest.importorskip("decord")
    with pytest.raises(Exception):
        decord.VideoReader(av1_video, ctx=decord.cpu(0))


# -- backend selection --------------------------------------------------------


def test_av1_uses_pyav_backend(av1_video):
    vr = VideoReader(av1_video, ctx=cpu(0))
    assert vr._decord is None  # fell back to PyAV
    assert len(vr) == N_FRAMES


def test_h264_uses_decord_backend(h264_video):
    pytest.importorskip("decord")
    vr = VideoReader(h264_video, ctx=cpu(0))
    assert vr._decord is not None  # decord handles H.264, no fallback
    assert len(vr) == N_FRAMES


# -- frame indexing correctness (both backends) -------------------------------


@pytest.mark.parametrize("fixture", ["av1_video", "h264_video"])
def test_monotonic_access_maps_to_correct_frame(fixture, request):
    video = request.getfixturevalue(fixture)
    vr = VideoReader(video, ctx=cpu(0))
    for i in range(N_FRAMES):
        arr = vr[i].asnumpy()
        assert arr.shape == (HEIGHT, WIDTH, 3)
        assert arr.dtype == np.uint8
        assert _frame_index(arr) == i, f"{fixture}: index {i} decoded wrong frame"


@pytest.mark.parametrize("fixture", ["av1_video", "h264_video"])
def test_out_of_order_access_returns_correct_frames(fixture, request):
    video = request.getfixturevalue(fixture)
    vr = VideoReader(video, ctx=cpu(0))
    # Jump to the end, then backwards (forces a decoder reset on the PyAV path).
    for i in [N_FRAMES - 1, 1, 5, 0, N_FRAMES - 2]:
        assert _frame_index(vr[i].asnumpy()) == i


@pytest.mark.parametrize("fixture", ["av1_video", "h264_video"])
def test_sparse_increasing_subset(fixture, request):
    video = request.getfixturevalue(fixture)
    vr = VideoReader(video, ctx=cpu(0))
    indices = [0, 3, 7, N_FRAMES - 1]  # the pattern sample_frame_indices produces
    frames = [vr[i].asnumpy() for i in indices]
    assert [_frame_index(f) for f in frames] == indices


def test_pyav_out_of_range_raises(av1_video):
    vr = VideoReader(av1_video, ctx=cpu(0))
    with pytest.raises(IndexError):
        vr[N_FRAMES]
