"""decord-compatible video reader with a PyAV fallback for AV1.

decord's bundled FFmpeg only attempts *hardware* AV1 decoding and has no
software AV1 decoder compiled in, so it fails on AV1-encoded clips (LeRobot's
default codec) with "cannot find video stream with wanted index: -1". PyAV's
binary wheel bundles an FFmpeg built with libdav1d, which decodes AV1 in pure
software.

This module exposes a drop-in replacement for decord's ``VideoReader`` (plus a
no-op ``cpu`` context stub), so call sites only swap their import line:

    from .video_io import VideoReader, cpu   # was: from decord import ...

``VideoReader`` tries decord first and transparently falls back to PyAV when
decord cannot open the file. Both backends index frames 0..N-1 in presentation
order and return RGB uint8 ``(H, W, 3)`` arrays from ``frame.asnumpy()``.
"""

from __future__ import annotations

import numpy as np


def cpu(index: int = 0):
    """No-op stand-in for ``decord.cpu`` so the import line stays unchanged.

    The real decord context is constructed internally by the decord backend;
    the value passed by callers is ignored.
    """
    return index


class _Frame:
    """Wraps a decoded RGB array to mirror decord's ``frame.asnumpy()`` API."""

    __slots__ = ("_array",)

    def __init__(self, array: np.ndarray):
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array


class VideoReader:
    """Sequential-friendly video reader: decord when possible, else PyAV.

    Supports the subset of decord's interface the pipeline uses: ``len(vr)`` and
    ``vr[i].asnumpy()``. Random access is O(1) on the decord backend; on the
    PyAV backend it is optimized for monotonically increasing indices (a single
    forward decode pass) and falls back to a re-decode for out-of-order access.
    """

    def __init__(self, video_path: str, ctx=None, **kwargs):
        self._path = video_path
        self._decord = None
        self._container = None
        try:
            from decord import VideoReader as _DecordVR
            from decord import cpu as _decord_cpu

            self._decord = _DecordVR(video_path, ctx=_decord_cpu(0))
        except Exception:
            self._decord = None
            self._open_pyav()

    # -- PyAV backend -----------------------------------------------------

    def _open_pyav(self) -> None:
        import av

        self._container = av.open(self._path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._len = (
            int(self._stream.frames)
            if self._stream.frames and self._stream.frames > 0
            else self._count_pyav_frames()
        )
        rate = self._stream.average_rate or self._stream.guessed_rate
        self._avg_fps = float(rate) if rate else 0.0
        self._reset_decoder()

    def _count_pyav_frames(self) -> int:
        import av

        with av.open(self._path) as probe:
            stream = probe.streams.video[0]
            return sum(1 for _ in probe.decode(stream))

    def _reset_decoder(self) -> None:
        """(Re)start the forward decode generator from frame 0."""
        if self._container is not None:
            self._container.close()
        import av

        self._container = av.open(self._path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._decoder = self._container.decode(self._stream)
        self._pos = 0  # index of the next frame the generator will yield

    def _get_pyav(self, i: int) -> _Frame:
        if i < 0 or i >= self._len:
            raise IndexError(f"frame index {i} out of range (0..{self._len - 1})")
        if i < self._pos:
            self._reset_decoder()
        frame = None
        while self._pos <= i:
            frame = next(self._decoder)
            self._pos += 1
        return _Frame(frame.to_ndarray(format="rgb24"))

    # -- decord-compatible API -------------------------------------------

    def __len__(self) -> int:
        if self._decord is not None:
            return len(self._decord)
        return self._len

    def __getitem__(self, i: int):
        if self._decord is not None:
            return self._decord[i]
        return self._get_pyav(int(i))

    def get_avg_fps(self) -> float:
        """Average frames-per-second (decord-compatible)."""
        if self._decord is not None:
            return float(self._decord.get_avg_fps())
        return self._avg_fps
