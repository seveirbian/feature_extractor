"""生成可控的合成视频,供功能验证使用(确定、可移植)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_ramp_video(
    path: str | Path,
    *,
    codec: str = "libx264",
    n_frames: int = 12,
    width: int = 128,
    height: int = 96,
    step: int = 20,
) -> None:
    """写一段每帧填充灰度值 ``i*step`` 的纯色视频。

    解码后每帧均值仍能映射回索引 ``i``,可用于验证帧索引映射。
    """
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((height, width, 3), i * step, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_gradient_video(
    path: str | Path,
    *,
    codec: str = "libx264",
    n_frames: int = 12,
    width: int = 128,
    height: int = 96,
) -> None:
    """写一段每帧带水平灰度渐变的视频,为 depth/dino 提供空间信号。"""
    import av

    grad = np.linspace(0, 255, width, dtype=np.uint8)
    base = np.broadcast_to(grad[None, :, None], (height, width, 3)).copy()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(n_frames):
            frame = av.VideoFrame.from_ndarray(base, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
