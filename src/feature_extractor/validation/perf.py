"""真实数据上的性能基准:吞吐、峰值显存、规模扩展。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PerfRecord:
    video_id: str
    branch: str          # 分支名,或 "model_load" / "decode"
    frames: int
    seconds: float
    fps: Optional[float] = None
    peak_mem_mb: Optional[float] = None
    note: str = ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return None


def measure_branch(extractor, video_id: str, video_path: str,
                   frame_indices: list[int], branch: str,
                   device: torch.device) -> PerfRecord:
    """计量单分支在给定帧上的耗时、FPS、峰值显存(含一次预热)。

    计时包含 ``extract_video`` 内部的解码开销;若要单独看推理耗时,可减去
    :func:`measure_decode` 在同一组帧上的结果。``frames`` 取输出特征数(提取器
    可能跳帧),与 :func:`measure_decode` 以请求帧数计不同,FPS 口径据此略有差异。
    """
    # 预热(前 2 帧),不计时
    warm = frame_indices[:2] if len(frame_indices) >= 2 else frame_indices
    extractor.extract_video(video_path, frame_indices=warm)
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    feats = extractor.extract_video(video_path, frame_indices=frame_indices)
    _sync(device)
    sec = time.perf_counter() - t0
    n = len(feats)
    return PerfRecord(video_id, branch, n, sec,
                      fps=(n / sec if sec > 0 else None),
                      peak_mem_mb=_peak_mb(device))


def measure_decode(video_id: str, video_path: str, frame_indices: list[int]) -> PerfRecord:
    """单独计量解码耗时(读取指定帧),用于解码 vs 推理拆分。"""
    from feature_extractor.video_io import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    try:
        t0 = time.perf_counter()
        for i in frame_indices:
            _ = vr[int(i)].asnumpy()
        sec = time.perf_counter() - t0
    finally:
        del vr  # 释放底层容器/文件句柄
    n = len(frame_indices)
    return PerfRecord(video_id, "decode", n, sec,
                      fps=(n / sec if sec > 0 else None), note="纯解码")
