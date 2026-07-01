#!/usr/bin/env python3
"""探测 32GB 显存上 VGGT 单窗口能塞多少帧,给流式 --pose_window 定一个合理初值。

用法: CUDA_VISIBLE_DEVICES=1 uv run python scripts/probe_pose_window.py <video.mp4>
对递增的窗口帧数跑一次真实 VGGT 推理(走 PoseExtractor._window_extrinsics),
记录成功/OOM 与峰值显存,输出建议的 pose_window / pose_overlap。
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from feature_extractor.extractors.pose import PoseExtractor
from feature_extractor.video_io import VideoReader, cpu

WINDOWS = [24, 32, 48, 64, 80, 96, 112, 128]


def main():
    video = sys.argv[1]
    vr = VideoReader(video, ctx=cpu(0))
    total = len(vr)
    need = max(WINDOWS)
    idx = np.linspace(0, total - 1, min(need, total)).astype(int)
    frames_all = [vr[int(i)].asnumpy() for i in idx]
    print(f"video frames={total}, decoded {len(frames_all)} probe frames")

    pose = PoseExtractor(device="cuda")
    last_ok = 0
    peak_ok = 0.0
    for w in WINDOWS:
        if w > len(frames_all):
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _ = pose._window_extrinsics(frames_all[:w])
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  window={w:4d}  OK    peak={peak:5.1f} GiB")
            last_ok, peak_ok = w, peak
        except torch.cuda.OutOfMemoryError:
            print(f"  window={w:4d}  OOM")
            torch.cuda.empty_cache()
            break

    print("\n=== 建议 ===")
    if last_ok == 0:
        print("  连最小窗口都 OOM,检查显存占用")
        return
    # 留 ~20% 余量作为默认
    rec = max(16, int(last_ok * 0.8) // 4 * 4)
    ov = max(4, rec // 4)
    print(f"  单窗口实测上限 ≈ {last_ok} 帧 (峰值 {peak_ok:.1f} GiB)")
    print(f"  建议 --pose_window {rec}  --pose_overlap {ov}   (窗口留 ~20% 余量,重叠约 1/4)")


if __name__ == "__main__":
    main()
