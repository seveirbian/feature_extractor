#!/usr/bin/env python3
"""运行所有用到 third_party 的分支,记录 sys.modules 中真实加载的文件 → 保留清单。

用法: CUDA_VISIBLE_DEVICES=7 uv run python scripts/trace_thirdparty_usage.py
产物: scripts/keep/<repo>.txt(每行一个相对 third_party/<repo>/ 的文件路径)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = (ROOT / "third_party").resolve()
REPOS = ["dinov3", "VGGT", "Video-Depth-Anything", "ml-depth-pro"]


def exercise_all_branches() -> None:
    import numpy as np  # noqa: F401
    from feature_extractor.extractors.dino import DINOExtractor
    from feature_extractor.extractors.depth import DepthExtractor
    from feature_extractor.extractors.pose import PoseExtractor
    from feature_extractor.validation.synthetic import make_gradient_video

    dev = "cuda"
    with tempfile.TemporaryDirectory() as td:
        vid = str(Path(td) / "g.mp4")
        make_gradient_video(vid, n_frames=6)
        idx = list(range(6))

        # DINO:两个 vits16* 变体都触发(都走本地 dinov3)
        dino = None
        for mn in ("dinov3_vits16plus", "dinov3_vits16"):
            dino = DINOExtractor(model_name=mn, device=dev)
            dino.extract_video(vid, frame_indices=idx)

        # Depth:VDA(内部带 DINOv2 编码器)+ depth_pro
        DepthExtractor(mode="video_depth_anything", device=dev,
                       dino_extractor=dino).extract_video(vid, frame_indices=idx)
        try:
            DepthExtractor(mode="depth_pro", device=dev,
                           dino_extractor=dino).extract_video(vid, frame_indices=idx)
        except Exception as e:  # depth_pro 权重缺失等不应阻断其余追踪
            print(f"[trace] depth_pro 跳过: {e}")

        # Pose:VGGT
        PoseExtractor(device=dev).extract_video(vid, frame_indices=idx)


def collect_used() -> dict[str, set[str]]:
    used = {r: set() for r in REPOS}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            fp = Path(f).resolve()
        except Exception:
            continue
        for r in REPOS:
            base = (TP / r).resolve()
            prefix = str(base) + os.sep
            if str(fp).startswith(prefix):
                used[r].add(str(fp.relative_to(base)))
    return used


def main() -> int:
    exercise_all_branches()
    used = collect_used()
    out_dir = ROOT / "scripts" / "keep"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in REPOS:
        files = sorted(used[r])
        (out_dir / f"{r}.txt").write_text("\n".join(files) + "\n")
        print(f"{r}: {len(files)} files used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
