"""feature-validate:跑功能不变量 + 性能基准,生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import datetime
import socket
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from feature_extractor import DINOExtractor, DepthExtractor, PoseExtractor, FeatureStore
from feature_extractor.cli import find_videos, sample_frame_indices
from feature_extractor.validation import sanity, perf as perfmod
from feature_extractor.validation.report import render_report
from feature_extractor.validation.synthetic import make_gradient_video


def _env_meta(command: str) -> dict:
    def _safe(fn, default="?"):
        try:
            return fn()
        except Exception:
            return default
    gpu = _safe(lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    commit = _safe(lambda: subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip())
    deps = ", ".join(f"{p}={_safe(lambda p=p: metadata.version(p))}"
                     for p in ("av", "decord", "torch"))
    return {
        "date": datetime.date.today().isoformat(),
        "commit": commit,
        "host": _safe(socket.gethostname),
        "gpu": gpu,
        "cuda": _safe(lambda: torch.version.cuda),
        "torch": torch.__version__,
        "deps": deps,
        "command": command,
    }


def _build_extractors(branches, depth_mode, device: torch.device, assets_root):
    """按需构建各分支提取器。

    注意:当请求 depth 但未请求 dino 时,仍会构建一个 DINO 作为 depth 的骨干
    (传给 ``dino_extractor``),但返回值里的 dino 置为 None——避免把仅作骨干用的
    实例当成独立 dino 分支去跑。
    """
    dino = depth = pose = None
    if "dino" in branches or "depth" in branches:
        dino = DINOExtractor(model_name="dinov3_vits16plus", device=device, assets_root=assets_root)
    if "depth" in branches:
        depth = DepthExtractor(mode=depth_mode, device=device, dino_extractor=dino,
                               vda_input_size=224, assets_root=assets_root)
    if "pose" in branches:
        pose = PoseExtractor(device=device, assets_root=assets_root)
    return dino if "dino" in branches else None, depth, pose


def run_sanity(branches, depth_mode, device, assets_root) -> list[sanity.CheckResult]:
    """在合成视频上跑功能不变量。重模型缺失则标 SKIPPED;
    单分支运行异常转为失败项,不中断整体报告。"""
    checks: list = []
    with tempfile.TemporaryDirectory() as td:
        vid = str(Path(td) / "gradient.mp4")
        make_gradient_video(vid, n_frames=8)
        idx = list(range(8))
        store = FeatureStore(td)
        try:
            dino, depth, pose = _build_extractors(branches, depth_mode, device, assets_root)
        except Exception as e:
            checks.append(sanity.CheckResult("pipeline", "模型加载", "成功",
                                             f"SKIPPED: {e}", False))
            return checks

        indices_by_branch = {}
        # 确定性检查仅覆盖 dino/depth(spec 约定);pose 不做两次比对。
        if dino is not None:  # dino 仅在请求 dino 分支时非 None(depth 骨干不计)
            try:
                f = dino.extract_video(vid, frame_indices=idx)
                checks += sanity.check_dino(f)
                f2 = dino.extract_video(vid, frame_indices=idx)
                checks.append(sanity.check_determinism("dino", f, f2))
                store.write_dino("syn", f, frame_indices=np.array(idx))
                checks.append(sanity.check_exact_roundtrip("dino", f, store.read_dino("syn")))
                indices_by_branch["dino"] = store.read_frame_indices("syn", "dino")
            except Exception as e:
                checks.append(sanity.CheckResult("dino", "运行", "无异常", f"EXCEPTION: {e}", False))
        if depth is not None:
            try:
                d = depth.extract_video(vid, frame_indices=idx)
                checks += sanity.check_depth(d)
                d2 = depth.extract_video(vid, frame_indices=idx)
                checks.append(sanity.check_determinism("depth", d, d2))
                store.write_depth("syn", d, frame_indices=np.array(idx))
                checks.append(sanity.check_depth_roundtrip(d, store.read_depth("syn")))
                indices_by_branch["depth"] = store.read_frame_indices("syn", "depth")
            except Exception as e:
                checks.append(sanity.CheckResult("depth", "运行", "无异常", f"EXCEPTION: {e}", False))
        if pose is not None:
            try:
                pse = pose.extract_video(vid, frame_indices=idx)
                checks += sanity.check_pose(pse)
                store.write_pose("syn", pse, frame_indices=np.array(idx))
                checks.append(sanity.check_exact_roundtrip("pose", pse, store.read_pose("syn")))
                indices_by_branch["pose"] = store.read_frame_indices("syn", "pose")
            except Exception as e:
                checks.append(sanity.CheckResult("pose", "运行", "无异常", f"EXCEPTION: {e}", False))

        checks += sanity.check_alignment(indices_by_branch, idx)
    return checks


def run_perf(data_root, branches, depth_mode, device, perf_frames, sweep,
             assets_root) -> list[perfmod.PerfRecord]:
    """真实数据性能基准。单项测量异常转为带 ERROR 备注的记录,不中断报告。"""
    records: list[perfmod.PerfRecord] = []
    videos = find_videos(data_root)
    if not videos:
        records.append(perfmod.PerfRecord("-", "perf", 0, 0.0, note="SKIPPED: 无视频"))
        return records
    video_path = videos[0]
    video_id = Path(video_path).stem

    t0 = time.perf_counter()
    try:
        dino, depth, pose = _build_extractors(branches, depth_mode, device, assets_root)
    except Exception as e:
        records.append(perfmod.PerfRecord(video_id, "model_load", 0,
                                          time.perf_counter() - t0, note=f"ERROR: {e}"))
        return records
    records.append(perfmod.PerfRecord(video_id, "model_load", 0,
                                      time.perf_counter() - t0, note="一次性加载"))
    extractors = [("dino", dino), ("depth", depth), ("pose", pose)]

    frame_indices = sample_frame_indices(video_path, perf_frames)
    try:
        records.append(perfmod.measure_decode(video_id, video_path, frame_indices))
    except Exception as e:
        records.append(perfmod.PerfRecord(video_id, "decode", 0, 0.0, note=f"ERROR: {e}"))
    for name, ex in extractors:
        if ex is None:
            continue
        try:
            records.append(perfmod.measure_branch(ex, video_id, video_path,
                                                  frame_indices, name, device))
        except Exception as e:
            records.append(perfmod.PerfRecord(video_id, name, 0, 0.0, note=f"ERROR: {e}"))
    # 扩展性扫描:用第一个可用分支(优先级 dino > depth > pose,取 extractors 顺序首个)
    name, ex = next(((n, e) for n, e in extractors if e is not None), (None, None))
    if ex is not None:
        for nf in sweep:
            idx = sample_frame_indices(video_path, nf)
            try:
                rec = perfmod.measure_branch(ex, f"{video_id}@{nf}", video_path, idx, name, device)
                rec.note = f"扫描 frames={nf}"
            except Exception as e:
                rec = perfmod.PerfRecord(f"{video_id}@{nf}", name, 0, 0.0,
                                         note=f"扫描 frames={nf} ERROR: {e}")
            records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="feature_extractor 自验证")
    parser.add_argument("--data_root", type=str, default=None, help="性能基准用真实数据目录")
    parser.add_argument("--report", type=str, default="validation_report.md")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--branches", type=str, default="dino,depth,pose")
    parser.add_argument("--depth_mode", type=str, default="video_depth_anything")
    parser.add_argument("--frames-sweep", type=str, default="16,32,64,128")
    parser.add_argument("--perf-frames", type=int, default=64)
    parser.add_argument("--assets_root", type=str, default=None)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    sweep = [int(x) for x in args.frames_sweep.split(",") if x.strip()]

    checks: list = []
    records: list = []
    if not args.skip_sanity:
        print("=== 功能不变量(合成视频)===")
        checks = run_sanity(branches, args.depth_mode, device, args.assets_root)
    if not args.skip_perf:
        if not args.data_root:
            parser.error("性能基准需要 --data_root(或加 --skip-perf)")
        print("=== 性能基准(真实数据)===")
        records = run_perf(args.data_root, branches, args.depth_mode, device,
                           args.perf_frames, sweep, args.assets_root)

    meta = _env_meta(" ".join(["feature-validate"] + sys.argv[1:]))
    md = render_report(meta, checks, records)
    Path(args.report).write_text(md, encoding="utf-8")
    n_pass = sum(1 for c in checks if c.passed)
    print(f"报告已写入 {args.report}　功能 {n_pass}/{len(checks)} 通过")
    # 作为 CI 门禁:任一功能不变量失败则非零退出
    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
