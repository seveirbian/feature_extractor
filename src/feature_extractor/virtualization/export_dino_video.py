#!/usr/bin/env python3
"""Export full-video DINO feature visualization as an MP4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from feature_extractor.video_io import VideoReader, cpu
from feature_extractor.extractors.dino import DINOExtractor


def _fit_pca_rgb_basis(
    patch_tokens: np.ndarray,
    frame_stride: int = 8,
    token_stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    sampled = patch_tokens[::frame_stride, ::token_stride]
    flat = sampled.reshape(-1, sampled.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    centered = flat - mean
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:3].T.astype(np.float32)
    return mean.astype(np.float32), basis


def _project_pca_rgb(
    tokens: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    proj = (tokens - mean) @ basis
    proj = proj.astype(np.float32)
    lo = np.percentile(proj, 2.0, axis=0)
    hi = np.percentile(proj, 98.0, axis=0)
    hi = np.where(hi <= lo + 1e-8, lo + 1.0, hi)
    proj = np.clip((proj - lo) / (hi - lo), 0.0, 1.0)
    return proj


def _project_cosine_cls(tokens: np.ndarray, cls_tokens: np.ndarray) -> np.ndarray:
    cls = cls_tokens[:, None, :]
    sim = np.sum(tokens * cls, axis=-1)
    lo = np.percentile(sim, 2.0)
    hi = np.percentile(sim, 98.0)
    if hi <= lo + 1e-8:
        hi = lo + 1.0
    sim = np.clip((sim - lo) / (hi - lo), 0.0, 1.0)
    return sim.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export DINO feature visualization video")
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--output_mp4", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dino_model", type=str, default="dinov3_vits16plus")
    parser.add_argument("--render_scale", type=float, default=1.0)
    parser.add_argument("--panel_mode", type=str, default="rgb_pca_cls", choices=["rgb_pca", "rgb_pca_cls"])
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧(默认 0)")
    parser.add_argument("--frame_stride", type=int, default=1, help="抽帧步长(默认 1)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最多处理多少帧(默认全部);长视频务必设置,否则会逐帧跑完整段")
    args = parser.parse_args()

    video_path = args.video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_mp4 = args.output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_mp4.with_suffix(".json") if args.output_json is None else args.output_json.resolve()

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    frame0 = vr[0].asnumpy()
    orig_h, orig_w = int(frame0.shape[0]), int(frame0.shape[1])
    render_w = max(2, int(round(orig_w * args.render_scale)))
    render_h = max(2, int(round(orig_h * args.render_scale)))

    frame_indices = list(range(args.start_frame, total_frames, max(1, args.frame_stride)))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise RuntimeError(
            f"No frames selected (start={args.start_frame}, stride={args.frame_stride}, "
            f"max={args.max_frames}, total={total_frames})"
        )
    num_proc = len(frame_indices)

    print(f"video: {video_path}")
    print(f"frames: {total_frames} (processing {num_proc}: "
          f"start={args.start_frame} stride={args.frame_stride} max={args.max_frames})")
    print(f"fps: {fps:.3f}")
    print(f"resolution: {orig_w}x{orig_h}")
    print(f"render_resolution: {render_w}x{render_h}")
    print(f"dino_model: {args.dino_model}")
    print(f"panel_mode: {args.panel_mode}")

    extractor = DINOExtractor(model_name=args.dino_model, device=args.device)
    features = extractor.extract_video(str(video_path), frame_indices=frame_indices)
    if len(features) != num_proc:
        raise RuntimeError(f"Feature count mismatch: got {len(features)} for {num_proc} frames")

    cls_tokens = features[:, 0, :]
    patch_tokens = features[:, 1:, :]
    num_patches = patch_tokens.shape[1]
    grid_side = int(round(math.sqrt(num_patches)))
    if grid_side * grid_side != num_patches:
        raise RuntimeError(f"Patch token count is not a square grid: {num_patches}")

    mean, basis = _fit_pca_rgb_basis(patch_tokens)
    pca_rgb = _project_pca_rgb(patch_tokens.reshape(-1, patch_tokens.shape[-1]), mean, basis)
    pca_rgb = pca_rgb.reshape(num_proc, grid_side, grid_side, 3)

    cls_sim = _project_cosine_cls(patch_tokens, cls_tokens).reshape(num_proc, grid_side, grid_side)

    if args.panel_mode == "rgb_pca_cls":
        panel_w = render_w * 3
    else:
        panel_w = render_w * 2

    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_w, render_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_mp4}")

    try:
        for i, src_idx in enumerate(frame_indices):
            rgb = vr[src_idx].asnumpy()
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if (render_w, render_h) != (orig_w, orig_h):
                rgb_bgr = cv2.resize(rgb_bgr, (render_w, render_h), interpolation=cv2.INTER_AREA)

            feat_rgb = (pca_rgb[i] * 255.0).round().astype(np.uint8)
            feat_rgb = cv2.resize(feat_rgb, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
            feat_bgr = cv2.cvtColor(feat_rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(feat_bgr, "DINO PCA RGB", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_bgr, f"frame={src_idx:05d} t={src_idx / fps:06.2f}s", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            panels = [rgb_bgr, feat_bgr]
            if args.panel_mode == "rgb_pca_cls":
                sim = (cls_sim[i] * 255.0).round().astype(np.uint8)
                sim = cv2.resize(sim, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
                sim_vis = cv2.applyColorMap(sim, cv2.COLORMAP_TURBO)
                cv2.putText(sim_vis, "CLS Similarity", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                panels.append(sim_vis)

            writer.write(np.concatenate(panels, axis=1))
    finally:
        writer.release()

    report = {
        "video_path": str(video_path),
        "output_mp4": str(output_mp4),
        "frames": num_proc,
        "total_frames": total_frames,
        "start_frame": args.start_frame,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "fps": fps,
        "original_resolution": [orig_h, orig_w],
        "render_resolution": [render_h, render_w],
        "dino_model": args.dino_model,
        "panel_mode": args.panel_mode,
        "grid_side": grid_side,
        "feature_shape": list(features.shape),
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"saved mp4: {output_mp4}")
    print(f"saved report: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
