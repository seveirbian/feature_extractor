#!/usr/bin/env python3
"""Batch feature extraction entry point for egoWM Phase 1.

Extracts DINO, Depth, and Pose features from videos and stores them in HDF5.

Usage:
    python scripts/extract_features.py \
        --data_root data/openego/videos \
        --output_root data/features \
        --num_samples 10 \
        --device cuda \
        --branches dino,depth,pose

    # Resume: only extract missing videos
    python scripts/extract_features.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from feature_extractor.storage import FeatureStore
from feature_extractor.extractors import DINOExtractor, DepthExtractor, PoseExtractor


# ----------------------------------------------------------------------
# Supported video extensions
# ----------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".MP4", ".AVI"}


def find_videos(root: str) -> list[str]:
    """Recursively find all video files under root."""
    root_path = Path(root)
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend([str(p) for p in root_path.rglob(f"*{ext}")])
    return sorted(videos)


def video_id_from_path(path: str, stem_only: bool = False) -> str:
    """Derive a unique video ID from the file path."""
    stem = Path(path).stem
    if stem_only:
        return stem
    parent = Path(path).parent.name
    return f"{parent}_{stem}"


def create_annotation(
    video_path: str,
    video_id: str,
    fps: float = 30.0,
) -> dict:
    """Create a minimal annotation stub for a video (no labels)."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
    except Exception:
        num_frames = 0

    return {
        "video_id": video_id,
        "video_path": video_path,
        "fps": fps,
        "num_frames": num_frames,
        "has_state": False,
        "has_action": False,
        "has_weak_action": False,
        "has_strong_action": False,
        "has_depth": True,
        "has_pose": True,
        "state_dim": None,
        "action_dim": None,
        "state": None,
        "action": None,
        "source": "openego_auto",
    }


def sample_frame_indices(video_path: str, frames_per_video: int | None) -> list[int]:
    """Choose original video frame indices once so every branch aligns."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if frames_per_video is None or frames_per_video <= 0 or frames_per_video >= total_frames:
        return list(range(total_frames))
    step = max(1, total_frames // frames_per_video)
    return list(range(0, total_frames, step))[:frames_per_video]


# ----------------------------------------------------------------------
# Main extraction
# ----------------------------------------------------------------------


def extract_single_video(
    video_path: str,
    video_id: str,
    extractor_dino: DINOExtractor | None,
    extractor_depth: DepthExtractor | None,
    extractor_pose: PoseExtractor | None,
    store: FeatureStore,
    frame_indices: list[int],
    future_horizon: int = 4,
    resume: bool = False,
) -> bool:
    """Extract all features for one video. Returns True if successful."""
    if resume and store.exists(video_id):
        print(f"  [SKIP] {video_id} already extracted")
        return True

    try:
        # Extract DINO
        dino_feats = None
        if extractor_dino is not None:
            dino_feats = extractor_dino.extract_video(video_path, frame_indices=frame_indices)
            store.write_dino(video_id, dino_feats, frame_indices=frame_indices)

        # Extract Depth
        depth_inv = None
        if extractor_depth is not None:
            depth_inv = extractor_depth.extract_video(video_path, frame_indices=frame_indices)
            store.write_depth(video_id, depth_inv, frame_indices=frame_indices)

        # Extract Pose
        pose_se3 = None
        if extractor_pose is not None:
            pose_se3 = extractor_pose.extract_video(video_path, frame_indices=frame_indices)
            store.write_pose(video_id, pose_se3, frame_indices=frame_indices)

        print(f"  [OK] {video_id}: DINO={dino_feats.shape if dino_feats is not None else 'N/A'}, "
              f"Depth={depth_inv.shape if depth_inv is not None else 'N/A'}, "
              f"Pose={pose_se3.shape if pose_se3 is not None else 'N/A'}")
        return True

    except Exception as e:
        print(f"  [ERROR] {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="EgoWM feature extraction")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for HDF5 feature files")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max number of videos to process (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for extraction (cuda/cpu)")
    parser.add_argument("--branches", type=str, default="dino,depth,pose",
                        help="Comma-separated branches to extract")
    parser.add_argument("--resume", action="store_true",
                        help="Skip videos already extracted")
    parser.add_argument("--future_horizon", type=int, default=4,
                        help="Number of future frames for prediction")
    parser.add_argument("--frames_per_video", type=int, default=120,
                        help="Number of original video frames to sample per video; <=0 means all")
    parser.add_argument("--dino_model", type=str, default="dinov3_vits16plus")
    parser.add_argument("--depth_mode", type=str, default="dino_attention",
                        choices=["video_depth_anything", "vda", "da3", "depth_pro", "dino_attention"])
    parser.add_argument("--vda_input_size", type=int, default=224,
                        help="Video Depth Anything input size. Lower this to avoid GPU OOM.")
    parser.add_argument("--annotation_dir", type=str, default=None,
                        help="Directory to save annotation JSONs")
    parser.add_argument("--id_from_stem", action="store_true",
                        help="Use video filename stem as video_id instead of parent_stem")
    parser.add_argument("--assets_root", type=str, default=None,
                        help="Root dir containing third_party/ model assets "
                             "(default: env FEATURE_EXTRACTOR_ASSETS or package root)")

    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but unavailable, falling back to CPU")
        device = "cpu"

    branches = [b.strip() for b in args.branches.split(",")]

    print(f"=== EgoWM Feature Extraction ===")
    print(f"  Data root:   {args.data_root}")
    print(f"  Output root: {args.output_root}")
    print(f"  Device:      {device}")
    print(f"  Branches:    {branches}")
    print(f"  Resume:      {args.resume}")
    print()

    # Find videos
    videos = find_videos(args.data_root)
    if not videos:
        print(f"[ERROR] No video files found under {args.data_root}")
        sys.exit(1)

    if args.num_samples is not None:
        videos = videos[: args.num_samples]

    print(f"Found {len(videos)} videos to process\n")

    # Initialize store
    store = FeatureStore(args.output_root)

    # Initialize extractors
    extractor_dino = None
    extractor_depth = None
    extractor_pose = None

    if "dino" in branches:
        print("Loading DINO extractor...")
        extractor_dino = DINOExtractor(model_name=args.dino_model, device=device,
                                       assets_root=args.assets_root)

    if "depth" in branches:
        print("Loading Depth extractor...")
        extractor_depth = DepthExtractor(
            mode=args.depth_mode,
            device=device,
            dino_extractor=extractor_dino,
            vda_input_size=args.vda_input_size,
            assets_root=args.assets_root,
        )

    if "pose" in branches:
        print("Loading Pose extractor...")
        extractor_pose = PoseExtractor(device=device, assets_root=args.assets_root)

    # Create annotation output dir
    ann_dir = args.annotation_dir
    if ann_dir is None:
        ann_dir = os.path.join(args.output_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # Process videos
    successes = 0
    failures = 0

    for video_path in tqdm(videos, desc="Extracting features"):
        video_id = video_id_from_path(video_path, stem_only=args.id_from_stem)
        frame_indices = sample_frame_indices(video_path, args.frames_per_video)

        ok = extract_single_video(
            video_path=video_path,
            video_id=video_id,
            extractor_dino=extractor_dino,
            extractor_depth=extractor_depth,
            extractor_pose=extractor_pose,
            store=store,
            frame_indices=frame_indices,
            future_horizon=args.future_horizon,
            resume=args.resume,
        )

        if ok:
            successes += 1
            ann_path = os.path.join(ann_dir, f"{video_id}.json")
            if not os.path.exists(ann_path):
                ann = create_annotation(video_path, video_id)
                with open(ann_path, "w") as f:
                    json.dump(ann, f, indent=2)
        else:
            failures += 1

    print(f"\n=== Done ===")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")
    print(f"  Output:    {args.output_root}")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
