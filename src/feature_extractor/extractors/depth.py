"""Depth feature extractor using Video Depth Anything + Depth Pro.

Outputs normalized inverse depth d_inv = normalize(1 / clamp(z, z_min, z_max))
for each pixel. Also extracts metric depth on keyframes via Depth Pro.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from feature_extractor.assets import resolve_assets_root

_WARNED_FALLBACKS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


# ----------------------------------------------------------------------
# Depth Anything v3 model loading  (pip package: depth_anything_3)
# ----------------------------------------------------------------------
def _load_depth_anything_v3(device: torch.device):
    """Load Depth Anything V3 model from pip package."""
    try:
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3(model_name="da3-large")
        model = model.to(device)
        model.eval()
        print("[DepthExtractor] Loaded DA3 (pip package)")
        return model
    except Exception as e:
        _warn_once(
            "depth_da3_load_failed",
            f"Depth fallback: failed to load Depth Anything V3; another depth source may be used. Error: {e}",
        )
        print(f"[DepthExtractor] Failed to load DA3: {e}")
        return None


# ----------------------------------------------------------------------
# Video Depth Anything model loading  (local repo)
# ----------------------------------------------------------------------
def _load_video_depth_anything(
    device: torch.device,
    encoder: str = "vitl",
    metric: bool = False,
    assets_root=None,
):
    """Load Video Depth Anything from local repo."""
    try:
        import sys

        vda_path = resolve_assets_root(assets_root) / "third_party" / "Video-Depth-Anything"
        ckpt_prefix = "metric_video_depth_anything" if metric else "video_depth_anything"
        ckpt_path = vda_path / "checkpoints" / f"{ckpt_prefix}_{encoder}.pth"
        print("ckpt path ", ckpt_path)
        if not ckpt_path.exists():
            _warn_once(
                f"depth_vda_checkpoint_missing:{encoder}:{metric}",
                f"Depth fallback: Video Depth Anything checkpoint not found: {ckpt_path}",
            )
            print(f"[DepthExtractor] Video Depth Anything checkpoint not found: {ckpt_path}")
            return None
        if str(vda_path) not in sys.path:
            sys.path.insert(0, str(vda_path))

        from video_depth_anything.video_depth import VideoDepthAnything

        model_cfgs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        model = VideoDepthAnything(**model_cfgs[encoder], metric=metric)
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model = model.to(device)
        model.eval()
        print(f"[DepthExtractor] Loaded Video Depth Anything ({encoder})")
        return model
    except Exception as e:
        _warn_once(
            f"depth_vda_load_failed:{encoder}:{metric}",
            f"Depth fallback: failed to load Video Depth Anything; another depth source may be used. Error: {e}",
        )
        print(f"[DepthExtractor] Failed to load Video DA: {e}")
        return None


# ----------------------------------------------------------------------
# Depth Pro (metric depth on keyframes)
# ----------------------------------------------------------------------
def _load_depth_pro(device: torch.device, assets_root=None):
    """Load Depth Pro model."""
    try:
        import sys
        from dataclasses import replace

        project_root = resolve_assets_root(assets_root)
        local_src = project_root / "third_party" / "ml-depth-pro" / "src"
        checkpoint_path = project_root / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
        if local_src.exists() and str(local_src) not in sys.path:
            sys.path.insert(0, str(local_src))
        try:
            import depth_pro

            config = replace(
                depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
                checkpoint_uri=str(checkpoint_path),
            )
            precision = torch.float16 if device.type == "cuda" else torch.float32
            model, transform = depth_pro.create_model_and_transforms(
                config=config,
                device=device,
                precision=precision,
            )
            model = model.to(device)
            model.eval()
            return model, transform
        except Exception as e:
            _warn_once(
                "depth_pro_load_failed",
                f"Depth fallback: failed to load Depth Pro metric model; metric correction/keyframe depth is unavailable. Error: {e}",
            )
            print(f"[DepthExtractor] Failed to load Depth Pro: {e}")
            return None, None
    except Exception as e:
        _warn_once(
            "depth_pro_loader_init_failed",
            f"Depth fallback: failed to initialize Depth Pro loader; metric correction/keyframe depth is unavailable. Error: {e}",
        )
        print(f"[DepthExtractor] Failed to initialize Depth Pro loader: {e}")
        return None, None


class DepthExtractor:
    """Video Depth Anything + Depth Pro depth extractor.

    Combines temporal Video Depth Anything (per-pixel relative depth)
    with Depth Pro metric depth (on keyframes) to produce:
    - Inverse normalized depth per pixel: (H, W, 1)
    - Stored as uint16 for compression efficiency

    Args:
        mode: Which depth source to use.
            - "video_depth_anything" / "vda": local Video Depth Anything
            - "da3": Depth Anything v3 if available
            - "depth_pro": Depth Pro (metric, keyframes only)
        device: torch device.
        z_min / z_max: Clamping range for metric depth (meters).
        keyframe_interval: Extract Depth Pro every N frames (default 30).
    """

    def __init__(
        self,
        mode: Literal["video_depth_anything", "vda", "da3", "depth_pro"] = "video_depth_anything",
        device: Optional[str] = None,
        z_min: float = 0.1,
        z_max: float = 100.0,
        keyframe_interval: int = 30,
        vda_encoder: Literal["vits", "vitb", "vitl"] = "vitl",
        vda_metric: bool = False,
        vda_input_size: int = 518,
        input_color: Literal["rgb", "bgr"] = "rgb",
        assets_root: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.mode = "video_depth_anything" if mode == "vda" else mode
        self.z_min = z_min
        self.z_max = z_max
        self.keyframe_interval = keyframe_interval
        self.vda_encoder = vda_encoder
        self.vda_metric = vda_metric
        self.vda_input_size = vda_input_size
        self.input_color = input_color
        self.assets_root = assets_root

        self.model = self._load_model()
        self.depth_pro_model = None
        self.depth_pro_transform = None
        # VDA temporal buffer: store previous N frames for sliding-window inference
        self._vda_buffer = []  # list of (frame_tensor, original_hw)
        self._vda_buffer_max = 31  # keep enough for INFER_LEN=32 overlap
        if self.mode in ("da3", "video_depth_anything", "depth_pro"):
            self.depth_pro_model, self.depth_pro_transform = _load_depth_pro(self.device, self.assets_root)
        if self.mode == "video_depth_anything":
            if self.model is None:
                raise RuntimeError("Video Depth Anything failed to load; refusing to fall back for depth extraction.")
            if self.depth_pro_model is None or self.depth_pro_transform is None:
                raise RuntimeError("Depth Pro failed to load; video depth mode requires Depth Pro keyframe metric correction.")

    def _load_model(self):
        if self.mode == "da3":
            model = _load_depth_anything_v3(self.device)
            if model is None:
                raise RuntimeError("Depth Anything V3 (da3) 不可用;请改用 --depth_mode video_depth_anything。")
            model = model.to(self.device)
            model.eval()
            return model
        elif self.mode == "video_depth_anything":
            model = _load_video_depth_anything(
                self.device,
                encoder=self.vda_encoder,
                metric=self.vda_metric,
                assets_root=self.assets_root,
            )
            if model is None:
                raise RuntimeError("Video Depth Anything could not be loaded from local repo/checkpoint.")
            else:
                model = model.to(self.device)
                model.eval()
                return model
        return None

    @staticmethod
    def _fit_depth_affine(
        source_depth: np.ndarray,
        target_depth: np.ndarray,
        max_points: int = 200_000,
    ) -> tuple[float, float]:
        """Fit affine map source -> target using valid metric pixels."""
        src = np.asarray(source_depth, dtype=np.float32).reshape(-1)
        tgt = np.asarray(target_depth, dtype=np.float32).reshape(-1)
        mask = np.isfinite(src) & np.isfinite(tgt) & (src > 1e-6) & (tgt > 1e-6)
        if mask.sum() < 1024:
            raise RuntimeError("Too few valid pixels to align VDA depth with Depth Pro.")

        src = src[mask]
        tgt = tgt[mask]

        if src.size > max_points:
            step = max(1, src.size // max_points)
            src = src[::step]
            tgt = tgt[::step]

        src_mean = float(src.mean())
        tgt_mean = float(tgt.mean())
        src_centered = src - src_mean
        denom = float(np.dot(src_centered, src_centered))
        if denom < 1e-8:
            scale = 1.0
            shift = max(0.0, tgt_mean - src_mean)
            return scale, shift

        scale = float(np.dot(src_centered, tgt - tgt_mean) / denom)
        shift = float(tgt_mean - scale * src_mean)
        return scale, shift

    @staticmethod
    def _interpolate_keyframe_params(
        params: list[tuple[int, float, float]],
        num_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate affine parameters across frames."""
        if not params:
            raise RuntimeError("No keyframe depth-alignment parameters available.")

        frame_ids = np.asarray([p[0] for p in params], dtype=np.float32)
        scales = np.asarray([p[1] for p in params], dtype=np.float32)
        shifts = np.asarray([p[2] for p in params], dtype=np.float32)
        query = np.arange(num_frames, dtype=np.float32)

        interp_scales = np.interp(query, frame_ids, scales).astype(np.float32)
        interp_shifts = np.interp(query, frame_ids, shifts).astype(np.float32)
        return interp_scales, interp_shifts

    def _metricize_vda_depth_sequence(
        self,
        relative_depths: np.ndarray,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> np.ndarray:
        """Calibrate temporally consistent VDA depth with sparse metric Depth Pro keyframes."""
        if self.depth_pro_model is None or self.depth_pro_transform is None:
            raise RuntimeError("Depth Pro is required to metricize Video Depth Anything outputs.")

        num_frames = len(frames)
        if num_frames == 0:
            return np.empty((0, 0, 0), dtype=np.float32)

        keyframe_rows = [0]
        last_source_idx = frame_indices[0]
        for row, source_idx in enumerate(frame_indices[1:], start=1):
            if source_idx - last_source_idx >= self.keyframe_interval:
                keyframe_rows.append(row)
                last_source_idx = source_idx
        if keyframe_rows[-1] != num_frames - 1:
            keyframe_rows.append(num_frames - 1)

        params: list[tuple[int, float, float]] = []
        for row in keyframe_rows:
            metric_depth = self._extract_depth_pro(frames[row])
            scale, shift = self._fit_depth_affine(relative_depths[row], metric_depth)
            params.append((row, scale, shift))

        interp_scales, interp_shifts = self._interpolate_keyframe_params(params, num_frames)
        metric_depths = []
        for row in range(num_frames):
            depth = relative_depths[row] * interp_scales[row] + interp_shifts[row]
            depth = np.clip(depth, self.z_min, self.z_max).astype(np.float32)
            metric_depths.append(depth)
        return np.stack(metric_depths, axis=0)

    @torch.no_grad()
    def extract_frame(self, image: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """Extract inverse depth for a single frame.

        Args:
            image: (H, W, 3) uint8 BGR/RGB.
            frame_idx: Frame index for keyframe detection.

        Returns:
            inv_depth: (H, W, 1) float32, normalized to [0, 1].
        """
        h, w = image.shape[:2]

        if self.mode == "da3" and self.model is not None:
            depth_map = self._extract_da3(image)
        elif self.mode == "video_depth_anything" and self.model is not None:
            depth_map = self._extract_vda(image)
        elif self.mode == "depth_pro" and self.depth_pro_model is not None:
            depth_map = self._extract_depth_pro(image)
        else:
            raise RuntimeError(
                f"depth mode={self.mode!r} 没有可用模型;支持 video_depth_anything / da3 / depth_pro。"
            )

        inv_depth = self._to_inverse_depth(depth_map)
        return inv_depth[..., None].astype(np.float32)

    def _to_inverse_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert metric or relative depth to normalized inverse depth."""
        z = np.clip(depth_map, self.z_min, self.z_max)
        inv_depth = 1.0 / z

        # Normalize to [0, 1]
        inv_min = 1.0 / self.z_max
        inv_max = 1.0 / self.z_min
        inv_depth = (inv_depth - inv_min) / (inv_max - inv_min + 1e-8)
        inv_depth = np.clip(inv_depth, 0.0, 1.0)

        return inv_depth.astype(np.float32)

    def _extract_da3(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything V3 on a single frame."""
        import cv2

        if image.dtype == np.uint8:
            image_uint8 = image
        else:
            image_uint8 = (image * 255).astype(np.uint8)

        if self.input_color == "bgr" and image_uint8.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_uint8

        prediction = self.model.inference([image_rgb])

        # prediction.depth is (N, H, W) numpy array
        depth = prediction.depth[0]  # take first frame

        # Resize to original resolution
        depth = cv2.resize(
            depth,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return depth.astype(np.float32)

    def _extract_vda(self, image: np.ndarray) -> np.ndarray:
        """Run Video Depth Anything on a single frame using temporal buffer.

        Uses a sliding window: keeps previous frames in a buffer, runs
        inference on a window of 32 frames, and returns the depth for the
        current (last) frame.
        """
        import cv2
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import Compose

        INFER_LEN = 32
        OVERLAP = 10
        KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
        INTERP_LEN = 8

        # Precompute transform (cached per h/w ratio — rebuild only on change)
        h, w = image.shape[:2]
        ratio = max(h, w) / min(h, w)
        input_size = 518
        if ratio > 1.78:
            input_size = round(int(input_size * 1.777 / ratio) / 14) * 14
            input_size = max(input_size, 14)

        # Build transform pipeline
        from video_depth_anything.util.transform import (
            Resize, NormalizeImage, PrepareForNet,
        )
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Convert image to normalized tensor and add to buffer
        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        else:
            img_f = image.astype(np.float32)

        img_rgb = img_f[..., ::-1] if self.input_color == "bgr" else img_f
        tensor = torch.from_numpy(transform({"image": img_rgb})["image"]).unsqueeze(0)
        self._vda_buffer.append((tensor, (h, w)))

        # Need at least INFER_LEN frames to run inference
        if len(self._vda_buffer) < INFER_LEN:
            return np.ones((h, w), dtype=np.float32) * 5.0

        # Take window of INFER_LEN frames
        window = self._vda_buffer[-INFER_LEN:]
        window_tensors = [t for (t, _) in window]

        # Concatenate along time dim → (1, T, C, H', W')
        batch = torch.cat(window_tensors, dim=0).unsqueeze(0).to(self.device)

        # Run model (FP16 when on CUDA)
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                                enabled=self.device.type == "cuda"):
                depth_raw = self.model.forward(batch)  # (1, T, H', W')

        depth_raw = depth_raw.float()
        depth_raw = F.interpolate(
            depth_raw.flatten(0, 1).unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )  # (T, 1, H, W)

        # Return depth for the LAST (current) frame
        return depth_raw[-1, 0].cpu().numpy().astype(np.float32)

    def _extract_depth_pro(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Pro on a single frame for metric depth."""
        from PIL import Image

        if self.depth_pro_transform is None:
            return np.ones_like(image[..., 0], dtype=np.float32) * 5.0

        if image.dtype != np.uint8:
            image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image

        if self.input_color == "bgr" and image_uint8.shape[2] == 3:
            image_rgb = image_uint8[..., ::-1]
        else:
            image_rgb = image_uint8

        pil_img = Image.fromarray(image_rgb)
        x = self.depth_pro_transform(pil_img).to(self.device)

        # Depth Pro infers metric depth; use default focal length if unknown
        result = self.depth_pro_model.infer(x)  # returns dict with "depth" key
        if isinstance(result, dict):
            depth = result["depth"]
        else:
            depth = result

        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        depth = np.squeeze(depth)

        # Resize to original resolution
        import cv2
        depth = cv2.resize(
            depth,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return depth.astype(np.float32)

    @torch.no_grad()
    def extract_video(
        self,
        video_path: str,
        frame_indices: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Extract inverse depth for all specified frames.

        Args:
            video_path: Path to video file.
            frame_indices: Frame indices to process.

        Returns:
            inv_depths: (T, H, W, 1) float32.
        """
        from ..video_io import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if frame_indices is None:
            frame_indices = list(range(total_frames))
            if len(frame_indices) > 1000:
                frame_indices = frame_indices[:: max(1, len(frame_indices) // 1000)]

        if self.mode == "video_depth_anything" and self.model is not None:
            frames = []
            valid_indices = []
            for i in frame_indices:
                if i >= total_frames:
                    break
                frames.append(vr[i].asnumpy())
                valid_indices.append(i)
            if not frames:
                return np.empty((0, 0, 0, 1), dtype=np.float32)
            depths, _ = self.model.infer_video_depth(
                np.stack(frames, axis=0),
                target_fps=30,
                input_size=self.vda_input_size,
                device=self.device.type,
                fp32=self.device.type == "cpu",
            )
            depths = np.asarray(depths[: len(valid_indices)], dtype=np.float32)
            if self.vda_metric:
                metric_depths = np.clip(depths, self.z_min, self.z_max)
            else:
                metric_depths = self._metricize_vda_depth_sequence(depths, frames, valid_indices)
            inv_depths = [self._to_inverse_depth(depth)[..., None] for depth in metric_depths]
            return np.stack(inv_depths, axis=0).astype(np.float32)

        all_depths = []
        for i in tqdm(frame_indices, desc=f"Depth [{Path(video_path).name}]"):
            if i >= total_frames:
                break
            frame = vr[i].asnumpy()
            inv_d = self.extract_frame(frame, frame_idx=i)
            all_depths.append(inv_d)

        return np.stack(all_depths, axis=0).astype(np.float32)  # (T, H, W, 1)

    def save_depth(self, inv_depths: np.ndarray, output_path: str) -> None:
        """Save inverse depth as uint16 HDF5 (compression-friendly)."""
        import h5py

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to uint16 for storage efficiency
        inv_depths_uint16 = (np.clip(inv_depths, 0, 1) * 65535).astype(np.uint16)

        with h5py.File(output_path, "w", libver="v1_10") as f:
            dataset = f.create_dataset(
                "inv_depth",
                data=inv_depths_uint16,
                dtype=np.uint16,
                compression="lzf",
                chunks=(1, inv_depths.shape[1], inv_depths.shape[2], 1),
            )
            dataset.attrs["mode"] = self.mode
            dataset.attrs["scale"] = 65535.0

    def __repr__(self) -> str:
        return f"DepthExtractor(mode={self.mode}, device={self.device})"
