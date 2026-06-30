"""DINO feature extractor using a frozen DINOv3 HuggingFace backbone.

Defaults to the local HF-format DINOv3 ViT-S/16+ checkpoint under
``third_party/dinov3/checkpoints/dinov3-vits16plus-hf``.

Extracts patch-level features for future H frames from egocentric video.
Features are L2-normalized and stored as FP32 in HDF5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from feature_extractor.assets import resolve_assets_root


class DINOExtractor:
    """Frozen DINO patch-token feature extractor.

    Extracts patch-level features plus one CLS token. The default DINOv3
    ``dinov3_vits16plus`` output is ``(N_patches + 1, 384)`` at stride 16.

    Args:
        model_name: Model variant. Options:
            - "dinov3_vits16plus" (ViT-S/16+, 384-dim, default)
            - "dinov3_vits16"    (ViT-S/16, 384-dim)
        device: torch device (cuda/cpu). If None, auto-detects.
        compile: Whether to torch.compile() the model (CUDA only).
    """

    MODEL_CONFIGS = {
        "dinov3_vits16plus": {
            "family": "dinov3_hf",
            "img_size": 512,
            "patch_size": 16,
            "embed_dim": 384,
            "weights": "third_party/dinov3/checkpoints/dinov3-vits16plus-hf",
        },
        "dinov3_vits16": {
            "family": "dinov3_hf",
            "img_size": 512,
            "patch_size": 16,
            "embed_dim": 384,
            "weights": "third_party/dinov3/checkpoints/dinov3-vits16-hf",
        },
    }

    MODEL_ALIASES = {
        "facebook/dinov3-vits16plus-pretrain-lvd1689m": "dinov3_vits16plus",
        "facebook/dinov3-vits16plus": "dinov3_vits16plus",
        "dinov3-vits16plus": "dinov3_vits16plus",
        "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
        "facebook/dinov3-vits16": "dinov3_vits16",
        "dinov3-vits16": "dinov3_vits16",
    }

    def __init__(
        self,
        model_name: str = "dinov3_vits16plus",
        device: Optional[str] = None,
        compile: bool = False,
        input_color: Literal["rgb", "bgr"] = "rgb",
        assets_root: Optional[str] = None,
    ):
        self.model_name = self.MODEL_ALIASES.get(model_name, model_name)
        self.input_color = input_color
        self.assets_root = assets_root
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown dino_model {model_name!r}; available: {sorted(self.MODEL_CONFIGS)}"
            )
        cfg = self.MODEL_CONFIGS[self.model_name]
        self.family = cfg["family"]
        self.img_size = cfg["img_size"]
        self.patch_size = cfg["patch_size"]
        self.embed_dim = cfg["embed_dim"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = self._load_model(compile)

    def _load_model(self, compile: bool) -> nn.Module:
        """Load and freeze the configured DINO model."""
        cfg = self.MODEL_CONFIGS[self.model_name]

        model = self._load_dinov3_hf(cfg)

        model = model.to(self.device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        if compile and self.device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("[DINOExtractor] torch.compile enabled")
            except Exception as exc:
                print(f"[DINOExtractor] torch.compile failed: {exc}")

        return model

    def _load_dinov3_hf(self, cfg: dict) -> nn.Module:
        """从本地 HF 格式目录加载 DINOv3(离线 from_pretrained)。"""
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "DINOv3 HF backend 需要 transformers(>=4.56)。请 `uv sync` 或 `pip install transformers`。"
            ) from e
        weights = resolve_assets_root(self.assets_root) / cfg["weights"]
        if not weights.exists():
            raise FileNotFoundError(
                f"DINOv3 HF weights dir not found: {weights}. "
                "请放入 HF 格式权重(config.json + safetensors)。"
            )
        model = AutoModel.from_pretrained(str(weights))
        print(f"[DINOExtractor] Loaded DINOv3 (HF): {self.model_name} ({weights})")
        return model

    @staticmethod
    def _slice_hf_tokens(last_hidden_state: torch.Tensor, num_register: int) -> torch.Tensor:
        """HF DINOv3 输出 [CLS, R×register, patches] → 还原 [CLS, patches](剔除 register)。"""
        cls = last_hidden_state[:, 0:1, :]
        patches = last_hidden_state[:, 1 + num_register:, :]
        return torch.cat([cls, patches], dim=1)

    def _extract_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Return DINO tokens as (1, N, D) = [CLS] + patches(剔除 register)。"""
        batch = image_tensor.unsqueeze(0)
        out = self.model(pixel_values=batch)
        num_register = int(getattr(self.model.config, "num_register_tokens", 0))
        return self._slice_hf_tokens(out.last_hidden_state, num_register)

    @torch.no_grad()
    def extract_frame(self, image: np.ndarray, *, resize: bool = True) -> np.ndarray:
        """Extract features from a single BGR/RGB image (HWC, 0-255 or 0-1).

        Args:
            image: numpy array, shape (H, W, 3), dtype uint8 or float32.

        Returns:
            features: numpy array, shape (N_patches + 1, embed_dim), FP32.
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if self.input_color == "bgr" and image.ndim == 3 and image.shape[2] == 3:
            image = image[..., ::-1]

        image_tensor = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)  # HWC → CHW
            .float()
            .to(self.device)
        )

        if resize:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        else:
            height, width = image_tensor.shape[-2:]
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    f"DINO native extraction requires H/W divisible by patch_size={self.patch_size}, "
                    f"got {(height, width)}."
                )

        mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        tokens = self._extract_tokens(image_tensor)
        features = tokens.squeeze(0).cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms

        return features.astype(np.float32)

    @torch.no_grad()
    def extract_video(
        self,
        video_path: str,
        frame_indices: Optional[list[int]] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Extract features for specified frame indices from a video.

        Args:
            video_path: Path to video file.
            frame_indices: List of frame indices to extract. If None, extracts all.
            batch_size: Frames per batch (only effective on GPU).

        Returns:
            features: numpy array, shape (T, N_patches + 1, embed_dim).
        """
        from ..video_io import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if frame_indices is None:
            frame_indices = list(range(total_frames))
            if len(frame_indices) > 1000:
                frame_indices = frame_indices[:: max(1, len(frame_indices) // 1000)]

        all_features = []
        for i in tqdm(frame_indices, desc=f"DINO [{Path(video_path).name}]"):
            if i >= total_frames:
                break
            frame = vr[i].asnumpy()
            feats = self.extract_frame(frame)
            all_features.append(feats)

        return np.stack(all_features, axis=0).astype(np.float32)

    def extract_video_streaming(
        self,
        video_path: str,
        frame_indices: list[int],
        store,
        video_id: str,
        block_size: int = 1024,
    ) -> None:
        """Stream DINO features to ``store`` block-by-block (bounded memory).

        Writes directly via ``store.write_dino_chunk`` instead of returning one
        large array, so host RAM stays bounded by a single block.
        """
        from ..chunking import iter_frame_blocks

        first = True
        for block_idx, frames, write_offset in iter_frame_blocks(
            video_path, frame_indices, block_size, overlap=0
        ):
            feats = np.stack(
                [self.extract_frame(frames[j]) for j in range(frames.shape[0])], axis=0
            ).astype(np.float32)
            feats = feats[write_offset:]
            out_idx = block_idx[write_offset:]
            if len(feats) == 0:
                continue
            store.write_dino_chunk(video_id, feats, out_idx, reset=first)
            first = False
            del feats
        store.mark_branch_complete(video_id, "dino")

    def get_patch_grid_size(self, h: int, w: int) -> tuple[int, int]:
        """Return the number of patches in height and width for a given image size."""
        return h // self.patch_size, w // self.patch_size

    def save_features(self, features: np.ndarray, output_path: str) -> None:
        """Save features to an HDF5 file with proper chunking."""
        import h5py

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w", libver="v1_10") as f:
            chunks = (1, features.shape[1]) if features.ndim == 2 else (1, features.shape[1], features.shape[2])
            f.create_dataset(
                "features",
                data=features,
                dtype=np.float32,
                compression="lzf",
                chunks=chunks,
            )
            f.attrs["model_name"] = self.model_name
            f.attrs["embed_dim"] = self.embed_dim
            f.attrs["patch_size"] = self.patch_size

    def __repr__(self) -> str:
        return (
            f"DINOExtractor(model={self.model_name}, family={self.family}, "
            f"device={self.device}, embed_dim={self.embed_dim})"
        )
