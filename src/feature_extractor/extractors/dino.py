"""DINO feature extractor using frozen DINOv3 or DINOv2 backbones.

Defaults to the local DINOv3 ViT-S/16+ checkpoint under ``third_party/dinov3``.
DINOv2 remains supported for older feature stores and experiments.

Extracts patch-level features for future H frames from egocentric video.
Features are L2-normalized and stored as FP32 in HDF5.
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


class DINOExtractor:
    """Frozen DINO patch-token feature extractor.

    Extracts patch-level features plus one CLS token. The default DINOv3
    ``dinov3_vits16plus`` output is ``(N_patches + 1, 384)`` at stride 16.

    Args:
        model_name: Model variant. Options:
            - "dinov3_vits16plus" (ViT-S/16+, 384-dim, default)
            - "dinov2_vitl14"  (ViT-L/14, 1024-dim, default)
            - "dinov2_vitb14"  (ViT-B/14, 768-dim)
            - "dinov2_vits14"  (ViT-S/14, 384-dim)
        device: torch device (cuda/cpu). If None, auto-detects.
        compile: Whether to torch.compile() the model (CUDA only).
    """

    MODEL_CONFIGS = {
        "dinov3_vits16plus": {
            "family": "dinov3",
            "img_size": 512,
            "patch_size": 16,
            "embed_dim": 384,
            "local_repo": "third_party/dinov3",
            "weights": "third_party/dinov3/checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        },
        "dinov2_vitl14": {
            "family": "dinov2",
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 1024,
            "hub_repo": "facebookresearch/dinov2:main",
        },
        "dinov2_vitb14": {
            "family": "dinov2",
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 768,
            "hub_repo": "facebookresearch/dinov2:main",
        },
        "dinov2_vits14": {
            "family": "dinov2",
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 384,
            "hub_repo": "facebookresearch/dinov2:main",
        },
    }

    MODEL_ALIASES = {
        "facebook/dinov3-vits16plus-pretrain-lvd1689m": "dinov3_vits16plus",
        "facebook/dinov3-vits16plus": "dinov3_vits16plus",
        "dinov3-vits16plus": "dinov3_vits16plus",
        "dinov3_vits16plus_pretrain_lvd1689m": "dinov3_vits16plus",
        "facebook/dinov2-vitl14": "dinov2_vitl14",
        "facebook/dinov2-vitb14": "dinov2_vitb14",
        "facebook/dinov2-vits14": "dinov2_vits14",
        "dinov2-vitl14": "dinov2_vitl14",
        "dinov2-vitb14": "dinov2_vitb14",
        "dinov2-vits14": "dinov2_vits14",
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
        cfg = self.MODEL_CONFIGS.get(self.model_name, self.MODEL_CONFIGS["dinov3_vits16plus"])
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
        cfg = self.MODEL_CONFIGS.get(self.model_name, self.MODEL_CONFIGS["dinov3_vits16plus"])

        if cfg["family"] == "dinov3":
            model = self._load_dinov3(cfg)
        else:
            try:
                hub_repo = cfg["hub_repo"]
                model = torch.hub.load(hub_repo, self.model_name)
                print(f"[DINOExtractor] Loaded via torch.hub: {hub_repo}/{self.model_name}")
            except Exception as e:
                print(f"[DINOExtractor] torch.hub failed: {e}, trying alternative...")
                model = self._load_alternative()

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

    def _load_dinov3(self, cfg: dict) -> nn.Module:
        """Load DINOv3 from the local third_party checkout and checkpoint."""
        import sys

        root = resolve_assets_root(self.assets_root)
        repo = root / cfg["local_repo"]
        weights = root / cfg["weights"]
        if not repo.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {repo}")
        if not weights.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from dinov3.hub import backbones

        build_model = getattr(backbones, self.model_name)
        model = build_model(pretrained=False, check_hash=False)
        state_dict = torch.load(str(weights), map_location="cpu")
        if isinstance(state_dict, dict) and "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        model.load_state_dict(state_dict, strict=True)
        print(f"[DINOExtractor] Loaded DINOv3: {self.model_name} ({weights})")
        return model

    def _load_alternative(self) -> nn.Module:
        """Try timm as fallback."""
        try:
            import timm

            model = timm.create_model(
                f"vit_{self.model_name.replace('dinov2_', '')}",
                pretrained=True,
                num_classes=0,
            )
            _warn_once(
                f"dino_timm_fallback:{self.model_name}",
                f"DINO fallback: loaded {self.model_name} through timm after the primary loader failed.",
            )
            print(f"[DINOExtractor] Loaded via timm")
            return model
        except Exception as e:
            _warn_once(
                f"dino_timm_fallback_failed:{self.model_name}",
                f"DINO fallback failed: timm could not load {self.model_name}: {e}",
            )
            print(f"[DINOExtractor] timm fallback failed: {e}")
            raise RuntimeError(
                f"Cannot load DINO model '{self.model_name}'. "
                "torch.hub and timm both failed. "
                "Ensure the requested backbone is available locally."
            )

    def _extract_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Return DINO tokens as (1, N, D), preferring patch-level features."""
        batch = image_tensor.unsqueeze(0)

        if hasattr(self.model, "forward_features"):
            output = self.model.forward_features(batch)
        else:
            output = self.model(batch)

        if isinstance(output, dict):
            cls_token = None
            patch_tokens = None
            for key in ("x_norm_clstoken", "cls_token", "pooler_output"):
                if key in output and output[key] is not None:
                    cls_token = output[key]
                    break
            for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
                if key in output and output[key] is not None:
                    patch_tokens = output[key]
                    break

            if patch_tokens is not None and patch_tokens.ndim == 3:
                if cls_token is not None and cls_token.ndim == 2:
                    return torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
                return patch_tokens
            if cls_token is not None and cls_token.ndim == 2:
                return cls_token.unsqueeze(1)

            for value in output.values():
                if isinstance(value, torch.Tensor):
                    output = value
                    break

        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif isinstance(output, tuple):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Unsupported DINO output type: {type(output)!r}")
        if output.ndim == 2:
            output = output.unsqueeze(1)
        if output.ndim != 3:
            raise RuntimeError(f"Unsupported DINO token shape: {tuple(output.shape)}")
        return output

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
        from decord import VideoReader, cpu

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
