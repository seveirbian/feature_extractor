"""Pose feature extractor backed by a local VGGT checkpoint."""

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from feature_extractor.assets import resolve_assets_root

_WARNED_FALLBACKS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def rotation_to_6d(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (ortho6d)."""
    if rot.shape == (3, 3):
        rot_6d = np.concatenate([rot[:3, 0], rot[:3, 1]], axis=-1).astype(np.float32)
    else:
        rot_6d = np.asarray(rot, dtype=np.float32).flatten()
    return rot_6d


def pose_to_se3(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Package translation + rotation-6D into a 9D pose vector.

    The historical function name is kept for compatibility with existing
    storage paths. This is not a Lie algebra se(3) log vector.
    """
    rot_6d = rotation_to_6d(rotation)
    return np.concatenate([translation.flatten(), rot_6d], axis=-1).astype(np.float32)


def _resolve_local_vggt_paths(assets_root=None) -> tuple[Path, Path]:
    """Resolve the local VGGT repo and checkpoint path, or raise immediately."""
    repo_root = resolve_assets_root(assets_root)
    vggt_repo = repo_root / "third_party" / "VGGT"
    if not vggt_repo.exists():
        raise FileNotFoundError(f"VGGT repo not found: {vggt_repo}")

    snapshots_dir = vggt_repo / "checkpoints" / "models--facebook--VGGT-1B" / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"VGGT snapshot directory not found: {snapshots_dir}")

    # Prefer .safetensors over .pt (pt files can be corrupted during download)
    safetensor_candidates = sorted(snapshots_dir.glob("*/model.safetensors"))
    if safetensor_candidates:
        return vggt_repo, safetensor_candidates[-1]

    model_candidates = sorted(snapshots_dir.glob("*/model.pt"))
    if model_candidates:
        return vggt_repo, model_candidates[-1]

    raise FileNotFoundError(
        f"No local VGGT checkpoint found under {snapshots_dir}. Expected model.pt or model.safetensors."
    )


def _find_vggt_state_blob(assets_root=None) -> Path:
    """Find the VGGT model state_dict blob file in the LFS cache."""
    repo_root = resolve_assets_root(assets_root)
    blobs_dir = repo_root / "third_party" / "VGGT" / "checkpoints" / "models--facebook--VGGT-1B" / "blobs"
    if not blobs_dir.exists():
        raise FileNotFoundError(f"VGGT blobs dir not found: {blobs_dir}")

    # The main model weight blob is the large (~4.7GB) one.
    # We identify it by size > 100MB to skip config/metadata blobs.
    candidates = []
    for blob in blobs_dir.iterdir():
        if not blob.is_file():
            continue
        try:
            size = blob.stat().st_size
            if size > 100 * 1024 * 1024:  # > 100MB
                candidates.append((blob, size))
        except OSError:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No VGGT model blob (>100MB) found in {blobs_dir}. "
            "The checkpoint may not have been fully downloaded."
        )

    # Sort by size descending; the largest is the primary model weights
    candidates.sort(key=lambda x: x[1], reverse=True)
    chosen = candidates[0][0]
    print(f"[PoseExtractor] Using VGGT blob: {chosen} ({candidates[0][1] / 1e9:.1f} GB)")
    return chosen


def _load_vggt_model(device: torch.device, assets_root=None):
    """Load VGGT from the local LFS blob cache."""
    import sys

    vggt_repo = resolve_assets_root(assets_root) / "third_party" / "VGGT"
    if str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))

    from vggt.models.vggt import VGGT

    model = VGGT()
    blob_path = _find_vggt_state_blob(assets_root)
    state_dict = torch.load(str(blob_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"[PoseExtractor] Loaded VGGT from blob: {blob_path}")
    return model


class PoseExtractor:
    """VGGT-based head pose and trajectory extractor.

    Processes each frame through a frozen local VGGT checkpoint to obtain
    camera translation and rotation relative to the first frame.

    Args:
        model_name: VGGT model variant.
        device: torch device.
        max_frames: Maximum frames to process per video.
        use_relative: If True, all poses relative to frame 0.
    """

    def __init__(
        self,
        model_name: str = "ashleve/vggt-s",
        device: Optional[str] = None,
        max_frames: int = 600,
        use_relative: bool = True,
        input_color: Literal["rgb", "bgr"] = "rgb",
        assets_root: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_frames = max_frames
        self.use_relative = use_relative
        self.input_color = input_color
        self.assets_root = assets_root
        self.autocast_dtype = None
        if self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.model = self._load_model()

    def _load_model(self):
        model = _load_vggt_model(self.device, self.assets_root)
        for param in model.parameters():
            param.requires_grad = False
        print(f"[PoseExtractor] Using VGGT on {self.device}")
        return model

    def _preprocess_image_tensor(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert one uint8 image to a normalized VGGT input tensor."""
        if image.dtype == np.uint8:
            image_f = image.astype(np.float32) / 255.0
        else:
            image_f = image.astype(np.float32, copy=False)
        if self.input_color == "bgr" and image_f.shape[2] == 3:
            image_f = image_f[..., ::-1]

        import math

        orig_h, orig_w = image_f.shape[:2]
        target_h = 518
        target_w = 518
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = math.ceil(orig_h * scale / 14) * 14
        new_w = math.ceil(orig_w * scale / 14) * 14

        img_tensor = torch.from_numpy(image_f).permute(2, 0, 1).float().to(self.device)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        return img_tensor, (new_h, new_w)

    @torch.no_grad()
    def _infer_sequence_pose_enc(self, images: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
        """Run VGGT jointly on a frame sequence and return pose encodings."""
        if not images:
            raise ValueError("Expected at least one image for sequence pose inference")

        tensors = []
        image_hw = None
        for image in images:
            img_tensor, current_hw = self._preprocess_image_tensor(image)
            if image_hw is None:
                image_hw = current_hw
            elif current_hw != image_hw:
                raise ValueError(f"Inconsistent preprocessed image size: {current_hw} vs {image_hw}")
            tensors.append(img_tensor)

        batch = torch.stack(tensors, dim=0)
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = self.model(batch)
        pose_enc = outputs["pose_enc"].squeeze(0).float().cpu().numpy()
        return pose_enc, image_hw

    def _pose_enc_sequence_to_relative_se3(
        self,
        pose_enc: np.ndarray,
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert VGGT sequence pose encodings into frame-0-relative extrinsics."""
        import sys

        vggt_repo, _checkpoint_path = _resolve_local_vggt_paths(self.assets_root)
        if str(vggt_repo) not in sys.path:
            sys.path.insert(0, str(vggt_repo))
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pose_tensor = torch.from_numpy(np.asarray(pose_enc, dtype=np.float32)).to(self.device).unsqueeze(0)
        extrinsics, _intrinsics = pose_encoding_to_extri_intri(
            pose_tensor,
            image_size_hw=image_hw,
            build_intrinsics=False,
        )
        extrinsics_np = extrinsics.squeeze(0).float().cpu().numpy()

        extrinsics_h = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (extrinsics_np.shape[0], 1, 1))
        extrinsics_h[:, :3, :4] = extrinsics_np
        ref_inv = np.linalg.inv(extrinsics_h[0])
        rel_extrinsics = extrinsics_h @ ref_inv[None, :, :]

        rel_se3 = []
        for extr in rel_extrinsics:
            rel_t = extr[:3, 3].astype(np.float32)
            rel_r6d = rotation_to_6d(extr[:3, :3].astype(np.float32))
            rel_se3.append(pose_to_se3(rel_t, rel_r6d))
        return np.stack(rel_se3, axis=0).astype(np.float32)

    @torch.no_grad()
    def extract_frame(self, image: np.ndarray, ref_image: Optional[np.ndarray] = None) -> dict:
        """Extract pose for a single frame.

        Args:
            image: (H, W, 3) uint8 BGR.
            ref_image: Reference frame for relative pose. If None, returns absolute.

        Returns:
            dict with keys: translation (3,), rotation_6d (6,), se3 (9,).
        """
        return self._extract_vggt(image, ref_image)

    def _extract_vggt(self, image: np.ndarray, ref_image: Optional[np.ndarray]) -> dict:
        """Extract pose via VGGT.

        VGGT outputs:
          - pose_enc: [B, S, 9] = [translation(3), quaternion(4), focal_length(2)]
          - depth:    [B, S, H, W, 1]
          - extrinsics/intrinsics are NOT directly output; pose_enc must be decoded.
        For relative pose, we compare world_points or use the centroid of depth.
        """
        if image.dtype == np.uint8:
            image_f = image.astype(np.float32) / 255.0
        else:
            image_f = image
        if self.input_color == "bgr" and image_f.shape[2] == 3:
            image_f = image_f[..., ::-1]  # BGR → RGB

        # VGGT requires H and W to be multiples of 14 (ViT patch size).
        # Use size divisible by 14 and close to the original aspect ratio.
        # Common ViT input size: 518. Here we round up to nearest multiple of 14.
        import math
        orig_h, orig_w = image_f.shape[:2]
        target_h = 518
        target_w = 518
        # Round to nearest multiple of 14 while keeping aspect ratio
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = math.ceil(orig_h * scale / 14) * 14
        new_w = math.ceil(orig_w * scale / 14) * 14

        img_tensor = (
            torch.from_numpy(image_f).permute(2, 0, 1).float().to(self.device)
        )
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=(new_h, new_w),
            mode="bicubic", align_corners=False,
        )
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = self.model(img_tensor)  # single-frame input → [1, 1, ...]

        # pose_enc: [1, 1, 9] → [9]
        pose_enc = outputs["pose_enc"].squeeze(0).squeeze(0).cpu().numpy()
        translation = pose_enc[:3].astype(np.float32)
        quat_xyzw = pose_enc[3:7].astype(np.float32)  # xyzw quaternion

        # depth: [1, 1, H, W, 1]
        depth_squeezed = outputs["depth"].squeeze(0)  # [1, H, W, 1] or [H, W, 1]

        if ref_image is not None:
            ref_pose_enc = self._extract_vggt_pose_encoding(ref_image)
            rel_T = translation - ref_pose_enc["translation"]

            # Relative quaternion: q_rel = q_cur * q_ref^-1
            q_cur = quat_xyzw
            q_ref = ref_pose_enc["quat_xyzw"]
            q_rel = self._quaternion_multiply(q_cur, self._quaternion_conjugate(q_ref))
            rel_rot_6d = self._quaternion_to_6d(q_rel)
            se3 = pose_to_se3(rel_T, rel_rot_6d)
        else:
            rot_6d = self._quaternion_to_6d(quat_xyzw)
            se3 = pose_to_se3(translation, rot_6d)

        return {
            "translation": translation,
            "rotation_6d": self._quaternion_to_6d(quat_xyzw),
            "se3": se3,
        }

    def _extract_vggt_pose_encoding(self, image: np.ndarray) -> dict:
        """Extract just the pose encoding (translation + quaternion) from VGGT."""
        if image.dtype == np.uint8:
            image_f = image.astype(np.float32) / 255.0
        else:
            image_f = image
        if self.input_color == "bgr" and image_f.shape[2] == 3:
            image_f = image_f[..., ::-1]

        import math
        orig_h, orig_w = image_f.shape[:2]
        scale = min(518 / orig_h, 518 / orig_w)
        new_h = math.ceil(orig_h * scale / 14) * 14
        new_w = math.ceil(orig_w * scale / 14) * 14

        img_tensor = (
            torch.from_numpy(image_f).permute(2, 0, 1).float().to(self.device)
        )
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=(new_h, new_w),
            mode="bicubic", align_corners=False,
        )
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = self.model(img_tensor)
        pose_enc = outputs["pose_enc"].squeeze(0).squeeze(0).cpu().numpy()
        return {
            "translation": pose_enc[:3].astype(np.float32),
            "quat_xyzw": pose_enc[3:7].astype(np.float32),
        }

    def _quaternion_to_6d(self, quat_xyzw: np.ndarray) -> np.ndarray:
        """Convert xyzw quaternion to 6D rotation representation."""
        x = quat_xyzw.copy()
        # Normalize
        x = x / (np.linalg.norm(x) + 1e-8)
        # First two columns of rotation matrix from quaternion
        # R from quaternion q=[w,x,y,z]:
        w, a, b, c = x[3], x[0], x[1], x[2]
        R = np.array([
            [1 - 2*(b*b + c*c),     2*(a*b - c*w),     2*(a*c + b*w)],
            [    2*(a*b + c*w), 1 - 2*(a*a + c*c),     2*(b*c - a*w)],
            [    2*(a*c - b*w),     2*(b*c + a*w), 1 - 2*(a*a + b*b)],
        ], dtype=np.float32)
        # ortho6d: first two columns
        return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two xyzw quaternions: q1 * q2."""
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float32)

    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Conjugate of xyzw quaternion."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    def _extract_orb(self, image: np.ndarray, ref_image: Optional[np.ndarray]) -> dict:
        """Extract relative pose via ORB optical flow + essential/homography decomposition.

        Uses ORB keypoint matching between current and reference frame to
        estimate camera motion via Essential matrix decomposition.
        Produces rotation and relative translation direction (up to scale).
        """
        if ref_image is None:
            _warn_once(
                "pose_orb_missing_reference",
                "Pose fallback: ORB pose requested without a reference image; using identity pose.",
            )
            return self._identity_pose()

        color_code = cv2.COLOR_BGR2GRAY if self.input_color == "bgr" else cv2.COLOR_RGB2GRAY
        gray_cur = cv2.cvtColor(image, color_code) if image.ndim == 3 else image
        gray_ref = cv2.cvtColor(ref_image, color_code) if ref_image.ndim == 3 else ref_image

        kp_ref, des_ref = self._orb.detectAndCompute(gray_ref, None)
        kp_cur, des_cur = self._orb.detectAndCompute(gray_cur, None)

        if des_ref is None or des_cur is None or len(kp_ref) < 8 or len(kp_cur) < 8:
            _warn_once(
                "pose_orb_insufficient_keypoints",
                "Pose fallback: ORB found fewer than 8 keypoints/descriptors; using identity pose.",
            )
            return self._dummy_pose()

        matches = self._bf.match(des_ref, des_cur)
        if len(matches) < 8:
            _warn_once(
                "pose_orb_insufficient_matches",
                "Pose fallback: ORB found fewer than 8 matches; using identity pose.",
            )
            return self._dummy_pose()
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        src_pts = np.float64([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float64([kp_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h, w = gray_cur.shape[:2]
        focal = float(w)
        K = np.float64([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])

        E, mask_e = cv2.findEssentialMat(
            src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        inliers_e = int(mask_e.ravel().sum()) if mask_e is not None else 0

        if inliers_e < 8:
            H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if mask_h is not None and mask_h.ravel().sum() >= 8:
                try:
                    n_sols, Rs, ts, _ = cv2.decomposeHomographyMat(H, K)
                    R, t = _pick_valid_homography_solution(Rs, ts, src_pts, dst_pts, K)
                except Exception:
                    _warn_once(
                        "pose_homography_decompose_failed",
                        "Pose fallback: homography decomposition failed; using identity pose.",
                    )
                    R, t = np.eye(3), np.zeros(3)
            else:
                _warn_once(
                    "pose_essential_and_homography_failed",
                    "Pose fallback: essential matrix and homography estimation both failed; using identity pose.",
                )
                R, t = np.eye(3), np.zeros(3)
        else:
            _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask_e)
            t = t.ravel()

        se3 = pose_to_se3(t, R)
        return {
            "translation": t.astype(np.float32),
            "rotation_6d": rotation_to_6d(R).astype(np.float32),
            "se3": se3.astype(np.float32),
        }

    def _dummy_pose(self) -> dict:
        """Return zero pose when model unavailable and ORB fails."""
        return self._identity_pose()

    @staticmethod
    def _identity_pose() -> dict:
        translation = np.zeros(3, dtype=np.float32)
        rotation_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return {
            "translation": translation,
            "rotation_6d": rotation_6d,
            "se3": np.concatenate([translation, rotation_6d], axis=-1).astype(np.float32),
        }

    @torch.no_grad()
    def extract_video(
        self,
        video_path: str,
        frame_indices: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Extract frame-0-relative camera extrinsics for a video.

        Translation is the relative extrinsic translation term from VGGT and
        remains in VGGT's normalized scene scale, not metric meters.
        """
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if frame_indices is None:
            frame_indices = list(range(min(total_frames, self.max_frames)))

        if len(frame_indices) > self.max_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist()

        valid_indices = [int(i) for i in frame_indices if 0 <= int(i) < total_frames]
        if not valid_indices:
            return np.zeros((0, 9), dtype=np.float32)

        frames = [vr[i].asnumpy() for i in tqdm(valid_indices, desc=f"PoseFrames [{Path(video_path).name}]")]
        pose_enc, image_hw = self._infer_sequence_pose_enc(frames)
        return self._pose_enc_sequence_to_relative_se3(pose_enc, image_hw)

    def save_trajectory(self, se3_trajectory: np.ndarray, output_path: str) -> None:
        """Save SE(3) trajectory to HDF5."""
        import h5py
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w", libver="v1_10") as f:
            f.create_dataset(
                "se3_trajectory",
                data=se3_trajectory, dtype=np.float32,
                compression="lzf", chunks=(1, se3_trajectory.shape[1]),
            )
            f.attrs["description"] = "translation + rotation-6D trajectory relative to frame 0"
            f.attrs["columns"] = "tx,ty,tz,r6d_0,r6d_1,r6d_2,r6d_3,r6d_4,r6d_5"

    def __repr__(self) -> str:
        return f"PoseExtractor(device={self.device}, model={'VGGT' if self.model else 'ORB'})"


def _pick_valid_homography_solution(
    Rs: list, ts: list, src_pts: np.ndarray, dst_pts: np.ndarray, K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the homography decomposition solution with most positive-depth points."""
    best_R, best_t, best_count = np.eye(3), np.zeros(3), -1
    for R, t in zip(Rs, ts):
        t = t.ravel()
        count = 0
        for i in range(len(src_pts)):
            try:
                A = np.column_stack([K @ R[:, 0], K @ R[:, 1], K @ t])
                X = np.linalg.solve(A, (K @ dst_pts[i, 0].T).flatten())
                X = X / (X[2] + 1e-8)
                if X[2] > 0:
                    count += 1
            except Exception:
                pass
        if count > best_count:
            best_count = count
            best_R, best_t = R, t
    return best_R, best_t
