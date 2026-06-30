"""HDF5 feature storage for egoWM data production.

The store keeps one file per video/episode and preserves the original video
frame index for every extracted feature row. Training code samples chunks over
feature rows, then uses ``frame_indices`` only as metadata or to align external
labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import h5py
import numpy as np


FeatureBranch = Literal["dino", "depth", "pose"]


class FeatureStore:
    """HDF5-backed feature store for DINO, depth, and camera-pose targets.

    Canonical layout per video:

        /dino/features         (T, N, D) or legacy (T, D), float32
        /dino/frame_indices    (T,), int64
        /depth/inv_depth       (T, H, W, 1), uint16 on disk
        /depth/frame_indices   (T,), int64
        /pose/se3_trajectory   (T, P), float32, P can be 6 or 9
        /pose/frame_indices    (T,), int64

    Depth is returned as float32 in [0, 1] with shape (T, H, W, 1), even for
    legacy stores that were written as (T, H, W).
    """

    def __init__(
        self,
        store_dir: str,
        compression: Literal["lzf", "gzip"] | None = "lzf",
        gzip_level: int = 4,
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.gzip_level = gzip_level

    def _path(self, video_id: str) -> Path:
        return self.store_dir / f"{video_id}.h5"

    def _get_file(self, video_id: str, mode: str = "r") -> h5py.File:
        return h5py.File(self._path(video_id), mode)

    def _compression_kwargs(self) -> dict[str, Any]:
        if self.compression is None:
            return {}
        kwargs: dict[str, Any] = {"compression": self.compression}
        if self.compression == "gzip":
            kwargs["compression_opts"] = self.gzip_level
        return kwargs

    @staticmethod
    def _default_frame_indices(length: int, frame_indices: Optional[np.ndarray]) -> np.ndarray:
        if frame_indices is None:
            return np.arange(length, dtype=np.int64)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices.ndim != 1:
            raise ValueError(f"frame_indices must be 1D, got shape {frame_indices.shape}")
        if len(frame_indices) != length:
            raise ValueError(
                f"frame_indices length {len(frame_indices)} does not match feature length {length}"
            )
        return frame_indices

    @staticmethod
    def _select_rows(
        values: np.ndarray,
        stored_frame_indices: np.ndarray,
        requested_frame_indices: Optional[np.ndarray],
    ) -> np.ndarray:
        if requested_frame_indices is None:
            return values
        idx_map = {int(frame): row for row, frame in enumerate(stored_frame_indices)}
        rows = [idx_map[int(frame)] for frame in requested_frame_indices if int(frame) in idx_map]
        return values[rows]

    @staticmethod
    def _read_requested_rows(
        dataset: h5py.Dataset,
        stored_frame_indices: np.ndarray,
        requested_frame_indices: Optional[np.ndarray],
    ) -> np.ndarray:
        if requested_frame_indices is None:
            return dataset[:]
        idx_map = {int(frame): row for row, frame in enumerate(stored_frame_indices)}
        rows = [idx_map[int(frame)] for frame in requested_frame_indices if int(frame) in idx_map]
        if not rows:
            return dataset[:0]
        order = np.argsort(rows)
        sorted_rows = np.asarray(rows, dtype=np.int64)[order]
        values = dataset[sorted_rows]
        inverse = np.argsort(order)
        return values[inverse]

    @staticmethod
    def _normalize_depth(inv_depth: np.ndarray) -> np.ndarray:
        inv_depth = np.asarray(inv_depth)
        if inv_depth.ndim == 3:
            inv_depth = inv_depth[..., None]
        elif inv_depth.ndim == 4 and inv_depth.shape[1] == 1 and inv_depth.shape[-1] != 1:
            inv_depth = np.moveaxis(inv_depth, 1, -1)
        if inv_depth.ndim != 4 or inv_depth.shape[-1] != 1:
            raise ValueError(
                "inv_depth must have shape (T,H,W), (T,H,W,1), or (T,1,H,W); "
                f"got {inv_depth.shape}"
            )
        return inv_depth.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_pose(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32)
        if pose.ndim == 1:
            pose = pose[None, :]
        if pose.ndim != 2:
            raise ValueError(f"pose trajectory must have shape (T,P), got {pose.shape}")
        return pose

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_dino(
        self,
        video_id: str,
        features: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Write DINO features as FP32 without quantization."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim not in (2, 3):
            raise ValueError(f"DINO features must have shape (T,D) or (T,N,D), got {features.shape}")
        frame_indices = self._default_frame_indices(len(features), frame_indices)
        chunks = (1, features.shape[1]) if features.ndim == 2 else (1, features.shape[1], features.shape[2])

        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("dino")
            if "features" in grp:
                del grp["features"]
            grp.create_dataset(
                "features",
                data=features,
                dtype=np.float32,
                chunks=chunks,
                **self._compression_kwargs(),
            )
            if "frame_indices" in grp:
                del grp["frame_indices"]
            grp.create_dataset(
                "frame_indices",
                data=frame_indices,
                dtype=np.int64,
                **self._compression_kwargs(),
            )
            grp.attrs["video_id"] = video_id
            grp.attrs["shape"] = features.shape
            grp.attrs["representation"] = "patch_tokens" if features.ndim == 3 else "global_descriptor"

    def _append_or_create(
        self,
        grp: h5py.Group,
        name: str,
        data: np.ndarray,
        *,
        reset: bool,
        chunks: tuple[int, ...],
    ) -> None:
        """Create a resizable dataset on reset, else resize axis 0 and append."""
        if reset:
            if name in grp:
                del grp[name]
            maxshape = (None,) + data.shape[1:]
            grp.create_dataset(
                name,
                data=data,
                maxshape=maxshape,
                chunks=chunks,
                dtype=data.dtype,
                **self._compression_kwargs(),
            )
        else:
            dset = grp[name]
            n = dset.shape[0]
            dset.resize(n + data.shape[0], axis=0)
            dset[n:] = data

    def write_dino_chunk(
        self,
        video_id: str,
        features: np.ndarray,
        frame_indices: np.ndarray,
        *,
        reset: bool,
    ) -> None:
        """Append a block of DINO features to a resizable dataset."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim not in (2, 3):
            raise ValueError(f"DINO features must be (T,D) or (T,N,D), got {features.shape}")
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        chunks = (1,) + features.shape[1:]
        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("dino")
            self._append_or_create(grp, "features", features, reset=reset, chunks=chunks)
            self._append_or_create(
                grp, "frame_indices", frame_indices, reset=reset,
                chunks=(min(1024, len(frame_indices) or 1),),
            )
            if reset:
                grp.attrs["complete"] = False
                grp.attrs["representation"] = (
                    "patch_tokens" if features.ndim == 3 else "global_descriptor"
                )

    def mark_branch_complete(self, video_id: str, branch: FeatureBranch) -> None:
        with self._get_file(video_id, mode="a") as f:
            f.require_group(branch).attrs["complete"] = True

    def is_branch_complete(self, video_id: str, branch: FeatureBranch) -> bool:
        if not self.exists(video_id):
            return False
        with self._get_file(video_id, mode="r") as f:
            if branch not in f:
                return False
            return bool(f[branch].attrs.get("complete", False))

    def is_video_complete(self, video_id: str, branches: list[FeatureBranch]) -> bool:
        return all(self.is_branch_complete(video_id, b) for b in branches)

    def write_dino_camera(
        self,
        video_id: str,
        camera_key: str,
        features: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
        *,
        patch_grid: tuple[int, int] | None = None,
        has_cls_token: bool = True,
    ) -> None:
        """Write per-camera DINO features without camera concatenation."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError(f"Per-camera DINO features must have shape (T,N,D), got {features.shape}")
        frame_indices = self._default_frame_indices(len(features), frame_indices)
        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("dino").require_group("cameras").require_group(camera_key)
            if "features" in grp:
                del grp["features"]
            grp.create_dataset(
                "features",
                data=features,
                dtype=np.float32,
                chunks=(1, features.shape[1], features.shape[2]),
                **self._compression_kwargs(),
            )
            if "frame_indices" in grp:
                del grp["frame_indices"]
            grp.create_dataset("frame_indices", data=frame_indices, dtype=np.int64, **self._compression_kwargs())
            grp.attrs["camera_key"] = camera_key
            grp.attrs["shape"] = features.shape
            grp.attrs["representation"] = "patch_tokens"
            grp.attrs["has_cls_token"] = bool(has_cls_token)
            if patch_grid is not None:
                grp.attrs["patch_grid"] = tuple(int(v) for v in patch_grid)

    def write_depth(
        self,
        video_id: str,
        inv_depth: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Write normalized inverse depth in [0, 1] as uint16."""
        inv_depth = self._normalize_depth(inv_depth)
        frame_indices = self._default_frame_indices(len(inv_depth), frame_indices)
        inv_depth_u16 = (np.clip(inv_depth, 0.0, 1.0) * 65535.0).round().astype(np.uint16)

        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("depth")
            if "inv_depth" in grp:
                del grp["inv_depth"]
            grp.create_dataset(
                "inv_depth",
                data=inv_depth_u16,
                dtype=np.uint16,
                chunks=(1, inv_depth.shape[1], inv_depth.shape[2], 1),
                **self._compression_kwargs(),
            )
            if "frame_indices" in grp:
                del grp["frame_indices"]
            grp.create_dataset(
                "frame_indices",
                data=frame_indices,
                dtype=np.int64,
                **self._compression_kwargs(),
            )
            grp.attrs["scale"] = 65535.0
            grp.attrs["shape"] = inv_depth.shape
            grp.attrs["representation"] = "normalized_inverse_depth"

    def write_depth_camera(
        self,
        video_id: str,
        camera_key: str,
        inv_depth: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Write per-camera normalized inverse depth at the extractor input resolution."""
        inv_depth = self._normalize_depth(inv_depth)
        frame_indices = self._default_frame_indices(len(inv_depth), frame_indices)
        inv_depth_u16 = (np.clip(inv_depth, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("depth").require_group("cameras").require_group(camera_key)
            if "inv_depth" in grp:
                del grp["inv_depth"]
            grp.create_dataset(
                "inv_depth",
                data=inv_depth_u16,
                dtype=np.uint16,
                chunks=(1, inv_depth.shape[1], inv_depth.shape[2], 1),
                **self._compression_kwargs(),
            )
            if "frame_indices" in grp:
                del grp["frame_indices"]
            grp.create_dataset("frame_indices", data=frame_indices, dtype=np.int64, **self._compression_kwargs())
            grp.attrs["camera_key"] = camera_key
            grp.attrs["scale"] = 65535.0
            grp.attrs["shape"] = inv_depth.shape
            grp.attrs["representation"] = "normalized_inverse_depth"

    def write_pose(
        self,
        video_id: str,
        se3_trajectory: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
        representation: str | None = None,
    ) -> None:
        """Write camera pose targets.

        ``se3_trajectory`` may be true se(3) log vectors (T, 6) or the more
        stable first-version format translation + rotation-6D (T, 9).
        """
        pose = self._normalize_pose(se3_trajectory)
        frame_indices = self._default_frame_indices(len(pose), frame_indices)
        if representation is None:
            representation = "se3_log" if pose.shape[1] == 6 else "translation_rot6d"

        with self._get_file(video_id, mode="a") as f:
            grp = f.require_group("pose")
            if "se3_trajectory" in grp:
                del grp["se3_trajectory"]
            grp.create_dataset(
                "se3_trajectory",
                data=pose,
                dtype=np.float32,
                chunks=(1, pose.shape[1]),
                **self._compression_kwargs(),
            )
            if "frame_indices" in grp:
                del grp["frame_indices"]
            grp.create_dataset(
                "frame_indices",
                data=frame_indices,
                dtype=np.int64,
                **self._compression_kwargs(),
            )
            grp.attrs["pose_dim"] = pose.shape[1]
            grp.attrs["representation"] = representation

    def write_episode(
        self,
        video_id: str,
        dino_features: np.ndarray,
        depth_inv: np.ndarray,
        pose_se3: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.write_dino(video_id, dino_features, frame_indices)
        self.write_depth(video_id, depth_inv, frame_indices)
        self.write_pose(video_id, pose_se3, frame_indices)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def read_dino(
        self,
        video_id: str,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Any:
        with self._get_file(video_id, mode="r") as f:
            if "dino/cameras" in f:
                out: dict[str, dict[str, Any]] = {}
                for camera_key, grp in f["dino/cameras"].items():
                    stored_frames = grp["frame_indices"][:]
                    values = self._read_requested_rows(grp["features"], stored_frames, frame_indices)
                    out[camera_key] = {
                        "features": values.astype(np.float32, copy=False),
                        "patch_grid": tuple(int(v) for v in grp.attrs.get("patch_grid", ())),
                        "has_cls_token": bool(grp.attrs.get("has_cls_token", True)),
                    }
                return out
            stored_frames = f["dino/frame_indices"][:]
            values = self._read_requested_rows(f["dino/features"], stored_frames, frame_indices)
        return values.astype(np.float32, copy=False)

    def read_depth(
        self,
        video_id: str,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Any:
        with self._get_file(video_id, mode="r") as f:
            if "depth/cameras" in f:
                out: dict[str, np.ndarray] = {}
                for camera_key, grp in f["depth/cameras"].items():
                    scale = float(grp.attrs.get("scale", 65535.0))
                    stored_frames = grp["frame_indices"][:]
                    values = self._read_requested_rows(grp["inv_depth"], stored_frames, frame_indices)
                    out[camera_key] = self._normalize_depth(values.astype(np.float32) / scale)
                return out
            scale = float(f["depth"].attrs.get("scale", 65535.0))
            stored_frames = f["depth/frame_indices"][:]
            values = self._read_requested_rows(f["depth/inv_depth"], stored_frames, frame_indices)
        inv_depth = self._normalize_depth(values.astype(np.float32) / scale)
        return inv_depth

    def read_pose(
        self,
        video_id: str,
        frame_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        with self._get_file(video_id, mode="r") as f:
            stored_frames = f["pose/frame_indices"][:]
            values = self._read_requested_rows(f["pose/se3_trajectory"], stored_frames, frame_indices)
        return values.astype(np.float32, copy=False)

    def read_all(
        self,
        video_id: str,
        frame_indices: Optional[np.ndarray] = None,
        strict: bool = False,
    ) -> dict[str, Optional[np.ndarray]]:
        """Read all available branches.

        When ``strict`` is False, missing branches are returned as None. This is
        useful while producing data incrementally.
        """
        out: dict[str, Optional[np.ndarray]] = {"dino": None, "depth": None, "pose": None}
        readers = {
            "dino": self.read_dino,
            "depth": self.read_depth,
            "pose": self.read_pose,
        }
        for key, reader in readers.items():
            try:
                out[key] = reader(video_id, frame_indices)
            except KeyError:
                if strict:
                    raise
        return out

    # ------------------------------------------------------------------
    # Metadata and inspection
    # ------------------------------------------------------------------

    def exists(self, video_id: str) -> bool:
        return self._path(video_id).exists()

    def list_videos(self) -> list[str]:
        return sorted(p.stem for p in self.store_dir.glob("*.h5"))

    def read_frame_indices(self, video_id: str, branch: FeatureBranch = "dino") -> np.ndarray:
        with self._get_file(video_id, mode="r") as f:
            return f[f"{branch}/frame_indices"][:].astype(np.int64, copy=False)

    def primary_frame_indices(self, video_id: str) -> np.ndarray:
        """Return frame indices from the first available feature branch."""
        with self._get_file(video_id, mode="r") as f:
            for branch in ("dino", "depth", "pose"):
                if branch in f and "frame_indices" in f[branch]:
                    return f[f"{branch}/frame_indices"][:].astype(np.int64, copy=False)
        raise KeyError(f"No feature frame_indices found for {video_id}")

    def get_shape(self, video_id: str) -> dict[str, dict[str, tuple[int, ...]]]:
        with self._get_file(video_id, mode="r") as f:
            shapes: dict[str, dict[str, tuple[int, ...]]] = {}
            for branch in ("dino", "depth", "pose"):
                if branch in f:
                    shapes[branch] = {
                        name: tuple(dataset.shape)
                        for name, dataset in f[branch].items()
                        if isinstance(dataset, h5py.Dataset)
                    }
            return shapes

    def inspect_video(self, video_id: str) -> dict[str, Any]:
        with self._get_file(video_id, mode="r") as f:
            info: dict[str, Any] = {"video_id": video_id, "path": str(self._path(video_id)), "branches": {}}
            for branch in ("dino", "depth", "pose"):
                if branch not in f:
                    continue
                branch_info: dict[str, Any] = {"attrs": dict(f[branch].attrs)}
                for name, dataset in f[branch].items():
                    if isinstance(dataset, h5py.Dataset):
                        branch_info[name] = {
                            "shape": tuple(dataset.shape),
                            "dtype": str(dataset.dtype),
                        }
                info["branches"][branch] = branch_info
            return info

    def __repr__(self) -> str:
        return f"FeatureStore(dir={self.store_dir}, compression={self.compression})"
