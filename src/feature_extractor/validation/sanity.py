"""功能不变量检查:纯几何工具 + 各分支 sanity 检查。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CheckResult:
    branch: str          # "dino" | "depth" | "pose" | "pipeline"
    name: str
    expected: str
    observed: str
    passed: bool


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D 表示(R 的前两列)经 Gram-Schmidt 重建 3x3 旋转矩阵。

    对退化输入(``a1`` 或正交化后的 ``a2`` 接近零向量)不抛错,而是返回一个
    近退化矩阵——交由调用方用 :func:`is_valid_rotation` 判定其是否为有效旋转。
    """
    r6d = np.asarray(r6d, dtype=np.float64).reshape(6)
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # 按列拼,与 rotation_to_6d 取列一致


def is_valid_rotation(R: np.ndarray, atol: float = 1e-4) -> bool:
    """检查 R 正交且 det≈+1。"""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    ortho = np.allclose(R @ R.T, np.eye(3), atol=atol)
    det = np.allclose(np.linalg.det(R), 1.0, atol=atol)
    return bool(ortho and det)


def _finite(arr) -> bool:
    return bool(np.all(np.isfinite(arr)))


def check_dino(features: np.ndarray) -> list[CheckResult]:
    f = np.asarray(features)
    is3 = f.ndim == 3
    fin = _finite(f)
    out = [
        CheckResult("dino", "ndim==3", "3", str(f.ndim), is3),
        CheckResult("dino", "dtype==float32", "float32", str(f.dtype), f.dtype == np.float32),
        CheckResult("dino", "embed_dim==384", "384",
                    str(f.shape[-1]) if is3 else "n/a",
                    is3 and f.shape[-1] == 384),
        CheckResult("dino", "有限值", "all finite", str(fin), fin),
        CheckResult("dino", "含CLS(N≥2)", ">=2",
                    str(f.shape[1]) if is3 else "n/a",
                    is3 and f.shape[1] >= 2),
    ]
    if is3 and f.shape[0] > 0 and f.shape[1] >= 2:
        cls_diff = float(np.abs(f[:, 0] - f[:, 1:].mean(axis=1)).max())
        out.append(CheckResult("dino", "CLS≠patch均值", ">1e-6",
                               f"{cls_diff:.3g}", cls_diff > 1e-6))
    return out


def check_depth(inv_depth: np.ndarray) -> list[CheckResult]:
    d = np.asarray(inv_depth)
    nonneg = bool(np.all(d >= -1e-6))
    fin = _finite(d)
    std = float(d.std()) if d.size else 0.0
    return [
        CheckResult("depth", "shape==(T,H,W,1)", "4D 末维1",
                    str(d.shape), d.ndim == 4 and d.shape[-1] == 1),
        CheckResult("depth", "dtype==float32", "float32", str(d.dtype), d.dtype == np.float32),
        CheckResult("depth", "逆深度≥0", ">=0", str(nonneg), nonneg),
        CheckResult("depth", "有限值", "all finite", str(fin), fin),
        CheckResult("depth", "非全常数", "std>0", f"{std:.3g}", std > 0),
    ]


def check_pose(pose: np.ndarray) -> list[CheckResult]:
    p = np.asarray(pose)
    out = [
        CheckResult("pose", "shape==(T,9)", "2D 末维9",
                    str(p.shape), p.ndim == 2 and p.shape[-1] == 9),
        CheckResult("pose", "dtype==float32", "float32", str(p.dtype), p.dtype == np.float32),
        CheckResult("pose", "有限值", "all finite", str(_finite(p)), _finite(p)),
    ]
    if p.ndim == 2 and p.shape[-1] == 9 and len(p) > 0:
        ident = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        err0 = float(np.abs(p[0] - ident).max())
        out.append(CheckResult("pose", "pose[0]≈单位变换", "max|·|<1e-3",
                               f"{err0:.3g}", err0 < 1e-3))
        valid = all(is_valid_rotation(rot6d_to_matrix(row[3:9])) for row in p)
        out.append(CheckResult("pose", "每帧6D→有效旋转", "正交且det≈1",
                               str(valid), valid))
    return out


def check_alignment(indices_by_branch: dict, requested: list) -> list[CheckResult]:
    req = list(int(i) for i in requested)
    out = []
    for branch, idx in indices_by_branch.items():
        same = list(int(i) for i in idx) == req
        out.append(CheckResult("pipeline", f"{branch} 帧索引对齐请求",
                               str(req[:4]) + "…", str(list(idx)[:4]) + "…", same))
    return out


def check_depth_roundtrip(written: np.ndarray, read_back: np.ndarray) -> CheckResult:
    """depth 存 uint16,往返用量化容差(1/65535 量级)。"""
    from feature_extractor.storage import FeatureStore
    expected = FeatureStore._normalize_depth(np.asarray(written, dtype=np.float32))
    read_back = np.asarray(read_back)
    tol = 2.0 / 65535.0
    if read_back.shape != expected.shape or read_back.size == 0:
        return CheckResult("pipeline", "depth 往返(量化容差)", f"max|·|<{tol:.2g}",
                           f"形状不符 {read_back.shape} vs {expected.shape}", False)
    err = float(np.abs(expected - read_back).max())
    return CheckResult("pipeline", "depth 往返(量化容差)", f"max|·|<{tol:.2g}",
                       f"{err:.2g}", err < tol)


def check_exact_roundtrip(branch: str, written: np.ndarray, read_back: np.ndarray) -> CheckResult:
    """DINO/Pose 为 float32 无损,往返应完全相等。"""
    equal = np.array_equal(np.asarray(written, dtype=np.float32), np.asarray(read_back))
    return CheckResult("pipeline", f"{branch} 往返(无损)", "完全相等", str(equal), equal)


def check_determinism(branch: str, a: np.ndarray, b: np.ndarray,
                      atol: float = 1e-3) -> CheckResult:
    """同输入两次运行应在容差内一致(cuDNN 可能非确定,用 allclose)。"""
    close = bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol))
    return CheckResult(branch, "确定性(两次allclose)", f"allclose atol={atol}",
                       str(close), close)
