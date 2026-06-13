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
