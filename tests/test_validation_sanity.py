import numpy as np

from feature_extractor.validation.sanity import (
    CheckResult,
    rot6d_to_matrix,
    is_valid_rotation,
)
from feature_extractor.extractors.pose import rotation_to_6d


def test_rot6d_identity_roundtrips_to_identity():
    R = rot6d_to_matrix(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    assert np.allclose(R, np.eye(3), atol=1e-5)


def test_rot6d_recovers_known_rotation():
    # 绕 z 轴 90°
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    r6d = rotation_to_6d(Rz)                # 生产编码:R 的前两列
    R = rot6d_to_matrix(r6d)
    assert np.allclose(R, Rz, atol=1e-5)


def test_is_valid_rotation_accepts_rotation_rejects_garbage():
    assert is_valid_rotation(np.eye(3))
    assert not is_valid_rotation(np.full((3, 3), 2.0, dtype=np.float32))


def test_is_valid_rotation_rejects_reflection():
    # 反射矩阵:正交但 det=-1,不是旋转
    refl = np.diag([1.0, 1.0, -1.0])
    assert not is_valid_rotation(refl)


def test_is_valid_rotation_rejects_wrong_shape():
    assert not is_valid_rotation(np.eye(2))
    assert not is_valid_rotation(np.zeros((3, 4)))


def test_checkresult_is_a_dataclass_with_fields():
    c = CheckResult(branch="dino", name="shape", expected="(T,N,384)",
                    observed="(4,1025,384)", passed=True)
    assert c.passed and c.branch == "dino"
