# feature_extractor 自验证工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `feature_extractor` 增加 `feature-validate` 命令,跑合成视频做功能不变量检查、跑真实数据做性能基准,自动生成交付用的 Markdown 报告。

**Architecture:** 新增 `src/feature_extractor/validation/` 子包,拆为 `synthetic`(造合成视频)、`sanity`(不变量检查 + 纯几何工具)、`perf`(性能基准)、`report`(渲染 Markdown)、`cli`(编排)。纯函数走 TDD 单测;依赖重模型/GPU 的集成部分通过运行 CLI 冒烟验证。

**Tech Stack:** Python 3.12、PyAV(合成视频)、torch/CUDA(模型与显存计量)、h5py(`FeatureStore`)、pytest(纯函数单测)。

> 实现期相对设计 spec 的两处更正(已在对应任务中体现):
> 1. **depth 存储有损**:`FeatureStore.write_depth` 把逆深度归一化到 [0,1] 后存 **uint16**;depth 往返用量化容差(≈1/65535),非逐位相等。DINO/Pose 为 float32 无损。
> 2. **DINO patch 数**取决于提取器内部 resize(实现细节),故只断言 `embed_dim==384`、含 CLS、形状逐次稳定,不硬编码 patch 数。

---

### Task 1: 合成视频生成器 `synthetic.py`(并让 video_io 测试复用)

**Files:**
- Create: `src/feature_extractor/validation/__init__.py`
- Create: `src/feature_extractor/validation/synthetic.py`
- Modify: `tests/test_video_io.py`(改为引用新模块)

- [ ] **Step 1: 建空包文件**

`src/feature_extractor/validation/__init__.py`:
```python
"""feature_extractor 自验证工具(功能不变量 + 性能基准)。"""
```

- [ ] **Step 2: 写 `synthetic.py`**

`src/feature_extractor/validation/synthetic.py`:
```python
"""生成可控的合成视频,供功能验证使用(确定、可移植)。"""

from __future__ import annotations

import numpy as np


def make_ramp_video(
    path,
    *,
    codec: str = "libx264",
    n_frames: int = 12,
    width: int = 128,
    height: int = 96,
    step: int = 20,
) -> None:
    """写一段每帧填充灰度值 ``i*step`` 的纯色视频。

    解码后每帧均值仍能映射回索引 ``i``,可用于验证帧索引映射。
    """
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((height, width, 3), i * step, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_gradient_video(
    path,
    *,
    codec: str = "libx264",
    n_frames: int = 12,
    width: int = 128,
    height: int = 96,
) -> None:
    """写一段每帧带水平灰度渐变的视频,为 depth/dino 提供空间信号。"""
    import av

    grad = np.linspace(0, 255, width, dtype=np.uint8)
    base = np.broadcast_to(grad[None, :, None], (height, width, 3)).copy()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(n_frames):
            frame = av.VideoFrame.from_ndarray(base, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
```

- [ ] **Step 3: 改 `tests/test_video_io.py` 引用 `make_ramp_video`**

把原本的本地 `_make_video` 实现替换为调用新模块(删掉文件内的 `import av` 编码细节,保留常量与各用例):
```python
from feature_extractor.validation.synthetic import make_ramp_video

N_FRAMES = 12
STEP = 20
WIDTH, HEIGHT = 128, 96


def _make_video(path, codec):
    make_ramp_video(path, codec=codec, n_frames=N_FRAMES, width=WIDTH, height=HEIGHT, step=STEP)
```
其余 fixture / 用例不变。

- [ ] **Step 4: 跑测试确认仍全绿**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_video_io.py -q`
Expected: PASS(10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/validation/__init__.py src/feature_extractor/validation/synthetic.py tests/test_video_io.py
git commit -m "feat(validation): add synthetic video generators; reuse in video_io tests"
```

---

### Task 2: 纯几何工具 + `CheckResult`(`sanity.py` 第一部分,TDD)

**Files:**
- Create: `src/feature_extractor/validation/sanity.py`
- Test: `tests/test_validation_sanity.py`

- [ ] **Step 1: 写失败测试**

`tests/test_validation_sanity.py`:
```python
import numpy as np
import pytest

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


def test_checkresult_is_a_dataclass_with_fields():
    c = CheckResult(branch="dino", name="shape", expected="(T,N,384)",
                    observed="(4,1025,384)", passed=True)
    assert c.passed and c.branch == "dino"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_validation_sanity.py -q`
Expected: FAIL(ImportError: cannot import name ...)

- [ ] **Step 3: 写 `sanity.py` 第一部分**

`src/feature_extractor/validation/sanity.py`:
```python
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
    """6D 表示(R 的前两列)经 Gram-Schmidt 重建 3x3 旋转矩阵。"""
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_validation_sanity.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/validation/sanity.py tests/test_validation_sanity.py
git commit -m "feat(validation): add CheckResult and 6D-rotation geometry helpers"
```

---

### Task 3: 报告渲染 `report.py`(TDD)

**Files:**
- Create: `src/feature_extractor/validation/report.py`
- Test: `tests/test_validation_report.py`

> `PerfRecord` 在 Task 5 才正式定义于 `perf.py`;`report.py` 仅按属性名访问(鸭子类型),测试用一个等价的轻量对象即可,不产生导入依赖。

- [ ] **Step 1: 写失败测试**

`tests/test_validation_report.py`:
```python
from dataclasses import dataclass
from typing import Optional

from feature_extractor.validation.sanity import CheckResult
from feature_extractor.validation.report import render_report


@dataclass
class _Perf:
    video_id: str
    branch: str
    frames: int
    seconds: float
    fps: Optional[float]
    peak_mem_mb: Optional[float]
    note: str = ""


def test_render_report_has_sections_and_counts():
    meta = {"date": "2026-06-13", "commit": "abc1234", "gpu": "TestGPU",
            "torch": "2.5.0", "command": "feature-validate --x"}
    checks = [
        CheckResult("dino", "embed_dim", "384", "384", True),
        CheckResult("pose", "pose0=identity", "~I", "max|.|=0.3", False),
    ]
    perf = [_Perf("vid0", "dino", 64, 4.0, 16.0, 1200.0)]
    md = render_report(meta, checks, perf)

    assert "# feature_extractor 验证报告" in md
    assert "1/2" in md                 # 通过计数
    assert "FAIL" in md and "PASS" in md
    assert "pose0=identity" in md      # 失败项可见
    assert "TestGPU" in md             # 环境
    assert "feature-validate --x" in md  # 复现命令
    assert "16.0" in md                # fps
```

- [ ] **Step 2: 跑测试确认失败**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_validation_report.py -q`
Expected: FAIL(ImportError: render_report)

- [ ] **Step 3: 写 `report.py`**

`src/feature_extractor/validation/report.py`:
```python
"""把检查结果与性能记录渲染成交付用的 Markdown 报告。"""

from __future__ import annotations


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def render_report(meta: dict, checks: list, perf: list) -> str:
    n_pass = sum(1 for c in checks if c.passed)
    n_total = len(checks)
    overall = "PASS" if n_pass == n_total else "FAIL"

    lines: list[str] = []
    lines.append("# feature_extractor 验证报告")
    lines.append("")
    lines.append(f"- 日期:{meta.get('date', '')}")
    lines.append(f"- Git commit:`{meta.get('commit', '')}`")
    lines.append(f"- GPU:{meta.get('gpu', '')}　torch:{meta.get('torch', '')}"
                 f"　CUDA:{meta.get('cuda', '')}")
    lines.append(f"- 依赖:{meta.get('deps', '')}")
    lines.append("")

    # 摘要
    lines.append("## 摘要")
    lines.append("")
    lines.append(f"- 总体:**{overall}**　功能不变量 **{n_pass}/{n_total}** 通过")
    lines.append("")

    # 功能不变量
    lines.append("## 功能不变量")
    lines.append("")
    lines.append("| 分支 | 检查项 | 期望 | 实测 | 结果 |")
    lines.append("|------|--------|------|------|------|")
    for c in checks:
        result = "PASS" if c.passed else "FAIL"
        lines.append(f"| {c.branch} | {c.name} | {c.expected} | {c.observed} | {result} |")
    lines.append("")

    # 性能
    lines.append("## 性能")
    lines.append("")
    lines.append("| 视频 | 分支 | 帧数 | 耗时(s) | FPS | 峰值显存(MB) | 备注 |")
    lines.append("|------|------|------|---------|-----|--------------|------|")
    for p in perf:
        lines.append(
            f"| {p.video_id} | {p.branch} | {p.frames} | {_fmt(p.seconds)} | "
            f"{_fmt(p.fps)} | {_fmt(p.peak_mem_mb)} | {p.note} |"
        )
    lines.append("")

    # 复现
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append(meta.get("command", ""))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_validation_report.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/validation/report.py tests/test_validation_report.py
git commit -m "feat(validation): add Markdown report renderer"
```

---

### Task 4: 各分支 sanity 运行器(`sanity.py` 第二部分,集成)

**Files:**
- Modify: `src/feature_extractor/validation/sanity.py`(追加运行器)

> 这些函数调用真实模型,不写 pytest;由 Task 6 的 CLI 冒烟运行验证。

- [ ] **Step 1: 追加共享小工具与各分支运行器**

在 `sanity.py` 末尾追加:
```python
def _finite(arr) -> bool:
    return bool(np.all(np.isfinite(arr)))


def check_dino(features: np.ndarray) -> list[CheckResult]:
    f = np.asarray(features)
    out = [
        CheckResult("dino", "ndim==3", "3", str(f.ndim), f.ndim == 3),
        CheckResult("dino", "dtype==float32", "float32", str(f.dtype), f.dtype == np.float32),
        CheckResult("dino", "embed_dim==384", "384",
                    str(f.shape[-1]) if f.ndim == 3 else "n/a",
                    f.ndim == 3 and f.shape[-1] == 384),
        CheckResult("dino", "有限值", "all finite", str(_finite(f)), _finite(f)),
        CheckResult("dino", "含CLS(N≥2)", ">=2",
                    str(f.shape[1]) if f.ndim == 3 else "n/a",
                    f.ndim == 3 and f.shape[1] >= 2),
    ]
    if f.ndim == 3 and f.shape[1] >= 2:
        cls_diff = float(np.abs(f[:, 0] - f[:, 1:].mean(axis=1)).max())
        out.append(CheckResult("dino", "CLS≠patch均值", ">0",
                               f"{cls_diff:.3g}", cls_diff > 0))
    return out


def check_depth(inv_depth: np.ndarray) -> list[CheckResult]:
    d = np.asarray(inv_depth)
    nonneg = bool(np.all(d >= -1e-6))
    std = float(d.std())
    return [
        CheckResult("depth", "shape==(T,H,W,1)", "4D 末维1",
                    str(d.shape), d.ndim == 4 and d.shape[-1] == 1),
        CheckResult("depth", "dtype==float32", "float32", str(d.dtype), d.dtype == np.float32),
        CheckResult("depth", "逆深度≥0", ">=0", str(nonneg), nonneg),
        CheckResult("depth", "有限值", "all finite", str(_finite(d)), _finite(d)),
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
    tol = 2.0 / 65535.0
    err = float(np.abs(expected - np.asarray(read_back)).max()) if read_back.size else 1.0
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
```

- [ ] **Step 2: 语法/导入自检**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "import feature_extractor.validation.sanity as s; print([n for n in dir(s) if n.startswith('check')])"`
Expected: 打印 `['check_alignment', 'check_depth', 'check_depth_roundtrip', 'check_determinism', 'check_dino', 'check_exact_roundtrip', 'check_pose']`

- [ ] **Step 3: Commit**

```bash
git add src/feature_extractor/validation/sanity.py
git commit -m "feat(validation): add per-branch sanity invariant checks"
```

---

### Task 5: 性能基准 `perf.py`(集成)

**Files:**
- Create: `src/feature_extractor/validation/perf.py`

> 计时/显存依赖 GPU,不写 pytest;由 Task 6 的 CLI 冒烟运行验证。

- [ ] **Step 1: 写 `perf.py`**

`src/feature_extractor/validation/perf.py`:
```python
"""真实数据上的性能基准:吞吐、峰值显存、规模扩展。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PerfRecord:
    video_id: str
    branch: str          # 分支名,或 "model_load" / "decode"
    frames: int
    seconds: float
    fps: Optional[float] = None
    peak_mem_mb: Optional[float] = None
    note: str = ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return None


def measure_branch(extractor, video_id: str, video_path: str,
                   frame_indices: list, branch: str,
                   device: torch.device) -> PerfRecord:
    """计量单分支在给定帧上的耗时、FPS、峰值显存(含一次预热)。"""
    # 预热(前 2 帧),不计时
    warm = frame_indices[:2] if len(frame_indices) >= 2 else frame_indices
    extractor.extract_video(video_path, frame_indices=warm)
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    feats = extractor.extract_video(video_path, frame_indices=frame_indices)
    _sync(device)
    sec = time.perf_counter() - t0
    n = len(feats)
    return PerfRecord(video_id, branch, n, sec,
                      fps=(n / sec if sec > 0 else None),
                      peak_mem_mb=_peak_mb(device))


def measure_decode(video_id: str, video_path: str, frame_indices: list) -> PerfRecord:
    """单独计量解码耗时(读取指定帧),用于解码 vs 推理拆分。"""
    from feature_extractor.video_io import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    t0 = time.perf_counter()
    for i in frame_indices:
        _ = vr[int(i)].asnumpy()
    sec = time.perf_counter() - t0
    n = len(frame_indices)
    return PerfRecord(video_id, "decode", n, sec,
                      fps=(n / sec if sec > 0 else None), note="纯解码")
```

- [ ] **Step 2: 导入自检**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "from feature_extractor.validation.perf import PerfRecord, measure_branch, measure_decode; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/feature_extractor/validation/perf.py
git commit -m "feat(validation): add throughput/memory/decode perf measurements"
```

---

### Task 6: 编排 CLI `cli.py` + console entry + 冒烟验证

**Files:**
- Create: `src/feature_extractor/validation/cli.py`
- Modify: `pyproject.toml`(加 `feature-validate` entry)

- [ ] **Step 1: 写 `cli.py`**

`src/feature_extractor/validation/cli.py`:
```python
"""feature-validate:跑功能不变量 + 性能基准,生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import datetime
import socket
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from feature_extractor import DINOExtractor, DepthExtractor, PoseExtractor, FeatureStore
from feature_extractor.cli import find_videos, sample_frame_indices
from feature_extractor.validation import sanity, perf as perfmod
from feature_extractor.validation.report import render_report
from feature_extractor.validation.synthetic import make_gradient_video


def _env_meta(command: str) -> dict:
    def _safe(fn, default="?"):
        try:
            return fn()
        except Exception:
            return default
    gpu = _safe(lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    commit = _safe(lambda: subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip())
    deps = ", ".join(f"{p}={_safe(lambda p=p: metadata.version(p))}"
                     for p in ("av", "decord", "torch"))
    return {
        "date": datetime.date.today().isoformat(),
        "commit": commit,
        "host": _safe(socket.gethostname),
        "gpu": gpu,
        "cuda": _safe(lambda: torch.version.cuda),
        "torch": torch.__version__,
        "deps": deps,
        "command": command,
    }


def _build_extractors(branches, depth_mode, device, assets_root):
    dino = depth = pose = None
    if "dino" in branches or "depth" in branches:
        dino = DINOExtractor(model_name="dinov3_vits16plus", device=device, assets_root=assets_root)
    if "depth" in branches:
        depth = DepthExtractor(mode=depth_mode, device=device, dino_extractor=dino,
                               vda_input_size=224, assets_root=assets_root)
    if "pose" in branches:
        pose = PoseExtractor(device=device, assets_root=assets_root)
    return dino if "dino" in branches else None, depth, pose


def run_sanity(branches, depth_mode, device, assets_root) -> list:
    """在合成视频上跑功能不变量。重模型缺失则标 SKIPPED。"""
    checks: list = []
    with tempfile.TemporaryDirectory() as td:
        vid = str(Path(td) / "gradient.mp4")
        make_gradient_video(vid, n_frames=8)
        idx = list(range(8))
        store = FeatureStore(td)
        try:
            dino, depth, pose = _build_extractors(branches, depth_mode, device, assets_root)
        except Exception as e:
            checks.append(sanity.CheckResult("pipeline", "模型加载", "成功",
                                             f"SKIPPED: {e}", False))
            return checks

        indices_by_branch = {}
        if dino is not None and "dino" in branches:
            f = dino.extract_video(vid, frame_indices=idx)
            checks += sanity.check_dino(f)
            f2 = dino.extract_video(vid, frame_indices=idx)
            checks.append(sanity.check_determinism("dino", f, f2))
            store.write_dino("syn", f, frame_indices=np.array(idx))
            checks.append(sanity.check_exact_roundtrip("dino", f, store.read_dino("syn")))
            indices_by_branch["dino"] = store.read_frame_indices("syn", "dino")
        if depth is not None:
            d = depth.extract_video(vid, frame_indices=idx)
            checks += sanity.check_depth(d)
            d2 = depth.extract_video(vid, frame_indices=idx)
            checks.append(sanity.check_determinism("depth", d, d2))
            store.write_depth("syn", d, frame_indices=np.array(idx))
            checks.append(sanity.check_depth_roundtrip(d, store.read_depth("syn")))
            indices_by_branch["depth"] = store.read_frame_indices("syn", "depth")
        if pose is not None:
            pse = pose.extract_video(vid, frame_indices=idx)
            checks += sanity.check_pose(pse)
            store.write_pose("syn", pse, frame_indices=np.array(idx))
            checks.append(sanity.check_exact_roundtrip("pose", pse, store.read_pose("syn")))
            indices_by_branch["pose"] = store.read_frame_indices("syn", "pose")

        checks += sanity.check_alignment(indices_by_branch, idx)
    return checks


def run_perf(data_root, branches, depth_mode, device, perf_frames, sweep, assets_root) -> list:
    records: list = []
    videos = find_videos(data_root)
    if not videos:
        records.append(perfmod.PerfRecord("-", "perf", 0, 0.0, note="SKIPPED: 无视频"))
        return records
    video_path = videos[0]
    video_id = Path(video_path).stem

    import time
    t0 = time.perf_counter()
    dino, depth, pose = _build_extractors(branches, depth_mode, device, assets_root)
    records.append(perfmod.PerfRecord(video_id, "model_load", 0,
                                      time.perf_counter() - t0, note="一次性加载"))
    extractors = [("dino", dino), ("depth", depth), ("pose", pose)]

    frame_indices = sample_frame_indices(video_path, perf_frames)
    records.append(perfmod.measure_decode(video_id, video_path, frame_indices))
    for name, ex in extractors:
        if ex is None:
            continue
        records.append(perfmod.measure_branch(ex, video_id, video_path,
                                              frame_indices, name, device))
    # 扩展性扫描:用第一个可用分支
    name, ex = next(((n, e) for n, e in extractors if e is not None), (None, None))
    if ex is not None:
        for nf in sweep:
            idx = sample_frame_indices(video_path, nf)
            rec = perfmod.measure_branch(ex, f"{video_id}@{nf}", video_path, idx, name, device)
            rec.note = f"扫描 frames={nf}"
            records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="feature_extractor 自验证")
    parser.add_argument("--data_root", type=str, default=None, help="性能基准用真实数据目录")
    parser.add_argument("--report", type=str, default="validation_report.md")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--branches", type=str, default="dino,depth,pose")
    parser.add_argument("--depth_mode", type=str, default="video_depth_anything")
    parser.add_argument("--frames-sweep", type=str, default="16,32,64,128")
    parser.add_argument("--perf-frames", type=int, default=64)
    parser.add_argument("--assets_root", type=str, default=None)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    sweep = [int(x) for x in args.frames_sweep.split(",") if x.strip()]

    checks: list = []
    records: list = []
    if not args.skip_sanity:
        print("=== 功能不变量(合成视频)===")
        checks = run_sanity(branches, args.depth_mode, device, args.assets_root)
    if not args.skip_perf:
        if not args.data_root:
            parser.error("性能基准需要 --data_root(或加 --skip-perf)")
        print("=== 性能基准(真实数据)===")
        records = run_perf(args.data_root, branches, args.depth_mode, device,
                           args.perf_frames, sweep, args.assets_root)

    meta = _env_meta(" ".join(["feature-validate"] + _argv_tail()))
    md = render_report(meta, checks, records)
    Path(args.report).write_text(md, encoding="utf-8")
    n_pass = sum(1 for c in checks if c.passed)
    print(f"报告已写入 {args.report}　功能 {n_pass}/{len(checks)} 通过")


def _argv_tail() -> list:
    import sys
    return sys.argv[1:]


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 加 console entry**

`pyproject.toml` 的 `[project.scripts]` 段在 `feature-extract` 下加一行:
```toml
[project.scripts]
feature-extract = "feature_extractor.cli:main"
feature-validate = "feature_extractor.validation.cli:main"
```

- [ ] **Step 3: 同步并冒烟(仅 dino + 跳过性能,合成视频)**

Run:
```bash
uv sync
CUDA_VISIBLE_DEVICES=7 uv run feature-validate --branches dino --skip-perf --report /tmp/val_dino.md
```
Expected: 打印 `报告已写入 /tmp/val_dino.md　功能 N/N 通过`(dino 各项 PASS);`grep -c PASS /tmp/val_dino.md` > 0,无 Traceback。

- [ ] **Step 4: 冒烟报告内容自检**

Run: `grep -E "验证报告|功能不变量|PASS|dino" /tmp/val_dino.md | head`
Expected: 出现报告标题、表头与 dino 行。

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/validation/cli.py pyproject.toml uv.lock
git commit -m "feat(validation): add feature-validate CLI orchestration + report output"
```

---

### Task 7: 全配置端到端验证(交付配置)

**Files:** 无新增,仅运行验证产出样例报告。

- [ ] **Step 1: 全分支 + 性能,跑真实数据**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-validate \
    --data_root data/libero_10/videos \
    --branches dino,depth,pose \
    --depth_mode video_depth_anything \
    --perf-frames 64 --frames-sweep 16,32,64,128 \
    --report /tmp/val_full.md
```
Expected: 无 Traceback;功能项绝大多数 PASS;`/tmp/val_full.md` 含性能表(decode、dino/depth/pose 三行、扫描 4 行)。

- [ ] **Step 2: 人读报告,确认数字合理**

Run: `cat /tmp/val_full.md`
Expected: pose[0] 误差 < 1e-3;depth 往返误差 < ~3e-5;FPS/显存为正且随 frames 扫描单调;无空白/NaN。

- [ ] **Step 3: 跑全套单测确认无回归**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/ -q`
Expected: 全部通过(原 16 + 新增 sanity/report 用例)。

- [ ] **Step 4(可选): 把样例报告纳入交付**

如需随仓库交付一份样例报告:
```bash
mkdir -p docs/validation
cp /tmp/val_full.md docs/validation/sample_report.md
git add docs/validation/sample_report.md
git commit -m "docs(validation): add sample validation report"
```
