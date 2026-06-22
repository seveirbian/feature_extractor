# Depth Pro HF-only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Depth Pro 后端换成 HuggingFace `transformers.DepthProForDepthEstimation`,删除 vendored `third_party/ml-depth-pro` 源码;`_extract_depth_pro` 输出契约(米制 `(H,W)` float32)与调用方不变。

**Architecture:** 仅改 `depth.py` 的 `_load_depth_pro`(HF 加载,本地 HF 目录)与 `_extract_depth_pro`(HF 推理),并把 `self.depth_pro_transform` 改名 `self.depth_pro_processor`;`_metricize_vda_depth_sequence`/`_fit_depth_affine`/独立 depth_pro 模式不变。删 vendored ml-depth-pro 源码 + 更新 PROVENANCE/README。VGGT/VDA 不动。

**Tech Stack:** transformers(已在依赖,`DepthProForDepthEstimation` + `AutoImageProcessor`)、torch、pytest。

> **执行建议:Inline**(跨 depth.py 多点 + 删 vendored + 条件性 e2e)。

---

### Task 1: depth.py → HF Depth Pro(loader + 推理 + 改名)

**Files:**
- Modify: `src/feature_extractor/extractors/depth.py`
- Test: `tests/test_depthpro_hf.py`

- [ ] **Step 1: 写失败测试(缺权重目录 → FileNotFoundError;不需真模型)**

`tests/test_depthpro_hf.py`:
```python
"""Depth Pro HF 后端:缺本地 HF 权重目录时明确报错。"""

import tempfile
import pytest

from feature_extractor.extractors.depth import _load_depth_pro
import torch


def test_load_depth_pro_missing_hf_dir_raises():
    # assets_root 指向空目录 → 期望目录不存在的 FileNotFoundError
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FileNotFoundError):
            _load_depth_pro(torch.device("cpu"), assets_root=td)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_depthpro_hf.py -q`
Expected: FAIL（当前 `_load_depth_pro` 走 vendored,缺目录不抛 FileNotFoundError 而是返回 (None,None) 或别的)

- [ ] **Step 3: 替换 `_load_depth_pro` 为 HF 版**

把整个 `def _load_depth_pro(device, assets_root=None): ...` 函数替换为:
```python
def _load_depth_pro(device: torch.device, assets_root=None):
    """Load Apple Depth Pro via HuggingFace transformers from a local HF-format dir."""
    try:
        from transformers import AutoImageProcessor, DepthProForDepthEstimation
    except ImportError as e:
        raise RuntimeError(
            "Depth Pro HF backend 需要 transformers(>=4.56)。请 `uv sync` 或 `pip install transformers`。"
        ) from e
    hf_dir = resolve_assets_root(assets_root) / "third_party" / "ml-depth-pro" / "checkpoints" / "depth-pro-hf"
    if not hf_dir.exists():
        raise FileNotFoundError(
            f"Depth Pro HF weights dir not found: {hf_dir}. "
            "请放入 HF 格式权重(config.json + safetensors),来源 HF 仓库 apple/DepthPro-hf。"
        )
    processor = AutoImageProcessor.from_pretrained(str(hf_dir))
    model = DepthProForDepthEstimation.from_pretrained(str(hf_dir), use_fov_model=False)
    model = model.to(device)
    model.eval()
    print(f"[DepthExtractor] Loaded Depth Pro (HF): {hf_dir}")
    return model, processor
```

- [ ] **Step 4: __init__ 改名 `depth_pro_transform` → `depth_pro_processor`**

把 `__init__` 中:
```python
        self.depth_pro_model = None
        self.depth_pro_transform = None
```
改为:
```python
        self.depth_pro_model = None
        self.depth_pro_processor = None
```
把:
```python
        if self.mode in ("da3", "video_depth_anything", "depth_pro"):
            self.depth_pro_model, self.depth_pro_transform = _load_depth_pro(self.device, self.assets_root)
```
改为:
```python
        if self.mode in ("da3", "video_depth_anything", "depth_pro"):
            self.depth_pro_model, self.depth_pro_processor = _load_depth_pro(self.device, self.assets_root)
```
把 VDA 必需检查:
```python
            if self.depth_pro_model is None or self.depth_pro_transform is None:
```
改为:
```python
            if self.depth_pro_model is None or self.depth_pro_processor is None:
```

- [ ] **Step 5: 替换 `_extract_depth_pro` 为 HF 推理**

把整个 `def _extract_depth_pro(self, image): ...` 方法替换为:
```python
    def _extract_depth_pro(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Pro (HF) on a single frame for metric depth, at original resolution."""
        from PIL import Image

        if self.depth_pro_processor is None:
            return np.ones_like(image[..., 0], dtype=np.float32) * 5.0

        image_uint8 = image if image.dtype == np.uint8 else image.astype(np.uint8)
        if self.input_color == "bgr" and image_uint8.shape[2] == 3:
            image_rgb = image_uint8[..., ::-1]
        else:
            image_rgb = image_uint8

        pil_img = Image.fromarray(np.ascontiguousarray(image_rgb))
        inputs = self.depth_pro_processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_pro_model(**inputs)
        post = self.depth_pro_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.shape[0], image.shape[1])]
        )
        depth = post[0]["predicted_depth"]
        return depth.detach().cpu().numpy().astype(np.float32)
```

- [ ] **Step 6: docstring 更新**

把文件顶部 `"""Depth feature extractor using Video Depth Anything + Depth Pro. ...`(及类 docstring)中
"Depth Pro" 相关描述无需大改;若有"vendored"/"local repo"措辞指 Depth Pro 的,改为 "Depth Pro (HuggingFace)"。
(不强制;保持准确即可。)

- [ ] **Step 7: 跑测试 + 残留检查**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_depthpro_hf.py -q` → PASS。
Run: `grep -nE "depth_pro_transform|create_model_and_transforms|ml-depth-pro/src|\.infer\(" src/feature_extractor/extractors/depth.py || echo clean` → `clean`。
Run 全套: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q` → 全过。

- [ ] **Step 8: Commit**

```bash
git add src/feature_extractor/extractors/depth.py tests/test_depthpro_hf.py
git commit -m "refactor(depth): load Depth Pro via HuggingFace transformers"
```

---

### Task 2: 删除 vendored ml-depth-pro 源码 + PROVENANCE

**Files:**
- Delete: `third_party/ml-depth-pro/src`、`third_party/ml-depth-pro/LICENSE`
- Keep: `third_party/ml-depth-pro/checkpoints/`
- Modify: `third_party/PROVENANCE.md`

- [ ] **Step 1: 删 vendored 代码,保留 checkpoints**

```bash
cd /root/codes/feature_extractor
git rm -r --quiet third_party/ml-depth-pro/src third_party/ml-depth-pro/LICENSE
rm -rf third_party/ml-depth-pro/src   # 清理可能残留的 __pycache__
ls third_party/ml-depth-pro/          # 应只剩 checkpoints
find third_party/ml-depth-pro -name '*.py' | wc -l   # 期望 0
```

- [ ] **Step 2: PROVENANCE 去掉 ml-depth-pro 行**

编辑 `third_party/PROVENANCE.md`,删掉表格里 `ml-depth-pro` 那一行(保留 VGGT、Video-Depth-Anything)。

- [ ] **Step 3: 确认无代码引用 vendored depth_pro**

Run: `grep -rnE "ml-depth-pro/src|import depth_pro|from depth_pro|create_model_and_transforms" src/feature_extractor || echo clean`
Expected: `clean`

- [ ] **Step 4: Commit**

```bash
git add -A third_party/
git commit -m "build: drop vendored ml-depth-pro sources (Depth Pro now via transformers)"
```

---

### Task 3: README 更新(Depth Pro 归入 HF)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: §3 模型资源:vendored 列表去掉 ml-depth-pro**

把第 3 节里 vendored 三仓的句子改为只列两仓:
`**VGGT / Video-Depth-Anything**:精简后的 vendored 源码内置在 `third_party/<repo>/`(非子模块;裁剪溯源见 `third_party/PROVENANCE.md`)。`
并在 DINO 那条之后补一句:`- **Depth Pro**:走 HuggingFace transformers,权重为本地 HF 格式目录,见第 7 节。`

- [ ] **Step 2: §3 OBS 下载去掉 ml-depth-pro 行**

删除 `obsutil cp ... /ml-depth-pro/checkpoints ...` 那一行;只保留 Video-Depth-Anything 与 VGGT 两行。
把"DINO(HF)权重单独获取"那句扩为"DINO 与 Depth Pro(HF)权重单独获取,见第 6/7 节"。
"只用部分后端"那句把 depth 的依赖改为 "默认 depth `video_depth_anything` 需 `Video-Depth-Anything`(vendored)+ Depth Pro(HF 权重)"。

- [ ] **Step 3: §7 深度模式:Depth Pro 权重说明改 HF**

在第 7 节深度模式表下补一段:
```markdown
`video_depth_anything`(默认)在关键帧用 **Depth Pro** 做米制校正,`depth_pro` 模式直接用 Depth Pro。
Depth Pro 现走 HuggingFace `transformers`,需本地 HF 格式权重目录:

```bash
uv run hf download apple/DepthPro-hf \
    --local-dir third_party/ml-depth-pro/checkpoints/depth-pro-hf
```
```

- [ ] **Step 4: 残留检查 + Commit**

Run: `grep -nE "ml-depth-pro/checkpoints.*obsutil|depth_pro.*\.pt\b" README.md || echo "check ok"`(确认 OBS 行已去)
```bash
git add README.md
git commit -m "docs: Depth Pro via HF; drop ml-depth-pro from vendored/OBS"
```

---

### Task 4: 验证

- [ ] **Step 1: 全套单测**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全过(含新 `test_depthpro_hf.py`)。

- [ ] **Step 2: 确认 vendored ml-depth-pro 彻底无引用**

Run:
```bash
find third_party/ml-depth-pro -name '*.py' | wc -l   # 0
grep -rnE "ml-depth-pro/src|create_model_and_transforms|depth_pro_transform" src/feature_extractor || echo clean
```
Expected: `0` 和 `clean`。

- [ ] **Step 3(有真权重时):端到端**

若 `third_party/ml-depth-pro/checkpoints/depth-pro-hf/config.json` 存在:
```bash
CUDA_VISIBLE_DEVICES=7 uv run python - <<'PY'
import numpy as np, torch
from feature_extractor.extractors.depth import DepthExtractor
ex = DepthExtractor(mode="depth_pro", device="cuda")
img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
d = ex._extract_depth_pro(img)
print("depth shape", d.shape, "dtype", d.dtype, "finite", bool(np.isfinite(d).all()))
assert d.shape == (120, 160) and d.dtype == np.float32
print("Depth Pro HF OK")
PY
```
Expected: `(120, 160) float32 finite True` → `Depth Pro HF OK`。
若无权重目录:跳过并说明(`_load_depth_pro` 会抛 FileNotFoundError,符合预期);Task 1 的缺目录单测已覆盖错误路径。

- [ ] **Step 4: VGGT/VDA 未受影响(导入冒烟)**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "import feature_extractor.extractors.depth, feature_extractor.extractors.pose; print('import ok')"`
Expected: `import ok`。
