# DINOv3 HuggingFace backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `DINOExtractor` 增加基于 HuggingFace transformers 的 DINOv3 可选后端(`dinov3_vits16_hf` / `dinov3_vits16plus_hf`),与现有 vendored 实现并存,输出契约一致 `(T, 1025, 384)`。

**Architecture:** 在现有 `MODEL_CONFIGS` 加 `family="dinov3_hf"` 两个条目;`_load_model` 增分支调 `_load_dinov3_hf`(本地 HF 目录、懒导入 transformers);`_extract_tokens` 增 HF 分支,用纯函数 `_slice_hf_tokens` 切掉 register tokens 还原 `[CLS]+patches`。预处理/L2-norm/extract_video 全复用。

**Tech Stack:** transformers>=4.56(已验证解析得 5.12.1,与 numpy<2/torch-cu124 兼容)、torch、pytest。

**已验证前提:** `uv pip compile` 显示 `transformers==5.12.1` 与 `numpy==1.26.4` 共存解析成功。

---

### Task 1: 加入 transformers 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 在 dependencies 加 transformers**

`pyproject.toml` 的 `dependencies` 列表里,在 `"timm>=1.0.0",` 之后加一行:
```toml
    # DINOv3 HuggingFace 可选后端(family=dinov3_hf);DINOv3 自 transformers 4.56 起原生支持
    "transformers>=4.56",
```

- [ ] **Step 2: 同步并确认可导入**

Run: `uv sync 2>&1 | tail -3 && CUDA_VISIBLE_DEVICES="" uv run python -c "import transformers, numpy; print('transformers', transformers.__version__, '| numpy', numpy.__version__)"`
Expected: 打印 transformers 版本(5.x)与 `numpy 1.26.x`;无解析/安装错误。

- [ ] **Step 3: 确认 DINOv3 类可导入**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "from transformers import AutoModel, DINOv3ViTModel, DINOv3ViTConfig; print('ok')"`
Expected: `ok`

- [ ] **Step 4: 全套测试无回归**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全部通过(28)。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add transformers>=4.56 for DINOv3 HF backend"
```

---

### Task 2: 加 HF MODEL_CONFIGS 条目(TDD)

**Files:**
- Modify: `src/feature_extractor/extractors/dino.py`
- Test: `tests/test_dino_hf_config.py`

- [ ] **Step 1: 写失败测试**

`tests/test_dino_hf_config.py`:
```python
"""dinov3_*_hf 后端配置的纯单测(不加载模型/权重)。"""

from feature_extractor.extractors.dino import DINOExtractor


def test_hf_configs_present():
    cfgs = DINOExtractor.MODEL_CONFIGS
    for name in ("dinov3_vits16_hf", "dinov3_vits16plus_hf"):
        assert name in cfgs, name
        c = cfgs[name]
        assert c["family"] == "dinov3_hf"
        assert c["embed_dim"] == 384
        assert c["patch_size"] == 16
        assert c["img_size"] == 512
        assert "checkpoints" in c["weights"]


def test_hf_does_not_change_vendored_aliases():
    # HF 风格别名仍指向 vendored,不被 _hf 抢占
    al = DINOExtractor.MODEL_ALIASES
    assert al["facebook/dinov3-vits16-pretrain-lvd1689m"] == "dinov3_vits16"
    assert al["facebook/dinov3-vits16plus-pretrain-lvd1689m"] == "dinov3_vits16plus"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_dino_hf_config.py -q`
Expected: FAIL（KeyError: 'dinov3_vits16_hf'）

- [ ] **Step 3: 加两个配置条目**

在 `dino.py` 的 `MODEL_CONFIGS` 中,`dinov3_vits16` 条目之后加入:
```python
        "dinov3_vits16_hf": {
            "family": "dinov3_hf",
            "img_size": 512,
            "patch_size": 16,
            "embed_dim": 384,
            "weights": "third_party/dinov3/checkpoints/dinov3-vits16-hf",
        },
        "dinov3_vits16plus_hf": {
            "family": "dinov3_hf",
            "img_size": 512,
            "patch_size": 16,
            "embed_dim": 384,
            "weights": "third_party/dinov3/checkpoints/dinov3-vits16plus-hf",
        },
```

- [ ] **Step 4: 跑测试确认通过**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_dino_hf_config.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor/extractors/dino.py tests/test_dino_hf_config.py
git commit -m "feat(dino): add dinov3_*_hf MODEL_CONFIGS entries"
```

---

### Task 3: HF 加载器 + token 切片(TDD)

**Files:**
- Modify: `src/feature_extractor/extractors/dino.py`
- Test: `tests/test_dino_hf_tokens.py`

- [ ] **Step 1: 写失败测试(纯切片 + 随机 HF 模型布局)**

`tests/test_dino_hf_tokens.py`:
```python
"""DINOv3 HF 后端的 token 切片契约测试(不需真权重)。"""

import torch

from feature_extractor.extractors.dino import DINOExtractor


def test_slice_hf_tokens_drops_register():
    # 构造 (1, 1 + R + P, D);R=4 register,P=6 patch
    R, P, D = 4, 6, 8
    h = torch.arange(1 * (1 + R + P) * D, dtype=torch.float32).reshape(1, 1 + R + P, D)
    out = DINOExtractor._slice_hf_tokens(h, R)
    assert out.shape == (1, 1 + P, D)          # CLS + patches,register 被剔除
    assert torch.equal(out[:, 0, :], h[:, 0, :])           # CLS 保留
    assert torch.equal(out[:, 1:, :], h[:, 1 + R:, :])     # patches = 跳过 register 之后


def test_hf_model_output_layout_and_slice():
    # 用随机初始化的真实 DINOv3ViT 验证布局假设:512 输入 → 1+R+1024,切片得 1025
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, patch_size=16, image_size=512, num_register_tokens=4,
    )
    model = DINOv3ViTModel(cfg).eval()
    px = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        h = model(pixel_values=px).last_hidden_state
    P = (512 // 16) * (512 // 16)              # 1024
    assert h.shape == (1, 1 + 4 + P, cfg.hidden_size)
    sliced = DINOExtractor._slice_hf_tokens(h, cfg.num_register_tokens)
    assert sliced.shape == (1, 1 + P, cfg.hidden_size)   # 1025
```

- [ ] **Step 2: 跑测试确认失败**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_dino_hf_tokens.py -q`
Expected: FAIL（AttributeError: _slice_hf_tokens）

- [ ] **Step 3: 实现切片静态方法 + 加载器 + _extract_tokens 分支 + 调度**

在 `dino.py` 加静态方法(放在 `_extract_tokens` 之前):
```python
    @staticmethod
    def _slice_hf_tokens(last_hidden_state: torch.Tensor, num_register: int) -> torch.Tensor:
        """HF DINOv3 输出 [CLS, R×register, patches] → 还原 [CLS, patches](剔除 register)。"""
        cls = last_hidden_state[:, 0:1, :]
        patches = last_hidden_state[:, 1 + num_register:, :]
        return torch.cat([cls, patches], dim=1)
```

加 HF 加载器(放在 `_load_dinov3` 之后):
```python
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
```

在 `_load_model` 的分支里加 `dinov3_hf`(把现有 `if cfg["family"] == "dinov3":` 改为含 elif):
```python
        if cfg["family"] == "dinov3":
            model = self._load_dinov3(cfg)
        elif cfg["family"] == "dinov3_hf":
            model = self._load_dinov3_hf(cfg)
        else:
```

在 `_extract_tokens` 开头加 HF 分支(在现有 `batch = image_tensor.unsqueeze(0)` 之后、`if hasattr(...forward_features)` 之前):
```python
        if self.family == "dinov3_hf":
            out = self.model(pixel_values=batch)
            num_register = int(getattr(self.model.config, "num_register_tokens", 0))
            return self._slice_hf_tokens(out.last_hidden_state, num_register)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_dino_hf_tokens.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 全套测试无回归**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全部通过。

- [ ] **Step 6: Commit**

```bash
git add src/feature_extractor/extractors/dino.py tests/test_dino_hf_tokens.py
git commit -m "feat(dino): add DINOv3 HF loader + register-token slicing"
```

---

### Task 4: README 文档

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在「DINO backbone」表加 HF 变体**

在 README 第 6 章「DINO backbone(`--dino_model`)」的表格里,`dinov3_vits16` 行之后加两行:
```markdown
| `dinov3_vits16_hf` | DINOv3 ViT-S/16(HuggingFace 后端) | 本地 HF 格式目录 `third_party/dinov3/checkpoints/dinov3-vits16-hf/` |
| `dinov3_vits16plus_hf` | DINOv3 ViT-S+/16(HuggingFace 后端) | 本地 HF 格式目录 `third_party/dinov3/checkpoints/dinov3-vits16plus-hf/` |
```

- [ ] **Step 2: 在该节末尾追加 HF 后端说明**

在「DINO backbone」节末尾(`获取权重` 段之后)追加:
```markdown
### HuggingFace 后端(可选)

`*_hf` 变体用 `transformers` 的原生 DINOv3 实现,输出与 vendored 同契约
(`(T, 1025, 384)`,已剔除 register tokens)。需要:

- `transformers`(已在依赖中);
- **本地 HF 格式权重目录**(`config.json` + `*.safetensors`)放在上表对应路径——
  可从 HF 仓库 `facebook/dinov3-vits16-pretrain-lvd1689m`(gated)下载后置于该目录,离线加载。

```bash
uv run feature-extract --branches dino --dino_model dinov3_vits16_hf ...
```

vendored 后端(`dinov3_vits16` / `dinov3_vits16plus`)与 `facebook/*` 别名行为不变。
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document DINOv3 HF optional backend"
```

---

### Task 5: 端到端验证(有真权重时)+ 收尾

**Files:** 无新增。

- [ ] **Step 1: 检测是否有本地 HF 权重目录**

Run: `ls -d third_party/dinov3/checkpoints/dinov3-vits16-hf 2>/dev/null && echo PRESENT || echo ABSENT`

- [ ] **Step 2(仅当 PRESENT): 端到端跑 HF 后端**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-validate \
  --branches dino --dino_model dinov3_vits16_hf --skip-perf --report /tmp/hf_val.md 2>&1 | grep -iE "Loaded|功能|Error" | tail
echo "EXIT ${PIPESTATUS[0]}"; grep -c PASS /tmp/hf_val.md
```
Expected: 加载 HF 后端、DINO 各项 PASS、输出形状 `(T,1025,384)`、EXIT 0。
> 注:`feature-validate` 当前硬编码 vits16plus,需临时用直连脚本验证:
> ```bash
> CUDA_VISIBLE_DEVICES=7 uv run python -c "
> from feature_extractor.extractors.dino import DINOExtractor
> from feature_extractor.validation.synthetic import make_gradient_video
> from feature_extractor.validation import sanity
> import tempfile, pathlib
> ex=DINOExtractor(model_name='dinov3_vits16_hf', device='cuda')
> with tempfile.TemporaryDirectory() as td:
>     v=str(pathlib.Path(td)/'g.mp4'); make_gradient_video(v,n_frames=6)
>     f=ex.extract_video(v, frame_indices=list(range(6)))
> print('shape', f.shape)
> assert all(c.passed for c in sanity.check_dino(f)), 'sanity failed'
> print('HF backend OK')
> "
> ```

- [ ] **Step 2(仅当 ABSENT): 记录跳过**

说明:无本地 HF 权重目录,跳过端到端;配置 + 切片单测已覆盖 HF 路径契约。要真跑需按 README 放置
`third_party/dinov3/checkpoints/dinov3-vits16-hf/`(HF 格式)。

- [ ] **Step 3: 全套测试最终确认**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全部通过。
