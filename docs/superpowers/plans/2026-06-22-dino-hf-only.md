# DINO HF-only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DINO 后端只保留 HuggingFace 实现:删除 vendored dinov3、dinov2-hub、timm fallback、`dino_attention` 深度模式,以及 `third_party/dinov3` 的 vendored 源码。

**Architecture:** `dino.py` 精简为 HF-only(两个配置 `dinov3_vits16`/`dinov3_vits16plus`,family `dinov3_hf`);`depth.py`/`cli.py` 去掉 `dino_attention`,默认 depth_mode 改 `video_depth_anything`;删除 `third_party/dinov3` 的 python 包(保留 checkpoints)。VGGT/VDA/depth_pro 不动。

**Tech Stack:** transformers(已在依赖)、torch、pytest。

> **执行建议:Inline**。改动跨多文件、有删整段函数与 import 验证,需会话连续性。

---

### Task 1: dino.py → HF-only(配置/别名/加载/token)+ 测试

**Files:**
- Modify: `src/feature_extractor/extractors/dino.py`
- Modify: `tests/test_dino_config.py`(重写)
- Delete: `tests/test_dino_hf_config.py`(并入 test_dino_config.py)
- Keep: `tests/test_dino_hf_tokens.py`(切片测试,`_slice_hf_tokens` 不变)

- [ ] **Step 1: 重写 tests/test_dino_config.py 为 HF-only 断言**

整文件替换为:
```python
"""DINO 后端为 HF-only 的配置/别名断言(不加载模型)。"""

from feature_extractor.extractors.dino import DINOExtractor


def test_only_hf_dinov3_configs():
    cfgs = DINOExtractor.MODEL_CONFIGS
    assert set(cfgs) == {"dinov3_vits16", "dinov3_vits16plus"}
    for name in cfgs:
        c = cfgs[name]
        assert c["family"] == "dinov3_hf"
        assert c["embed_dim"] == 384
        assert c["patch_size"] == 16
        assert c["img_size"] == 512
        assert "checkpoints" in c["weights"] and ".pth" not in c["weights"]


def test_vendored_and_dinov2_removed():
    cfgs = DINOExtractor.MODEL_CONFIGS
    assert "dinov3_vits16_hf" not in cfgs and "dinov3_vits16plus_hf" not in cfgs
    assert not any(n.startswith("dinov2") for n in cfgs)


def test_aliases_point_to_hf():
    al = DINOExtractor.MODEL_ALIASES
    assert al["facebook/dinov3-vits16-pretrain-lvd1689m"] == "dinov3_vits16"
    assert al["facebook/dinov3-vits16plus-pretrain-lvd1689m"] == "dinov3_vits16plus"
    assert not any(v.startswith("dinov2") for v in al.values())
```

- [ ] **Step 2: 删除 tests/test_dino_hf_config.py**

Run: `git rm tests/test_dino_hf_config.py`

- [ ] **Step 3: 跑测试确认失败**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_dino_config.py -q`
Expected: FAIL（当前 MODEL_CONFIGS 仍含 vendored/dinov2/_hf 名)

- [ ] **Step 4: 替换 MODEL_CONFIGS(只留两条 HF)**

把 `dino.py` 中从 `"dinov3_vits16plus": {` 到 `dinov2_vits14` 条目结束的整个块,替换为:
```python
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
```

- [ ] **Step 5: 替换 MODEL_ALIASES(去 dinov2、去 `_hf` 旧名、指向新名)**

把整个 `MODEL_ALIASES = { ... }` 替换为:
```python
    MODEL_ALIASES = {
        "facebook/dinov3-vits16plus-pretrain-lvd1689m": "dinov3_vits16plus",
        "facebook/dinov3-vits16plus": "dinov3_vits16plus",
        "dinov3-vits16plus": "dinov3_vits16plus",
        "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
        "facebook/dinov3-vits16": "dinov3_vits16",
        "dinov3-vits16": "dinov3_vits16",
    }
```

- [ ] **Step 6: _load_model 只剩 HF 路径**

把 `_load_model` 中的分发块:
```python
        if cfg["family"] == "dinov3":
            model = self._load_dinov3(cfg)
        elif cfg["family"] == "dinov3_hf":
            model = self._load_dinov3_hf(cfg)
        else:
            try:
                hub_repo = cfg["hub_repo"]
                model = torch.hub.load(hub_repo, self.model_name)
                print(f"[DINOExtractor] Loaded via torch.hub: {hub_repo}/{self.model_name}")
            except Exception as e:
                print(f"[DINOExtractor] torch.hub failed: {e}, trying alternative...")
                model = self._load_alternative()
```
替换为:
```python
        model = self._load_dinov3_hf(cfg)
```

- [ ] **Step 7: 删除 `_load_dinov3`(vendored)与 `_load_alternative`(timm)两个方法**

删除 `def _load_dinov3(self, cfg)`(整个方法)与 `def _load_alternative(self)`(整个方法)。保留 `_load_dinov3_hf`。

- [ ] **Step 8: _extract_tokens 只剩 HF 逻辑**

把整个 `_extract_tokens` 方法体替换为:
```python
    def _extract_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Return DINO tokens as (1, N, D) = [CLS] + patches(剔除 register)。"""
        batch = image_tensor.unsqueeze(0)
        out = self.model(pixel_values=batch)
        num_register = int(getattr(self.model.config, "num_register_tokens", 0))
        return self._slice_hf_tokens(out.last_hidden_state, num_register)
```

- [ ] **Step 9: 跑 dino 配置 + 切片测试确认通过**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest tests/test_dino_config.py tests/test_dino_hf_tokens.py -q`
Expected: PASS

- [ ] **Step 10: 确认 import / 残留**

Run: `grep -nE "_load_dinov3\b|_load_alternative|dinov2|torch.hub|forward_features|x_norm_" src/feature_extractor/extractors/dino.py || echo "clean"`
Expected: `clean`（无 vendored/dinov2/timm 残留)

- [ ] **Step 11: Commit**

```bash
git add src/feature_extractor/extractors/dino.py tests/test_dino_config.py tests/test_dino_hf_config.py
git commit -m "refactor(dino): HF-only backend (drop vendored/dinov2/timm)"
```

---

### Task 2: depth.py / cli.py 去掉 dino_attention

**Files:**
- Modify: `src/feature_extractor/extractors/depth.py`
- Modify: `src/feature_extractor/cli.py`

- [ ] **Step 1: 删除 `_depth_from_dino_attention` 函数**

删除 `depth.py` 顶部的注释块 `# Fallback: MiDaS-style depth from DINO attention (no GPU needed)` 起、到 `def _depth_from_dino_attention(...)` 整个函数结束(即到 `class DepthExtractor:` 之前)的所有内容。

- [ ] **Step 2: __init__ 改默认 mode、去 dino_attention 类型与 dino_extractor**

把 `__init__` 签名行:
```python
        mode: Literal["video_depth_anything", "vda", "da3", "depth_pro", "dino_attention"] = "da3",
```
改为:
```python
        mode: Literal["video_depth_anything", "vda", "da3", "depth_pro"] = "video_depth_anything",
```
删除签名里的 `dino_extractor=None,` 一行;删除体内 `self.dino_extractor = dino_extractor` 一行。

- [ ] **Step 3: _load_model 的 da3 分支不再回退 dino_attention**

把:
```python
            if model is None:
                _warn_once(
                    "depth_da3_to_dino_attention",
                    "Depth fallback: DA3 is unavailable; switching extraction mode to DINO attention proxy.",
                )
                print("[DepthExtractor] Falling back to dino_attention mode.")
                self.mode = "dino_attention"
            else:
                model = model.to(self.device)
                model.eval()
                return model
```
替换为:
```python
            if model is None:
                raise RuntimeError("Depth Anything V3 (da3) 不可用;请改用 --depth_mode video_depth_anything。")
            model = model.to(self.device)
            model.eval()
            return model
```

- [ ] **Step 4: 帧分发去掉 dino_attention 兜底**

把帧分发的 `else:` 兜底块(从 `else:` 到 `depth_map = self._extract_dino_attention(image, h, w)`):
```python
        else:
            # DINO attention proxy. ...(整段注释与 _warn_once 分支)...
            depth_map = self._extract_dino_attention(image, h, w)
```
替换为:
```python
        else:
            raise RuntimeError(
                f"depth mode={self.mode!r} 没有可用模型;支持 video_depth_anything / da3 / depth_pro。"
            )
```

- [ ] **Step 5: 删除 `_extract_dino_attention` 方法**

删除 `def _extract_dino_attention(self, image, h, w)` 整个方法。

- [ ] **Step 6: docstring 清理**

删除类 docstring 中 `- "dino_attention": DINO attention as proxy (fallback, no GPU)` 一行(若存在)。

- [ ] **Step 7: cli.py 改默认 depth_mode、去掉 dino_extractor 传参**

`cli.py`:把
```python
    parser.add_argument("--depth_mode", type=str, default="dino_attention",
```
改为
```python
    parser.add_argument("--depth_mode", type=str, default="video_depth_anything",
```
并删除 `DepthExtractor(` 构造里的 `dino_extractor=extractor_dino,` 一行。

- [ ] **Step 8: 语法/残留自检**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "import feature_extractor.extractors.depth, feature_extractor.cli; print('import ok')" && grep -rnE "dino_attention|_depth_from_dino_attention|_extract_dino_attention" src/feature_extractor || echo clean`
Expected: `import ok` 然后 `clean`

- [ ] **Step 9: 全套测试**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全部通过。

- [ ] **Step 10: Commit**

```bash
git add src/feature_extractor/extractors/depth.py src/feature_extractor/cli.py
git commit -m "refactor(depth): remove dino_attention mode; default depth_mode=video_depth_anything"
```

---

### Task 3: 删除 vendored dinov3 源码 + 更新 PROVENANCE

**Files:**
- Delete: `third_party/dinov3/dinov3/`(vendored python 包)、`third_party/dinov3/LICENSE.md`
- Keep: `third_party/dinov3/checkpoints/`(HF 权重,gitignored)
- Modify: `third_party/PROVENANCE.md`

- [ ] **Step 1: 删除 vendored 代码,保留 checkpoints**

```bash
cd /root/codes/feature_extractor
git rm -r third_party/dinov3/dinov3 third_party/dinov3/LICENSE.md
ls third_party/dinov3/   # 应只剩 checkpoints(软链/目录,gitignored)
```

- [ ] **Step 2: PROVENANCE 去掉 dinov3 行**

编辑 `third_party/PROVENANCE.md`,删除表格里 dinov3 那一行(保留 VGGT / Video-Depth-Anything / ml-depth-pro 三行)。

- [ ] **Step 3: 确认没有代码再触发 vendored dinov3 导入**

Run: `grep -rnE "from dinov3|import dinov3|dinov3.hub|local_repo" src/feature_extractor || echo "clean"`
Expected: `clean`

- [ ] **Step 4: Commit**

```bash
git add -A third_party/
git commit -m "build: drop vendored dinov3 sources (DINO now via transformers)"
```

---

### Task 4: README 改为 DINO HF-only

**Files:**
- Modify: `README.md`

- [ ] **Step 1: DINO backbone 表只留两名(HF)**

把「DINO backbone」表内 4 行(含 `_hf`)替换为 2 行:
```markdown
| `dinov3_vits16plus`(默认) | DINOv3 ViT-S+/16(HuggingFace) | 本地 HF 格式目录 `dinov3-vits16plus-hf/` |
| `dinov3_vits16` | DINOv3 ViT-S/16(HuggingFace) | 本地 HF 格式目录 `dinov3-vits16-hf/` |
```

- [ ] **Step 2: 删除 vendored `.pth` / Meta 下载段,统一为 HF 获取**

把「获取权重」段中讲 Meta 官方 `.pth`、`strict=True`、vendored builder 的内容删除;保留/合并到 HF 获取说明(`hf download ... --local-dir third_party/dinov3/checkpoints/dinov3-vits16-hf`)。`### HuggingFace 后端(可选)` 标题改为 `### 权重获取`(不再是"可选",是唯一)。

- [ ] **Step 3: 深度模式表去掉 dino_attention 行、默认改 VDA**

在第 7 章「深度模式」表中删除 `| \`dino_attention\`(默认) | DINO 注意力代理 | 无 |` 一行;把 `video_depth_anything` 标注为默认。把节首"默认 `dino_attention` 是 DINO 注意力代理"那句改为说明默认是 `video_depth_anything`(需 VDA 权重)。

- [ ] **Step 4: 第 6 章 CLI 示例与第 1 章功能表里 `dino_attention` 提法清理**

Run 检查并改:`grep -n "dino_attention" README.md` —— 把出现处改写或删除,使其与"已移除 dino_attention"一致。

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: DINO HF-only; drop dino_attention from depth modes"
```

---

### Task 5: 端到端验证

- [ ] **Step 1: 全套单测**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全部通过。

- [ ] **Step 2: 确认 vendored dinov3 彻底不被引用,VGGT/VDA/depth_pro 仍可加载**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run python - <<'PY'
import sys
# DINO(HF)用 stand-in 随机权重验证加载+提取路径
import tempfile, pathlib, torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from feature_extractor.extractors.dino import DINOExtractor
from feature_extractor.validation.synthetic import make_gradient_video
from feature_extractor.validation import sanity
root = pathlib.Path(tempfile.mkdtemp())
hf = root/"third_party/dinov3/checkpoints/dinov3-vits16plus-hf"; hf.mkdir(parents=True)
DINOv3ViTModel(DINOv3ViTConfig(hidden_size=384,num_hidden_layers=1,num_attention_heads=6,
    intermediate_size=768,patch_size=16,image_size=512,num_register_tokens=4)).save_pretrained(str(hf))
ex = DINOExtractor(device="cuda", assets_root=str(root))   # 默认 dinov3_vits16plus(HF)
with tempfile.TemporaryDirectory() as td:
    v=str(pathlib.Path(td)/"g.mp4"); make_gradient_video(v,n_frames=4)
    f=ex.extract_video(v, frame_indices=list(range(4)))
assert f.shape==(4,1025,384) and all(c.passed for c in sanity.check_dino(f))
print("DINO HF default OK", f.shape)
# 确认 vendored dinov3 不在 sys.modules(没人 import 它)
assert not any(m=="dinov3" or m.startswith("dinov3.") for m in sys.modules), "vendored dinov3 still imported!"
print("no vendored dinov3 import OK")
import shutil; shutil.rmtree(root)
PY
```
Expected: `DINO HF default OK (4, 1025, 384)` 与 `no vendored dinov3 import OK`,无报错。

- [ ] **Step 3(有真权重时):跑 feature-extract dino + VDA depth**

Run(若已放好 HF 权重 + VDA 权重):
```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-extract --data_root <数据> --output_root /tmp/o \
  --branches dino,depth --depth_mode video_depth_anything --frames_per_video 8
```
Expected: 无报错,产出 h5。无权重则跳过并说明。

- [ ] **Step 4: 度量 third_party/dinov3 代码已清零**

Run: `find third_party/dinov3 -name '*.py' | wc -l`
Expected: `0`(只剩 checkpoints 权重,非代码)。
