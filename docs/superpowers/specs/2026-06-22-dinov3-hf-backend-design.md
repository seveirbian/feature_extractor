# DINOv3 HuggingFace backend(可选,与 vendored 并存)设计

- 日期:2026-06-22
- 范围:为 `DINOExtractor` 增加基于 HuggingFace `transformers` 的 DINOv3 后端,作为**可选 backend**
  与现有 vendored 实现**并存**;不动 VGGT / Video-Depth-Anything / depth_pro / 现有 vendored DINO。
- 依据:`docs/huggingface-migration-feasibility.md`(仅 DINOv3 在稳定版 transformers 可干净替换)。

## 1. 目标与约束

| 项 | 决策 |
|----|------|
| 形态 | 可选 backend,新增 `dinov3_vits16_hf` / `dinov3_vits16plus_hf`,经 `--dino_model` 选择 |
| 现有行为 | vendored 条目与 HF 风格别名(`facebook/dinov3-*`)**不变**(仍指向 vendored) |
| 输出契约 | 与 vendored 一致:`(T, 1025, 384)` = `[CLS] + 1024 patches`,**切掉 register tokens** |
| 依赖 | `transformers>=4.56`(DINOv3 落地版本)加入 base dependencies |
| 权重 | **本地 HF 格式目录**(config.json + safetensors),离线 `from_pretrained(本地路径)`;用户自备 |
| 不改 | 预处理(resize 512 + ImageNet 归一化)、L2-norm、`extract_video` 循环、其余分支 |

非目标:不迁移 VGGT/VDA/depth_pro;不联网下载 gated 权重;不追求与 vendored 逐位相同(验证为不变量式)。

## 2. 背景(现状关键点)

- vendored 路径:`extract_frame` 预处理 → `_extract_tokens` 调 `forward_features` 取 `x_norm_clstoken` +
  `x_norm_patchtokens`(register 已排除)→ `[CLS, patches]` → L2-norm → `(N+1, D)`。
- HF `DINOv3ViTModel(pixel_values=...)` 返回 `last_hidden_state` 形如 `[CLS, R×register, patches]`
  (R=`config.num_register_tokens`,vits16/plus 为 4)。512×512 输入 → 32×32=1024 patches → 1+4+1024=1029。
- 故 HF 分支取 `cls = h[:,0:1]`、`patches = h[:, 1+R:]`,拼成 `[CLS, patches]` = 1025,匹配契约。

## 3. 改动

### 3.1 `src/feature_extractor/extractors/dino.py`

- `MODEL_CONFIGS` 新增:
  ```python
  "dinov3_vits16_hf": {
      "family": "dinov3_hf",
      "img_size": 512, "patch_size": 16, "embed_dim": 384,
      "weights": "third_party/dinov3/checkpoints/dinov3-vits16-hf",
  },
  "dinov3_vits16plus_hf": {
      "family": "dinov3_hf",
      "img_size": 512, "patch_size": 16, "embed_dim": 384,
      "weights": "third_party/dinov3/checkpoints/dinov3-vits16plus-hf",
  },
  ```
  (`weights` 指向本地 HF 格式目录;用户把 HF checkpoint 放这里。)
- `_load_model`:`cfg["family"] == "dinov3_hf"` 时走新 `_load_dinov3_hf(cfg)`。
- `_load_dinov3_hf(cfg)`:
  ```python
  from transformers import AutoModel  # 懒导入
  weights = resolve_assets_root(self.assets_root) / cfg["weights"]
  if not weights.exists():
      raise FileNotFoundError(f"DINOv3 HF weights dir not found: {weights}")
  model = AutoModel.from_pretrained(str(weights))
  return model
  ```
  缺 `transformers` 时捕获 ImportError,报"pip/uv 安装 transformers>=4.56"。
- `_extract_tokens`:`self.family == "dinov3_hf"` 时走新分支:
  ```python
  out = self.model(pixel_values=image_tensor.unsqueeze(0))
  h = out.last_hidden_state                      # (1, 1+R+P, D)
  R = int(getattr(self.model.config, "num_register_tokens", 0))
  return torch.cat([h[:, 0:1, :], h[:, 1 + R:, :]], dim=1)  # (1, 1+P, D)
  ```
  其余(`extract_frame` 预处理、L2-norm、`extract_video`)复用不改。

### 3.2 `pyproject.toml`

- `dependencies` 加 `transformers>=4.56`。

### 3.3 README

- 「DINO backbone」表加 `dinov3_vits16_hf` / `dinov3_vits16plus_hf`,说明:需 `transformers`、
  需本地 HF 格式权重目录、输出与 vendored 同契约。

## 4. 测试

- **纯配置单测**(无需权重/transformers 运行模型):断言两个 `*_hf` 在 `MODEL_CONFIGS`,
  `family=="dinov3_hf"`、`embed_dim==384`、`weights` 指向 HF 目录。
- **HF 路径形状单测**(需 transformers,**不需真权重**):用
  `DINOv3ViTModel(DINOv3ViTConfig(num_register_tokens=4, ...))` 随机初始化,跑一张 512×512 张量经 HF 提取分支,
  断言输出 `(1, 1025, 384)`、且等于 `cat([h[:,0:1], h[:,1+R:]])`(register 被排除)。
- **端到端(条件性,有真权重时)**:`feature-validate --branches dino --dino_model dinov3_vits16_hf --skip-perf`
  → DINO 各不变量 PASS、形状 `(T,1025,384)`。无本地 HF 权重目录时此步跳过并说明。

## 5. 风险

- **numpy<2 冲突**:`transformers>=4.56` 须与 vendored VGGT 要求的 `numpy<2`、torch-cu124 共存。
  **第一步**用 `uv` 验证解析成功;若冲突,缩小 transformers 版本范围到既含 DINOv3 又兼容 numpy<2 的版本。
- **输出非逐位一致**:HF 实现与 vendored 数值略有差异(同权重同架构,差异极小);验证为不变量式,可接受。
- **本地 HF 权重格式**:须为 HF 格式(config.json+safetensors),非原始 `.pth`;用户自备,缺失时明确报错。

## 6. 交付

- `dino.py`:两个 `*_hf` 配置 + `_load_dinov3_hf` + `_extract_tokens` HF 分支。
- `pyproject.toml`:`transformers>=4.56`。
- 两个单测(配置 + HF 形状)。
- README 更新。
- 验证记录(单测全绿;有权重时端到端 PASS)。
