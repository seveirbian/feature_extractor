# Depth Pro 改用 HuggingFace transformers(HF-only)设计

- 日期:2026-06-22
- 范围:把 Depth Pro 后端从 vendored `third_party/ml-depth-pro` 改为 HuggingFace `transformers` 的
  `DepthProForDepthEstimation`,并删除 vendored ml-depth-pro 源码。VGGT / Video-Depth-Anything 不动。
- 依据:`docs/huggingface-migration-feasibility.md`(Depth Pro 在稳定版 transformers 原生支持);
  做法对齐已完成的 DINO HF-only。

## 1. 决策

| 项 | 决策 |
|----|------|
| 后端 | Depth Pro 只走 transformers `DepthProForDepthEstimation`;删除 vendored ml-depth-pro |
| 权重 | 本地 HF 格式目录 `third_party/ml-depth-pro/checkpoints/depth-pro-hf/`(离线 from_pretrained,用户自备;来源 HF 仓库 `apple/DepthPro-hf`) |
| 依赖 | `transformers` 已在依赖中(DINO 引入),无新增 |
| 输出契约 | `_extract_depth_pro(image)` 仍返回**米制深度 `(H,W)` float32**(原分辨率),两个调用方不变 |
| FOV | 加载时 `use_fov_model=False`(本模块只用深度,不用焦距/FOV,省算力) |

非目标:不动 VGGT/VDA;不改 VDA↔DepthPro 的标定逻辑(`_metricize_vda_depth_sequence`/`_fit_depth_affine`);不追求与 vendored 逐位一致(标定为仿射拟合,对尺度差异稳健)。

## 2. 现状

- `_load_depth_pro(device, assets_root)` 从 `third_party/ml-depth-pro/src` 加载 vendored `depth_pro`,
  返回 `(model, transform)`,赋给 `self.depth_pro_model` / `self.depth_pro_transform`。
- `_extract_depth_pro(image)`:PIL→transform→`model.infer(x)["depth"]`→resize 到原分辨率→`(H,W)` float32。
- 消费方:① `_metricize_vda_depth_sequence` 关键帧上 `_extract_depth_pro` → `_fit_depth_affine`(默认 VDA 路径依赖);
  ② 独立 `--depth_mode depth_pro`。
- `__init__` 在 `mode in ("da3","video_depth_anything","depth_pro")` 时加载 depth_pro;VDA 模式要求其非空。

## 3. 改动

### 3.1 `src/feature_extractor/extractors/depth.py`
- `_load_depth_pro(device, assets_root)` 改为(懒导入 transformers):
  ```python
  from transformers import AutoImageProcessor, DepthProForDepthEstimation
  hf_dir = resolve_assets_root(assets_root) / "third_party" / "ml-depth-pro" / "checkpoints" / "depth-pro-hf"
  if not hf_dir.exists(): raise FileNotFoundError(...HF 格式权重目录...)
  processor = AutoImageProcessor.from_pretrained(str(hf_dir))
  model = DepthProForDepthEstimation.from_pretrained(str(hf_dir), use_fov_model=False).to(device).eval()
  return model, processor
  ```
  保留返回二元组形状;`__init__` 接收处改名 `self.depth_pro_model, self.depth_pro_processor = _load_depth_pro(...)`
  (把原 `self.depth_pro_transform` 统一改名为 `self.depth_pro_processor`,涉及 `__init__` 与 `_extract_depth_pro`)。
  缺 transformers → RuntimeError 提示;缺目录 → FileNotFoundError。
- `_extract_depth_pro(image)` 改为 HF 推理:
  ```python
  from PIL import Image
  if self.depth_pro_processor is None: return np.ones_like(image[...,0], np.float32) * 5.0
  # uint8 + bgr→rgb 处理同现状
  pil = Image.fromarray(image_rgb)
  inputs = self.depth_pro_processor(images=pil, return_tensors="pt").to(self.device)
  out = self.depth_pro_model(**inputs)
  post = self.depth_pro_processor.post_process_depth_estimation(out, target_sizes=[(image.shape[0], image.shape[1])])
  depth = post[0]["predicted_depth"]
  return depth.detach().cpu().numpy().astype(np.float32)   # (H,W),已是原分辨率
  ```
  (post_process 已按 target_sizes 还原到原分辨率,无需手动 cv2.resize。)
- 文件顶部模块 docstring/类 docstring 把 "vendored Depth Pro" 措辞更新为 "Depth Pro (HuggingFace)"。

### 3.2 third_party
- 删除 `third_party/ml-depth-pro/src`(vendored python 包,9 文件)+ `third_party/ml-depth-pro/LICENSE`。
- 保留 `third_party/ml-depth-pro/checkpoints/`(HF 权重,gitignored)。
- `third_party/PROVENANCE.md` 删除 ml-depth-pro 行(只剩 VGGT、Video-Depth-Anything)。

### 3.3 依赖
- `transformers` 已在 deps。`pillow_heif` 当初注释为 ml-depth-pro 所需;HF 路径用标准 PIL,
  **本设计不动 deps**(保守;`pillow_heif` 是否还需留待后续单独评估,避免误删影响其他路径)。

### 3.4 README
- §3 模型资源:vendored 列表去掉 ml-depth-pro(只剩 VGGT、Video-Depth-Anything);OBS 下载去掉
  ml-depth-pro 行;把 Depth Pro 归到 "HF 获取"(与 DINO 并列,需本地 HF 目录)。
- §7 深度模式:`depth_pro` 模式与默认 `video_depth_anything`(度量校正用 Depth Pro)的权重说明改为
  "Depth Pro 需 HF 格式权重目录";给出 `hf download apple/DepthPro-hf --local-dir .../depth-pro-hf`。

## 4. 测试 / 验证
- **纯单测**:`_extract_depth_pro` 难以零成本单测(模型重)。改为:
  - 用一个**小随机** `DepthProForDepthEstimation`(tiny config)`save_pretrained` 到临时
    `third_party/ml-depth-pro/checkpoints/depth-pro-hf`,经 `assets_root` 覆盖构造
    `DepthExtractor(mode="depth_pro")`,对合成帧跑 `_extract_depth_pro`,断言返回 `(H,W)` float32、有限。
    若 tiny 构造不可行则降级为"有真权重时跑"。
- **端到端(有真权重)**:`feature-extract --branches depth --depth_mode depth_pro`,以及默认
  `video_depth_anything`(走 DepthPro 关键帧校正)能跑通、出 h5。
- 确认删除后无任何代码 `import depth_pro` / 引用 `ml-depth-pro/src`;VGGT/VDA 仍正常。
- 全套既有单测green。

## 5. 风险
- HF DepthPro 预处理/输出与 vendored 略有数值差异;`video_depth_anything` 默认路径靠仿射标定吸收尺度差,
  影响小;独立 `depth_pro` 模式绝对值可能微移(可接受,验证为不变量式)。
- `apple/DepthPro-hf` 默认带 FOV 编码器;本设计 `use_fov_model=False` 只取深度。
- 权重缺失 → 明确报错(同 DINO)。
