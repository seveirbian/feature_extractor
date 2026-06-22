# DINO 改为 HF-only(移除 vendored / dinov2-hub / dino_attention)设计

- 日期:2026-06-22
- 范围:DINO 后端只保留 HuggingFace transformers 实现;删除 vendored dinov3 与 dinov2(torch.hub)两类后端,
  删除依赖 vendored 注意力的 `dino_attention` 深度模式。VGGT / Video-Depth-Anything / depth_pro 不动。

## 1. 决策

| 项 | 决策 |
|----|------|
| DINO 后端 | 只留 HF;删 vendored `dinov3`(family=dinov3)与 `dinov2`(torch.hub)条目与加载路径 |
| 命名 | 去掉 `_hf` 后缀:`dinov3_vits16` / `dinov3_vits16plus` 现指 HF 实现;默认 `dinov3_vits16plus` |
| dino_attention | **删除**该深度代理模式(依赖 vendored `MemEffAttention`) |
| 默认 depth_mode | 由 `dino_attention` 改为 `video_depth_anything` |
| vendored 源码 | 删 `third_party/dinov3/` 的 python 包 + LICENSE;保留 `checkpoints/`(HF 权重,gitignored) |

非目标:不动 VGGT/VDA/depth_pro;不改 HF token 切片契约(仍 `(T,1025,384)`)。

## 2. 改动明细

### 2.1 `src/feature_extractor/extractors/dino.py`
- `MODEL_CONFIGS`:只保留两条(family `dinov3_hf`),命名为 `dinov3_vits16` / `dinov3_vits16plus`,
  字段 img_size=512/patch16/embed384、`weights` 指向本地 HF 目录
  (`third_party/dinov3/checkpoints/dinov3-vits16-hf` 与 `…plus-hf`)。删除原 vendored `dinov3_vits16(plus)`、
  全部 `dinov2_*` 条目。
- `MODEL_ALIASES`:`facebook/dinov3-vits16-pretrain-lvd1689m`→`dinov3_vits16`、
  `facebook/dinov3-vits16plus-pretrain-lvd1689m`→`dinov3_vits16plus`、`dinov3-vits16(plus)`→同名;
  删除全部 dinov2 别名与 `*_hf` 旧名(已并入新名)。
- 默认 `model_name="dinov3_vits16plus"`(现为 HF)。
- 删除 `_load_dinov3`(vendored)、`_load_alternative`(timm fallback)。
- `_load_model`:只剩 `model = self._load_dinov3_hf(cfg)`(family 都是 dinov3_hf)。
- `_extract_tokens`:只剩 HF 分支(`self.model(pixel_values=...)` → `_slice_hf_tokens` 切 register);
  删除 vendored 的 `forward_features`/dict/last_hidden_state 通用处理。
- 保留 `_slice_hf_tokens`、`_load_dinov3_hf`、`extract_frame`/`extract_video`(预处理与循环不变)。

### 2.2 `src/feature_extractor/extractors/depth.py`
- 删除 `_depth_from_dino_attention` 及 `dino_attention` 分支。
- DepthExtractor 默认 `mode` 改为 `video_depth_anything`;`dino_extractor` 仅 dino_attention 用到 →
  从 depth 路径清理(构造仍可接受该参数但不再使用,或移除;实现期最小改动为准)。

### 2.3 `src/feature_extractor/cli.py`
- `--depth_mode` 默认 `dino_attention` → `video_depth_anything`。
- `--dino_model` 默认仍 `dinov3_vits16plus`(现 HF)。

### 2.4 `third_party/`
- 删除 `third_party/dinov3/<vendored python package>`(19 文件)+ `third_party/dinov3/LICENSE.md`。
- 保留 `third_party/dinov3/checkpoints/`(HF 权重目录,gitignored)。
- `third_party/PROVENANCE.md`:删除 dinov3 行(只剩 VGGT/VDA/ml-depth-pro)。

### 2.5 测试
- `tests/test_dino_config.py`:vendored 名/别名断言失效 → 改为断言:`dinov3_vits16`/`dinov3_vits16plus`
  存在且 `family=="dinov3_hf"`;`facebook/*` 别名指向它们;旧 vendored/dinov2 名**不在** `MODEL_CONFIGS`。
- `tests/test_dino_hf_config.py`、`tests/test_dino_hf_tokens.py`:`_hf` 名已改 → 更新为新名(或保留断言切片纯函数)。
- 其余单测(validation/video_io/storage)不依赖 DINO 加载,不受影响。

### 2.6 文档
- README「DINO backbone」节改为 HF-only(只列两名、HF 权重获取、去掉 vendored `.pth`/Meta 下载段、
  去掉 dino_attention 提法);depth 模式表去掉 `dino_attention` 行、默认标注 `video_depth_anything`。

## 3. 后果(已与用户确认)

- DINO 分支**必须**有 HF 格式权重 + transformers,无 vendored 回退。
- 默认 depth_mode 变 `video_depth_anything`(需 VDA 权重);不再有无权重快速深度代理。
- 合规面:DINO 第三方源码从 19 文件降为 0(改由 PyPI 的 transformers 提供)。

## 4. 验证
- 单测全绿(更新后的 dino 配置测试 + 既有)。
- 端到端(有 HF 权重时):`feature-extract --branches dino --dino_model dinov3_vits16`、
  及 `--depth_mode video_depth_anything` 跑通;无权重时以 stand-in 随机 HF 模型(save_pretrained)验证加载+提取路径。
- 确认删除后 `import dinov3` 不再被任何代码触发;VGGT/VDA/depth_pro 仍各自正常。
