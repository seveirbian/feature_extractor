# 支持 DINOv3 ViT-S/16(dinov3_vits16)backbone 设计

- 日期:2026-06-16
- 范围:让 `DINOExtractor` 支持 `facebook/dinov3-vits16-pretrain-lvd1689m` 权重,作为
  **可选** backbone(默认仍为 `dinov3_vits16plus`)。

## 背景与验证

`_load_dinov3` 已用 `getattr(backbones, model_name)` 泛型查找 builder,vendored
`third_party/dinov3` 仓库已含 `dinov3_vits16` builder(`embed_dim=384`、`patch_size=16`、
depth 12 / 6 heads,比 vits16plus 更轻)。已实测:官方 `.pth`
(`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`)对该 builder `strict=True` 加载
**0 missing / 0 unexpected**。因此**加载器无需改动**。

## 改动

1. `src/feature_extractor/extractors/dino.py`
   - `MODEL_CONFIGS` 新增 `dinov3_vits16`:
     ```python
     "dinov3_vits16": {
         "family": "dinov3",
         "img_size": 512,
         "patch_size": 16,
         "embed_dim": 384,
         "local_repo": "third_party/dinov3",
         "weights": "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
     },
     ```
     (镜像 vits16plus 设置,沿用 `img_size=512` 使输出形状不变 `(T,1025,384)`。)
   - `MODEL_ALIASES` 新增:
     ```python
     "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
     "facebook/dinov3-vits16": "dinov3_vits16",
     "dinov3-vits16": "dinov3_vits16",
     "dinov3_vits16_pretrain_lvd1689m": "dinov3_vits16",
     ```

2. 权重:已放在 `third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
   (该目录软链到 egoWM)。获取方式见 README(Meta 官方下载页 / HF gated)。

3. README:DINO 部分补充可选 `dinov3_vits16`,并给官方权重获取说明。

## 测试

- **纯单测**(不加载权重):断言 `MODEL_ALIASES["facebook/dinov3-vits16-pretrain-lvd1689m"]`
  解析到 `dinov3_vits16`,且 `MODEL_CONFIGS["dinov3_vits16"]` 的 family/embed_dim/patch_size
  正确。
- **端到端**(权重已就位):`feature-validate --branches dino --dino_model dinov3_vits16
  --skip-perf` → dino 各项 PASS(embed_dim=384、含 CLS、确定性)。

## 非目标

- 不改默认模型;不重构加载器;不新增 OBS 同步(可后续手动上传)。
