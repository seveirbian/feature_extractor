# 可行性调研:将 third_party 模型换成 HuggingFace transformers 实现

- 日期:2026-06-22
- 结论先行:**不建议整体迁移**。只有 DINOv3(及勉强 Depth Pro)在稳定版 transformers 里、可干净替换;
  VGGT、Video-Depth-Anything 在稳定版**没有**,任何迁移路径都会破坏刚完成的"去子模块 / 自包含 / 减少扫描面"合规目标。

## 背景

当前四个模型以**裁剪后 vendored 源码**形式放在 `third_party/`(见
`docs/superpowers/specs/2026-06-21-vendor-trim-thirdparty-design.md`)。本调研评估"改用 HF transformers
实现是否更简单"。

## 概念区分(决定能否使用)

| 层次 | 含义 | 对合规/可复现 |
|------|------|----------------|
| **稳定版(PyPI release)** | 已发布、带版本号(如 `transformers==5.12.0`),`uv add transformers` 装到的 | ✅ 版本固定、可复现、内源/合规通常只认这种 |
| **main(开发分支)** | GitHub 上未发布的最新代码;文档 `/docs/transformers/main/...`。"只在 main 有"= 已合入但**未进任何发行版** | ⚠️ 只能 `pip install git+...@commit`,移动靶、难复现,合规一般不接受;或等发版 |
| **trust_remote_code** | 模型实现代码放在 HF 模型仓库,加载时**运行时下载并执行远程 Python** | ❌ 等于运行未审计远程代码,与"自包含/去外部代码"相悖 |

## 各模型支持现状(2026-06 实测自 HF 官方文档)

| 模型 | 稳定版原生? | 类 / checkpoint | 迁移代价 |
|------|--------------|------------------|----------|
| **DINOv3** | ✅ 稳定版(v5.12.0) | `AutoModel` / `DINOv3ViTModel` + `DINOv3ViTImageProcessor`;`facebook/dinov3-vits16-pretrain-lvd1689m` | 输出布局不同:HF 为 `[CLS, 4×register, patches]`(如 1+4+1024=1029),当前为 `[CLS, patches]`=1025 → 需切掉 register tokens 对齐;需重验;权重来自 HF(gated) |
| **Depth Pro** | ✅ 稳定版 | `DepthProForDepthEstimation`;`apple/DepthPro-hf` | 当前主要用于 VDA 度量校正,接口不同;收益小、要改适配 |
| **VGGT** | ❌ 仅 main/dev | —(稳定版 `model_doc/vggt` 404) | 无干净路径:钉 main(不稳/难复现)或 trust_remote_code(远程代码),均踩合规雷 |
| **Video-Depth-Anything** | ❌ 仅 main/dev | —(稳定版 404;稳定版只有图像版 Depth Anything V2) | 同上 |

## 共性代价(无论换哪个)

1. **新增重依赖**:`transformers` 现为 v5.x,体量大、v5 为大版本变更。
2. **输出语义改变**:DINO 的 register token、VGGT 的位姿编码接口等都与现实现不同 → 整条管道需用
   `feature-validate` 重新验证并对齐下游(DINO 的 CLS+patch 被 depth 的 `dino_attention` 代理与可视化 PCA 消费)。
3. **权重来源变化**:从 OBS/官方下载改为 HF Hub(DINOv3/VGGT 为 gated),与现流程不一致。

## 建议

- **整体不换。** VGGT/VDA 稳定版不可得,迁移会破坏合规且不可复现。
- 若将来要简化,**唯一较干净的候选是 DINOv3**:可作为**可选 backend** 与现有 vendored 并存(不动 VGGT/VDA),
  但仍需承担 register-token 对齐、重验、gated 权重三项成本,需权衡"省 ~19 个文件 / 19M"是否值得。
- **Depth Pro** 原生可用但收益小、耦合度量校正,优先级低。
- 触发再评估的条件:transformers **稳定版**纳入 VGGT 与 Video-Depth-Anything 后,可重跑本调研。

## 参考

- DINOv3:https://huggingface.co/docs/transformers/en/model_doc/dinov3
- Depth Pro:https://huggingface.co/docs/transformers/en/model_doc/depth_pro
- VGGT / Video-Depth-Anything:稳定版 `model_doc` 页 404,仅存在于 `main`(dev)
