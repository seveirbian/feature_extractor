# DINO 特征可视化(export_dino_video.py)

把一段视频的 DINO 特征逐帧渲染成并排对比的 MP4,直观看 DINO 在"看"什么。

每帧并排若干面板:

- **原始帧**(带帧号/时间戳)
- **DINO PCA RGB**:对 patch token 做 PCA 取前 3 主成分映射成 RGB(语义相近的区域颜色相近)
- **CLS Similarity**(仅 `rgb_pca_cls` 模式):每个 patch 与 CLS token 的余弦相似度热力图

## 用法

```bash
CUDA_VISIBLE_DEVICES=7 uv run python src/feature_extractor/virtualization/export_dino_video.py \
    --video_path  path/to/clip.mp4 \
    --output_mp4  /tmp/dino_vis.mp4 \
    --dino_model  dinov3_vits16 \
    --render_scale 0.5
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--video_path` | 必填 | 输入视频(支持 AV1/H.264 等,经 `video_io` 解码) |
| `--output_mp4` | 必填 | 输出可视化视频路径 |
| `--output_json` | 同名 `.json` | 输出元数据报告(帧数/fps/分辨率/grid/特征形状等) |
| `--dino_model` | `dinov3_vits16plus` | DINO backbone;可填 `dinov3_vits16` 或 HF 别名(见仓库根 README「DINO backbone」) |
| `--device` | `cuda` | 计算设备 |
| `--render_scale` | `1.0` | 输出相对原分辨率的缩放;长视频/大图建议调小加速 |
| `--panel_mode` | `rgb_pca_cls` | `rgb_pca`(原帧+PCA)或 `rgb_pca_cls`(再加 CLS 相似度) |

## 输出

- **MP4**:每帧为各面板横向拼接,帧率取自源视频。
- **JSON**:`video_path / output_mp4 / frames / fps / original_resolution / render_resolution / dino_model / panel_mode / grid_side / feature_shape`。

## 注意

- **整段抽帧**:脚本对视频的**每一帧**都跑 DINO,长视频(上万帧)会很慢、显存/耗时高;
  建议先用短片或截取片段。
- **AV1**:经 `feature_extractor.video_io` 解码,AV1 走 PyAV(libdav1d)回退,无需额外处理。
- **权重**:`--dino_model` 对应的权重需已放在 `third_party/dinov3/checkpoints/`
  (获取方式见仓库根 README)。
- patch token 数须为完全平方数(用于摆成方形 grid),DINOv3 ViT-S/16 默认满足
  (`1024 patch → 32×32`)。
