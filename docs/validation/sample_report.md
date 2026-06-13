# feature_extractor 验证报告

- 日期:2026-06-13
- Git commit:`946738f`
- GPU:Tesla V100S-PCIE-32GB　torch:2.6.0+cu124　CUDA:12.4
- 依赖:av=17.1.0, decord=0.6.0, torch=2.6.0+cu124

## 摘要

- 总体:**PASS**　功能不变量 **24/24** 通过

## 功能不变量

| 分支 | 检查项 | 期望 | 实测 | 结果 |
|------|--------|------|------|------|
| dino | ndim==3 | 3 | 3 | PASS |
| dino | dtype==float32 | float32 | float32 | PASS |
| dino | embed_dim==384 | 384 | 384 | PASS |
| dino | 有限值 | all finite | True | PASS |
| dino | 含CLS(N≥2) | >=2 | 1025 | PASS |
| dino | CLS≠patch均值 | >1e-6 | 0.26 | PASS |
| dino | 确定性(两次allclose) | allclose atol=0.001 | True | PASS |
| pipeline | dino 往返(无损) | 完全相等 | True | PASS |
| depth | shape==(T,H,W,1) | 4D 末维1 | (8, 96, 128, 1) | PASS |
| depth | dtype==float32 | float32 | float32 | PASS |
| depth | 逆深度≥0 | >=0 | True | PASS |
| depth | 有限值 | all finite | True | PASS |
| depth | 非全常数 | std>0 | 0.121 | PASS |
| depth | 确定性(两次allclose) | allclose atol=0.001 | True | PASS |
| pipeline | depth 往返(量化容差) | max\|·\|<3.1e-05 | 7.6e-06 | PASS |
| pose | shape==(T,9) | 2D 末维9 | (8, 9) | PASS |
| pose | dtype==float32 | float32 | float32 | PASS |
| pose | 有限值 | all finite | True | PASS |
| pose | pose[0]≈单位变换 | max\|·\|<1e-3 | 6.21e-11 | PASS |
| pose | 每帧6D→有效旋转 | 正交且det≈1 | True | PASS |
| pipeline | pose 往返(无损) | 完全相等 | True | PASS |
| pipeline | dino 帧索引对齐请求 | [0, 1, 2, 3]… | [0, 1, 2, 3]… | PASS |
| pipeline | depth 帧索引对齐请求 | [0, 1, 2, 3]… | [0, 1, 2, 3]… | PASS |
| pipeline | pose 帧索引对齐请求 | [0, 1, 2, 3]… | [0, 1, 2, 3]… | PASS |

## 性能

| 视频 | 分支 | 帧数 | 耗时(s) | FPS | 峰值显存(MB) | 备注 |
|------|------|------|---------|-----|--------------|------|
| file-000 | model_load | 0 | 33.5 | — | — | 一次性加载 |
| file-000 | decode | 64 | 33.9 | 1.9 | — | 纯解码 |
| file-000 | dino | 64 | 35.5 | 1.8 | 10610.6 |  |
| file-000 | depth | 64 | 57.7 | 1.1 | 10674.1 |  |
| file-000 | pose | 64 | 68.3 | 0.9 | 29365.9 |  |
| file-000@16 | dino | 16 | 32.9 | 0.5 | 8743.9 | 扫描 frames=16 |
| file-000@32 | dino | 32 | 34.4 | 0.9 | 8743.9 | 扫描 frames=32 |
| file-000@64 | dino | 64 | 35.6 | 1.8 | 8743.9 | 扫描 frames=64 |
| file-000@128 | dino | 128 | 36.9 | 3.5 | 8743.9 | 扫描 frames=128 |

## 复现

```bash
feature-validate --data_root data/libero_10/videos --branches dino,depth,pose --depth_mode video_depth_anything --perf-frames 64 --frames-sweep 16,32,64,128 --report /tmp/val_full.md
```
