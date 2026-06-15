# feature_extractor 自验证(feature-validate)

对 feature_extractor 模块做**功能**与**性能**两方面的自验证,跑完自动生成一份
Markdown 报告,供交付/复核使用。

- **功能**:在**合成视频**上检查各分支输出的合理性/不变量(无需真实数据、确定可复现)。
- **性能**:在**真实数据**上测吞吐、峰值显存、规模扩展性。
- **退出码**:任一功能不变量失败 → 退出码非零(可直接做 CI 门禁)。

样例报告见 [`sample_report.md`](./sample_report.md)。

## 前置条件

1. 子模块与权重已就位(见仓库根 [README](../../README.md) 的「模型资源」一节)。
   - 仅跑 `dino` 分支只需 dinov3 权重;`depth(video_depth_anything)` 还需 VDA + ml-depth-pro;`pose` 需 VGGT。
2. 依赖已安装:`uv sync`。

## 快速开始

### 1) 只验功能(最快,不需要真实数据)

```bash
# 仅 dino 分支(最轻量,几十秒)
CUDA_VISIBLE_DEVICES=7 uv run feature-validate --branches dino --skip-perf --report report_dino.md

# 全部分支的功能检查(需要 VDA/ml-depth-pro/VGGT 权重)
CUDA_VISIBLE_DEVICES=7 uv run feature-validate \
    --branches dino,depth,pose --depth_mode video_depth_anything \
    --skip-perf --report report_func.md
```

### 2) 功能 + 性能(交付配置,跑真实数据)

```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-validate \
    --data_root data/libero_10/videos \
    --branches dino,depth,pose \
    --depth_mode video_depth_anything \
    --perf-frames 64 \
    --frames-sweep 16,32,64,128 \
    --report report_full.md
```

跑完终端会打印 `报告已写入 <path>　功能 N/N 通过`,并据此设置退出码:

```bash
echo $?   # 0 = 全部功能不变量通过;非 0 = 有失败项
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data_root` | 无 | 性能基准用的真实数据目录(递归找视频)。配 `--skip-perf` 时可省 |
| `--report` | `validation_report.md` | 报告输出路径 |
| `--branches` | `dino,depth,pose` | 验证哪些分支 |
| `--depth_mode` | `video_depth_anything` | depth 后端(同主 CLI) |
| `--perf-frames` | `64` | 单视频吞吐基准的帧数 |
| `--frames-sweep` | `16,32,64,128` | 扩展性扫描的帧数档位(超视频长度自动截断) |
| `--device` | `cuda` | 计算设备;CPU-only 环境用 `--device cpu` |
| `--assets_root` | 无 | 覆盖模型资源根目录(同主 CLI) |
| `--skip-perf` | 关 | 跳过性能基准(只验功能) |
| `--skip-sanity` | 关 | 跳过功能验证(只测性能) |

## 报告怎么读

报告分四部分:

1. **元数据/环境**:日期、git commit、主机、GPU、torch/CUDA、关键依赖版本——保证结果可追溯。
2. **摘要**:总体 PASS/FAIL 与 `通过数/总数`。
3. **功能不变量**:逐项 `分支 | 检查项 | 期望 | 实测 | 结果`。挑的都是与模型质量无关、构造上必然成立的硬不变量,例如:
   - 形状/`dtype=float32`/有限值;三分支帧索引对齐;存储往返(dino/pose 无损,depth 量化容差)。
   - DINO:`embed_dim=384`、含 CLS、两次运行 `allclose`。
   - Depth:逆深度 ≥0、非全常数、两次运行 `allclose`。
   - Pose:`pose[0]≈单位变换`(相对第 0 帧)、每帧 6D 旋转重建后正交且 `det≈1`。
4. **性能**:模型加载、纯解码、各分支吞吐+峰值显存,以及 frames 扫描。

> **解读性能时注意**:各分支「耗时/FPS」**包含解码开销**。对长视频做稀疏采样时,
> 解码可能成为主导成本(报告里单列了 `decode` 行,可据此把推理耗时与解码耗时拆开看)。

## 失败时

功能项失败不会中断报告——失败/异常会如实记成 `FAIL`/`EXCEPTION`/`SKIPPED` 行,报告照常生成,
便于定位是哪个分支、哪条不变量出问题。权重缺失则对应项标 `SKIPPED`。

## 在 CI 里用

```bash
uv run feature-validate --branches dino --skip-perf --report report.md || {
    echo "功能验证未通过"; exit 1;
}
```
