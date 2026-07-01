# feature_extractor

从 egoWM 中独立出来的 **DINO / Depth / Pose 特征提取**工具。给定视频,逐帧抽取三类
特征并以 HDF5 存储;提取行为与 egoWM 一致,模型资源根目录可配置。

> 本文档面向**模块交接**:涵盖功能、输入/输出格式与要求、命令行/库用法、自验证,以及
> 已知限制。

## 1. 功能

对每个视频按统一的帧采样,跑下列分支(可按需选择),并把结果写入一个 HDF5 文件:

| 分支 | 内容 | 模型 |
|------|------|------|
| `dino` | 逐帧 DINOv3 patch token 特征(视觉表征) | DINOv3(默认 `dinov3_vits16plus`) |
| `depth` | 逐帧**归一化逆深度**图 | 见下方「深度模式」(默认 `video_depth_anything`) |
| `pose` | 相对第 0 帧的**相机位姿轨迹**(平移 + 6D 旋转) | VGGT |

三个分支共享**同一组帧索引**,保证跨分支严格对齐。

模块组成:

```
src/feature_extractor/
  cli.py            # feature-extract 命令:遍历视频 → 调用各分支 → 写 store
  extractors/       # dino.py / depth.py / pose.py 三个提取器
  storage.py        # FeatureStore:HDF5 读写(含增量分块写 + 完成标记)
  chunking.py       # 流式分块:plan_blocks / iter_frame_blocks(见第 6.1 节)
  video_io.py       # decord 兼容的视频读取(AV1 走 PyAV 回退)
  assets.py         # 模型资源根目录解析
  validation/       # feature-validate 自验证工具(见第 8 节)
```

## 2. 安装

```bash
uv sync
```

`third_party/` 下的第三方模型源码已**内置在仓库**(精简 vendored,无 git 子模块)。
模型权重不在 git 里,需另行获取(见第 3 节)。

## 3. 模型资源(third_party/)

- **DINO** 与 **Depth Pro**:走 HuggingFace `transformers`(PyPI 依赖,已随 `uv sync` 安装),
  **不在** `third_party/`;权重为 HF 格式(DINO 见第 6 节,Depth Pro 见第 7 节)。
- **VGGT / Video-Depth-Anything**:精简后的 **vendored 源码内置**在
  `third_party/<repo>/`(非子模块;裁剪溯源见 `third_party/PROVENANCE.md`)。

### 权重文件(不在 git 里)

vendored 后端权重位于 `third_party/<repo>/checkpoints/`,托管在华为云 OBS,在仓库根目录用
[`obsutil`](https://support.huaweicloud.com/utiltg-obs/obs_11_0003.html) 下载(先 `obsutil config` 配 AK/SK):

```bash
obsutil cp obs://cloudrobo-model/wangchao/egoWM/third_party/Video-Depth-Anything/checkpoints ./third_party/Video-Depth-Anything/checkpoints -r -f
obsutil cp obs://cloudrobo-model/wangchao/egoWM/third_party/VGGT/checkpoints                  ./third_party/VGGT/checkpoints                  -r -f
```

**DINO 与 Depth Pro(均 HF)权重单独获取**(`hf download ...`),见第 6 / 7 节。

只用部分后端时只下对应的几项(`dino` 需 DINOv3 HF 权重,见第 6 节;默认 depth `video_depth_anything`
需 `Video-Depth-Anything`(OBS)+ Depth Pro(HF,见第 7 节);`pose` 需 `VGGT`(OBS))。

### 指向其他位置

用其他位置的模型资源,可覆盖资源根目录(优先级从高到低):
`--assets_root /path`(CLI)/ `assets_root=`(构造函数) > 环境变量
`FEATURE_EXTRACTOR_ASSETS=/path` > 默认包根目录(即内置 `third_party/`)。根目录指
**包含** `third_party/` 的那个目录。

## 4. 输入:格式与要求

- **来源**:`--data_root` 目录被**递归遍历**查找视频文件
  (如 `data/openego/HO-Cap/.../video.mp4`)。
- **支持的扩展名**:`.mp4 .avi .mkv .webm .mov`(大小写 `.MP4/.AVI` 亦可)。
- **编解码**:凡 decord 或 PyAV 能解的都行。**AV1** 由 PyAV(libdav1d)回退解码
  (decord 自带 FFmpeg 无 AV1 软解);H.264 等走 decord。
- **分辨率/帧率**:任意;各模型内部自行 resize。
- **帧采样**:`--frames_per_video N` 在整段视频上**均匀步采**最多 N 帧;`<=0` 或大于
  总帧数则取全部。三分支用同一组索引。
  > 采样帧数超过 `--stream_threshold`(默认 2000,如全帧长视频)时自动切换到**流式分块
  > 提取**,内存按块封顶、逐块写盘,避免长视频 OOM(见第 6.1 节)。
- **video_id(输出文件名)**:默认 `<父目录名>_<文件名stem>`;加 `--id_from_stem` 则只用
  `<stem>`。
  > ⚠️ **注意命名冲突**:若不同子目录下存在同名视频(如 LeRobot 的
  > `observation.images.image/chunk-000/file-000.mp4` 与
  > `observation.images.wrist_image/chunk-000/file-000.mp4`),两种命名都会得到相同
  > video_id,**后处理的会覆盖先处理的**。这类多相机/多 chunk 布局需自行保证 id 唯一
  > (例如改用包含相机目录的命名,或分目录分别跑)。

## 5. 输出:格式与要求

每个视频生成两份产物,落在 `--output_root` 下:

```
<output_root>/
  <video_id>.h5                # 特征
  annotations/<video_id>.json  # 元数据 stub
```

### 5.1 HDF5 schema(`<video_id>.h5`)

每个分支一个 group;每个 group 都带一个 `frame_indices`(int64,长度 T,**指向原视频的
帧号**,三分支一致)。`T` = 实际处理帧数。

| group/dataset | dtype | 形状 | 含义 |
|---|---|---|---|
| `dino/features` | float32 | `(T, N+1, D)` | patch token 特征;含 1 个 CLS + N 个 patch。`dinov3_vits16plus` 下 `D=384`(默认配置 `N+1=1025`)。退化为全局描述子时为 `(T, D)` |
| `dino/frame_indices` | int64 | `(T,)` | 帧号 |
| `depth/inv_depth` | **uint16** | `(T, H, W, 1)` | **归一化逆深度**,真实值 = `stored / 65535.0` ∈ [0,1]。`H,W` 取决于深度后端/输入 |
| `depth/frame_indices` | int64 | `(T,)` | 帧号 |
| `pose/se3_trajectory` | float32 | `(T, 9)` | 相对第 0 帧的位姿:`[tx,ty,tz, r6d_0..r6d_5]`(平移 + 6D 旋转)。也支持 `(T,6)` 的 se(3) log 格式 |
| `pose/frame_indices` | int64 | `(T,)` | 帧号 |

关键属性(attrs):`dino` 有 `representation`(`patch_tokens`/`global_descriptor`)、`shape`;
`depth` 有 `scale=65535.0`、`representation=normalized_inverse_depth`;`pose` 有 `pose_dim`、
`representation`(`translation_rot6d`/`se3_log`)。

要点:
- **DINO / Pose 为 float32 无损存储**;**Depth 为 uint16 有损**(量化精度 ≈ 1/65535)。
- 平移量是 VGGT 归一化场景尺度下的相对平移,**非米制**。
- 位姿是**相对第 0 帧**:`pose[0] ≈ [0,0,0,1,0,0,0,1,0]`(单位变换)。

### 5.2 标注 JSON(`annotations/<video_id>.json`)

最小元数据 stub(无标签),字段含 `video_id`、`video_path`、`fps`、`num_frames`、
`has_depth`、`has_pose`、`source` 等,供下游数据集装配占位。

### 5.3 读取

```python
from feature_extractor import FeatureStore

store = FeatureStore("data/features")
dino = store.read_dino("file-000")    # (T, N+1, 384) float32
depth = store.read_depth("file-000")  # (T, H, W, 1) float32,已 /65535 还原到 [0,1]
pose = store.read_pose("file-000")    # (T, 9) float32
allf = store.read_all("file-000")     # {"dino":..., "depth":..., "pose":...}
idx  = store.read_frame_indices("file-000", "dino")  # 帧号
```

## 6. 命令行(feature-extract)

```bash
# 通过 CUDA_VISIBLE_DEVICES 选择 GPU
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/openego/videos \
    --output_root data/features \
    --device cuda \
    --branches dino,depth,pose \
    --depth_mode video_depth_anything \
    --frames_per_video 64 \
    --id_from_stem
```

选择 DINO 权重(默认 `dinov3_vits16plus`,加 `--dino_model` 切换;详见下方「DINO backbone」):

```bash
# 用更轻的 dinov3_vits16(权重须已放在 third_party/dinov3/checkpoints/)
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/openego/videos \
    --output_root data/features \
    --branches dino,depth,pose \
    --depth_mode video_depth_anything \
    --frames_per_video 64 \
    --id_from_stem \
    --dino_model dinov3_vits16
# 也可用 HF 风格别名:--dino_model facebook/dinov3-vits16-pretrain-lvd1689m
```

常用参数:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data_root` | 必填 | 递归查找视频的根目录 |
| `--output_root` | 必填 | 输出目录 |
| `--branches` | `dino,depth,pose` | 要跑的分支 |
| `--depth_mode` | `video_depth_anything` | 深度后端(见第 7 节) |
| `--frames_per_video` | `120` | 每视频最多采样帧数;`<=0` 取全部 |
| `--device` | `cuda` | 计算设备 |
| `--id_from_stem` | 关 | video_id 只用文件名 stem(见第 4 节命名冲突) |
| `--num_samples` | 全部 | 只处理前 N 个视频(调试用) |
| `--resume` | 关 | 跳过**所有请求分支都已完成**的视频(见第 6.1 节) |
| `--dino_model` | `dinov3_vits16plus` | DINO 模型(见下方可选 backbone) |
| `--vda_input_size` | `224` | Video-Depth-Anything 输入边长 |
| `--assets_root` | 无 | 覆盖模型资源根目录 |
| `--annotation_dir` | `<output_root>/annotations` | 标注输出目录 |

全帧流式提取相关参数(见第 6.1 节):

| 参数 | 默认 | 说明 |
|------|------|------|
| `--stream_threshold` | `2000` | 采样帧数超过此值时切换到流式分块提取 |
| `--block_size` | `1024` | DINO/Depth 流式分块的块长(帧) |
| `--depth_overlap` | `96` | Depth 分段间的重叠帧数(预留,Phase 2 生效) |
| `--pose_window` | `600` | Pose(VGGT)窗口长度,受显存约束(预留,Phase 3 生效) |
| `--pose_overlap` | `120` | Pose 窗口间的重叠帧数(预留,Phase 3 生效) |

> **`--frames_per_video` 与 `--stream_threshold` 的关系**:两者作用在不同阶段、互相正交。
> `--frames_per_video` 决定**采样多少帧**(选哪些帧、共几帧);`--stream_threshold` 再拿这个
> **采样后的帧数**去比较,决定**用什么内存策略处理**(`采样帧数 > 阈值` → 流式分块,否则一次性
> 内存路径)。`--stream_threshold` **只影响处理方式,不改变提取哪些帧**,两种路径输出完全一致。
> 举例(阈值默认 2000):
> - `--frames_per_video 120`(默认):采样 120 帧 < 2000 → 恒走内存路径,与视频长短无关。
> - `--frames_per_video 0`(全帧)+ 长视频(如 18000 帧):采样 18000 > 2000 → 走流式。
> - `--frames_per_video 0`(全帧)+ 短视频(如 300 帧):采样 300 < 2000 → 仍走内存路径。
>
> 即:`--stream_threshold` 只在 `--frames_per_video` 产生了**大帧数**(大 N,或对长视频取全帧)
> 时才真正起作用。

### DINO backbone(`--dino_model`)

DINO 用 HuggingFace `transformers` 的原生 DINOv3 实现(`transformers` 已在依赖中)。可选两个:

| `--dino_model` | 架构 | 权重(本地 HF 格式目录) |
|----------------|------|-------------------------|
| `dinov3_vits16plus`(默认) | DINOv3 ViT-S+/16,embed_dim 384 | `third_party/dinov3/checkpoints/dinov3-vits16plus-hf/` |
| `dinov3_vits16` | DINOv3 ViT-S/16(更轻),embed_dim 384 | `third_party/dinov3/checkpoints/dinov3-vits16-hf/` |

两者输出形状一致(`(T, 1025, 384)` = CLS + 1024 patch,已剔除 register tokens)。
`--dino_model` 也接受 HF 风格别名(如 `facebook/dinov3-vits16-pretrain-lvd1689m`、`dinov3-vits16`)。

**1) 准备本地 HF 格式权重**(`config.json` + `*.safetensors`)。HF 仓库是 gated,需先在
huggingface.co 同意许可、再登录下载(在仓库根目录执行;`hf` 是 huggingface_hub 自带 CLI):

```bash
uv run hf auth login        # 输入有该 gated 仓库访问权的 token
uv run hf download facebook/dinov3-vits16-pretrain-lvd1689m \
    --local-dir third_party/dinov3/checkpoints/dinov3-vits16-hf
# 默认 vits16plus:
uv run hf download facebook/dinov3-vits16plus-pretrain-lvd1689m \
    --local-dir third_party/dinov3/checkpoints/dinov3-vits16plus-hf
```

> 代码按约定查找 `<assets_root>/third_party/dinov3/checkpoints/dinov3-vits16-hf`
> (`assets_root` 默认=仓库根)。放别处时用 `--assets_root /ROOT` 或
> `FEATURE_EXTRACTOR_ASSETS=/ROOT`(仅改根,子路径不变),或把该约定路径软链过去。
> 缺目录会明确报错 `DINOv3 HF weights dir not found: ...`。

**2) 命令行 / 库用法**:

```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/clips --output_root data/features \
    --branches dino --dino_model dinov3_vits16
```

```python
from feature_extractor import DINOExtractor

dino = DINOExtractor(model_name="dinov3_vits16", device="cuda")   # 或 dinov3_vits16plus(默认)
feats = dino.extract_video("clip.mp4", frame_indices=[0, 8, 16])  # (3, 1025, 384)
```

### 6.1 全帧流式提取与批处理容错

针对**长视频全帧提取**(如 10 分钟 @30Hz ≈ 18000 帧)新增流式路径,解决"整段一次性载入
内存导致 OOM、进程被杀、目录内剩余视频不再处理"的问题。

- **自动触发**:当某视频的采样帧数 `> --stream_threshold` 时,该视频走**流式分块提取**——
  按 `--block_size` 逐块读帧、推理、**增量写入 HDF5**,任一时刻内存只占一个块,不随视频
  长度增长。帧数少于阈值时仍走原内存路径,输出格式完全一致。
- **增量写盘 + 完成标记**:每块写完即落盘;分支全部写完后打 `complete=True` 标记。
  崩溃只会留下带部分数据的 `.h5`,不会丢已写块。
- **`--resume` 语义**:改为跳过"**所请求分支全部 `complete`**"的视频;上次崩溃留下的
  半成品(无 `complete` 标记)会被**重跑覆盖**而非误跳过。
- **批处理容错**:单个视频失败(损坏文件、读帧异常等)只计入 `Failures` 并继续下一个,
  **不再中止整批**。

**命令行用法**:

```bash
# 1) DINO 全帧流式提取(--frames_per_video 0 取全部帧;块长 512)
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/clips --output_root data/features \
    --branches dino --frames_per_video 0 --block_size 512

# 2) 长视频 + 断点续跑(崩溃/中断后重跑,已完成的视频自动 [SKIP],半成品自动重跑覆盖)
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/clips --output_root data/features \
    --branches dino --frames_per_video 0 --block_size 512 --resume

# 3) 显式调低触发阈值:让采样帧数 >64 的视频也走流式(调试/小显存时验证流式路径)
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/clips --output_root data/features \
    --branches dino --frames_per_video 0 --stream_threshold 64 --block_size 512
```

跑完控制台会打印 `Successes / Failures / Output`;损坏或读帧失败的视频计入 `Failures`
且**不影响其余视频**(见上文「批处理容错」)。也可用 `python -m feature_extractor.cli ...`
代替 `feature-extract`,参数完全相同。

> ⚠️ **当前仅 `dino` 分支支持全帧流式**。`depth` / `pose` 仍走内存路径,对超长视频全帧仍会
> OOM;`--depth_overlap` / `--pose_window` / `--pose_overlap` 参数已预留,将在 Phase 2/3
> (Depth 分段对齐、Pose 滑窗拼接)启用。做长视频全帧时,请先只对 `--branches dino` 使用。

## 7. 深度模式(`--depth_mode`)

默认 `video_depth_anything`(需对应权重)。各模式权重须已在 `third_party/.../checkpoints/`:

| 模式 | 后端 | 额外依赖 |
|------|------|---------|
| `video_depth_anything`(默认) | Video Depth Anything(`vitl`,vendored)+ Depth Pro 度量校正 | VDA 权重(OBS)+ Depth Pro(HF) |
| `da3` | Depth Anything V3 | `pip install depth_anything_3` |
| `depth_pro` | Apple Depth Pro(HuggingFace) | Depth Pro HF 权重 |

`video_depth_anything`(默认)在关键帧用 **Depth Pro** 做米制校正,`depth_pro` 模式直接用 Depth Pro。
Depth Pro 现走 HuggingFace `transformers`,需本地 HF 格式权重目录:

```bash
uv run hf download apple/DepthPro-hf \
    --local-dir third_party/ml-depth-pro/checkpoints/depth-pro-hf
```

## 8. 自验证(feature-validate)

模块自带一套功能(合理性/不变量)+ 性能(吞吐/显存/扩展性)的自验证工具,跑完出一份
Markdown 报告。用法、参数、报告解读见 [`docs/validation/README.md`](docs/validation/README.md),
样例报告见 [`docs/validation/sample_report.md`](docs/validation/sample_report.md)。

```bash
# 最快:只验功能(合成视频,几十秒)
CUDA_VISIBLE_DEVICES=7 uv run feature-validate --branches dino --skip-perf --report report.md
```

## 9. 作为库使用

**一次性提取**(返回完整数组,适合帧数不多的场景):

```python
from feature_extractor import DINOExtractor, DepthExtractor, PoseExtractor, FeatureStore

dino = DINOExtractor(model_name="dinov3_vits16plus", device="cuda")
feats = dino.extract_video("clip.mp4", frame_indices=[0, 8, 16])  # (3, N+1, 384)
```

**流式提取**(全帧/长视频,内存按块封顶、逐块写入 store;见第 6.1 节)。
`extract_video_streaming` **不返回数组**,而是直接把每块写进 `FeatureStore` 并在结束时打
`complete` 标记,结果通过 `store.read_*` 取回:

```python
from feature_extractor import DINOExtractor, FeatureStore

store = FeatureStore("data/features")
dino = DINOExtractor(model_name="dinov3_vits16plus", device="cuda")

# frame_indices 传全部帧即可做全帧;block_size 控制每块帧数(内存上限)
dino.extract_video_streaming(
    "clip.mp4",
    frame_indices=list(range(n_frames)),
    store=store,
    video_id="clip",
    block_size=512,
)

feats = store.read_dino("clip")                 # (n_frames, N+1, 384) float32
assert store.is_branch_complete("clip", "dino") # 写盘完成标记
```

> 目前仅 `DINOExtractor` 提供 `extract_video_streaming`;`DepthExtractor` / `PoseExtractor`
> 的流式接口将在 Phase 2/3 加入(见第 6.1 节)。底层分块工具在
> `feature_extractor.chunking`(`plan_blocks` / `iter_frame_blocks`),增量写入接口为
> `FeatureStore.write_{dino,depth,pose}_chunk(...)`。

## 10. 已知限制与注意事项

- **video_id 命名冲突**:见第 4 节;多相机/多 chunk 同名文件会互相覆盖,交接后若处理
  LeRobot 类数据需先解决命名。
- **Depth 有损**:逆深度按 uint16 存储,读出有 ≈1/65535 量化误差;平移为归一化尺度,非米制。
- **全帧流式仅 DINO**:`depth` / `pose` 分支暂未流式化,对超长视频全帧仍会 OOM(Phase 2/3 处理);
  相关重叠/窗口参数已预留但未生效。见第 6.1 节。
- **LeRobot 数据**:常把多条 episode 打包进少数 chunk mp4,模块按**整文件**采样,不按
  episode 切分。如需逐 episode,需额外读 `episode_index` 自行分段。
- **性能为解码受限**:对长视频做稀疏采样时,解码(尤其 AV1 软解)往往是主要开销,推理占比小;
  自验证报告里单列了 `decode` 行可据此拆分。
- **依赖钉版**:`numpy<2`(vendored VGGT 要求);torch/torchvision 钉 CUDA 12.4 wheel
  (见 `pyproject.toml`,避免在 12.5 驱动上回退 CPU)。
- **权重外置**:third_party 第三方源码已 vendored 进仓库,但模型权重不在 git 里;换机器需按
  第 3 / 6 节重新下载(VGGT/VDA/depth-pro 走 OBS,DINO 走 HF)。
