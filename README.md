# feature_extractor

从 egoWM 中独立出来的 DINO / Depth / Pose 特征提取工具。基于 HDF5 的特征存储,提取行为与 egoWM 完全一致,模型资源根目录可配置。

## 安装(uv)

```bash
# 克隆时带上子模块,或克隆后再初始化:
git submodule update --init --recursive

uv sync
```

## 模型资源(third_party/)

提取器依赖四个内置模型仓库,以 **git 子模块** 形式放在 `third_party/` 下
(`dinov3`、`Video-Depth-Anything`、`ml-depth-pro`、`VGGT`)。
`git submodule update --init` 会从 GitHub 拉取钉定版本的代码。

### 权重文件(不在 git 里)

模型**权重不纳入版本管理**(体积达数 GB,上游仓库本身也不含)。各后端期望权重位于
`third_party/<repo>/checkpoints/` 下,例如:

```
third_party/dinov3/checkpoints/dinov3_vits16plus_pretrain_lvd1689m-*.pth
third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth
third_party/ml-depth-pro/checkpoints/depth_pro.pt
third_party/VGGT/checkpoints/models--facebook--VGGT-1B/...
```

权重已托管在华为云 OBS,用 [`obsutil`](https://support.huaweicloud.com/utiltg-obs/obs_11_0003.html)
在仓库根目录执行下面的命令下载到对应目录(需先用 `obsutil config` 配好 AK/SK):

```bash
obsutil cp obs://cloudrobo-model/wangchao/egoWM/third_party/dinov3/checkpoints               ./third_party/dinov3/checkpoints               -r -f
obsutil cp obs://cloudrobo-model/wangchao/egoWM/third_party/Video-Depth-Anything/checkpoints ./third_party/Video-Depth-Anything/checkpoints -r -f
obsutil cp obs://cloudrobo-model/wangchao/egoWM/third_party/ml-depth-pro/checkpoints          ./third_party/ml-depth-pro/checkpoints          -r -f
obsutil cp obs://cloudrobo-model/wangchao/egoWM/third_party/VGGT/checkpoints                  ./third_party/VGGT/checkpoints                  -r -f
```

只用部分后端时,只下对应的那几行即可(例如默认 `dino_attention` 不需要任何权重;
`video_depth_anything` 需要 `Video-Depth-Anything` 与 `ml-depth-pro` 两份)。

## 命令行(CLI)

```bash
uv run feature-extract \
    --data_root data/openego/ \
    --output_root data/features \
    --device cuda \
    --branches dino,depth,pose \
    --depth_mode video_depth_anything \
    --frames_per_video 64 \
    --id_from_stem
```

`--data_root` 会被递归遍历查找视频文件(例如
`data/openego/HO-Cap/Ho-Cap/demo_*/video.mp4`)。

### 深度模式(`--depth_mode`)

默认是 `dino_attention`,一种 **DINO 注意力代理**——快且无额外依赖,但**不是**几何深度。
若需要真实深度,请选择某个模型后端(其权重须已位于 `third_party/.../checkpoints/` 下):

| 模式 | 后端 | 额外依赖 |
|------|------|---------|
| `dino_attention`(默认) | DINO 注意力代理 | 无 |
| `video_depth_anything` | Video Depth Anything(`vitl`)+ Depth Pro 度量校正 | 已包含在 `uv sync` 中 |
| `da3` | Depth Anything V3 | `pip install depth_anything_3` |
| `depth_pro` | Apple Depth Pro | 已包含在 `uv sync` 中 |

## 作为库使用

```python
from feature_extractor import DINOExtractor, DepthExtractor, PoseExtractor, FeatureStore
```