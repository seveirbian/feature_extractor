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
third_party/VGGT/checkpoints/models--facebook--VGGT-1B/...
```

请从各模型的上游下载;若本机已有一份(例如某个 egoWM 检出),可直接软链复用,免去重复下载:

```bash
for d in VGGT Video-Depth-Anything dinov3 ml-depth-pro; do
  ln -sfn /root/codes/egoWM/third_party/$d/checkpoints third_party/$d/checkpoints
done
```

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