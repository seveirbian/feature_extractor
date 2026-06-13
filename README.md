# feature_extractor

Standalone DINO / Depth / Pose feature extraction, extracted from egoWM.
HDF5-backed feature store, identical extraction behavior, configurable model-asset root.

## Install (uv)

```bash
uv sync
```

## Model assets (third_party/)

The extractors need the `third_party/` model repos + checkpoints
(`dinov3`, `Video-Depth-Anything`, `ml-depth-pro`, `VGGT`). Point the package at them
in one of three ways (priority order):

1. `--assets_root /path/to/root` (CLI) or `assets_root=` (constructor)
2. env var `FEATURE_EXTRACTOR_ASSETS=/path/to/root`
3. default: the package root — symlink the assets in:

   ```bash
   ln -sfn /root/codes/egoWM/third_party ./third_party
   ```

The root is the directory that *contains* `third_party/`.

## CLI

```bash
# choose GPU via CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=7 uv run feature-extract \
    --data_root data/openego/ \
    --output_root data/features \
    --device cuda \
    --branches dino,depth,pose \
    --depth_mode video_depth_anything \
    --frames_per_video 64 \
    --id_from_stem
```

`--data_root` is walked recursively for video files (e.g.
`data/openego/HO-Cap/Ho-Cap/demo_*/video.mp4`).

### Depth modes (`--depth_mode`)

The default is `dino_attention`, a **DINO-attention proxy** — fast and
dependency-free, but *not* geometric depth. For real depth, pick a model
backend (its checkpoint must already be under `third_party/.../checkpoints/`):

| mode | backend | extra deps |
|------|---------|-----------|
| `dino_attention` (default) | DINO attention proxy | none |
| `video_depth_anything` | Video Depth Anything (`vitl`) + Depth Pro metric correction | bundled in `uv sync` |
| `da3` | Depth Anything V3 | `pip install depth_anything_3` |
| `depth_pro` | Apple Depth Pro | bundled in `uv sync` |

## Library

```python
from feature_extractor import DINOExtractor, DepthExtractor, PoseExtractor, FeatureStore
```
