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
CUDA_VISIBLE_DEVICES=0 uv run feature-extract \
    --data_root data/openego/videos \
    --output_root data/features \
    --device cuda \
    --branches dino,depth,pose
```

## Library

```python
from feature_extractor import DINOExtractor, DepthExtractor, PoseExtractor, FeatureStore
```
