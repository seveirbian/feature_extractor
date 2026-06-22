"""Depth Pro HF 后端:缺本地 HF 权重目录时明确报错。"""

import tempfile

import pytest
import torch

from feature_extractor.extractors.depth import _load_depth_pro


def test_load_depth_pro_missing_hf_dir_raises():
    # assets_root 指向空目录 → 期望目录不存在的 FileNotFoundError
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FileNotFoundError):
            _load_depth_pro(torch.device("cpu"), assets_root=td)
