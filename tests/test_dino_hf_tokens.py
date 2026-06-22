"""DINOv3 HF 后端的 token 切片契约测试(不需真权重)。"""

import torch

from feature_extractor.extractors.dino import DINOExtractor


def test_slice_hf_tokens_drops_register():
    # 构造 (1, 1 + R + P, D);R=4 register,P=6 patch
    R, P, D = 4, 6, 8
    h = torch.arange(1 * (1 + R + P) * D, dtype=torch.float32).reshape(1, 1 + R + P, D)
    out = DINOExtractor._slice_hf_tokens(h, R)
    assert out.shape == (1, 1 + P, D)          # CLS + patches,register 被剔除
    assert torch.equal(out[:, 0, :], h[:, 0, :])           # CLS 保留
    assert torch.equal(out[:, 1:, :], h[:, 1 + R:, :])     # patches = 跳过 register 之后


def test_hf_model_output_layout_and_slice():
    # 用随机初始化的真实 DINOv3ViT 验证布局假设:512 输入 → 1+R+1024,切片得 1025
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, patch_size=16, image_size=512, num_register_tokens=4,
    )
    model = DINOv3ViTModel(cfg).eval()
    px = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        h = model(pixel_values=px).last_hidden_state
    P = (512 // 16) * (512 // 16)              # 1024
    assert h.shape == (1, 1 + 4 + P, cfg.hidden_size)
    sliced = DINOExtractor._slice_hf_tokens(h, cfg.num_register_tokens)
    assert sliced.shape == (1, 1 + P, cfg.hidden_size)   # 1025
