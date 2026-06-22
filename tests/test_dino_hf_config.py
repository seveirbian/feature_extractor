"""dinov3_*_hf 后端配置的纯单测(不加载模型/权重)。"""

from feature_extractor.extractors.dino import DINOExtractor


def test_hf_configs_present():
    cfgs = DINOExtractor.MODEL_CONFIGS
    for name in ("dinov3_vits16_hf", "dinov3_vits16plus_hf"):
        assert name in cfgs, name
        c = cfgs[name]
        assert c["family"] == "dinov3_hf"
        assert c["embed_dim"] == 384
        assert c["patch_size"] == 16
        assert c["img_size"] == 512
        assert "checkpoints" in c["weights"]


def test_hf_does_not_change_vendored_aliases():
    # HF 风格别名仍指向 vendored,不被 _hf 抢占
    al = DINOExtractor.MODEL_ALIASES
    assert al["facebook/dinov3-vits16-pretrain-lvd1689m"] == "dinov3_vits16"
    assert al["facebook/dinov3-vits16plus-pretrain-lvd1689m"] == "dinov3_vits16plus"
