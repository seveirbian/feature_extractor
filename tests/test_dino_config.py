"""DINO 后端为 HF-only 的配置/别名断言(不加载模型)。"""

from feature_extractor.extractors.dino import DINOExtractor


def test_only_hf_dinov3_configs():
    cfgs = DINOExtractor.MODEL_CONFIGS
    assert set(cfgs) == {"dinov3_vits16", "dinov3_vits16plus"}
    for name in cfgs:
        c = cfgs[name]
        assert c["family"] == "dinov3_hf"
        assert c["embed_dim"] == 384
        assert c["patch_size"] == 16
        assert c["img_size"] == 512
        assert "checkpoints" in c["weights"] and ".pth" not in c["weights"]


def test_vendored_and_dinov2_removed():
    cfgs = DINOExtractor.MODEL_CONFIGS
    assert "dinov3_vits16_hf" not in cfgs and "dinov3_vits16plus_hf" not in cfgs
    assert not any(n.startswith("dinov2") for n in cfgs)


def test_aliases_point_to_hf():
    al = DINOExtractor.MODEL_ALIASES
    assert al["facebook/dinov3-vits16-pretrain-lvd1689m"] == "dinov3_vits16"
    assert al["facebook/dinov3-vits16plus-pretrain-lvd1689m"] == "dinov3_vits16plus"
    assert not any(v.startswith("dinov2") for v in al.values())
