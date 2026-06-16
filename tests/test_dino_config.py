"""dinov3_vits16 backbone 配置/别名的纯单测(不加载权重)。"""

from feature_extractor.extractors.dino import DINOExtractor


def test_vits16_alias_resolves():
    aliases = DINOExtractor.MODEL_ALIASES
    assert aliases["facebook/dinov3-vits16-pretrain-lvd1689m"] == "dinov3_vits16"
    assert aliases["dinov3-vits16"] == "dinov3_vits16"


def test_vits16_config_fields():
    cfg = DINOExtractor.MODEL_CONFIGS["dinov3_vits16"]
    assert cfg["family"] == "dinov3"
    assert cfg["patch_size"] == 16
    assert cfg["embed_dim"] == 384
    assert cfg["local_repo"] == "third_party/dinov3"
    assert cfg["weights"].endswith("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")


def test_vits16_not_confused_with_vits16plus():
    # 别名不应把 vits16 解析成 vits16plus
    assert DINOExtractor.MODEL_ALIASES.get("dinov3-vits16") != "dinov3_vits16plus"
    assert "dinov3_vits16" in DINOExtractor.MODEL_CONFIGS
