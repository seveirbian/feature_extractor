def test_top_level_exports():
    import feature_extractor as fe

    assert hasattr(fe, "DINOExtractor")
    assert hasattr(fe, "DepthExtractor")
    assert hasattr(fe, "PoseExtractor")
    assert hasattr(fe, "FeatureStore")
