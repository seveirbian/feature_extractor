def test_feature_store_importable_and_constructs(tmp_path):
    from feature_extractor.storage import FeatureStore

    store = FeatureStore(str(tmp_path / "store"))
    assert store.list_videos() == []
