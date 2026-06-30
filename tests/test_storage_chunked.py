import numpy as np

from feature_extractor.storage import FeatureStore


def test_write_dino_chunk_appends_and_reads_back_in_order(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "clip0"
    chunk0 = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    chunk1 = (chunk0 + 100).copy()

    store.write_dino_chunk(vid, chunk0, np.array([0, 1]), reset=True)
    store.write_dino_chunk(vid, chunk1, np.array([2, 3]), reset=False)

    feats = store.read_dino(vid)
    assert feats.shape == (4, 3, 5)
    np.testing.assert_allclose(feats[:2], chunk0)
    np.testing.assert_allclose(feats[2:], chunk1)
    np.testing.assert_array_equal(store.read_frame_indices(vid, "dino"), [0, 1, 2, 3])


def test_branch_completion_markers(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    vid = "clip1"
    store.write_dino_chunk(vid, np.zeros((1, 2, 2), np.float32), np.array([0]), reset=True)
    assert store.is_branch_complete(vid, "dino") is False
    store.mark_branch_complete(vid, "dino")
    assert store.is_branch_complete(vid, "dino") is True
    assert store.is_video_complete(vid, ["dino"]) is True
    assert store.is_video_complete(vid, ["dino", "depth"]) is False


def test_is_video_complete_false_when_missing(tmp_path):
    store = FeatureStore(str(tmp_path / "store"))
    assert store.is_video_complete("nope", ["dino"]) is False
