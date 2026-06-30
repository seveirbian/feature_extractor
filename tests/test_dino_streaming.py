import numpy as np

from feature_extractor.storage import FeatureStore
from feature_extractor.validation.synthetic import make_ramp_video


class _StubDINO:
    """Exercises extract_video_streaming without loading a real model."""

    embed_dim = 4

    # bind the real method under test
    from feature_extractor.extractors.dino import DINOExtractor
    extract_video_streaming = DINOExtractor.extract_video_streaming

    def extract_frame(self, frame):
        # deterministic per-frame "feature": mean of the frame, broadcast
        return np.full((1, self.embed_dim), float(frame.mean()), dtype=np.float32)


def test_dino_streaming_writes_every_frame(tmp_path):
    path = str(tmp_path / "ramp.mp4")
    make_ramp_video(path, codec="libx264", n_frames=10, width=64, height=48, step=20)
    store = FeatureStore(str(tmp_path / "store"))

    _StubDINO().extract_video_streaming(
        video_path=path,
        frame_indices=list(range(10)),
        store=store,
        video_id="clip",
        block_size=4,
    )

    feats = store.read_dino("clip")
    assert feats.shape == (10, 1, 4)
    np.testing.assert_array_equal(store.read_frame_indices("clip", "dino"), list(range(10)))
    assert store.is_branch_complete("clip", "dino") is True
