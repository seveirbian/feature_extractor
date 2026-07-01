"""Pure segment-boundary helpers for streaming VDA depth metricization.

No torch/model imports: given a segment's source frame indices and carried
keyframe state, decide which rows are Depth Pro keyframes and interpolate the
per-frame affine (scale, shift) that maps VDA raw depth to metric depth.
"""

from __future__ import annotations

import numpy as np


def plan_segment_keyframes(
    source_indices: list[int],
    last_kf_source_idx: int | None,
    keyframe_interval: int,
) -> list[int]:
    """Return row indices (into this segment) that should be Depth Pro keyframes.

    Keyframes are spaced by ``keyframe_interval`` over the *global* source frame
    index. The first-ever segment (``last_kf_source_idx is None``) anchors its
    own row 0. The last row is always forced so interpolation is bounded and the
    next segment inherits a valid anchor.
    """
    n = len(source_indices)
    if n == 0:
        return []

    rows: list[int] = []
    if last_kf_source_idx is None:
        rows.append(0)
        last = source_indices[0]
    else:
        last = last_kf_source_idx

    for r in range(n):
        if r in rows:
            continue
        if source_indices[r] - last >= keyframe_interval:
            rows.append(r)
            last = source_indices[r]

    if rows[-1] != n - 1:
        rows.append(n - 1)
    return rows
