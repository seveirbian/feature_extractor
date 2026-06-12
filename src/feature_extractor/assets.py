"""Resolve the asset root directory that contains ``third_party/``.

Priority: explicit argument > env var FEATURE_EXTRACTOR_ASSETS > package root.
The package root is where the ``third_party/`` symlink is expected to live.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_assets_root(assets_root: str | Path | None = None) -> Path:
    """Return the directory that contains the ``third_party/`` model assets."""
    if assets_root is not None:
        return Path(assets_root).expanduser().resolve()
    env = os.environ.get("FEATURE_EXTRACTOR_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    # assets.py is at <pkg_root>/src/feature_extractor/assets.py
    return Path(__file__).resolve().parents[2]
