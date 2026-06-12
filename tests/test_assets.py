import os
from pathlib import Path

import pytest

from feature_extractor.assets import resolve_assets_root


PKG_ROOT = Path(__file__).resolve().parents[1]  # /root/codes/feature_extractor


def test_explicit_param_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("FEATURE_EXTRACTOR_ASSETS", str(tmp_path / "env"))
    assert resolve_assets_root(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_env_var_used_when_no_param(monkeypatch, tmp_path):
    monkeypatch.setenv("FEATURE_EXTRACTOR_ASSETS", str(tmp_path / "env"))
    assert resolve_assets_root(None) == (tmp_path / "env").resolve()


def test_default_is_package_root(monkeypatch):
    monkeypatch.delenv("FEATURE_EXTRACTOR_ASSETS", raising=False)
    assert resolve_assets_root(None) == PKG_ROOT


def test_user_expansion(monkeypatch, tmp_path):
    monkeypatch.delenv("FEATURE_EXTRACTOR_ASSETS", raising=False)
    result = resolve_assets_root("~")
    assert result == Path.home().resolve()
