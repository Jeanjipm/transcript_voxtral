"""Tests de config.py — chargement, fusion defaults+user, save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    Config,
    HotkeyConfig,
    ModelConfig,
    SoundsConfig,
    TranscriptionConfig,
    UIConfig,
    UpdatesConfig,
    _build,
    _deep_merge,
    _dict_to_config,
    ensure_user_config_exists,
    load_config,
    save_config,
)


# ---- _build : tolérance aux clés inconnues / manquantes ----


def test_build_accepts_known_fields():
    cfg = _build(HotkeyConfig, {"combo": "ctrl+space"})
    assert cfg.combo == "ctrl+space"


def test_build_ignores_unknown_fields():
    """Anciens config.yaml avec champs retirés ne doivent pas crasher."""
    cfg = _build(HotkeyConfig, {"combo": "alt_r", "obsolete_field": "x"})
    assert cfg.combo == "alt_r"


def test_build_uses_defaults_for_missing_fields():
    cfg = _build(HotkeyConfig, {})
    assert cfg.combo == "alt_r"  # default


# ---- _deep_merge ----


def test_deep_merge_overrides_top_level():
    base = {"a": 1, "b": 2}
    override = {"b": 20}
    assert _deep_merge(base, override) == {"a": 1, "b": 20}


def test_deep_merge_recurses_into_dicts():
    base = {"section": {"a": 1, "b": 2}}
    override = {"section": {"b": 20}}
    assert _deep_merge(base, override) == {"section": {"a": 1, "b": 20}}


def test_deep_merge_preserves_base_keys_absent_from_override():
    base = {"a": 1, "b": 2, "c": 3}
    override = {"b": 20}
    assert _deep_merge(base, override) == {"a": 1, "b": 20, "c": 3}


def test_deep_merge_replaces_dict_with_scalar():
    """Edge case : si override remplace un dict par autre chose."""
    base = {"section": {"a": 1}}
    override = {"section": "scalaire"}
    assert _deep_merge(base, override) == {"section": "scalaire"}


# ---- _dict_to_config ----


def test_dict_to_config_with_full_dict():
    data = {
        "model": {"name": "test/model"},
        "hotkey": {"combo": "ctrl+s"},
        "transcription": {"language": "fr", "task": "translate"},
        "sounds": {"enabled": False, "volume": 0.8},
        "ui": {"auto_paste": False},
        "updates": {"auto_check": False},
    }
    cfg = _dict_to_config(data)
    assert cfg.model.name == "test/model"
    assert cfg.hotkey.combo == "ctrl+s"
    assert cfg.transcription.language == "fr"
    assert cfg.transcription.task == "translate"
    assert cfg.sounds.enabled is False
    assert cfg.sounds.volume == 0.8
    assert cfg.ui.auto_paste is False
    assert cfg.updates.auto_check is False


def test_dict_to_config_with_empty_dict_uses_defaults():
    cfg = _dict_to_config({})
    assert cfg.model.name == ModelConfig().name
    assert cfg.hotkey.combo == HotkeyConfig().combo
    assert cfg.updates.auto_check is True


def test_dict_to_config_partial_overrides():
    """Partial override : juste hotkey, le reste reste default."""
    cfg = _dict_to_config({"hotkey": {"combo": "f13"}})
    assert cfg.hotkey.combo == "f13"
    assert cfg.model.name == ModelConfig().name
    assert cfg.updates.auto_check is True  # default


# ---- load_config ----


def test_load_config_defaults_only(tmp_path: Path):
    """User file absent → on retourne les defaults projet."""
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "model:\n  name: defaults-model\n",
        encoding="utf-8",
    )
    user_path = tmp_path / "user.yaml"  # absent

    cfg = load_config(user_path=user_path, default_path=default_path)
    assert cfg.model.name == "defaults-model"


def test_load_config_user_overrides_defaults(tmp_path: Path):
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "model:\n  name: defaults-model\nhotkey:\n  combo: alt_r\n",
        encoding="utf-8",
    )
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        "hotkey:\n  combo: ctrl+space\n",
        encoding="utf-8",
    )

    cfg = load_config(user_path=user_path, default_path=default_path)
    assert cfg.hotkey.combo == "ctrl+space"
    assert cfg.model.name == "defaults-model"  # défault préservé


def test_load_config_empty_user_file(tmp_path: Path):
    """User file vide (yaml.safe_load → None) ne doit pas crasher."""
    default_path = tmp_path / "default.yaml"
    default_path.write_text("model:\n  name: x\n", encoding="utf-8")
    user_path = tmp_path / "user.yaml"
    user_path.write_text("", encoding="utf-8")

    cfg = load_config(user_path=user_path, default_path=default_path)
    assert cfg.model.name == "x"


# ---- save_config + roundtrip ----


def test_save_and_reload_roundtrip(tmp_path: Path):
    default_path = tmp_path / "default.yaml"
    default_path.write_text("model:\n  name: defaults\n", encoding="utf-8")
    user_path = tmp_path / "user.yaml"

    cfg = Config(
        model=ModelConfig(name="custom/model"),
        hotkey=HotkeyConfig(combo="cmd+shift+v"),
        transcription=TranscriptionConfig(language="fr", task="translate"),
        sounds=SoundsConfig(enabled=False, volume=0.3),
        ui=UIConfig(auto_paste=False),
        updates=UpdatesConfig(auto_check=False),
    )
    save_config(cfg, user_path=user_path)

    reloaded = load_config(user_path=user_path, default_path=default_path)
    assert reloaded.model.name == "custom/model"
    assert reloaded.hotkey.combo == "cmd+shift+v"
    assert reloaded.transcription.task == "translate"
    assert reloaded.sounds.enabled is False
    assert reloaded.ui.auto_paste is False
    assert reloaded.updates.auto_check is False


def test_save_config_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "nested" / "dir" / "config.yaml"
    cfg = Config()
    save_config(cfg, user_path=target)
    assert target.exists()


# ---- ensure_user_config_exists ----


def test_ensure_user_config_exists_creates_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Premier lancement : pas de fichier → on le crée depuis les defaults."""
    user_dir = tmp_path / ".voxtral"
    user_path = user_dir / "config.yaml"
    monkeypatch.setattr("config.USER_CONFIG_PATH", user_path)
    monkeypatch.setattr("config.USER_CONFIG_DIR", user_dir)

    result = ensure_user_config_exists()
    assert result == user_path
    assert user_path.exists()


def test_ensure_user_config_exists_keeps_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Si le fichier user existe, on ne l'écrase pas."""
    user_dir = tmp_path / ".voxtral"
    user_dir.mkdir()
    user_path = user_dir / "config.yaml"
    user_path.write_text("model:\n  name: my-precious\n", encoding="utf-8")

    monkeypatch.setattr("config.USER_CONFIG_PATH", user_path)
    monkeypatch.setattr("config.USER_CONFIG_DIR", user_dir)

    ensure_user_config_exists()
    assert "my-precious" in user_path.read_text(encoding="utf-8")


# ---- to_dict ----


def test_config_to_dict_roundtrip():
    cfg = Config(
        hotkey=HotkeyConfig(combo="f13"),
        sounds=SoundsConfig(enabled=False),
    )
    d = cfg.to_dict()
    assert d["hotkey"]["combo"] == "f13"
    assert d["sounds"]["enabled"] is False
    # roundtrip via dict
    cfg2 = _dict_to_config(d)
    assert cfg2.hotkey.combo == "f13"
    assert cfg2.sounds.enabled is False
