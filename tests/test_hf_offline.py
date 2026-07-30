"""Tests d'hf_offline.py — détection du cache, bascule hors-ligne, résolution locale."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import hf_offline


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch):
    """Remplace huggingface_hub par un stub contrôlable.

    `hf_offline` importe hf_hub à l'intérieur des fonctions (import retardé),
    donc on patche l'entrée de sys.modules plutôt qu'un attribut du module.
    """
    hub = MagicMock(name="huggingface_hub")
    hub.constants = MagicMock(name="constants")
    hub.constants.HF_HUB_OFFLINE = False
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub)
    return hub


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Isole la variable d'environnement entre les tests."""
    monkeypatch.delenv(hf_offline.ENV_VAR, raising=False)


# ---- is_model_cached ----


def test_is_model_cached_true_when_path_returned(fake_hub):
    """try_to_load_from_cache qui rend un chemin => modèle en cache."""
    fake_hub.try_to_load_from_cache.return_value = "/cache/models--x/snapshots/a/config.json"
    assert hf_offline.is_model_cached("org/modele") is True


def test_is_model_cached_false_when_none(fake_hub):
    """None => repo absent du cache."""
    fake_hub.try_to_load_from_cache.return_value = None
    assert hf_offline.is_model_cached("org/modele") is False


def test_is_model_cached_false_on_cached_no_exist_sentinel(fake_hub):
    """L'objet sentinelle _CACHED_NO_EXIST ne doit pas compter comme un hit."""
    fake_hub.try_to_load_from_cache.return_value = object()
    assert hf_offline.is_model_cached("org/modele") is False


def test_is_model_cached_false_when_hub_raises(fake_hub):
    """Une API hf_hub qui lève ne doit pas propager : on répond False, donc
    on n'active pas le hors-ligne, donc on ne bloque aucun téléchargement."""
    fake_hub.try_to_load_from_cache.side_effect = RuntimeError("api cassée")
    assert hf_offline.is_model_cached("org/modele") is False


# ---- set_offline / is_offline ----


def test_set_offline_sets_both_env_and_constant(fake_hub):
    """La constante hf_hub est figée à l'import : il faut poser LES DEUX,
    sinon le mode hors-ligne n'a aucun effet réel."""
    hf_offline.set_offline(True)
    assert os.environ[hf_offline.ENV_VAR] == "1"
    assert fake_hub.constants.HF_HUB_OFFLINE is True


def test_set_offline_false_clears_both(fake_hub):
    hf_offline.set_offline(True)
    hf_offline.set_offline(False)
    assert os.environ[hf_offline.ENV_VAR] == "0"
    assert fake_hub.constants.HF_HUB_OFFLINE is False


# ---- refresh ----


def test_refresh_enables_offline_when_cached(fake_hub):
    fake_hub.try_to_load_from_cache.return_value = "/cache/x/config.json"
    assert hf_offline.refresh("org/modele") is True
    assert fake_hub.constants.HF_HUB_OFFLINE is True


def test_refresh_allows_network_when_not_cached(fake_hub, capsys):
    """Modèle absent => réseau autorisé, et on le dit sur stderr (sinon
    l'utilisateur ne comprend pas pourquoi un téléchargement démarre)."""
    fake_hub.try_to_load_from_cache.return_value = None
    assert hf_offline.refresh("org/absent") is False
    assert fake_hub.constants.HF_HUB_OFFLINE is False
    assert "absent du cache" in capsys.readouterr().err


def test_refresh_respects_prefer_offline_false(fake_hub):
    """prefer_offline=false dans la config => on n'active jamais le hors-ligne,
    même si le modèle est en cache."""
    fake_hub.try_to_load_from_cache.return_value = "/cache/x/config.json"
    assert hf_offline.refresh("org/modele", prefer_offline=False) is False
    assert fake_hub.constants.HF_HUB_OFFLINE is False


# ---- allow_network ----


def test_allow_network_restores_previous_state(fake_hub):
    hf_offline.set_offline(True)
    with hf_offline.allow_network():
        assert fake_hub.constants.HF_HUB_OFFLINE is False
    assert fake_hub.constants.HF_HUB_OFFLINE is True


def test_allow_network_restores_even_on_exception(fake_hub):
    """Un téléchargement qui échoue ne doit pas laisser l'app en mode
    connecté : sinon on retombe sur les erreurs DNS au chargement suivant."""
    hf_offline.set_offline(True)
    with pytest.raises(RuntimeError):
        with hf_offline.allow_network():
            raise RuntimeError("téléchargement interrompu")
    assert fake_hub.constants.HF_HUB_OFFLINE is True


# ---- resolve_local_path ----


def test_resolve_local_path_returns_snapshot_dir(fake_hub, tmp_path: Path):
    """Le contournement du bug mistral_common : on doit rendre le DOSSIER du
    snapshot, pas l'identifiant du repo."""
    snapshot = tmp_path / "models--org--m" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    fake_hub.try_to_load_from_cache.return_value = str(snapshot / "config.json")

    assert hf_offline.resolve_local_path("org/m") == str(snapshot)


def test_resolve_local_path_falls_back_to_repo_id_when_absent(fake_hub):
    """Pas en cache => on rend le repo_id, pour que le téléchargement
    normal puisse avoir lieu."""
    fake_hub.try_to_load_from_cache.return_value = None
    assert hf_offline.resolve_local_path("org/m") == "org/m"


def test_resolve_local_path_falls_back_when_dir_missing(fake_hub, tmp_path: Path):
    """Cache incohérent (chemin annoncé mais dossier absent) => repli sûr."""
    fake_hub.try_to_load_from_cache.return_value = str(
        tmp_path / "inexistant" / "config.json"
    )
    assert hf_offline.resolve_local_path("org/m") == "org/m"


def test_resolve_local_path_falls_back_when_hub_raises(fake_hub):
    fake_hub.try_to_load_from_cache.side_effect = RuntimeError("boom")
    assert hf_offline.resolve_local_path("org/m") == "org/m"


# ---- install_early ----


def test_install_early_defaults_to_offline():
    hf_offline.install_early()
    assert os.environ[hf_offline.ENV_VAR] == "1"


def test_install_early_respects_user_override(monkeypatch: pytest.MonkeyPatch):
    """Un utilisateur qui a explicitement mis HF_HUB_OFFLINE=0 dans son
    environnement garde son choix."""
    monkeypatch.setenv(hf_offline.ENV_VAR, "0")
    hf_offline.install_early()
    assert os.environ[hf_offline.ENV_VAR] == "0"
