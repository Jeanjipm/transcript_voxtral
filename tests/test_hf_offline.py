"""Tests d'hf_offline.py — détection du cache, bascule hors-ligne, résolution locale."""

from __future__ import annotations

import json
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


def make_snapshot(
    root: Path,
    fake_hub,
    weights: tuple[str, ...] = ("model.safetensors",),
    index: dict | None = None,
    incomplete_blob: bool = False,
) -> Path:
    """Fabrique un repo du cache HuggingFace tel qu'il est sur le disque :
    `models--org--m/blobs/` et `models--org--m/snapshots/<sha>/`, et branche
    `try_to_load_from_cache` dessus. Les poids sont des liens vers des blobs,
    comme en vrai — un morceau listé dans `index` mais absent de `weights`
    n'a donc ni lien ni blob."""
    repo = root / "models--org--m"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "cfg").write_text("{}", encoding="utf-8")
    (snapshot / "config.json").symlink_to(blobs / "cfg")
    for i, name in enumerate(weights):
        (blobs / f"w{i}").write_bytes(b"\0" * 16)
        (snapshot / name).symlink_to(blobs / f"w{i}")
    if index is not None:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index}), encoding="utf-8"
        )
    if incomplete_blob:
        (blobs / "deadbeef.incomplete").write_bytes(b"\0" * 16)
    fake_hub.try_to_load_from_cache.return_value = str(snapshot / "config.json")
    return snapshot


# ---- is_model_cached ----


def test_is_model_cached_true_for_a_complete_snapshot(fake_hub, tmp_path: Path):
    """config.json + un fichier de poids => modèle en cache."""
    make_snapshot(tmp_path, fake_hub)
    assert hf_offline.is_model_cached("org/modele") is True


def test_is_model_cached_false_when_only_config_json(fake_hub, tmp_path: Path):
    """Le cas vécu : téléchargement tué juste après config.json. Se fier à
    config.json seul activait le hors-ligne sur un modèle sans poids."""
    make_snapshot(tmp_path, fake_hub, weights=())
    assert hf_offline.is_model_cached("org/modele") is False


def test_is_model_cached_false_while_a_blob_is_incomplete(fake_hub, tmp_path: Path):
    """Un blob .incomplete = huggingface_hub n'a pas fini : pas complet,
    même si d'autres poids sont déjà là."""
    make_snapshot(tmp_path, fake_hub, incomplete_blob=True)
    assert hf_offline.is_model_cached("org/modele") is False


def test_is_model_cached_checks_every_shard_of_the_index(fake_hub, tmp_path: Path):
    """Poids découpés (Voxtral 8 bits : 2 morceaux) : il faut TOUS les
    morceaux annoncés par model.safetensors.index.json."""
    index = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    make_snapshot(
        tmp_path, fake_hub, weights=("model-00001-of-00002.safetensors",), index=index
    )
    assert hf_offline.is_model_cached("org/modele") is False


def test_is_model_cached_true_when_all_shards_present(fake_hub, tmp_path: Path):
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    make_snapshot(tmp_path, fake_hub, weights=shards, index={"a": shards[0], "b": shards[1]})
    assert hf_offline.is_model_cached("org/modele") is True


def test_is_model_cached_false_when_index_is_unreadable(fake_hub, tmp_path: Path):
    """Index corrompu : on répond « incomplet », le côté sûr (réseau autorisé,
    snapshot_download vérifie), plutôt que de deviner."""
    snapshot = make_snapshot(tmp_path, fake_hub, weights=("model.safetensors",))
    (snapshot / "model.safetensors.index.json").write_text("{pas du json", encoding="utf-8")
    assert hf_offline.is_model_cached("org/modele") is False


def test_is_model_cached_accepts_whisper_style_weights(fake_hub, tmp_path: Path):
    """Les Whisper MLX n'ont pas d'index : un seul weights.safetensors."""
    make_snapshot(tmp_path, fake_hub, weights=("weights.safetensors",))
    assert hf_offline.is_model_cached("org/modele") is True


def test_is_model_cached_false_when_weight_symlink_dangles(fake_hub, tmp_path: Path):
    """Lien vers un blob supprimé (cache nettoyé à la main) : absent."""
    snapshot = make_snapshot(tmp_path, fake_hub)
    (snapshot / "model.safetensors").resolve().unlink()
    assert hf_offline.is_model_cached("org/modele") is False


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


def test_refresh_enables_offline_when_cached(fake_hub, tmp_path: Path):
    make_snapshot(tmp_path, fake_hub)
    assert hf_offline.refresh("org/modele") is True
    assert fake_hub.constants.HF_HUB_OFFLINE is True


def test_refresh_allows_network_when_not_cached(fake_hub, capsys):
    """Modèle absent => réseau autorisé, et on le dit sur stderr (sinon
    l'utilisateur ne comprend pas pourquoi un téléchargement démarre)."""
    fake_hub.try_to_load_from_cache.return_value = None
    assert hf_offline.refresh("org/absent") is False
    assert fake_hub.constants.HF_HUB_OFFLINE is False
    assert "absent du cache" in capsys.readouterr().err


def test_refresh_respects_prefer_offline_false(fake_hub, tmp_path: Path):
    """prefer_offline=false dans la config => on n'active jamais le hors-ligne,
    même si le modèle est en cache."""
    make_snapshot(tmp_path, fake_hub)
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
    snapshot = make_snapshot(tmp_path, fake_hub)

    assert hf_offline.resolve_local_path("org/m") == str(snapshot)


def test_resolve_local_path_falls_back_when_snapshot_is_partial(
    fake_hub, tmp_path: Path
):
    """Snapshot sans poids : rendre le dossier ferait échouer from_pretrained
    sur « No weight files found ». On rend le repo_id, pour que le
    téléchargement reprenne (le réseau est ouvert par network_if_missing,
    puisque is_model_cached dit la même chose)."""
    make_snapshot(tmp_path, fake_hub, weights=(), incomplete_blob=True)
    assert hf_offline.resolve_local_path("org/m") == "org/m"
    assert hf_offline.is_model_cached("org/m") is False


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


# ---- network_if_missing ----


def test_network_if_missing_opens_network_for_an_absent_model(fake_hub):
    """Modèle absent du cache : le réseau est ouvert le temps du bloc, puis
    l'état précédent (hors-ligne) est rétabli."""
    fake_hub.try_to_load_from_cache.return_value = None
    hf_offline.set_offline(True)

    with hf_offline.network_if_missing("org/absent"):
        assert hf_offline.is_offline() is False

    assert hf_offline.is_offline() is True


def test_network_if_missing_keeps_offline_for_a_cached_model(fake_hub, tmp_path: Path):
    """Modèle en cache : on ne rouvre jamais le réseau sans raison."""
    make_snapshot(tmp_path, fake_hub)
    hf_offline.set_offline(True)

    with hf_offline.network_if_missing("org/present"):
        assert hf_offline.is_offline() is True

    assert hf_offline.is_offline() is True
