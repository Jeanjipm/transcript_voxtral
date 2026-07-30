"""Tests de model_manager.py — catalogue, usages, et gestion du disque."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import model_manager as mm
from model_manager import (
    USAGE_DICTATION,
    USAGE_FILES,
    format_size,
    list_available_models,
    models_in_use,
)


# ---- Catalogue ----


def test_every_model_supports_dictation():
    """La dictée est l'usage de base : tout modèle du catalogue doit l'assurer."""
    assert all(m.supports(USAGE_DICTATION) for m in mm.AVAILABLE_MODELS)


def test_only_whisper_models_support_files():
    """Les fichiers exigent des horodatages pour se repérer dans un long
    enregistrement — Voxtral n'en produit aucun."""
    for m in list_available_models(USAGE_FILES):
        assert "whisper" in m.repo_id.lower()


def test_file_capable_models_exist():
    assert len(list_available_models(USAGE_FILES)) >= 1


def test_default_file_model_is_in_the_catalog():
    """Régression : le modèle par défaut des fichiers manquait au catalogue,
    donc les Préférences affichaient quatre modèles dont aucun n'était celui
    qui travaillait réellement."""
    from config import FileTranscriptionConfig

    default = FileTranscriptionConfig().model
    assert mm.find_model(default) is not None
    assert mm.find_model(default).supports(USAGE_FILES)


def test_default_dictation_model_is_in_the_catalog():
    from config import ModelConfig

    assert mm.find_model(ModelConfig().name) is not None


def test_full_precision_voxtral_was_removed():
    """8,7 Go pour un gain marginal face au 8-bit : retiré pour ne pas
    déclencher un téléchargement énorme sur un clic de curiosité."""
    assert mm.find_model("mistralai/Voxtral-Mini-3B-2507") is None


def test_turbo_is_flagged_as_unable_to_translate():
    """Les modèles distillés rendent la langue source au lieu de l'anglais."""
    turbo = mm.find_model("mlx-community/whisper-large-v3-turbo")
    assert turbo is not None
    assert turbo.can_translate is False


def test_large_v3_can_translate():
    model = mm.find_model("mlx-community/whisper-large-v3-mlx")
    assert model is not None and model.can_translate is True


def test_list_without_filter_returns_everything():
    assert len(list_available_models()) == len(mm.AVAILABLE_MODELS)


def test_list_returns_a_copy():
    """L'appelant ne doit pas pouvoir muter le catalogue."""
    listed = list_available_models()
    listed.clear()
    assert len(mm.AVAILABLE_MODELS) > 0


# ---- format_size ----


@pytest.mark.parametrize(
    "size,expected",
    [
        (5_400_000_000, "5.4 Go"),
        (1_600_000_000, "1.6 Go"),
        (236_000_000, "236 Mo"),
        (12_000, "12 Ko"),
    ],
)
def test_format_size(size: int, expected: str):
    assert format_size(size) == expected


# ---- models_in_use ----


def test_models_in_use_covers_both_roles():
    used = models_in_use("org/dictee", "org/fichier")
    assert "org/dictee" in used
    assert "org/fichier" in used


def test_models_in_use_includes_translation_fallback():
    """VoxtralTranscriber délègue la traduction à Whisper : le supprimer
    casserait cette fonction sans que le lien soit visible."""
    from transcriber import WHISPER_TRANSLATE_REPO

    assert WHISPER_TRANSLATE_REPO in models_in_use("a/b", "c/d")


# ---- scan_cached_models ----


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch):
    """Simule scan_cache_dir avec deux repos de tailles connues."""

    def make_repo(repo_id: str, size: int, repo_type: str = "model"):
        repo = MagicMock()
        repo.repo_id = repo_id
        repo.size_on_disk = size
        repo.repo_type = repo_type
        return repo

    info = MagicMock()
    info.repos = [
        make_repo("mzbac/voxtral-mini-3b-8bit", 5_400_000_000),
        make_repo("inconnu/modele-exotique", 900_000_000),
        make_repo("un/dataset", 100, repo_type="dataset"),
    ]
    hub = MagicMock()
    hub.scan_cache_dir.return_value = info
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


def test_scan_lists_models_largest_first(fake_cache):
    entries = mm.scan_cached_models()
    assert [e.repo_id for e in entries] == [
        "mzbac/voxtral-mini-3b-8bit",
        "inconnu/modele-exotique",
    ]


def test_scan_ignores_non_model_repos(fake_cache):
    """Un dataset dans le cache n'est pas un modèle à proposer à la
    suppression depuis cet écran."""
    assert all(e.repo_id != "un/dataset" for e in mm.scan_cached_models())


def test_scan_labels_known_models(fake_cache):
    entry = mm.scan_cached_models()[0]
    assert entry.in_catalog is True
    assert entry.label == "Voxtral Mini 3B (8-bit)"


def test_scan_falls_back_to_repo_id_for_unknown(fake_cache):
    """Un modèle hors catalogue (essayé à la main, ou reliquat) doit tout de
    même apparaître : c'est souvent lui qui occupe la place."""
    entry = mm.scan_cached_models()[1]
    assert entry.in_catalog is False
    assert entry.label == "inconnu/modele-exotique"


def test_scan_returns_empty_when_cache_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Un cache absent ou corrompu ne doit pas empêcher d'ouvrir les
    Préférences."""
    hub = MagicMock()
    hub.scan_cache_dir.side_effect = OSError("cache illisible")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert mm.scan_cached_models() == []


def test_total_cache_size_sums_models(fake_cache):
    assert mm.total_cache_size() == 5_400_000_000 + 900_000_000


def test_size_str_is_human_readable(fake_cache):
    assert mm.scan_cached_models()[0].size_str == "5.4 Go"


# ---- delete_cached_model ----


def test_delete_removes_all_revisions_of_the_repo(monkeypatch: pytest.MonkeyPatch):
    revision = MagicMock()
    revision.commit_hash = "abc123"
    repo = MagicMock()
    repo.repo_id = "org/modele"
    repo.revisions = [revision]

    strategy = MagicMock()
    strategy.expected_freed_size = 1_500_000_000
    info = MagicMock()
    info.repos = [repo]
    info.delete_revisions.return_value = strategy

    hub = MagicMock()
    hub.scan_cache_dir.return_value = info
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    freed = mm.delete_cached_model("org/modele")

    info.delete_revisions.assert_called_once_with("abc123")
    strategy.execute.assert_called_once()
    assert freed == 1_500_000_000


def test_delete_unknown_repo_is_a_noop(monkeypatch: pytest.MonkeyPatch):
    info = MagicMock()
    info.repos = []
    hub = MagicMock()
    hub.scan_cache_dir.return_value = info
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert mm.delete_cached_model("org/absent") == 0
    info.delete_revisions.assert_not_called()
