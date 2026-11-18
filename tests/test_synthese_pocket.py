"""Tests de synthese_pocket.py — la voix Pocket, sans torch ni réseau.

Le paquet `pocket_tts` est simulé. On vérifie ce que le module décide : où il
cherche ses fichiers, quand il rouvre le réseau, ce qu'il dit quand le paquet
manque, et que le son sort bien par morceaux.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

import synthese_pocket
from synthese_pocket import (
    PocketError,
    PocketSynthetiseur,
    code_voix,
    est_en_cache,
    langue_du_code,
    libelles,
)


# ---- Codes de voix ----


def test_voice_codes_carry_a_prefix_that_tells_the_engine():
    assert code_voix("french_24l") == "pocket:french_24l"
    assert langue_du_code("pocket:french_24l") == "french_24l"
    assert langue_du_code("fr_female") is None  # une voix Voxtral


def test_the_large_french_voice_is_listed_first():
    codes = [code for code, _ in libelles()]
    assert codes == ["pocket:french_24l", "pocket:french"]
    assert synthese_pocket.DEFAULT_LANGUE == "french_24l"


# ---- Ce qui est déjà sur le disque ----


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Un cache HuggingFace vide, vers lequel pointe huggingface_hub."""
    constants = types.SimpleNamespace(HF_HUB_CACHE=str(tmp_path))
    monkeypatch.setattr(sys.modules["huggingface_hub"], "constants", constants, raising=False)
    return tmp_path


def _poser(cache: Path, repo: str, revision: str, relatif: str) -> None:
    fichier = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / relatif
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_bytes(b"x")


def test_nothing_cached_means_a_download_is_needed(cache: Path):
    assert est_en_cache("french_24l") is False


def test_cached_when_weights_tokenizer_and_voice_are_all_there(cache: Path):
    """Les trois fichiers peuvent être dans des révisions différentes du
    dépôt : c'est ainsi que Pocket TTS les télécharge."""
    repo = synthese_pocket.REPO_LIBRE
    _poser(cache, repo, "rev1", "languages/french_24l/model.safetensors")
    assert est_en_cache("french_24l") is False
    _poser(cache, repo, "rev2", "languages/french_24l/tokenizer.json")
    assert est_en_cache("french_24l") is False
    _poser(cache, repo, "rev3", "languages/french_24l/embeddings/estelle.safetensors")
    assert est_en_cache("french_24l") is True
    # L'autre taille du modèle n'en profite pas.
    assert est_en_cache("french") is False


def test_weights_from_the_gated_repo_count_too(cache: Path):
    _poser(cache, "kyutai/pocket-tts", "r", "languages/french/model.safetensors")
    _poser(cache, synthese_pocket.REPO_LIBRE, "r", "languages/french/tokenizer.json")
    _poser(cache, synthese_pocket.REPO_LIBRE, "r", "languages/french/embeddings/estelle.safetensors")
    assert est_en_cache("french") is True


def test_a_dangling_link_is_not_a_cached_file(cache: Path):
    """Téléchargement interrompu : le lien existe, pas le fichier qu'il vise."""
    repo = synthese_pocket.REPO_LIBRE
    _poser(cache, repo, "r", "languages/french/tokenizer.json")
    _poser(cache, repo, "r", "languages/french/embeddings/estelle.safetensors")
    lien = (cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
            / "languages/french/model.safetensors")
    lien.symlink_to(cache / "blob-absent")
    assert est_en_cache("french") is False


def test_cache_check_never_raises_without_huggingface(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "constants",
        types.SimpleNamespace(HF_HUB_CACHE=None), raising=False,
    )
    assert est_en_cache("french_24l") is False


# ---- Le moteur ----


class FakeChunk:
    """Ce que rend pocket_tts : un tenseur torch d'une tranche de 80 ms."""

    def __init__(self, value: float) -> None:
        self._data = np.full(1920, value, dtype=np.float32)

    def detach(self) -> "FakeChunk":
        return self

    def cpu(self) -> "FakeChunk":
        return self

    def numpy(self) -> np.ndarray:
        return self._data


@pytest.fixture
def pocket(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Un faux paquet pocket_tts, et la bibliothèque déclarée présente."""
    state: dict = {"loads": [], "offline_at_load": [], "generated": [], "cached": True}

    class FakeModel:
        sample_rate = 24_000

        def get_state_for_audio_prompt(self, voice: str) -> str:
            state["voice"] = voice
            return f"état<{voice}>"

        def generate_audio_stream(self, voice_state, text, stop=None):  # noqa: ANN001
            state["generated"].append((voice_state, text, stop))
            for i in range(3):
                if stop is not None and stop.is_set():
                    return
                yield FakeChunk(float(i))

    class FakeTTSModel:
        @staticmethod
        def load_model(language: str) -> FakeModel:
            state["loads"].append(language)
            state["offline_at_load"].append(synthese_pocket.hf_offline.is_offline())
            return FakeModel()

    module = types.ModuleType("pocket_tts")
    module.TTSModel = FakeTTSModel
    monkeypatch.setitem(sys.modules, "pocket_tts", module)
    monkeypatch.setattr(synthese_pocket, "bibliotheque_presente", lambda: True)
    monkeypatch.setattr(
        synthese_pocket, "est_en_cache", lambda langue, voix="estelle": state["cached"]
    )
    return state


def test_sound_comes_out_in_chunks(pocket: dict):
    moteur = PocketSynthetiseur("french_24l")
    morceaux = list(moteur.flux("Bonjour."))
    assert len(morceaux) == 3
    assert all(m.dtype == np.float32 and m.shape == (1920,) for m in morceaux)
    assert pocket["loads"] == ["french_24l"]
    assert pocket["voice"] == "estelle"
    assert pocket["generated"][0][:2] == ("état<estelle>", "Bonjour.")


def test_the_model_is_loaded_once(pocket: dict):
    moteur = PocketSynthetiseur()
    assert moteur.is_loaded is False
    list(moteur.flux("Un."))
    list(moteur.flux("Deux."))
    assert pocket["loads"] == ["french_24l"]
    assert moteur.is_loaded is True


def test_stop_event_reaches_the_model(pocket: dict):
    """Arrêter la lecture doit aussi arrêter la génération, pas seulement le son."""
    arret = threading.Event()
    moteur = PocketSynthetiseur()
    flux = moteur.flux("Une longue phrase.", arret)
    next(flux)
    arret.set()
    assert list(flux) == []
    assert pocket["generated"][0][2] is arret


def test_empty_text_generates_nothing(pocket: dict):
    assert list(PocketSynthetiseur().flux("   ")) == []
    assert pocket["loads"] == []


def test_network_is_opened_only_for_the_first_download(pocket: dict):
    """Le reste de l'app tourne hors-ligne forcé : en cache, on n'y touche
    pas ; au premier usage, on rouvre le réseau le temps du chargement."""
    hf = synthese_pocket.hf_offline
    hf.set_offline(True)
    try:
        PocketSynthetiseur("french").preload()
        assert pocket["offline_at_load"] == [True]

        pocket["cached"] = False
        PocketSynthetiseur("french_24l").preload()
        assert pocket["offline_at_load"] == [True, False]
        assert hf.is_offline() is True
    finally:
        hf.set_offline(False)


def test_missing_library_gives_a_message_for_the_user(monkeypatch: pytest.MonkeyPatch):
    """Installation d'avant ce moteur : on dit quoi faire."""
    monkeypatch.setattr(synthese_pocket, "bibliotheque_presente", lambda: False)
    with pytest.raises(PocketError, match="Relance l'installation"):
        PocketSynthetiseur().preload()


def test_lire_concatenates_the_stream(pocket: dict):
    audio = PocketSynthetiseur().lire("Bonjour.")
    assert audio.shape == (3 * 1920,)
