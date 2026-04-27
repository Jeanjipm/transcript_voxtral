"""Tests de transcriber.py — factory + delegation translate + preload."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import transcriber
from transcriber import (
    Transcriber,
    VoxtralTranscriber,
    WhisperTranscriber,
    WHISPER_REPO,
    make_transcriber,
)
from config import Config, ModelConfig


# ---- make_transcriber : routing ----


def test_make_transcriber_routes_whisper_repo_to_whisper():
    """Si le nom contient 'whisper', on prend WhisperTranscriber."""
    cfg = Config(model=ModelConfig(name="mlx-community/whisper-large-v3-mlx"))
    t = make_transcriber(cfg)
    assert isinstance(t, WhisperTranscriber)
    assert t.model_repo == "mlx-community/whisper-large-v3-mlx"


def test_make_transcriber_routes_voxtral_to_voxtral_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    """Modèle Voxtral + lib mlx_voxtral importable → VoxtralTranscriber."""
    cfg = Config(model=ModelConfig(name="mzbac/voxtral-mini-3b-4bit-mixed"))
    # is_available est appelé via import mlx_voxtral. Le mock de conftest
    # rend cet import OK.
    t = make_transcriber(cfg)
    assert isinstance(t, VoxtralTranscriber)
    assert t.model_repo == "mzbac/voxtral-mini-3b-4bit-mixed"


def test_make_transcriber_falls_back_to_whisper_when_voxtral_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
):
    """mlx_voxtral introuvable → fallback Whisper + warning sur stderr."""
    cfg = Config(model=ModelConfig(name="mzbac/voxtral-mini-3b-4bit-mixed"))

    # On simule un VoxtralTranscriber.is_available() qui retourne False.
    monkeypatch.setattr(VoxtralTranscriber, "is_available", lambda self: False)

    t = make_transcriber(cfg)
    assert isinstance(t, WhisperTranscriber)
    assert t.model_repo == WHISPER_REPO

    # Vérifie qu'un warning explicite est loggé sur stderr.
    captured = capsys.readouterr()
    assert "AVERTISSEMENT" in captured.err
    assert "mlx_voxtral" in captured.err


# ---- VoxtralTranscriber : preload + lazy load ----


def test_voxtral_preload_calls_ensure_loaded(monkeypatch: pytest.MonkeyPatch):
    """preload() doit déclencher le chargement du modèle."""
    t = VoxtralTranscriber("any/model")
    called = {"count": 0}

    def fake_ensure():
        called["count"] += 1
        t._model = MagicMock()
        t._processor = MagicMock()

    monkeypatch.setattr(t, "_ensure_loaded", fake_ensure)
    t.preload()
    assert called["count"] == 1


def test_voxtral_ensure_loaded_idempotent():
    """2 appels ne re-chargent pas le modèle (cache _model)."""
    t = VoxtralTranscriber("any/model")
    t._model = MagicMock()  # Simule un modèle déjà chargé
    t._processor = MagicMock()

    # On patch from_pretrained pour vérifier qu'il n'est PAS rappelé.
    fake_model_class = MagicMock()
    fake_processor_class = MagicMock()
    sys.modules["mlx_voxtral"].VoxtralForConditionalGeneration = (
        fake_model_class
    )
    sys.modules["mlx_voxtral"].VoxtralProcessor = fake_processor_class

    t._ensure_loaded()  # ne devrait rien faire
    fake_model_class.from_pretrained.assert_not_called()
    fake_processor_class.from_pretrained.assert_not_called()


# ---- VoxtralTranscriber : translate délègue à Whisper ----


def test_voxtral_translate_creates_whisper_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """task='translate' → instancie un WhisperTranscriber et délègue."""
    t = VoxtralTranscriber("voxtral/model")

    # Mock du WhisperTranscriber.transcribe pour intercepter l'appel.
    fake_translate = MagicMock(return_value="texte traduit")
    monkeypatch.setattr(WhisperTranscriber, "transcribe", fake_translate)

    wav = tmp_path / "test.wav"
    wav.touch()

    result = t.transcribe(wav, language="fr", task="translate")
    assert result == "texte traduit"

    # Vérifie qu'un Whisper a bien été instancié et utilisé.
    assert t._whisper_for_translate is not None
    assert isinstance(t._whisper_for_translate, WhisperTranscriber)
    assert t._whisper_for_translate.model_repo == WHISPER_REPO

    # Vérifie l'appel a bien la bonne task.
    args, kwargs = fake_translate.call_args
    assert kwargs.get("task") == "translate"


def test_voxtral_translate_reuses_whisper_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """2e translate ne crée pas un nouveau Whisper, réutilise le 1er."""
    t = VoxtralTranscriber("voxtral/model")
    monkeypatch.setattr(
        WhisperTranscriber, "transcribe", lambda *a, **kw: "ok"
    )

    wav = tmp_path / "test.wav"
    wav.touch()

    t.transcribe(wav, task="translate")
    first_whisper = t._whisper_for_translate
    t.transcribe(wav, task="translate")
    assert t._whisper_for_translate is first_whisper


def test_voxtral_transcribe_does_not_create_whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """task='transcribe' (default) ne doit PAS instancier de Whisper."""
    t = VoxtralTranscriber("voxtral/model")
    t._model = MagicMock()
    t._processor = MagicMock()

    # Mock le pipeline Voxtral pour qu'il retourne un texte
    t._processor.apply_transcrition_request = MagicMock()
    inputs = MagicMock()
    inputs.input_ids.shape = (1, 5)
    t._processor.apply_transcrition_request.return_value = inputs
    t._model.generate = MagicMock(return_value=[[1, 2, 3, 4, 5, 6]])
    t._processor.decode = MagicMock(return_value=" hello ")

    wav = tmp_path / "test.wav"
    wav.touch()

    result = t.transcribe(wav, task="transcribe")
    assert result == "hello"  # strip
    assert t._whisper_for_translate is None


# ---- WhisperTranscriber ----


def test_whisper_preload_is_noop():
    """WhisperTranscriber.preload() est un no-op pour la v0 (mlx-whisper
    n'expose pas d'API publique de préchargement)."""
    t = WhisperTranscriber("any/model")
    # Ne doit pas lever, ne doit rien faire de visible.
    result = t.preload()
    assert result is None


def test_whisper_is_available_returns_true_when_lib_present():
    """mlx_whisper est mocké dans conftest → import réussit → True."""
    t = WhisperTranscriber("any/model")
    assert t.is_available() is True


def test_voxtral_is_available_returns_true_when_lib_present():
    t = VoxtralTranscriber("any/model")
    assert t.is_available() is True


# ---- Transcriber base class : preload no-op default ----


class _DummyTranscriber(Transcriber):
    """Sous-classe de test pour vérifier le comportement par défaut."""

    def transcribe(self, wav_path, language="auto", task="transcribe", max_new_tokens=1024):
        return ""

    def is_available(self):
        return True


def test_transcriber_preload_default_is_noop():
    """Sous-classe qui n'override pas preload → no-op silencieux."""
    t = _DummyTranscriber()
    assert t.preload() is None
