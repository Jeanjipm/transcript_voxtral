"""Tests de transcribe.py — la CLI de debug.

Sprint 6c (P3-17) : la CLI refusait tout WAV qui n'était pas en 16 kHz. Elle
passe maintenant par `audio_convert.ensure_16k_mono`, comme la transcription
de fichiers de l'app, donc accepte m4a, mp3, aiff…
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import transcribe
from audio_convert import AudioConversionError
from config import Config


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> dict:
    cfg = Config()
    cfg.transcription.language = "fr"
    monkeypatch.setattr(transcribe, "load_config", lambda: cfg)
    transcriber = MagicMock(name="transcriber")
    transcriber.transcribe.return_value = "bonjour"
    monkeypatch.setattr(transcribe, "make_transcriber", lambda c: transcriber)
    convert = MagicMock(name="ensure_16k_mono")
    monkeypatch.setattr(transcribe, "ensure_16k_mono", convert)
    return {"cfg": cfg, "transcriber": transcriber, "convert": convert}


def test_converts_then_transcribes_and_removes_the_temporary(
    cli: dict, tmp_path: Path, capsys
):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"...")
    tmp = tmp_path / "voxtral_conv_x.wav"
    tmp.write_bytes(b"RIFF")
    cli["convert"].return_value = (tmp, True)

    code = transcribe.main([str(src), "--task", "translate"])

    assert code == 0
    cli["convert"].assert_called_once_with(src)
    cli["transcriber"].transcribe.assert_called_once_with(
        tmp, language="fr", task="translate", max_new_tokens=1024
    )
    assert not tmp.exists(), "le WAV temporaire de conversion est nettoyé"
    assert capsys.readouterr().out.strip() == "bonjour"


def test_keeps_a_file_that_needed_no_conversion(cli: dict, tmp_path: Path):
    src = tmp_path / "deja16k.wav"
    src.write_bytes(b"RIFF")
    cli["convert"].return_value = (src, False)

    assert transcribe.main([str(src), "--lang", "en"]) == 0
    assert src.exists()
    cli["transcriber"].transcribe.assert_called_once_with(
        src, language="en", task="transcribe", max_new_tokens=1024
    )


def test_unreadable_file_is_an_error_without_loading_the_model(
    cli: dict, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
):
    cli["convert"].side_effect = AudioConversionError("Fichier introuvable : x")
    loaded = MagicMock(name="make_transcriber")
    monkeypatch.setattr(transcribe, "make_transcriber", loaded)

    assert transcribe.main([str(tmp_path / "x")]) == 1
    assert "Fichier introuvable" in capsys.readouterr().err
    loaded.assert_not_called()
