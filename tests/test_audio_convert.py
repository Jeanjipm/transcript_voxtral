"""Tests d'audio_convert.py — normalisation via afconvert + lecture par blocs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

import audio_convert
from audio_convert import (
    AudioConversionError,
    AudioInfo,
    ensure_16k_mono,
    probe,
    read_block,
)


def _write_wav(
    path: Path, seconds: float, rate: int = 16_000, channels: int = 1
) -> Path:
    frames = int(seconds * rate)
    data = np.zeros((frames, channels), dtype="int16")
    sf.write(path, data, rate, subtype="PCM_16")
    return path


@pytest.fixture
def ok_afconvert(monkeypatch: pytest.MonkeyPatch):
    """Simule un afconvert qui réussit en produisant un WAV 16 kHz mono."""
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN003
        calls.append(cmd)
        _write_wav(Path(cmd[-1]), 2.0)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("audio_convert.subprocess.run", fake_run)
    return calls


# ---- probe ----


def test_probe_reads_wav_metadata(tmp_path: Path):
    path = _write_wav(tmp_path / "a.wav", 3.0, rate=44_100, channels=2)
    info = probe(path)
    assert info is not None
    assert info.sample_rate == 44_100
    assert info.channels == 2
    assert info.duration_s == pytest.approx(3.0, abs=0.01)


def test_probe_returns_none_for_unreadable_format(tmp_path: Path):
    """Cas normal du .m4a : libsndfile refuse. None ne veut pas dire invalide,
    afconvert saura le lire."""
    fake = tmp_path / "voix.m4a"
    fake.write_bytes(b"pas un conteneur audio valide")
    assert probe(fake) is None


def test_needs_conversion_flags():
    assert AudioInfo(Path("x"), 1.0, 16_000, 1).needs_conversion is False
    assert AudioInfo(Path("x"), 1.0, 44_100, 1).needs_conversion is True
    assert AudioInfo(Path("x"), 1.0, 16_000, 2).needs_conversion is True


# ---- ensure_16k_mono ----


def test_already_16k_mono_is_returned_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Pas de conversion inutile : on ne recopie pas un fichier déjà conforme."""
    path = _write_wav(tmp_path / "ok.wav", 2.0)
    fake_run = MagicMock(name="run")
    monkeypatch.setattr("audio_convert.subprocess.run", fake_run)

    out, is_temp = ensure_16k_mono(path)

    assert out == path
    assert is_temp is False
    fake_run.assert_not_called()


def test_resampling_invokes_afconvert_with_exact_flags(
    tmp_path: Path, ok_afconvert
):
    src = _write_wav(tmp_path / "hi.wav", 2.0, rate=44_100, channels=2)

    out, is_temp = ensure_16k_mono(src)

    assert is_temp is True
    cmd = ok_afconvert[0]
    assert cmd[0] == audio_convert.AFCONVERT
    assert "-f" in cmd and "WAVE" in cmd
    assert "LEI16@16000" in cmd
    assert cmd[cmd.index("-c") + 1] == "1"
    out.unlink()


def test_unreadable_format_is_converted(tmp_path: Path, ok_afconvert):
    """Le cas .m4a : probe échoue, on convertit quand même."""
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"conteneur que libsndfile ignore")

    out, is_temp = ensure_16k_mono(src)

    assert is_temp is True
    assert len(ok_afconvert) == 1
    out.unlink()


def test_missing_source_raises_french_error(tmp_path: Path):
    with pytest.raises(AudioConversionError, match="introuvable"):
        ensure_16k_mono(tmp_path / "pas_la.m4a")


def test_afconvert_failure_includes_its_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Une conversion qui échoue en silence serait le bug le plus difficile à
    diagnostiquer : le détail d'afconvert doit remonter dans le message."""
    src = tmp_path / "casse.m4a"
    src.write_bytes(b"xxx")
    monkeypatch.setattr(
        "audio_convert.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="Error: Couldn't open input file"
        ),
    )

    with pytest.raises(AudioConversionError) as excinfo:
        ensure_16k_mono(src)

    assert "Couldn't open input file" in str(excinfo.value)
    assert "non reconnu" in str(excinfo.value)


def test_afconvert_missing_binary_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src = tmp_path / "x.m4a"
    src.write_bytes(b"xxx")
    monkeypatch.setattr(
        "audio_convert.subprocess.run",
        MagicMock(side_effect=FileNotFoundError("no afconvert")),
    )

    with pytest.raises(AudioConversionError, match="introuvable"):
        ensure_16k_mono(src)


def test_afconvert_timeout_raises_french_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src = tmp_path / "x.m4a"
    src.write_bytes(b"xxx")
    monkeypatch.setattr(
        "audio_convert.subprocess.run",
        MagicMock(side_effect=subprocess.TimeoutExpired("afconvert", 1)),
    )

    with pytest.raises(AudioConversionError, match="dépassé"):
        ensure_16k_mono(src)


def test_empty_output_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """afconvert peut rendre 0 sur un fichier sans piste audio (une vidéo
    muette) : le WAV produit est vide et doit être signalé."""
    src = tmp_path / "muet.mov"
    src.write_bytes(b"xxx")

    def fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN003
        Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("audio_convert.subprocess.run", fake_run)

    with pytest.raises(AudioConversionError, match="aucune donnée audio"):
        ensure_16k_mono(src)


def test_temp_file_is_cleaned_up_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Un échec ne doit pas laisser traîner de WAV temporaire."""
    src = tmp_path / "x.m4a"
    src.write_bytes(b"xxx")
    created: list[Path] = []

    def fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN003
        created.append(Path(cmd[-1]))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr("audio_convert.subprocess.run", fake_run)

    with pytest.raises(AudioConversionError):
        ensure_16k_mono(src)

    assert created and not created[0].exists()


# ---- read_block ----


def test_read_block_returns_requested_frames(tmp_path: Path):
    path = _write_wav(tmp_path / "long.wav", 10.0)
    block = read_block(path, 2.0, 3.0)
    assert block.shape == (3 * 16_000,)
    assert block.dtype == np.float32


def test_read_block_is_one_dimensional(tmp_path: Path):
    """Whisper et Voxtral attendent un signal à une dimension."""
    path = _write_wav(tmp_path / "s.wav", 4.0)
    assert read_block(path, 0.0, 2.0).ndim == 1


def test_read_block_past_end_returns_empty(tmp_path: Path):
    """Sert de condition d'arrêt à la boucle de transcription."""
    path = _write_wav(tmp_path / "s.wav", 5.0)
    assert read_block(path, 99.0, 3.0).size == 0


def test_read_block_truncates_at_end_of_file(tmp_path: Path):
    path = _write_wav(tmp_path / "s.wav", 5.0)
    block = read_block(path, 4.0, 30.0)
    assert block.shape[0] == pytest.approx(1 * 16_000, abs=10)


def test_read_block_downmixes_stereo(tmp_path: Path):
    path = _write_wav(tmp_path / "st.wav", 3.0, channels=2)
    assert read_block(path, 0.0, 1.0).ndim == 1
