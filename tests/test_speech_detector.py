"""Tests de speech_detector.py — Silero VAD derrière une façade prudente.

Le modèle MLX n'existe pas dans l'environnement de test : on en simule un. Ce
qu'on vérifie ici, c'est la prudence de la façade — elle ne doit jamais
bloquer ni faire perdre une dictée — et les réglages qu'elle passe à Silero.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

import speech_detector
from speech_detector import MIN_SPEECH_MS, SpeechDetector


@pytest.fixture
def cached(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Poids « sur le disque », et un faux `mlx_audio.vad.load`."""
    state: dict = {"cached": True, "stamps": [{"start": 0.3, "end": 0.9}], "calls": []}
    monkeypatch.setattr(
        speech_detector.hf_offline, "is_model_cached", lambda repo: state["cached"]
    )
    monkeypatch.setattr(
        speech_detector.hf_offline, "resolve_local_path", lambda repo: "/cache/silero"
    )

    class FakeModel:
        def get_speech_timestamps(self, audio, **kwargs):  # noqa: ANN001, ANN003
            state["calls"].append((audio, kwargs))
            if state.get("boom"):
                raise RuntimeError("modèle cassé")
            return state["stamps"]

    def load(path):  # noqa: ANN001, ANN202
        state.setdefault("loaded", []).append(path)
        return FakeModel()

    fake_vad = types.ModuleType("mlx_audio.vad")
    fake_vad.load = load
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", fake_vad)
    return state


def _wav(tmp_path: Path, seconds: float = 1.0, channels: int = 1) -> Path:
    path = tmp_path / "d.wav"
    audio = np.zeros((int(16_000 * seconds), channels), dtype="int16")
    sf.write(path, audio, 16_000, subtype="PCM_16")
    return path


def test_speech_is_reported(cached: dict, tmp_path: Path):
    assert SpeechDetector().has_speech(_wav(tmp_path)) is True


def test_no_speech_is_reported(cached: dict, tmp_path: Path):
    cached["stamps"] = []
    assert SpeechDetector().has_speech(_wav(tmp_path)) is False


def test_short_words_are_not_missed(cached: dict, tmp_path: Path):
    """Le défaut de Silero (250 ms) laissait passer un « Oui. » dit seul pour
    du silence : on lui demande de reconnaître dès 100 ms."""
    SpeechDetector().has_speech(_wav(tmp_path))
    audio, kwargs = cached["calls"][0]
    assert kwargs["min_speech_duration_ms"] == MIN_SPEECH_MS == 100
    assert kwargs["sample_rate"] == 16_000
    assert kwargs["return_seconds"] is True
    assert audio.dtype == np.float32 and audio.ndim == 1


def test_stereo_is_mixed_down(cached: dict, tmp_path: Path):
    SpeechDetector().has_speech(_wav(tmp_path, channels=2))
    assert cached["calls"][0][0].ndim == 1


def test_model_is_loaded_once_from_the_local_folder(cached: dict, tmp_path: Path):
    """Le dossier local, pas l'identifiant du dépôt : aucun appel réseau."""
    detector = SpeechDetector()
    detector.has_speech(_wav(tmp_path))
    detector.has_speech(_wav(tmp_path))
    assert cached["loaded"] == ["/cache/silero"]


def test_missing_weights_mean_i_dont_know(cached: dict, tmp_path: Path):
    """Pas encore téléchargé : None, et l'appelant retombe sur le seuil de
    volume. Surtout, aucun chargement n'est tenté."""
    cached["cached"] = False
    assert SpeechDetector().has_speech(_wav(tmp_path)) is None
    assert "loaded" not in cached


def test_a_failing_model_means_i_dont_know(cached: dict, tmp_path: Path, capsys):
    cached["boom"] = True
    assert SpeechDetector().has_speech(_wav(tmp_path)) is None


def test_an_unreadable_file_means_i_dont_know(cached: dict, tmp_path: Path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"pas un wav")
    assert SpeechDetector().has_speech(bad) is None


def test_an_empty_recording_has_no_speech(cached: dict, tmp_path: Path):
    assert SpeechDetector().has_speech(_wav(tmp_path, seconds=0)) is False
    assert cached["calls"] == []


# ---- Téléchargement en arrière-plan ----


def test_download_is_skipped_when_weights_are_there(cached: dict, monkeypatch):
    download = MagicMock()
    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "snapshot_download", download, raising=False
    )
    assert SpeechDetector().ensure_downloaded() is True
    download.assert_not_called()


def test_download_opens_the_network_then_reports(cached: dict, monkeypatch):
    cached["cached"] = False
    seen: dict = {}

    def download(repo_id):  # noqa: ANN001, ANN202
        seen["repo"] = repo_id
        seen["offline"] = speech_detector.hf_offline.is_offline()
        cached["cached"] = True

    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "snapshot_download", download, raising=False
    )
    speech_detector.hf_offline.set_offline(True)
    try:
        assert SpeechDetector().ensure_downloaded() is True
        assert seen == {"repo": speech_detector.REPO, "offline": False}
        assert speech_detector.hf_offline.is_offline() is True
    finally:
        speech_detector.hf_offline.set_offline(False)


def test_failed_download_is_not_an_error(cached: dict, monkeypatch, capsys):
    """Hors-ligne au premier lancement : on continue avec le seuil de volume."""
    cached["cached"] = False
    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "snapshot_download",
        MagicMock(side_effect=OSError("pas de réseau")), raising=False,
    )
    assert SpeechDetector().ensure_downloaded() is False
    assert "téléchargement impossible" in capsys.readouterr().out
