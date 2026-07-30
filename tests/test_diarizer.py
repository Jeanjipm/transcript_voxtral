"""Tests de diarizer.py — fusion locuteurs × segments, et dégradation propre.

`assign_speakers` est une fonction pure : c'est là que se joue la qualité du
résultat final, donc elle est testée sans modèle.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import diarizer
from diarizer import (
    DiarizationUnavailable,
    SpeakerTurn,
    assign_speakers,
    speaker_label,
)
from transcriber import Segment


def _seg(start: float, end: float, text: str = "texte") -> Segment:
    return Segment(start=start, end=end, text=text)


# ---- assign_speakers ----


def test_segment_gets_the_most_overlapping_speaker():
    segments = [_seg(0.0, 5.0), _seg(5.0, 10.0)]
    turns = [
        SpeakerTurn(0.0, 5.0, 0),
        SpeakerTurn(5.0, 10.0, 1),
    ]

    result = assign_speakers(segments, turns)

    assert result[0].speaker == 0
    assert result[1].speaker == 1


def test_partial_overlap_picks_the_majority_speaker():
    """Un segment de texte à cheval sur deux tours va au locuteur dominant."""
    segments = [_seg(0.0, 10.0)]
    turns = [
        SpeakerTurn(0.0, 3.0, 0),
        SpeakerTurn(3.0, 10.0, 1),  # 7 s contre 3 s
    ]

    assert assign_speakers(segments, turns)[0].speaker == 1


def test_overlaps_are_summed_per_speaker():
    """Un locuteur qui intervient plusieurs fois cumule son temps."""
    segments = [_seg(0.0, 10.0)]
    turns = [
        SpeakerTurn(0.0, 2.0, 0),
        SpeakerTurn(2.0, 4.0, 1),
        SpeakerTurn(4.0, 6.0, 0),
        SpeakerTurn(6.0, 8.0, 0),  # locuteur 0 : 6 s au total contre 2 s
    ]

    assert assign_speakers(segments, turns)[0].speaker == 0


def test_no_overlap_leaves_speaker_none():
    """Mieux vaut ne rien affirmer que d'attribuer une phrase à la mauvaise
    personne."""
    segments = [_seg(100.0, 105.0)]
    turns = [SpeakerTurn(0.0, 10.0, 0)]

    assert assign_speakers(segments, turns)[0].speaker is None


def test_negligible_overlap_leaves_speaker_none():
    """Un contact de quelques millisecondes est du bruit de frontière."""
    segments = [_seg(9.99, 15.0)]
    turns = [SpeakerTurn(0.0, 10.0, 0)]

    assert assign_speakers(segments, turns)[0].speaker is None


def test_empty_turns_returns_segments_unchanged():
    """Diarisation sans résultat : le texte doit rester intact."""
    segments = [_seg(0.0, 5.0, "bonjour"), _seg(5.0, 9.0, "salut")]

    result = assign_speakers(segments, [])

    assert [s.text for s in result] == ["bonjour", "salut"]
    assert all(s.speaker is None for s in result)


def test_text_and_timestamps_are_preserved():
    segments = [_seg(1.5, 4.25, "phrase exacte")]
    turns = [SpeakerTurn(0.0, 10.0, 2)]

    result = assign_speakers(segments, turns)[0]

    assert result.text == "phrase exacte"
    assert result.start == 1.5
    assert result.end == 4.25
    assert result.speaker == 2


def test_empty_segments_gives_empty_result():
    assert assign_speakers([], [SpeakerTurn(0.0, 5.0, 0)]) == []


# ---- speaker_label ----


def test_speaker_label_is_one_based():
    """« Locuteur 0 » n'a aucun sens pour un lecteur non-technique."""
    assert speaker_label(0) == "Locuteur 1"
    assert speaker_label(1) == "Locuteur 2"
    assert speaker_label(3) == "Locuteur 4"


def test_speaker_label_none_is_empty():
    assert speaker_label(None) == ""


# ---- Disponibilité ----


def test_is_available_false_without_the_package(monkeypatch: pytest.MonkeyPatch):
    """Sans mlx_audio, tout le module se désactive proprement.

    Poser None dans sys.modules fait lever ImportError à l'import — c'est le
    mécanisme standard de CPython pour marquer un module absent.
    """
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", None)

    assert diarizer.is_available() is False


def test_is_available_true_with_the_package(monkeypatch: pytest.MonkeyPatch):
    fake_vad = MagicMock(name="vad")
    monkeypatch.setitem(sys.modules, "mlx_audio", MagicMock(vad=fake_vad))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", fake_vad)

    assert diarizer.is_available() is True


def test_diarize_raises_french_error_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Le message doit dire quoi faire, pas seulement que ça ne marche pas."""
    monkeypatch.setattr(diarizer, "is_available", lambda: False)

    with pytest.raises(DiarizationUnavailable) as excinfo:
        diarizer.diarize(tmp_path / "a.wav")

    message = str(excinfo.value)
    assert "mlx-audio" in message
    assert "pip install" in message


def test_diarize_maps_model_segments(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Les segments du modèle deviennent des SpeakerTurn triés, en filtrant les
    intervalles vides."""
    fake_seg = lambda s, e, spk: MagicMock(start=s, end=e, speaker=spk)  # noqa: E731
    fake_model = MagicMock()
    fake_model.generate.return_value = MagicMock(
        segments=[
            fake_seg(5.0, 9.0, 1),
            fake_seg(0.0, 5.0, 0),
            fake_seg(9.0, 9.0, 0),  # vide, doit être ignoré
        ]
    )
    fake_vad = MagicMock()
    fake_vad.load.return_value = fake_model
    monkeypatch.setitem(sys.modules, "mlx_audio", MagicMock(vad=fake_vad))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", fake_vad)
    monkeypatch.setattr(diarizer, "is_available", lambda: True)

    turns = diarizer.diarize(tmp_path / "a.wav")

    assert [(t.start, t.speaker) for t in turns] == [(0.0, 0), (5.0, 1)]
