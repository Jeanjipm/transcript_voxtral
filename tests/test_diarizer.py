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


def _fake_model(probs) -> MagicMock:
    model = MagicMock()
    model.generate.return_value = MagicMock(speaker_probs=probs)
    # Force le repli de _frame_duration : un MagicMock rendrait un mock
    # là où on attend un nombre.
    model.config.processor_config.hop_length = 160
    model.config.processor_config.sampling_rate = 16_000
    model.config.fc_encoder_config.subsampling_factor = 8
    return model


def test_diarize_builds_turns_from_probabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """`diarize` repart des probabilités brutes, pas des segments tout faits
    de mlx-audio : ceux-ci sortent d'un seuillage trame à trame sans lissage
    et donnaient des rafales de tours de 80 ms."""
    # 50 trames de 80 ms : locuteur 0 pendant 2 s, puis locuteur 1 pendant 2 s.
    probs = [[0.9, 0.0] for _ in range(25)] + [[0.0, 0.9] for _ in range(25)]
    fake_vad = MagicMock()
    fake_vad.load.return_value = _fake_model(probs)
    monkeypatch.setitem(sys.modules, "mlx_audio", MagicMock(vad=fake_vad))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", fake_vad)
    monkeypatch.setattr(diarizer, "is_available", lambda: True)
    monkeypatch.setattr(diarizer, "_audio_duration", lambda _p: 4.0)

    turns = diarizer.diarize(tmp_path / "a.wav")

    assert [(t.speaker, round(t.start, 2), round(t.end, 2)) for t in turns] == [
        (0, 0.0, 2.0),
        (1, 2.0, 4.0),
    ]


def test_diarize_downloads_when_the_model_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Le mode hors-ligne est activé au démarrage pour que la dictée ne
    dépende jamais du réseau. Sans la parenthèse `allow_network`, le tout
    premier usage de la diarisation échouerait sur une erreur hors sujet."""
    import hf_offline

    fake_vad = MagicMock()
    fake_vad.load.return_value = _fake_model([[0.9, 0.0]] * 20)
    monkeypatch.setitem(sys.modules, "mlx_audio", MagicMock(vad=fake_vad))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", fake_vad)
    monkeypatch.setattr(diarizer, "is_available", lambda: True)
    monkeypatch.setattr(diarizer, "_audio_duration", lambda _p: 1.6)
    monkeypatch.setattr(hf_offline, "is_model_cached", lambda _r: False)

    ouvert: list[bool] = []
    monkeypatch.setattr(
        hf_offline, "allow_network", lambda: _Recording(ouvert)
    )

    diarizer.diarize(tmp_path / "a.wav")

    assert ouvert == [True], "le réseau doit être rouvert le temps du téléchargement"
    fake_vad.load.assert_called_once_with(diarizer.DEFAULT_MODEL)


class _Recording:
    """Contexte factice qui note son ouverture."""

    def __init__(self, journal: list[bool]) -> None:
        self._journal = journal

    def __enter__(self):  # noqa: ANN204
        self._journal.append(True)
        return self

    def __exit__(self, *_exc) -> bool:
        return False


# ---- turns_from_probs : la mise en forme des tours ----


def _probs(sequence: list[int | None], speakers: int = 4) -> list[list[float]]:
    """Construit une matrice de probabilités depuis une suite d'étiquettes."""
    rows = []
    for label in sequence:
        row = [0.02] * speakers
        if label is not None:
            row[label] = 0.9
        rows.append(row)
    return rows


def test_turns_follow_the_dominant_channel():
    turns = diarizer.turns_from_probs(_probs([0] * 20 + [1] * 20), frame_s=0.1)
    assert [(t.speaker, round(t.start, 1), round(t.end, 1)) for t in turns] == [
        (0, 0.0, 2.0),
        (1, 2.0, 4.0),
    ]


def test_silence_is_not_attributed():
    """Sous le seuil, aucun canal ne gagne : le silence reste du silence."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 20 + [None] * 20 + [0] * 20), frame_s=0.1, max_gap_s=0.5
    )
    assert len(turns) == 2, "un silence de 2 s sépare bien deux tours"


def test_short_gap_is_closed():
    """Une respiration au milieu d'une phrase ne fait pas deux tours."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 20 + [None] * 3 + [0] * 20), frame_s=0.1, max_gap_s=0.5
    )
    assert len(turns) == 1


def test_isolated_frame_is_smoothed_away():
    """Le modèle décide trame par trame : une bascule isolée en plein milieu
    d'une phrase produirait un tour de 100 ms attribué à quelqu'un d'autre."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 20 + [1] + [0] * 20), frame_s=0.1
    )
    assert [t.speaker for t in turns] == [0]


def test_short_turn_is_dropped():
    """Artefact de transition mesuré : le modèle allume brièvement un canal
    tiers dans le silence entre deux phrases."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 20 + [2] * 4 + [1] * 20), frame_s=0.1, min_turn_s=0.5
    )
    assert [t.speaker for t in turns] == [0, 1]


def test_dropping_a_short_turn_lets_the_neighbours_merge():
    """L'ordre compte : filtrer AVANT de recoller. L'inverse figerait
    l'intrus au milieu de deux tours du même locuteur."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 20 + [2] * 2 + [0] * 20), frame_s=0.1, min_turn_s=0.5
    )
    assert len(turns) == 1
    assert turns[0].speaker == 0


def test_marginal_speaker_is_dropped():
    """Le modèle hésite parfois sur la première seconde avant de se fixer."""
    turns = diarizer.turns_from_probs(
        _probs([3] * 8 + [0] * 40 + [1] * 40),
        frame_s=0.1, min_speaker_s=1.5,
    )
    assert [t.speaker for t in turns] == [0, 1]


def test_all_speakers_marginal_keeps_everything():
    """Sur un très court extrait, tout est sous le seuil. Mieux vaut des
    étiquettes imparfaites qu'un transcript sans aucune étiquette."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 8 + [1] * 8), frame_s=0.1, min_speaker_s=5.0
    )
    assert len(turns) == 2


def test_speakers_are_renumbered_in_order_of_appearance():
    """Le modèle rend des numéros de canal quelconques — couramment 0 et 3
    pour un dialogue à deux. Sans renumérotation, le lecteur verrait
    « Locuteur 1 » puis « Locuteur 4 »."""
    turns = diarizer.turns_from_probs(
        _probs([3] * 20 + [0] * 20), frame_s=0.1
    )
    assert [t.speaker for t in turns] == [0, 1]


def test_turns_are_clipped_to_the_audio_duration():
    """Les caractéristiques sont complétées à un multiple de 16 trames : sans
    rognage, le dernier tour déborde de la fin du fichier."""
    turns = diarizer.turns_from_probs(
        _probs([0] * 40), frame_s=0.1, duration_s=3.5
    )
    assert turns[-1].end == 3.5


def test_empty_probs_gives_no_turns():
    assert diarizer.turns_from_probs([]) == []


def test_turns_from_probs_is_deterministic():
    """Deux exécutions doivent produire exactement le même transcript."""
    sequence = _probs([0] * 15 + [1] * 15 + [0] * 15)
    assert diarizer.turns_from_probs(sequence) == diarizer.turns_from_probs(
        sequence
    )
