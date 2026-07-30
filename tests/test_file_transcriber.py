"""Tests de file_transcriber.py — boucle long-form, garde-fous, mise en forme.

Module pur : ces tests n'ont besoin d'aucun modèle ni d'aucun fichier audio,
seulement d'un faux Transcriber qui rend des segments scriptés.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

import file_transcriber
from file_transcriber import (
    FileTranscript,
    format_timestamp,
    format_transcript,
    transcribe_file,
)
from transcriber import Segment, TranscriptionResult


class _ScriptedTranscriber:
    """Transcriber factice : rend des résultats prédéfinis, un par appel.

    Enregistre les arguments reçus pour que les tests vérifient l'avance,
    l'amorce et la langue.
    """

    def __init__(self, results: list[TranscriptionResult]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    def transcribe_array(self, audio, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"frames": int(np.asarray(audio).shape[0]), **kwargs})
        if self._results:
            return self._results.pop(0)
        return TranscriptionResult(text="", segments=[], language=None)


@pytest.fixture
def fake_audio(monkeypatch: pytest.MonkeyPatch):
    """Remplace read_block : rend `duration_s` secondes de zéros, ou du vide
    au-delà de la durée totale simulée."""
    state = {"total": 120.0}

    def read_block(_path: Path, start_s: float, duration_s: float):
        remaining = state["total"] - start_s
        if remaining <= 0:
            return np.zeros(0, dtype="float32")
        seconds = min(float(duration_s), remaining)
        return np.zeros(int(seconds * 16_000), dtype="float32")

    monkeypatch.setattr("file_transcriber.audio_convert.read_block", read_block)
    return state


def _res(*spans, language: str | None = None) -> TranscriptionResult:
    """Construit un TranscriptionResult depuis des tuples (start, end, texte)."""
    segs = [Segment(start=a, end=b, text=t) for a, b, t in spans]
    return TranscriptionResult(
        text=" ".join(s.text for s in segs), segments=segs, language=language
    )


# ---- Avance : la règle centrale ----


def test_advance_follows_last_segment_end(fake_audio):
    """On repart de la fin du dernier segment, pas de la fin du bloc : c'est
    ce qui évite de couper une phrase en deux à la jointure."""
    fake_audio["total"] = 60.0
    t = _ScriptedTranscriber([
        _res((0.0, 25.0, "premier bloc")),
        _res((0.0, 20.0, "deuxieme bloc")),
        _res((0.0, 10.0, "troisieme")),
    ])

    transcribe_file(t, Path("x.wav"), 60.0, block_duration_s=30)

    # 1er appel à 0, 2e à 25 (pas 30), 3e à 45.
    assert t.calls[0]["frames"] == 30 * 16_000
    assert len(t.calls) >= 3


def test_segment_timestamps_are_offset_to_absolute_time(fake_audio):
    """Les horodatages rendus par le backend sont relatifs au bloc ; le
    résultat final doit être en temps absolu depuis le début du fichier."""
    fake_audio["total"] = 60.0
    t = _ScriptedTranscriber([
        _res((0.0, 28.0, "un")),
        _res((2.0, 25.0, "deux")),
        _res((0.0, 5.0, "trois")),
    ])

    result = transcribe_file(t, Path("x.wav"), 60.0, block_duration_s=30)

    assert result.segments[0].start == 0.0
    # 2e bloc démarré à 28 → segment relatif 2.0 devient 30.0
    assert result.segments[1].start == pytest.approx(30.0)


def test_no_segments_advances_a_full_block(fake_audio):
    """Bloc silencieux : aucun segment, donc aucune avance naturelle. Sans le
    plancher, la boucle tournerait indéfiniment."""
    fake_audio["total"] = 90.0
    t = _ScriptedTranscriber([
        TranscriptionResult(text="", segments=[], language=None),
        TranscriptionResult(text="", segments=[], language=None),
        TranscriptionResult(text="", segments=[], language=None),
    ])

    result = transcribe_file(t, Path("x.wav"), 90.0, block_duration_s=30)

    assert len(t.calls) == 3  # 90 s / 30 s, pas de boucle infinie
    assert result.segments == []


def test_tiny_advance_is_forced_to_a_full_block(fake_audio):
    """Une avance sous 1 s (segment parasite en début de bloc) doit être
    forcée, sinon progression quasi nulle et boucle sans fin."""
    fake_audio["total"] = 60.0
    t = _ScriptedTranscriber([
        _res((0.0, 0.3, "tic")),
        _res((0.0, 0.2, "tac")),
    ])

    transcribe_file(t, Path("x.wav"), 60.0, block_duration_s=30)

    assert len(t.calls) == 2  # 2 blocs de 30 s couvrent les 60 s


def test_advance_is_capped_at_the_audio_actually_provided(fake_audio):
    """Whisper remplit ses fenêtres de 30 s et peut annoncer une fin AU-DELÀ
    de l'audio fourni (constaté : end=29.98 pour 8 s d'audio). Sans plafond,
    on sauterait de l'audio jamais transcrit."""
    fake_audio["total"] = 20.0
    # Bloc de 20 s d'audio réel, mais le backend annonce une fin à 29.98.
    t = _ScriptedTranscriber([
        _res((0.0, 29.98, "bloc rembourre")),
        _res((0.0, 5.0, "fin")),
    ])

    transcribe_file(t, Path("x.wav"), 20.0, block_duration_s=30)

    # L'avance est plafonnée à 20 s (l'audio réel), donc on sort proprement.
    assert len(t.calls) == 1


def test_loop_stops_when_read_block_returns_empty(fake_audio):
    """Fin de fichier atteinte : read_block rend du vide, on s'arrête."""
    fake_audio["total"] = 10.0
    t = _ScriptedTranscriber([_res((0.0, 10.0, "tout"))])

    result = transcribe_file(t, Path("x.wav"), 60.0, block_duration_s=30)

    assert len(result.segments) == 1


# ---- Amorce et langue ----


def test_no_prompt_tail_by_default(fake_audio):
    """Régression mesurée : passer la queue du texte précédent comme amorce
    fait PERDRE du contenu avec Whisper (un bloc a sauté ~15 s de discours et
    rendu une phrase de conclusion à la place). Désactivé par défaut."""
    fake_audio["total"] = 60.0
    t = _ScriptedTranscriber([
        _res((0.0, 28.0, "Monsieur Dupont est arrivé")),
        _res((0.0, 25.0, "il a parlé")),
    ])

    transcribe_file(t, Path("x.wav"), 60.0, block_duration_s=30)

    assert all(call["initial_prompt"] is None for call in t.calls)


def test_prompt_tail_can_be_enabled_explicitly(fake_audio):
    """Le mécanisme reste disponible pour les backends qui sauraient s'en
    servir sans y perdre du contenu."""
    fake_audio["total"] = 60.0
    t = _ScriptedTranscriber([
        _res((0.0, 28.0, "Monsieur Dupont est arrivé")),
        _res((0.0, 25.0, "il a parlé")),
    ])

    transcribe_file(
        t, Path("x.wav"), 60.0, block_duration_s=30, use_prompt_tail=True
    )

    assert t.calls[0]["initial_prompt"] is None
    assert "Monsieur Dupont" in t.calls[1]["initial_prompt"]


def test_prompt_tail_is_truncated(fake_audio):
    fake_audio["total"] = 60.0
    long_text = "mot " * 300
    t = _ScriptedTranscriber([
        _res((0.0, 28.0, long_text)),
        _res((0.0, 25.0, "suite")),
    ])

    transcribe_file(
        t, Path("x.wav"), 60.0, block_duration_s=30, use_prompt_tail=True
    )

    prompt = t.calls[1]["initial_prompt"]
    assert len(prompt) <= file_transcriber._PROMPT_TAIL_CHARS


def test_language_is_pinned_from_the_first_block(fake_audio):
    """Détectée une fois, puis figée : évite le papillonnage de langue en
    cours de fichier et la passe de détection sur chaque bloc."""
    fake_audio["total"] = 90.0
    t = _ScriptedTranscriber([
        _res((0.0, 28.0, "bonjour"), language="fr"),
        _res((0.0, 28.0, "suite"), language="en"),  # ignorée
        _res((0.0, 28.0, "fin"), language="de"),  # ignorée
    ])

    result = transcribe_file(t, Path("x.wav"), 90.0, block_duration_s=30, language="auto")

    assert result.language == "fr"
    assert t.calls[1]["language"] == "fr"
    assert t.calls[2]["language"] == "fr"


def test_explicit_language_is_passed_from_the_start(fake_audio):
    fake_audio["total"] = 30.0
    t = _ScriptedTranscriber([_res((0.0, 28.0, "bonjour"))])

    transcribe_file(t, Path("x.wav"), 30.0, block_duration_s=30, language="fr")

    assert t.calls[0]["language"] == "fr"


# ---- Progression et annulation ----


def test_progress_is_reported_and_monotonic(fake_audio):
    fake_audio["total"] = 90.0
    t = _ScriptedTranscriber([
        _res((0.0, 28.0, "a")), _res((0.0, 28.0, "b")), _res((0.0, 28.0, "c")),
        _res((0.0, 28.0, "d")),
    ])
    seen: list[tuple[float, float]] = []

    transcribe_file(
        t, Path("x.wav"), 90.0, block_duration_s=30,
        on_progress=lambda cur, tot: seen.append((cur, tot)),
    )

    assert seen
    assert all(b >= a for (a, _), (b, _) in zip(seen, seen[1:]))
    assert seen[-1][0] <= seen[-1][1]  # jamais au-dessus du total


def test_cancel_returns_partial_marked_as_interrupted(fake_audio):
    """Annuler ne doit pas jeter le travail fait : jeter 90 % d'un job de
    55 minutes serait inacceptable."""
    fake_audio["total"] = 300.0
    cancel = threading.Event()

    class _CancelAfterTwo(_ScriptedTranscriber):
        def transcribe_array(self, audio, **kwargs):  # noqa: ANN001, ANN003
            result = super().transcribe_array(audio, **kwargs)
            if len(self.calls) >= 2:
                cancel.set()
            return result

    t = _CancelAfterTwo([
        _res((0.0, 28.0, "bloc un")), _res((0.0, 28.0, "bloc deux")),
    ])

    result = transcribe_file(
        t, Path("x.wav"), 300.0, block_duration_s=30, cancel=cancel
    )

    assert result.cancelled is True
    assert result.cancelled_at_s is not None
    assert len(result.segments) == 2  # le travail fait est conservé
    assert "bloc un" in result.text


def test_cancel_before_first_block_returns_empty_but_valid(fake_audio):
    cancel = threading.Event()
    cancel.set()
    t = _ScriptedTranscriber([])

    result = transcribe_file(
        t, Path("x.wav"), 120.0, block_duration_s=30, cancel=cancel
    )

    assert result.cancelled is True
    assert result.segments == []
    assert len(t.calls) == 0


def test_file_shorter_than_one_block(fake_audio):
    fake_audio["total"] = 7.0
    t = _ScriptedTranscriber([_res((0.0, 7.0, "court"))])

    result = transcribe_file(t, Path("x.wav"), 7.0, block_duration_s=30)

    assert len(t.calls) == 1
    assert result.text == "court"


# ---- format_timestamp ----


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "00:00:00"),
        (5.4, "00:00:05"),
        (61.0, "00:01:01"),
        (3600.0, "01:00:00"),
        (3661.0, "01:01:01"),
        (-3.0, "00:00:00"),  # jamais de négatif affiché
    ],
)
def test_format_timestamp(seconds: float, expected: str):
    assert format_timestamp(seconds) == expected


# ---- format_transcript ----


def _transcript(*spans, **kwargs) -> FileTranscript:
    return FileTranscript(
        segments=[Segment(start=a, end=b, text=t) for a, b, t in spans],
        duration_s=kwargs.pop("duration_s", 100.0),
        **kwargs,
    )


def test_format_includes_header_metadata():
    out = format_transcript(
        _transcript((0.0, 5.0, "Bonjour."), language="fr"),
        source_name="reunion.m4a",
        model_name="mlx-community/whisper-large-v3-mlx",
        generated_at="30/07/2026 à 14:22",
    )
    assert "Transcription — reunion.m4a" in out
    assert "Modèle : mlx-community/whisper-large-v3-mlx" in out
    assert "00:01:40" in out  # durée 100 s
    assert "30/07/2026 à 14:22" in out
    assert "Langue : fr" in out


def test_format_prefixes_paragraphs_with_timestamps():
    out = format_transcript(
        _transcript((65.0, 70.0, "Bonjour à tous.")),
        source_name="a.wav", model_name="m", generated_at="x",
    )
    assert "[00:01:05] Bonjour à tous." in out


def test_format_without_timestamps_omits_them():
    out = format_transcript(
        _transcript((65.0, 70.0, "Bonjour à tous.")),
        source_name="a.wav", model_name="m",
        include_timestamps=False, generated_at="x",
    )
    assert "Bonjour à tous." in out
    assert "[00:01:05]" not in out


def test_format_splits_paragraph_on_long_gap():
    """Un silence notable ouvre un paragraphe."""
    out = format_transcript(
        _transcript(
            (0.0, 5.0, "Premier passage."),
            (5.5, 8.0, "Encore le premier."),
            (30.0, 33.0, "Nouveau passage."),
        ),
        source_name="a.wav", model_name="m", generated_at="x",
    )
    assert "[00:00:00] Premier passage. Encore le premier." in out
    assert "[00:00:30] Nouveau passage." in out


def test_format_splits_paragraph_when_too_long():
    """Quelqu'un qui parle sans pause ne doit pas produire un pavé illisible."""
    spans = [(float(i), float(i) + 0.9, "phrase de remplissage") for i in range(40)]
    out = format_transcript(
        _transcript(*spans), source_name="a.wav", model_name="m", generated_at="x"
    )
    paragraphs = [line for line in out.splitlines() if line.startswith("[")]
    assert len(paragraphs) > 1


def test_format_marks_cancelled_transcript():
    """Un .txt partiel doit se signaler comme tel, sans ambiguïté."""
    out = format_transcript(
        _transcript((0.0, 5.0, "Début seulement."), cancelled_at_s=2952.0),
        source_name="a.wav", model_name="m", generated_at="x",
    )
    assert "[Transcription interrompue à 00:49:12]" in out


def test_format_empty_transcript_still_has_header():
    out = format_transcript(
        _transcript(), source_name="vide.wav", model_name="m", generated_at="x"
    )
    assert "Transcription — vide.wav" in out
    assert out.endswith("\n")


# ---- Bornage des horodatages et blocs résiduels ----


def test_segment_timestamps_are_clamped_to_the_block(fake_audio):
    """Régression mesurée : Whisper a annoncé un segment finissant à 51,6 s
    dans un bloc de 30 s. Non borné, le .txt affiche des horaires faux et le
    regroupement en paragraphes (qui raisonne sur les écarts) part en vrille.
    """
    fake_audio["total"] = 30.0
    t = _ScriptedTranscriber([_res((0.0, 51.6, "segment qui deborde"))])

    result = transcribe_file(t, Path("x.wav"), 30.0, block_duration_s=30)

    assert result.segments[0].end <= 30.0


def test_negative_timestamps_are_clamped(fake_audio):
    fake_audio["total"] = 30.0
    t = _ScriptedTranscriber([_res((-5.0, 10.0, "debut negatif"))])

    result = transcribe_file(t, Path("x.wav"), 30.0, block_duration_s=30)

    assert result.segments[0].start >= 0.0


def test_inverted_span_is_normalised(fake_audio):
    """Fin avant début : on ne doit pas produire de segment incohérent."""
    fake_audio["total"] = 30.0
    t = _ScriptedTranscriber([_res((10.0, 4.0, "inverse"))])

    result = transcribe_file(t, Path("x.wav"), 30.0, block_duration_s=30)

    seg = result.segments[0]
    assert seg.end >= seg.start


def test_tail_block_too_short_is_skipped(fake_audio):
    """Régression mesurée : un résidu de 0,1 s a produit l'hallucination
    « Merci. ». Sous 1 s on s'arrête au lieu de transcrire du remplissage."""
    fake_audio["total"] = 30.4
    t = _ScriptedTranscriber([
        _res((0.0, 29.9, "bloc plein")),
        _res((0.0, 1.0, "HALLUCINATION")),
    ])

    result = transcribe_file(t, Path("x.wav"), 30.4, block_duration_s=30)

    assert len(t.calls) == 1, "le résidu de 0,4 s n'aurait pas dû être transcrit"
    assert "HALLUCINATION" not in result.text


def test_default_block_is_several_minutes():
    """Les blocs larges laissent Whisper appliquer sa propre logique long-form,
    testée en amont, au lieu de la remplacer par la nôtre."""
    import inspect

    default = inspect.signature(transcribe_file).parameters["block_duration_s"].default
    assert default >= 120
