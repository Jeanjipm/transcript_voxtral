"""
Transcription de fichiers longs : boucle par blocs + mise en forme du texte.

Module pur : aucun thread, aucune UI, aucun accès réseau. Il reçoit un
`Transcriber` et un chemin de fichier déjà normalisé en 16 kHz mono, et rend
un `FileTranscript`. C'est ce qui le rend entièrement testable.

## Pourquoi des blocs LARGES, et non de 30 secondes

Leçon apprise en mesurant. `mlx_whisper` possède déjà sa propre boucle
long-form : fenêtres de 30 s, ré-ancrage sur la fin du dernier segment décodé,
conditionnement sur le texte précédent, seuils anti-hallucination. Tout ça est
testé en amont.

Découper l'audio en tranches de 30 s **remplace** cette logique par la nôtre,
et réintroduit exactement les défauts qu'elle corrige. Constaté sur un
enregistrement de 80 s : des horodatages incohérents (un segment finissant à
51,6 s dans un bloc de 30 s) et un paragraphe entier de discours perdu.

On donne donc à Whisper des blocs de plusieurs minutes (`block_duration_s`,
300 s par défaut), à l'intérieur desquels il applique sa propre logique. Le
découpage ne sert plus qu'à trois choses, toutes des besoins d'interface :
annuler, afficher la progression, et laisser une dictée passer devant. Une
jointure toutes les 5 minutes au lieu de toutes les 30 secondes, soit dix fois
moins d'occasions de se tromper.

## Pourquoi on ne passe PAS d'amorce de texte

`initial_prompt` semblait la bonne idée pour assurer la continuité d'une
jointure à l'autre. Mesuré : il **fait perdre du contenu**. Sur le même
enregistrement, le bloc 60→80 s transcrit correctement seul, mais avec une
amorce prise dans le texte précédent il rend « Je vous propose de passer aux
questions » et saute une quinzaine de secondes de discours. C'est un travers
connu de Whisper — l'amorce biaise le décodage et il peut se mettre à résumer.
Le bénéfice était théorique, la perte est démontrée : on n'en passe pas.
Le paramètre reste dans l'API pour les backends qui sauraient s'en servir.

## Garde-fous

- **Avance minimale.** Un bloc de pur silence ne rend aucun segment, donc
  aucune avance, donc une boucle infinie. En dessous de `_MIN_ADVANCE_S` on
  avance d'un bloc entier.
- **Avance maximale.** Whisper peut annoncer une fin de segment au-delà de
  l'audio fourni. Sans plafond, on sauterait de l'audio jamais transcrit.
- **Horodatages bornés.** Même cause : un segment qui dépasse la fin du bloc
  est ramené dedans, sinon le .txt affiche des horaires faux et les
  paragraphes sont mal regroupés.
- **Bloc résiduel ignoré.** Un dernier bloc de quelques dixièmes de seconde
  est du remplissage, et Whisper hallucine dessus (constaté : 0,1 s d'audio
  → « Merci. »). En dessous de `_MIN_BLOCK_S` on s'arrête.
- **Langue figée** après le premier bloc : évite le papillonnage d'une langue
  à l'autre en cours de fichier.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import audio_convert
from transcriber import Segment, Transcriber


# En dessous de cette avance, on considère que le bloc n'a rien produit
# d'exploitable et on avance d'un bloc entier. Garantit la terminaison.
_MIN_ADVANCE_S = 1.0

# Un bloc plus court que ça n'est pas transcrit : c'est un résidu de fin de
# fichier, et Whisper hallucine sur du quasi-vide (mesuré : 0,1 s d'audio a
# produit « Merci. »).
_MIN_BLOCK_S = 1.0

# Longueur de la queue de texte utilisable comme amorce. Conservé pour les
# backends qui savent s'en servir, mais NON utilisé avec Whisper (cf. la
# docstring du module : ça fait perdre du contenu).
_PROMPT_TAIL_CHARS = 200

# Au-delà de cet écart entre deux segments, on considère qu'il y a une rupture
# et on ouvre un paragraphe.
_PARAGRAPH_GAP_S = 2.0

# Longueur cible d'un paragraphe, pour ne pas produire un pavé illisible quand
# quelqu'un parle sans pause.
_PARAGRAPH_MAX_CHARS = 400


@dataclass
class FileTranscript:
    """Résultat complet d'une transcription de fichier."""

    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration_s: float = 0.0
    # Position atteinte quand le job a été interrompu ; None si terminé.
    cancelled_at_s: float | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancelled_at_s is not None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()


def transcribe_file(
    transcriber: Transcriber,
    audio_path: Path,
    total_duration_s: float,
    block_duration_s: int = 300,
    language: str = "auto",
    task: str = "transcribe",
    cancel: threading.Event | None = None,
    on_progress: Callable[[float, float], None] | None = None,
    use_prompt_tail: bool = False,
) -> FileTranscript:
    """Transcrit `audio_path` bloc par bloc et rend les segments horodatés.

    `audio_path` doit déjà être en 16 kHz mono (cf.
    `audio_convert.ensure_16k_mono`). `total_duration_s` évite un nouveau
    sondage du fichier.

    `cancel` permet l'interruption : elle est testée entre deux blocs, donc la
    latence d'annulation vaut au pire la durée de traitement d'un bloc.
    Le résultat partiel est conservé et marqué comme interrompu — jeter 90 %
    d'un job de 55 minutes serait inacceptable.

    `use_prompt_tail` est désactivé par défaut : mesuré comme provoquant des
    pertes de contenu avec Whisper (cf. la docstring du module).
    """
    segments: list[Segment] = []
    pinned_language: str | None = None if language in ("", "auto") else language
    start = 0.0

    while start < total_duration_s:
        if cancel is not None and cancel.is_set():
            return FileTranscript(
                segments=segments,
                language=pinned_language,
                duration_s=total_duration_s,
                cancelled_at_s=start,
            )

        audio = audio_convert.read_block(audio_path, start, block_duration_s)
        if audio.size == 0:
            break

        block_seconds = audio.shape[0] / float(audio_convert.TARGET_SAMPLE_RATE)
        if block_seconds < _MIN_BLOCK_S:
            # Résidu de fin de fichier : Whisper hallucinerait dessus.
            break

        result = transcriber.transcribe_array(
            audio,
            sample_rate=audio_convert.TARGET_SAMPLE_RATE,
            language=pinned_language,
            task=task,
            initial_prompt=_prompt_tail(segments) if use_prompt_tail else None,
        )

        # Langue figée dès le premier bloc qui en révèle une.
        if pinned_language is None and result.language:
            pinned_language = result.language

        segments.extend(
            _absolute_segments(result.segments, start, block_seconds)
        )

        start += _next_advance(result.segments, block_seconds, block_duration_s)

        if on_progress is not None:
            on_progress(min(start, total_duration_s), total_duration_s)

    return FileTranscript(
        segments=segments,
        language=pinned_language,
        duration_s=total_duration_s,
        cancelled_at_s=None,
    )


def _absolute_segments(
    block_segments: list[Segment], offset_s: float, block_seconds: float
) -> list[Segment]:
    """Décale les horodatages en temps absolu, en les bornant au bloc.

    Le bornage n'est pas cosmétique : Whisper annonce parfois une fin de
    segment bien au-delà de l'audio fourni (constaté : fin à 51,6 s dans un
    bloc de 30 s). Sans ça le .txt afficherait des horaires faux et le
    regroupement en paragraphes, qui raisonne sur les écarts entre segments,
    produirait n'importe quoi.
    """
    result: list[Segment] = []
    for seg in block_segments:
        start = min(max(seg.start, 0.0), block_seconds)
        end = min(max(seg.end, start), block_seconds)
        result.append(
            Segment(start=start + offset_s, end=end + offset_s, text=seg.text)
        )
    return result


def _next_advance(
    block_segments: list[Segment], block_seconds: float, block_duration_s: int
) -> float:
    """De combien avancer après un bloc.

    On repart de la fin du dernier segment, borné des deux côtés (cf. la
    docstring du module pour le pourquoi de chaque borne).
    """
    if not block_segments:
        return float(block_duration_s)

    advance = block_segments[-1].end
    # Plafond : Whisper remplit ses fenêtres de 30 s et peut annoncer une fin
    # au-delà de l'audio réellement fourni. Sans ce min, on sauterait des
    # secondes d'audio jamais transcrites.
    ceiling = min(float(block_duration_s), block_seconds)
    if advance > ceiling:
        return ceiling
    if advance < _MIN_ADVANCE_S:
        return float(block_duration_s)
    return advance


def _prompt_tail(segments: list[Segment]) -> str | None:
    """Queue du texte déjà transcrit, pour amorcer le bloc suivant."""
    if not segments:
        return None
    tail = " ".join(s.text for s in segments[-5:]).strip()
    if not tail:
        return None
    return tail[-_PROMPT_TAIL_CHARS:]


# ---- Mise en forme du .txt ----


def format_timestamp(seconds: float) -> str:
    """Formate en hh:mm:ss (toujours 3 groupes, pour un alignement stable)."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_duration(seconds: float) -> str:
    return format_timestamp(seconds)


def format_transcript(
    transcript: FileTranscript,
    source_name: str,
    model_name: str,
    include_timestamps: bool = True,
    generated_at: str | None = None,
) -> str:
    """Rend le contenu du fichier .txt.

    `generated_at` est injectable pour que les tests soient déterministes.
    """
    if generated_at is None:
        generated_at = time.strftime("%d/%m/%Y à %H:%M")

    lines: list[str] = [
        f"Transcription — {source_name}",
        f"Modèle : {model_name}",
        f"Durée : {format_duration(transcript.duration_s)} "
        f"— généré le {generated_at}",
    ]
    if transcript.language:
        lines.append(f"Langue : {transcript.language}")
    if transcript.cancelled:
        # En-tête explicite : un .txt partiel qui ne se signale pas comme tel
        # serait pris pour une transcription complète.
        lines.append(
            f"[Transcription interrompue à "
            f"{format_timestamp(transcript.cancelled_at_s or 0.0)}]"
        )
    lines.append("")

    for paragraph in _paragraphs(transcript.segments):
        prefix = ""
        if include_timestamps:
            prefix = f"[{format_timestamp(paragraph[0].start)}] "
        speaker = _paragraph_speaker(paragraph)
        if speaker:
            prefix = f"{prefix}{speaker} : "
        body = " ".join(s.text for s in paragraph).strip()
        if body:
            lines.append(f"{prefix}{body}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _paragraph_speaker(paragraph: list[Segment]) -> str:
    """Étiquette du locuteur d'un paragraphe, vide s'il n'y en a pas.

    Import local : `diarizer` est une dépendance optionnelle côté modèle, mais
    `speaker_label` est du pur formatage et toujours disponible. L'import est
    quand même local pour ne pas créer de cycle avec `transcriber`.
    """
    speaker = paragraph[0].speaker
    if speaker is None:
        return ""
    from diarizer import speaker_label

    return speaker_label(speaker)


def _paragraphs(segments: list[Segment]) -> list[list[Segment]]:
    """Regroupe les segments en paragraphes lisibles.

    Rupture sur un changement de locuteur, sur un silence notable, ou quand le
    paragraphe devient trop long pour rester lisible. Le changement de locuteur
    est la rupture la plus importante : mélanger deux personnes dans un même
    paragraphe rendrait le transcript trompeur.
    """
    if not segments:
        return []

    groups: list[list[Segment]] = [[segments[0]]]
    length = len(segments[0].text)

    for previous, current in zip(segments, segments[1:]):
        gap = current.start - previous.end
        speaker_changed = current.speaker != previous.speaker
        if (
            speaker_changed
            or gap > _PARAGRAPH_GAP_S
            or length > _PARAGRAPH_MAX_CHARS
        ):
            groups.append([current])
            length = len(current.text)
        else:
            groups[-1].append(current)
            length += len(current.text) + 1

    return groups
