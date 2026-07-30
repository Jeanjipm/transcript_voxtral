"""
Identification des locuteurs (« qui parle quand »), et fusion avec le texte.

Modèle : NVIDIA Sortformer porté sur MLX
(`mlx-community/diar_sortformer_4spk-v1-fp16`, 117 M paramètres, ~235 Mo).

## Pourquoi Sortformer et pas pyannote

pyannote est le standard du domaine, mais il exige un **token HuggingFace et
l'acceptation d'une licence** sur le site avant de pouvoir télécharger le
modèle. Pour un utilisateur non-développeur — et plus encore pour distribuer
l'outil à des amis — c'est un blocage net. Sortformer est public, sans
compte ni token.

## ⚠️ CE PORTAGE N'EST PAS EXPLOITABLE EN L'ÉTAT

Testé, et il échoue sur son cas d'usage principal. Mesuré en faisant varier la
hauteur de la seconde voix (même phrase, même enregistrement de départ) :

| Écart de hauteur | Locuteurs détectés | Probabilité du canal 2 |
|---|---|---|
| ×1,00 (voix identique) | 1 | 0,0002 |
| ×1,05 | 1 | 0,0001 |
| ×1,10 | 1 | 0,0002 |
| ×1,20 | 1 | 0,17 |
| ×1,40 | 2 | 0,995 |
| ×1,60 | 2 | 0,9999 |

Il ne sépare donc qu'au-delà d'environ **40 % d'écart de hauteur**, soit près
d'une octave — très au-delà de ce qui distingue deux personnes réelles. Une
voix masculine et une voix féminine ne sont PAS séparées, et un test
utilisateur sur un enregistrement réel à deux voix a également échoué.

Ce n'est pas un seuil à ajuster : la détection d'activité vocale (le découpage
en segments de parole) est correcte, seule l'attribution des locuteurs
s'effondre. Symptôme typique d'un défaut de prétraitement dans le portage, qui
détruit l'information de timbre en préservant les différences spectrales
grossières. Vérifié aussi en fp32 : même résultat, ce n'est pas une saturation
fp16.

Ce module reste en place — le code de fusion (`assign_speakers`) est correct et
réutilisable avec un autre moteur — mais la fonctionnalité doit rester
**désactivée par défaut** et signalée comme expérimentale tant qu'un backend
fiable n'a pas remplacé Sortformer. La piste sérieuse est `pyannote.audio`
(~10 % de DER en conditions réelles), au prix d'un compte HuggingFace et de
l'acceptation d'une licence.

## Ce qui fonctionne quand même

- **Performance** : 600 s d'audio diarisés en 2,7 s (224× le temps réel), donc
  une heure d'enregistrement coûterait une quinzaine de secondes.
- **La détection de parole** est juste : les frontières de segments
  correspondent bien aux tours de parole.

## Limites structurelles

- **4 locuteurs maximum.** Au-delà, les voix supplémentaires sont rabattues
  sur les 4 canaux existants.
- Dépendance optionnelle : sans le paquet `mlx-audio`, tout ce module se
  désactive proprement avec un message explicite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcriber import Segment


DEFAULT_MODEL = "mlx-community/diar_sortformer_4spk-v1-fp16"

# Sortformer v1 gère jusqu'à 4 locuteurs simultanés.
MAX_SPEAKERS = 4

# Seuil de présence d'un locuteur sur une trame.
DEFAULT_THRESHOLD = 0.5

# Recouvrement minimum, en secondes, pour attribuer un passage à un locuteur.
# En dessous, c'est du bruit de frontière et on préfère ne rien affirmer.
_MIN_OVERLAP_S = 0.05


class DiarizationUnavailable(RuntimeError):
    """Message en français, destiné à l'utilisateur."""


@dataclass(frozen=True)
class SpeakerTurn:
    """Un intervalle pendant lequel un locuteur donné parle."""

    start: float
    end: float
    speaker: int


def is_available() -> bool:
    """True si le paquet de diarisation est installé."""
    try:
        import mlx_audio.vad  # noqa: F401
    except ImportError:
        return False
    return True


def diarize(
    audio_path: Path,
    model_repo: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[SpeakerTurn]:
    """Retourne les tours de parole détectés dans `audio_path`.

    Le fichier doit être en 16 kHz mono (cf. `audio_convert.ensure_16k_mono`).
    On traite le fichier entier d'un coup : mesuré à 224× le temps réel, donc
    même une heure d'audio reste rapide, et ça évite d'avoir à recoller
    l'identité des locuteurs entre des morceaux traités séparément — ce qui
    serait la principale source d'erreur.
    """
    if not is_available():
        raise DiarizationUnavailable(
            "L'identification des locuteurs nécessite le paquet « mlx-audio », "
            "qui n'est pas installé.\n\n"
            "Pour l'activer :\n"
            "  ~/.voxtral/venv/bin/pip install mlx-audio\n\n"
            "Puis relance Voxtral."
        )

    from mlx_audio.vad import load  # type: ignore[import-not-found]

    import hf_offline

    model = load(hf_offline.resolve_local_path(model_repo))
    result = model.generate(
        str(audio_path), sample_rate=16_000, threshold=threshold
    )

    turns = [
        SpeakerTurn(
            start=float(seg.start), end=float(seg.end), speaker=int(seg.speaker)
        )
        for seg in result.segments
        if float(seg.end) > float(seg.start)
    ]
    turns.sort(key=lambda t: t.start)
    return turns


def assign_speakers(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> list[Segment]:
    """Attribue un locuteur à chaque segment de texte, par recouvrement.

    Fonction pure — c'est ici que se joue la qualité du résultat final, donc
    elle est isolée et testable sans modèle.

    Chaque segment de texte reçoit le locuteur avec lequel il partage le plus
    de temps. Un segment sans recouvrement significatif garde `speaker=None`
    plutôt que de se voir attribuer un locuteur au hasard : mieux vaut ne rien
    affirmer que d'attribuer une phrase à la mauvaise personne.
    """
    if not turns:
        return list(segments)

    result: list[Segment] = []
    for seg in segments:
        overlaps: dict[int, float] = {}
        for turn in turns:
            shared = min(seg.end, turn.end) - max(seg.start, turn.start)
            if shared > 0:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + shared

        speaker: int | None = None
        if overlaps:
            best, best_overlap = max(overlaps.items(), key=lambda kv: kv[1])
            if best_overlap >= _MIN_OVERLAP_S:
                speaker = best

        result.append(
            Segment(
                start=seg.start, end=seg.end, text=seg.text, speaker=speaker
            )
        )
    return result


def speaker_label(speaker: int | None) -> str:
    """Étiquette affichée. Numérotée à partir de 1 : « Locuteur 0 » n'a aucun
    sens pour un lecteur non-technique."""
    if speaker is None:
        return ""
    return f"Locuteur {speaker + 1}"
