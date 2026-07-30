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

## Ce qui est mesuré, et ce qui ne l'est pas

Mesuré sur cette machine :
- 600 s d'audio diarisés en 2,7 s (224× le temps réel), donc une heure
  d'enregistrement coûte une quinzaine de secondes ;
- sur deux voix franchement distinctes, les deux locuteurs sont correctement
  séparés.

**Non validé** : la qualité sur de la vraie parole humaine à plusieurs. Les
voix de synthèse macOS (`say`) partagent le même moteur et se ressemblent
trop — Sortformer les a fusionnées en un seul locuteur. Ce n'est donc pas un
matériel de test représentatif, et la qualité réelle reste à confirmer sur un
enregistrement authentique.

## Limites à connaître

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
