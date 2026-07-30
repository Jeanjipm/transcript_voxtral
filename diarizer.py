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

## Pourquoi la v2.1 et pas la v1 — ce que la mesure a montré

La v1 ne marchait pas, et cette version du fichier a d'abord accusé le
portage MLX. C'était faux. Voici ce qui a été réellement mesuré, sur un
dialogue synthétique construit avec `say` (deux voix macOS très distinctes,
Thomas et Audrey, quatre tours de parole, vérité terrain connue à la
milliseconde) :

| Modèle | Attribution correcte | Non détecté | Locuteurs distincts |
|---|---|---|---|
| v1, Thomas parle en premier | 56 % | 44 % | 2 |
| v1, Audrey parle en premier | 49 % | 7 % | 2 (mais fusionnés) |
| **v2.1, Thomas en premier** | **96 %** | 4 % | 2 |
| **v2.1, Audrey en premier** | **96 %** | 4 % | 2 |

49 %, c'est le niveau du hasard : la v1 rangeait les deux voix dans le même
canal. Les probabilités brutes disent pourquoi — sur le premier locuteur elle
donne 1,00 de façon parfaitement stable, sur le second elle s'effondre entre
0,00 et 0,38 avec quelques pics isolés. Ce n'est pas une confusion entre deux
personnes : le second locuteur n'est tout simplement pas détecté.

Deux vérifications ont écarté le portage :
- **chaque voix seule** est détectée à 1,00 sur 94-99 % des trames. Le
  frontal acoustique (mel, préaccentuation, normalisation) et l'encodeur
  fonctionnent donc ;
- le prétraitement de `mlx-audio` a été relu ligne à ligne contre la
  référence NeMo (échelle mel *slaney*, `log(x + 2⁻²⁴)`, normalisation par
  bande avec correction de Bessel, fenêtre de Hann symétrique) : conforme.

Erreur méthodologique à ne pas refaire : le premier test comparait deux
copies **de la même voix** transposées en hauteur. Deux transpositions d'une
même personne peuvent légitimement être jugées comme un seul locuteur — ce
test ne pouvait rien prouver. Le test valable utilise deux voix réellement
différentes.

## Ce que fait la v2.1, mesuré

- **96 % d'attribution correcte** sur deux voix, dans les deux ordres.
- **Trois voix** successives : les trois séparées correctement.
- **Parole simultanée** : les deux locuteurs qui se chevauchent sont bien
  détectés — avec un faux positif sur un troisième canal, ce que le passage
  par `argmax` de `turns_from_probs` élimine.
- **57× le temps réel** : une heure d'enregistrement coûte environ une minute.
- Poids : 225 Mo.

## Limites qui restent

- **4 locuteurs maximum.** Au-delà, les voix supplémentaires sont rabattues
  sur les 4 canaux existants.
- Le modèle attribue des numéros de canal quelconques (0 et 3 pour deux
  personnes, par exemple) : `turns_from_probs` renumérote dans l'ordre
  d'apparition, sinon l'utilisateur lirait « Locuteur 1 » puis « Locuteur 4 ».
- Testé sur des voix de synthèse, pas sur du terrain réel avec réverbération
  et micro lointain. À confirmer sur un vrai enregistrement.
- Dépendance optionnelle : sans le paquet `mlx-audio`, tout ce module se
  désactive proprement avec un message explicite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcriber import Segment


DEFAULT_MODEL = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"

# Sortformer gère jusqu'à 4 locuteurs simultanés.
MAX_SPEAKERS = 4

# Seuil de présence d'un locuteur sur une trame.
DEFAULT_THRESHOLD = 0.5

# Durée d'une trame de sortie : pas de 10 ms du spectrogramme × sous-
# échantillonnage 8 du FastConformer. Repli seulement — la vraie valeur est
# lue dans la configuration du modèle.
_FALLBACK_FRAME_S = 0.08

# Fenêtre de lissage des étiquettes, en trames impaires (5 × 80 ms = 400 ms).
# Sans elle, une trame isolée au-dessus du seuil crée un « tour de parole »
# de 80 ms qui n'existe pas.
_SMOOTHING_FRAMES = 5

# Deux tours du même locuteur séparés par moins que ça sont recollés : une
# respiration au milieu d'une phrase ne fait pas deux tours de parole.
_MAX_GAP_S = 0.5

# En dessous, on jette : ce n'est pas un tour de parole, c'est du scintillement.
# 0,5 s parce que c'est la durée mesurée des artefacts de transition — le
# modèle allume brièvement un canal tiers pendant le silence qui sépare deux
# phrases. Une vraie interjection (« Oui. », « D'accord. ») dépasse ce seuil.
_MIN_TURN_S = 0.5

# Temps de parole cumulé en dessous duquel un locuteur est considéré comme un
# artefact et retiré. Le modèle hésite parfois sur la première seconde d'un
# fichier avant de se fixer ; sans ce filtre, ce faux locuteur consomme un
# numéro et le lecteur voit « Locuteur 1 » puis « Locuteur 3 ».
_MIN_SPEAKER_S = 1.5

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


def _smooth(labels: list[int | None], window: int) -> list[int | None]:
    """Lissage par vote majoritaire sur une fenêtre glissante impaire.

    Le modèle décide trame par trame, indépendamment. Une seule trame qui
    bascule au milieu d'une phrase produirait un tour de parole de 80 ms
    attribué à quelqu'un d'autre — visible et faux dans le transcript.

    Départage, dans l'ordre : le plus de voix ; puis un locuteur plutôt que
    du silence (mieux vaut un mot attribué qu'un trou) ; puis le plus petit
    numéro. Entièrement déterministe — deux exécutions doivent donner le
    même transcript.
    """
    if window <= 1 or not labels:
        return list(labels)

    def rank(item: tuple[int | None, int]) -> tuple[int, int, int]:
        label, count = item
        return (count, 0 if label is None else 1, -(label or 0))

    half = window // 2
    out: list[int | None] = []
    for i in range(len(labels)):
        votes: dict[int | None, int] = {}
        for j in range(max(0, i - half), min(len(labels), i + half + 1)):
            votes[labels[j]] = votes.get(labels[j], 0) + 1
        out.append(max(votes.items(), key=rank)[0])
    return out


def _runs(labels: list[int | None], frame_s: float) -> list[SpeakerTurn]:
    """Regroupe les trames consécutives de même étiquette en intervalles."""
    turns: list[SpeakerTurn] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] is not None:
                turns.append(
                    SpeakerTurn(
                        start=start * frame_s,
                        end=i * frame_s,
                        speaker=int(labels[start]),  # type: ignore[arg-type]
                    )
                )
            start = i
    return turns


def _merge_gaps(turns: list[SpeakerTurn], max_gap_s: float) -> list[SpeakerTurn]:
    """Recolle deux tours du même locuteur séparés par un court silence."""
    merged: list[SpeakerTurn] = []
    for turn in turns:
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end <= max_gap_s
        ):
            merged[-1] = SpeakerTurn(merged[-1].start, turn.end, turn.speaker)
        else:
            merged.append(turn)
    return merged


def _drop_marginal_speakers(
    turns: list[SpeakerTurn], min_speaker_s: float
) -> list[SpeakerTurn]:
    """Retire les locuteurs dont le temps de parole cumulé est négligeable.

    Un « locuteur » qui totalise moins d'une seconde et demie sur tout un
    enregistrement n'existe pas : c'est le modèle qui hésite, typiquement sur
    les premières trames avant de se fixer. On ne garde jamais un seul
    locuteur par ce chemin — si tout tombe sous le seuil, on rend la liste
    intacte plutôt que de renvoyer un transcript sans aucune étiquette.
    """
    totals: dict[int, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + (turn.end - turn.start)

    kept = {s for s, total in totals.items() if total >= min_speaker_s}
    if not kept:
        return turns
    return [t for t in turns if t.speaker in kept]


def _renumber(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Renumérote les locuteurs de 0 à N-1, dans l'ordre d'apparition.

    Le modèle utilise des numéros de canal quelconques : sur un dialogue à
    deux, il rend couramment les canaux 0 et 3. Sans cette étape, le lecteur
    verrait « Locuteur 1 » et « Locuteur 4 » et se demanderait où sont passés
    les deux autres.
    """
    order: dict[int, int] = {}
    for turn in turns:
        if turn.speaker not in order:
            order[turn.speaker] = len(order)
    return [
        SpeakerTurn(t.start, t.end, order[t.speaker]) for t in turns
    ]


def turns_from_probs(
    probs,  # noqa: ANN001  — (trames, locuteurs), numpy ou mlx
    frame_s: float = _FALLBACK_FRAME_S,
    threshold: float = DEFAULT_THRESHOLD,
    smoothing_frames: int = _SMOOTHING_FRAMES,
    max_gap_s: float = _MAX_GAP_S,
    min_turn_s: float = _MIN_TURN_S,
    min_speaker_s: float = _MIN_SPEAKER_S,
    duration_s: float | None = None,
) -> list[SpeakerTurn]:
    """Transforme la sortie brute du modèle en tours de parole exploitables.

    Fonction pure — c'est ici que se joue la lisibilité du transcript, donc
    elle est testable sans modèle.

    On prend l'`argmax` par trame plutôt que « tous les canaux au-dessus du
    seuil » : chaque passage de texte ne peut porter qu'une étiquette, et sur
    les zones de parole simultanée le modèle allume un troisième canal
    parasite que l'argmax écarte naturellement.
    """
    import numpy as np

    matrix = np.asarray(probs, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return []

    best = matrix.argmax(axis=1)
    labels: list[int | None] = [
        int(b) if matrix[i, b] >= threshold else None
        for i, b in enumerate(best)
    ]

    turns = _runs(_smooth(labels, smoothing_frames), frame_s)
    # Filtrer AVANT de recoller : aux transitions, le modèle allume brièvement
    # un canal tiers dans le silence entre deux phrases. Le supprimer d'abord
    # permet aux deux vrais tours de se rejoindre ensuite ; l'ordre inverse
    # figerait l'intrus au milieu.
    turns = [t for t in turns if t.end - t.start >= min_turn_s]
    turns = _drop_marginal_speakers(turns, min_speaker_s)
    turns = _merge_gaps(turns, max_gap_s)

    if duration_s is not None:
        # Les caractéristiques sont complétées à un multiple de 16 trames :
        # sans ce rognage, le dernier tour déborde de la fin du fichier.
        turns = [
            SpeakerTurn(t.start, min(t.end, duration_s), t.speaker)
            for t in turns
            if t.start < duration_s
        ]

    return _renumber(turns)


def diarize(
    audio_path: Path,
    model_repo: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[SpeakerTurn]:
    """Retourne les tours de parole détectés dans `audio_path`.

    Le fichier doit être en 16 kHz mono (cf. `audio_convert.ensure_16k_mono`).
    On traite le fichier entier d'un coup : mesuré à 57× le temps réel, donc
    même une heure d'audio reste rapide, et ça évite d'avoir à recoller
    l'identité des locuteurs entre des morceaux traités séparément — ce qui
    serait la principale source d'erreur.

    On repart des probabilités brutes plutôt que des segments tout faits de
    `mlx-audio` : ceux-ci sortent du seuillage trame à trame, sans lissage ni
    renumérotation, et donnaient des rafales de tours de 80 ms.
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

    if hf_offline.is_model_cached(model_repo):
        model = load(hf_offline.resolve_local_path(model_repo))
    else:
        # Premier usage : le modèle n'est pas encore là. Sans cette
        # parenthèse, le mode hors-ligne — activé au démarrage pour que la
        # dictée ne dépende jamais du réseau — ferait échouer le
        # téléchargement avec une erreur incompréhensible.
        with hf_offline.allow_network():
            model = load(model_repo)

    result = model.generate(
        str(audio_path), sample_rate=16_000, threshold=threshold
    )

    return turns_from_probs(
        result.speaker_probs,
        frame_s=_frame_duration(model),
        threshold=threshold,
        duration_s=_audio_duration(audio_path),
    )


def _frame_duration(model) -> float:  # noqa: ANN001
    """Durée d'une trame de sortie, lue dans la config du modèle."""
    try:
        proc = model.config.processor_config
        factor = model.config.fc_encoder_config.subsampling_factor
        return (proc.hop_length * factor) / proc.sampling_rate
    except Exception:  # noqa: BLE001
        return _FALLBACK_FRAME_S


def _audio_duration(audio_path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return None


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
