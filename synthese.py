"""
Synthèse vocale locale : un texte devient du son, avec Voxtral 4B TTS sur MLX.

C'est le chemin inverse de la dictée. Deux usages :
  - le menu « Lire le texte copié » de l'app (lecture immédiate au haut-parleur) ;
  - `vocaliser_cahier.py`, qui produit des MP3 pour le cahier de révisions.

Le modèle (`mlx-community/Voxtral-4B-TTS-2603-mlx-4bit`, ~2,5 Go) vit dans le
cache HuggingFace comme les autres, et se charge à la première demande. Même
règle hors-ligne que `transcriber.py` : on donne au chargeur le dossier local
du snapshot quand il est en cache, et le réseau n'est ouvert que le temps du
téléchargement s'il manque (`hf_offline.network_if_missing`). On ne touche
jamais à `hf_offline.refresh`, réservé au modèle de dictée.

Les imports de mlx_audio sont retardés : les tests tournent sans MLX, et
l'app ne doit pas payer le chargement de la bibliothèque si personne ne lit
jamais de texte.

Pourquoi mlx-audio 0.4.6 et pas plus récent : c'est la version figée dans
requirements.lock, et elle contient déjà Voxtral TTS (arrivé en 0.4.2). Les
versions 0.5+/0.6+ imposent de monter transformers puis mlx, ce qui touche
la dictée et obligerait tous les utilisateurs à réinstaller.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import hf_offline


DEFAULT_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
# Fréquence d'échantillonnage de sortie du modèle (fixe, cf. son config.json).
SAMPLE_RATE = 24_000
# Garde-fou pour « Lire le texte copié » : au-delà, on lit le début seulement.
# Un presse-papier peut contenir un article entier, et le modèle plafonne de
# toute façon à 4 096 jetons audio (quelques minutes).
MAX_CHARS = 2_000
# Dossier du dépôt HF qui contient une empreinte par voix (`<nom>.safetensors`).
VOICE_DIR = "voice_embedding"
PREFIXE_FRANCAIS = "fr_"

# Libellés lisibles pour les Préférences (règle de settings_ui : jamais de
# code technique à l'écran). Les deux voix françaises d'abord. La liste
# complète des 20 voix vient du modèle ; une voix absente d'ici s'affiche
# avec son nom brut.
LIBELLES_VOIX: list[tuple[str, str]] = [
    ("fr_female", "Français — femme"),
    ("fr_male", "Français — homme"),
    ("casual_female", "Anglais — femme, naturelle"),
    ("casual_male", "Anglais — homme, naturel"),
    ("cheerful_female", "Anglais — femme, enjouée"),
    ("neutral_female", "Anglais — femme, neutre"),
    ("neutral_male", "Anglais — homme, neutre"),
    ("es_female", "Espagnol — femme"),
    ("es_male", "Espagnol — homme"),
    ("de_female", "Allemand — femme"),
    ("de_male", "Allemand — homme"),
    ("it_female", "Italien — femme"),
    ("it_male", "Italien — homme"),
    ("pt_female", "Portugais — femme"),
    ("pt_male", "Portugais — homme"),
    ("nl_female", "Néerlandais — femme"),
    ("nl_male", "Néerlandais — homme"),
    ("ar_male", "Arabe — homme"),
    ("hi_female", "Hindi — femme"),
    ("hi_male", "Hindi — homme"),
]


def libelle_voix(voix: str) -> str:
    return dict(LIBELLES_VOIX).get(voix, voix)


class SynthesisError(RuntimeError):
    """Erreur de synthèse ou d'encodage, avec un message destiné à l'utilisateur."""


# ----------------------------------------------------------------------
# Voix
# ----------------------------------------------------------------------


def voix_disponibles(repo: str = DEFAULT_MODEL) -> list[str]:
    """Noms exacts des voix que le modèle fournit, triés.

    Lus dans le snapshot en cache (un fichier par voix), sans charger le
    modèle. Si le modèle n'est pas encore téléchargé, on se rabat sur la
    table interne de mlx-audio, puis sur notre liste de libellés.
    """
    local = hf_offline.resolve_local_path(repo)
    if local != repo:
        dossier = Path(local) / VOICE_DIR
        noms = sorted(p.stem for p in dossier.glob("*.safetensors"))
        if noms:
            return noms
    try:
        from mlx_audio.tts.models.voxtral_tts.voxtral_tts import (  # type: ignore[import-not-found]
            VOICE_MAP,
        )

        return sorted(VOICE_MAP)
    except Exception:  # noqa: BLE001 — mlx_audio absent ou version différente
        return [code for code, _ in LIBELLES_VOIX]


def voix_francaises(repo: str = DEFAULT_MODEL) -> list[str]:
    """Les voix françaises (homme et femme), repérées à leur préfixe `fr_`."""
    return [v for v in voix_disponibles(repo) if v.startswith(PREFIXE_FRANCAIS)]


# ----------------------------------------------------------------------
# Moteur
# ----------------------------------------------------------------------


def _vers_numpy(audio: Any) -> np.ndarray:
    """Tableau MLX (ou autre) → numpy float32 mono, à plat."""
    try:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    except TypeError:
        # Un mx.array en bfloat16 ne se convertit pas directement en numpy.
        import mlx.core as mx  # type: ignore[import-not-found]

        return np.asarray(audio.astype(mx.float32)).reshape(-1)


class Synthetiseur:
    """Un modèle Voxtral TTS, chargé à la première lecture.

    Thread-safe pour le chargement ; les lectures elles-mêmes doivent être
    sérialisées par l'appelant (dans l'app, c'est l'inference-worker, seul
    propriétaire des modèles MLX).
    """

    def __init__(self, repo: str = DEFAULT_MODEL) -> None:
        self.repo = repo
        self.sample_rate = SAMPLE_RATE
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # Import retardé : mlx_audio tire mlx, lourd, et n'existe pas
            # dans l'environnement de test.
            from mlx_audio.tts.utils import load  # type: ignore[import-not-found]

            source = hf_offline.resolve_local_path(self.repo)
            with hf_offline.network_if_missing(self.repo):
                model = load(source)
            self.sample_rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
            self._model = model

    def preload(self) -> None:
        """Charge le modèle maintenant, pour que la première lecture soit rapide."""
        self._ensure_loaded()

    def voix(self) -> list[str]:
        """Voix connues du modèle chargé ; sinon celles du cache."""
        if self._model is not None:
            fichiers = getattr(self._model, "_voice_embedding_files", None)
            if fichiers:
                return sorted(fichiers)
        return voix_disponibles(self.repo)

    def lire(self, texte: str, voix: str) -> np.ndarray:
        """Synthétise `texte` avec la voix `voix`.

        Retourne un tableau numpy float32 mono à `self.sample_rate` Hz
        (24 kHz). Lève `SynthesisError` si le texte est vide ou la voix
        inconnue — mlx-audio, lui, ignore en silence une voix inexistante et
        lit avec une voix par défaut, ce qu'on ne veut surtout pas pour des
        MP3 destinés à des enfants.
        """
        texte = texte.strip()
        if not texte:
            raise SynthesisError("Rien à lire : le texte est vide.")
        self._ensure_loaded()
        connues = self.voix()
        if voix not in connues:
            raise SynthesisError(
                f"Voix inconnue « {voix} ». Voix disponibles : {', '.join(connues)}."
            )
        morceaux = [
            _vers_numpy(resultat.audio)
            for resultat in self._model.generate(text=texte, voice=voix)
        ]
        morceaux = [m for m in morceaux if m.size]
        if not morceaux:
            raise SynthesisError("Le modèle n'a produit aucun son.")
        return np.concatenate(morceaux)


_partage: Synthetiseur | None = None
_partage_lock = threading.Lock()


def synthetiseur(repo: str = DEFAULT_MODEL) -> Synthetiseur:
    """Instance partagée (une par processus), reconstruite si le modèle change."""
    global _partage
    with _partage_lock:
        if _partage is None or _partage.repo != repo:
            _partage = Synthetiseur(repo)
        return _partage


def lire(texte: str, voix: str, repo: str = DEFAULT_MODEL) -> np.ndarray:
    """Texte → son (numpy float32 mono, 24 kHz), avec l'instance partagée."""
    return synthetiseur(repo).lire(texte, voix)


def tronquer(texte: str, max_chars: int = MAX_CHARS) -> tuple[str, bool]:
    """Coupe un texte trop long pour une lecture, sur une fin de mot.

    Retourne (texte, True si coupé).
    """
    texte = texte.strip()
    if len(texte) <= max_chars:
        return texte, False
    coupe = texte[:max_chars]
    espace = coupe.rfind(" ")
    if espace > max_chars // 2:
        coupe = coupe[:espace]
    return coupe.rstrip(), True


# ----------------------------------------------------------------------
# Encodage MP3 (ffmpeg)
# ----------------------------------------------------------------------

# ffmpeg n'est pas une dépendance de l'app : seul vocaliser_cahier.py s'en
# sert. Une app lancée par LaunchAgent ou Voxtral.command a un PATH minimal
# sans Homebrew, d'où les emplacements explicites.
FFMPEG_CANDIDATES = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
MP3_BITRATE = "64k"
# Léger ralentissement pour des enfants de 7 ans. `atempo` conserve la
# hauteur de la voix, contrairement à un rééchantillonnage.
TEMPO = 0.92
# Normalisation EBU R128 : volume régulier d'une phrase à l'autre, quelle que
# soit l'énergie que le modèle a mise dans sa lecture.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
FFMPEG_TIMEOUT_S = 120


def chemin_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def ffmpeg_disponible() -> bool:
    return chemin_ffmpeg() is not None


def commande_ffmpeg(ffmpeg: str, sortie: Path, sample_rate: int) -> list[str]:
    """La ligne de commande : PCM float32 mono sur stdin → MP3 mono 64 kb/s."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-ac", "1",
        # loudnorm travaille en 192 kHz et laisserait l'encodeur choisir 48 kHz :
        # on revient à la fréquence du modèle, plus efficace à 64 kb/s.
        "-ar", str(sample_rate),
        "-b:a", MP3_BITRATE,
        "-af", f"atempo={TEMPO},{LOUDNORM}",
        "-f", "mp3",
        str(sortie),
    ]


def vers_mp3(
    audio: Iterable[float] | np.ndarray,
    chemin: Path | str,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Écrit `audio` en MP3 mono 64 kb/s, volume normalisé, débit ralenti.

    Écriture atomique : ffmpeg produit `<chemin>.tmp`, renommé à la fin. Une
    interruption ne laisse donc jamais un MP3 à moitié écrit que la reprise
    incrémentale prendrait pour terminé.
    """
    ffmpeg = chemin_ffmpeg()
    if ffmpeg is None:
        raise SynthesisError(
            "ffmpeg introuvable. Installe-le avec : brew install ffmpeg"
        )
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.name + ".tmp")

    data = np.asarray(audio, dtype=np.float32).reshape(-1).tobytes()
    try:
        result = subprocess.run(
            commande_ffmpeg(ffmpeg, tmp, sample_rate),
            input=data,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        tmp.unlink(missing_ok=True)
        raise SynthesisError(f"ffmpeg a échoué : {exc}") from exc
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-400:]
        raise SynthesisError(f"ffmpeg a échoué ({result.returncode}) : {detail}")
    tmp.replace(chemin)


# ----------------------------------------------------------------------
# Lecture au haut-parleur
# ----------------------------------------------------------------------


def jouer(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Joue le son sur la sortie par défaut et attend la fin."""
    import sounddevice as sd

    sd.play(np.asarray(audio, dtype=np.float32), sample_rate)
    sd.wait()


def arreter() -> None:
    """Interrompt une lecture en cours (sans erreur s'il n'y en a pas)."""
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:  # noqa: BLE001
        pass
