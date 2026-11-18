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

import re
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
# Depuis que la lecture se fait phrase par phrase (cf. lecture.py), la limite
# ne vient plus du modèle mais du bon sens : 20 000 caractères font une
# vingtaine de minutes d'écoute. Un presse-papier peut contenir un livre.
MAX_CHARS = 20_000
# Taille d'un morceau lu d'un seul tenant. Assez long pour que la voix garde
# son élan, assez court pour que le premier son arrive vite et que le budget
# de génération (cf. `budget_tokens` : 600 caractères tiennent en ~128 s sur
# les 160 du plafond) ne soit jamais approché.
PHRASE_MAX_CHARS = 300
# En dessous, un morceau est collé au suivant.
PHRASE_MIN_CHARS = 25
# Budget de génération. Le modèle produit 12,5 trames audio par seconde et
# s'arrête normalement tout seul ; mais il lui arrive, au hasard de
# l'échantillonnage, de ne jamais émettre sa fin et de dérouler jusqu'au
# plafond par défaut de mlx-audio (4 096 trames ≈ 5 min 30), où le décodage
# audio a tenté d'allouer 32 Go et tué le processus (constaté sur une phrase
# de 38 caractères). On plafonne donc selon la longueur du texte — une
# lecture lente fait ~8 caractères par seconde, on prend large — et une
# génération qui touche son plafond est tenue pour « partie en boucle » :
# on la rejoue, puis on renonce avec une erreur propre.
# Débit mesuré sur le cahier : de 5,6 caractères/s (phrase courte, pauses)
# à 15 (phrase longue) — on prend le plus lent, plus une marge fixe.
FRAMES_PAR_SECONDE = 12.5
CHARS_PAR_SECONDE = 5.0
MARGE_SECONDES = 8.0
MAX_TOKENS_PLAFOND = 2_000  # 160 s ; le plus long texte du cahier (600 car.) tient en ~128 s
TENTATIVES = 3
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


def budget_tokens(texte: str) -> int:
    """Nombre maximal de trames audio à générer pour ce texte."""
    secondes = len(texte) / CHARS_PAR_SECONDE + MARGE_SECONDES
    return min(MAX_TOKENS_PLAFOND, int(secondes * FRAMES_PAR_SECONDE))


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
        max_tokens = budget_tokens(texte)
        for tentative in range(1, TENTATIVES + 1):
            morceaux: list[np.ndarray] = []
            trames = 0
            for resultat in self._model.generate(
                text=texte, voice=voix, max_tokens=max_tokens
            ):
                morceaux.append(_vers_numpy(resultat.audio))
                trames += int(getattr(resultat, "token_count", 0) or 0)
            if trames < max_tokens:
                break
            print(
                f"[synthese] lecture partie en boucle ({trames} trames, "
                f"tentative {tentative}/{TENTATIVES}) : « {texte[:60]} »"
            )
        else:
            raise SynthesisError(
                f"Le modèle n'arrive pas à finir cette phrase après {TENTATIVES} "
                f"tentatives : « {texte[:80]} »"
            )
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


# Une phrase : du premier caractère jusqu'à une ponctuation forte (avec son
# guillemet ou sa parenthèse fermante) suivie d'une espace, ou jusqu'à la fin
# de la ligne. « 12.5 » ou « kixit.fr » ne coupent donc pas.
# Le guillemet français fermant est précédé d'une espace (« non. ») : elle
# ne doit pas couper la phrase en laissant le guillemet orphelin.
_PHRASE = re.compile(r"\S.*?(?:[.!?…]+(?:\s?»|[\"')\]])*(?=\s|$)|$)")
# Où recouper une phrase trop longue, du meilleur endroit au pire.
_COUPURES = (re.compile(r"(?<=[;:])\s+"), re.compile(r"(?<=,)\s+"), re.compile(r"\s+"))


def decouper_en_phrases(
    texte: str,
    max_chars: int = PHRASE_MAX_CHARS,
    min_chars: int = PHRASE_MIN_CHARS,
) -> list[str]:
    """Découpe un texte en morceaux à lire l'un après l'autre.

    C'est ce qui permet à la lecture de démarrer dès la première phrase, et
    à une phrase ratée de ne pas emporter tout le texte (cf. lecture.py).

    - On coupe aux fins de phrase et aux retours à la ligne.
    - Un morceau trop court est collé au suivant : « M. » ou « Oui. » lus
      seuls sonnent faux, et le modèle déraille sur les textes minuscules.
    - Un morceau trop long est recoupé, de préférence à un point-virgule,
      sinon à une virgule, sinon entre deux mots : le budget de génération
      du modèle est borné (cf. `budget_tokens`).
    """
    bruts: list[str] = []
    for ligne in texte.splitlines():
        ligne = " ".join(ligne.split())
        if ligne:
            bruts.extend(_PHRASE.findall(ligne))

    phrases: list[str] = []
    attente = ""
    for brut in bruts:
        candidat = f"{attente} {brut}" if attente else brut
        if len(candidat) < min_chars:
            attente = candidat
            continue
        attente = ""
        phrases.extend(_recouper(candidat, max_chars))
    if attente:
        # Un reste trop court en fin de texte : on le rattache à ce qui
        # précède si ça tient, sinon il est lu seul.
        if phrases and len(phrases[-1]) + 1 + len(attente) <= max_chars:
            phrases[-1] = f"{phrases[-1]} {attente}"
        else:
            phrases.append(attente)
    return phrases


def _recouper(phrase: str, max_chars: int) -> list[str]:
    """Coupe une phrase plus longue que `max_chars` en morceaux qui tiennent."""
    morceaux: list[str] = []
    reste = phrase
    while len(reste) > max_chars:
        coupe = -1
        for motif in _COUPURES:
            candidats = [
                m.start() for m in motif.finditer(reste, 0, max_chars + 1)
                # Pas de miette en tête : on ne coupe pas dans le premier tiers.
                if m.start() >= max_chars // 3
            ]
            if candidats:
                coupe = candidats[-1]
                break
        if coupe < 0:
            # Un « mot » plus long que la limite (une URL) : coupe franche.
            coupe = max_chars
        morceaux.append(reste[:coupe].rstrip())
        reste = reste[coupe:].lstrip()
    if reste:
        morceaux.append(reste)
    return morceaux


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


def demarrer(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Lance le son sur la sortie par défaut et rend la main tout de suite."""
    import sounddevice as sd

    sd.play(np.asarray(audio, dtype=np.float32), sample_rate)


def attendre() -> None:
    """Attend la fin du son lancé par `demarrer` (ou son arrêt par `arreter`)."""
    import sounddevice as sd

    sd.wait()


def jouer(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Joue le son sur la sortie par défaut et attend la fin."""
    demarrer(audio, sample_rate)
    attendre()


class SortieFlux:
    """Joue un son qui arrive par morceaux, à mesure qu'il est produit.

    Pour un moteur qui génère en continu (Pocket TTS, cf. synthese_pocket.py) :
    le premier morceau part au haut-parleur dès qu'il existe, sans attendre la
    fin de la phrase. Même contrat que `demarrer` / `attendre` / `arreter`,
    pour que `lecture.Lecture` s'en serve sans rien savoir du moteur.

    Tout ce qui touche à la sortie audio se passe sur un seul fil (le nôtre) :
    `arreter()` ne fait que lever un drapeau, lu entre deux morceaux de 80 ms.
    Fermer un flux PortAudio depuis un autre fil que celui qui y écrit est une
    source connue de blocages.
    """

    def __init__(self) -> None:
        # Posé par `arreter()`. Public : le moteur le reçoit aussi, pour
        # cesser de générer et pas seulement de jouer.
        self.arret = threading.Event()
        self._thread: threading.Thread | None = None
        self._erreur: BaseException | None = None

    def demarrer(self, morceaux: Iterable[np.ndarray], sample_rate: int) -> None:
        """Commence à jouer `morceaux` et rend la main tout de suite."""
        self._erreur = None
        self._thread = threading.Thread(
            target=self._jouer, args=(morceaux, sample_rate),
            name="speech-stream", daemon=True,
        )
        self._thread.start()

    def _jouer(self, morceaux: Iterable[np.ndarray], sample_rate: int) -> None:
        try:
            import sounddevice as sd

            with sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="float32"
            ) as stream:
                for morceau in morceaux:
                    if self.arret.is_set():
                        break
                    audio = np.asarray(morceau, dtype=np.float32).reshape(-1, 1)
                    if audio.size:
                        stream.write(audio)
                if self.arret.is_set():
                    # Couper net : sans ça la fermeture du flux attendrait la
                    # fin de ce qui est déjà dans le tampon.
                    stream.abort()
        except BaseException as exc:  # noqa: BLE001 — rendue à `attendre()`
            self._erreur = exc

    def attendre(self) -> None:
        """Attend la fin du son (ou son arrêt). Relève l'erreur du moteur ou
        de la sortie audio, s'il y en a eu une."""
        if self._thread is not None:
            self._thread.join()
        if self._erreur is not None and not self.arret.is_set():
            raise self._erreur

    def arreter(self) -> None:
        self.arret.set()


def arreter() -> None:
    """Interrompt une lecture en cours (sans erreur s'il n'y en a pas)."""
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:  # noqa: BLE001
        pass
