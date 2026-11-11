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
from typing import Any, Callable, Iterable

import numpy as np

import hf_offline


DEFAULT_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
# Fréquence d'échantillonnage de sortie du modèle (fixe, cf. son config.json).
SAMPLE_RATE = 24_000
# Garde-fou pour « Lire le texte copié » : au-delà, on lit le début seulement.
# Un presse-papier peut contenir un article entier, et la génération est de
# toute façon plafonnée à MAX_TOKENS_PLAFOND trames (160 s de son).
MAX_CHARS = 2_000
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
# Second garde-fou, pour les lectures qui s'arrêtent d'elles-mêmes mais en
# charabia : un mot seul (« oiseau », « heure ») sort normalement en 0,4 à
# 0,7 s, et une fois sur dix environ en 2 à 6 s de syllabes inventées. Le
# débit le plus lent mesuré sur une vraie phrase est 5,6 caractères/s ; au-delà
# de ce maximum (texte lu à 4 car./s, plus 0,3 s), la lecture n'est pas
# plausible et on la rejoue. Pour 5 lettres : 1,55 s ; pour 38 : 9,8 s.
# Un chiffre se lit comme un mot entier (« 60 » → « soixante ») : il compte
# pour quatre caractères, sinon les listes de nombres du cahier
# (« 54 + … = 60 6, 4, 5 », lue en 6,8 s) passaient pour du charabia.
CHARS_PAR_SECONDE_MIN = 4.0
MARGE_DUREE_S = 0.3
POIDS_CHIFFRE = 4
# Décodage du son par tranches. Le décodeur audio de mlx-audio calcule, pour N
# trames, des tableaux d'attention de 8 × (8 N)² nombres : la mémoire croît
# comme le CARRÉ de la durée. Mesuré : 2,8 Go pour 40 s, 12 Go pour 2 min,
# 21 Go pour 160 s (notre plafond de 2 000 trames), et à 4 096 trames un seul
# tableau de 32 Gio que Metal refuse — le plantage des générations en série.
# Le plafond de génération ci-dessus évite les 32 Gio, mais 21 Go restent
# hors de portée d'un Mac de 8 ou 16 Go. Or le
# décodeur ne regarde que le passé proche (attention à fenêtre glissante,
# convolutions causales) : on peut donc décoder par tranches, chacune précédée
# de quelques trames déjà décodées qui servent de contexte et dont on jette le
# son. Mesuré sur les codes de vraies lectures de 23 à 41 s : l'influence du
# début de tranche s'éteint en 16 à 24 trames ; au-delà, le son recollé reste
# à ~60 dB sous le signal de celui du décodage d'un bloc, écart qui ne diminue
# pas quand on élargit le contexte (ce n'est donc pas un défaut de raccord) et
# qui est 15 à 25 dB plus faible que l'erreur d'arrondi du modèle lui-même
# (livré en demi-précision, comparé au même en simple précision). Whisper
# entend le même texte des deux façons.
# La mémoire ne dépend plus de la durée : ~1,7 Go par tranche.
# Une lecture de 20 s ou moins — presque tout le cahier — tient en une tranche
# et suit exactement le même chemin qu'avant.
TRANCHE_TRAMES = 250  # 20 s de son
CONTEXTE_TRAMES = 32
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


def duree_max_plausible(texte: str) -> float:
    """Au-delà de cette durée (en secondes), la lecture est du charabia."""
    chiffres = sum(c.isdigit() for c in texte)
    longueur = len(texte) + chiffres * (POIDS_CHIFFRE - 1)
    return longueur / CHARS_PAR_SECONDE_MIN + MARGE_DUREE_S


def decoder_par_tranches(
    decode: Callable[[Any], Any],
    codes: Any,
    tranche: int = TRANCHE_TRAMES,
    contexte: int = CONTEXTE_TRAMES,
) -> Any:
    """Décode `codes` (1, trames, 37) en son (1, échantillons), par tranches.

    `decode` est le décodeur d'origine de mlx-audio. Jusqu'à `tranche` trames,
    on l'appelle une seule fois, comme avant. Au-delà, chaque tranche est
    décodée avec les `contexte` trames qui la précèdent, dont le son est jeté :
    la mémoire reste celle d'une tranche, quelle que soit la durée.
    """
    total = codes.shape[1]
    if total <= tranche:
        return decode(codes)
    import mlx.core as mx  # type: ignore[import-not-found]

    morceaux = []
    for debut in range(0, total, tranche):
        depart = max(0, debut - contexte)
        fin = min(total, debut + tranche)
        onde = decode(codes[:, depart:fin, :])
        par_trame = onde.shape[1] // (fin - depart)
        onde = onde[:, (debut - depart) * par_trame :]
        # Calculé tout de suite : MLX est paresseux, et sans cela toutes les
        # tranches attendraient en mémoire le calcul final.
        mx.eval(onde)
        morceaux.append(onde)
    return mx.concatenate(morceaux, axis=1)


def appliquer_decodage_par_tranches(model: Any) -> bool:
    """Borne la mémoire du décodeur audio de `model`. Retourne True si posé."""
    tokenizer = getattr(model, "audio_tokenizer", None)
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return False

    def decode_borne(codes: Any) -> Any:
        return decoder_par_tranches(decode, codes)

    tokenizer.decode = decode_borne
    return True


def _est_mlx(tableau: Any) -> bool:
    return type(tableau).__module__.split(".")[0] == "mlx"


def _vers_numpy(audio: Any) -> np.ndarray:
    """Tableau MLX (ou autre) → numpy float32 mono, à plat."""
    if _est_mlx(audio):
        # MLX ne calcule le son qu'ici. Si Metal refuse une allocation pendant
        # la conversion numpy, l'erreur ne peut pas remonter jusqu'à Python et
        # tue le processus (« abort ») ; déclenché explicitement, le même
        # calcul lève une RuntimeError ordinaire, que `lire` sait traiter.
        import mlx.core as mx  # type: ignore[import-not-found]

        mx.eval(audio)
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
        # True une fois le modèle chargé si la génération corrigée a été posée.
        self.correctif_debuts = False

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
            # Cf. le bloc « Correctif des débuts de phrase » plus bas.
            self.correctif_debuts = appliquer_correctif_debuts(model)
            # Cf. « Décodage du son par tranches » en tête de module.
            appliquer_decodage_par_tranches(model)
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
        duree_max = duree_max_plausible(texte)
        for tentative in range(1, TENTATIVES + 1):
            morceaux: list[np.ndarray] = []
            trames = 0
            try:
                for resultat in self._model.generate(
                    text=texte, voice=voix, max_tokens=max_tokens
                ):
                    morceaux.append(_vers_numpy(resultat.audio))
                    trames += int(getattr(resultat, "token_count", 0) or 0)
            except RuntimeError as exc:
                if "malloc" not in str(exc):
                    raise
                # Metal n'a pas pu fournir la mémoire demandée. Une erreur
                # propre, que le lot du cahier note avant de passer à la suite.
                raise SynthesisError(
                    f"Mémoire insuffisante pour lire ce texte : « {texte[:80]} » ({exc})"
                ) from exc
            duree = sum(m.size for m in morceaux) / self.sample_rate
            if trames < max_tokens and duree <= duree_max:
                break
            motif = (
                "partie en boucle" if trames >= max_tokens
                else f"trop longue pour le texte ({duree:.1f} s, max {duree_max:.1f} s)"
            )
            print(
                f"[synthese] lecture {motif}, tentative {tentative}/{TENTATIVES} : "
                f"« {texte[:60]} »"
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


# ----------------------------------------------------------------------
# Correctif des débuts de phrase (mlx-audio < 0.5.8)
#
# Dans mlx-audio 0.4.6, `Model.generate` glisse un jeton [AUDIO] de trop entre
# la lecture du texte et la première trame de son. Toute la séquence est
# décalée d'une position et le début de chaque phrase est abîmé : bruit,
# syllabe inventée (« Zal écrit »), premier mot avalé. Mesuré sur 16 lectures
# transcrites par Whisper : 9 débuts faux avec la 0.4.6, 0 avec la fonction
# ci-dessous. Corrigé en amont dans la 0.5.8 (5 oct. 2026), que l'on ne peut
# pas installer sans monter transformers puis mlx — donc réinstallation pour
# tous. On embarque la fonction corrigée (copie à l'identique de la 0.5.8,
# seul l'import de `make_prompt_cache` est adapté ; le reste du module est
# identique entre 0.4.6 et 0.6.4) et on la pose sur le modèle chargé, tant que
# la version installée est antérieure à la 0.5.8. Quand requirements.lock
# passera au-dessus, ce bloc cessera de s'appliquer tout seul.
# ----------------------------------------------------------------------

VERSION_CORRIGEE = (0, 5, 8)


def _version_mlx_audio() -> tuple[int, ...] | None:
    try:
        from mlx_audio.version import __version__  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    nombres = re.findall(r"\d+", __version__)
    return tuple(int(n) for n in nombres[:3]) if nombres else None


def correctif_debuts_necessaire(version: tuple[int, ...] | None = None) -> bool:
    """True si la version de mlx-audio a encore le jeton [AUDIO] de trop."""
    if version is None:
        version = _version_mlx_audio()
    return version is not None and version < VERSION_CORRIGEE


def appliquer_correctif_debuts(model: Any) -> bool:
    """Pose la génération corrigée sur `model` si nécessaire. Retourne True si
    elle a été appliquée."""
    if not correctif_debuts_necessaire():
        return False
    import types

    model.generate = types.MethodType(_generate_corrige, model)
    return True


# Copie de `Model.generate` de mlx-audio 0.5.8 (sans le jeton [AUDIO] de trop),
# rendue autonome par ses imports en tête ; le corps est inchangé.
def _generate_corrige(  # noqa: PLR0912, PLR0915
    self,
    text: str,
    voice: str = "casual_male",
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    verbose: bool = False,
    stream: bool = False,
    streaming_interval: float = 2.0,
    **kwargs,
):
    """Generate speech from text.

    Args:
        text: Input text to synthesize.
        voice: Voice preset name (e.g. "casual_male", "fr_female").
        temperature: Sampling temperature for the LM.
        top_k: Top-k sampling.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum number of audio tokens to generate.
        verbose: Show progress bar.
        stream: Enable streaming output. When True, intermediate audio
            chunks are yielded during generation for lower latency.
        streaming_interval: Approximate seconds of audio per streaming
            chunk. Each frame is 80 ms, so the interval is converted to
            a frame count (``max(1, int(streaming_interval / 0.08))``).

    Yields:
        GenerationResult with audio waveform. When *stream* is True,
        intermediate results have ``is_streaming_chunk=True`` and the
        last result additionally has ``is_final_chunk=True``.
    """
    import time

    import mlx.core as mx  # type: ignore[import-not-found]
    from mlx_audio.tts.models.base import GenerationResult  # type: ignore[import-not-found]
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore[import-not-found]
    from tqdm import tqdm

    if self.tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded. Ensure post_load_hook was called."
        )

    time_start = time.time()

    input_ids = self._encode_text(text, voice)
    input_ids_mx = mx.array(input_ids)[None, :]  # (1, seq_len)
    input_embeddings = self._build_input_embeddings(input_ids_mx, voice)

    # LM backbone returns hidden states; full model returns logits
    lm_backbone = self.language_model.model.model

    # Prefill: run prompt through LM with KV cache. The prompt ends with
    # [BEGIN_AUDIO], so its last hidden state already predicts the first
    # audio frame (as in vllm-omni). Feeding an extra [AUDIO] token here
    # shifts every frame by one position and garbles the opening words.
    cache = make_prompt_cache(self.language_model.model)
    hidden = lm_backbone(
        input_ids_mx, cache=cache, input_embeddings=input_embeddings
    )

    all_codes = []
    yielded_frames = 0
    chunk_idx = 0
    # Convert streaming_interval (seconds) to frames (1 frame = 80 ms)
    frames_per_chunk = max(1, int(streaming_interval / 0.08))
    # Context frames for overlap-add decoding.  The codec decoder uses
    # sliding-window attention with windows up to 16 (at the final stage).
    # Including context frames from the previous chunk ensures smooth
    # transitions without boundary artifacts.
    context_frames = 16
    # Each codec frame produces 1920 samples (8x upsample × 240 patch)
    samples_per_frame = 1920

    for i in tqdm(range(max_tokens), disable=not verbose):
        h = hidden[:, -1, :]  # (1, dim)

        codes = self.acoustic_transformer.decode_one_frame(h)  # (1, 37)

        # End-of-audio: semantic code == END_AUDIO special token (index 1)
        semantic_code = codes[0, 0].item()
        if semantic_code <= 1:  # 0=empty_audio, 1=end_audio
            break

        all_codes.append(codes[:, None, :])  # (1, 1, 37)

        # Embed audio codes back as LLM input for next step.
        # Each codebook has its own offset in the global embedding table:
        #   Codebook 0 (semantic): offset=0, size=semantic_codebook_size+2
        #   Codebook 1..36 (acoustic): offset=prev+prev_size, size=acoustic_codebook_size+2
        global_codes = self._codes_to_global_indices(codes)  # (1, 37)
        code_embeddings = self.audio_codebook_embeddings["embeddings"](global_codes)
        next_embedding = code_embeddings.sum(axis=1, keepdims=True)  # (1, 1, dim)

        # Feed back through LM for next step using KV cache
        dummy_input = mx.array([[self.config.audio_token_id]])
        hidden = lm_backbone(
            dummy_input, cache=cache, input_embeddings=next_embedding
        )

        if i % 50 == 0:
            mx.clear_cache()

        # Streaming: yield chunk when buffer is full
        if stream and len(all_codes) - yielded_frames >= frames_per_chunk:
            # Include context frames from earlier in the sequence so the
            # codec decoder's sliding-window attention has proper context,
            # avoiding boundary artifacts between chunks.
            ctx_start = max(0, yielded_frames - context_frames)
            chunk_codes = mx.concatenate(all_codes[ctx_start:], axis=1)
            full_waveform = self.audio_tokenizer.decode(chunk_codes)
            full_waveform = full_waveform.squeeze(0)

            # Trim the context portion — keep only new audio
            ctx_used = yielded_frames - ctx_start
            trim_samples = ctx_used * samples_per_frame
            chunk_waveform = full_waveform[trim_samples:]

            chunk_samples = chunk_waveform.shape[0]
            chunk_duration = chunk_samples / self.config.sample_rate
            chunk_time = time.time() - time_start
            chunk_token_count = len(all_codes) - yielded_frames

            yield GenerationResult(
                audio=chunk_waveform,
                sample_rate=self.config.sample_rate,
                samples=chunk_samples,
                segment_idx=chunk_idx,
                token_count=chunk_token_count,
                audio_samples={
                    "samples": chunk_samples,
                    "samples-per-sec": self.config.sample_rate,
                },
                audio_duration=self._format_duration(chunk_duration),
                real_time_factor=(
                    chunk_duration / chunk_time if chunk_time > 0 else 0
                ),
                prompt={
                    "tokens": chunk_token_count,
                    "tokens-per-sec": (
                        round(chunk_token_count / chunk_time, 2)
                        if chunk_time > 0
                        else 0
                    ),
                },
                processing_time_seconds=chunk_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=True,
                is_final_chunk=False,
            )
            yielded_frames = len(all_codes)
            chunk_idx += 1
            time_start = time.time()

    if not all_codes:
        raise RuntimeError("No audio frames generated")

    # Final chunk: decode remaining frames (or all frames if not streaming)
    remaining = len(all_codes) - yielded_frames
    if stream and yielded_frames > 0 and remaining > 0:
        # Decode remainder with context for smooth transition
        ctx_start = max(0, yielded_frames - context_frames)
        final_codes = mx.concatenate(all_codes[ctx_start:], axis=1)
        full_waveform = self.audio_tokenizer.decode(final_codes).squeeze(0)
        ctx_used = yielded_frames - ctx_start
        trim_samples = ctx_used * samples_per_frame
        waveform = full_waveform[trim_samples:]
    elif stream and yielded_frames > 0 and remaining == 0:
        # Everything already yielded — emit a zero-length final marker
        waveform = mx.zeros((0,))
    else:
        # Non-streaming (or no intermediate chunks were yielded):
        # decode everything at once — identical to the original path
        audio_codes = mx.concatenate(all_codes, axis=1)
        waveform = self.audio_tokenizer.decode(audio_codes).squeeze(0)

    time_end = time.time()

    audio_samples = waveform.shape[0]
    audio_duration = audio_samples / self.config.sample_rate

    processing_time = time_end - time_start

    yield GenerationResult(
        audio=waveform,
        sample_rate=self.config.sample_rate,
        samples=audio_samples,
        segment_idx=chunk_idx if stream else 0,
        token_count=remaining if stream and yielded_frames > 0 else len(all_codes),
        audio_samples={
            "samples": audio_samples,
            "samples-per-sec": self.config.sample_rate,
        },
        audio_duration=self._format_duration(audio_duration),
        real_time_factor=(
            audio_duration / processing_time if processing_time > 0 else 0
        ),
        prompt={
            "tokens": (
                remaining if stream and yielded_frames > 0 else len(all_codes)
            ),
            "tokens-per-sec": (
                round(
                    (remaining if stream and yielded_frames > 0 else len(all_codes))
                    / processing_time,
                    2,
                )
                if processing_time > 0
                else 0
            ),
        },
        processing_time_seconds=processing_time,
        peak_memory_usage=mx.get_peak_memory() / 1e9,
        is_streaming_chunk=stream,
        is_final_chunk=stream,
    )
    mx.clear_cache()
