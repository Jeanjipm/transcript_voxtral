"""
Transcription audio → texte.

Architecture : `Transcriber` abstrait + 2 implémentations concrètes :
- `VoxtralTranscriber` (par défaut, via `mlx_voxtral`)
- `WhisperTranscriber` (fallback libre de droits, via `mlx_whisper`)

Le modèle est chargé en mémoire au premier `transcribe()` puis réutilisé
(évite de retélécharger / reparser ~3 Go à chaque dictée).

Pourquoi un abstrait ? Si la licence Voxtral devient bloquante, ou si
mlx-voxtral est en panne, on swap d'une seule ligne (`make_transcriber`).
"""

from __future__ import annotations

import os
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import hf_offline
from config import Config


AUDIO_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Segment:
    """Un passage transcrit, horodaté en secondes depuis le début de l'audio."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Résultat d'une transcription : le texte, et son découpage si connu.

    `segments` est vide pour les backends sans horodatage (Voxtral) ; les
    appelants qui en ont besoin — la transcription de fichiers longs, la
    diarisation — doivent gérer ce cas.
    """

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None


# Modèle Whisper utilisé pour :
#   1. fallback quand mlx-voxtral est introuvable (paquet MLX absent)
#   2. délégation de traduction depuis Voxtral (mlx-voxtral 0.0.4 ne
#      supporte pas task="translate")
#   3. choix explicite utilisateur dans Préférences → Modèle
# On prend le large-v3 non-distillé (≠ turbo) car turbo est distillé pour
# la transcription uniquement et retourne la langue source au lieu
# d'anglais pour task="translate".
WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"


class Transcriber(ABC):
    """Interface commune à tous les backends de transcription."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        task: str = "transcribe",
        max_new_tokens: int = 1024,
    ) -> str:
        """Retourne le texte transcrit (chaîne UTF-8, espaces nettoyés)."""

    @abstractmethod
    def is_available(self) -> bool:
        """True si le backend peut être utilisé (modèle + lib OK)."""

    def preload(self) -> None:
        """Charge le modèle en mémoire de façon proactive.

        Appelé au lancement de l'app via l'inference-worker, pour que la 1re
        transcription ne paye pas le coût (5-15s pour un modèle MLX 3B).
        Implémentation par défaut : no-op — surchargée par les backends qui
        supportent le pré-chargement.
        """
        return None

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
        max_new_tokens: int = 1024,
    ) -> TranscriptionResult:
        """Transcrit un tableau audio et retourne texte + segments horodatés.

        Implémentation par défaut, pour les backends sans horodatage : on écrit
        un WAV temporaire, on appelle `transcribe()`, et on retourne UN segment
        couvrant tout le bloc. Ça permet à la transcription de fichiers longs
        de fonctionner avec n'importe quel backend — simplement sans découpage
        interne, donc avec des blocs qu'il faut garder courts.

        `initial_prompt` est ignoré par défaut (les backends qui savent s'en
        servir surchargent cette méthode).
        """
        if audio.size == 0:
            return TranscriptionResult(text="", segments=[], language=language)

        duration = audio.shape[0] / float(sample_rate)
        fd, path_str = tempfile.mkstemp(suffix=".wav", prefix="voxtral_block_")
        os.close(fd)
        wav_path = Path(path_str)
        try:
            import soundfile as sf

            sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
            text = self.transcribe(
                wav_path,
                language=language or "auto",
                task=task,
                max_new_tokens=max_new_tokens,
            )
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

        segments = (
            [Segment(start=0.0, end=duration, text=text)] if text.strip() else []
        )
        return TranscriptionResult(
            text=text, segments=segments, language=language
        )


class VoxtralTranscriber(Transcriber):
    """Backend Voxtral via le package `mlx_voxtral` (mzbac).

    mlx-voxtral 0.0.4 ne supporte PAS `task="translate"` — la signature de
    `apply_transcrition_request()` ne prend que `audio`, `language`, et
    `sampling_rate`, et le code émet toujours un token `[TRANSCRIBE]`. On
    délègue donc à Whisper (qui supporte la traduction→anglais nativement)
    quand l'utilisateur choisit translate.
    """

    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo
        self._model = None
        self._processor = None
        self._whisper_for_translate: "WhisperTranscriber | None" = None

    def is_available(self) -> bool:
        try:
            import mlx_voxtral  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Import retardé : ne charge mlx (lourd) qu'à la première transcription.
        from mlx_voxtral import (  # type: ignore[import-not-found]
            VoxtralForConditionalGeneration,
            VoxtralProcessor,
        )

        # On passe le chemin du snapshot local plutôt que l'identifiant du
        # repo quand le modèle est en cache : sinon mistral_common appelle
        # list_repo_files() sans condition et le chargement échoue dès qu'il
        # n'y a pas de réseau, alors que tout est sur le disque
        # (cf. hf_offline.resolve_local_path).
        source = hf_offline.resolve_local_path(self.model_repo)

        self._model = VoxtralForConditionalGeneration.from_pretrained(source)
        self._processor = VoxtralProcessor.from_pretrained(source)

    def preload(self) -> None:
        """Charge le modèle Voxtral en mémoire dès le démarrage."""
        self._ensure_loaded()

    def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        task: str = "transcribe",
        max_new_tokens: int = 1024,
    ) -> str:
        if task == "translate":
            # Voxtral Mini ne sait pas traduire via mlx-voxtral 0.0.4.
            # On délègue à Whisper large-v3 (le turbo est distillé pour la
            # transcription uniquement et retourne la langue source).
            if self._whisper_for_translate is None:
                self._whisper_for_translate = WhisperTranscriber(WHISPER_REPO)
            return self._whisper_for_translate.transcribe(
                wav_path, language=language, task="translate",
                max_new_tokens=max_new_tokens,
            )

        self._ensure_loaded()
        if self._model is None or self._processor is None:
            raise RuntimeError("Modèle Voxtral non chargé.")

        # "auto" doit devenir None : apply_transcrition_request ne saute le
        # bloc de langue que si l'argument est None, sinon il tokenise
        # littéralement « lang:auto » dans le prompt et pollue la sortie
        # (cf. processing_voxtral.py ~ligne 265).
        voxtral_lang = None if language in (None, "", "auto") else language

        # NB : la méthode upstream s'écrit bien "transcrition" (typo du
        # package mzbac).
        inputs = self._processor.apply_transcrition_request(
            language=voxtral_lang,
            audio=str(wav_path),
        )
        # mlx-voxtral retourne un TranscriptionInputs (objet, pas dict) ;
        # `**inputs` échoue avec "must be a mapping". vars() déballe le
        # __dict__ de l'objet.
        outputs = self._model.generate(
            **vars(inputs),
            max_new_tokens=max_new_tokens,
        )
        text = self._processor.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return text.strip()


class WhisperTranscriber(Transcriber):
    """
    Backend Whisper (mlx-whisper). Sert de fallback libre de droits.
    """

    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo

    def is_available(self) -> bool:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        task: str = "transcribe",
        max_new_tokens: int = 1024,  # noqa: ARG002 — non utilisé par Whisper
    ) -> str:
        import mlx_whisper  # type: ignore[import-not-found]
        import soundfile as sf

        # mlx-whisper utilise ffmpeg pour décoder un fichier audio depuis un
        # chemin. Pour éviter cette dep système, on charge le WAV via
        # soundfile et on passe un numpy array (AudioRecorder enregistre
        # déjà en 16 kHz mono, exactement ce qu'attend Whisper).
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != AUDIO_SAMPLE_RATE:
            # Un `assert` disparaîtrait sous `-O` et ne porte aucun message
            # utile. Les fichiers arbitraires doivent passer par
            # audio_convert.ensure_16k_mono avant d'arriver ici.
            raise ValueError(
                f"Whisper attend un audio à {AUDIO_SAMPLE_RATE} Hz, "
                f"reçu {sr} Hz."
            )

        # Whisper utilise None pour la détection automatique
        whisper_lang = None if language in (None, "", "auto") else language

        result = mlx_whisper.transcribe(
            audio,
            # Même raison que pour Voxtral : un chemin local évite tout appel
            # réseau (cf. hf_offline.resolve_local_path).
            path_or_hf_repo=hf_offline.resolve_local_path(self.model_repo),
            language=whisper_lang,
            task=task,
        )
        return str(result.get("text", "")).strip()

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
        max_new_tokens: int = 1024,  # noqa: ARG002 — non utilisé par Whisper
    ) -> TranscriptionResult:
        """Transcrit un tableau audio en exploitant le long-form de Whisper.

        Contrairement à l'implémentation par défaut, on récupère ici de vrais
        segments horodatés : `mlx_whisper` découpe nativement par fenêtres de
        30 s et rend `result["segments"]` avec `start`/`end`. C'est ce qui rend
        possibles la transcription de fichiers longs et la diarisation.

        On passe l'array directement plutôt qu'un chemin : `mlx_whisper`
        décode les chemins via ffmpeg, absent de la machine.
        """
        import mlx_whisper  # type: ignore[import-not-found]

        if audio.size == 0:
            return TranscriptionResult(text="", segments=[], language=language)
        if sample_rate != AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"Whisper attend un audio à {AUDIO_SAMPLE_RATE} Hz, "
                f"reçu {sample_rate} Hz."
            )

        whisper_lang = None if language in (None, "", "auto") else language

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=hf_offline.resolve_local_path(self.model_repo),
            language=whisper_lang,
            task=task,
            initial_prompt=initial_prompt,
        )

        segments = [
            Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
            )
            for seg in result.get("segments", [])
            if str(seg.get("text", "")).strip()
        ]
        return TranscriptionResult(
            text=str(result.get("text", "")).strip(),
            segments=segments,
            language=result.get("language") or whisper_lang,
        )


def make_transcriber(config: Config) -> Transcriber:
    """
    Factory : choisit le backend selon le nom du modèle dans la config.
    Tombe en fallback sur Whisper si Voxtral indisponible.

    Si un modèle Voxtral est configuré mais que `mlx_voxtral` n'est pas
    importable, on bascule sur Whisper Turbo et on log un avertissement
    explicite sur stderr (sinon l'utilisateur ne comprend pas pourquoi sa
    config est ignorée).
    """
    model_name = config.model.name

    if "whisper" in model_name.lower():
        return WhisperTranscriber(model_name)

    voxtral = VoxtralTranscriber(model_name)
    if voxtral.is_available():
        return voxtral

    print(
        f"[transcriber] AVERTISSEMENT : paquet 'mlx_voxtral' introuvable, "
        f"modèle '{model_name}' non utilisable. "
        f"Fallback sur Whisper ({WHISPER_REPO}). "
        f"Installe mlx-voxtral ou choisis un modèle Whisper dans Préférences.",
        file=sys.stderr,
    )
    return WhisperTranscriber(WHISPER_REPO)
