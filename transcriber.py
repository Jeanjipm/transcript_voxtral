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

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from config import Config


# Repo HuggingFace utilisé quand Voxtral est indisponible (paquet MLX absent).
WHISPER_FALLBACK_REPO = "mlx-community/whisper-large-v3-turbo"

# Modèle utilisé pour la traduction (via Voxtral → délégation) : turbo est
# distillé pour la transcription uniquement et retourne la langue source
# au lieu d'anglais. large-v3 (non-turbo) supporte le vrai task="translate".
# ~3 Go au premier usage translate, téléchargé lazy.
WHISPER_TRANSLATE_REPO = "mlx-community/whisper-large-v3-mlx"


def _is_hf_model_cached(repo_id: str) -> bool:
    """Best-effort : True si le repo semble déjà présent dans le cache HF.

    On s'appuie sur `config.json` comme sentinelle (présent dans à peu près
    tous les repos MLX/transformers). Si huggingface_hub est trop vieux pour
    exposer l'API, on renvoie True (on évite une fausse notif — le pire
    scénario est juste un download silencieux, comme avant).
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return True
    result = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    return isinstance(result, (str, bytes))


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


class VoxtralTranscriber(Transcriber):
    """Backend Voxtral via le package `mlx_voxtral` (mzbac).

    mlx-voxtral 0.0.4 ne supporte PAS `task="translate"` — la signature de
    `apply_transcrition_request()` ne prend que `audio`, `language`, et
    `sampling_rate`, et le code émet toujours un token `[TRANSCRIBE]`. On
    délègue donc à Whisper (qui supporte la traduction→anglais nativement)
    quand l'utilisateur choisit translate.
    """

    def __init__(
        self,
        model_repo: str,
        on_model_download: Callable[[str], None] | None = None,
    ) -> None:
        self.model_repo = model_repo
        self._model = None
        self._processor = None
        self._whisper_for_translate: "WhisperTranscriber | None" = None
        # Callback appelé avec le repo_id quand un modèle n'est pas en cache
        # HF et va être téléchargé. Permet à l'app de notifier l'utilisateur
        # (téléchargement = plusieurs Go, plusieurs minutes).
        self._on_model_download = on_model_download

    def is_available(self) -> bool:
        try:
            import mlx_voxtral  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Prévenir AVANT `from_pretrained` (qui télécharge sync et bloque).
        if (
            self._on_model_download is not None
            and not _is_hf_model_cached(self.model_repo)
        ):
            self._on_model_download(self.model_repo)
        # Import retardé : ne charge mlx (lourd) qu'à la première transcription.
        from mlx_voxtral import (  # type: ignore[import-not-found]
            VoxtralForConditionalGeneration,
            VoxtralProcessor,
        )

        self._model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_repo
        )
        self._processor = VoxtralProcessor.from_pretrained(self.model_repo)

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
                self._whisper_for_translate = WhisperTranscriber(
                    WHISPER_TRANSLATE_REPO,
                    on_model_download=self._on_model_download,
                )
            return self._whisper_for_translate.transcribe(
                wav_path, language=language, task="translate",
                max_new_tokens=max_new_tokens,
            )

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        # NB : la méthode upstream s'écrit bien "transcrition" (typo du
        # package mzbac).
        inputs = self._processor.apply_transcrition_request(
            language=language,
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

    def __init__(
        self,
        model_repo: str,
        on_model_download: Callable[[str], None] | None = None,
    ) -> None:
        self.model_repo = model_repo
        self._on_model_download = on_model_download
        self._cache_checked = False

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
        # Prévenir AVANT le 1er `mlx_whisper.transcribe` (qui télécharge sync
        # via HF Hub si le modèle n'est pas en cache). On ne check qu'une
        # fois — les appels suivants utilisent forcément le cache.
        if not self._cache_checked:
            self._cache_checked = True
            if (
                self._on_model_download is not None
                and not _is_hf_model_cached(self.model_repo)
            ):
                self._on_model_download(self.model_repo)

        import mlx_whisper  # type: ignore[import-not-found]
        import soundfile as sf

        # mlx-whisper utilise ffmpeg pour décoder un fichier audio depuis un
        # chemin. Pour éviter cette dep système, on charge le WAV via
        # soundfile et on passe un numpy array (AudioRecorder enregistre
        # déjà en 16 kHz mono, exactement ce qu'attend Whisper).
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16_000, f"Whisper attend 16 kHz, pas {sr} Hz"

        # Whisper utilise None pour la détection automatique
        whisper_lang = None if language == "auto" else language

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            language=whisper_lang,
            task=task,
        )
        return str(result.get("text", "")).strip()


def make_transcriber(
    config: Config,
    on_model_download: Callable[[str], None] | None = None,
) -> Transcriber:
    """
    Factory : choisit le backend selon le nom du modèle dans la config.
    Tombe en fallback sur Whisper si Voxtral indisponible.

    Si un modèle Voxtral est configuré mais que `mlx_voxtral` n'est pas
    importable, on bascule sur Whisper Turbo et on log un avertissement
    explicite sur stderr (sinon l'utilisateur ne comprend pas pourquoi sa
    config est ignorée).

    `on_model_download(repo_id)` (optionnel) est appelé juste avant qu'un
    modèle absent du cache HF ne soit téléchargé — permet à l'app de prévenir
    l'utilisateur. Propagé au backend de traduction Whisper instancié lazy.
    """
    model_name = config.model.name

    if "whisper" in model_name.lower():
        return WhisperTranscriber(model_name, on_model_download=on_model_download)

    voxtral = VoxtralTranscriber(
        model_name,
        on_model_download=on_model_download,
    )
    if voxtral.is_available():
        return voxtral

    print(
        f"[transcriber] AVERTISSEMENT : paquet 'mlx_voxtral' introuvable, "
        f"modèle '{model_name}' non utilisable. "
        f"Fallback sur Whisper ({WHISPER_FALLBACK_REPO}). "
        f"Installe mlx-voxtral ou choisis un modèle Whisper dans Préférences.",
        file=sys.stderr,
    )
    return WhisperTranscriber(
        WHISPER_FALLBACK_REPO, on_model_download=on_model_download
    )
