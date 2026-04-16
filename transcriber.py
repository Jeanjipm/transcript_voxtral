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

from abc import ABC, abstractmethod
from pathlib import Path

from config import Config


class Transcriber(ABC):
    """Interface commune à tous les backends de transcription."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        task: str = "transcribe",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
    ) -> str:
        """Retourne le texte transcrit (chaîne UTF-8, espaces nettoyés)."""

    @abstractmethod
    def is_available(self) -> bool:
        """True si le backend peut être utilisé (modèle + lib OK)."""


class VoxtralTranscriber(Transcriber):
    """
    Backend Voxtral via le package `mlx_voxtral` (mzbac).
    """

    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo
        self._model = None
        self._processor = None

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

        self._model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_repo
        )
        self._processor = VoxtralProcessor.from_pretrained(self.model_repo)

    def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        task: str = "transcribe",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
    ) -> str:
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        # mlx-voxtral attend un code langue type "fr"/"en" ; "auto" est
        # géré côté processor sur les versions récentes — on passe tel quel.
        # NB : la méthode upstream s'écrit bien "transcrition" (typo du package mzbac).
        inputs = self._processor.apply_transcrition_request(
            language=language,
            audio=str(wav_path),
            task=task,
        )
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
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
        temperature: float = 0.0,
        max_new_tokens: int = 1024,  # noqa: ARG002 — non utilisé par Whisper
    ) -> str:
        import mlx_whisper  # type: ignore[import-not-found]

        # Whisper utilise None pour la détection automatique
        whisper_lang = None if language == "auto" else language

        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=self.model_repo,
            language=whisper_lang,
            task=task,
            temperature=temperature,
        )
        return str(result.get("text", "")).strip()


def make_transcriber(config: Config) -> Transcriber:
    """
    Factory : choisit le backend selon le nom du modèle dans la config.
    Tombe en fallback sur Whisper si Voxtral indisponible.
    """
    model_name = config.model.name

    if "whisper" in model_name.lower():
        return WhisperTranscriber(model_name)

    voxtral = VoxtralTranscriber(model_name)
    if voxtral.is_available():
        return voxtral

    # Fallback structuré : Voxtral indisponible → Whisper Turbo
    fallback_repo = "mlx-community/whisper-large-v3-turbo"
    return WhisperTranscriber(fallback_repo)
