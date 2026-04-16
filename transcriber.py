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

from config import Config


# Repo HuggingFace utilisé quand Voxtral est indisponible (paquet MLX absent).
WHISPER_FALLBACK_REPO = "mlx-community/whisper-large-v3-turbo"


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
        # L'argument task n'est PAS supporté par mlx-voxtral 0.0.4 ; pour la
        # traduction, bascule sur Whisper dans Préférences (cf. task != "transcribe").
        if task != "transcribe":
            print(
                f"[transcriber] task={task!r} non supporté par mlx-voxtral 0.0.4, "
                f"fallback sur transcribe. Bascule sur un modèle Whisper dans "
                f"Préférences pour activer la traduction.",
                file=sys.stderr,
            )
        inputs = self._processor.apply_transcrition_request(
            language=language,
            audio=str(wav_path),
        )
        # mlx-voxtral 0.0.4 retourne un TranscriptionInputs (objet, pas dict) ;
        # `**inputs` échoue avec "must be a mapping". On déballe via vars()
        # qui fonctionne pour les classes ordinaires avec __dict__.
        outputs = self._model.generate(
            **vars(inputs),
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
        f"Fallback sur Whisper ({WHISPER_FALLBACK_REPO}). "
        f"Installe mlx-voxtral ou choisis un modèle Whisper dans Préférences.",
        file=sys.stderr,
    )
    return WhisperTranscriber(WHISPER_FALLBACK_REPO)
