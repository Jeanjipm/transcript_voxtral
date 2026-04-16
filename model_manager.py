"""
Gestion des modèles : téléchargement / mise à jour depuis HuggingFace.

Stockage : ~/.voxtral/models/<repo_id>/  (snapshot complet du repo HF).

On utilise `huggingface_hub.snapshot_download` qui gère le cache, le
déduplication, la reprise, et les hashs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download


# Catalogue des modèles connus, exposé dans l'UI Préférences.
# Source : brief technique v0.2 § F3 (Voxtral) et § Fallback (Whisper).
@dataclass(frozen=True)
class ModelInfo:
    repo_id: str
    label: str
    size_gb: float
    description: str


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        repo_id="mzbac/voxtral-mini-3b-4bit-mixed",
        label="Voxtral Mini 3B (4-bit)",
        size_gb=3.2,
        description="Rapide, recommandé. Quantification mixte 4-bit.",
    ),
    ModelInfo(
        repo_id="mzbac/voxtral-mini-3b-8bit",
        label="Voxtral Mini 3B (8-bit)",
        size_gb=5.3,
        description="Qualité supérieure, plus lourd.",
    ),
    ModelInfo(
        repo_id="mistralai/Voxtral-Mini-3B-2507",
        label="Voxtral Mini 3B (full)",
        size_gb=8.0,
        description="Qualité maximale, nécessite plus de RAM.",
    ),
    ModelInfo(
        repo_id="mlx-community/whisper-large-v3-turbo",
        label="Whisper Large V3 Turbo",
        size_gb=1.6,
        description="Fallback libre de droits (MIT).",
    ),
]


def list_available_models() -> list[ModelInfo]:
    return list(AVAILABLE_MODELS)


def find_model(repo_id: str) -> ModelInfo | None:
    for m in AVAILABLE_MODELS:
        if m.repo_id == repo_id:
            return m
    return None


def model_local_path(repo_id: str, models_root: Path) -> Path:
    """Chemin local attendu pour un modèle (qu'il soit téléchargé ou non)."""
    safe = repo_id.replace("/", "_")
    return models_root.expanduser() / safe


def is_downloaded(repo_id: str, models_root: Path) -> bool:
    path = model_local_path(repo_id, models_root)
    if not path.exists():
        return False
    # Heuristique simple : au moins un fichier non vide dans le dossier.
    for p in path.rglob("*"):
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False


def download_model(
    repo_id: str,
    models_root: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Télécharge (ou met à jour) un modèle HuggingFace dans
    `models_root/<repo_safe>/`. Retourne le chemin local.

    `progress_callback(current_bytes, total_bytes)` est invoqué périodiquement
    si fourni — pour brancher une barre de progression dans l'UI.
    Note : `snapshot_download` gère lui-même la progression via `tqdm` ;
    le callback ici est best-effort (HF Hub ne propose pas d'API officielle
    de callback fin par fichier).
    """
    dest = model_local_path(repo_id, models_root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    if progress_callback is not None:
        # Best-effort : on signale juste la fin (100%).
        progress_callback(1, 1)
    return Path(local_path)
