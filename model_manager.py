"""
Gestion des modèles : téléchargement / mise à jour depuis HuggingFace.

Stockage : le cache HuggingFace standard (~/.cache/huggingface/hub), et lui
seul.

Pourquoi pas ~/.voxtral/models/ comme avant : `snapshot_download(local_dir=…)`
copie les fichiers dans le dossier demandé SANS peupler le cache HF, alors
que tous les `from_pretrained(repo_id)` à l'exécution résolvent via le
cache. On stockait donc chaque modèle deux fois — 8 Go de doublon constatés
sur une installation réelle — dont une copie que rien ne lisait jamais.
On s'en tient au cache : `snapshot_download` y gère déjà la déduplication
par hash, la reprise de téléchargement et les révisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

import hf_offline


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
        repo_id="mlx-community/whisper-large-v3-mlx",
        label="Whisper Large V3",
        size_gb=3.0,
        description="MIT, transcrit + traduit.",
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


def is_downloaded(repo_id: str, models_root: Path) -> bool:  # noqa: ARG001
    """True si le modèle est utilisable hors-ligne, c'est-à-dire présent dans
    le cache HuggingFace.

    On ne regarde QUE le cache, délibérément : c'est le seul endroit que
    `from_pretrained` sait lire. L'ancienne version répondait True quand le
    modèle était dans `models_root` mais absent du cache — un état où l'app
    croyait pouvoir travailler hors-ligne alors que le chargement partait
    quand même sur le réseau.

    `models_root` est conservé dans la signature pour ne pas casser les
    appelants (`settings_ui`, `download_model.py`) mais n'est plus consulté.
    """
    return hf_offline.is_model_cached(repo_id)


def download_model(
    repo_id: str,
    models_root: Path,  # noqa: ARG001 — conservé pour compat. des appelants
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Télécharge (ou met à jour) un modèle HuggingFace dans le cache HF.
    Retourne le chemin local du snapshot.

    `progress_callback(current_bytes, total_bytes)` est invoqué périodiquement
    si fourni — pour brancher une barre de progression dans l'UI.
    Note : `snapshot_download` gère lui-même la progression via `tqdm` ;
    le callback ici est best-effort (HF Hub ne propose pas d'API officielle
    de callback fin par fichier).

    `allow_network()` est indispensable : le reste de l'app tourne en mode
    hors-ligne forcé (cf. hf_offline), qui ferait échouer ce téléchargement.
    """
    with hf_offline.allow_network():
        local_path = snapshot_download(repo_id=repo_id)

    if progress_callback is not None:
        # Best-effort : on signale juste la fin (100%).
        progress_callback(1, 1)
    return Path(local_path)
