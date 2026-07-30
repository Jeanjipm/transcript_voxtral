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


# Usages possibles d'un modèle.
#
# Tous les modèles savent transcrire une dictée, mais seuls ceux qui rendent
# des HORODATAGES conviennent aux fichiers : sans eux, pas de repères dans le
# .txt ni d'identification des locuteurs. Voxtral n'en produit aucun.
USAGE_DICTATION = "dictation"
USAGE_FILES = "files"


# Catalogue des modèles connus, exposé dans l'UI Préférences.
@dataclass(frozen=True)
class ModelInfo:
    repo_id: str
    label: str
    size_gb: float
    description: str
    usages: tuple[str, ...] = (USAGE_DICTATION,)
    # False pour les modèles distillés « turbo », qui rendent la langue source
    # au lieu de l'anglais quand on demande une traduction.
    can_translate: bool = True

    def supports(self, usage: str) -> bool:
        return usage in self.usages


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        repo_id="mzbac/voxtral-mini-3b-4bit-mixed",
        label="Voxtral Mini 3B (4-bit)",
        size_gb=3.2,
        description="Rapide et léger. Bon compromis pour la dictée.",
        usages=(USAGE_DICTATION,),
        can_translate=False,  # délégué à Whisper par VoxtralTranscriber
    ),
    ModelInfo(
        repo_id="mzbac/voxtral-mini-3b-8bit",
        label="Voxtral Mini 3B (8-bit)",
        size_gb=5.3,
        description="Meilleure qualité de dictée, plus lourd.",
        usages=(USAGE_DICTATION,),
        can_translate=False,
    ),
    ModelInfo(
        repo_id="mlx-community/whisper-large-v3-turbo",
        label="Whisper Large V3 Turbo",
        size_gb=1.6,
        description=(
            "Le plus rapide et le plus léger. Mesuré à 64× le temps réel "
            "contre 17× pour large-v3, avec une fidélité au moins égale. "
            "Ne sait pas traduire."
        ),
        usages=(USAGE_DICTATION, USAGE_FILES),
        can_translate=False,
    ),
    ModelInfo(
        repo_id="mlx-community/whisper-large-v3-mlx",
        label="Whisper Large V3",
        size_gb=3.0,
        description=(
            "Le seul à savoir traduire vers l'anglais. Plus lent que Turbo, "
            "à ne prendre que si tu utilises la traduction."
        ),
        usages=(USAGE_DICTATION, USAGE_FILES),
        can_translate=True,
    ),
]
# Retiré du catalogue : mistralai/Voxtral-Mini-3B-2507 (8,7 Go pleine
# précision). Le gain face au 8-bit ne justifie pas le poids, et un
# utilisateur qui le sélectionnait par curiosité déclenchait un
# téléchargement de 8,7 Go.


def list_available_models(usage: str | None = None) -> list[ModelInfo]:
    """Catalogue, éventuellement filtré sur un usage (dictée ou fichiers)."""
    if usage is None:
        return list(AVAILABLE_MODELS)
    return [m for m in AVAILABLE_MODELS if m.supports(usage)]


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


# ----------------------------------------------------------------------
# Occupation disque
#
# Le cache HuggingFace n'efface jamais rien de lui-même : chaque modèle
# essayé y reste pour toujours. Constaté sur une installation réelle,
# 22 Go accumulés dont 15 inutilisés, sans aucun moyen de s'en rendre compte
# depuis l'app. D'où ces fonctions, exposées dans Préférences → Stockage.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CachedModel:
    """Un modèle présent dans le cache HuggingFace."""

    repo_id: str
    size_bytes: int
    label: str  # libellé du catalogue, ou le repo_id s'il est inconnu
    in_catalog: bool

    @property
    def size_str(self) -> str:
        return format_size(self.size_bytes)


def format_size(size_bytes: int) -> str:
    """Taille lisible. On reste en Go/Mo : les utilisateurs raisonnent comme ça."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} Go"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.0f} Mo"
    return f"{size_bytes / 1_000:.0f} Ko"


def scan_cached_models() -> list[CachedModel]:
    """Liste les modèles présents dans le cache HuggingFace, du plus gros au
    plus petit.

    Best-effort : si l'API de cache est indisponible, on rend une liste vide
    plutôt que de faire échouer l'ouverture des Préférences.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return []

    try:
        info = scan_cache_dir()
    except Exception:  # noqa: BLE001 — cache absent ou corrompu
        return []

    result: list[CachedModel] = []
    for repo in info.repos:
        if getattr(repo, "repo_type", "model") != "model":
            continue
        known = find_model(repo.repo_id)
        result.append(
            CachedModel(
                repo_id=repo.repo_id,
                size_bytes=int(repo.size_on_disk),
                label=known.label if known else repo.repo_id,
                in_catalog=known is not None,
            )
        )
    result.sort(key=lambda m: -m.size_bytes)
    return result


def cached_size_bytes(repo_id: str) -> int:
    """Taille sur disque d'un modèle précis, en octets. 0 s'il est absent.

    On parcourt le dossier directement plutôt que d'appeler `scan_cache_dir`,
    qui inspecte tout le cache : ici on veut pouvoir interroger la taille
    plusieurs fois par seconde pour animer une barre de progression pendant
    un téléchargement.
    """
    folder = _repo_cache_dir(repo_id)
    if folder is None or not folder.is_dir():
        return 0
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _repo_cache_dir(repo_id: str) -> Path | None:
    """Dossier du cache HuggingFace correspondant à `repo_id`."""
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    safe = "models--" + repo_id.replace("/", "--")
    return Path(constants.HF_HUB_CACHE) / safe


def total_cache_size() -> int:
    """Occupation totale du cache HuggingFace, en octets."""
    return sum(m.size_bytes for m in scan_cached_models())


def delete_cached_model(repo_id: str) -> int:
    """Supprime un modèle du cache. Retourne le nombre d'octets libérés.

    L'appelant est responsable de vérifier que le modèle n'est pas en cours
    d'utilisation — cf. `models_in_use`. On ne le vérifie pas ici pour garder
    la fonction sans dépendance à la config.
    """
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    revisions = [
        rev.commit_hash
        for repo in info.repos
        if repo.repo_id == repo_id
        for rev in repo.revisions
    ]
    if not revisions:
        return 0

    strategy = info.delete_revisions(*revisions)
    freed = int(strategy.expected_freed_size)
    strategy.execute()
    return freed


def models_in_use(
    dictation_repo: str, file_repo: str, diarization: bool = False
) -> set[str]:
    """Les modèles qu'il ne faut pas proposer à la suppression.

    Inclut le repli de traduction : `VoxtralTranscriber` délègue à Whisper
    quand on demande une traduction, donc le supprimer casserait cette
    fonction sans que le lien soit évident pour l'utilisateur.

    Et, si l'identification des locuteurs est activée, le modèle de
    diarisation : il n'apparaît dans aucune liste de choix, donc rien ne
    signalerait à l'utilisateur que ces 225 Mo servent à quelque chose.
    """
    from transcriber import WHISPER_TRANSLATE_REPO

    in_use = {dictation_repo, file_repo, WHISPER_TRANSLATE_REPO}
    if diarization:
        from diarizer import DEFAULT_MODEL as DIARIZATION_REPO

        in_use.add(DIARIZATION_REPO)
    return in_use


def download_model(
    repo_id: str,
    models_root: Path | None = None,  # noqa: ARG001 — compat. des appelants
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
