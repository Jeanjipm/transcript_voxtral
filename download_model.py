"""
CLI : télécharge ou met à jour un modèle de transcription.

Usage :
    python download_model.py                    # modèle de la config
    python download_model.py --model NAME       # modèle spécifique
    python download_model.py --list             # liste les modèles connus
"""

from __future__ import annotations

import argparse
import sys

from config import load_config
from model_manager import (
    AVAILABLE_MODELS,
    download_model,
    is_downloaded,
)


def _print_catalog() -> None:
    print("Modèles disponibles :\n")
    for m in AVAILABLE_MODELS:
        print(f"  {m.repo_id}")
        print(f"    {m.label} — {m.size_gb:.1f} Go")
        print(f"    {m.description}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Télécharge / met à jour un modèle MLX pour Voxtral Dictée."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="repo_id HuggingFace (défaut : celui de config.yaml)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liste les modèles connus et quitte",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_catalog()
        return 0

    cfg = load_config()
    repo_id = args.model if args.model is not None else cfg.model.name
    models_root = cfg.model.resolved_path

    if is_downloaded(repo_id, models_root):
        print(f"Modèle déjà présent : {repo_id}")
        print(f"Mise à jour vers la dernière version disponible...")
    else:
        print(f"Téléchargement de : {repo_id}")
        print(f"Destination : {models_root}")

    try:
        local_path = download_model(repo_id, models_root)
    except Exception as exc:
        print(f"Erreur durant le téléchargement : {exc}", file=sys.stderr)
        return 2

    print(f"OK — modèle disponible dans : {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
