"""
CLI debug : transcrit un fichier WAV et imprime le texte sur stdout.

Usage :
    python transcribe.py recording.wav
    python transcribe.py recording.wav --lang fr
    python transcribe.py recording.wav --task translate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import load_config
from transcriber import make_transcriber


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcrit un fichier audio via le modèle MLX configuré."
    )
    parser.add_argument("wav", type=Path, help="Chemin vers le fichier WAV")
    parser.add_argument(
        "--lang",
        default=None,
        help="Code langue (fr, en, de, ...) ou 'auto' (défaut : config.yaml)",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default=None,
        help="Tâche : transcribe (défaut config) ou translate vers anglais",
    )
    args = parser.parse_args(argv)

    if not args.wav.exists():
        print(f"Erreur : fichier introuvable : {args.wav}", file=sys.stderr)
        return 1

    cfg = load_config()
    language = args.lang if args.lang is not None else cfg.transcription.language
    task = args.task if args.task is not None else cfg.transcription.task

    transcriber = make_transcriber(cfg)
    text = transcriber.transcribe(
        args.wav,
        language=language,
        task=task,
        temperature=cfg.transcription.temperature,
        max_new_tokens=cfg.transcription.max_new_tokens,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
