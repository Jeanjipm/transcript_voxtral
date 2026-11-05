"""
CLI debug : transcrit un fichier audio et imprime le texte sur stdout.

Usage :
    python transcribe.py recording.wav
    python transcribe.py memo.m4a --lang fr
    python transcribe.py recording.wav --task translate

Tout format lisible par macOS est accepté (wav, m4a, mp3, aiff, caf, flac,
piste audio d'un mp4) : la conversion en 16 kHz mono passe par le même
`audio_convert.ensure_16k_mono` que la transcription de fichiers de l'app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audio_convert import AudioConversionError, ensure_16k_mono
from config import load_config
from transcriber import make_transcriber


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcrit un fichier audio via le modèle MLX configuré."
    )
    parser.add_argument(
        "audio", type=Path, help="Fichier audio (wav, m4a, mp3, aiff, flac…)"
    )
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

    cfg = load_config()
    language = args.lang if args.lang is not None else cfg.transcription.language
    task = args.task if args.task is not None else cfg.transcription.task

    try:
        wav_path, is_temporary = ensure_16k_mono(args.audio)
    except AudioConversionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    try:
        transcriber = make_transcriber(cfg)
        text = transcriber.transcribe(
            wav_path,
            language=language,
            task=task,
            max_new_tokens=cfg.transcription.max_new_tokens,
        )
    finally:
        if is_temporary:
            wav_path.unlink(missing_ok=True)

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
