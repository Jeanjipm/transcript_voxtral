#!/usr/bin/env python3
"""
Worker du banc « voix » : Pocket TTS (Kyutai), un modèle par process.

Tourne dans le venv dédié `~/.voxtral/bench/tts` (paquet `pocket-tts`, qui
tire torch) — jamais dans celui de l'app. Pour cette raison il n'importe RIEN
du dépôt : bibliothèque standard, numpy et pocket_tts seulement.

    python pocket_worker.py --language french --voice estelle \
        --texts textes.json --out-dir sons/ --result resultat.json

`textes.json` : {"nom": "texte à dire", ...}. Pour chaque texte on mesure le
délai avant le premier son (génération en flux), le temps total et la durée
du son produit, et on écrit `<out-dir>/<nom>.wav` (24 kHz mono 16 bits).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import wave
from pathlib import Path


def _rss_mb() -> float:
    """Pic mémoire du process. macOS compte en octets, Linux en kilo-octets."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)  # french, french_24l
    parser.add_argument("--voice", default="estelle")
    parser.add_argument("--texts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    out: dict = {"engine": f"pocket-{args.language}", "voice": args.voice, "items": {}}
    result_path = Path(args.result)
    try:
        import torch
        from pocket_tts import TTSModel

        texts: dict[str, str] = json.loads(Path(args.texts).read_text(encoding="utf-8"))
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        model = TTSModel.load_model(language=args.language)
        out["load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        voice_state = model.get_state_for_audio_prompt(args.voice)
        out["voice_s"] = time.perf_counter() - t0
        sample_rate = int(model.sample_rate)
        out["sample_rate"] = sample_rate

        # Première génération « à froid », hors mesure : ce qu'on attend à la
        # toute première lecture après le lancement de l'app.
        t0 = time.perf_counter()
        for _ in model.generate_audio_stream(voice_state, "Bonjour."):
            pass
        out["cold_s"] = time.perf_counter() - t0

        for name, text in texts.items():
            item: dict = {"chars": len(text)}
            try:
                chunks = []
                first_s = None
                t0 = time.perf_counter()
                for chunk in model.generate_audio_stream(voice_state, text):
                    if first_s is None:
                        first_s = time.perf_counter() - t0
                    chunks.append(chunk)
                item["total_s"] = time.perf_counter() - t0
                item["first_audio_s"] = first_s
                audio = torch.cat(chunks, dim=0) if chunks else torch.zeros(0)
                item["audio_s"] = audio.shape[-1] / sample_rate
                wav = out_dir / f"{name}.wav"
                _write_wav(wav, audio.detach().cpu().numpy(), sample_rate)
                item["wav"] = str(wav)
            except Exception as exc:  # noqa: BLE001 — une phrase ratée n'arrête pas le banc
                item["error"] = f"{type(exc).__name__}: {exc}"
            out["items"][name] = item
            etat = item.get("error") or f"{item['audio_s']:.1f} s de son"
            print(f"[pocket] {args.language} × {name} : {etat}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["peak_rss_mb"] = _rss_mb()
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    sys.exit(main())
