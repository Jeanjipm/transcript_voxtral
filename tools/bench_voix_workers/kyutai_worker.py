#!/usr/bin/env python3
"""
Worker du banc « voix » : transcription en continu Kyutai STT (stt-1b-en_fr).

Tourne dans le venv dédié `~/.voxtral/bench/stt` (Python 3.12, paquet
`moshi_mlx==0.2.12`, qui impose une version ancienne de mlx) — jamais dans
celui de l'app. Il n'importe donc RIEN du dépôt.

    python kyutai_worker.py --wavs a.wav b.wav --result resultat.json

La boucle est celle du script officiel `stt_from_file_mlx.py` (dépôt
kyutai-labs/delayed-streams-modeling) : l'audio est donné au modèle par
tranches de 80 ms, comme s'il arrivait du micro, et le modèle rend le texte
au fil de l'eau avec 0,5 s de retard par construction. On note, pour chaque
mot, l'instant de l'audio où il a été prononcé et l'instant où il a été
écrit : c'est ce qui dit si le texte suivrait la voix en dictée réelle.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO = "kyutai/stt-1b-en_fr-mlx"
FRAME = 1920  # 80 ms à 24 kHz
SAMPLE_RATE = 24_000
TEXT_DELAY_S = 0.5  # retard du texte sur l'audio, propre à ce modèle


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs", nargs="+", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    out: dict = {"engine": "kyutai-stt-1b-en_fr", "repo": args.repo, "items": {}}
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import sentencepiece
        import sphn
        from huggingface_hub import hf_hub_download
        from moshi_mlx import models, utils

        t0 = time.perf_counter()
        with open(hf_hub_download(args.repo, "config.json"), "r") as fobj:
            raw_config = json.load(fobj)
        mimi_weights = hf_hub_download(args.repo, raw_config["mimi_name"])
        moshi_weights = hf_hub_download(
            args.repo, raw_config.get("moshi_name", "model.safetensors")
        )
        tokenizer_path = hf_hub_download(args.repo, raw_config["tokenizer_name"])

        lm_config = models.LmConfig.from_config_dict(raw_config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(model, bits=8, group_size=64)
        model.load_weights(moshi_weights, strict=True)
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(32))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)
        model.warmup()
        out["load_s"] = time.perf_counter() - t0

        for wav in args.wavs:
            name = Path(wav).stem
            item: dict = {}
            try:
                audio, _ = sphn.read(wav, sample_rate=SAMPLE_RATE)
                audio_s = audio.shape[-1] / SAMPLE_RATE
                # 2 s de silence en queue, comme le script officiel : le
                # modèle écrit avec retard, il faut le laisser finir.
                audio = mx.concat([mx.array(audio), mx.zeros((1, 2 * SAMPLE_RATE))], axis=-1)
                steps = audio.shape[-1] // FRAME
                gen = models.LmGen(
                    model=model,
                    max_steps=steps + 8,
                    text_sampler=utils.Sampler(top_k=25, temp=0),
                    audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                    check=False,
                )
                # Le tokenizer audio garde un état d'un appel à l'autre.
                if hasattr(audio_tokenizer, "reset_state"):
                    audio_tokenizer.reset_state()
                pieces: list[str] = []
                lags: list[float] = []
                t0 = time.perf_counter()
                for step in range(steps):
                    start = step * FRAME
                    block = audio[:, None, start : start + FRAME]
                    tokens = audio_tokenizer.encode_step(block).transpose(0, 2, 1)
                    text_token = gen.step(tokens[0])
                    text_token = text_token[0].item()
                    if text_token not in (0, 3):
                        piece = text_tokenizer.id_to_piece(text_token)
                        pieces.append(piece.replace("▁", " "))
                        # Retard d'affichage si l'audio arrivait du micro : le
                        # mot ne peut pas sortir avant la fin de sa tranche,
                        # ni avant que le calcul ait rattrapé le temps réel.
                        available_at = (step + 1) * FRAME / SAMPLE_RATE
                        computed_at = time.perf_counter() - t0
                        spoken_at = max(0.0, available_at - TEXT_DELAY_S)
                        lags.append(max(available_at, computed_at) - spoken_at)
                item["proc_s"] = time.perf_counter() - t0
                item["audio_s"] = audio_s
                item["text"] = "".join(pieces).strip()
                if lags:
                    lags.sort()
                    item["lag_median_s"] = lags[len(lags) // 2]
                    item["lag_max_s"] = lags[-1]
            except Exception as exc:  # noqa: BLE001 — un fichier raté n'arrête pas le banc
                item["error"] = f"{type(exc).__name__}: {exc}"
            out["items"][name] = item
            print(f"[kyutai] {name} : {item.get('error') or item['text'][:70]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["peak_rss_mb"] = _rss_mb()
    Path(args.result).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    sys.exit(main())
