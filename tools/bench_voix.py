#!/usr/bin/env python3
"""
Banc d'essai « voix » — outil de développement, HORS app.

Trois questions, posées après l'écoute de l'épisode Underscore_ sur Kyutai
(novembre 2026), et auxquelles seules des mesures sur le Mac répondent :

1. **Lecture** (`tts`) : Pocket TTS (Kyutai, CPU seul, licence CC-BY) peut-il
   remplacer Voxtral TTS (2,5 Go, non commercial) pour « Lire le texte
   copié » ? On mesure le délai avant le premier son, la vitesse, la mémoire,
   et — contrôle objectif de l'intelligibilité — on fait retranscrire chaque
   son par Whisper pour compter les mots déformés. Les sons sont gardés pour
   être écoutés : la qualité d'une voix se juge à l'oreille.
2. **Dictée en continu** (`stt`) : Kyutai STT (stt-1b-en_fr) écrit pendant
   qu'on parle. Est-il aussi juste que les modèles en place, et suit-il le
   temps réel ? Mêmes fichiers et même mesure (WER) que `bench_models.py`.
3. **Mains libres** (`vad`) : Silero VAD (y a-t-il de la parole ?) et Smart
   Turn (la phrase est-elle finie ?), tous deux déjà fournis par le
   `mlx-audio` de l'app. Silero est comparé au filtre anti-silence actuel
   (seuil de volume) ; Smart Turn est essayé sur des phrases complètes et
   des phrases laissées en suspens.

Rien ici n'est importé par Voxtral Dictée, et rien n'est installé dans le
venv de l'app : Pocket TTS et Kyutai STT vivent chacun dans leur venv
(`~/.voxtral/bench/tts` et `~/.voxtral/bench/stt`), créés par
`tools/bench_voix.sh`. Leurs bibliothèques sont incompatibles entre elles
(deux versions de sentencepiece) et avec l'app (Kyutai impose un mlx ancien).

Lancement — le plus simple, tout d'un coup :

    bash tools/bench_voix.sh

ou, une fois les venv créés, partie par partie :

    ~/.voxtral/venv/bin/python tools/bench_voix.py tts
    ~/.voxtral/venv/bin/python tools/bench_voix.py stt
    ~/.voxtral/venv/bin/python tools/bench_voix.py vad
    ~/.voxtral/venv/bin/python tools/bench_voix.py --report   # tableau seul

Pour mesurer sur SA voix (dictée et fin de phrase), un enregistreur guidé fait
dire douze phrases, puis relance les deux mesures concernées :

    bash tools/bench_voix.sh enregistrer

Résultats : `tools/bench_voix_results/summary.md` (chiffres) et
`~/Documents/Voxtral/banc-voix/` (sons à écouter).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
WORKERS_DIR = TOOLS_DIR / "bench_voix_workers"
RESULTS_DIR = TOOLS_DIR / "bench_voix_results"
SOUNDS_DIR = Path.home() / "Documents" / "Voxtral" / "banc-voix"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

import bench_models  # noqa: E402 — WER, jeu d'essai et résultats de novembre
import hf_offline  # noqa: E402 — savoir si un modèle est déjà sur le disque

# Le venv de l'app (mlx, mlx-audio, mlx-whisper) et les deux venv du banc.
APP_PYTHON = Path.home() / ".voxtral" / "venv" / "bin" / "python"
BENCH_DIR = Path.home() / ".voxtral" / "bench"
TTS_PYTHON = BENCH_DIR / "tts" / "bin" / "python"
STT_PYTHON = BENCH_DIR / "stt" / "bin" / "python"

WORKER_TIMEOUT_S = 3600

# ----------------------------------------------------------------------
# Jeu d'essai
# ----------------------------------------------------------------------

# Textes à lire. Tous tiennent en un morceau de l'app (≤ 300 caractères), sauf
# le paragraphe : l'app le découpe en phrases, Pocket TTS le reçoit entier.
TTS_TEXTS: dict[str, str] = {
    "court": "Peux-tu me rappeler demain matin à neuf heures pour finaliser le "
             "devis du client de Chambéry ?",
    "chiffres": "La réunion est fixée au 12 mars 2026 à quatorze heures trente, "
                "salle B, avec un budget de 3500 euros hors taxes.",
    "noms_sigles": "Jean-Jérôme envoie le PDF sur la TVA à l'URSSAF, puis il "
                   "appelle l'office de tourisme de la Tarentaise.",
    "question": "Est-ce que tu as bien reçu mon message d'hier soir, ou "
                "faut-il que je te le renvoie ?",
    "paragraphe": "Le tourisme de montagne traverse une période de "
                  "transformation profonde. Les stations de moyenne altitude "
                  "voient leur enneigement reculer d'année en année, et les "
                  "offices de tourisme doivent réinventer leur offre pour "
                  "attirer des visiteurs toute l'année. La randonnée, le vélo "
                  "électrique, les activités en famille et le patrimoine "
                  "local deviennent des arguments aussi importants que le ski.",
}

VOXTRAL_VOICE = "fr_female"
POCKET_VOICE = "estelle"  # la seule voix française du catalogue Pocket TTS
POCKET_LANGUAGES = ("french", "french_24l")  # petit modèle, et aperçu du grand
WHISPER_ROUNDTRIP = "mlx-community/whisper-large-v3-turbo"

# Modèles du banc de novembre auxquels comparer Kyutai STT.
STT_REFERENCES = ("whisper-large-v3-turbo", "whisper-large-v3", "voxtral-mini-3b-8bit")
KYUTAI_KEY = "kyutai-stt-1b-en_fr"

# Fins de tour : (phrase complète, la même laissée en suspens). Dites par les
# voix de macOS — donc un test de ce que les MOTS annoncent. L'intonation
# d'une vraie hésitation ne s'imite pas : dépose tes propres enregistrements
# `tour_complet_*.wav` et `tour_suspendu_*.wav` dans bench_samples/wav/ pour
# un verdict sur ta voix.
TURN_PAIRS: list[tuple[str, str]] = [
    ("Je voudrais réserver une table pour quatre personnes demain soir.",
     "Je voudrais réserver une table pour"),
    ("Peux-tu envoyer le devis à Claire avant midi ?",
     "Peux-tu envoyer le devis à Claire et ensuite"),
    ("Rappelle-moi de téléphoner au garage lundi matin.",
     "Rappelle-moi de téléphoner au"),
    ("La réunion est reportée à jeudi quatorze heures.",
     "La réunion est reportée à jeudi parce que"),
    ("Ajoute du lait et des œufs à la liste de courses.",
     "Ajoute du lait, des œufs, et aussi"),
    ("Quel temps fera-t-il à Annecy ce week-end ?",
     "Quel temps fera-t-il à Annecy si jamais on"),
    ("Note que le client préfère être livré le matin.",
     "Note que le client préfère, euh,"),
    ("Écris à Thomas que j'arrive dans dix minutes.",
     "Écris à Thomas que"),
]
TURN_VOICES = ("Thomas", "Audrey")

# Phrases à dire soi-même (`bench_voix.py enregistrer`). Les complètes servent
# deux fois : à Smart Turn (« c'est fini »), et à mesurer la dictée sur une
# vraie voix, puisqu'on sait ce qui a été lu. Elles contiennent donc ce qui
# fait trébucher les modèles : un nom de société, des sigles, une date.
RECORD_COMPLETE: list[str] = [
    "Peux-tu rappeler Kixit Consulting demain matin à neuf heures ?",
    "La TVA est à envoyer à l'URSSAF avant vendredi.",
    "Le rendez-vous est fixé au 12 mars à quatorze heures trente.",
    "Ajoute du lait et des œufs à la liste de courses.",
    "Écris à Claire que j'arrive dans dix minutes.",
    "Note que le client préfère être livré le matin.",
]
RECORD_SUSPENDED: list[str] = [
    "Je voudrais réserver une table pour…",
    "Peux-tu envoyer le devis à Claire et ensuite…",
    "La réunion est reportée à jeudi parce que…",
    "Ajoute du lait, des œufs, et aussi…",
    "Note que le client préfère, euh…",
    "Rappelle-moi de téléphoner au…",
]

# Seuil du filtre anti-silence de l'app (config : recording.silence_threshold_dbfs).
APP_SILENCE_DBFS = -45.0


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------


def _rss_mb() -> float:
    """Pic mémoire du process. macOS compte en octets, Linux en kilo-octets."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _run(cmd: list[str], result: Path, label: str) -> dict:
    """Lance un worker dans son process et rend son fichier de résultat.

    Un worker qui plante sans rien écrire laisse un résultat « erreur » avec
    la fin de sa sortie : le banc continue, et le rapport dit ce qui a manqué.
    """
    print(f"[voix] {label} …", flush=True)
    result.unlink(missing_ok=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=WORKER_TIMEOUT_S,
        )
        tail = (proc.stdout + "\n" + proc.stderr)[-2500:]
        if not result.exists():
            _write_json(result, {"error": f"worker sans résultat (code {proc.returncode})\n{tail}"})
        elif proc.returncode != 0 and "error" not in _read_json(result):
            data = _read_json(result)
            data["error"] = f"code {proc.returncode}\n{tail}"
            _write_json(result, data)
    except subprocess.TimeoutExpired:
        _write_json(result, {"error": f"abandonné après {WORKER_TIMEOUT_S} s"})
    except OSError as exc:
        _write_json(result, {"error": f"lancement impossible : {exc}"})
    data = _read_json(result)
    if data.get("error"):
        print(f"[voix]   → ÉCHEC : {str(data['error']).splitlines()[0][:200]}", flush=True)
    return data


def _load_16k(path: Path):
    """WAV quelconque → numpy float32 mono 16 kHz (interpolation linéaire :
    suffisant pour de la parole destinée à un modèle de reconnaissance)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000 and audio.size:
        positions = np.arange(0, audio.shape[0], sr / 16_000)
        audio = np.interp(positions, np.arange(audio.shape[0]), audio).astype(np.float32)
    return audio


def _is_dictation_sample(stem: str) -> bool:
    """Un fichier du jeu d'essai qui est une dictée entière (pas une phrase
    en suspens ni une phrase du test de fin de tour)."""
    return not stem.startswith(("tour_", "reel_suspendu_"))


def _say(text: str, voice: str, wav: Path) -> None:
    """Fait dire `text` par une voix de macOS, en WAV 16 kHz mono."""
    if wav.exists():
        return
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["say", "-v", voice, "-o", str(wav), "--data-format=LEI16@16000", text],
        check=True,
    )


# ----------------------------------------------------------------------
# 1. Lecture : Voxtral TTS (venv de l'app) contre Pocket TTS (venv tts)
# ----------------------------------------------------------------------


def worker_voxtral_tts(out_path: Path) -> None:
    """Voxtral TTS, exactement comme l'app s'en sert : phrase par phrase."""
    import numpy as np
    import soundfile as sf

    import synthese

    out: dict = {"engine": "voxtral-tts-4b-4bit", "voice": VOXTRAL_VOICE, "items": {}}
    try:
        synth = synthese.Synthetiseur(synthese.DEFAULT_MODEL)
        t0 = time.perf_counter()
        synth.preload()
        out["load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        synth.lire("Bonjour.", VOXTRAL_VOICE)
        out["cold_s"] = time.perf_counter() - t0
        out["sample_rate"] = synth.sample_rate
        dest = SOUNDS_DIR / "voxtral"
        dest.mkdir(parents=True, exist_ok=True)

        for name, text in TTS_TEXTS.items():
            item: dict = {"chars": len(text)}
            try:
                phrases = synthese.decouper_en_phrases(text)
                item["phrases"] = len(phrases)
                morceaux = []
                first_s = None
                t0 = time.perf_counter()
                for phrase in phrases:
                    morceaux.append(synth.lire(phrase, VOXTRAL_VOICE))
                    if first_s is None:
                        # Dans l'app, le son part quand la 1re phrase est prête.
                        first_s = time.perf_counter() - t0
                item["total_s"] = time.perf_counter() - t0
                item["first_audio_s"] = first_s
                audio = np.concatenate(morceaux)
                item["audio_s"] = audio.shape[0] / synth.sample_rate
                wav = dest / f"{name}.wav"
                sf.write(str(wav), audio, synth.sample_rate, subtype="PCM_16")
                item["wav"] = str(wav)
            except Exception as exc:  # noqa: BLE001
                item["error"] = f"{type(exc).__name__}: {exc}"
            out["items"][name] = item
            etat = item.get("error") or f"{item['audio_s']:.1f} s de son"
            print(f"[voxtral-tts] {name} : {etat}", flush=True)

        # Ce que donnerait le mode « flux » de mlx-audio, que l'app n'utilise
        # pas : le son sort par tranches de 0,5 s pendant la génération.
        try:
            texte = TTS_TEXTS["court"]
            t0 = time.perf_counter()
            first = None
            for res in synth._model.generate(  # noqa: SLF001 — mesure, hors app
                text=texte, voice=VOXTRAL_VOICE,
                max_tokens=synthese.budget_tokens(texte),
                stream=True, streaming_interval=0.5,
            ):
                if first is None and np.asarray(synthese._vers_numpy(res.audio)).size:  # noqa: SLF001
                    first = time.perf_counter() - t0
            out["stream_first_audio_s"] = first
            out["stream_total_s"] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            out["stream_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["peak_rss_mb"] = _rss_mb()
    try:
        import mlx.core as mx

        out["mlx_peak_mb"] = mx.get_peak_memory() / 1e6
    except Exception:  # noqa: BLE001
        pass
    _write_json(out_path, out)


def worker_roundtrip(out_path: Path) -> None:
    """Fait retranscrire par Whisper tous les sons produits par les voix.

    Une voix peut être agréable et manger des mots : le taux de mots faux de
    l'aller-retour texte → voix → texte le mesure sans oreille humaine.
    """
    from transcriber import WhisperTranscriber

    out: dict = {"model": WHISPER_ROUNDTRIP, "items": {}}
    try:
        whisper = WhisperTranscriber(WHISPER_ROUNDTRIP)
        for engine_file in sorted(RESULTS_DIR.glob("tts__*.json")):
            engine = _read_json(engine_file)
            for name, item in engine.get("items", {}).items():
                wav = item.get("wav")
                if not wav or not Path(wav).exists():
                    continue
                key = f"{engine.get('engine')}/{name}"
                try:
                    result = whisper.transcribe_array(_load_16k(Path(wav)), 16_000, language="fr")
                    rate, words, errors = bench_models.wer(TTS_TEXTS[name], result.text)
                    out["items"][key] = {
                        "text": result.text, "wer": rate, "ref_words": words, "errors": errors,
                    }
                except Exception as exc:  # noqa: BLE001
                    out["items"][key] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[aller-retour] {key}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    _write_json(out_path, out)


def run_tts() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    me = str(Path(__file__).resolve())
    _run([str(APP_PYTHON), me, "--worker", "voxtral-tts",
          str(RESULTS_DIR / "tts__voxtral.json")],
         RESULTS_DIR / "tts__voxtral.json", "Voxtral TTS (lecture actuelle)")

    texts = RESULTS_DIR / "tts_texts.json"
    _write_json(texts, TTS_TEXTS)
    for language in POCKET_LANGUAGES:
        result = RESULTS_DIR / f"tts__pocket-{language}.json"
        if not TTS_PYTHON.exists():
            _write_json(result, {
                "engine": f"pocket-{language}",
                "error": "venv ~/.voxtral/bench/tts absent : lance bash tools/bench_voix.sh",
            })
            continue
        _run([str(TTS_PYTHON), str(WORKERS_DIR / "pocket_worker.py"),
              "--language", language, "--voice", POCKET_VOICE,
              "--texts", str(texts), "--out-dir", str(SOUNDS_DIR / f"pocket-{language}"),
              "--result", str(result)],
             result, f"Pocket TTS {language}")

    _run([str(APP_PYTHON), me, "--worker", "roundtrip",
          str(RESULTS_DIR / "tts_roundtrip.json")],
         RESULTS_DIR / "tts_roundtrip.json", "aller-retour par Whisper")


# ----------------------------------------------------------------------
# 2. Dictée en continu : Kyutai STT (venv stt)
# ----------------------------------------------------------------------


def run_stt(only: list[str] | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_models.prepare_synth_samples()
    wavs = [w for w in bench_models.list_samples(only) if _is_dictation_sample(w.stem)]
    # Les modèles déjà dans l'app n'ont pas encore vu les enregistrements
    # faits avec `enregistrer` : on les y passe, pour comparer à voix égale.
    for wav in wavs:
        if wav.stem.startswith("reel_"):
            for model in STT_REFERENCES:
                # Seulement les modèles déjà sur le disque : pas question de
                # télécharger 5 Go pour une ligne de comparaison.
                if not hf_offline.is_model_cached(bench_models.MODELS[model].repo):
                    continue
                if not bench_models.result_path(model, wav.stem, False).exists():
                    print(f"[voix] {model} × {wav.stem} …", flush=True)
                    bench_models.run_pair(model, wav, offline=False, force=False)
    result = RESULTS_DIR / "stt__kyutai.json"
    if not STT_PYTHON.exists():
        _write_json(result, {
            "engine": KYUTAI_KEY,
            "error": "venv ~/.voxtral/bench/stt absent : lance bash tools/bench_voix.sh",
        })
        return
    _run([str(STT_PYTHON), str(WORKERS_DIR / "kyutai_worker.py"),
          "--wavs", *[str(w) for w in wavs], "--result", str(result)],
         result, f"Kyutai STT sur {len(wavs)} fichiers")


# ----------------------------------------------------------------------
# 3. Mains libres : Silero VAD et Smart Turn (mlx-audio de l'app)
# ----------------------------------------------------------------------


def _noise(seconds: float, dbfs: float):
    """Bruit blanc au niveau RMS demandé (graine fixe : mesure reproductible)."""
    import numpy as np

    rng = np.random.default_rng(2026)
    samples = rng.standard_normal(int(seconds * 16_000)).astype(np.float32)
    return samples * (10 ** (dbfs / 20.0))


def worker_vad(out_path: Path) -> None:
    import numpy as np

    import audio_level
    from mlx_audio.vad import load

    out: dict = {"silero": {}, "turns": {}}
    vad = None

    # ---- Silero : parole ou pas ? ----
    try:
        t0 = time.perf_counter()
        vad = load("mlx-community/silero-vad")
        out["silero_load_s"] = time.perf_counter() - t0

        cases: dict[str, tuple] = {
            "silence absolu (3 s)": (np.zeros(48_000, dtype=np.float32), False),
            "souffle faible, −60 dBFS (3 s)": (_noise(3, -60), False),
            "bruit moyen, −40 dBFS (3 s)": (_noise(3, -40), False),
            "bruit fort, −25 dBFS (3 s)": (_noise(3, -25), False),
        }
        bench_models.prepare_synth_samples()
        for wav in bench_models.list_samples(None):
            if not _is_dictation_sample(wav.stem):
                continue
            audio = _load_16k(wav)
            cases[wav.stem] = (audio, True)
            if wav.stem == "synth_court_dictee":
                # La même phrase dite très bas : le filtre de volume la jette.
                peak = float(np.abs(audio).max()) or 1.0
                cases["phrase chuchotée, crête −50 dBFS"] = (
                    audio / peak * (10 ** (-50 / 20.0)), True,
                )

        for name, (audio, has_speech) in cases.items():
            item: dict = {"has_speech": has_speech, "audio_s": audio.shape[0] / 16_000}
            try:
                t0 = time.perf_counter()
                stamps = vad.get_speech_timestamps(audio, sample_rate=16_000, return_seconds=True)
                item["proc_s"] = time.perf_counter() - t0
                item["speech_s"] = sum(float(s["end"]) - float(s["start"]) for s in stamps)
                item["silero_says_speech"] = bool(stamps)
                level = audio_level.peak_rms_dbfs(audio, 16_000)
                item["level_dbfs"] = None if math.isinf(level) else level
                item["filter_says_speech"] = not audio_level.is_silent(level, APP_SILENCE_DBFS)
            except Exception as exc:  # noqa: BLE001
                item["error"] = f"{type(exc).__name__}: {exc}"
            out["silero"][name] = item
            print(f"[silero] {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["silero_error"] = f"{type(exc).__name__}: {exc}"

    # ---- Smart Turn : phrase finie ou en suspens ? ----
    try:
        t0 = time.perf_counter()
        turn = load("mlx-community/smart-turn-v3", strict=True)
        out["turn_load_s"] = time.perf_counter() - t0

        wav_dir = bench_models.WAV_DIR
        cases_t: list[tuple[str, Path, bool, str]] = []
        for i, (complete, suspended) in enumerate(TURN_PAIRS, start=1):
            for voice in TURN_VOICES:
                done = wav_dir / f"tour_synth_{i:02d}_{voice}_complet.wav"
                open_ = wav_dir / f"tour_synth_{i:02d}_{voice}_suspendu.wav"
                _say(complete, voice, done)
                _say(suspended, voice, open_)
                cases_t.append((done.stem, done, True, complete))
                cases_t.append((open_.stem, open_, False, suspended))
        # Enregistrements de l'utilisateur, s'il en a déposé.
        for pattern, complete in (
            ("reel_complet_*.wav", True), ("tour_complet_*.wav", True),
            ("reel_suspendu_*.wav", False), ("tour_suspendu_*.wav", False),
        ):
            for wav in sorted(wav_dir.glob(pattern)):
                note = wav.with_suffix(".prompt")
                text = note.read_text(encoding="utf-8").strip() if note.exists() else ""
                cases_t.append((wav.stem, wav, complete, text))

        turn.predict_endpoint(np.zeros(16_000, dtype=np.float32), sample_rate=16_000)  # à froid
        for name, wav, complete, text in cases_t:
            item = {"complete": complete, "text": text, "real": not name.startswith("tour_synth_")}
            try:
                audio = _load_16k(wav)
                # Comme en vrai : Smart Turn est interrogé 0,2 s après la fin
                # de la parole, pas après le silence qui traîne dans le fichier.
                if vad is not None:
                    stamps = vad.get_speech_timestamps(
                        audio, sample_rate=16_000, return_seconds=True
                    )
                    if stamps:
                        end = int((float(stamps[-1]["end"]) + 0.2) * 16_000)
                        audio = audio[:end]
                t0 = time.perf_counter()
                res = turn.predict_endpoint(audio, sample_rate=16_000)
                item["proc_s"] = time.perf_counter() - t0
                item["probability"] = float(res.probability)
                item["says_complete"] = int(res.prediction) == 1
            except Exception as exc:  # noqa: BLE001
                item["error"] = f"{type(exc).__name__}: {exc}"
            out["turns"][name] = item
            print(f"[smart-turn] {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["turn_error"] = f"{type(exc).__name__}: {exc}"

    out["peak_rss_mb"] = _rss_mb()
    _write_json(out_path, out)


def run_vad() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    me = str(Path(__file__).resolve())
    _run([str(APP_PYTHON), me, "--worker", "vad", str(RESULTS_DIR / "vad.json")],
         RESULTS_DIR / "vad.json", "Silero VAD et Smart Turn")


# ----------------------------------------------------------------------
# Enregistreur guidé : douze phrases dites par l'utilisateur
# ----------------------------------------------------------------------


def _record_one(path: Path) -> float:
    """Enregistre le micro entre deux appuis sur Entrée. Rend la durée."""
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    blocks: list = []
    input("   Entrée pour commencer… ")
    with sd.InputStream(samplerate=16_000, channels=1, dtype="int16",
                        callback=lambda data, *_: blocks.append(data.copy())):
        input("   ● ça enregistre. Parle, puis Entrée pour arrêter… ")
    audio = np.concatenate(blocks) if blocks else np.zeros((0, 1), dtype="int16")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, 16_000, subtype="PCM_16")
    return audio.shape[0] / 16_000


def run_record() -> None:
    """Fait dire douze phrases : six finies, six laissées en suspens."""
    wav_dir = bench_models.WAV_DIR
    print(
        "\nEnregistrement de ta voix : 12 phrases, environ 3 minutes.\n"
        "Pour chacune : appuie sur Entrée, lis la phrase à voix haute, comme\n"
        "tu dicterais, puis appuie de nouveau sur Entrée.\n"
        "(Si macOS demande l'accès au micro pour le Terminal, accepte.)\n"
    )
    print("— Première série : des phrases COMPLÈTES. Dis-les jusqu'au bout.\n")
    for i, text in enumerate(RECORD_COMPLETE, start=1):
        print(f"{i}/6  « {text} »")
        stem = f"reel_complet_{i:02d}"
        seconds = _record_one(wav_dir / f"{stem}.wav")
        # Référence pour le taux de mots faux, et rappel du texte pour le rapport.
        (bench_models.SAMPLES_DIR / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        (wav_dir / f"{stem}.prompt").write_text(text, encoding="utf-8")
        print(f"   ✓ {seconds:.1f} s\n")
    print(
        "— Seconde série : des phrases EN SUSPENS. Arrête-toi net aux points\n"
        "  de suspension, comme quand on cherche la suite, sans baisser la voix.\n"
    )
    for i, text in enumerate(RECORD_SUSPENDED, start=1):
        print(f"{i}/6  « {text} »")
        stem = f"reel_suspendu_{i:02d}"
        seconds = _record_one(wav_dir / f"{stem}.wav")
        (wav_dir / f"{stem}.prompt").write_text(text, encoding="utf-8")
        print(f"   ✓ {seconds:.1f} s\n")
    print("Merci. Les mesures sur ta voix démarrent.\n")


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------


def _num(value, fmt: str = "{:.1f}", empty: str = "—") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return empty
    return fmt.format(value).replace(".", ",").replace("-", "−")


def _pct(value) -> str:
    """Pourcentage, ou un simple tiret quand il n'y a rien à montrer."""
    return "—" if value is None else f"{_num(value)} %"


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _first_line(error) -> str:
    return str(error).strip().splitlines()[0][:160] if error else ""


def report_tts() -> list[str]:
    lines = ["## 1. Lecture à voix haute : Voxtral TTS contre Pocket TTS", ""]
    engines = [
        _read_json(RESULTS_DIR / "tts__voxtral.json"),
        *[_read_json(RESULTS_DIR / f"tts__pocket-{lang}.json") for lang in POCKET_LANGUAGES],
    ]
    engines = [e for e in engines if e]
    if not engines:
        return lines + ["Pas encore mesuré (`bench_voix.py tts`).", ""]
    roundtrip = _read_json(RESULTS_DIR / "tts_roundtrip.json").get("items", {})

    lines += [
        "| Moteur | Chargement | 1re lecture à froid | Délai avant le 1er son (moyenne) | "
        "Vitesse (× temps réel) | Pic mémoire | Mots déformés (aller-retour Whisper) |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in engines:
        name = e.get("engine", "?")
        if e.get("error"):
            lines.append(f"| {name} | ÉCHEC : {_first_line(e['error'])} | | | | | |")
            continue
        items = [i for i in e.get("items", {}).values() if "error" not in i]
        audio = sum(i["audio_s"] for i in items)
        total = sum(i["total_s"] for i in items)
        errs = sum(roundtrip.get(f"{name}/{k}", {}).get("errors", 0) for k in e["items"])
        words = sum(roundtrip.get(f"{name}/{k}", {}).get("ref_words", 0) for k in e["items"])
        mem = e.get("peak_rss_mb")
        if e.get("mlx_peak_mb"):
            mem_txt = f"{_num(mem / 1000)} Go (+ {_num(e['mlx_peak_mb'] / 1000)} Go côté GPU)"
        else:
            mem_txt = f"{_num(mem / 1000 if mem else None)} Go"
        lines.append(
            f"| {name} | {_num(e.get('load_s'))} s | {_num(e.get('cold_s'))} s | "
            f"{_num(_mean([i.get('first_audio_s') for i in items]), '{:.2f}')} s | "
            f"{_num(audio / total if total else None)}× | {mem_txt} | "
            f"{_pct(100 * errs / words if words else None)} |"
        )
    lines += ["", "Détail par texte (délai avant le 1er son / temps total / durée du son, en secondes) :", ""]
    lines += ["| Texte | " + " | ".join(e.get("engine", "?") for e in engines) + " |",
              "|---|" + "---|" * len(engines)]
    for name in TTS_TEXTS:
        cells = []
        for e in engines:
            i = e.get("items", {}).get(name)
            if not i:
                cells.append("—")
            elif "error" in i:
                cells.append(f"ÉCHEC : {_first_line(i['error'])}")
            else:
                cells.append(f"{_num(i.get('first_audio_s'), '{:.2f}')} / "
                             f"{_num(i.get('total_s'))} / {_num(i.get('audio_s'))}")
        lines.append(f"| {name} ({len(TTS_TEXTS[name])} car.) | " + " | ".join(cells) + " |")
    vox = engines[0]
    if vox.get("stream_first_audio_s") is not None:
        lines += ["", f"Voxtral TTS en mode « flux » (non utilisé par l'app), phrase courte : "
                      f"1er son après {_num(vox['stream_first_audio_s'], '{:.2f}')} s, "
                      f"fin après {_num(vox.get('stream_total_s'))} s."]
    elif vox.get("stream_error"):
        lines += ["", f"Voxtral TTS en mode « flux » : échec ({_first_line(vox['stream_error'])})."]
    lines += ["", f"**À écouter** (c'est l'oreille qui tranche) : `{SOUNDS_DIR}` — un "
                  f"dossier par moteur, les mêmes cinq textes dans chacun.", ""]
    return lines


def report_stt() -> list[str]:
    lines = ["## 2. Dictée en continu : Kyutai STT", ""]
    data = _read_json(RESULTS_DIR / "stt__kyutai.json")
    if not data:
        return lines + ["Pas encore mesuré (`bench_voix.py stt`).", ""]
    if data.get("error"):
        return lines + [f"ÉCHEC : {_first_line(data['error'])}", ""]

    rows: dict[str, dict] = {}
    for sample, item in data.get("items", {}).items():
        if "error" in item:
            continue
        ref = bench_models.SAMPLES_DIR / f"{sample}.txt"
        if ref.exists():
            item["wer"], item["ref_words"], item["errors"] = bench_models.wer(
                ref.read_text(encoding="utf-8"), item["text"]
            )
        rows[sample] = item

    def group_wer(samples: list[str], getter) -> float | None:
        errors = words = 0
        for s in samples:
            e, w = getter(s)
            if w:
                errors, words = errors + e, words + w
        return 100 * errors / words if words else None

    def reference(model: str):
        def getter(sample: str):
            r = _read_json(bench_models.RESULTS_DIR / f"{model}__{sample}.json")
            ref = bench_models.SAMPLES_DIR / f"{sample}.txt"
            if not r.get("ok") or not ref.exists():
                return 0, 0
            _, words, errors = bench_models.wer(ref.read_text(encoding="utf-8"), r.get("text", ""))
            return errors, words
        return getter

    def kyutai(sample: str):
        return rows[sample].get("errors", 0), rows[sample].get("ref_words", 0)

    scored = [s for s in rows if rows[s].get("ref_words")]
    synth = [s for s in scored if not s.startswith("reel_")]
    real = [s for s in scored if s.startswith("reel_")]
    audio = sum(i["audio_s"] for i in rows.values())
    proc = sum(i["proc_s"] for i in rows.values())
    lines += [
        "| Modèle | Mots faux, phrases synthé | Mots faux, réel | Vitesse (× temps réel) |",
        "|---|---|---|---|",
        f"| **Kyutai STT 1B en/fr** | {_pct(group_wer(synth, kyutai))} | "
        f"{_pct(group_wer(real, kyutai))} | {_num(audio / proc if proc else None)}× |",
    ]
    for model in STT_REFERENCES:
        spec = bench_models.MODELS.get(model)
        lines.append(
            f"| {spec.label if spec else model} (banc de novembre, mêmes fichiers) | "
            f"{_pct(group_wer(synth, reference(model)))} | "
            f"{_pct(group_wer(real, reference(model)))} | — |"
        )
    lags = [i["lag_median_s"] for i in rows.values() if "lag_median_s" in i]
    lag_max = [i["lag_max_s"] for i in rows.values() if "lag_max_s" in i]
    lines += [
        "",
        f"Chargement : {_num(data.get('load_s'))} s. Pic mémoire : "
        f"{_num(data.get('peak_rss_mb', 0) / 1000)} Go.",
        f"**Retard d'affichage** (entre le moment où un mot est dit et celui où il "
        f"s'écrirait, si l'audio venait du micro) : médiane "
        f"{_num(_mean(lags), '{:.2f}')} s, pire cas {_num(max(lag_max) if lag_max else None, '{:.2f}')} s. "
        f"Le modèle écrit par construction 0,5 s après la voix ; au-delà, c'est "
        f"le Mac qui ne suit pas le temps réel.",
        "",
        "| Fichier | Durée | Mots faux | Retard médian | Retard max |",
        "|---|---|---|---|---|",
    ]
    for sample in sorted(rows):
        i = rows[sample]
        wer_txt = _num(100 * i["wer"]) + " %" if "wer" in i else "—"
        lines.append(f"| {sample} | {_num(i['audio_s'])} s | {wer_txt} | "
                     f"{_num(i.get('lag_median_s'), '{:.2f}')} s | "
                     f"{_num(i.get('lag_max_s'), '{:.2f}')} s |")
    failed = [s for s, i in data.get("items", {}).items() if "error" in i]
    if failed:
        lines += ["", f"Fichiers en échec : {', '.join(failed)}."]
    lines += ["", "Textes produits (phrases synthétiques seulement — les enregistrements "
                  "réels sont des paroles privées) :", ""]
    for sample in sorted(synth):
        lines.append(f"- **{sample}** : {rows[sample]['text']}")
    lines.append("")
    return lines


def report_vad() -> list[str]:
    lines = ["## 3. Mains libres : Silero VAD et Smart Turn", ""]
    data = _read_json(RESULTS_DIR / "vad.json")
    if not data:
        return lines + ["Pas encore mesuré (`bench_voix.py vad`).", ""]
    if data.get("error"):
        return lines + [f"ÉCHEC : {_first_line(data['error'])}", ""]

    lines += ["### Y a-t-il de la parole ? Silero contre le filtre de volume de l'app", ""]
    if data.get("silero_error"):
        lines += [f"ÉCHEC : {_first_line(data['silero_error'])}", ""]
    else:
        silero = {k: v for k, v in data.get("silero", {}).items() if "error" not in v}
        ok_s = sum(v["silero_says_speech"] == v["has_speech"] for v in silero.values())
        ok_f = sum(v["filter_says_speech"] == v["has_speech"] for v in silero.values())
        audio = sum(v["audio_s"] for v in silero.values())
        proc = sum(v["proc_s"] for v in silero.values())
        lines += [
            f"Bonnes réponses : **Silero {ok_s}/{len(silero)}**, filtre de volume "
            f"actuel (seuil {_num(APP_SILENCE_DBFS, '{:.0f}')} dBFS) {ok_f}/{len(silero)}. "
            f"Silero traite {_num(audio / proc if proc else None, '{:.0f}')}× le temps réel "
            f"(chargement {_num(data.get('silero_load_s'), '{:.2f}')} s).",
            "",
            "| Cas | Parole réelle | Niveau | Filtre de volume | Silero | Parole détectée |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in silero.items():
            def verdict(said: bool, truth: bool = v["has_speech"]) -> str:
                return ("parole" if said else "rien") + ("" if said == truth else " ✗")
            lines.append(
                f"| {name} | {'oui' if v['has_speech'] else 'non'} | "
                f"{_num(v.get('level_dbfs'), '{:.0f}', '−∞')} dBFS | "
                f"{verdict(v['filter_says_speech'])} | {verdict(v['silero_says_speech'])} | "
                f"{_num(v['speech_s'])} s sur {_num(v['audio_s'])} s |"
            )
        lines.append("")

    lines += ["### La phrase est-elle finie ? Smart Turn", ""]
    if data.get("turn_error"):
        lines += [f"ÉCHEC : {_first_line(data['turn_error'])}", ""]
        return lines
    turns = {k: v for k, v in data.get("turns", {}).items() if "error" not in v}
    for label, subset in (
        ("Phrases dites par les voix de macOS", {k: v for k, v in turns.items() if not v["real"]}),
        ("Tes enregistrements", {k: v for k, v in turns.items() if v["real"]}),
    ):
        if not subset:
            continue
        done = [v for v in subset.values() if v["complete"]]
        open_ = [v for v in subset.values() if not v["complete"]]
        lines += [
            f"**{label}** : phrases complètes reconnues finies "
            f"{sum(v['says_complete'] for v in done)}/{len(done)} ; phrases en suspens "
            f"reconnues inachevées {sum(not v['says_complete'] for v in open_)}/{len(open_)}. "
            f"Temps de réponse moyen : "
            f"{_num(1000 * (_mean([v['proc_s'] for v in subset.values()]) or 0), '{:.0f}')} ms.",
            "",
        ]
    lines += ["| Phrase | Attendu | Réponse | Probabilité « finie » |", "|---|---|---|---|"]
    for name, v in turns.items():
        said = "finie" if v["says_complete"] else "en suspens"
        expected = "finie" if v["complete"] else "en suspens"
        mark = "" if said == expected else " ✗"
        if v["text"] and not v["real"]:
            text = f"« {v['text']} » ({name.split('_')[3]})"
        elif v["text"]:
            text = f"« {v['text']} » (ta voix)"
        else:
            text = name
        lines.append(f"| {text} | {expected} | {said}{mark} | {_num(v['probability'], '{:.2f}')} |")
    lines += [
        "",
        "Limite : les voix de macOS disent une phrase tronquée d'un ton posé, "
        "sans l'intonation d'une vraie hésitation. Ce test dit ce que les mots "
        "annoncent ; pour ta voix : `bash tools/bench_voix.sh enregistrer`.",
        "",
    ]
    return lines


def write_report() -> str:
    lines = [
        "# Banc d'essai « voix » — résultats",
        "",
        f"Produit par `tools/bench_voix.py` le {time.strftime('%d/%m/%Y à %H:%M')}.",
        "",
        *report_tts(), *report_stt(), *report_vad(),
    ]
    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.md").write_text(report, encoding="utf-8")
    return report


# ----------------------------------------------------------------------
# Entrée
# ----------------------------------------------------------------------

PARTS = {"tts": run_tts, "stt": run_stt, "vad": run_vad}
WORKERS = {
    "voxtral-tts": worker_voxtral_tts,
    "roundtrip": worker_roundtrip,
    "vad": worker_vad,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Banc d'essai « voix » (hors app).")
    parser.add_argument("parts", nargs="*",
                        help="tts, stt, vad (défaut : tout), ou enregistrer")
    parser.add_argument("--samples", nargs="*",
                        help="stt : limiter aux fichiers nommés (sans .wav)")
    parser.add_argument("--report", action="store_true", help="n'écrire que le rapport")
    parser.add_argument("--worker", nargs=2, metavar=("NOM", "SORTIE"))
    args = parser.parse_args(argv)

    if args.worker:
        name, out = args.worker
        WORKERS[name](Path(out))
        return 0

    parts = list(args.parts)
    if "enregistrer" in parts:
        # Ta voix d'abord, puis les deux mesures qui s'en servent.
        run_record()
        parts = [p for p in parts if p != "enregistrer"] or ["stt", "vad"]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        print(f"[voix] parties inconnues : {unknown} "
              f"(attendu : tts, stt, vad, enregistrer)", file=sys.stderr)
        return 2
    if not args.report:
        for part in parts or list(PARTS):
            if part == "stt":
                run_stt(args.samples)
            else:
                PARTS[part]()
    print(write_report())
    print(f"\n[voix] Rapport : {RESULTS_DIR / 'summary.md'}")
    print(f"[voix] Sons à écouter : {SOUNDS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
