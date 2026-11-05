#!/usr/bin/env python3
"""
Banc d'essai des modèles de transcription — outil de développement, HORS app.

Rien ici n'est importé par Voxtral Dictée : ce script sert à décider, mesures
à l'appui, si un nouveau modèle mérite d'entrer dans le catalogue
(`model_manager.AVAILABLE_MODELS`). Résultats commentés dans `BENCH-2026-11.md`.

Lancement (depuis la racine du dépôt, avec le venv de l'app, celui qui a
mlx-audio) :

    ~/.voxtral/venv/bin/python tools/bench_models.py              # tout
    ~/.voxtral/venv/bin/python tools/bench_models.py --models parakeet-tdt-0.6b-v3
    ~/.voxtral/venv/bin/python tools/bench_models.py --samples synth_court_dictee
    ~/.voxtral/venv/bin/python tools/bench_models.py --report     # tableau seul

## Comment c'est mesuré, et pourquoi comme ça

- **Un sous-processus par couple (modèle, fichier).** C'est le seul moyen
  d'obtenir un pic mémoire propre à chaque modèle (`ru_maxrss` du
  sous-processus) et un vrai temps de chargement : dans un seul process, le
  deuxième modèle profiterait du cache disque du premier et sa mémoire
  s'additionnerait à celle du précédent.
- **Trois temps par couple** : chargement, première transcription (« à froid »,
  compilation Metal comprise : c'est ce qu'on attend à la première dictée) et
  seconde transcription (« à chaud » : le régime de croisière). Le facteur
  temps réel est calculé sur le temps à chaud.
- **WER** (word error rate, taux de mots faux) = (substitutions + insertions
  + suppressions) / mots de la référence, sur du texte normalisé : minuscules,
  ponctuation ôtée, élisions séparées. Calculé par le parent à partir des
  fichiers `bench_samples/<nom>.txt`, donc une référence corrigée se répercute
  sans relancer les modèles. Implémentation maison (Levenshtein sur les mots)
  pour ne pas ajouter de dépendance.
- **Hors-ligne** : `--offline` rejoue un fichier court avec `HF_HUB_OFFLINE=1`
  ET un point d'accès HuggingFace injoignable, en passant au backend le dossier
  local rendu par `hf_offline.resolve_local_path` — exactement ce que ferait
  l'app. Un modèle qui échoue ici n'est pas intégrable tel quel.

## Jeu d'essai

- `synth_*` : phrases lues par la synthèse vocale macOS (`say`), donc
  référence exacte. Générées à la volée dans `bench_samples/wav/` (ignoré par
  git).
- `reel_*` : enregistrements de l'utilisateur, convertis en WAV 16 kHz mono
  (`afconvert -f WAVE -d LEI16@16000 -c 1`) et déposés dans
  `bench_samples/wav/`. Jamais commités. Référence relue à la main dans
  `bench_samples/reel_<nom>.txt` ; sans référence, seule la vitesse est
  mesurée.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
SAMPLES_DIR = TOOLS_DIR / "bench_samples"
WAV_DIR = SAMPLES_DIR / "wav"
RESULTS_DIR = TOOLS_DIR / "bench_results"

# Le venv de l'app : le seul qui a mlx, mlx-voxtral, mlx-whisper et mlx-audio.
PYTHON = Path.home() / ".voxtral" / "venv" / "bin" / "python"

# Une transcription qui dépasse ça est un modèle qui boucle : on l'abandonne.
WORKER_TIMEOUT_S = 1800


# ----------------------------------------------------------------------
# Phrases synthétiques (référence exacte = le texte lu par `say`)
# ----------------------------------------------------------------------

SYNTH_SAMPLES: dict[str, tuple[str, str]] = {
    "synth_court_dictee": (
        "Thomas",
        "Peux-tu me rappeler demain matin à neuf heures pour finaliser le "
        "devis du client de Chambéry ?",
    ),
    "synth_court_chiffres": (
        "Thomas",
        "La réunion est fixée au 12 mars 2026 à quatorze heures trente, salle "
        "B, avec un budget de 3500 euros hors taxes.",
    ),
    "synth_court_noms": (
        "Thomas",
        "Jean-Jérôme Danton rencontre l'office de tourisme de Serre Chevalier "
        "Briançon et la mairie de Saint-Pourçain-sur-Sioule.",
    ),
    "synth_court_sigles": (
        "Thomas",
        "Le PDF de la direction des finances sur la TVA est à envoyer à "
        "l'URSSAF et à la CCI avant vendredi.",
    ),
    "synth_court_question": (
        "Thomas",
        "Est-ce que l'outil reconnaît bien les questions, les virgules et les "
        "points de suspension, ou pas ?",
    ),
    "synth_court_audrey": (
        "Audrey",
        "Bonjour, je voudrais réserver deux places pour le concert de samedi "
        "soir, si possible côté balcon.",
    ),
    "synth_moyen_mail": (
        "Thomas",
        "Bonjour Claire, merci pour ton retour rapide. Je te confirme que la "
        "formation sur l'intelligence artificielle aura lieu les 3 et 4 "
        "décembre à Annecy. Peux-tu réserver la salle du deuxième étage, "
        "celle avec le vidéoprojecteur, et prévenir les douze participants ? "
        "Je t'envoie le programme détaillé en fin de semaine. Bonne journée, "
        "Jean-Jérôme.",
    ),
    "synth_moyen_audrey": (
        "Audrey",
        "Pour la commande du marché, il nous faut trois kilos de pommes, deux "
        "bouquets de basilic, un litre d'huile d'olive, six œufs et une "
        "baguette de tradition. Le total ne doit pas dépasser quarante euros, "
        "sinon on reporte le fromage à la semaine prochaine.",
    ),
    "synth_long_tourisme": (
        "Thomas",
        "Le tourisme de montagne traverse une période de transformation "
        "profonde. Les stations de moyenne altitude voient leur enneigement "
        "reculer d'année en année, et les offices de tourisme doivent "
        "réinventer leur offre pour attirer des visiteurs toute l'année. "
        "La randonnée, le vélo électrique, les activités en famille et le "
        "patrimoine local deviennent des arguments aussi importants que le "
        "ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi "
        "de mutualiser leurs moyens pour financer une plateforme de "
        "réservation commune. L'objectif est double : simplifier le parcours "
        "du client et récupérer des données fiables sur la fréquentation. "
        "Les premiers résultats montrent une hausse de dix-huit pour cent des "
        "séjours hors saison, portée surtout par une clientèle de proximité "
        "venue de Lyon, de Genève et de Grenoble. Reste la question du "
        "financement : les budgets communaux sont contraints, et les "
        "partenaires privés attendent des garanties avant de s'engager sur "
        "plusieurs années.",
    ),
    "synth_long_dictee_pro": (
        "Thomas",
        "Compte rendu de la réunion du comité de pilotage. Présents : la "
        "directrice générale, le responsable informatique, deux représentants "
        "des utilisateurs et le prestataire. Premier point, l'état "
        "d'avancement du projet. Le lot un est livré, le lot deux est en "
        "recette depuis lundi, et le lot trois démarre le quinze janvier. "
        "Deuxième point, les risques. Le prestataire signale un retard "
        "possible de trois semaines sur la reprise des données, lié à la "
        "qualité des fichiers sources. Décision : une relecture manuelle des "
        "cinq cents fiches les plus anciennes sera confiée au service "
        "archives. Troisième point, le budget. L'enveloppe initiale de deux "
        "cent mille euros est consommée à soixante-cinq pour cent, ce qui "
        "est conforme au calendrier. Prochaine réunion le huit février à "
        "dix heures, en visioconférence.",
    ),
}


# ----------------------------------------------------------------------
# Modèles comparés
# ----------------------------------------------------------------------

BACKEND_VOXTRAL = "voxtral"          # mlx-voxtral, comme la dictée actuelle
BACKEND_WHISPER = "whisper"          # mlx-whisper, comme les fichiers actuels
BACKEND_MLX_AUDIO = "mlx_audio"      # candidats, via mlx_audio.stt.load


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    backend: str
    label: str
    reference: bool = False          # modèle déjà dans l'app (point de comparaison)
    language_arg: str | None = None  # valeur à passer pour imposer le français
    cast_bf16: bool = False          # Parakeet : poids fp32 convertis en bf16
    aligner_repo: str | None = None  # Qwen3 : aligneur pour les horodatages
    chunk_s: float | None = None     # Parakeet : découpage en blocs (secondes)


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("voxtral-mini-3b-8bit", "mzbac/voxtral-mini-3b-8bit",
                  BACKEND_VOXTRAL, "Voxtral Mini 3B 8-bit (dictée actuelle)",
                  reference=True, language_arg="fr"),
        ModelSpec("voxtral-mini-3b-4bit", "mzbac/voxtral-mini-3b-4bit-mixed",
                  BACKEND_VOXTRAL, "Voxtral Mini 3B 4-bit (catalogue)",
                  reference=True, language_arg="fr"),
        ModelSpec("whisper-large-v3-turbo", "mlx-community/whisper-large-v3-turbo",
                  BACKEND_WHISPER, "Whisper large-v3-turbo (fichiers actuels)",
                  reference=True, language_arg="fr"),
        ModelSpec("whisper-large-v3", "mlx-community/whisper-large-v3-mlx",
                  BACKEND_WHISPER, "Whisper large-v3 (traduction)",
                  reference=True, language_arg="fr"),
        ModelSpec("voxtral-rt-4b-4bit",
                  "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
                  BACKEND_MLX_AUDIO, "Voxtral Mini 4B Realtime 4-bit (Apache 2.0)"),
        ModelSpec("qwen3-asr-1.7b-8bit", "mlx-community/Qwen3-ASR-1.7B-8bit",
                  BACKEND_MLX_AUDIO, "Qwen3-ASR 1.7B 8-bit (Apache 2.0)",
                  language_arg="French"),
        ModelSpec("qwen3-asr-1.7b-4bit", "mlx-community/Qwen3-ASR-1.7B-4bit",
                  BACKEND_MLX_AUDIO, "Qwen3-ASR 1.7B 4-bit (Apache 2.0)",
                  language_arg="French"),
        ModelSpec("qwen3-asr-1.7b-4bit+aligner", "mlx-community/Qwen3-ASR-1.7B-4bit",
                  BACKEND_MLX_AUDIO, "Qwen3-ASR 4-bit + aligneur 0.6B (horodatages)",
                  language_arg="French",
                  aligner_repo="mlx-community/Qwen3-ForcedAligner-0.6B-4bit"),
        ModelSpec("parakeet-tdt-0.6b-v3", "mlx-community/parakeet-tdt-0.6b-v3",
                  BACKEND_MLX_AUDIO, "Parakeet TDT 0.6B v3 (CC BY 4.0, fp32)"),
        ModelSpec("parakeet-tdt-0.6b-v3-bf16", "mlx-community/parakeet-tdt-0.6b-v3",
                  BACKEND_MLX_AUDIO, "Parakeet TDT 0.6B v3 converti en bf16",
                  cast_bf16=True),
        # Sans découpage, Parakeet saute des phrases entières sur un
        # enregistrement de 2 min (mesuré) : variante par blocs de 60 s,
        # recouvrement de 2 s (défaut mlx-audio), bf16.
        ModelSpec("parakeet-tdt-0.6b-v3-bf16-blocs60",
                  "mlx-community/parakeet-tdt-0.6b-v3",
                  BACKEND_MLX_AUDIO, "Parakeet TDT 0.6B v3 bf16, blocs de 60 s",
                  cast_bf16=True, chunk_s=60.0),
        ModelSpec("parakeet-tdt-0.6b-v3-bf16-blocs20",
                  "mlx-community/parakeet-tdt-0.6b-v3",
                  BACKEND_MLX_AUDIO, "Parakeet TDT 0.6B v3 bf16, blocs de 20 s",
                  cast_bf16=True, chunk_s=20.0),
    ]
}


# ----------------------------------------------------------------------
# Résultat d'un couple (modèle, fichier)
# ----------------------------------------------------------------------

@dataclass
class RunResult:
    model: str
    sample: str
    audio_s: float
    offline: bool = False
    ok: bool = False
    error: str = ""
    load_s: float = 0.0
    cold_s: float = 0.0
    warm_s: float = 0.0
    peak_rss_mb: float = 0.0
    mlx_peak_mb: float = 0.0
    text: str = ""
    segments: list[dict] = field(default_factory=list)
    timestamps: str = "non"          # "non", "bloc", "phrase", "mot"
    wer: float | None = None
    ref_words: int = 0
    errors: int = 0

    @property
    def rtf(self) -> float:
        """Facteur temps réel : 10 = une minute d'audio en six secondes."""
        return self.audio_s / self.warm_s if self.warm_s > 0 else 0.0


# ----------------------------------------------------------------------
# WER
# ----------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
_THOUSANDS = re.compile(r"(?<=\d)[  ](?=\d{3}\b)")
_HOUR_MIN = re.compile(r"\b(\d+)\s*h\s*(\d+)\b")
_HOUR = re.compile(r"\b(\d+)\s*h\b")

# Nombres en toutes lettres → chiffres. Les modèles écrivent « 14h30 », « 18 % »
# ou « 65 pour cent » là où la référence dictée à `say` dit « quatorze heures
# trente » : sans cette étape, la moitié des « erreurs » seraient des
# différences de notation, pas d'écoute. Appliqué aux deux côtés.
_NUM_UNITS = {
    "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
    "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
}
_NUM_TENS = {
    "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60, "quatrevingt": 80, "quatrevingts": 80,
}


def _words_to_digits(words: list[str]) -> list[str]:
    """« deux cent mille » → « 200000 », « quatre vingt dix neuf » → « 99 »."""
    out: list[str] = []
    i = 0
    while i < len(words):
        total = 0
        current = 0
        j = i
        consumed = False
        while j < len(words):
            w = words[j]
            if w in _NUM_UNITS:
                current += _NUM_UNITS[w]
            elif w in _NUM_TENS:
                current += _NUM_TENS[w]
            elif w == "cent" or w == "cents":
                current = (current or 1) * 100
            elif w == "mille":
                total += (current or 1) * 1000
                current = 0
            elif (w == "et" and current in (20, 30, 40, 50, 60)
                  and j + 1 < len(words) and words[j + 1] in _NUM_UNITS):
                j += 1  # « vingt et un », « soixante et onze »
                continue
            else:
                break
            consumed = True
            j += 1
        if consumed:
            out.append(str(total + current))
            i = j
        else:
            out.append(words[i])
            i += 1
    return out


def normalize_words(text: str) -> list[str]:
    """Mots comparables : casse, accents composés, ponctuation, élisions et
    notation des nombres neutralisées.
    « L'outil, 3 500 € à 14h30 ! » → ['l', 'outil', '3500', 'a', '14', 'heures', '30']."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("œ", "oe")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = _THOUSANDS.sub("", text)
    # « pour cent » et « % » deviennent un mot à part, pour que « cent » n'y
    # soit pas lu comme un nombre ; « quatre-vingt » devient un seul mot.
    text = text.replace("%", " pourcent ").replace("pour cent", "pourcent")
    text = re.sub(r"quatre[\s-]vingt", "quatrevingt", text)
    text = _HOUR_MIN.sub(r"\1 heures \2", text)
    text = _HOUR.sub(r"\1 heures", text)
    # La ponctuation devient une barrière « | » le temps de lire les nombres :
    # « trois, quatre » reste deux nombres, pas sept.
    text = text.replace("-", " ")  # « dix-huit » doit rester un seul nombre
    text = _PUNCT.sub(" | ", text)
    text = text.replace("'", " ")
    words = _words_to_digits(text.split())
    return [("heures" if w == "heure" else w) for w in words if w != "|"]


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Distance de Levenshtein entre deux listes de mots."""
    prev = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        cur = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, start=1):
            cost = 0 if ref_word == hyp_word else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """(taux, mots de référence, erreurs). Taux = 0 si la référence est vide."""
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0, 0, len(hyp)
    errors = word_errors(ref, hyp)
    return errors / len(ref), len(ref), errors


# ----------------------------------------------------------------------
# Worker : un modèle, un fichier, dans un process neuf
# ----------------------------------------------------------------------

def _local_source(repo: str) -> Path | str:
    """Dossier local du modèle s'il est en cache, sinon l'identifiant du repo.

    On rend un `Path` et pas une chaîne : `mlx_audio` déduit le type de
    certains modèles (Parakeet, sans `model_type` dans config.json) du NOM du
    dossier de cache, et il ne sait le faire que sur un `Path`.
    """
    import hf_offline

    resolved = hf_offline.resolve_local_path(repo)
    return Path(resolved) if resolved != repo else repo


def _load_backend(spec: ModelSpec):
    """Charge le modèle et rend `transcribe(wav_path) -> (texte, segments)`."""
    if spec.backend == BACKEND_VOXTRAL:
        from transcriber import VoxtralTranscriber

        t = VoxtralTranscriber(spec.repo)
        t.preload()

        def run(wav: Path):
            return t.transcribe(wav, language=spec.language_arg or "auto"), []

        return run

    if spec.backend == BACKEND_WHISPER:
        import mlx.core as mx
        import soundfile as sf
        from mlx_whisper.transcribe import ModelHolder

        import hf_offline
        from transcriber import WhisperTranscriber

        # mlx_whisper charge le modèle au premier transcribe() ; on force le
        # chargement ici pour le chronométrer à part.
        ModelHolder.get_model(hf_offline.resolve_local_path(spec.repo), mx.float16)
        t = WhisperTranscriber(spec.repo)

        def run(wav: Path):
            audio, sr = sf.read(str(wav), dtype="float32")
            result = t.transcribe_array(audio, sr, language=spec.language_arg)
            segs = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in result.segments
            ]
            return result.text, segs

        return run

    # Candidats via mlx-audio.
    import mlx.core as mx
    from mlx_audio.stt.utils import load

    model = load(_local_source(spec.repo))
    if spec.cast_bf16:
        from mlx.utils import tree_map

        model.update(
            tree_map(
                lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a,
                model.parameters(),
            )
        )
        mx.eval(model.parameters())
    aligner = load(_local_source(spec.aligner_repo)) if spec.aligner_repo else None

    def run(wav: Path):
        kwargs = {"language": spec.language_arg} if spec.language_arg else {}
        if spec.chunk_s:
            kwargs["chunk_duration"] = spec.chunk_s
        out = model.generate(str(wav), **kwargs)
        text = str(out.text).strip()
        segs: list[dict] = []
        if hasattr(out, "sentences"):  # Parakeet : AlignedResult
            segs = [
                {"start": float(s.start), "end": float(s.end), "text": s.text,
                 "words": len(s.tokens)}
                for s in out.sentences
            ]
        elif getattr(out, "segments", None):
            segs = [
                {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
                 "text": str(s.get("text", ""))}
                for s in out.segments
            ]
        if aligner is not None and text:
            aligned = aligner.generate(str(wav), text, language=spec.language_arg)
            segs = [
                {"start": float(i.start_time), "end": float(i.end_time),
                 "text": i.text, "word": True}
                for i in aligned.items
            ]
        return text, segs

    return run


def _timestamp_grain(spec: ModelSpec, segments: list[dict], audio_s: float) -> str:
    if not segments:
        return "non"
    if any(s.get("word") for s in segments):
        return "mot"
    if spec.backend == BACKEND_MLX_AUDIO and any("words" in s for s in segments):
        return "phrase"
    if spec.backend == BACKEND_WHISPER:
        return "phrase"
    # Un seul segment qui couvre tout : ce n'est pas un vrai horodatage.
    if len(segments) == 1 and segments[0]["end"] - segments[0]["start"] >= audio_s - 1:
        return "non"
    return "bloc"


def run_worker(model_key: str, wav: Path, out_path: Path, offline: bool) -> None:
    import resource

    if offline:
        # Même hypothèse que hf_offline : hors-ligne forcé ET hub injoignable,
        # pour qu'une requête réseau échoue vite au lieu de passer.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"

    sys.path.insert(0, str(ROOT))
    import soundfile as sf

    spec = MODELS[model_key]
    audio_s = float(sf.info(str(wav)).duration)
    result = RunResult(model=model_key, sample=wav.stem, audio_s=audio_s, offline=offline)

    try:
        t0 = time.perf_counter()
        run = _load_backend(spec)
        result.load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        text, segments = run(wav)
        result.cold_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        text, segments = run(wav)
        result.warm_s = time.perf_counter() - t0

        result.text = text
        result.segments = segments
        result.timestamps = _timestamp_grain(spec, segments, audio_s)
        result.ok = True
    except Exception as exc:  # noqa: BLE001 — on veut l'erreur dans le rapport
        import traceback

        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"

    # macOS rend ru_maxrss en octets.
    result.peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    try:
        import mlx.core as mx

        result.mlx_peak_mb = mx.get_peak_memory() / 2**20
    except Exception:  # noqa: BLE001
        pass

    out_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------
# Parent : jeu d'essai, orchestration, rapport
# ----------------------------------------------------------------------

def prepare_synth_samples() -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    for name, (voice, text) in SYNTH_SAMPLES.items():
        ref = SAMPLES_DIR / f"{name}.txt"
        if not ref.exists():
            ref.write_text(text + "\n")
        wav = WAV_DIR / f"{name}.wav"
        if wav.exists():
            continue
        subprocess.run(
            ["say", "-v", voice, "-o", str(wav), "--data-format=LEI16@16000", text],
            check=True,
        )
        print(f"[bench] synthèse : {wav.name} ({voice})")


def list_samples(only: list[str] | None) -> list[Path]:
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if only:
        wavs = [w for w in wavs if w.stem in only]
    return wavs


def result_path(model_key: str, sample: str, offline: bool) -> Path:
    suffix = "__offline" if offline else ""
    return RESULTS_DIR / f"{model_key}__{sample}{suffix}.json"


def run_pair(model_key: str, wav: Path, offline: bool, force: bool) -> RunResult:
    out = result_path(model_key, wav.stem, offline)
    if out.exists() and not force:
        return RunResult(**json.loads(out.read_text()))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(Path(__file__).resolve()), "--worker", model_key,
           str(wav), str(out)]
    if offline:
        cmd.append("--offline")
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=WORKER_TIMEOUT_S,
        )
        if not out.exists():
            r = RunResult(model=model_key, sample=wav.stem, audio_s=0.0, offline=offline,
                          error=f"worker sans résultat (code {proc.returncode})\n"
                                f"{proc.stderr[-2000:]}")
            out.write_text(json.dumps(asdict(r), ensure_ascii=False, indent=1))
    except subprocess.TimeoutExpired:
        r = RunResult(model=model_key, sample=wav.stem, audio_s=0.0, offline=offline,
                      error=f"abandonné après {WORKER_TIMEOUT_S} s")
        out.write_text(json.dumps(asdict(r), ensure_ascii=False, indent=1))
    return RunResult(**json.loads(out.read_text()))


def score(result: RunResult) -> RunResult:
    ref = SAMPLES_DIR / f"{result.sample}.txt"
    if result.ok and ref.exists():
        result.wer, result.ref_words, result.errors = wer(ref.read_text(), result.text)
    return result


def _fmt_wer(r: RunResult) -> str:
    if not r.ok:
        return "ÉCHEC"
    if r.wer is None:
        return "—"
    return f"{100 * r.wer:.1f} %"


def _disk_gb(repo: str) -> float:
    sys.path.insert(0, str(ROOT))
    import model_manager

    return model_manager.cached_size_bytes(repo) / 1e9


def _is_real(sample: str) -> bool:
    return sample.startswith("reel_")


def write_report(results: list[RunResult], offline: list[RunResult]) -> str:
    lines: list[str] = []
    models = [k for k in MODELS if any(r.model == k for r in results)]

    lines.append("## Synthèse par modèle\n")
    lines.append("| Modèle | Disque | WER synthé | WER réel | Temps réel (× à chaud) "
                 "| 1re transcription 10 s | Chargement | Pic RAM | Horodatages | Hors-ligne |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in models:
        spec = MODELS[key]
        rs = [r for r in results if r.model == key and r.ok]
        if not rs:
            errs = [r for r in results if r.model == key]
            msg = errs[0].error.splitlines()[0] if errs and errs[0].error else "?"
            lines.append(f"| {spec.label} | | ÉCHEC : {msg} | | | | | | | |")
            continue
        synth = [r.wer for r in rs if not _is_real(r.sample) and r.wer is not None]
        real = [r.wer for r in rs if _is_real(r.sample) and r.wer is not None]
        rtfs = [r.rtf for r in rs if r.warm_s > 0]
        short = [r for r in rs if 4 <= r.audio_s <= 15]
        cold = (sum(r.cold_s for r in short) / len(short)) if short else 0.0
        load = sum(r.load_s for r in rs) / len(rs)
        peak = max(r.peak_rss_mb for r in rs)
        grains = sorted({r.timestamps for r in rs})
        off = [o for o in offline if o.model == key]
        off_txt = ("oui" if off and off[0].ok else
                   ("NON : " + off[0].error.splitlines()[0]) if off else "non testé")
        lines.append(
            f"| {spec.label} | {_disk_gb(spec.repo):.1f} Go "
            f"| {(f'{100 * sum(synth) / len(synth):.1f} %' if synth else '—')} | "
            f"{(f'{100 * sum(real) / len(real):.1f} %' if real else '—')} "
            f"| {sum(rtfs) / len(rtfs):.0f}× (min {min(rtfs):.0f}×) "
            f"| {cold:.1f} s | {load:.1f} s | {peak / 1024:.1f} Go "
            f"| {', '.join(grains)} | {off_txt} |"
        )

    lines.append("\n## Détail par fichier\n")
    lines.append("| Modèle | Fichier | Durée | WER | Erreurs / mots | À chaud | À froid | Temps réel |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in models:
        for r in sorted((r for r in results if r.model == key), key=lambda r: r.sample):
            if not r.ok:
                first = r.error.splitlines()[0] if r.error else "?"
                lines.append(f"| {key} | {r.sample} | | ÉCHEC | {first} | | | |")
                continue
            lines.append(
                f"| {key} | {r.sample} | {r.audio_s:.0f} s | {_fmt_wer(r)} "
                f"| {r.errors} / {r.ref_words} | {r.warm_s:.2f} s | {r.cold_s:.2f} s "
                f"| {r.rtf:.0f}× |"
            )

    # Les textes des enregistrements réels restent dans les JSON (ignorés par
    # git) : ce sont des paroles privées, le dépôt est public.
    lines.append("\n## Textes produits (phrases synthétiques seulement)\n")
    for key in models:
        lines.append(f"### {MODELS[key].label}\n")
        for r in sorted((r for r in results if r.model == key and r.ok
                         and not _is_real(r.sample)), key=lambda r: r.sample):
            lines.append(f"- **{r.sample}** ({_fmt_wer(r)}) : {r.text}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--models", nargs="*", help="clés de MODELS (défaut : tous)")
    parser.add_argument("--samples", nargs="*", help="noms de fichiers sans .wav")
    parser.add_argument("--force", action="store_true", help="ignorer les résultats en cache")
    parser.add_argument("--offline", action="store_true",
                        help="ajouter le passage hors-ligne sur le fichier le plus court")
    parser.add_argument("--report", action="store_true", help="n'écrire que le rapport")
    parser.add_argument("--worker", nargs=3, metavar=("MODEL", "WAV", "OUT"))
    parser.add_argument("--offline-worker", action="store_true", dest="offline_flag")
    args = parser.parse_args(argv)

    if args.worker:
        model_key, wav, out = args.worker
        run_worker(model_key, Path(wav), Path(out), offline=args.offline_flag)
        return 0

    prepare_synth_samples()
    models = args.models or list(MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        print(f"[bench] modèles inconnus : {unknown}", file=sys.stderr)
        return 2
    samples = list_samples(args.samples)

    results: list[RunResult] = []
    offline: list[RunResult] = []
    if not args.report:
        for key in models:
            for wav in samples:
                print(f"[bench] {key} × {wav.stem} …", flush=True)
                r = score(run_pair(key, wav, offline=False, force=args.force))
                state = f"WER {_fmt_wer(r)}, {r.rtf:.0f}×" if r.ok else "ÉCHEC"
                print(f"[bench]   → {state}", flush=True)
                results.append(r)
            if args.offline and samples:
                shortest = min(samples, key=lambda w: w.stat().st_size)
                print(f"[bench] {key} × {shortest.stem} HORS-LIGNE …", flush=True)
                offline.append(run_pair(key, shortest, offline=True, force=args.force))
    else:
        for path in sorted(RESULTS_DIR.glob("*.json")):
            r = RunResult(**json.loads(path.read_text()))
            if r.model not in models:
                continue
            (offline if r.offline else results).append(score(r) if not r.offline else r)

    report = write_report(results, offline)
    (RESULTS_DIR / "summary.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    # `--offline` du worker est transmis en dernier argument ; on le distingue
    # de l'option du parent pour ne pas confondre les deux usages.
    if "--worker" in sys.argv and sys.argv[-1] == "--offline":
        sys.argv[-1] = "--offline-worker"
    sys.exit(main())
