"""
Amener n'importe quel fichier audio en 16 kHz mono lisible par soundfile.

## Pourquoi ce module existe

`libsndfile` (le moteur derrière soundfile) ne lit pas le `.m4a` — or c'est
exactement ce que produit Dictaphone sur macOS et iPhone, donc le format le
plus probable en entrée. Et `ffmpeg` n'est pas installé, ni installable sans
ajouter une dépendance système à un projet qui s'en passe.

La réponse est `/usr/bin/afconvert`, livré avec macOS : il décode tout ce que
CoreAudio sait lire (m4a/AAC, mp3, wav, aiff, caf…) et produit un WAV 16 kHz
mono que soundfile relit sans problème. Zéro nouvelle dépendance.

Vérifié de bout en bout :
    afconvert -f WAVE -d LEI16@16000 -c 1 entree.m4a sortie.wav

## Lecture par blocs

Les fichiers longs ne sont jamais chargés entièrement : `read_block` utilise
`sf.SoundFile.seek()` puis `.read(frames=…)`. Un enregistrement de 2 h fait
460 Mo en float32 — on n'en garde qu'un bloc de ~2 Mo à la fois.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


AFCONVERT = "/usr/bin/afconvert"

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

# Une conversion est bornée dans le temps : un fichier corrompu peut faire
# tourner afconvert indéfiniment. 30 min couvrent largement plusieurs heures
# d'audio (afconvert est bien plus rapide que le temps réel).
CONVERT_TIMEOUT_S = 1800

# Extensions proposées dans le sélecteur de fichiers. Tout ce que CoreAudio
# sait décoder ; la liste sert à filtrer le dialogue, pas à valider (on tente
# la conversion de toute façon).
SUPPORTED_EXTENSIONS = (
    "wav", "m4a", "mp3", "aiff", "aif", "caf", "flac", "ogg",
    "mp4", "mov", "aac", "m4b", "wma",
)


class AudioConversionError(RuntimeError):
    """Conversion impossible — message destiné à l'utilisateur, en français."""


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_s: float
    sample_rate: int
    channels: int

    @property
    def needs_conversion(self) -> bool:
        return (
            self.sample_rate != TARGET_SAMPLE_RATE
            or self.channels != TARGET_CHANNELS
        )


def probe(path: Path) -> AudioInfo | None:
    """Lit les caractéristiques du fichier, ou None si soundfile ne sait pas.

    None ne veut pas dire « fichier invalide » : c'est le cas normal pour un
    `.m4a`, que libsndfile refuse et qu'`afconvert` traitera très bien.
    """
    try:
        info = sf.info(str(path))
    except Exception:  # noqa: BLE001 — LibsndfileError et parents varient
        return None
    return AudioInfo(
        path=path,
        duration_s=float(info.duration),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
    )


def probe_duration(path: Path) -> float | None:
    """Durée en secondes, y compris pour les formats que soundfile refuse.

    Passe par `afinfo` (également livré avec macOS) en dernier recours, ce qui
    permet de refuser un fichier trop long AVANT de payer la conversion.
    """
    info = probe(path)
    if info is not None:
        return info.duration_s

    try:
        result = subprocess.run(
            ["/usr/bin/afinfo", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    # afinfo écrit une ligne « estimated duration: 123.456 sec ».
    for line in result.stdout.splitlines():
        if "duration" in line.lower():
            for token in line.replace(":", " ").split():
                try:
                    return float(token)
                except ValueError:
                    continue
    return None


def ensure_16k_mono(src: Path) -> tuple[Path, bool]:
    """Retourne `(chemin utilisable, est_temporaire)`.

    Si le fichier est déjà en 16 kHz mono et lisible par soundfile, on le rend
    tel quel sans rien copier. Sinon on convertit via `afconvert` vers un WAV
    temporaire, et l'appelant est responsable de le supprimer (le booléen
    retourné le lui dit).
    """
    if not src.exists():
        raise AudioConversionError(f"Fichier introuvable : {src}")

    info = probe(src)
    if info is not None and not info.needs_conversion:
        return src, False

    fd, out_str = tempfile.mkstemp(suffix=".wav", prefix="voxtral_conv_")
    os.close(fd)
    out = Path(out_str)

    try:
        _run_afconvert(src, out)
    except AudioConversionError:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return out, True


def _run_afconvert(src: Path, dest: Path) -> None:
    """Lance afconvert. Lève AudioConversionError avec un message utile.

    On remonte le stderr d'afconvert dans le message : une conversion qui
    échoue en silence serait le bug le plus difficile à diagnostiquer de cette
    fonctionnalité.
    """
    cmd = [
        AFCONVERT,
        "-f", "WAVE",
        "-d", f"LEI16@{TARGET_SAMPLE_RATE}",
        "-c", str(TARGET_CHANNELS),
        str(src),
        str(dest),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=CONVERT_TIMEOUT_S, check=False,
        )
    except FileNotFoundError as exc:
        raise AudioConversionError(
            f"{AFCONVERT} est introuvable. C'est un outil livré avec macOS ; "
            f"son absence indique une installation inhabituelle."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioConversionError(
            f"La conversion de {src.name} a dépassé "
            f"{CONVERT_TIMEOUT_S // 60} minutes — fichier probablement corrompu."
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AudioConversionError(
            f"Impossible de convertir {src.name} : format audio non reconnu "
            f"par macOS.\n\nDétail : {detail[:300]}"
        )

    if not dest.exists() or dest.stat().st_size == 0:
        raise AudioConversionError(
            f"La conversion de {src.name} n'a produit aucune donnée audio. "
            f"Le fichier est peut-être vide ou ne contient pas de piste audio."
        )


def read_block(path: Path, start_s: float, duration_s: float) -> np.ndarray:
    """Lit `duration_s` secondes à partir de `start_s`, en float32 mono.

    Le fichier doit déjà être en 16 kHz (cf. `ensure_16k_mono`). On lit par
    positionnement plutôt qu'en chargeant tout : la mémoire reste
    proportionnelle au bloc, pas au fichier.

    Retourne un tableau vide si `start_s` est au-delà de la fin.
    """
    with sf.SoundFile(str(path)) as handle:
        start_frame = int(start_s * handle.samplerate)
        if start_frame >= handle.frames:
            return np.zeros(0, dtype="float32")
        handle.seek(start_frame)
        frames = int(duration_s * handle.samplerate)
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)

    # Repli mono : Whisper et Voxtral attendent un signal à une dimension.
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio.reshape(-1)
    return audio.astype("float32", copy=False)
