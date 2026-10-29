"""
Niveau sonore d'un enregistrement : y a-t-il eu de la voix, ou que du
silence ?

## Pourquoi

Les modèles de reconnaissance vocale n'ont pas de notion de « rien » : donnés
deux secondes de silence, ils inventent la phrase la plus probable de leurs
données d'entraînement — « Merci. », « Thank you », « Sous-titres réalisés
par… ». Collée telle quelle au curseur, c'est pire que de ne rien coller. Le
seul garde jusqu'ici était la durée (0,5 s).

## La mesure : RMS de la fenêtre de 100 ms la plus forte

On découpe l'audio en fenêtres de 100 ms et on garde l'énergie (RMS) de la
plus forte. Pas la moyenne du fichier : un « oui » prononcé après trois
secondes de réflexion aurait une moyenne de silence, alors que sa fenêtre la
plus forte est clairement de la voix.

Unité : dBFS, décibels sous la pleine échelle. 0 dBFS = saturation ; une
sinusoïde pleine échelle vaut −3 dBFS ; le silence numérique vaut −∞. Repères
sur un micro de MacBook : bruit d'une pièce calme entre −70 et −50 dBFS, voix
normale entre −30 et −15 dBFS sur sa fenêtre la plus forte.

Fonctions pures, sans état : testables avec des tableaux numpy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


# Fenêtre d'analyse : assez courte pour isoler une syllabe, assez longue pour
# qu'un clic de souris ou un claquement de touche ne pèse pas comme une voix.
DEFAULT_WINDOW_MS = 100

# Seuil (dBFS) en dessous duquel on considère que l'enregistrement est du
# silence. Volontairement bas : jeter une vraie phrase dite doucement serait
# pire qu'une hallucination de temps en temps.
DEFAULT_SILENCE_THRESHOLD_DBFS = -45.0

# À ce seuil ou en dessous, le filtre est désactivé : aucun micro réel ne
# produit un signal aussi faible, donc rien n'est jamais jeté. Valeur à
# mettre dans config.yaml pour couper le filtre.
SILENCE_FILTER_OFF_DBFS = -120.0


def _to_float(samples: np.ndarray) -> np.ndarray:
    """Ramène n'importe quel format PCM sur [-1, 1] en mono."""
    if samples.dtype.kind in ("i", "u"):
        # int16 : diviser par 32768 pour qu'une pleine échelle vaille 1.0
        scale = float(np.iinfo(samples.dtype).max) + 1.0
        audio = samples.astype(np.float64) / scale
    else:
        audio = samples.astype(np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def peak_rms_dbfs(
    samples: np.ndarray,
    sample_rate: int,
    window_ms: int = DEFAULT_WINDOW_MS,
) -> float:
    """Énergie (RMS, en dBFS) de la fenêtre de `window_ms` la plus forte.

    Retourne −inf pour un tableau vide ou entièrement nul. Un dernier bout
    plus court qu'une fenêtre compte comme une fenêtre à lui seul : on ne
    veut pas qu'un mot prononcé en toute fin d'enregistrement soit dilué.
    """
    audio = _to_float(samples)
    n = audio.shape[0]
    if n == 0:
        return -math.inf

    window = max(1, int(sample_rate * window_ms / 1000))
    power = audio * audio
    full = (n // window) * window
    peak = 0.0
    if full > 0:
        peak = float(power[:full].reshape(-1, window).mean(axis=1).max())
    if full < n:
        peak = max(peak, float(power[full:].mean()))

    if peak <= 0.0:
        return -math.inf
    return 10.0 * math.log10(peak)


def measure_peak_rms_dbfs(
    wav_path: Path, window_ms: int = DEFAULT_WINDOW_MS
) -> float:
    """Même mesure, sur un fichier WAV. Lève si le fichier est illisible :
    c'est à l'appelant de décider quoi faire (le contrôleur transcrit alors
    quand même, comme avant ce filtre)."""
    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    return peak_rms_dbfs(np.asarray(audio), int(sample_rate), window_ms)


def is_silent(level_dbfs: float, threshold_dbfs: float) -> bool:
    """Décision : `level_dbfs` est-il sous le seuil ? Le filtre est désactivé
    (jamais silencieux) quand le seuil vaut `SILENCE_FILTER_OFF_DBFS` ou
    moins."""
    if threshold_dbfs <= SILENCE_FILTER_OFF_DBFS:
        return False
    return level_dbfs < threshold_dbfs
