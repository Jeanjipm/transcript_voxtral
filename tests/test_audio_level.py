"""Tests d'audio_level.py — mesure de niveau (RMS de la fenêtre la plus forte).

Les valeurs attendues sont celles de la théorie du signal : une sinusoïde
pleine échelle a un RMS de 1/√2, soit −3,01 dBFS.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_level import (
    DEFAULT_SILENCE_THRESHOLD_DBFS,
    SILENCE_FILTER_OFF_DBFS,
    is_silent,
    measure_peak_rms_dbfs,
    peak_rms_dbfs,
)


SR = 16_000


def _sine(seconds: float, amplitude: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return amplitude * np.sin(2 * np.pi * freq * t)


# ---- peak_rms_dbfs ----


def test_empty_is_minus_infinity():
    assert peak_rms_dbfs(np.zeros(0), SR) == -math.inf


def test_digital_silence_is_minus_infinity():
    assert peak_rms_dbfs(np.zeros(SR * 2, dtype="int16"), SR) == -math.inf


def test_full_scale_sine_is_minus_3_dbfs():
    level = peak_rms_dbfs(_sine(1.0, 1.0), SR)
    assert level == pytest.approx(-3.01, abs=0.05)


def test_minus_20_db_sine_is_minus_23_dbfs():
    level = peak_rms_dbfs(_sine(1.0, 0.1), SR)
    assert level == pytest.approx(-23.01, abs=0.05)


def test_int16_is_normalised_like_float():
    """Un WAV 16 bits et le même signal en flottant donnent le même niveau :
    la mesure ne dépend pas du format de stockage."""
    as_float = _sine(0.5, 0.5)
    as_int16 = (as_float * 32768).astype("int16")
    assert peak_rms_dbfs(as_int16, SR) == pytest.approx(
        peak_rms_dbfs(as_float, SR), abs=0.01
    )


def test_stereo_is_averaged_to_mono():
    mono = _sine(0.5, 0.5)
    stereo = np.stack([mono, mono], axis=1)
    assert peak_rms_dbfs(stereo, SR) == pytest.approx(peak_rms_dbfs(mono, SR))


def test_short_burst_in_long_silence_is_measured_at_its_own_level():
    """LE cas qui justifie la fenêtre glissante : un « oui » de 150 ms après
    trois secondes de réflexion. Sur la moyenne du fichier, ce serait du
    silence (−16 dB de dilution) ; sur la fenêtre la plus forte, c'est une
    voix à son vrai niveau."""
    silence = np.zeros(SR * 3)
    burst = _sine(0.15, 0.1)  # −23 dBFS
    audio = np.concatenate([silence, burst, np.zeros(SR * 3)])

    whole_file_rms = 20 * math.log10(math.sqrt(float(np.mean(audio * audio))))
    assert whole_file_rms < -35  # la moyenne ment

    assert peak_rms_dbfs(audio, SR) == pytest.approx(-23.01, abs=0.1)


def test_tail_shorter_than_a_window_still_counts():
    """Un mot prononcé dans les 50 dernières millisecondes n'est pas dilué
    ni ignoré : le bout de fin compte comme une fenêtre à lui seul."""
    audio = np.concatenate([np.zeros(SR), _sine(0.05, 0.1)])
    assert peak_rms_dbfs(audio, SR, window_ms=100) == pytest.approx(
        -23.01, abs=0.1
    )


def test_window_size_changes_resolution():
    """Fenêtre de 1 s sur une salve de 100 ms : dilution ×10 → −10 dB."""
    audio = np.concatenate([_sine(0.1, 0.1), np.zeros(int(SR * 0.9))])
    fine = peak_rms_dbfs(audio, SR, window_ms=100)
    coarse = peak_rms_dbfs(audio, SR, window_ms=1000)
    assert fine == pytest.approx(-23.01, abs=0.1)
    assert coarse == pytest.approx(fine - 10.0, abs=0.1)


def test_realistic_room_noise_is_below_default_threshold():
    """Bruit blanc à −60 dBFS (pièce calme sur un micro de MacBook) : jeté.
    Le seuil par défaut doit laisser une marge nette de chaque côté."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.001, SR * 2)  # RMS 0.001 → −60 dBFS
    level = peak_rms_dbfs(noise, SR)
    assert level < DEFAULT_SILENCE_THRESHOLD_DBFS - 10


def test_quiet_voice_is_above_default_threshold():
    """Voix douce à −35 dBFS : transmise. Jeter une vraie phrase serait pire
    qu'une hallucination de temps en temps."""
    level = peak_rms_dbfs(_sine(1.0, 0.025), SR)  # ≈ −35 dBFS
    assert level > DEFAULT_SILENCE_THRESHOLD_DBFS + 5


# ---- measure_peak_rms_dbfs (fichier) ----


def test_measure_reads_a_real_wav(tmp_path: Path):
    path = tmp_path / "voix.wav"
    sf.write(path, (_sine(1.0, 0.5) * 32767).astype("int16"), SR, subtype="PCM_16")
    assert measure_peak_rms_dbfs(path) == pytest.approx(-9.03, abs=0.05)


def test_measure_silent_wav(tmp_path: Path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros((SR, 1), dtype="int16"), SR, subtype="PCM_16")
    assert measure_peak_rms_dbfs(path) == -math.inf


def test_measure_unreadable_wav_raises(tmp_path: Path):
    """L'appelant décide (le contrôleur transcrit quand même)."""
    path = tmp_path / "corrompu.wav"
    path.write_bytes(b"pas un wav")
    with pytest.raises(sf.LibsndfileError):
        measure_peak_rms_dbfs(path)


# ---- is_silent ----


@pytest.mark.parametrize(
    "level, threshold, expected",
    [
        (-60.0, -45.0, True),
        (-45.0, -45.0, False),  # au seuil = transmis
        (-30.0, -45.0, False),
        (-math.inf, -45.0, True),
        (-math.inf, SILENCE_FILTER_OFF_DBFS, False),  # filtre coupé
        (-200.0, -130.0, False),  # sous -120 : coupé aussi
    ],
)
def test_is_silent(level, threshold, expected):
    assert is_silent(level, threshold) is expected
