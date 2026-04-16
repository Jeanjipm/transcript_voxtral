"""
Capture micro pour Voxtral Dictée.

On enregistre en mono 16 kHz (format attendu par Voxtral et Whisper),
on tamponne en RAM dans une liste de chunks numpy, puis on écrit un WAV
temporaire à `stop()`. Pas de conversion ni resampling : sounddevice gère.

Pourquoi 16 kHz ? C'est la fréquence d'échantillonnage native des modèles
de speech-to-text grand public (économie de calcul vs 44.1 kHz, qualité
vocale identique).
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"  # économise la RAM vs float32, qualité identique pour la voix


class AudioRecorder:
    """
    Enregistreur audio non-bloquant.

    Usage :
        rec = AudioRecorder()
        rec.start()
        ...  # parler
        wav_path = rec.stop()  # Path vers un WAV temporaire
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False

    # ---- API publique ----

    def start(self) -> None:
        """Démarre la capture. Idempotent : un appel start() pendant un
        enregistrement déjà en cours est ignoré."""
        with self._lock:
            if self._recording:
                return
            self._chunks = []
            self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> Path:
        """
        Arrête la capture, écrit un WAV temporaire et retourne son chemin.
        Lève RuntimeError si aucun enregistrement n'est en cours.
        """
        with self._lock:
            if not self._recording:
                raise RuntimeError("stop() appelé sans start() préalable")
            self._recording = False

        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            # Cas où l'utilisateur a relâché immédiatement : on écrit
            # quand même un WAV vide pour ne pas casser le pipeline aval.
            audio = np.zeros((0, self.channels), dtype=self.dtype)
        else:
            audio = np.concatenate(chunks, axis=0)

        # mkstemp renvoie (fd, path) : on ferme le fd immédiatement pour
        # éviter une fuite de descripteur à chaque dictée (soundfile
        # rouvre le fichier en écriture indépendamment).
        fd, path_str = tempfile.mkstemp(suffix=".wav", prefix="voxtral_")
        os.close(fd)
        wav_path = Path(path_str)
        sf.write(wav_path, audio, self.sample_rate, subtype="PCM_16")
        return wav_path

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    # ---- Callback sounddevice ----

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        # status peut signaler un overflow/underflow ; on ignore en v0
        # (rare en capture micro, et non bloquant).
        with self._lock:
            self._chunks.append(indata.copy())
