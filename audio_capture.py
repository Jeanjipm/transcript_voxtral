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
        enregistrement déjà en cours est ignoré.

        Le `sd.InputStream` est créé lazy au premier appel et conservé
        entre les dictées : sa fermeture libère le device CoreAudio qui
        rendort alors le hardware micro, et le réveil au prochain start
        coûte 2-5s sur Apple Silicon (cf. macos-mic-keepwarm). En gardant
        le stream ouvert, on évite ce coût à chaque dictée.

        Le hardware reste-t-il "warm" entre stream.stop() et stream.start() ?
        C'est empirique : la doc PortAudio ne le garantit pas formellement
        sur CoreAudio. Si insuffisant, il faudra passer à un stream toujours
        actif (avec voyant orange permanent côté macOS).
        """
        with self._lock:
            if self._recording:
                return
            self._chunks = []
            self._recording = True
        if self._stream is None:
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

        On stop() le stream sans le close() : le device reste alloué côté
        PortAudio/CoreAudio pour éviter le coût de re-init au prochain start.
        Le close() final est délégué à shutdown().
        """
        with self._lock:
            if not self._recording:
                raise RuntimeError("stop() appelé sans start() préalable")
            self._recording = False

        assert self._stream is not None
        # stream.stop() attend la fin du callback en cours (Pa_StopStream)
        # avant de retourner, donc plus aucun sample ne sera ajouté à
        # _chunks après ce point — pas de race avec le _chunks = [] de
        # start() suivant.
        self._stream.stop()

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

    def shutdown(self) -> None:
        """Ferme proprement le stream — à appeler au quit de l'app.

        Pendant la durée de vie de l'app on garde le stream ouvert (cf. start()).
        Au shutdown on libère le device pour ne pas laisser de fuite
        côté CoreAudio.
        """
        with self._lock:
            self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
            except sd.PortAudioError:
                # déjà stoppé : pas grave
                pass
            self._stream.close()
            self._stream = None

    def prewarm(self) -> None:
        """Pré-initialise le stream micro pour amortir le coût d'init CoreAudio.

        Crée + start + stop le stream sans toucher au flag _recording, donc
        n'interfère pas avec un éventuel hotkey concurrent. Les samples
        captés pendant le bref start/stop sont ignorés par _on_audio
        (check _recording=False).

        À appeler en thread daemon au lancement de l'app : la 1re vraie
        dictée devient instantanée comme les suivantes.
        """
        with self._lock:
            # Si _recording=True, l'utilisateur a déjà déclenché un hotkey
            # avant qu'on prewarm — le stream est déjà chaud par le start
            # réel, on n'a rien à faire.
            if self._recording:
                return
            if self._stream is None:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._on_audio,
                )
            try:
                self._stream.start()
                self._stream.stop()
            except sd.PortAudioError:
                # Stream peut être en état imprévu (déjà started par
                # un start() concurrent qui a contourné le lock — ne
                # devrait pas arriver, mais on est défensif).
                pass

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
        # Garde-fou : si on est en transition stop(), on ignore les samples
        # résiduels pour ne pas polluer l'enregistrement suivant. En théorie
        # PortAudio attend la fin du callback à Pa_StopStream, mais ce check
        # est gratuit et défensif.
        with self._lock:
            if not self._recording:
                return
            self._chunks.append(indata.copy())
