"""
Capture micro pour Voxtral Dictée.

On enregistre en mono 16 kHz (format attendu par Voxtral et Whisper),
on tamponne en RAM dans une liste de chunks numpy, puis on écrit un WAV
temporaire à `stop()`. Pas de conversion ni resampling : sounddevice gère.

Pourquoi 16 kHz ? C'est la fréquence d'échantillonnage native des modèles
de speech-to-text grand public (économie de calcul vs 44.1 kHz, qualité
vocale identique).

## Contrat de threads (important)

Cette classe est conçue pour être pilotée par UN SEUL thread
(`dictation-worker`, cf. dictation_controller.py) : `start`, `stop`,
`prewarm` et `shutdown` ne doivent jamais être appelés en parallèle.
C'est ce qui permet de garder le verrou interne réduit au strict minimum.

Le verrou `_state_lock` ne protège QUE le drapeau `_recording`, et n'est
jamais tenu pendant un appel PortAudio. Raison : construire un
`sd.InputStream` coûte ~4,1 s sur cette machine (mesuré). L'ancienne
version tenait le verrou pendant cette construction, si bien qu'un appui
sur le raccourci se bloquait dessus — et comme le callback clavier de macOS
a un budget d'environ 1 seconde, macOS désactivait la surveillance clavier
et le relâchement de touche n'arrivait jamais.

`_on_audio` tourne dans le thread temps-réel CoreAudio et ne prend AUCUN
verrou : y bloquer provoquerait des pertes d'échantillons, et prendre un
verrou que d'autres threads peuvent tenir longtemps est une inversion de
priorité.
"""

from __future__ import annotations

import os
import sys
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

# Nombre de reconstructions du stream après un échec de démarrage. Une seule :
# un second échec signifie que le micro est réellement indisponible
# (permission refusée, aucun périphérique d'entrée) et doit remonter en
# erreur visible, pas boucler.
DEFAULT_START_RETRIES = 1


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
        start_retries: int = DEFAULT_START_RETRIES,
        silence_padding_ms: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.start_retries = start_retries
        self.silence_padding_ms = silence_padding_ms
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._state_lock = threading.Lock()
        self._recording = False

    # ---- API publique ----

    def start(self) -> None:
        """Démarre la capture. Idempotent : un appel start() pendant un
        enregistrement déjà en cours est ignoré.

        Le `sd.InputStream` est créé lazy au premier appel et conservé
        entre les dictées : sa fermeture libère le device CoreAudio qui
        rendort alors le hardware micro, et le réveil au prochain start
        coûte ~4 s sur Apple Silicon (mesuré). En gardant le stream
        ouvert, on évite ce coût à chaque dictée.

        En cas d'échec, `_recording` est remis à False. C'est le correctif
        d'un bug où le drapeau restait bloqué à True après un
        `PortAudioError` : l'app affichait alors l'icône « enregistrement »
        et n'enregistrait plus jamais rien, jusqu'au redémarrage.
        """
        with self._state_lock:
            if self._recording:
                return
            # Réassignation, PAS .clear() : un callback temps-réel retardataire
            # peut encore détenir une référence à l'ancienne liste. Il y
            # écrira sans polluer l'enregistrement qui commence.
            self._chunks = []
            self._recording = True

        try:
            self._start_stream_with_retry()
        except Exception:
            with self._state_lock:
                self._recording = False
            raise

    def stop(self) -> Path:
        """
        Arrête la capture, écrit un WAV temporaire et retourne son chemin.
        Lève RuntimeError si aucun enregistrement n'est en cours.

        On stop() le stream sans le close() : le device reste alloué côté
        PortAudio/CoreAudio pour éviter le coût de re-init au prochain start.
        Le close() final est délégué à shutdown().
        """
        with self._state_lock:
            if not self._recording:
                raise RuntimeError("stop() appelé sans start() préalable")
            # Baisser le drapeau AVANT stream.stop() : c'est ce qui garantit
            # qu'aucun callback ne peut plus ajouter d'échantillons une fois
            # Pa_StopStream revenu (cf. _on_audio).
            self._recording = False

        if self._stream is not None:
            # stream.stop() attend la fin du callback en cours (Pa_StopStream)
            # avant de retourner : après ce point, plus aucun sample n'arrive.
            try:
                self._stream.stop()
            except sd.PortAudioError:
                # Périphérique disparu en cours d'enregistrement (casque
                # débranché) : on garde ce qui a été capté avant la coupure.
                self._discard_stream()

        chunks = self._chunks
        self._chunks = []

        if not chunks:
            # Cas où l'utilisateur a relâché immédiatement : on écrit
            # quand même un WAV vide pour ne pas casser le pipeline aval.
            audio = np.zeros((0, self.channels), dtype=self.dtype)
        else:
            audio = np.concatenate(chunks, axis=0)

        return self._write_wav(audio)

    def shutdown(self) -> None:
        """Ferme proprement le stream — à appeler au quit de l'app.

        Pendant la durée de vie de l'app on garde le stream ouvert (cf. start()).
        Au shutdown on libère le device pour ne pas laisser de fuite
        côté CoreAudio.
        """
        with self._state_lock:
            self._recording = False
        self._discard_stream()

    def prewarm(self) -> None:
        """Pré-initialise le stream micro pour amortir le coût d'init CoreAudio.

        Crée + start + stop le stream sans toucher au drapeau `_recording`,
        donc sans interférer avec une dictée. Les samples captés pendant le
        bref start/stop sont ignorés par `_on_audio` (`_recording` est False).

        La construction du stream (~4,1 s) se fait HORS de tout verrou : c'est
        le point qui bloquait le callback clavier dans l'ancienne version.
        """
        with self._state_lock:
            # Si _recording est True, l'utilisateur a déjà déclenché une
            # dictée — le stream est chaud par le start réel, rien à faire.
            if self._recording:
                return

        self._ensure_stream()  # ~4,1 s, hors verrou
        try:
            assert self._stream is not None
            self._stream.start()
            self._stream.stop()
        except (sd.PortAudioError, OSError):
            # Stream dans un état imprévu : on le jette pour que le prochain
            # start() en reconstruise un propre plutôt que d'en hériter.
            self._discard_stream()

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recording

    # ---- Gestion du stream ----

    def _ensure_stream(self) -> None:
        """Construit le stream s'il n'existe pas. ~4,1 s. Jamais sous verrou."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._on_audio,
        )

    def _discard_stream(self) -> None:
        """Arrête et ferme le stream courant, sans jamais lever."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:  # noqa: BLE001 — on ferme quoi qu'il arrive
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    def _start_stream_with_retry(self) -> None:
        """Démarre le stream, en le reconstruisant une fois si nécessaire.

        Pourquoi une reprise : PortAudio met en cache la liste des
        périphériques à son initialisation. Un casque branché (ou débranché,
        ou un Mac sortant de veille) après le lancement de l'app rend le
        stream existant invalide et l'index du périphérique « par défaut »
        périmé — `start()` lève alors `PortAudioError`. On force donc une
        réinitialisation de PortAudio avant de reconstruire, sinon on
        rebâtirait un stream sur la même liste périmée.
        """
        last_error: Exception | None = None
        for attempt in range(self.start_retries + 1):
            try:
                self._ensure_stream()
                assert self._stream is not None
                self._stream.start()
                return
            except (sd.PortAudioError, OSError) as exc:
                last_error = exc
                self._discard_stream()
                if attempt == self.start_retries:
                    break
                print(
                    f"[audio_capture] démarrage micro échoué ({exc}) — "
                    f"réinitialisation de PortAudio et nouvelle tentative.",
                    file=sys.stderr,
                )
                self._reinit_portaudio()

        assert last_error is not None
        raise last_error

    @staticmethod
    def _reinit_portaudio() -> None:
        """Purge le cache de périphériques de PortAudio. Best-effort.

        `_terminate`/`_initialize` sont des API privées de sounddevice, mais
        c'est le seul moyen de faire relire la liste des périphériques sans
        redémarrer le process. Si elles disparaissent d'une version future,
        on continue sans : la reprise se contentera de reconstruire le stream.
        """
        try:
            sd._terminate()
            sd._initialize()
        except Exception:  # noqa: BLE001
            pass

    def _pad(self, audio: np.ndarray) -> np.ndarray:
        """Ajoute du silence de chaque côté de l'enregistrement.

        Les modèles de reconnaissance vocale sont entraînés sur des fenêtres
        rembourrées de silence et se comportent mal quand la parole commence
        ou finit exactement au bord du fichier — le premier ou le dernier mot
        y perd des phonèmes. Quelques centaines de millisecondes de zéros
        coûtent 8 Ko et suppriment cette classe d'erreur.
        """
        if self.silence_padding_ms <= 0 or audio.shape[0] == 0:
            return audio
        pad_frames = int(self.sample_rate * self.silence_padding_ms / 1000)
        silence = np.zeros((pad_frames, self.channels), dtype=self.dtype)
        return np.concatenate([silence, audio, silence], axis=0)

    def _write_wav(self, audio: np.ndarray) -> Path:
        """Écrit `audio` dans un WAV temporaire et retourne son chemin."""
        audio = self._pad(audio)
        # mkstemp renvoie (fd, path) : on ferme le fd immédiatement pour
        # éviter une fuite de descripteur à chaque dictée (soundfile
        # rouvre le fichier en écriture indépendamment).
        fd, path_str = tempfile.mkstemp(suffix=".wav", prefix="voxtral_")
        os.close(fd)
        wav_path = Path(path_str)
        sf.write(wav_path, audio, self.sample_rate, subtype="PCM_16")
        return wav_path

    # ---- Callback sounddevice ----

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Thread temps-réel CoreAudio. AUCUN verrou, aucun syscall.

        La lecture non synchronisée de `_recording` est correcte : `stop()`
        met le drapeau à False *puis* appelle `Pa_StopStream`, qui attend la
        fin du callback en vol. Un callback qui aurait passé le test juste
        avant termine donc son `append` avant que `stop()` ne revienne, et
        ses échantillons sont bien inclus. Et comme `start()` réassigne
        `_chunks` à une liste neuve, un callback vraiment retardataire écrit
        dans l'ancienne liste, sans polluer la dictée suivante.

        `status` peut signaler un overflow ; on l'ignore en v0 (rare en
        capture micro, et non bloquant).
        """
        if not self._recording:
            return
        self._chunks.append(indata.copy())
