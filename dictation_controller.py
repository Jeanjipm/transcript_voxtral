"""
Machine à états de la dictée + sérialisation des accès au micro.

## Le problème que ce module résout

Les callbacks du raccourci clavier s'exécutent DANS le callback de l'event
tap de macOS (vérifié par `sample` : `m_CGEventTapCallBack` → pynput →
notre code). macOS accorde à ce callback un budget d'environ 1 seconde ;
au-delà, il **désactive le tap** en émettant `kCGEventTapDisabledByTimeout`.
Et pynput 1.8.1 ne traite jamais cet événement : le tap reste mort, donc
l'événement « touche relâchée » n'arrive plus jamais. L'app restait en
écoute, icône rouge, raccourci perdu jusqu'au redémarrage.

Or `recorder.start()` peut coûter plusieurs secondes quand le périphérique
audio est froid (4,1 s mesurés à froid contre ~50 ms à chaud) — un coût
imprévisible, largement au-dessus du budget.

## La solution

Les deux points d'entrée appelables depuis le thread de l'event tap
(`on_hotkey_start` / `on_hotkey_stop`) ne font qu'un `put_nowait` et
rendent la main en quelques microsecondes. Un thread dédié
(`dictation-worker`) consomme la file et fait le travail lent.

## Contrat de threads

| Thread                | Peut toucher                          | Budget    |
|-----------------------|---------------------------------------|-----------|
| main (NSApp().run())  | tout rumps/AppKit                     | < 100 ms  |
| event tap (pynput)    | `on_hotkey_start` / `on_hotkey_stop`  | < 1 ms    |
| CoreAudio temps-réel  | `AudioRecorder._on_audio`             | 0 verrou  |
| `dictation-worker`    | `AudioRecorder`, `AudioFeedback`      | bloquant  |
| `inference-worker`    | les `Transcriber`, le presse-papier   | bloquant  |

Ce module n'importe NI rumps NI AppKit : il communique avec l'UI par
callbacks injectés, ce qui le rend testable sans macOS.

## Tolérance aux pannes

Le flux clavier est intrinsèquement lacunaire — macOS peut perdre un
relâchement pendant un réarmement du tap. Toute transition illégale de la
machine à états **logue et abandonne, sans jamais lever**. Une exception
qui s'échapperait tuerait l'unique thread micro.

## Enregistrer et transcrire sont indépendants

Le micro est libre dès que le WAV est écrit ; l'inférence, elle, continue
sur son propre thread. Un nouvel appui est donc accepté pendant qu'une
transcription tourne encore, et `_in_flight` compte celles qui restent en
vol pour que l'état affiché reste juste.

Ça n'a pas toujours été le cas, et le refus coûtait cher. Enchaîner deux
dictées — le réflexe naturel quand on réfléchit à voix haute — voyait le
second appui jeté en silence : ni son, ni icône rouge, rien. On parlait
plusieurs secondes dans le vide avant de comprendre. Vu de l'utilisateur,
c'est indistinguable d'une app bloquée, et c'est bien comme ça que le bug a
été rapporté. La justification d'alors — « deux textes ne doivent jamais
courir au presse-papier » — était fausse : l'`inference-worker` est à thread
unique, il sérialise les transcriptions et l'ordre est préservé.
"""

from __future__ import annotations

import enum
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import soundfile as sf


# En dessous de ce seuil, l'audio n'est que du silence plus le son
# d'activation, et le modèle hallucine une phrase (« Thank you »). On
# abandonne sans rien notifier.
MIN_DICTATION_DURATION_S = 0.5

# Réveil du worker quand la file est vide : donne la précision de la
# surveillance de durée maximale (± cette valeur).
_TICK_S = 0.5

# Taille de la file micro. Un matraquage du raccourci ne doit jamais faire
# bloquer `put_nowait` dans le callback de l'event tap ; au-delà on abandonne
# les commandes, ce qui est sans conséquence (la machine à états déduplique
# déjà les appuis répétés).
_QUEUE_MAX = 8


class State(enum.Enum):
    """États de la dictée. Mutés UNIQUEMENT sur `dictation-worker`."""

    IDLE = "idle"
    ARMING = "arming"  # micro en cours d'ouverture (peut durer des secondes)
    RECORDING = "recording"
    PENDING = "pending"  # transcription en cours côté inference-worker


class _Command(enum.Enum):
    START = "start"
    STOP = "stop"
    PREWARM = "prewarm"
    DEVICE_CHANGED = "device_changed"
    TRANSCRIPTION_DONE = "transcription_done"
    SHUTDOWN = "shutdown"


class Recorder(Protocol):
    """Le sous-ensemble d'`AudioRecorder` dont ce module a besoin."""

    def start(self) -> None: ...
    def stop(self) -> Path: ...
    def prewarm(self) -> None: ...
    def shutdown(self) -> None: ...
    @property
    def is_recording(self) -> bool: ...


class Feedback(Protocol):
    """Le sous-ensemble d'`AudioFeedback` dont ce module a besoin."""

    def play_start(self) -> None: ...
    def play_stop(self) -> None: ...


@dataclass
class DictationCallbacks:
    """Points de sortie vers l'UI et l'inférence.

    Tous sont appelés depuis `dictation-worker`, donc les implémentations qui
    touchent Cocoa doivent elles-mêmes repasser sur le main thread (le garde
    `AppHelper.callAfter` déjà en usage dans app.py).
    """

    # (état, libellé à afficher)
    on_state_change: Callable[[State, str], None]
    # Soumet la transcription ; reçoit le WAV. Doit rendre la main tout de
    # suite (typiquement : mise en file sur l'inference-worker).
    submit_transcription: Callable[[Path], None]
    # (sous-titre, message) — erreur à rendre visible.
    on_error: Callable[[str, str], None]
    # Le tap clavier semble mort : le réarmer.
    on_rearm_needed: Callable[[], None]
    # Un enregistrement a été coupé pour dépassement de durée ; le WAV est
    # conservé et confié à l'appelant (typiquement : proposer de le
    # transcrire comme un fichier).
    on_recording_kept: Callable[[Path], None]


class DictationController:
    """Pilote la dictée : une file, un thread, une machine à états."""

    def __init__(
        self,
        recorder: Recorder,
        feedback: Feedback,
        callbacks: DictationCallbacks,
        max_duration_s: int = 300,
        recordings_dir: Path | None = None,
        tail_padding_ms: int = 350,
    ) -> None:
        self._recorder = recorder
        self._feedback = feedback
        self._cb = callbacks
        self._max_duration_s = max_duration_s
        self._tail_padding_ms = tail_padding_ms
        self._recordings_dir = recordings_dir or (
            Path.home() / ".voxtral" / "recordings"
        )

        self._queue: queue.Queue[_Command] = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._stopping = False

        self._state = State.IDLE
        # Un relâchement reçu pendant ARMING doit être honoré dès que
        # l'armement se termine. Sans cette mémorisation, un appui bref
        # pendant les secondes d'ouverture du micro laisserait l'app bloquée
        # en RECORDING sans jamais recevoir de stop.
        self._stop_requested = False
        self._recording_started_at = 0.0
        # Transcriptions soumises et pas encore terminées. Il peut y en avoir
        # plusieurs : on accepte de redémarrer une dictée pendant qu'une
        # transcription tourne encore (cf. `_handle_start`), donc l'état
        # affiché ne peut plus se déduire du seul `_state`.
        self._in_flight = 0

    # ---- Cycle de vie ----

    def start(self) -> None:
        """Démarre le thread worker. Idempotent."""
        if self._thread is not None:
            return
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dictation-worker"
        )
        self._thread.start()

    def shutdown(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Demande l'arrêt. `wait=False` pour ne jamais bloquer le main thread."""
        if self._thread is None:
            return
        thread = self._thread
        self._stopping = True
        self._enqueue(_Command.SHUTDOWN)
        if wait:
            thread.join(timeout=timeout)
        self._thread = None

    def request_prewarm(self) -> None:
        """Demande le pré-chauffage du micro (coûteux, donc sur le worker)."""
        self._enqueue(_Command.PREWARM)

    def request_device_refresh(self) -> None:
        """Le périphérique audio a changé : jeter et reconstruire le stream."""
        self._enqueue(_Command.DEVICE_CHANGED)

    @property
    def state(self) -> State:
        """Instantané de l'état. Lecture sans verrou : la valeur peut avoir
        changé à l'instant où l'appelant la lit, ce qui est acceptable pour
        de l'affichage et des tests."""
        return self._state

    # ---- Points d'entrée depuis le thread de l'event tap ----
    #
    # CES DEUX MÉTHODES SONT LE CŒUR DU CORRECTIF DE LA CAUSE 1.
    # Elles ne doivent JAMAIS faire autre chose qu'un put_nowait.

    def on_hotkey_start(self) -> None:
        """Appelée depuis le callback de l'event tap macOS. < 1 ms."""
        self._enqueue(_Command.START)

    def on_hotkey_stop(self) -> None:
        """Appelée depuis le callback de l'event tap macOS. < 1 ms."""
        self._enqueue(_Command.STOP)

    def _enqueue(self, command: _Command) -> None:
        """Met une commande en file sans jamais bloquer ni lever.

        `put_nowait` plutôt que `put` : si la file était pleine et qu'on
        attendait, on bloquerait le callback de l'event tap — exactement ce
        que ce module existe pour empêcher.
        """
        try:
            self._queue.put_nowait(command)
        except queue.Full:
            print(
                f"[dictation] file pleine, commande {command.value} abandonnée",
                file=sys.stderr,
            )

    # ---- Boucle du worker ----

    def _run(self) -> None:
        while True:
            try:
                command = self._queue.get(timeout=_TICK_S)
            except queue.Empty:
                self._safe(self._on_tick)
                continue

            if command is _Command.SHUTDOWN:
                self._safe(self._handle_shutdown)
                return

            self._safe(lambda: self._handle(command))

    def _safe(self, fn: Callable[[], None]) -> None:
        """Exécute `fn` en garantissant qu'aucune exception ne s'échappe.

        Ce thread est l'unique pilote du micro : le perdre rendrait la dictée
        impossible jusqu'au redémarrage. En cas d'erreur on force un retour à
        IDLE, micro libéré, et on rend l'erreur visible.
        """
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._force_idle()
            self._cb.on_error("Erreur de dictée", str(exc)[:200])

    def _handle(self, command: _Command) -> None:
        if command is _Command.START:
            self._handle_start()
        elif command is _Command.STOP:
            self._handle_stop()
        elif command is _Command.PREWARM:
            self._recorder.prewarm()
        elif command is _Command.DEVICE_CHANGED:
            self._handle_device_changed()
        elif command is _Command.TRANSCRIPTION_DONE:
            self._handle_transcription_done()

    # ---- Transitions ----

    def _handle_start(self) -> None:
        if self._state not in (State.IDLE, State.PENDING):
            # Appui en double : auto-repeat, ou appui reçu juste après un
            # réarmement du tap. Rien à faire, on enregistre déjà.
            self._log_drop("start", self._state)
            return

        # PENDING est accepté, et c'est un correctif, pas un détail.
        #
        # La version précédente refusait de démarrer tant que la transcription
        # précédente tournait, au motif que « deux textes ne doivent jamais
        # courir au presse-papier ». Ce raisonnement était faux : les
        # transcriptions passent par un worker à thread unique qui les
        # sérialise, donc les textes arrivent dans l'ordre de toute façon.
        #
        # Le prix de ce refus, lui, était réel et invisible. Enchaîner deux
        # dictées — ce qu'on fait naturellement quand on réfléchit à voix
        # haute — voyait le second appui jeté sans aucun signal : pas de son,
        # pas d'icône rouge, rien. L'utilisateur parlait dans le vide pendant
        # plusieurs secondes, puis relâchait dans le vide aussi. Vu de
        # l'extérieur, c'est exactement la signature d'une app bloquée, et
        # c'est ce qui a été observé en usage réel (deux occurrences dans
        # voxtral.log : « start ignorée en état pending » suivi de
        # « stop ignorée en état idle »).
        #
        # Le micro est libre dès que le WAV est écrit : il n'y a aucune raison
        # matérielle d'attendre la fin de l'inférence pour réenregistrer.

        self._trace("appui reçu")
        self._stop_requested = False
        self._state = State.ARMING
        self._feedback.play_start()
        t0 = time.monotonic()

        try:
            self._recorder.start()
        except Exception as exc:  # noqa: BLE001
            self._state = State.IDLE
            self._cb.on_error(
                "Micro indisponible",
                f"{exc}\n\nVérifie qu'un micro est branché et que Voxtral a "
                f"l'autorisation d'y accéder.",
            )
            return

        self._recording_started_at = time.monotonic()
        self._state = State.RECORDING
        self._trace(f"micro ouvert en {(time.monotonic() - t0) * 1000:.0f} ms")
        self._cb.on_state_change(State.RECORDING, "État : écoute en cours…")

        # Le relâchement a pu arriver pendant l'ouverture du micro (qui peut
        # durer plusieurs secondes) : on l'honore maintenant.
        if self._stop_requested:
            self._stop_requested = False
            self._handle_stop()

    def _handle_stop(self) -> None:
        if self._state is State.ARMING:
            # Mémorisé, honoré à la fin de l'armement (cf. _handle_start).
            self._stop_requested = True
            return

        if self._state is not State.RECORDING:
            self._log_drop("stop", self._state)
            return

        # On continue d'enregistrer un court instant après le relâchement.
        # Mesuré : sans ça, le micro s'arrêtait 0,1 ms après le relâchement,
        # alors qu'on lâche la touche en finissant de prononcer le dernier mot
        # — la fin de phrase était donc systématiquement tronquée en pleine
        # syllabe. On est sur `dictation-worker`, donc cette attente ne bloque
        # ni la menu bar ni le callback clavier.
        if self._tail_padding_ms > 0:
            time.sleep(self._tail_padding_ms / 1000.0)

        wav_path = self._recorder.stop()
        self._state = State.IDLE

        duration = self._safe_duration(wav_path)
        self._trace(f"relâchement, {duration:.1f} s enregistrées")
        if duration < MIN_DICTATION_DURATION_S:
            self._trace("trop court, dictée jetée")
            self._unlink(wav_path)
            self._settle()
            return

        self._feedback.play_stop()
        self._in_flight += 1
        self._state = State.PENDING
        self._cb.on_state_change(State.PENDING, "État : transcription…")
        self._trace("envoyé à la transcription")
        self._cb.submit_transcription(wav_path)

    def notify_transcription_done(self) -> None:
        """À appeler quand l'inference-worker a fini (succès ou échec).

        Appelable depuis n'importe quel thread : passe par la file, pour que
        l'état reste muté par le seul `dictation-worker`. Muter directement
        depuis le thread d'inférence ouvrirait une course avec un appui sur
        le raccourci traité au même instant.
        """
        self._enqueue(_Command.TRANSCRIPTION_DONE)

    def _handle_transcription_done(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self._trace("transcription terminée")
        # Une transcription qui se termine ne doit RIEN dire quand une
        # nouvelle dictée a déjà commencé : sinon l'icône repasserait au vert
        # « prêt » alors que le micro est ouvert et que l'utilisateur parle.
        if self._state in (State.PENDING, State.IDLE):
            self._settle()

    def _settle(self) -> None:
        """Repose l'état affiché en tenant compte des transcriptions en vol.

        Depuis qu'on peut réenregistrer pendant une transcription, « plus
        d'enregistrement en cours » ne veut plus dire « prêt » : il peut
        rester du travail côté inférence.
        """
        if self._in_flight > 0:
            self._state = State.PENDING
            self._cb.on_state_change(State.PENDING, "État : transcription…")
        else:
            self._state = State.IDLE
            self._cb.on_state_change(State.IDLE, "État : prêt")

    def _handle_device_changed(self) -> None:
        """Périphérique audio changé : on repart d'un stream propre.

        Pendant un enregistrement on ne touche à rien : `AudioRecorder.stop()`
        sait déjà survivre à la disparition du périphérique et conserve
        l'audio capté.
        """
        if self._state in (State.ARMING, State.RECORDING):
            return
        self._recorder.shutdown()
        self._recorder.prewarm()

    def _handle_shutdown(self) -> None:
        if self._state is State.RECORDING:
            try:
                self._recorder.stop()
            except Exception:  # noqa: BLE001
                pass
        self._state = State.IDLE
        try:
            self._recorder.shutdown()
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    # ---- Surveillance de la durée maximale ----

    def _on_tick(self) -> None:
        """Coupe-circuit : le relâchement n'est jamais arrivé.

        C'est le filet de sécurité de la cause 1. Le tap désactivé n'est pas
        détectable depuis l'extérieur — pynput garde le tap en variable
        locale, et `listener.running` reste True sur un tap mort. On détecte
        donc par le symptôme : un enregistrement anormalement long.
        """
        if self._state is not State.RECORDING:
            return
        elapsed = time.monotonic() - self._recording_started_at
        if elapsed < self._max_duration_s:
            return

        print(
            f"[dictation] enregistrement coupé après {elapsed:.0f}s "
            f"(limite {self._max_duration_s}s) — relâchement jamais reçu, "
            f"réarmement du raccourci.",
            file=sys.stderr,
        )

        wav_path = self._recorder.stop()
        self._state = State.IDLE

        # On CONSERVE l'audio : jeter cinq minutes de dictée réelle serait
        # pire qu'un fichier orphelin, et coller cinq minutes de bruit pire
        # encore. On ne colle donc rien, et on confie le fichier à l'appelant.
        kept = self._keep_recording(wav_path)
        self._settle()
        self._cb.on_error(
            "Enregistrement trop long",
            f"L'enregistrement a été coupé après {int(elapsed)} s car le "
            f"relâchement du raccourci n'a jamais été reçu. L'audio est "
            f"conservé ici :\n{kept}\n\nTu peux le transcrire via "
            f"« Transcrire un fichier audio… ».",
        )
        if kept is not None:
            self._cb.on_recording_kept(kept)
        self._cb.on_rearm_needed()

    def _keep_recording(self, wav_path: Path) -> Path | None:
        """Déplace le WAV temporaire vers ~/.voxtral/recordings/."""
        try:
            self._recordings_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            target = self._recordings_dir / f"dictee-{stamp}.wav"
            wav_path.replace(target)
            return target
        except OSError:
            traceback.print_exc()
            return None

    # ---- Utilitaires ----

    def _force_idle(self) -> None:
        """Retour à IDLE avec libération du micro, quoi qu'il arrive."""
        self._stop_requested = False
        try:
            if self._recorder.is_recording:
                self._unlink(self._recorder.stop())
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self._state = State.IDLE
        try:
            self._settle()
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    @staticmethod
    def _safe_duration(wav_path: Path) -> float:
        """Durée du WAV, ou 0.0 si illisible (fichier vide, disque plein…)."""
        try:
            return float(sf.info(str(wav_path)).duration)
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _unlink(wav_path: Path) -> None:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _trace(self, message: str) -> None:
        """Trace horodatée du cycle de vie d'une dictée.

        Existe parce qu'un blocage rapporté en usage réel n'avait laissé
        AUCUNE trace : ni erreur, ni commande jetée, ni crash. Impossible de
        savoir si l'appui était arrivé, si le micro s'était ouvert, ni
        combien de temps l'enregistrement avait duré — donc impossible
        d'enquêter autrement qu'en devinant.

        Une ligne par étape, horodatée à la milliseconde, avec l'état et le
        nombre de transcriptions en vol. Le volume est négligeable (quelques
        lignes par dictée) et c'est la différence entre un diagnostic et une
        hypothèse.
        """
        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        print(
            f"[{stamp}] [dictee] {message} "
            f"(état={self._state.value}, en vol={self._in_flight})",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _log_drop(command: str, state: State) -> None:
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"[dictee] commande '{command}' ignorée en état {state.value}",
            file=sys.stderr,
        )
