"""
Enregistrement d'un raccourci à partir de vraies touches pressées.

Remplace la saisie au clavier d'un texte du genre `alt+space` dans les
Préférences : l'utilisateur appuie sur les touches qu'il veut, on lit
l'appui, on en déduit le raccourci.

## Pourquoi pynput et pas les événements clavier de tkinter

C'est le choix structurant de ce module. tkinter aurait été plus simple —
aucune permission, aucun thread — mais il aurait fallu traduire ses
« keysym » (`Meta_R`, `Alt_L`…) vers notre vocabulaire de configuration
(`cmd_r`, `alt`…). Cette table de traduction est exactement le genre de
chose qui produit un raccourci enregistré qui ne se déclenche jamais : ce
n'est pas tkinter qui écoutera le raccourci à l'exécution, c'est pynput.

En capturant avec pynput et en normalisant via `hotkey_manager.normalize_key`,
la touche enregistrée passe par le MÊME code que celui qui la reconnaîtra
plus tard. Ce qu'on enregistre est, par construction, ce qui marchera.

Vérifié sur la machine cible avant d'écrire ce module :
`Quartz.CGPreflightListenEventAccess()` répond `True` pour le binaire Python
de l'app — le sous-processus des Préférences a donc bien le droit d'écouter
le clavier, c'est la même permission « Saisie au clavier » que la dictée.

## Contrat de threads

Les callbacks pynput tournent dans le thread du CGEventTap, celui-là même
que macOS désactive quand un callback dépasse ~1 seconde (cf. la cause n°1
du sprint 5). Ils ne font donc ici que trois choses : normaliser, faire
avancer une machine à états en mémoire, et `put_nowait`. **Aucun appel
tkinter** : Tk n'est pas thread-safe, et le toucher depuis le tap ferait
planter la fenêtre. C'est l'appelant qui dépile `events` depuis sa boucle.

Le listener n'est jamais suppressif : un tap suppressif qui se coincerait
rendrait le clavier inutilisable dans TOUTES les applications jusqu'à la
mort du processus. On accepte donc que les touches pressées pendant la
capture parviennent aussi au système.
"""

from __future__ import annotations

import enum
import queue
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

from pynput import keyboard

from hotkey_manager import format_combo, is_modifier, normalize_key, token_for_key


# Taille de la file d'événements. Un humain ne produit pas 64 appuis entre
# deux tours de boucle de l'interface ; la borne existe pour que le tap ne
# puisse jamais se bloquer sur une file pleine, pas pour être atteinte.
_QUEUE_MAX = 64


class Outcome(enum.Enum):
    """Où en est l'enregistrement."""

    WAITING = "waiting"          # rien de pressé pour l'instant
    HOLDING = "holding"          # touches en cours de maintien (aperçu)
    DONE = "done"                # raccourci complet, capture terminée
    CANCELLED = "cancelled"      # Échap seul : on garde l'ancien raccourci
    UNSUPPORTED = "unsupported"  # touche sans nom dans notre vocabulaire


@dataclass(frozen=True)
class CaptureEvent:
    outcome: Outcome
    combo: str = ""


class ComboRecorder:
    """Machine à états pure : reçoit des touches normalisées, rend un combo.

    Aucun thread, aucune UI, aucun accès système — donc testable directement.

    Le raccourci retenu est le **maximum de touches tenues simultanément**,
    pas ce qui reste au moment du relâchement. Sans ça, enregistrer ⌘⇧H en
    relâchant H en premier donnerait `cmd+shift`.

    Le raccourci est figé au premier relâchement, ce qui reproduit la
    sémantique du push-to-talk : on tient, on relâche, c'est fini.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._held: list[str] = []
        self._peak: list[str] = []
        self._finished = False

    @property
    def combo(self) -> str:
        """Raccourci en cours de formation (vide si rien n'a été pressé)."""
        return format_combo(self._peak)

    def press(self, key: keyboard.Key | str | None) -> CaptureEvent:
        if self._finished:
            return CaptureEvent(Outcome.DONE, self.combo)

        token = token_for_key(key) if key is not None else None
        if token is None:
            # Touche média, pavé exotique… : inexprimable dans la config, donc
            # on le dit plutôt que d'enregistrer un raccourci mort.
            return CaptureEvent(Outcome.UNSUPPORTED, self.combo)

        # Échap seul = sortie de secours. Après un premier appui, Échap est
        # une touche comme une autre (on peut vouloir ⌘⎋).
        if token == "esc" and not self._held:
            self._finished = True
            return CaptureEvent(Outcome.CANCELLED)

        if token in self._held:
            # Auto-répétition du système : la touche est déjà tenue.
            return CaptureEvent(Outcome.HOLDING, self.combo)

        # Une seule touche « finale » par raccourci : si l'utilisateur tape H
        # puis J sans relâcher, on garde H. `cmd+h+j` serait accepté par le
        # parseur mais ne se déclencherait quasiment jamais.
        if not is_modifier(token) and any(
            not is_modifier(t) for t in self._held
        ):
            return CaptureEvent(Outcome.HOLDING, self.combo)

        self._held.append(token)
        if len(self._held) > len(self._peak):
            self._peak = list(self._held)
        return CaptureEvent(Outcome.HOLDING, self.combo)

    def release(self, key: keyboard.Key | str | None) -> CaptureEvent:
        if self._finished:
            return CaptureEvent(Outcome.DONE, self.combo)

        token = token_for_key(key) if key is not None else None
        if token is not None and token in self._held:
            self._held.remove(token)

        if not self._peak:
            # Relâchement d'une touche déjà tenue avant le début de la
            # capture : on n'a jamais vu l'appui, il n'y a rien à conclure.
            return CaptureEvent(Outcome.WAITING)

        self._finished = True
        return CaptureEvent(Outcome.DONE, self.combo)


ListenerFactory = Callable[..., "keyboard.Listener"]


class HotkeyCapture:
    """Écoute le clavier le temps d'un enregistrement, et rien de plus.

    Usage :
        capture = HotkeyCapture()
        capture.start()
        ...  # dépiler capture.events depuis la boucle de l'interface
        capture.stop()

    `stop()` est idempotent et sûr depuis n'importe quel thread SAUF celui du
    listener — même règle que `HotkeyManager.stop()`, pour la même raison
    (le listener pynput *est* un Thread, s'arrêter depuis soi-même bloque).
    """

    def __init__(self, listener_factory: ListenerFactory | None = None) -> None:
        self.events: queue.Queue[CaptureEvent] = queue.Queue(maxsize=_QUEUE_MAX)
        self._recorder = ComboRecorder()
        self._listener: keyboard.Listener | None = None
        self._factory: ListenerFactory = (
            listener_factory
            if listener_factory is not None
            else keyboard.Listener
        )

    @property
    def active(self) -> bool:
        return self._listener is not None

    def start(self) -> None:
        """Démarre l'écoute. Sans effet si une capture est déjà en cours."""
        if self._listener is not None:
            return
        self._recorder.reset()
        # On vide la file : les événements d'une capture précédente annulée
        # ne doivent pas être pris pour ceux de la nouvelle.
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                break

        self._listener = self._factory(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()

    def stop(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    # -- callbacks du thread event-tap : rapides, et rien d'autre --

    def _on_press(self, key: object) -> None:
        self._emit(self._recorder.press, key)

    def _on_release(self, key: object) -> None:
        self._emit(self._recorder.release, key)

    def _emit(
        self,
        step: Callable[[keyboard.Key | str | None], CaptureEvent],
        key: object,
    ) -> None:
        try:
            event = step(normalize_key(key))
            self.events.put_nowait(event)
        except queue.Full:
            # L'interface ne dépile plus : perdre un aperçu est sans gravité,
            # bloquer le tap ne l'est pas.
            pass
        except Exception as exc:  # noqa: BLE001
            # Une exception qui remonte tuerait le listener, et la capture
            # resterait figée sur « Appuie sur les touches… » à vie.
            print(f"[HotkeyCapture] erreur de capture : {exc}", file=sys.stderr)
            traceback.print_exc()
