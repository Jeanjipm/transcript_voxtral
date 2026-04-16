"""
Écoute du raccourci clavier global.

Deux modes de raccourci supportés :

1. **Single-key tenue** (ex. `alt_r` = Option droite). Mode talkie-walkie :
   on appuie → on_start ; on relâche → on_stop. C'est le défaut Voxtral.

2. **Combinaison** (ex. `cmd+shift+h`). Deux sous-modes :
   - `push_to_talk` : maintenir la combinaison enregistre
   - `toggle`       : un appui démarre, le suivant arrête

Implémentation : `pynput.keyboard.Listener` non-suppressif. Ne bloque
PAS la propagation de la touche au système, donc Right Option continue
à servir aux caractères spéciaux (Option+E → é) tant qu'on ne tape
pas pendant qu'on parle.

Anti-rebond : on filtre les événements `auto-repeat` du système (la
touche enfoncée déclenche en boucle des `on_press` ; on ne déclenche
on_start qu'au premier).
"""

from __future__ import annotations

from typing import Callable

from pynput import keyboard


# Mapping nom config → pynput.Key (ou caractère)
_NAMED_KEYS: dict[str, keyboard.Key] = {
    "alt_l": keyboard.Key.alt_l,
    "alt_r": keyboard.Key.alt_r,
    "alt": keyboard.Key.alt,
    "cmd_l": keyboard.Key.cmd_l,
    "cmd_r": keyboard.Key.cmd_r,
    "cmd": keyboard.Key.cmd,
    "ctrl_l": keyboard.Key.ctrl_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl": keyboard.Key.ctrl,
    "shift_l": keyboard.Key.shift_l,
    "shift_r": keyboard.Key.shift_r,
    "shift": keyboard.Key.shift,
    "space": keyboard.Key.space,
    "enter": keyboard.Key.enter,
    "tab": keyboard.Key.tab,
    "esc": keyboard.Key.esc,
    "f13": keyboard.Key.f13,
    "f14": keyboard.Key.f14,
    "f15": keyboard.Key.f15,
    "f16": keyboard.Key.f16,
    "f17": keyboard.Key.f17,
    "f18": keyboard.Key.f18,
    "f19": keyboard.Key.f19,
}


def _parse_key(token: str) -> keyboard.Key | str:
    """Convertit un token ('alt_r', 'h', 'space') en clé pynput."""
    token = token.lower().strip()
    if token in _NAMED_KEYS:
        return _NAMED_KEYS[token]
    if len(token) == 1:
        return token  # caractère ASCII (ex. "h")
    raise ValueError(f"Touche inconnue : {token!r}")


def _is_single_key(combo: str) -> bool:
    """True si le combo désigne une touche unique (ex. 'alt_r')."""
    return "+" not in combo


class HotkeyManager:
    """
    Gestionnaire de raccourci global.

    Usage :
        mgr = HotkeyManager(
            combo="alt_r",
            mode="push_to_talk",
            on_start=lambda: ...,
            on_stop=lambda: ...,
        )
        mgr.start()       # lance le listener (non-bloquant)
        ...
        mgr.stop()        # arrête le listener
    """

    def __init__(
        self,
        combo: str,
        mode: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self.combo = combo.lower().strip()
        self.mode = mode
        self.on_start = on_start
        self.on_stop = on_stop

        self._listener: keyboard.Listener | None = None
        self._active = False  # True pendant qu'on enregistre

        if _is_single_key(self.combo):
            self._target_key: keyboard.Key | str = _parse_key(self.combo)
            self._modifier_keys: set[keyboard.Key | str] = set()
            self._final_key: keyboard.Key | str | None = None
        else:
            tokens = [t.strip() for t in self.combo.split("+")]
            *modifiers, final = tokens
            self._target_key = None  # type: ignore[assignment]
            self._modifier_keys = {_parse_key(t) for t in modifiers}
            self._final_key = _parse_key(final)

        self._pressed: set[keyboard.Key | str] = set()

    # ---- Lifecycle ----

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,  # CRITIQUE : ne pas bloquer la touche au système
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None
        self._pressed.clear()
        if self._active:
            self._active = False

    def update_binding(self, combo: str, mode: str) -> None:
        """Reconfigure le raccourci sans redémarrer l'app."""
        self.stop()
        self.__init__(combo, mode, self.on_start, self.on_stop)
        self.start()

    # ---- Callbacks pynput ----

    def _normalize(self, key: object) -> keyboard.Key | str | None:
        """Normalise key (KeyCode → char str, Key → Key, autre → None)."""
        if isinstance(key, keyboard.Key):
            return key
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        return None

    def _on_press(self, key: object) -> None:
        norm = self._normalize(key)
        if norm is None:
            return

        # Anti-rebond auto-repeat : si déjà dans le set, on ignore
        already_pressed = norm in self._pressed
        self._pressed.add(norm)
        if already_pressed:
            return

        if _is_single_key(self.combo):
            # Single-key : push-to-talk forcé
            if norm == self._target_key and not self._active:
                self._active = True
                self._safe_call(self.on_start)
            return

        # Combinaison : faut que tous les modifs ET la touche finale soient pressés
        if not self._modifier_keys.issubset(self._pressed):
            return
        if norm != self._final_key:
            return

        if self.mode == "push_to_talk":
            if not self._active:
                self._active = True
                self._safe_call(self.on_start)
        else:  # toggle
            if self._active:
                self._active = False
                self._safe_call(self.on_stop)
            else:
                self._active = True
                self._safe_call(self.on_start)

    def _on_release(self, key: object) -> None:
        norm = self._normalize(key)
        if norm is None:
            return
        self._pressed.discard(norm)

        if not self._active:
            return

        if _is_single_key(self.combo):
            if norm == self._target_key:
                self._active = False
                self._safe_call(self.on_stop)
            return

        # Combinaison push-to-talk : on stoppe dès qu'on relâche la touche
        # finale OU n'importe quel modificateur (sinon l'utilisateur reste
        # bloqué en "écoute" si le timing est imparfait).
        if self.mode == "push_to_talk":
            if norm == self._final_key or norm in self._modifier_keys:
                self._active = False
                self._safe_call(self.on_stop)

    # ---- Robustesse ----

    @staticmethod
    def _safe_call(fn: Callable[[], None]) -> None:
        """
        Encapsule l'appel callback : une exception dans on_start/on_stop
        ne doit PAS tuer le listener clavier (sinon le raccourci ne
        marchera plus jusqu'au redémarrage de l'app).
        """
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[HotkeyManager] callback erreur: {exc}")
            traceback.print_exc()


def display_combo(combo: str) -> str:
    """Joli label pour l'UI (ex. 'alt_r' → '⌥ droite', 'cmd+shift+h' → '⌘⇧H')."""
    pretty: dict[str, str] = {
        "cmd": "⌘", "cmd_l": "⌘ gauche", "cmd_r": "⌘ droite",
        "shift": "⇧", "shift_l": "⇧ gauche", "shift_r": "⇧ droite",
        "alt": "⌥", "alt_l": "⌥ gauche", "alt_r": "⌥ droite",
        "ctrl": "⌃", "ctrl_l": "⌃ gauche", "ctrl_r": "⌃ droite",
        "space": "␣", "enter": "↩", "tab": "⇥", "esc": "⎋",
    }
    if "+" not in combo:
        return pretty.get(combo, combo.upper())
    return "".join(pretty.get(t, t.upper()) for t in combo.split("+"))
