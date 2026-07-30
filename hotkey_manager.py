"""
Écoute du raccourci clavier global en mode push-to-talk.

Maintenir la touche/combinaison → enregistre ; relâcher → transcrit.

Implémentation : `pynput.keyboard.Listener` non-suppressif. Ne bloque
PAS la propagation de la touche au système, donc Right Option continue
à servir aux caractères spéciaux (Option+E → é) tant qu'on ne tape
pas pendant qu'on parle.

Anti-rebond : on filtre les événements `auto-repeat` du système (la
touche enfoncée déclenche en boucle des `on_press` ; on ne déclenche
on_start qu'au premier).
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from pynput import keyboard


# Intervalle minimum entre deux réarmements du listener. Empêche une cause de
# blocage persistante de faire boucler la reconstruction du tap.
_REARM_MIN_INTERVAL_S = 10.0

# Attente maximale de la fin du thread listener. pynput sort de sa boucle au
# tour de CFRunLoop suivant, soit jusqu'à 1 s ; on laisse une marge.
_LISTENER_JOIN_TIMEOUT_S = 2.0


# Mapping nom config → pynput.Key (ou caractère). Les alias "option"
# pointent vers alt — sur les claviers macOS la touche est labellisée
# "⌥ Option" et les utilisateurs tapent "option" plus souvent qu'"alt".
_NAMED_KEYS: dict[str, keyboard.Key] = {
    "alt_l": keyboard.Key.alt_l,
    "alt_r": keyboard.Key.alt_r,
    "alt": keyboard.Key.alt,
    "option_l": keyboard.Key.alt_l,
    "option_r": keyboard.Key.alt_r,
    "option": keyboard.Key.alt,
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


def parse_key(token: str) -> keyboard.Key | str:
    """Convertit un token ('alt_r', 'h', 'space') en clé pynput."""
    token = token.lower().strip()
    if token in _NAMED_KEYS:
        return _NAMED_KEYS[token]
    if len(token) == 1 and token != "+":
        return token  # caractère ASCII (ex. "h")
    raise ValueError(f"Touche inconnue : {token!r}")


def _is_single_key(combo: str) -> bool:
    """True si le combo désigne une touche unique (ex. 'alt_r', 'h').

    On vérifie strictement via le registre des touches connues plutôt que
    via `"+" not in combo` : ça évite d'accepter un combo mal formé
    ("xy", "+") comme single-key, puis d'échouer plus tard dans parse.
    """
    t = combo.lower().strip()
    return t in _NAMED_KEYS or (len(t) == 1 and t != "+")


def validate_combo(combo: str) -> str | None:
    """Retourne None si le combo est valide, sinon un message d'erreur."""
    combo = combo.strip()
    if not combo:
        return "Raccourci vide."
    tokens = [t.strip() for t in combo.split("+")]
    if any(not t for t in tokens):
        return f"Jetons vides dans '{combo}'."
    try:
        for t in tokens:
            parse_key(t)
    except ValueError as exc:
        return str(exc)
    return None


class HotkeyManager:
    """Gestionnaire de raccourci global, mode push-to-talk uniquement.

    Usage :
        mgr = HotkeyManager(
            combo="alt_r",
            on_start=lambda: ...,
            on_stop=lambda: ...,
        )
        mgr.start()
        ...
        mgr.stop()
    """

    def __init__(
        self,
        combo: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self.on_start = on_start
        self.on_stop = on_stop
        self._listener: keyboard.Listener | None = None
        self._active = False  # True pendant qu'on enregistre
        self._pressed: set[keyboard.Key | str] = set()
        self._last_rearm_at = 0.0
        self._configure(combo)

    def _configure(self, combo: str) -> None:
        """(Re)calcule les structures internes pour un combo donné."""
        self.combo = combo.lower().strip()

        if _is_single_key(self.combo):
            self._target_key: keyboard.Key | str = parse_key(self.combo)
            self._modifier_keys: set[keyboard.Key | str] = set()
            self._final_key: keyboard.Key | str | None = None
        else:
            tokens = [t.strip() for t in self.combo.split("+")]
            *modifiers, final = tokens
            self._target_key = None  # type: ignore[assignment]
            self._modifier_keys = {parse_key(t) for t in modifiers}
            self._final_key = parse_key(final)

        self._pressed.clear()

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

        # `pynput.keyboard.Listener` EST un Thread, et son callback tourne sur
        # lui. S'arrêter depuis son propre thread se bloquerait sur le join
        # ci-dessous. On transforme donc la règle « jamais depuis le callback »
        # en garde vérifiée plutôt qu'en simple commentaire.
        if threading.current_thread() is self._listener:
            print(
                "[HotkeyManager] stop() appelé depuis le thread du listener — "
                "ignoré (utilise le thread dictation-worker).",
                file=sys.stderr,
            )
            return

        listener = self._listener
        self._listener = None
        self._pressed.clear()
        self._active = False

        listener.stop()
        # Sans ce join, `update_binding` peut laisser deux listeners vivants
        # en parallèle jusqu'à 1 s (le tour de CFRunLoop de pynput), ce qui
        # délivre les appuis en double. La machine à états les tolère, mais
        # autant ne pas les créer.
        try:
            listener.join(timeout=_LISTENER_JOIN_TIMEOUT_S)
        except RuntimeError:
            # join() sur un thread jamais démarré : sans conséquence.
            pass

    def update_binding(self, combo: str) -> None:
        """Reconfigure le raccourci sans redémarrer l'app."""
        self.stop()
        self._configure(combo)
        self.start()

    def rearm(self) -> bool:
        """Recrée le listener — donc un CGEventTap neuf — après un blocage.

        macOS désactive un event tap dont le callback dépasse environ une
        seconde, en émettant `kCGEventTapDisabledByTimeout`. pynput 1.8.1 ne
        traite jamais cet événement : le tap reste mort et le raccourci est
        perdu jusqu'au redémarrage de l'app.

        On ne peut pas le réactiver proprement : pynput garde le tap dans une
        variable locale de `ListenerMixin._run`, donc `CGEventTapEnable` est
        hors d'atteinte, et `listener.running` reste `True` sur un tap mort —
        aucune détection directe n'est possible. La seule issue est de
        reconstruire le listener.

        Limité par `_REARM_MIN_INTERVAL_S` : sans ça, une cause de blocage
        persistante ferait boucler la reconstruction.

        NE JAMAIS appeler depuis le callback du tap (cf. le garde dans
        `stop()`) : réservé au thread `dictation-worker`.

        Retourne True si le réarmement a bien eu lieu.
        """
        now = time.monotonic()
        if now - self._last_rearm_at < _REARM_MIN_INTERVAL_S:
            return False
        self._last_rearm_at = now

        print(
            "[HotkeyManager] réarmement du raccourci (event tap probablement "
            "désactivé par macOS).",
            file=sys.stderr,
        )
        self.update_binding(self.combo)
        return True

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
            if norm != self._target_key:
                return
        else:
            # Combinaison : tous les modifs ET la touche finale doivent être pressés
            if not self._modifier_keys.issubset(self._pressed):
                return
            if norm != self._final_key:
                return

        if not self._active:
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

        # Combinaison : on stoppe dès qu'on relâche la touche finale OU
        # n'importe quel modificateur (sinon on reste bloqué en "écoute"
        # si le timing du release est imparfait).
        if norm == self._final_key or norm in self._modifier_keys:
            self._active = False
            self._safe_call(self.on_stop)

    @staticmethod
    def _safe_call(fn: Callable[[], None]) -> None:
        """Encapsule l'appel callback : une exception dans on_start/on_stop
        ne doit PAS tuer le listener clavier (sinon le raccourci ne marche
        plus jusqu'au redémarrage de l'app)."""
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
        "option": "⌥", "option_l": "⌥ gauche", "option_r": "⌥ droite",
        "ctrl": "⌃", "ctrl_l": "⌃ gauche", "ctrl_r": "⌃ droite",
        "space": "␣", "enter": "↩", "tab": "⇥", "esc": "⎋",
    }
    if "+" not in combo:
        return pretty.get(combo, combo.upper())
    return "".join(pretty.get(t, t.upper()) for t in combo.split("+"))
