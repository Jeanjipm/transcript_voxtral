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
from collections.abc import Sequence
from typing import Callable

from pynput import keyboard


# Intervalle minimum entre deux réarmements du listener. Empêche une cause de
# blocage persistante de faire boucler la reconstruction du tap.
_REARM_MIN_INTERVAL_S = 10.0

# Attente maximale de la fin du thread listener. pynput sort de sa boucle au
# tour de CFRunLoop suivant, soit jusqu'à 1 s ; on laisse une marge.
_LISTENER_JOIN_TIMEOUT_S = 2.0


# Vocabulaire des touches nommées, dans l'ordre où on veut les voir gagner
# quand plusieurs noms désignent la MÊME touche physique (cf. _TOKEN_BY_KEY).
#
# Détail macOS mesuré, et il compte : dans pynput, `Key.alt` et `Key.alt_l`
# partagent le même code virtuel (0x3A) et sont donc le même objet — pareil
# pour cmd/cmd_l, ctrl/ctrl_l, shift/shift_l. Autrement dit, sur un Mac,
# écrire « cmd » dans la config désigne le Commande *gauche*, pas « l'un ou
# l'autre ». Les touches de droite ont bien un code distinct (cmd_r = 0x36).
_KEY_NAMES: tuple[str, ...] = (
    # Modificateurs : la forme générique d'abord, c'est le nom canonique.
    "cmd", "cmd_l", "cmd_r",
    "ctrl", "ctrl_l", "ctrl_r",
    "alt", "alt_l", "alt_r",
    "shift", "shift_l", "shift_r",
    # Touches ordinaires
    "space", "enter", "tab", "esc", "backspace", "delete", "caps_lock",
    "up", "down", "left", "right",
    "home", "end", "page_up", "page_down",
) + tuple(f"f{i}" for i in range(1, 21))


# Les alias "option" pointent vers alt — sur les claviers macOS la touche est
# labellisée "⌥ Option" et les utilisateurs tapent "option" plus souvent
# qu'"alt". Ce sont des entrées acceptées en lecture, jamais produites.
_KEY_ALIASES: dict[str, str] = {
    "option": "alt",
    "option_l": "alt_l",
    "option_r": "alt_r",
}


def _build_named_keys() -> dict[str, keyboard.Key]:
    """Construit le mapping nom → touche pynput, en sautant les absentes.

    `getattr` plutôt qu'un dictionnaire littéral : toutes les touches
    listées n'existent pas sur toutes les plateformes ni dans toutes les
    versions de pynput, et un `AttributeError` à l'import rendrait l'app
    non lançable pour une touche que personne n'utilise.
    """
    named: dict[str, keyboard.Key] = {}
    for name in _KEY_NAMES:
        key = getattr(keyboard.Key, name, None)
        if key is not None:
            named[name] = key
    for alias, target in _KEY_ALIASES.items():
        if target in named:
            named[alias] = named[target]
    return named


_NAMED_KEYS: dict[str, keyboard.Key] = _build_named_keys()


# Mapping inverse : touche pynput → nom canonique. Le `setdefault` est ce qui
# fait le travail — plusieurs noms partageant un même objet touche, c'est le
# premier de `_KEY_NAMES` qui l'emporte. Sans cet ordre explicite, le nom rendu
# par la capture dépendrait de l'ordre d'itération d'un dictionnaire.
_TOKEN_BY_KEY: dict[keyboard.Key, str] = {}
for _token, _key in _NAMED_KEYS.items():
    _TOKEN_BY_KEY.setdefault(_key, _token)


# Modificateurs, par famille. Sert à deux choses : décider quel jeton est la
# touche « finale » d'une combinaison, et ramener un raccourci à sa forme
# générique pour la détection de conflits.
_MODIFIER_FAMILIES: dict[str, str] = {
    "cmd": "cmd", "cmd_l": "cmd", "cmd_r": "cmd",
    "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt",
    "option": "alt", "option_l": "alt", "option_r": "alt",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
}

# Ordre d'écriture des modificateurs dans un raccourci. Choisi pour coller à
# celui déjà utilisé dans `KNOWN_SYSTEM_CONFLICTS` (cmd+option+h,
# cmd+ctrl+space, cmd+shift+h) : sinon la capture produirait des raccourcis
# corrects mais dont la détection de conflit ne verrait rien.
_MODIFIER_ORDER: tuple[str, ...] = ("cmd", "ctrl", "alt", "shift")


def is_modifier(token: str) -> bool:
    """True si le jeton désigne une touche de modification (⌘ ⌃ ⌥ ⇧)."""
    return token.lower().strip() in _MODIFIER_FAMILIES


def generic_combo(combo: str) -> str:
    """Ramène un raccourci à sa forme sans gauche/droite ('cmd_r+space' →
    'cmd+space').

    macOS ne distingue pas les côtés pour ses propres raccourcis : ⌘ droite +
    Espace ouvre Spotlight tout autant que ⌘ gauche. La détection de conflit
    doit donc comparer sur cette forme, sinon elle laisse passer la moitié
    des cas.
    """
    return "+".join(
        _MODIFIER_FAMILIES.get(t.strip().lower(), t.strip().lower())
        for t in combo.split("+")
    )


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


def normalize_key(key: object) -> keyboard.Key | str | None:
    """Normalise un événement pynput (KeyCode → char, Key → Key, sinon None).

    Fonction de module et non méthode : l'enregistreur de raccourci
    (`hotkey_capture`) doit normaliser exactement comme le fait la mise en
    correspondance à l'exécution. Deux implémentations qui divergeraient d'un
    cheveu donneraient un raccourci enregistré qui ne se déclenche jamais.
    """
    if isinstance(key, keyboard.Key):
        return key
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return key.char.lower()
    return None


def token_for_key(key: keyboard.Key | str) -> str | None:
    """Nom de configuration d'une touche normalisée, ou None si inexprimable.

    None signifie « cette touche n'a pas de nom dans notre vocabulaire » —
    une touche média, un pavé numérique exotique. L'appelant doit le dire à
    l'utilisateur plutôt que d'enregistrer un raccourci qui ne marchera pas.
    """
    if isinstance(key, str):
        token = key.lower()
        return token if len(token) == 1 and token != "+" else None
    return _TOKEN_BY_KEY.get(key)


def format_combo(tokens: Sequence[str]) -> str:
    """Assemble des jetons en raccourci canonique ('h', 'cmd+shift+h').

    Deux règles :
    - la touche finale est la première non-modificateur ; s'il n'y en a pas,
      c'est le dernier modificateur pressé (tenir ⌘ puis appuyer sur ⇧ donne
      bien `cmd+shift`, pas `shift+cmd`) ;
    - les modificateurs restants sont écrits dans `_MODIFIER_ORDER`, pour que
      deux enregistrements des mêmes touches donnent toujours la même chaîne.
    """
    ordered = [t.lower().strip() for t in tokens if t and t.strip()]
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]

    plain = [t for t in ordered if not is_modifier(t)]
    final = plain[0] if plain else ordered[-1]
    modifiers = [t for t in ordered if t != final]
    modifiers.sort(
        key=lambda t: _MODIFIER_ORDER.index(_MODIFIER_FAMILIES[t])
        if t in _MODIFIER_FAMILIES
        else len(_MODIFIER_ORDER)
    )
    return "+".join([*modifiers, final])


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

    ## Un seul listener pour toute la session

    Règle de conception, apprise par un crash : **on ne détruit pas un
    listener pour en refabriquer un**. Ni pour changer de raccourci, ni pour
    suspendre temporairement la dictée.

    Pourquoi. `pynput.keyboard.Listener._run` ouvre un contexte de
    disposition clavier (`keycode_context`) qui appelle Carbon
    `TISGetInputSourceProperty` — et il le fait depuis le thread du listener,
    pas depuis celui qui a construit l'objet. Or macOS exige que le Text
    Services Manager soit interrogé depuis la file principale quand sa liste
    de sources de saisie doit être reconstruite. Quand ce n'est pas le cas,
    HIToolbox ne rend pas une erreur : il déclenche
    `dispatch_assert_queue_fail`, et le processus meurt sur SIGTRAP.

    Ce qui invalide cette liste, c'est un changement d'application active —
    typiquement la fermeture de la fenêtre de préférences. Fabriquer un
    listener à cet instant précis, c'est exactement la recette du crash
    observé (rapport `Python-2026-07-31-092748.ips`).

    Conséquence : `update_binding` reconfigure en place, et la mise en pause
    passe par un drapeau. Les callbacks lisent `self.combo` à chaque
    événement, donc changer le raccourci ne demande aucun nouveau tap.
    Reconstruire reste possible via `rearm()`, seul cas où c'est le but même
    de l'opération — et c'est un chemin de récupération, pas un chemin normal.
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
        self._paused = False  # True quand la fenêtre de réglages est ouverte
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
        """Change le raccourci écouté, sans toucher au listener.

        Les callbacks lisent `self.combo` à chaque événement : reconfigurer
        les structures internes suffit, le tap en cours reconnaît le nouveau
        raccourci dès l'appui suivant.

        L'ancienne version arrêtait puis recréait le listener. Ça marchait,
        mais c'était le chemin qui menait au crash décrit dans la docstring
        de la classe — et il se déclenchait au pire moment, juste après la
        fermeture de la fenêtre de préférences qui venait de changer le
        raccourci. Appelable depuis n'importe quel thread.
        """
        self._configure(combo)
        self._active = False

    def pause(self) -> None:
        """Suspend le raccourci sans démonter le listener.

        Utilisé pendant que la fenêtre de réglages est ouverte : sans ça,
        appuyer sur ⌥ droite pour l'enregistrer déclencherait une dictée.

        Un drapeau plutôt qu'un `stop()` : arrêter puis relancer le listener
        reviendrait à en fabriquer un neuf au retour, ce que la docstring de
        la classe explique être mortel. Appelable depuis n'importe quel
        thread, et instantané — pas d'attente de fin de thread.
        """
        self._paused = True
        # Si une dictée était en cours au moment de la pause, on ne verra
        # jamais le relâchement : on referme proprement plutôt que de laisser
        # la machine à états croire qu'on enregistre toujours.
        if self._active:
            self._active = False
            self._safe_call(self.on_stop)
        self._pressed.clear()

    def resume(self) -> None:
        """Réactive le raccourci. Sans effet s'il n'était pas en pause."""
        self._pressed.clear()
        self._active = False
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

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

        C'est le SEUL endroit qui refabrique un listener, et c'est assumé :
        sans tap neuf, le raccourci reste mort. Le risque décrit dans la
        docstring de la classe est accepté ici parce que l'alternative est
        une app inutilisable, et parce qu'on n'arrive dans ce chemin qu'après
        un dépassement de durée maximale — pas au fil de l'eau, et jamais au
        moment où macOS change d'application active.

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
        self.stop()
        self._configure(self.combo)
        self.start()
        return True

    def _normalize(self, key: object) -> keyboard.Key | str | None:
        """Normalise key (KeyCode → char str, Key → Key, autre → None)."""
        return normalize_key(key)

    def _on_press(self, key: object) -> None:
        if self._paused:
            return
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
        if self._paused:
            return
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


_PRETTY_KEYS: dict[str, str] = {
    "cmd": "⌘", "cmd_l": "⌘ gauche", "cmd_r": "⌘ droite",
    "shift": "⇧", "shift_l": "⇧ gauche", "shift_r": "⇧ droite",
    "alt": "⌥", "alt_l": "⌥ gauche", "alt_r": "⌥ droite",
    "option": "⌥", "option_l": "⌥ gauche", "option_r": "⌥ droite",
    "ctrl": "⌃", "ctrl_l": "⌃ gauche", "ctrl_r": "⌃ droite",
    "space": "␣", "enter": "↩", "tab": "⇥", "esc": "⎋",
    "backspace": "⌫", "delete": "⌦", "caps_lock": "⇪",
    "up": "↑", "down": "↓", "left": "←", "right": "→",
    "home": "↖", "end": "↘", "page_up": "⇞", "page_down": "⇟",
}

# Libellés complets, pour la fenêtre de préférences : « ⌥ » tout seul dans un
# champ ne dit pas à l'utilisateur quelle touche il vient d'enregistrer.
# Rappel du mapping macOS : la forme générique EST la touche de gauche.
_VERBOSE_KEYS: dict[str, str] = {
    "cmd": "⌘ Commande gauche", "cmd_l": "⌘ Commande gauche",
    "cmd_r": "⌘ Commande droite",
    "ctrl": "⌃ Contrôle gauche", "ctrl_l": "⌃ Contrôle gauche",
    "ctrl_r": "⌃ Contrôle droite",
    "alt": "⌥ Option gauche", "alt_l": "⌥ Option gauche",
    "alt_r": "⌥ Option droite",
    "option": "⌥ Option gauche", "option_l": "⌥ Option gauche",
    "option_r": "⌥ Option droite",
    "shift": "⇧ Majuscule gauche", "shift_l": "⇧ Majuscule gauche",
    "shift_r": "⇧ Majuscule droite",
    "space": "␣ Espace", "enter": "↩ Entrée", "tab": "⇥ Tabulation",
    "esc": "⎋ Échap", "backspace": "⌫ Retour arrière", "delete": "⌦ Suppr",
    "caps_lock": "⇪ Verr. maj",
}


def display_combo(combo: str, verbose: bool = False) -> str:
    """Joli label pour l'UI (ex. 'alt_r' → '⌥ droite', 'cmd+shift+h' → '⌘⇧H').

    `verbose=True` donne le nom complet d'une touche seule (« ⌥ Option
    droite ») : dans la fenêtre de préférences, le symbole seul ne suffit pas
    à savoir ce qu'on vient d'enregistrer.
    """
    combo = combo.strip()
    if not combo:
        return ""
    if "+" not in combo:
        if verbose and combo in _VERBOSE_KEYS:
            return _VERBOSE_KEYS[combo]
        return _PRETTY_KEYS.get(combo, combo.upper())
    return "".join(_PRETTY_KEYS.get(t, t.upper()) for t in combo.split("+"))
