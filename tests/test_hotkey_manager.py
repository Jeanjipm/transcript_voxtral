"""Tests d'hotkey_manager.py — parse, validate, state machine push-to-talk."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
import pynput.keyboard as kbd

import hotkey_manager
from hotkey_manager import (
    HotkeyManager,
    _is_single_key,
    display_combo,
    parse_key,
    token_for_key,
    validate_combo,
)


# ---- parse_key ----


def test_parse_key_single_char():
    """Lettre simple = char str."""
    assert parse_key("h") == "h"


def test_parse_key_lowercases_char():
    assert parse_key("H") == "h"


def test_parse_key_named_modifier():
    """alt_r → l'objet Key.alt_r."""
    result = parse_key("alt_r")
    # On a un stub _KeyStub avec name="alt_r"
    assert hasattr(result, "name")
    assert result.name == "alt_r"


def test_parse_key_option_alias():
    """'option' est un alias d'alt sur les claviers macOS."""
    alt = parse_key("alt")
    option = parse_key("option")
    assert alt.name == option.name == "alt"


def test_parse_key_unknown_raises():
    with pytest.raises(ValueError, match="inconnue"):
        parse_key("touche_imaginaire")


def test_parse_key_strips_whitespace():
    assert parse_key("  h  ") == "h"


def test_parse_key_function_keys():
    """F13-F19 sont les touches usuelles pour des hotkeys d'app perso."""
    for n in range(13, 20):
        result = parse_key(f"f{n}")
        assert result.name == f"f{n}"


# ---- _is_single_key ----


def test_is_single_key_named():
    assert _is_single_key("alt_r") is True
    assert _is_single_key("option") is True
    assert _is_single_key("space") is True


def test_is_single_key_char():
    assert _is_single_key("h") is True


def test_is_single_key_combo_with_plus():
    assert _is_single_key("ctrl+space") is False
    assert _is_single_key("cmd+shift+h") is False


def test_is_single_key_rejects_plus_alone():
    """Edge case : '+' tout seul n'est pas une touche valide."""
    assert _is_single_key("+") is False


def test_is_single_key_rejects_garbage():
    """'xy' n'est ni une touche connue ni un caractère unique."""
    assert _is_single_key("xy") is False


# ---- validate_combo ----


def test_validate_combo_accepts_single_key():
    assert validate_combo("alt_r") is None


def test_validate_combo_accepts_combo():
    assert validate_combo("ctrl+space") is None
    assert validate_combo("cmd+shift+h") is None


def test_validate_combo_rejects_empty():
    msg = validate_combo("")
    assert msg is not None
    assert "vide" in msg.lower()


def test_validate_combo_rejects_whitespace_only():
    msg = validate_combo("   ")
    assert msg is not None


def test_validate_combo_rejects_double_plus():
    """'a++b' donne un token vide → message d'erreur explicite."""
    msg = validate_combo("a++b")
    assert msg is not None
    assert "vide" in msg.lower()


def test_validate_combo_rejects_unknown_token():
    msg = validate_combo("ctrl+foobar")
    assert msg is not None
    assert "inconnue" in msg.lower()


# ---- HotkeyManager : state machine push-to-talk ----


@pytest.fixture
def starts_stops():
    """Tracker des appels on_start / on_stop pour les tests de state."""
    state = {"starts": 0, "stops": 0}

    def on_start():
        state["starts"] += 1

    def on_stop():
        state["stops"] += 1

    state["on_start"] = on_start
    state["on_stop"] = on_stop
    return state


def _press(mgr: HotkeyManager, name: str):
    """Helper : simule un appui sur une touche nommée."""
    if name in {"alt_r", "alt_l", "ctrl_l", "ctrl_r", "cmd_l", "cmd_r",
                "shift_l", "shift_r", "alt", "ctrl", "cmd", "shift", "space"}:
        # Stub _KeyStub
        class K:
            pass
        k = K()
        k.name = name
        # Le manager normalise via isinstance(key, keyboard.Key)
        # mais notre stub n'est pas instance de Key. On fait du monkey :
        # on appelle directement les méthodes internes en bypassant
        # _normalize, ou on utilise le stub global de conftest.
        # Solution simple : appeler _on_press avec un stub
        # qui imite l'API attendue.
        from pynput.keyboard import Key
        # _KeyEnumStub.__getattr__ retourne un _KeyStub
        actual_key = getattr(Key, name)
        mgr._on_press(actual_key)
    else:
        # KeyCode (caractère) — on utilise le KeyCodeStub
        from pynput.keyboard import KeyCode
        mgr._on_press(KeyCode(char=name))


def _release(mgr: HotkeyManager, name: str):
    if name in {"alt_r", "alt_l", "ctrl_l", "ctrl_r", "cmd_l", "cmd_r",
                "shift_l", "shift_r", "alt", "ctrl", "cmd", "shift", "space"}:
        from pynput.keyboard import Key
        actual_key = getattr(Key, name)
        mgr._on_release(actual_key)
    else:
        from pynput.keyboard import KeyCode
        mgr._on_release(KeyCode(char=name))


def test_single_key_press_triggers_start(starts_stops):
    """Push-to-talk avec touche unique : press → on_start."""
    mgr = HotkeyManager(
        combo="alt_r",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    _press(mgr, "alt_r")
    assert starts_stops["starts"] == 1
    assert starts_stops["stops"] == 0


def test_single_key_release_triggers_stop(starts_stops):
    mgr = HotkeyManager(
        combo="alt_r",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    _press(mgr, "alt_r")
    _release(mgr, "alt_r")
    assert starts_stops["stops"] == 1


def test_auto_repeat_does_not_double_trigger(starts_stops):
    """macOS répète les press sur touche maintenue → on doit ignorer."""
    mgr = HotkeyManager(
        combo="alt_r",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    _press(mgr, "alt_r")
    _press(mgr, "alt_r")  # auto-repeat
    _press(mgr, "alt_r")  # auto-repeat
    assert starts_stops["starts"] == 1


def test_other_key_does_not_trigger_single(starts_stops):
    """Une autre touche que celle configurée ne doit rien déclencher."""
    mgr = HotkeyManager(
        combo="alt_r",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    _press(mgr, "h")
    assert starts_stops["starts"] == 0


def test_combo_requires_all_modifiers(starts_stops):
    """ctrl+shift+h : ctrl seul ne déclenche rien, h seul non plus."""
    mgr = HotkeyManager(
        combo="ctrl+shift+h",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    _press(mgr, "ctrl")
    assert starts_stops["starts"] == 0
    _press(mgr, "shift")
    assert starts_stops["starts"] == 0
    _press(mgr, "h")
    # Ctrl ET shift ET h tous pressés → trigger
    assert starts_stops["starts"] == 1


def test_combo_release_any_modifier_stops(starts_stops):
    """En cours d'enregistrement combo, relâcher n'importe quelle touche
    du combo doit stopper (sinon on reste bloqué si timing imparfait)."""
    mgr = HotkeyManager(
        combo="ctrl+shift+h",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    _press(mgr, "ctrl")
    _press(mgr, "shift")
    _press(mgr, "h")
    assert starts_stops["starts"] == 1

    # Relâche ctrl en premier (avant h) → doit quand même stopper
    _release(mgr, "ctrl")
    assert starts_stops["stops"] == 1


def test_callback_exception_does_not_kill_listener(starts_stops):
    """Si on_start lève, on doit logger mais pas tuer le listener (sinon
    le hotkey serait mort jusqu'au redémarrage de l'app)."""
    def boom():
        raise RuntimeError("oops")

    mgr = HotkeyManager(combo="alt_r", on_start=boom, on_stop=lambda: None)
    # Ne doit pas raise
    _press(mgr, "alt_r")


# ---- update_binding (hot-reload) ----


def test_update_binding_changes_target_key(starts_stops):
    """Hot-reload du raccourci : on doit pouvoir changer sans redémarrer."""
    mgr = HotkeyManager(
        combo="alt_r",
        on_start=starts_stops["on_start"],
        on_stop=starts_stops["on_stop"],
    )
    mgr.update_binding("ctrl_l")

    # Ancien combo ne déclenche plus
    _press(mgr, "alt_r")
    assert starts_stops["starts"] == 0

    # Nouveau combo déclenche
    _press(mgr, "ctrl_l")
    assert starts_stops["starts"] == 1


# ---- display_combo ----


def test_display_combo_single_named():
    assert display_combo("alt_r") == "⌥ droite"
    assert display_combo("cmd_l") == "⌘ gauche"


def test_display_combo_single_char():
    """Caractère single → uppercase."""
    assert display_combo("h") == "H"


def test_display_combo_combination():
    """Combo : on concatène les jolies représentations."""
    result = display_combo("cmd+shift+h")
    assert "⌘" in result
    assert "⇧" in result
    assert "H" in result


def test_display_combo_unknown_named_falls_back():
    """Token inconnu → uppercase fallback."""
    assert display_combo("foo") == "FOO"


# ---- Régression sprint 5 : réarmement après désactivation du tap ----


@pytest.fixture
def distinct_listeners(monkeypatch: pytest.MonkeyPatch):
    """Remplace keyboard.Listener par une fabrique rendant des instances
    DISTINCTES.

    Le stub global de conftest est un MagicMock, dont chaque appel retourne
    le MÊME `return_value` : impossible de vérifier qu'un listener a bien été
    remplacé par un autre.
    """
    created: list = []

    def factory(**_kwargs):
        listener = MagicMock(name=f"Listener{len(created)}")
        created.append(listener)
        return listener

    monkeypatch.setattr("hotkey_manager.keyboard.Listener", factory)
    return created


def _mgr(combo: str = "alt_r") -> HotkeyManager:
    return HotkeyManager(combo=combo, on_start=lambda: None, on_stop=lambda: None)


def test_rearm_recreates_the_listener(distinct_listeners):
    """rearm() doit reconstruire le listener : c'est la seule façon d'obtenir
    un CGEventTap neuf, pynput gardant le tap hors d'atteinte."""
    mgr = _mgr()
    mgr.start()
    first = mgr._listener

    assert mgr.rearm() is True
    assert mgr._listener is not first
    assert mgr._listener is distinct_listeners[1]
    first.stop.assert_called_once()


def test_rearm_is_rate_limited(distinct_listeners):
    """Deux réarmements rapprochés = un seul effet. Sans ça, une cause de
    blocage persistante ferait boucler la reconstruction."""
    mgr = _mgr()
    mgr.start()

    assert mgr.rearm() is True
    assert mgr.rearm() is False
    assert len(distinct_listeners) == 2  # 1 au start + 1 au réarmement


def test_rearm_allowed_again_after_the_interval(distinct_listeners):
    mgr = _mgr()
    mgr.start()
    assert mgr.rearm() is True

    # Simule le passage du temps au-delà de l'intervalle minimum.
    mgr._last_rearm_at -= hotkey_manager._REARM_MIN_INTERVAL_S + 1.0
    assert mgr.rearm() is True


def test_rearm_preserves_the_combo(distinct_listeners):
    mgr = _mgr("cmd+shift+h")
    mgr.start()
    mgr.rearm()
    assert mgr.combo == "cmd+shift+h"
    assert mgr._final_key == "h"


def test_rearm_clears_pressed_keys(distinct_listeners):
    """Le set des touches enfoncées doit repartir de zéro : les relâchements
    survenus pendant que le tap était mort n'arriveront jamais."""
    mgr = _mgr()
    mgr.start()
    _press(mgr, "alt_r")
    assert mgr._pressed
    assert mgr._active is True

    mgr.rearm()
    assert mgr._pressed == set()
    assert mgr._active is False


def test_stop_refuses_to_run_from_the_listener_thread(distinct_listeners, capsys):
    """pynput.Listener EST un Thread et son callback tourne dessus : s'arrêter
    depuis là se bloquerait sur le join. Le garde doit refuser l'appel."""
    mgr = _mgr()
    mgr.start()

    # Fait croire au garde qu'on est sur le thread du listener.
    mgr._listener = threading.current_thread()  # type: ignore[assignment]
    mgr.stop()

    assert "thread du listener" in capsys.readouterr().err
    # L'appel a été refusé : le listener n'a pas été détaché.
    assert mgr._listener is threading.current_thread()


def test_stop_joins_the_listener(distinct_listeners):
    """Sans join, update_binding laisse deux taps vivants jusqu'à 1 s, ce qui
    délivre les appuis en double."""
    mgr = _mgr()
    mgr.start()
    listener = mgr._listener

    mgr.stop()

    listener.stop.assert_called_once()
    listener.join.assert_called_once()
    assert listener.join.call_args.kwargs["timeout"] > 0
    assert mgr._listener is None


def test_stop_survives_join_runtimeerror(distinct_listeners):
    """join() sur un thread jamais démarré lève RuntimeError : sans
    conséquence, ne doit pas remonter."""
    mgr = _mgr()
    mgr.start()
    mgr._listener.join.side_effect = RuntimeError("thread not started")

    mgr.stop()  # ne doit pas lever
    assert mgr._listener is None


# ---- Sprint 6 : vocabulaire de touches et noms canoniques ----
#
# Ces fonctions existent pour l'enregistrement de raccourci
# (cf. hotkey_capture.py). Elles sont testées ici parce que c'est
# hotkey_manager qui possède le vocabulaire.


def test_token_for_key_gives_the_canonical_name():
    assert token_for_key(kbd.Key.alt_r) == "alt_r"
    assert token_for_key(kbd.Key.f13) == "f13"


def test_token_for_left_modifier_is_the_generic_name():
    """Sur macOS, Key.alt et Key.alt_l sont un seul objet. Le nom rendu ne
    doit pas dépendre de l'ordre d'itération d'un dictionnaire."""
    assert token_for_key(kbd.Key.alt) == "alt"
    assert token_for_key(kbd.Key.alt_l) == "alt"


def test_token_for_key_accepts_a_character():
    assert token_for_key("h") == "h"
    assert token_for_key("H") == "h"


def test_token_for_key_returns_none_when_unnameable():
    """Une touche absente du vocabulaire n'est pas enregistrable : le None
    est ce qui permet de le DIRE au lieu d'écrire un raccourci mort."""
    assert token_for_key("+") is None
    assert token_for_key("ab") is None


def test_named_keys_cover_the_extended_vocabulary():
    for token in ("f1", "f12", "up", "page_down", "backspace", "caps_lock"):
        assert validate_combo(token) is None, token


def test_format_combo_orders_modifiers_canonically():
    assert hotkey_manager.format_combo(["shift", "cmd", "h"]) == "cmd+shift+h"
    assert hotkey_manager.format_combo(["h", "shift", "cmd"]) == "cmd+shift+h"


def test_format_combo_keeps_the_last_modifier_as_final():
    """Sans non-modificateur, c'est le dernier pressé qui déclenche."""
    assert hotkey_manager.format_combo(["cmd", "shift"]) == "cmd+shift"


def test_format_combo_single_token_is_unchanged():
    assert hotkey_manager.format_combo(["alt_r"]) == "alt_r"
    assert hotkey_manager.format_combo([]) == ""


def test_generic_combo_drops_the_side():
    """macOS ne distingue pas les côtés pour SES raccourcis : ⌘ droite +
    Espace ouvre Spotlight autant que ⌘ gauche. Sans cette normalisation, la
    détection de conflit rate un cas sur deux."""
    assert hotkey_manager.generic_combo("cmd_r+space") == "cmd+space"
    assert hotkey_manager.generic_combo("option_r") == "alt"
    assert hotkey_manager.generic_combo("cmd+shift+h") == "cmd+shift+h"


def test_is_modifier():
    assert hotkey_manager.is_modifier("cmd_r") is True
    assert hotkey_manager.is_modifier("option") is True
    assert hotkey_manager.is_modifier("space") is False


def test_display_combo_verbose_names_the_side():
    """« ⌥ » seul dans un champ ne dit pas quelle touche a été enregistrée."""
    assert display_combo("alt_r", verbose=True) == "⌥ Option droite"
    assert display_combo("alt", verbose=True) == "⌥ Option gauche"


def test_display_combo_verbose_leaves_combinations_compact():
    assert display_combo("cmd+shift+h", verbose=True) == "⌘⇧H"
