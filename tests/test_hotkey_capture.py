"""Tests d'hotkey_capture.py — enregistrement d'un raccourci au clavier.

Le fil rouge : ce qu'on enregistre doit être exactement ce que
`HotkeyManager` saura reconnaître ensuite. Plusieurs tests vérifient donc le
couple capture → mise en correspondance, pas seulement la capture.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

import pytest
from pynput import keyboard

from hotkey_capture import ComboRecorder, HotkeyCapture, Outcome
from hotkey_manager import HotkeyManager, normalize_key, validate_combo


K = keyboard.Key


@pytest.fixture
def rec() -> ComboRecorder:
    return ComboRecorder()


def _hold(rec: ComboRecorder, *keys: object) -> str:
    """Presse toutes les touches, puis relâche la première. Rend le combo.

    Les touches passent par `normalize_key`, exactement comme le fait
    `HotkeyCapture` : le recorder ne voit jamais un événement pynput brut.
    """
    for key in keys:
        rec.press(normalize_key(key))
    return rec.release(normalize_key(keys[0])).combo


# ---- Touche seule ----


def test_single_modifier_gives_its_canonical_name(rec: ComboRecorder):
    assert _hold(rec, K.alt_r) == "alt_r"


def test_left_modifier_uses_the_generic_name(rec: ComboRecorder):
    """Sur macOS, `Key.alt` et `Key.alt_l` sont la même touche (code 0x3A).
    Le nom canonique retenu doit être stable — sinon deux enregistrements de
    la même touche écriraient deux valeurs différentes dans la config."""
    assert _hold(rec, K.alt) == "alt"
    rec.reset()
    assert _hold(rec, K.alt_l) == "alt"


def test_plain_character_key(rec: ComboRecorder):
    assert _hold(rec, keyboard.KeyCode("H")) == "h"


def test_function_key(rec: ComboRecorder):
    assert _hold(rec, K.f13) == "f13"


# ---- Combinaisons ----


def test_modifiers_then_letter(rec: ComboRecorder):
    assert _hold(rec, K.cmd, K.shift, keyboard.KeyCode("h")) == "cmd+shift+h"


def test_modifier_order_is_canonical_not_press_order(rec: ComboRecorder):
    """Peu importe l'ordre des doigts : la chaîne produite est toujours la
    même, sinon la détection de conflit macOS (qui compare des chaînes) ne
    verrait qu'un cas sur deux."""
    assert _hold(rec, K.shift, K.cmd, keyboard.KeyCode("h")) == "cmd+shift+h"


def test_two_modifiers_alone_keep_the_last_as_final(rec: ComboRecorder):
    """Tenir ⌘ puis appuyer sur ⇧ donne `cmd+shift` : c'est ⇧ qui déclenche,
    ⌘ est le modificateur. L'inverse ne se déclencherait jamais."""
    assert _hold(rec, K.cmd, K.shift) == "cmd+shift"


def test_release_order_does_not_change_the_result(rec: ComboRecorder):
    """LE piège de ce module : si on retenait « ce qui reste pressé » au lieu
    du maximum atteint, relâcher H en premier donnerait `cmd+shift`."""
    for key in (K.cmd, K.shift, keyboard.KeyCode("h")):
        rec.press(normalize_key(key))
    assert rec.release(normalize_key(keyboard.KeyCode("h"))).combo == "cmd+shift+h"


def test_second_letter_is_ignored(rec: ComboRecorder):
    """Une seule touche finale : `cmd+h+j` serait accepté par le parseur mais
    ne se déclencherait quasiment jamais."""
    assert _hold(
        rec, K.cmd, keyboard.KeyCode("h"), keyboard.KeyCode("j")
    ) == "cmd+h"


# ---- Cas limites ----


def test_autorepeat_does_not_duplicate(rec: ComboRecorder):
    """Le système répète `on_press` tant que la touche est tenue."""
    rec.press(normalize_key(K.cmd))
    for _ in range(5):
        rec.press(normalize_key(K.cmd))
    rec.press(normalize_key(keyboard.KeyCode("h")))
    assert rec.release(normalize_key(K.cmd)).combo == "cmd+h"


def test_escape_alone_cancels(rec: ComboRecorder):
    assert rec.press(normalize_key(K.esc)).outcome is Outcome.CANCELLED


def test_escape_after_a_modifier_is_a_normal_key(rec: ComboRecorder):
    """⌘⎋ doit rester enregistrable : Échap n'est une sortie de secours
    qu'en tout premier appui."""
    rec.press(normalize_key(K.cmd))
    assert rec.press(normalize_key(K.esc)).outcome is Outcome.HOLDING
    assert rec.release(normalize_key(K.cmd)).combo == "cmd+esc"


def test_unsupported_key_is_reported_not_swallowed(rec: ComboRecorder):
    """Une touche média n'a pas de nom dans notre vocabulaire. Il faut le
    dire, sinon l'utilisateur croit avoir enregistré un raccourci mort."""
    media = keyboard.KeyCode(None)
    assert rec.press(normalize_key(media)).outcome is Outcome.UNSUPPORTED


def test_unsupported_key_does_not_enter_the_combo(rec: ComboRecorder):
    rec.press(normalize_key(K.cmd))
    rec.press(normalize_key(keyboard.KeyCode(None)))
    assert rec.release(normalize_key(K.cmd)).combo == "cmd"


def test_release_without_any_press_stays_waiting(rec: ComboRecorder):
    """Touche déjà tenue au moment où la capture démarre : on n'a pas vu
    l'appui, il n'y a rien à conclure."""
    assert rec.release(normalize_key(K.alt_r)).outcome is Outcome.WAITING


def test_recorder_is_frozen_after_the_first_release(rec: ComboRecorder):
    _hold(rec, K.alt_r)
    rec.press(normalize_key(K.cmd))
    assert rec.release(normalize_key(K.cmd)).combo == "alt_r"


def test_reset_allows_a_new_capture(rec: ComboRecorder):
    _hold(rec, K.alt_r)
    rec.reset()
    assert _hold(rec, K.f13) == "f13"


# ---- Le contrat qui compte : capture → reconnaissance ----


@pytest.mark.parametrize(
    "keys",
    [
        (K.alt_r,),
        (K.cmd, keyboard.KeyCode("h")),
        (K.cmd, K.shift, keyboard.KeyCode("h")),
        (K.f13,),
        (K.ctrl_r,),
    ],
)
def test_captured_combo_is_always_valid(keys):
    """Tout ce que la capture produit doit passer `validate_combo`, sinon la
    sauvegarde refuserait un raccourci que l'utilisateur vient de presser."""
    combo = _hold(ComboRecorder(), *keys)
    assert validate_combo(combo) is None


@pytest.mark.parametrize(
    "keys",
    [
        (K.alt_r,),
        (K.cmd, keyboard.KeyCode("h")),
        (K.cmd, K.shift, keyboard.KeyCode("h")),
    ],
)
def test_captured_combo_actually_triggers_the_hotkey(keys):
    """Le test de bout en bout du module : on rejoue les mêmes touches dans
    un `HotkeyManager` configuré avec le raccourci capturé, et il doit
    déclencher. C'est la raison d'être du choix de pynput pour la capture."""
    combo = _hold(ComboRecorder(), *keys)

    started: list[bool] = []
    stopped: list[bool] = []
    mgr = HotkeyManager(
        combo=combo,
        on_start=lambda: started.append(True),
        on_stop=lambda: stopped.append(True),
    )

    for key in keys:
        mgr._on_press(key)
    assert started == [True], f"{combo} n'a pas démarré"

    mgr._on_release(keys[0])
    assert stopped == [True], f"{combo} n'a pas arrêté"


# ---- HotkeyCapture : plomberie du listener ----


@pytest.fixture
def capture():
    """Capture branchée sur une fabrique de listener factice."""
    made: list[MagicMock] = []

    def factory(**kwargs):  # noqa: ANN003
        listener = MagicMock(name=f"listener{len(made)}")
        listener.kwargs = kwargs
        made.append(listener)
        return listener

    cap = HotkeyCapture(listener_factory=factory)
    return cap, made


def _drain(events: queue.Queue) -> list:
    out = []
    while True:
        try:
            out.append(events.get_nowait())
        except queue.Empty:
            return out


def test_start_creates_and_starts_a_listener(capture):
    cap, made = capture
    cap.start()
    assert len(made) == 1
    made[0].start.assert_called_once()
    assert cap.active is True


def test_listener_is_never_suppressive(capture):
    """Un tap suppressif qui se coincerait rendrait le clavier inutilisable
    dans toutes les applications jusqu'à la mort du processus."""
    cap, made = capture
    cap.start()
    assert made[0].kwargs["suppress"] is False


def test_start_twice_does_not_stack_listeners(capture):
    cap, made = capture
    cap.start()
    cap.start()
    assert len(made) == 1


def test_stop_is_idempotent(capture):
    cap, made = capture
    cap.start()
    cap.stop()
    cap.stop()
    made[0].stop.assert_called_once()
    assert cap.active is False


def test_events_reach_the_queue(capture):
    cap, _ = capture
    cap.start()
    cap._on_press(K.alt_r)
    cap._on_release(K.alt_r)

    events = _drain(cap.events)
    assert [e.outcome for e in events] == [Outcome.HOLDING, Outcome.DONE]
    assert events[-1].combo == "alt_r"


def test_restart_clears_stale_events(capture):
    """Les appuis d'une capture annulée ne doivent pas être pris pour ceux de
    la suivante."""
    cap, _ = capture
    cap.start()
    cap._on_press(K.cmd)
    cap.stop()

    cap.start()
    assert _drain(cap.events) == []


def test_callback_never_raises_on_a_full_queue(capture):
    """Le callback tourne dans le thread du CGEventTap : y lever une
    exception tuerait le listener, et bloquer y ferait désactiver le tap par
    macOS (la cause n°1 du sprint 5)."""
    cap, _ = capture
    cap.start()
    for _ in range(cap.events.maxsize + 10):
        cap._on_press(K.cmd)
        cap._on_release(K.cmd)
    assert cap.active is True


def test_callback_swallows_a_recorder_failure(capture, monkeypatch):
    cap, _ = capture
    cap.start()
    monkeypatch.setattr(
        cap._recorder, "press", MagicMock(side_effect=RuntimeError("boum"))
    )
    cap._on_press(K.cmd)  # ne doit pas lever
