"""Tests de clipboard.py — paste_text + préservation clipboard + edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import clipboard
from clipboard import paste_text


# ---- paste_text : early return sur texte vide ----


def test_paste_text_empty_string_does_nothing(monkeypatch: pytest.MonkeyPatch):
    """Texte vide → ne pas écraser le clipboard ni simuler de paste."""
    fake_copy = MagicMock()
    fake_paste = MagicMock()
    monkeypatch.setattr("clipboard.copy_to_clipboard", fake_copy)
    monkeypatch.setattr("clipboard.simulate_paste", fake_paste)

    paste_text("")
    fake_copy.assert_not_called()
    fake_paste.assert_not_called()


def test_paste_text_whitespace_only_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Que du whitespace = considéré vide."""
    fake_copy = MagicMock()
    fake_paste = MagicMock()
    monkeypatch.setattr("clipboard.copy_to_clipboard", fake_copy)
    monkeypatch.setattr("clipboard.simulate_paste", fake_paste)

    paste_text("   \n\t  ")
    fake_copy.assert_not_called()
    fake_paste.assert_not_called()


# ---- paste_text : suffixe espace ----


def test_paste_text_appends_trailing_space(monkeypatch: pytest.MonkeyPatch):
    """Le texte collé est suffixé d'un espace pour préparer la dictée
    suivante (cf. clipboard.py docstring)."""
    captured = {}

    def fake_copy(text):
        captured["text"] = text

    monkeypatch.setattr("clipboard.copy_to_clipboard", fake_copy)
    monkeypatch.setattr("clipboard.simulate_paste", MagicMock())
    monkeypatch.setattr(
        "clipboard._read_clipboard_text", lambda: None
    )

    paste_text("Bonjour", auto_paste=False)
    assert captured["text"] == "Bonjour "


def test_paste_text_strips_existing_trailing_space_before_adding(
    monkeypatch: pytest.MonkeyPatch,
):
    """Si le texte se termine déjà par un espace, on ne double pas."""
    captured = {}
    monkeypatch.setattr(
        "clipboard.copy_to_clipboard",
        lambda t: captured.update(text=t),
    )
    monkeypatch.setattr("clipboard.simulate_paste", MagicMock())
    monkeypatch.setattr(
        "clipboard._read_clipboard_text", lambda: None
    )

    paste_text("Bonjour   ", auto_paste=False)
    # rstrip puis " " → "Bonjour "
    assert captured["text"] == "Bonjour "


# ---- paste_text : auto_paste + préservation clipboard ----


def test_paste_text_auto_paste_true_calls_simulate(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_simulate = MagicMock()
    monkeypatch.setattr("clipboard.copy_to_clipboard", MagicMock())
    monkeypatch.setattr("clipboard.simulate_paste", fake_simulate)
    monkeypatch.setattr(
        "clipboard._read_clipboard_text", lambda: None
    )

    paste_text("hello", auto_paste=True)
    fake_simulate.assert_called_once()


def test_paste_text_auto_paste_false_skips_simulate(
    monkeypatch: pytest.MonkeyPatch,
):
    """Mode copy-only : on copie mais on ne simule pas Cmd+V."""
    fake_simulate = MagicMock()
    monkeypatch.setattr("clipboard.copy_to_clipboard", MagicMock())
    monkeypatch.setattr("clipboard.simulate_paste", fake_simulate)
    monkeypatch.setattr(
        "clipboard._read_clipboard_text", lambda: None
    )

    paste_text("hello", auto_paste=False)
    fake_simulate.assert_not_called()


def test_paste_text_preserves_clipboard_after_paste(
    monkeypatch: pytest.MonkeyPatch,
):
    """preserve_clipboard=True (default) : on lit l'ancien contenu, on
    paste le nouveau, puis on restaure l'ancien dans le clipboard."""
    copy_calls = []

    def fake_copy(text):
        copy_calls.append(text)

    monkeypatch.setattr("clipboard.copy_to_clipboard", fake_copy)
    monkeypatch.setattr("clipboard.simulate_paste", MagicMock())
    monkeypatch.setattr(
        "clipboard._read_clipboard_text", lambda: "ancien_contenu"
    )
    # On élimine le sleep pour que le test soit instantané
    monkeypatch.setattr("clipboard.time.sleep", lambda _: None)

    paste_text("nouveau", auto_paste=True)

    # 1er appel : on copie le nouveau texte
    # 2e appel : on restaure l'ancien
    assert len(copy_calls) == 2
    assert copy_calls[0] == "nouveau "
    assert copy_calls[1] == "ancien_contenu"


def test_paste_text_does_not_preserve_when_copy_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """En mode copy-only, on veut que le nouveau texte reste dans le
    clipboard (l'utilisateur veut probablement le re-utiliser)."""
    copy_calls = []
    monkeypatch.setattr(
        "clipboard.copy_to_clipboard",
        lambda t: copy_calls.append(t),
    )
    fake_read = MagicMock(return_value="ancien")
    monkeypatch.setattr("clipboard._read_clipboard_text", fake_read)

    paste_text("nouveau", auto_paste=False, preserve_clipboard=True)

    # On ne doit même pas avoir lu l'ancien clipboard.
    fake_read.assert_not_called()
    # Une seule copie : le nouveau texte.
    assert copy_calls == ["nouveau "]


def test_paste_text_preserve_disabled_does_not_restore(
    monkeypatch: pytest.MonkeyPatch,
):
    """preserve_clipboard=False explicite : pas de restauration même en
    auto_paste."""
    copy_calls = []
    monkeypatch.setattr(
        "clipboard.copy_to_clipboard",
        lambda t: copy_calls.append(t),
    )
    monkeypatch.setattr("clipboard.simulate_paste", MagicMock())
    fake_read = MagicMock(return_value="ancien")
    monkeypatch.setattr("clipboard._read_clipboard_text", fake_read)

    paste_text("nouveau", auto_paste=True, preserve_clipboard=False)

    fake_read.assert_not_called()
    assert copy_calls == ["nouveau "]


# ---- paste_text : ordre d'opérations ----


def test_paste_text_copies_before_simulating(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sequence critique : copier dans le clipboard AVANT le Cmd+V simulé,
    sinon on colle l'ancien contenu."""
    sequence = []

    def fake_copy(text):
        sequence.append(("copy", text))

    def fake_simulate():
        sequence.append(("simulate",))

    monkeypatch.setattr("clipboard.copy_to_clipboard", fake_copy)
    monkeypatch.setattr("clipboard.simulate_paste", fake_simulate)
    monkeypatch.setattr(
        "clipboard._read_clipboard_text", lambda: None
    )

    paste_text("test", auto_paste=True)

    # Le 1er événement doit être un copy, le 2e un simulate.
    assert sequence[0][0] == "copy"
    assert sequence[1][0] == "simulate"
