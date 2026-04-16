"""
Presse-papier macOS + paste simulé.

`paste_text(text)` :
  1. Écrit le texte dans le presse-papier système
  2. Si auto_paste=True, simule Cmd+V à la position du curseur

Backend presse-papier : `AppKit.NSPasteboard` (natif), avec fallback
`subprocess pbcopy` si pyobjc indisponible.

Backend paste : `pynput.keyboard.Controller` (Cmd+V virtuel).
Nécessite la permission Accessibilité (cf. README).
"""

from __future__ import annotations

import subprocess
import time

from pynput.keyboard import Controller, Key


# Tentative d'import pyobjc pour pasteboard natif
try:
    from AppKit import NSPasteboard, NSStringPboardType  # type: ignore[import-not-found]

    _HAS_NSPASTEBOARD = True
except ImportError:  # pragma: no cover (macOS+pyobjc only)
    _HAS_NSPASTEBOARD = False


_keyboard = Controller()


def copy_to_clipboard(text: str) -> None:
    """Place le texte dans le presse-papier système."""
    if _HAS_NSPASTEBOARD:
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSStringPboardType)
    else:
        # Fallback CLI macOS (toujours présent sur Mac)
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(text.encode("utf-8"))


def simulate_paste() -> None:
    """Simule Cmd+V (paste) à la position du curseur."""
    # Petit délai pour laisser l'OS enregistrer le clipboard avant de coller
    time.sleep(0.05)
    with _keyboard.pressed(Key.cmd):
        _keyboard.press("v")
        _keyboard.release("v")


def paste_text(text: str, auto_paste: bool = True) -> None:
    """
    Écrit `text` dans le presse-papier ; si auto_paste, colle aussi
    immédiatement à la position du curseur.

    Si `text` est vide ou ne contient que des espaces, ne fait rien
    (évite de coller du vide à la place du texte de l'utilisateur).
    """
    if not text or not text.strip():
        return

    copy_to_clipboard(text)
    if auto_paste:
        simulate_paste()
