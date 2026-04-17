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
    from AppKit import NSPasteboard, NSPasteboardTypeString  # type: ignore[import-not-found]

    _HAS_NSPASTEBOARD = True
except ImportError:  # pragma: no cover (macOS+pyobjc only)
    _HAS_NSPASTEBOARD = False


_keyboard = Controller()


def copy_to_clipboard(text: str) -> None:
    """Place le texte dans le presse-papier système."""
    if _HAS_NSPASTEBOARD:
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)
    else:
        # Fallback CLI macOS (toujours présent sur Mac)
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(text.encode("utf-8"))


def _read_clipboard_text() -> str | None:
    """Retourne le contenu texte actuel du presse-papier, ou None si vide /
    si pyobjc est absent (dans ce cas on ne préserve pas)."""
    if not _HAS_NSPASTEBOARD:
        return None
    pb = NSPasteboard.generalPasteboard()
    return pb.stringForType_(NSPasteboardTypeString)


def simulate_paste() -> None:
    """Simule Cmd+V (paste) à la position du curseur."""
    # 200 ms (vs 50 ms initialement) laisse le temps à l'utilisateur de
    # relâcher physiquement son raccourci en mode toggle (ex. cmd+shift+h
    # pour arrêter). Sans ça, macOS interprétait notre Cmd+V injecté comme
    # Cmd+Shift+V parce que shift était encore tenu — paste échouait
    # silencieusement. Imperceptible pour l'utilisateur à l'œil nu.
    time.sleep(0.2)
    with _keyboard.pressed(Key.cmd):
        _keyboard.press("v")
        _keyboard.release("v")


def paste_text(
    text: str,
    auto_paste: bool = True,
    preserve_clipboard: bool = True,
) -> None:
    """
    Écrit `text` dans le presse-papier ; si auto_paste, colle aussi
    immédiatement à la position du curseur.

    Si `preserve_clipboard` (et `auto_paste`), le contenu texte
    précédent du presse-papier est restauré après le paste. Utile pour
    ne pas écraser ce que l'utilisateur avait copié avant la dictée.
    Limite v0 : préservation texte uniquement (images/fichiers perdus).

    Si `text` est vide ou ne contient que des espaces, ne fait rien
    (évite de coller du vide à la place du texte de l'utilisateur).
    """
    if not text or not text.strip():
        return

    # Préfixe un espace : évite que la dictée vienne coller directement
    # contre la ponctuation du texte précédent (ex. "Bonjour.Comment…").
    # lstrip d'abord pour ne pas doubler l'espace si Voxtral en a déjà mis.
    text = " " + text.lstrip()

    # On ne préserve que si on va réellement paster — en mode copy-only
    # l'utilisateur veut que son clipboard reste sur le nouveau texte.
    saved = _read_clipboard_text() if (auto_paste and preserve_clipboard) else None

    copy_to_clipboard(text)

    if auto_paste:
        simulate_paste()
        if saved is not None:
            # Laisse macOS finir de traiter le Cmd+V avant de remplacer
            # le clipboard, sinon on écraserait notre propre paste.
            time.sleep(0.3)
            copy_to_clipboard(saved)
