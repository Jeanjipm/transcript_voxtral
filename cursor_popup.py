"""Popup HUD borderless affichée près du curseur souris, auto-dismiss.

Utilisée pour signaler une erreur non-bloquante à l'utilisateur directement
là où il regarde — alternative au `rumps.notification` qui s'affiche dans
le coin supérieur droit et passe souvent inaperçu en pleine dictée.

Thread-safe : `show_near_cursor` peut être appelée depuis n'importe quel
thread, la création NSWindow est dispatchée sur le main thread via
`AppHelper.callAfter`.
"""

from __future__ import annotations

from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSFont,
    NSTextField,
    NSWindow,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSMakeRect
from PyObjCTools import AppHelper


# NSWindow se libère d'elle-même sur close() par défaut (setReleasedWhenClosed=YES)
# ce qui laisserait des références Python dangling. On maintient les popups
# récentes vivantes dans cette liste et on orderOut: sans jamais close(),
# les fenêtres sont libérées proprement par Python GC quand elles sortent
# de la liste.
_live_windows: list = []
_MAX_LIVE = 3


def show_near_cursor(message: str, duration: float = 2.0) -> None:
    AppHelper.callAfter(_show, message, duration)


def _show(message: str, duration: float) -> None:
    loc = NSEvent.mouseLocation()  # coords écran, origine bas-gauche
    w, h = 340, 40
    x = loc.x + 12
    y = loc.y - h - 12  # popup sous-droite de la souris

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, w, h),
        NSWindowStyleMaskBorderless,
        NSBackingStoreBuffered,
        False,
    )
    # CRITIQUE : sinon close() libère NSWindow et toute référence ultérieure
    # (ex. _live_windows.remove via __eq__) segfault sur la mémoire freed.
    win.setReleasedWhenClosed_(False)
    win.setLevel_(24)  # NSPopUpMenuWindowLevel : au-dessus des apps
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.colorWithCalibratedWhite_alpha_(0.1, 0.92))
    win.setHasShadow_(True)
    win.setIgnoresMouseEvents_(True)

    label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
    label.setStringValue_(message)
    label.setBezeled_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setDrawsBackground_(False)
    label.setTextColor_(NSColor.whiteColor())
    label.setFont_(NSFont.systemFontOfSize_(13))
    label.setAlignment_(2)  # NSTextAlignmentCenter
    win.setContentView_(label)

    win.orderFrontRegardless()
    _live_windows.append(win)

    # Éviction LRU : borne la mémoire si l'utilisateur déclenche beaucoup
    # de popups. Les anciennes sortent de la liste → Python GC les libère.
    while len(_live_windows) > _MAX_LIVE:
        _live_windows.pop(0).orderOut_(None)

    # performSelector:withObject:afterDelay: planifie un appel Objective-C
    # pur sur le main run loop — pas de closure Python à gérer (qui était
    # la source du segfault avec NSTimer+bloc sur les popups successives).
    # orderOut: cache la fenêtre sans la libérer, nickel avec
    # setReleasedWhenClosed_(False) ci-dessus.
    win.performSelector_withObject_afterDelay_("orderOut:", None, duration)
