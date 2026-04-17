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
from Foundation import NSMakeRect, NSTimer
from PyObjCTools import AppHelper


# Cocoa détient des références faibles aux NSWindow créées dynamiquement —
# sans cette liste d'ancrage, la fenêtre serait collectée avant d'être vue.
_live_windows: list = []


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

    def _dismiss(_timer) -> None:
        win.close()
        try:
            _live_windows.remove(win)
        except ValueError:
            pass

    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        duration, False, _dismiss
    )
