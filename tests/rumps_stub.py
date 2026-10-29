"""
Faux module `rumps` pour les tests d'app.py.

Un `MagicMock` ne suffit pas ici : `class VoxtralApp(rumps.App)` échoue si
`rumps.App` est un mock (on ne peut pas hériter d'une instance de MagicMock —
Python prend son type comme métaclasse et le corps de la classe est perdu).
On fournit donc de vraies classes minimales, qui ne font rien de visuel :

- `App` : conserve le nom, le titre et le menu, et refuse `run()` (jamais
  appelé en test).
- `MenuItem` : un titre et un callback, exactement ce qu'app.py lit et
  écrit. Pas d'indexation par titre comme le vrai rumps.
- `Timer` : inerte. `start()` / `stop()` ne font que noter l'état ; le test
  déclenche `callback` lui-même quand il veut simuler un tick.
- `alert`, `quit_application` : des MagicMock, pour vérifier les appels.

Installé dans `sys.modules["rumps"]` par conftest.py AVANT tout import
d'app.py.
"""

from __future__ import annotations

import types
from typing import Any, Callable
from unittest.mock import MagicMock


class MenuItem:
    def __init__(
        self, title: str, callback: Callable[..., Any] | None = None
    ) -> None:
        self.title = title
        self.callback = callback

    def set_callback(self, callback: Callable[..., Any] | None) -> None:
        self.callback = callback

    def __repr__(self) -> str:
        return f"MenuItem({self.title!r})"


class Timer:
    def __init__(self, callback: Callable[..., Any], interval: float) -> None:
        self.callback = callback
        self.interval = interval
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def fire(self) -> None:
        """Simule un tick (le vrai rumps passe le timer en argument)."""
        self.callback(self)


class App:
    def __init__(
        self,
        name: str,
        title: str | None = None,
        icon: str | None = None,
        template: bool | None = None,
        menu: list[Any] | None = None,
        quit_button: str | None = "Quit",
    ) -> None:
        self.name = name
        self.title = title
        self.icon = icon
        self.quit_button = quit_button
        self.menu: list[Any] = list(menu or [])

    def run(self) -> None:
        raise RuntimeError("rumps_stub.App.run() ne doit pas être appelé en test")


def make_module() -> types.ModuleType:
    """Construit le module `rumps` factice."""
    module = types.ModuleType("rumps")
    module.App = App  # type: ignore[attr-defined]
    module.MenuItem = MenuItem  # type: ignore[attr-defined]
    module.Timer = Timer  # type: ignore[attr-defined]
    module.alert = MagicMock(name="rumps.alert", return_value=1)  # type: ignore[attr-defined]
    module.quit_application = MagicMock(name="rumps.quit_application")  # type: ignore[attr-defined]
    module.notification = MagicMock(name="rumps.notification")  # type: ignore[attr-defined]
    return module
