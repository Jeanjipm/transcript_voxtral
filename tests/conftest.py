"""
Configuration pytest globale + mocks des modules natifs macOS.

Voxtral dépend de plusieurs paquets uniquement disponibles sur macOS
Apple Silicon (AppKit/pyobjc, mlx_voxtral, sounddevice avec device audio
réel, pynput avec accès TCC). Pour exécuter les tests sur n'importe
quelle machine — y compris en CI Linux — on remplace ces imports par
des MagicMock dans `sys.modules` AVANT que le code de l'app ne les
importe.

Les tests unitaires couvrent la logique métier pure (config, parsing,
factory, état) qui ne nécessite pas le hardware. Les comportements qui
dépendent vraiment du système (lecture micro, paste Cmd+V, mutations
NSStatusItem) sont testés manuellement via le protocole de sprint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


# Permet `from config import Config` etc. depuis les tests sans avoir
# à installer le projet en mode editable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- Mocks des modules natifs ----
# Ils doivent être en place AVANT le premier import du code testé.
# Une fois enregistrés dans sys.modules, tout `import X` les retrouve.

_NATIVE_MODULES = (
    # macOS / pyobjc
    "AppKit",
    "Foundation",
    "PyObjCTools",
    "PyObjCTools.AppHelper",
    # rumps (menu bar) — dépend d'AppKit
    "rumps",
    # Audio temps-réel (devices CoreAudio)
    "sounddevice",
    # Hotkey global (CGEventTap, demande permission Accessibility)
    "pynput",
    "pynput.keyboard",
    # MLX (charge le modèle ~3 Go en RAM, lourd même sans transcribe)
    "mlx_voxtral",
    "mlx_whisper",
    "mlx",
    # huggingface_hub (try_to_load_from_cache n'est pas dispo offline)
    "huggingface_hub",
)

for _name in _NATIVE_MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()


# Pour les modules composés (X.Y), `from X import Y` fait `X.Y` (attribute
# access). Sur un MagicMock par défaut, X.Y retourne un *nouveau* mock à
# chaque accès, pas celui posé dans sys.modules["X.Y"]. On les lie
# explicitement pour que `from pynput import keyboard` retourne bien notre
# stub configuré.
def _link_submodule(parent: str, child: str) -> None:
    parent_mod = sys.modules[parent]
    child_mod = sys.modules[f"{parent}.{child}"]
    setattr(parent_mod, child, child_mod)


_link_submodule("pynput", "keyboard")
_link_submodule("PyObjCTools", "AppHelper")


# Quelques utilitaires que le code attend dans les modules mockés —
# on remplace les MagicMock par des stubs un peu plus précis.

import sounddevice as _sd  # noqa: E402  # mock posé ci-dessus
import numpy as _np  # noqa: E402


class _PortAudioErrorStub(Exception):
    """Stub minimaliste de sd.PortAudioError, qu'on doit pouvoir
    instancier ET catch dans le code prod."""


_sd.PortAudioError = _PortAudioErrorStub
_sd.CallbackFlags = MagicMock()


# pynput.keyboard avec les attributs que hotkey_manager.py importe.
# `Key` doit être une vraie classe (pas un MagicMock), pour que
# `isinstance(key, keyboard.Key)` dans hotkey_manager._normalize() puisse
# fonctionner. Idem pour KeyCode.

import pynput.keyboard as _kbd  # noqa: E402


class _Key:
    """Stub de pynput.keyboard.Key — chaque attribut accédé crée une
    instance avec le nom correspondant et la met en cache pour qu'une
    2e lecture renvoie la MÊME instance (égalité par identité, comme
    une enum)."""

    _cache: dict[str, "_Key"] = {}

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Key.{self.name}"


class _KeyEnum:
    """Imite l'énum pynput.keyboard.Key. `Key.alt_r` retourne une instance
    de _Key, et 2 appels successifs retournent la même (cf. cache)."""

    def __getattr__(self, name: str) -> _Key:
        if name not in _Key._cache:
            _Key._cache[name] = _Key(name)
        return _Key._cache[name]


# On expose `Key` comme la *classe* _Key, pour que isinstance(x, Key) marche
# dans hotkey_manager. Les "instances de l'enum" (Key.alt_r, etc.) sont
# créées via le _KeyEnum singleton — qui est ce qu'attend pynput.
_kbd.Key = _Key
# Backdoor pour les tests : hotkey_manager fait `keyboard.Key.alt_r`.
# Comme _Key est une classe, on doit donc lui ajouter des attributs de
# classe représentant les membres de l'enum. On utilise __class_getitem__
# style via une métaclasse simple.

_KEY_NAMES = (
    "alt_l", "alt_r", "alt", "cmd_l", "cmd_r", "cmd", "ctrl_l", "ctrl_r",
    "ctrl", "shift_l", "shift_r", "shift", "space", "enter", "tab", "esc",
    "f13", "f14", "f15", "f16", "f17", "f18", "f19",
)
for _name in _KEY_NAMES:
    setattr(_kbd.Key, _name, _Key(_name))


class _KeyCode:
    """Stub minimaliste de pynput.keyboard.KeyCode."""

    def __init__(self, char: str | None = None) -> None:
        self.char = char


_kbd.KeyCode = _KeyCode
_kbd.Listener = MagicMock()
_kbd.Controller = MagicMock


# Numpy est utilisé pour de vrai (pas mocké) — on s'assure juste qu'il
# est importable. Si l'environnement de test n'a pas numpy, ça lèvera
# ImportError ici, ce qui est le comportement attendu.
_ = _np.zeros(1)
