"""
Quel est le micro par défaut de macOS, et a-t-il changé ?

## Le problème

`AudioRecorder` garde son flux micro ouvert entre deux dictées (ouvrir un
flux coûte ~4 s à froid, cf. audio_capture.py). Ce flux reste attaché au
périphérique qui était le micro par défaut au moment de son ouverture.
Brancher un casque ou des AirPods *après* le lancement de l'app ne change
donc rien : on continue d'enregistrer avec le micro intégré — ou le flux
devient invalide et la dictée suivante paye une reconstruction.

## La solution retenue : sonder, pas écouter

On lit la propriété CoreAudio « périphérique d'entrée par défaut » toutes
les deux secondes (depuis un `rumps.Timer`, même rythme que le rechargement
de la configuration) et on ne reconstruit le flux **que** quand l'identifiant
change. La lecture coûte quelques microsecondes.

Pourquoi pas un écouteur CoreAudio (`AudioObjectAddPropertyListener`) ?
Il faudrait soit une nouvelle dépendance (`pyobjc-framework-CoreAudio`,
absente de l'installation), soit un callback C via `ctypes` appelé depuis
un thread CoreAudio — une classe de bugs (callback ramassé par le garbage
collector, plantage hors de l'interpréteur) impossible à diagnostiquer à
distance sur le Mac d'un ami. Lire une propriété via `ctypes` n'a aucun de
ces risques, et un délai de deux secondes est imperceptible.

Pourquoi pas reconstruire le flux à chaque dictée ? Ça coûterait ~4 s à
chaque appui et annulerait l'optimisation « flux gardé chaud ».

`DefaultInputWatcher` ne contient que la logique « a-t-on changé ? » et
reçoit la fonction de lecture en paramètre : c'est ce qui le rend testable
sans Mac.
"""

from __future__ import annotations

import ctypes
import sys
import traceback
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DeviceInfo:
    """Un périphérique audio : son identifiant CoreAudio et son nom."""

    device_id: int
    name: str


# ---- Constantes CoreAudio (AudioHardware.h) ----


def _fourcc(code: str) -> int:
    """Les sélecteurs CoreAudio sont des codes à quatre caractères."""
    return int.from_bytes(code.encode("ascii"), "big")


_SYSTEM_OBJECT = 1  # kAudioObjectSystemObject
_PROP_DEFAULT_INPUT = _fourcc("dIn ")  # kAudioHardwarePropertyDefaultInputDevice
_PROP_NAME = _fourcc("lnam")  # kAudioObjectPropertyName
_SCOPE_GLOBAL = _fourcc("glob")  # kAudioObjectPropertyScopeGlobal
_ELEMENT_MAIN = 0  # kAudioObjectPropertyElementMain
_CF_UTF8 = 0x08000100  # kCFStringEncodingUTF8
_UNKNOWN_DEVICE = 0  # kAudioObjectUnknown : aucun périphérique d'entrée


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    ]


_core_audio: ctypes.CDLL | None = None
_core_foundation: ctypes.CDLL | None = None


def _frameworks() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Charge CoreAudio et CoreFoundation une seule fois (frameworks système,
    présents sur tout Mac : aucune dépendance à installer)."""
    global _core_audio, _core_foundation
    if _core_audio is None or _core_foundation is None:
        _core_audio = ctypes.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        _core_foundation = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        _core_audio.AudioObjectGetPropertyData.restype = ctypes.c_int32
        _core_audio.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        _core_foundation.CFStringGetCString.restype = ctypes.c_bool
        _core_foundation.CFStringGetCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
        ]
        _core_foundation.CFRelease.restype = None
        _core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
    return _core_audio, _core_foundation


def _device_name(device_id: int) -> str:
    """Nom lisible du périphérique, pour le journal. Best-effort."""
    core_audio, core_foundation = _frameworks()
    address = _PropertyAddress(_PROP_NAME, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    cf_string = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(cf_string))
    status = core_audio.AudioObjectGetPropertyData(
        device_id, ctypes.byref(address), 0, None,
        ctypes.byref(size), ctypes.byref(cf_string),
    )
    if status != 0 or not cf_string.value:
        return f"périphérique {device_id}"
    try:
        buffer = ctypes.create_string_buffer(256)
        if core_foundation.CFStringGetCString(cf_string, buffer, 256, _CF_UTF8):
            return buffer.value.decode("utf-8", errors="replace")
        return f"périphérique {device_id}"
    finally:
        core_foundation.CFRelease(cf_string)


def default_input_device() -> DeviceInfo | None:
    """Le micro que macOS utilise par défaut en ce moment.

    `None` si la lecture échoue ou qu'aucun périphérique d'entrée n'existe.
    Ne lève jamais : cette fonction tourne dans un timer de la menu bar, et
    une exception y tuerait la surveillance jusqu'au redémarrage.
    """
    try:
        core_audio, _ = _frameworks()
        address = _PropertyAddress(
            _PROP_DEFAULT_INPUT, _SCOPE_GLOBAL, _ELEMENT_MAIN
        )
        device_id = ctypes.c_uint32(_UNKNOWN_DEVICE)
        size = ctypes.c_uint32(ctypes.sizeof(device_id))
        status = core_audio.AudioObjectGetPropertyData(
            _SYSTEM_OBJECT, ctypes.byref(address), 0, None,
            ctypes.byref(size), ctypes.byref(device_id),
        )
        if status != 0 or device_id.value == _UNKNOWN_DEVICE:
            return None
        return DeviceInfo(device_id.value, _device_name(device_id.value))
    except Exception:  # noqa: BLE001 — surveillance best-effort
        traceback.print_exc()
        return None


class DefaultInputWatcher:
    """Détecte les changements de micro par défaut entre deux sondages.

    - Le premier sondage mémorise le périphérique courant sans rien
      déclencher : au lancement, le flux micro n'est pas encore ouvert et il
      n'y a rien à reconstruire.
    - Une lecture qui échoue (`None`) est ignorée : un échec ponctuel n'est
      pas un changement, et « aucun micro » redeviendra « un micro » au
      prochain sondage réussi.
    - `on_change(ancien, nouveau)` ne doit pas pouvoir casser le timer qui
      nous appelle : ses exceptions sont journalisées et avalées.
    """

    def __init__(
        self,
        on_change: Callable[[DeviceInfo, DeviceInfo], None],
        read: Callable[[], DeviceInfo | None] = default_input_device,
    ) -> None:
        self._on_change = on_change
        self._read = read
        self._current: DeviceInfo | None = None

    @property
    def current(self) -> DeviceInfo | None:
        """Dernier périphérique vu (None avant le premier sondage réussi)."""
        return self._current

    def poll(self) -> bool:
        """Sonde une fois. Retourne True si un changement a été signalé."""
        try:
            info = self._read()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return False
        if info is None:
            return False

        previous = self._current
        if previous is None:
            self._current = info
            print(
                f"[audio_device] micro par défaut : {info.name}", file=sys.stderr
            )
            return False
        if info.device_id == previous.device_id:
            return False

        self._current = info
        try:
            self._on_change(previous, info)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return True
