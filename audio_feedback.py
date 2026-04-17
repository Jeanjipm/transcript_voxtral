"""
Feedback audio (sons d'activation / désactivation).

Sons système macOS Tink.aiff (start) et Pop.aiff (stop). Pas d'option de
thème custom — les sons système sont gratuits, toujours présents, et
l'ergonomie d'un son custom ne vaut pas la complexité (WAV à shipper,
chemins à résoudre, fallback silencieux obscur).

Backend principal : `AppKit.NSSound` (lecture asynchrone, volume réglable).
Fallback : `subprocess afplay` si pyobjc indisponible (pas de volume).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import Config


START_SOUND = Path("/System/Library/Sounds/Tink.aiff")
STOP_SOUND = Path("/System/Library/Sounds/Pop.aiff")


try:
    from AppKit import NSSound  # type: ignore[import-not-found]

    _HAS_NSSOUND = True
except ImportError:  # pragma: no cover (dispo seulement sur macOS+pyobjc)
    _HAS_NSSOUND = False


class AudioFeedback:
    """Joue les sons Tink/Pop selon la configuration utilisateur.

    Usage :
        fb = AudioFeedback(config)
        fb.play_start()
        fb.play_stop()
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # Cache des NSSound pré-chargées. Sans référence conservée, Python
        # peut collecter l'objet avant que Cocoa ait fini de jouer, et le
        # son est coupé.
        self._sound_cache: dict[str, "NSSound"] = {}

    def play_start(self) -> None:
        if not self.config.sounds.enabled:
            return
        self._play(START_SOUND)

    def play_stop(self) -> None:
        if not self.config.sounds.enabled:
            return
        self._play(STOP_SOUND)

    def _play(self, sound_path: Path) -> None:
        if not sound_path.exists():
            return
        if _HAS_NSSOUND:
            self._play_via_nssound(sound_path)
        else:
            self._play_via_afplay(sound_path)

    def _play_via_nssound(self, sound_path: Path) -> None:
        key = str(sound_path)
        sound = self._sound_cache.get(key)
        if sound is None:
            sound = NSSound.alloc().initWithContentsOfFile_byReference_(key, True)
            if sound is None:
                self._play_via_afplay(sound_path)
                return
            self._sound_cache[key] = sound
        sound.setVolume_(self.config.sounds.volume)
        # stop() avant play() permet de rejouer le son si l'utilisateur
        # enchaîne les dictées sans attendre la fin du précédent.
        sound.stop()
        sound.play()

    def _play_via_afplay(self, sound_path: Path) -> None:
        # afplay ne gère pas le volume — limite acceptée du fallback.
        try:
            subprocess.Popen(
                ["afplay", str(sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass
