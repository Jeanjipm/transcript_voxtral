"""
Feedback audio (sons d'activation / désactivation).

Pour le thème "system" on utilise les sons macOS natifs Tink.aiff (start)
et Pop.aiff (stop). Pour les autres thèmes, on lit un WAV depuis le dossier
`sounds/` du projet.

Backend principal : `AppKit.NSSound` (lecture asynchrone, intégration native).
Fallback : `subprocess.run(["afplay", path])` si pyobjc indisponible.

Pourquoi ne pas embarquer de WAV en v0 ? Les sons système macOS sont
parfaits, gratuits, déjà sur la machine, et reconnaissables par l'utilisateur.
On évite +1 Mo dans le repo et un choix esthétique prématuré.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import Config


# Sons système macOS (présents sur tout Mac)
SYSTEM_SOUND_DIR = Path("/System/Library/Sounds")
SYSTEM_START_SOUND = SYSTEM_SOUND_DIR / "Tink.aiff"
SYSTEM_STOP_SOUND = SYSTEM_SOUND_DIR / "Pop.aiff"


# Tentative d'import pyobjc — fallback transparent si absent
try:
    from AppKit import NSSound  # type: ignore[import-not-found]

    _HAS_NSSOUND = True
except ImportError:  # pragma: no cover (dispo seulement sur macOS+pyobjc)
    _HAS_NSSOUND = False


class AudioFeedback:
    """
    Joue les sons de feedback selon la configuration utilisateur.

    Usage :
        fb = AudioFeedback(config)
        fb.play_start()
        fb.play_stop()
    """

    def __init__(self, config: Config, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parent

    # ---- API publique ----

    def play_start(self) -> None:
        if not self.config.sounds.enabled:
            return
        self._play(self._resolve_sound(start=True))

    def play_stop(self) -> None:
        if not self.config.sounds.enabled:
            return
        self._play(self._resolve_sound(start=False))

    # ---- Résolution du fichier son ----

    def _resolve_sound(self, *, start: bool) -> Path:
        """Retourne le chemin du son à jouer selon le thème configuré."""
        if self.config.sounds.theme == "system":
            return SYSTEM_START_SOUND if start else SYSTEM_STOP_SOUND

        # Thème custom : chemin relatif au projet
        rel = (
            self.config.sounds.start_sound
            if start
            else self.config.sounds.stop_sound
        )
        return self.project_root / rel

    # ---- Lecture ----

    def _play(self, sound_path: Path) -> None:
        if not sound_path.exists():
            # Silencieux : on ne casse pas l'app pour un son manquant
            return

        if _HAS_NSSOUND:
            self._play_via_nssound(sound_path)
        else:
            self._play_via_afplay(sound_path)

    def _play_via_nssound(self, sound_path: Path) -> None:
        sound = NSSound.alloc().initWithContentsOfFile_byReference_(
            str(sound_path), True
        )
        if sound is None:
            self._play_via_afplay(sound_path)
            return
        sound.setVolume_(self.config.sounds.volume)
        sound.play()

    def _play_via_afplay(self, sound_path: Path) -> None:
        # afplay ne gère pas le volume directement ; on accepte cette limite
        # du fallback (rare en pratique sur macOS).
        try:
            subprocess.Popen(
                ["afplay", str(sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # Pas sur macOS (test depuis Linux/Windows) : silencieux
            pass
