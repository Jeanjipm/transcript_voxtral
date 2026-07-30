"""
Interactions Finder natives : choix d'un fichier audio, révélation du résultat.

MAIN THREAD UNIQUEMENT. `NSOpenPanel.runModal()` est une API AppKit ; l'appeler
depuis un thread secondaire est un comportement indéfini. Les fonctions d'ici
sont donc appelées depuis les callbacks de menu de rumps, qui tournent déjà sur
le main thread.

Deux pièges macOS traités ici :

1. L'app tourne en `NSApplicationActivationPolicyAccessory` (menu bar sans
   Dock). Sans `activateIgnoringOtherApps_`, le panneau s'ouvre DERRIÈRE la
   fenêtre active et l'utilisateur ne le voit pas.
2. Pendant qu'une modale est ouverte, les `rumps.Timer` (qui sont des NSTimer
   en mode par défaut) ne tournent plus : l'animation d'icône et le
   rechargement de config se figent le temps du dialogue. C'est sans
   conséquence, mais autant le savoir pour ne pas le diagnostiquer comme un
   blocage.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import traceback
from pathlib import Path

from AppKit import NSApp, NSOpenPanel

from audio_convert import SUPPORTED_EXTENSIONS


def _assert_main_thread(what: str) -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            f"{what} doit être appelé sur le main thread (API AppKit)."
        )


def choose_audio_file(title: str = "Choisir un fichier audio") -> Path | None:
    """Ouvre le sélecteur de fichiers. Retourne None si l'utilisateur annule.

    Main thread uniquement.
    """
    _assert_main_thread("choose_audio_file")

    panel = NSOpenPanel.openPanel()
    panel.setTitle_(title)
    panel.setCanChooseFiles_(True)
    panel.setCanChooseDirectories_(False)
    panel.setAllowsMultipleSelection_(False)
    panel.setAllowedFileTypes_(list(SUPPORTED_EXTENSIONS))

    # Sans ça, le panneau d'une app « accessory » s'ouvre derrière la fenêtre
    # active et passe inaperçu.
    try:
        NSApp.activateIgnoringOtherApps_(True)
    except Exception:  # noqa: BLE001 — jamais bloquant
        traceback.print_exc()

    if panel.runModal() != 1:  # NSModalResponseOK
        return None

    urls = panel.URLs()
    if not urls or len(urls) == 0:
        return None
    return Path(str(urls[0].path()))


def reveal_in_finder(path: Path) -> None:
    """Ouvre le Finder sur le fichier, sélectionné.

    Appelable depuis n'importe quel thread : c'est un sous-processus, pas une
    API AppKit. « Où est passé mon fichier ? » est la question la plus probable
    d'un utilisateur non-développeur, donc on répond avant qu'elle se pose.
    """
    try:
        subprocess.Popen(["open", "-R", str(path)])
    except OSError:
        print(
            f"[file_picker] impossible d'ouvrir le Finder sur {path}",
            file=sys.stderr,
        )
