"""
Entretien des fichiers de ~/.voxtral au démarrage de l'app.

Deux choses grossissaient sans limite (issue 10 du rapport du 26 octobre) :

- `voxtral.log` : 1,7 Mo après trois mois. C'est la sortie d'erreur du
  process lui-même, redirigée par `Voxtral.command` et par le LaunchAgent
  (cf. install.sh). Rien ne le tournait jamais.
- `~/.voxtral/recordings/` : les dictées coupées pour dépassement de durée y
  sont conservées (cf. dictation_controller._keep_recording) et n'en
  sortaient jamais.

Les deux fonctions de ce module sont pures sur des chemins : elles
s'appliquent à n'importe quel fichier ou dossier, ce qui permet de les tester
sur un dossier temporaire.

Pourquoi le journal est TRONQUÉ EN PLACE plutôt que renommé (rotation
classique `voxtral.log` → `voxtral.log.1`) : le fichier est ouvert par le
shell parent avant que Python ne démarre, et c'est ce descripteur que le
process écrit. Renommer le fichier ne changerait rien pour lui : il
continuerait d'écrire dans le fichier renommé, et le nouveau `voxtral.log`
resterait vide. Tronquer le fichier, puis y réécrire la queue qu'on garde,
fonctionne parce que les lanceurs ouvrent le journal en mode append (`>>`,
et launchd fait de même) : chaque écriture repart de la fin courante. Par
précaution, `trim_log` repositionne aussi les descripteurs 1 et 2 du process
en fin de fichier, pour un lancement à la main sans mode append.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from config import LOG_PATH, RECORDINGS_DIR, MaintenanceConfig


# Ce qu'on garde du journal quand on le tronque : assez pour diagnostiquer
# les derniers jours, pas de quoi rejouer tout l'historique.
LOG_KEEP_BYTES = 1_000_000

# Motif des fichiers créés par dictation_controller._keep_recording. On ne
# purge QUE ceux-là : un fichier déposé à la main dans le dossier n'est pas
# à nous, on n'y touche pas.
RECORDING_GLOB = "dictee-*.wav"

SECONDS_PER_DAY = 86_400


def trim_log(path: Path, max_bytes: int, keep_bytes: int = LOG_KEEP_BYTES) -> int:
    """Ramène `path` à ses derniers `keep_bytes` s'il dépasse `max_bytes`.

    Retourne le nombre d'octets retirés (0 si rien n'a été fait). La coupe
    se fait sur un début de ligne, pour ne pas laisser une demi-ligne en
    tête ; une ligne d'en-tête datée dit que le journal a été tronqué.
    `max_bytes <= 0` désactive. Un fichier absent n'est pas une erreur.
    """
    if max_bytes <= 0:
        return 0
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    if size <= max_bytes:
        return 0

    keep = max(0, min(keep_bytes, size))
    with open(path, "r+b") as f:
        f.seek(size - keep)
        tail = f.read()
        # Coupe sur une ligne entière : on jette tout jusqu'au premier \n.
        cut = tail.find(b"\n")
        if 0 <= cut < len(tail) - 1:
            tail = tail[cut + 1:]
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"[housekeeping] journal tronqué le {stamp} : "
            f"{size} octets > {max_bytes} autorisés, "
            f"{len(tail)} octets conservés.\n"
        ).encode("utf-8")
        f.seek(0)
        f.truncate(0)
        f.write(header + tail)
        new_size = f.tell()

    _reposition_std_streams(path)
    return size - new_size


def _reposition_std_streams(path: Path) -> None:
    """Place stdout/stderr en fin de fichier s'ils pointent sur `path`.

    Inutile en mode append (le noyau le fait à chaque écriture), nécessaire
    sinon : la position d'écriture du descripteur serait au-delà de la
    nouvelle fin, et le prochain message créerait un trou de zéros.
    Best-effort : tout échec (descripteur fermé, stderr capturé…) est ignoré.
    """
    try:
        target = os.stat(path)
    except OSError:
        return
    for fd in (1, 2):
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
                os.lseek(fd, 0, os.SEEK_END)
        except OSError:
            continue


def purge_recordings(
    directory: Path, max_age_days: int, now: float | None = None
) -> list[Path]:
    """Supprime les `dictee-*.wav` de `directory` plus vieux que `max_age_days`.

    Retourne la liste des fichiers supprimés. L'âge est celui de la dernière
    modification. `max_age_days <= 0` désactive ; un dossier absent ne
    contient rien à purger. Un fichier impossible à supprimer est ignoré,
    jamais bloquant.
    """
    if max_age_days <= 0 or not directory.is_dir():
        return []
    if now is None:
        now = time.time()
    cutoff = now - max_age_days * SECONDS_PER_DAY

    removed: list[Path] = []
    for wav in sorted(directory.glob(RECORDING_GLOB)):
        try:
            if wav.is_file() and wav.stat().st_mtime < cutoff:
                wav.unlink()
                removed.append(wav)
        except OSError:
            continue
    return removed


def run_startup_housekeeping(
    cfg: MaintenanceConfig,
    log_path: Path = LOG_PATH,
    recordings_dir: Path = RECORDINGS_DIR,
) -> None:
    """Enchaîne les deux entretiens au démarrage. Ne lève jamais.

    Un problème d'entretien ne doit pas empêcher la dictée de démarrer : on
    trace et on continue. Une ligne de résumé est écrite seulement quand
    quelque chose a été fait, pour ne pas remplir le journal qu'on cherche à
    contenir.
    """
    try:
        removed_bytes = trim_log(log_path, cfg.log_max_mb * 1_000_000)
        if removed_bytes:
            print(
                f"[housekeeping] {log_path.name} tronqué, "
                f"{removed_bytes} octets libérés.",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001 — l'entretien ne bloque jamais le démarrage
        traceback.print_exc()

    try:
        removed = purge_recordings(recordings_dir, cfg.recordings_max_age_days)
        if removed:
            print(
                f"[housekeeping] {len(removed)} enregistrement(s) de plus de "
                f"{cfg.recordings_max_age_days} jours supprimé(s) de "
                f"{recordings_dir}.",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
