"""
Vérification + application des mises à jour Voxtral depuis GitHub.

Au démarrage de l'app, un thread daemon interroge l'API GitHub pour
détecter une nouvelle version sur la branche main du repo officiel.
Si MAJ disponible, on prévient l'utilisateur (label menu modifié, parce
que macOS bloque les rumps.notification pour les apps non-signées).

L'utilisateur peut aussi forcer un check via le menu (bouton "Vérifier
les mises à jour…") qui montre une alerte modale "À jour" ou "MAJ dispo,
appliquer ?".

Comportement offline : tous les appels HTTP sont enveloppés dans des
try/except larges → si pas de réseau, on retourne None silencieusement,
l'app continue à fonctionner. Aucune notif d'erreur.

Sécurité : on n'interroge que api.github.com en HTTPS, et on git pull
depuis un remote qu'on contrôle (Jeanjipm/transcript_voxtral). Pas de
risque de supply chain.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_OWNER = "Jeanjipm"
REPO_NAME = "transcript_voxtral"
DEFAULT_BRANCH = "main"
USER_AGENT = "voxtral-app-updater/1.0"
HTTP_TIMEOUT = 5  # secondes — court pour ne pas bloquer si réseau lent

# Emplacement où l'app est installée (cf. install.sh).
APP_DIR = Path.home() / ".voxtral" / "app"

# Fichiers dont la modification nécessite une intervention manuelle :
# un simple `git pull` ne suffit pas, il faut re-run install.sh ou
# pip install. On les détecte pour avertir l'utilisateur.
RISKY_FILES = ("install.sh", "requirements.txt")


@dataclass
class UpdateInfo:
    """Représente une mise à jour disponible.

    `commits_behind` : nb de commits que la copie locale a de retard.
    `head_message` : message du commit le plus récent (1re ligne, court).
    `risky_files` : fichiers risqués modifiés (install.sh, requirements.txt).
    """

    local_sha: str
    remote_sha: str
    commits_behind: int
    head_message: str
    risky_files: list[str] = field(default_factory=list)

    @property
    def requires_manual_action(self) -> bool:
        return len(self.risky_files) > 0


@dataclass
class ApplyResult:
    """Résultat d'une tentative de git pull."""

    success: bool
    message: str
    requires_restart: bool = False


# ----------------------------------------------------------------------
# Helpers internes
# ----------------------------------------------------------------------


def _run_git(*args: str, cwd: Path = APP_DIR) -> str:
    """Exécute git et retourne stdout (strip)."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _http_get_json(url: str) -> Any | None:
    """GET HTTPS qui retourne un JSON parsé, ou None si erreur.

    Aucune exception ne remonte au caller : silent failure pour le mode
    offline et pour tolérer toute incident GitHub (rate limit, 5xx…).
    """
    try:
        req = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
        )
        with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.load(resp)
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None


# ----------------------------------------------------------------------
# API publique
# ----------------------------------------------------------------------


def get_local_sha() -> str | None:
    """SHA du HEAD local. None si pas dans un repo git ou erreur."""
    try:
        return _run_git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def check_for_update() -> UpdateInfo | None:
    """Vérifie si une MAJ est disponible sur le remote.

    Retourne UpdateInfo si MAJ dispo, None si à jour OU si erreur (réseau,
    repo absent, etc.). Le caller ne distingue pas "à jour" vs "erreur" —
    dans les 2 cas, rien à proposer à l'utilisateur.

    Utilise l'API GitHub /compare/{base}...{head} qui renvoie d'un coup :
    - le nb de commits d'écart (ahead_by)
    - la liste des fichiers modifiés (avec filename, status)
    - la liste des commits avec leur message
    """
    local_sha = get_local_sha()
    if local_sha is None:
        return None

    url = (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        f"/compare/{local_sha}...{DEFAULT_BRANCH}"
    )
    data = _http_get_json(url)
    if data is None:
        return None

    # ahead_by = nb de commits que `head` (=main) a en plus de `base` (=local)
    ahead = data.get("ahead_by", 0)
    if ahead == 0:
        return None  # à jour

    files = data.get("files") or []
    risky = [f["filename"] for f in files if f.get("filename") in RISKY_FILES]

    # `commits` est ordonné du plus ancien au plus récent.
    commits = data.get("commits") or []
    head_message = ""
    remote_sha = local_sha
    if commits:
        last = commits[-1]
        # 1re ligne du message, tronquée pour l'UI.
        head_message = (
            last.get("commit", {}).get("message", "").split("\n")[0][:120]
        )
        remote_sha = last.get("sha", local_sha)

    return UpdateInfo(
        local_sha=local_sha,
        remote_sha=remote_sha,
        commits_behind=ahead,
        head_message=head_message,
        risky_files=risky,
    )


def apply_update() -> ApplyResult:
    """Applique la MAJ via `git pull --ff-only` dans APP_DIR.

    --ff-only évite de créer un merge commit accidentel si le user a
    introduit un commit local par erreur (~/.voxtral/app n'a pas vocation
    à recevoir des modifs locales).

    Si le pull réussit, requires_restart=True : le code Python déjà en
    RAM ne reflète pas le nouveau code on-disk. L'app doit être relancée.
    """
    try:
        _run_git("fetch", "origin", DEFAULT_BRANCH)
        _run_git("pull", "--ff-only", "origin", DEFAULT_BRANCH)
    except subprocess.CalledProcessError as exc:
        # stderr de git contient en général la raison ("Not possible to
        # fast-forward", "Working tree has changes", etc.).
        msg = (exc.stderr or exc.stdout or "").strip() or "git pull a échoué."
        return ApplyResult(success=False, message=msg)
    except FileNotFoundError:
        return ApplyResult(
            success=False,
            message="git introuvable — vérifie que les Xcode CLT sont installés.",
        )

    return ApplyResult(
        success=True,
        message=(
            "Mise à jour appliquée. Redémarre Voxtral pour charger la "
            "nouvelle version."
        ),
        requires_restart=True,
    )
