"""
Seconde voix de lecture : Pocket TTS (Kyutai), à côté de Voxtral TTS.

Retenue après le banc du 18 novembre 2026 (BENCH-VOIX-2026-11.md) : sur le
Mac de l'auteur, le grand modèle français (`french_24l`) rend son premier son
en 0,08 s là où Voxtral TTS en demande 2 ; il a été préféré à l'écoute ; ses
poids sont sous licence CC-BY (usage commercial permis, à la différence de
Voxtral TTS).

Trois différences avec `synthese.py`, qui expliquent ce fichier à part :

- **Il tourne sur le processeur, avec torch, pas sur MLX.** Il n'a donc pas à
  passer par l'inference-worker (réservé aux modèles MLX) et ne dispute pas la
  puce graphique à la dictée.
- **Il rend le son au fil de la génération**, par tranches de 80 ms (`flux`).
  C'est tout son intérêt : `synthese.SortieFlux` les joue à mesure, sans
  attendre la fin de la phrase.
- **Il ne sait pas lire les chiffres** — d'où `oral.py`, appliqué en amont
  par l'app quel que soit le moteur.

La bibliothèque `pocket-tts` est une dépendance récente : une installation
d'avant ce sprint ne l'a pas. Tout ici le supporte (`bibliotheque_presente`),
et l'app dit alors comment l'obtenir au lieu d'échouer.

Une seule voix française existe au catalogue : « estelle ». Le clonage d'une
voix depuis un extrait demande des poids en accès conditionné (compte
Hugging Face) ; on s'en tient aux voix prédéfinies, libres d'accès.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import hf_offline


# Préfixe des codes de voix dans la config et les Préférences :
# « pocket:french_24l ». Sans préfixe, une voix est une voix Voxtral.
PREFIXE = "pocket:"
ENGINE = "pocket"

VOIX = "estelle"
SAMPLE_RATE = 24_000

# Les deux tailles du modèle français : (libellé des Préférences, taille du
# téléchargement en Go). La grande d'abord : c'est la recommandée.
LANGUES: dict[str, tuple[str, float]] = {
    "french_24l": ("Français — Estelle, voix naturelle (Pocket)", 0.7),
    "french": ("Français — Estelle, version légère (Pocket)", 0.2),
}
DEFAULT_LANGUE = "french_24l"

# Pocket TTS cherche ses poids dans le dépôt en accès conditionné, puis se
# rabat sur le dépôt libre (sans clonage de voix). Les voix prédéfinies et le
# découpeur de texte viennent toujours du second.
REPO_LIBRE = "kyutai/pocket-tts-without-voice-cloning"
REPOS = ("kyutai/pocket-tts", REPO_LIBRE)


class PocketError(RuntimeError):
    """Erreur de la voix Pocket, avec un message destiné à l'utilisateur."""


def bibliotheque_presente() -> bool:
    """True si le paquet `pocket-tts` est installé dans le venv de l'app."""
    try:
        return importlib.util.find_spec("pocket_tts") is not None
    except (ImportError, ValueError):
        return False


def code_voix(langue: str) -> str:
    """Code d'une voix Pocket, tel qu'il figure dans la liste des voix."""
    return f"{PREFIXE}{langue}"


def langue_du_code(code: str) -> str | None:
    """« pocket:french_24l » → « french_24l » ; None pour une voix Voxtral."""
    if not code.startswith(PREFIXE):
        return None
    return code[len(PREFIXE):]


def libelles() -> list[tuple[str, str]]:
    """Les voix Pocket pour la liste des Préférences : (code, libellé)."""
    return [(code_voix(langue), label) for langue, (label, _) in LANGUES.items()]


def taille_go(langue: str) -> float:
    return LANGUES.get(langue, ("", 0.7))[1]


def _dossiers_du_cache() -> list[Path]:
    """Dossiers `snapshots/<révision>` des dépôts Pocket présents sur le disque."""
    try:
        from huggingface_hub import constants

        racine = Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 — bibliothèque absente ou API différente
        return []
    dossiers: list[Path] = []
    for repo in REPOS:
        snapshots = racine / ("models--" + repo.replace("/", "--")) / "snapshots"
        if snapshots.is_dir():
            dossiers.extend(p for p in snapshots.iterdir() if p.is_dir())
    return dossiers


def est_en_cache(langue: str, voix: str = VOIX) -> bool:
    """True si tout ce qu'il faut pour lire est déjà sur le disque.

    Trois fichiers : les poids, le découpeur de texte et l'empreinte de la
    voix. On regarde le disque nous-mêmes plutôt que `hf_offline.is_model_cached` :
    ce dépôt-là range ses modèles par langue dans des sous-dossiers, sans le
    `config.json` que l'autre vérification attend.
    """
    attendus = {
        f"languages/{langue}/model.safetensors",
        f"languages/{langue}/tokenizer.json",
        f"languages/{langue}/embeddings/{voix}.safetensors",
    }
    trouves = {
        relatif
        for dossier in _dossiers_du_cache()
        for relatif in attendus
        if (dossier / relatif).exists()  # un lien vers un blob absent compte comme absent
    }
    return trouves == attendus


class PocketSynthetiseur:
    """Un modèle Pocket TTS, chargé à la première lecture.

    Une seule génération à la fois : le modèle n'est pas fait pour plusieurs
    fils d'exécution. L'app le garantit (une lecture à la fois).
    """

    def __init__(self, langue: str = DEFAULT_LANGUE, voix: str = VOIX) -> None:
        self.langue = langue
        self.voix = voix
        self.sample_rate = SAMPLE_RATE
        self._model: Any = None
        self._voice_state: Any = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not bibliotheque_presente():
                raise PocketError(
                    "La voix Pocket n'est pas installée sur ce Mac. Relance "
                    "l'installation de Voxtral Dictée pour l'obtenir."
                )
            # Import retardé : il tire torch (plusieurs secondes), et règle
            # torch sur un seul fil d'exécution pour tout le processus — sans
            # conséquence ici, l'app ne fait rien d'autre avec torch.
            from pocket_tts import TTSModel  # type: ignore[import-not-found]

            if est_en_cache(self.langue, self.voix):
                model = TTSModel.load_model(language=self.langue)
                state = model.get_state_for_audio_prompt(self.voix)
            else:
                # Premier usage : le reste de l'app tourne hors-ligne forcé
                # (cf. hf_offline), il faut rouvrir le réseau le temps du
                # téléchargement.
                with hf_offline.allow_network():
                    model = TTSModel.load_model(language=self.langue)
                    state = model.get_state_for_audio_prompt(self.voix)
            self.sample_rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
            self._voice_state = state
            self._model = model

    def preload(self) -> None:
        self._ensure_loaded()

    def flux(self, texte: str, arret: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Le son de `texte`, par tranches de 80 ms, à mesure qu'il est produit.

        Le modèle se charge à la première tranche demandée. `arret`, une fois
        posé, fait cesser la génération.
        """
        texte = texte.strip()
        if not texte:
            return
        self._ensure_loaded()
        for morceau in self._model.generate_audio_stream(
            self._voice_state, texte, stop=arret
        ):
            yield np.asarray(morceau.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

    def lire(self, texte: str) -> np.ndarray:
        """Tout le son d'un coup (essais, banc) : la concaténation du flux."""
        morceaux = [m for m in self.flux(texte) if m.size]
        if not morceaux:
            raise PocketError("Le modèle n'a produit aucun son.")
        return np.concatenate(morceaux)
