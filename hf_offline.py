"""
Mode hors-ligne HuggingFace.

Problème résolu : charger un modèle déjà présent en cache local déclenche
quand même une requête HEAD vers huggingface.co (transformers →
`hf_hub_download` → `get_hf_file_metadata`) pour vérifier s'il a changé en
amont. Sans réseau — Wi-Fi coupé, ou Mac qui vient de sortir de veille et
dont le DNS n'est pas encore prêt — ça donne `[Errno 8] nodename nor
servname provided`, puis 5 tentatives avec attente progressive, puis
`RuntimeError: Cannot send a request, as the client has been closed`.
Le préchargement du modèle échouait donc silencieusement, et la première
dictée repayait tout le coût de chargement.

C'était aussi une violation de la règle du projet « pas de dépendance
réseau à l'exécution » : le seul moment où le réseau est légitime, c'est le
téléchargement initial du modèle.

Solution : quand le modèle est déjà en cache, on force le mode hors-ligne
de huggingface_hub. Plus aucune requête, chargement immédiat. Le réseau
reste disponible à la demande via `allow_network()` pour les
téléchargements explicites.

Pourquoi poser DEUX choses (variable d'environnement ET constante) :
`huggingface_hub.constants.HF_HUB_OFFLINE` est figée à l'import du module,
donc modifier `os.environ` après coup n'a aucun effet. Mais les appelants
passent par `constants.is_offline_mode()`, qui relit la constante à chaque
appel — l'assigner directement fonctionne donc, et reste révocable. On pose
quand même la variable d'environnement pour les sous-processus et pour les
bibliothèques qui la lisent en direct.

Et pourquoi le mode hors-ligne NE SUFFIT PAS, d'où `resolve_local_path` :
le tokenizer Voxtral est au format Mistral (`tekken.json`), donc
`transformers` délègue à `mistral_common`, dont
`download_tokenizer_from_hf_hub` appelle `list_repo_files()` **sans aucune
condition** et sans repli sur le cache. En mode hors-ligne ça lève
`OfflineModeIsEnabled` ; sans réseau ça lève une erreur DNS. Dans les deux
cas le chargement échoue alors que tous les fichiers sont sur le disque.

La seule parade fiable est de ne jamais passer un identifiant de repo à
`from_pretrained` quand le modèle est en cache, mais le **chemin du
snapshot local** : `transformers` reconnaît un dossier et court-circuite
entièrement le hub. Mesuré : processor en 0,39 s et modèle en 0,07 s avec
un endpoint HuggingFace volontairement injoignable.
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Iterator


ENV_VAR = "HF_HUB_OFFLINE"

# Sentinelle présente dans à peu près tous les repos MLX / transformers :
# si elle est en cache, le repo est utilisable hors-ligne.
_CACHE_SENTINEL = "config.json"


def is_model_cached(repo_id: str) -> bool:
    """True si `repo_id` est présent dans le cache HuggingFace local.

    Best-effort : si `huggingface_hub` est absent ou trop ancien pour
    exposer l'API de cache, on répond False — donc on n'active pas le mode
    hors-ligne, donc on ne risque pas de bloquer un téléchargement légitime.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    try:
        result = try_to_load_from_cache(
            repo_id=repo_id, filename=_CACHE_SENTINEL
        )
    except Exception:  # noqa: BLE001 — API instable entre versions de hf_hub
        return False

    # try_to_load_from_cache retourne le chemin (str) si le fichier est là,
    # un objet sentinelle _CACHED_NO_EXIST si le fichier est connu absent,
    # ou None si le repo n'est pas en cache. Seul le chemin nous intéresse.
    return isinstance(result, (str, bytes))


def resolve_local_path(repo_id: str) -> str:
    """Traduit un identifiant de repo en chemin de snapshot local, si possible.

    Retourne le dossier du snapshot en cache (ex.
    `~/.cache/huggingface/hub/models--org--nom/snapshots/<sha>/`) quand le
    modèle est déjà téléchargé, sinon `repo_id` inchangé — auquel cas
    l'appelant fera un vrai téléchargement, ce qui est le comportement voulu.

    C'est LE contournement du bug `mistral_common` décrit dans la docstring
    du module : passer un dossier à `from_pretrained` évite tout appel au
    hub, là où passer un identifiant de repo en déclenche un même en mode
    hors-ligne.

    Best-effort et sans effet de bord : toute anomalie (hf_hub absent, cache
    corrompu, `repo_id` qui est déjà un chemin) retourne `repo_id` tel quel.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return repo_id

    try:
        cached = try_to_load_from_cache(
            repo_id=repo_id, filename=_CACHE_SENTINEL
        )
    except Exception:  # noqa: BLE001 — API instable entre versions de hf_hub
        return repo_id

    if not isinstance(cached, (str, bytes)):
        return repo_id

    snapshot_dir = Path(os.fsdecode(cached)).parent
    if not snapshot_dir.is_dir():
        return repo_id
    return str(snapshot_dir)


def set_offline(enabled: bool) -> None:
    """Active ou désactive le mode hors-ligne de huggingface_hub."""
    os.environ[ENV_VAR] = "1" if enabled else "0"
    try:
        from huggingface_hub import constants

        constants.HF_HUB_OFFLINE = enabled
    except ImportError:
        pass


def is_offline() -> bool:
    """État courant du mode hors-ligne, tel que hf_hub le voit."""
    try:
        from huggingface_hub import constants

        return bool(constants.HF_HUB_OFFLINE)
    except ImportError:
        return os.environ.get(ENV_VAR, "0") == "1"


def refresh(repo_id: str, prefer_offline: bool = True) -> bool:
    """(Ré)évalue le mode hors-ligne pour le modèle courant.

    À appeler au démarrage et après tout changement de modèle : si
    l'utilisateur choisit dans les Préférences un modèle pas encore
    téléchargé, il faut re-autoriser le réseau.

    Retourne l'état appliqué (True = hors-ligne).
    """
    if not prefer_offline:
        set_offline(False)
        return False

    cached = is_model_cached(repo_id)
    set_offline(cached)
    if not cached:
        print(
            f"[hf_offline] '{repo_id}' absent du cache : réseau autorisé "
            f"pour le téléchargement.",
            file=sys.stderr,
        )
    return cached


@contextlib.contextmanager
def allow_network() -> Iterator[None]:
    """Autorise temporairement le réseau, pour un téléchargement explicite.

    Restaure l'état précédent même si le bloc lève — sans ça, un
    téléchargement échoué laisserait l'app en mode connecté et on
    retomberait sur les erreurs DNS au prochain chargement de modèle.
    """
    previous = is_offline()
    set_offline(False)
    try:
        yield
    finally:
        set_offline(previous)


def install_early() -> None:
    """Pose le mode hors-ligne AVANT le premier import de huggingface_hub.

    À appeler tout en haut d'`app.py`, avant les imports qui tirent
    `huggingface_hub` (`model_manager`, `transcriber`). À ce stade on ne
    connaît pas encore le modèle configuré — on part donc du principe
    « hors-ligne », que `refresh()` corrigera une fois la config chargée.

    `setdefault` respecte une variable posée par l'utilisateur dans son
    environnement : s'il a explicitement mis HF_HUB_OFFLINE=0, on n'écrase
    pas son choix.
    """
    try:
        os.environ.setdefault(ENV_VAR, "1")
    except Exception:  # noqa: BLE001 — ne doit jamais empêcher le démarrage
        traceback.print_exc()
