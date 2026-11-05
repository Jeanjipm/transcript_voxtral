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
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterator


ENV_VAR = "HF_HUB_OFFLINE"

# Point d'entrée dans le cache : présent dans à peu près tous les repos MLX /
# transformers, et c'est le premier fichier téléchargé. Sa présence prouve
# que le repo est CONNU du cache — pas qu'il est complet, d'où
# `_complete_snapshot`.
_CACHE_SENTINEL = "config.json"
# Carte des fichiers de poids quand ils sont découpés en plusieurs morceaux.
_WEIGHTS_INDEX = "model.safetensors.index.json"
# Extensions de poids que nos modèles utilisent (Voxtral, Whisper MLX,
# Sortformer : safetensors ; anciens Whisper MLX : npz).
_WEIGHT_SUFFIXES = (".safetensors", ".npz")
# Suffixe que huggingface_hub donne à un blob en cours de téléchargement.
_INCOMPLETE_SUFFIX = ".incomplete"


def _complete_snapshot(repo_id: str) -> Path | None:
    """Dossier du snapshot local de `repo_id` s'il est COMPLET, sinon None.

    Constaté en vrai : un téléchargement interrompu (l'app tuée pendant le
    préchargement d'un modèle qu'on venait de choisir) laisse `config.json`
    et un blob `.incomplete` de 200 Mo. Se fier à `config.json` seul faisait
    alors tout dérailler : mode hors-ligne activé, chargement sur « No weight
    files found », et Préférences qui affichait « ✓ sur ton Mac ».

    Trois vérifications, toutes locales et sans réseau :
    1. aucun blob `.incomplete` dans le repo ;
    2. si `model.safetensors.index.json` est là, chaque morceau qu'il nomme
       existe ;
    3. sinon, au moins un fichier de poids est présent.

    Best-effort : hf_hub absent, API qui lève, dossier incohérent, index
    illisible → None. Répondre « incomplet » est le côté sûr : le réseau
    reste autorisé et `snapshot_download` reprend ou vérifie, là où un faux
    « complet » bloque tout.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    try:
        cached = try_to_load_from_cache(
            repo_id=repo_id, filename=_CACHE_SENTINEL
        )
    except Exception:  # noqa: BLE001 — API instable entre versions de hf_hub
        return None

    # try_to_load_from_cache retourne le chemin (str) si le fichier est là,
    # un objet sentinelle _CACHED_NO_EXIST si le fichier est connu absent,
    # ou None si le repo n'est pas en cache. Seul le chemin nous intéresse.
    if not isinstance(cached, (str, bytes)):
        return None

    snapshot = Path(os.fsdecode(cached)).parent
    if not snapshot.is_dir():
        return None

    # 1) models--org--nom/blobs/ est le frère de snapshots/.
    blobs = snapshot.parent.parent / "blobs"
    if blobs.is_dir() and any(blobs.glob(f"*{_INCOMPLETE_SUFFIX}")):
        return None

    # 2) Poids découpés : chaque morceau annoncé doit exister (un lien
    # symbolique vers un blob absent compte comme absent).
    index = snapshot / _WEIGHTS_INDEX
    if index.is_file():
        try:
            with open(index, "r", encoding="utf-8") as f:
                shards = set(json.load(f)["weight_map"].values())
        except (OSError, ValueError, KeyError, AttributeError):
            return None
        if not shards or not all((snapshot / s).exists() for s in shards):
            return None
        return snapshot

    # 3) Poids en un seul fichier.
    for entry in snapshot.iterdir():
        if entry.name.endswith(_WEIGHT_SUFFIXES) and entry.exists():
            return snapshot
    return None


def is_model_cached(repo_id: str) -> bool:
    """True si `repo_id` est présent ET complet dans le cache HuggingFace.

    Best-effort : si `huggingface_hub` est absent ou trop ancien pour
    exposer l'API de cache, on répond False — donc on n'active pas le mode
    hors-ligne, donc on ne risque pas de bloquer un téléchargement légitime.
    Cf. `_complete_snapshot` pour ce que « complet » veut dire.
    """
    return _complete_snapshot(repo_id) is not None


def resolve_local_path(repo_id: str) -> str:
    """Traduit un identifiant de repo en chemin de snapshot local, si possible.

    Retourne le dossier du snapshot en cache (ex.
    `~/.cache/huggingface/hub/models--org--nom/snapshots/<sha>/`) quand le
    modèle est déjà téléchargé EN ENTIER, sinon `repo_id` inchangé — auquel
    cas l'appelant fera un vrai téléchargement (ou le reprendra), ce qui est
    le comportement voulu.

    C'est LE contournement du bug `mistral_common` décrit dans la docstring
    du module : passer un dossier à `from_pretrained` évite tout appel au
    hub, là où passer un identifiant de repo en déclenche un même en mode
    hors-ligne.

    Best-effort et sans effet de bord : toute anomalie (hf_hub absent, cache
    corrompu ou partiel, `repo_id` qui est déjà un chemin) retourne `repo_id`
    tel quel.
    """
    snapshot = _complete_snapshot(repo_id)
    return repo_id if snapshot is None else str(snapshot)


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
    """(Ré)évalue le mode hors-ligne du process pour le modèle de DICTÉE.

    À appeler au démarrage et après tout changement de ce modèle : si
    l'utilisateur choisit dans les Préférences un modèle pas encore
    téléchargé, il faut re-autoriser le réseau.

    C'est un état GLOBAL, et le dernier appelant gagne : ne pas l'appeler
    pour un modèle secondaire (fichiers, traduction, diarisation). Pour
    ceux-là, `network_if_missing` ouvre le réseau le temps du chargement et
    le referme, sans effet sur le reste du process.

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


@contextlib.contextmanager
def network_if_missing(repo_id: str) -> Iterator[None]:
    """Autorise le réseau le temps du bloc si, et seulement si, `repo_id`
    n'est pas en cache.

    Pour les chargements de modèle « à la demande » : la délégation de la
    traduction vers Whisper Large V3, le modèle de diarisation. Le mode
    hors-ligne est posé au démarrage d'après le modèle de DICTÉE ; un second
    modèle absent du cache se heurtait alors à `OfflineModeIsEnabled`, une
    erreur sans rapport visible avec ce que l'utilisateur venait de faire.

    Quand le modèle est déjà là, le bloc tourne tel quel, hors-ligne compris :
    on ne rouvre jamais le réseau sans raison.
    """
    if is_model_cached(repo_id):
        yield
        return
    with allow_network():
        yield


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
