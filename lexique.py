"""
Lexique personnel : les mots que les modèles écorchent, remis d'aplomb.

Tous les modèles de transcription butent sur les mêmes choses : noms propres,
sigles, noms de clients (« KickSit » pour « Kixit », « la TV a été envoyé »
pour « la TVA est à envoyer » — cf. BENCH-2026-11.md). Deux remèdes, de
nature très différente, vivent ici :

1. **Les corrections** (`corriger`) : un tableau « ce que le modèle écrit »
   → « ce que je veux », appliqué APRÈS la transcription. C'est un simple
   rechercher-remplacer : aucun risque pour le reste du texte, et ça marche
   avec tous les modèles, en dictée comme sur les fichiers.

2. **L'amorce** (`amorce`) : une liste de mots soufflée à Whisper AVANT qu'il
   transcrive (son paramètre `initial_prompt`). Il peut alors écrire juste du
   premier coup un mot qu'on n'a jamais vu passer en erreur. Mais c'est
   expérimental, et désactivé par défaut : on a mesuré sur les fichiers
   (cf. file_transcriber.py, « Pourquoi on ne passe PAS d'amorce de texte »)
   qu'une amorce peut faire résumer Whisper et lui faire sauter du discours.
   Une courte liste de mots biaise moins qu'un paragraphe, mais rien ne le
   garantit ; d'où l'interrupteur dans les Préférences. Voxtral, lui, ne
   prend aucune amorce (mlx-voxtral n'en offre pas le moyen).

Ce module ne fait que du texte : ni modèle, ni fichier, ni interface.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping


# Séparateurs acceptés dans les Préférences entre « écrit » et « voulu ».
# « = » est celui qu'on affiche (il se tape sur tous les claviers) ; les
# flèches sont tolérées parce que c'est ce qu'on écrit spontanément.
_SEPARATEURS = ("→", "=>", "->", "=")
SEPARATEUR_AFFICHE = " = "

# Les deux apostrophes sont une même lettre pour nous : les modèles rendent
# tantôt la droite, tantôt la typographique, et l'utilisateur tape la droite.
_APOSTROPHES = "'’"

# Whisper ne lit que les ~224 derniers jetons de son amorce, et plus elle est
# longue plus elle pèse sur le décodage. 400 caractères ≈ une quarantaine de
# mots : assez pour un carnet de clients, assez court pour rester une liste.
AMORCE_MAX_CHARS = 400
AMORCE_PREFIXE = "Glossaire : "


# ----------------------------------------------------------------------
# Corrections (après la transcription)
# ----------------------------------------------------------------------


def analyser_corrections(texte: str) -> dict[str, str]:
    """Le champ des Préférences (une correction par ligne) → dictionnaire.

    Une ligne s'écrit `écrit = voulu` (ou avec `->`, `→`). Les lignes vides,
    sans séparateur ou sans partie gauche sont ignorées : un champ de texte
    libre ne doit jamais empêcher de fermer la fenêtre. Une partie droite
    vide est permise — c'est « supprime ce mot ».
    """
    corrections: dict[str, str] = {}
    for ligne in texte.splitlines():
        coupe = _couper(ligne)
        if coupe is None:
            continue
        ecrit, voulu = coupe
        if ecrit:
            corrections[ecrit] = voulu
    return corrections


def _couper(ligne: str) -> tuple[str, str] | None:
    """Coupe sur le premier séparateur rencontré dans la ligne."""
    positions = [
        (ligne.find(sep), sep) for sep in _SEPARATEURS if sep in ligne
    ]
    if not positions:
        return None
    # À position égale (« => » contient « = »), le plus long l'emporte.
    index, sep = min(positions, key=lambda p: (p[0], -len(p[1])))
    return ligne[:index].strip(), ligne[index + len(sep):].strip()


def formater_corrections(corrections: Mapping[str, str]) -> str:
    """Dictionnaire → texte du champ des Préférences (inverse d'`analyser`)."""
    return "\n".join(
        f"{ecrit}{SEPARATEUR_AFFICHE}{voulu}" for ecrit, voulu in corrections.items()
    )


def _motif(ecrit: str) -> str:
    """Expression régulière qui reconnaît `ecrit` dans un texte transcrit.

    Souple là où les modèles varient (espaces multiples, type d'apostrophe),
    stricte sur les bords : « TV » ne doit pas mordre dans « TVA ».
    """
    morceaux: list[str] = []
    for mot in ecrit.split():
        echappe = "".join(
            f"[{_APOSTROPHES}]" if c in _APOSTROPHES else re.escape(c) for c in mot
        )
        morceaux.append(echappe)
    return r"(?<!\w)" + r"\s+".join(morceaux) + r"(?!\w)"


def corriger(texte: str, corrections: Mapping[str, str] | None) -> str:
    """Applique les corrections à un texte transcrit.

    Mot entier, sans tenir compte des majuscules. Tout se fait en UNE passe :
    le résultat d'une correction n'est jamais recorrigé par une autre (sinon
    « A = B » et « B = A » tourneraient en rond), et quand deux entrées
    peuvent s'appliquer au même endroit, la plus longue gagne.
    """
    if not texte or not corrections:
        return texte
    table = {
        _cle(str(ecrit)): str(voulu)
        for ecrit, voulu in corrections.items()
        if str(ecrit).strip()
    }
    if not table:
        return texte
    alternatives = sorted(
        (str(e) for e in corrections if str(e).strip()), key=len, reverse=True
    )
    motif = re.compile("|".join(_motif(e) for e in alternatives), re.IGNORECASE)
    corrige = motif.sub(lambda m: table.get(_cle(m.group(0)), m.group(0)), texte)
    if "" in table.values():
        # Un mot supprimé laisse deux espaces côte à côte : on referme.
        corrige = re.sub(r" {2,}", " ", corrige).strip()
    return corrige


def _cle(texte: str) -> str:
    """Forme de comparaison : minuscules, espaces et apostrophes unifiés."""
    unifie = " ".join(texte.split()).casefold()
    return unifie.replace("’", "'")


# ----------------------------------------------------------------------
# Amorce (avant la transcription, Whisper seulement)
# ----------------------------------------------------------------------


def analyser_mots(texte: str) -> list[str]:
    """Le champ « mes mots » (séparés par des virgules ou des retours à la
    ligne) → liste sans vides ni doublons, dans l'ordre de saisie."""
    mots: list[str] = []
    vus: set[str] = set()
    for brut in re.split(r"[,;\n]", texte):
        mot = " ".join(brut.split())
        if mot and mot.casefold() not in vus:
            vus.add(mot.casefold())
            mots.append(mot)
    return mots


def amorce(mots: Iterable[str]) -> str | None:
    """La phrase soufflée à Whisper, ou None s'il n'y a rien à souffler.

    Les mots en trop (au-delà d'`AMORCE_MAX_CHARS`) sont laissés de côté
    plutôt que coupés au milieu : un demi-nom propre ferait pire que rien.
    """
    retenus: list[str] = []
    longueur = len(AMORCE_PREFIXE) + 1  # le point final
    for mot in mots:
        mot = " ".join(str(mot).split())
        if not mot:
            continue
        ajout = len(mot) + (2 if retenus else 0)
        if longueur + ajout > AMORCE_MAX_CHARS:
            break
        retenus.append(mot)
        longueur += ajout
    if not retenus:
        return None
    return AMORCE_PREFIXE + ", ".join(retenus) + "."


def est_echo_amorce(texte: str, amorce_envoyee: str | None) -> bool:
    """True si le modèle n'a fait que recracher l'amorce.

    Travers connu de Whisper : sur un audio presque vide, il « transcrit »
    le texte qu'on lui a soufflé. Mieux vaut ne rien coller que coller
    « Glossaire : Kixit, TVA. » au milieu d'un courriel.
    """
    if not amorce_envoyee or not texte.strip():
        return False
    rendu, soufflee = _squelette(texte), _squelette(amorce_envoyee)
    # Tout ou début de l'amorce, étiquette « Glossaire » comprise. Une vraie
    # dictée qui commence par le mot « glossaire » continue autrement, et
    # n'est donc pas prise pour un écho.
    return len(rendu) > len(_squelette(AMORCE_PREFIXE)) and soufflee.startswith(rendu)


def _squelette(texte: str) -> str:
    """Lettres et chiffres seulement, en minuscules : pour comparer deux
    textes sans se soucier de la ponctuation que le modèle a pu changer."""
    return re.sub(r"[\W_]+", "", texte.casefold())
