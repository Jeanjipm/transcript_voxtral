"""
De l'écrit à l'oral : préparer un texte avant de le donner à la voix.

Les modèles de synthèse lisent des mots, pas des chiffres. Mesuré le
18 novembre 2026 (cf. BENCH-VOIX-2026-11.md) : la phrase « au 12 mars 2026 à
14h30 » sort en charabia de Pocket TTS, et parfaite une fois écrite « au
douze mars deux mille vingt-six à quatorze heures trente ». Voxtral TTS s'en
tire un peu mieux avec des chiffres, pas bien pour autant.

Ce module fait donc deux choses, dans cet ordre (`preparer`) :

1. **Les prononciations de l'utilisateur** : un tableau « mot écrit » →
   « façon de l'écrire pour qu'il soit bien dit », pour les mots qu'une voix
   écorche. Même mécanique que les corrections de dictée (`lexique.corriger`),
   dans l'autre sens. En premier, pour qu'une entrée de l'utilisateur
   l'emporte sur les règles générales (« 3M = trois M »).
2. **Les nombres en toutes lettres** : entiers, décimaux, heures, dates,
   pourcentages, euros, ordinaux.

Ce qui n'est PAS fait, exprès : les chiffres romains (« XXe siècle » se lit
très bien tel quel, et « MIX » n'est pas mille neuf), les numéros de version
(« 1.2.3 »), et tout ce qui dépasse le millier de milliards.

Texte seulement : ni modèle, ni fichier, ni interface. `vocaliser_cahier.py`
garde sa propre adaptation (`adapter_oral`) : ses MP3 sont identifiés par leur
texte, et changer la règle obligerait à tous les régénérer.
"""

from __future__ import annotations

import re
from typing import Mapping

import lexique


# ----------------------------------------------------------------------
# Un entier en toutes lettres
# ----------------------------------------------------------------------

_UNITES = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf",
]
_DIZAINES = {
    20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante",
}
# Au-delà, on laisse le nombre tel quel : personne ne dicte un billion.
_MAX = 999_999_999_999


def _sous_cent(n: int) -> str:
    """0 à 99, sans le « s » de quatre-vingts (ajouté par l'appelant)."""
    if n < 20:
        return _UNITES[n]
    if n < 70:
        dizaine, unite = n - n % 10, n % 10
        if unite == 0:
            return _DIZAINES[dizaine]
        if unite == 1:
            return f"{_DIZAINES[dizaine]} et un"
        return f"{_DIZAINES[dizaine]}-{_UNITES[unite]}"
    if n < 80:
        # soixante-dix … soixante-dix-neuf ; « soixante et onze » seul avec « et ».
        return "soixante et onze" if n == 71 else f"soixante-{_UNITES[n - 60]}"
    # quatre-vingt, quatre-vingt-un … quatre-vingt-dix-neuf (jamais de « et »).
    return "quatre-vingt" if n == 80 else f"quatre-vingt-{_UNITES[n - 80]}"


def _sous_mille(n: int, final: bool) -> str:
    """1 à 999. `final` : ce groupe termine le nombre (ou précède « millions »),
    et prend donc les « s » de « cents » et de « quatre-vingts » ; devant
    « mille » ils tombent (« deux cent mille », « quatre-vingt mille »)."""
    centaines, reste = divmod(n, 100)
    mots: list[str] = []
    if centaines:
        if centaines > 1:
            mots.append(_UNITES[centaines])
        mots.append("cents" if centaines > 1 and reste == 0 and final else "cent")
    if reste:
        texte = _sous_cent(reste)
        if reste == 80 and final:
            texte += "s"
        mots.append(texte)
    return " ".join(mots)


def nombre_en_lettres(n: int) -> str:
    """Un entier positif en toutes lettres (orthographe traditionnelle).

    Lève `ValueError` hors de l'intervalle géré : à l'appelant de laisser
    alors le nombre tel quel.
    """
    if n < 0 or n > _MAX:
        raise ValueError(f"nombre hors de l'intervalle géré : {n}")
    if n == 0:
        return "zéro"
    milliards, n = divmod(n, 1_000_000_000)
    millions, n = divmod(n, 1_000_000)
    milliers, unites = divmod(n, 1_000)
    mots: list[str] = []
    if milliards:
        mots.append(f"{_sous_mille(milliards, True)} milliard{'s' if milliards > 1 else ''}")
    if millions:
        mots.append(f"{_sous_mille(millions, True)} million{'s' if millions > 1 else ''}")
    if milliers:
        # « mille », jamais « un mille » ; et invariable.
        mots.append("mille" if milliers == 1 else f"{_sous_mille(milliers, False)} mille")
    if unites:
        mots.append(_sous_mille(unites, True))
    return " ".join(mots)


def _chiffres(texte: str) -> str:
    """Suite de chiffres → nombre en lettres. Les zéros de tête se disent un
    par un (« 06 » → « zéro six », « 007 » → « zéro zéro sept ») : c'est ainsi
    qu'on lit un numéro de téléphone ou un code postal."""
    zeros = len(texte) - len(texte.lstrip("0"))
    if zeros == len(texte):
        return " ".join(["zéro"] * len(texte))
    try:
        reste = nombre_en_lettres(int(texte))
    except ValueError:
        return texte
    return " ".join(["zéro"] * zeros + [reste])


def _ordinal(n: int, feminin: bool = False) -> str:
    """1 → premier / première, 2 → deuxième, 9 → neuvième, 21 → vingt et unième."""
    if n == 1:
        return "première" if feminin else "premier"
    base = nombre_en_lettres(n)
    if base.endswith("cinq"):
        return base + "uième"
    if base.endswith("neuf"):
        return base[:-1] + "vième"
    if base.endswith(("cents", "vingts")):
        base = base[:-1]
    if base.endswith("e"):
        base = base[:-1]
    return base + "ième"


# ----------------------------------------------------------------------
# Les nombres d'un texte
# ----------------------------------------------------------------------

# Espaces qui séparent les milliers : espace, insécable, fine insécable.
_SEP = "[   ]"
# Un entier, avec ou sans séparateurs de milliers (« 3500 », « 3 500 »).
_ENTIER = rf"\d{{1,3}}(?:{_SEP}\d{{3}})+(?!\d)|\d+"
_MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
    "septembre", "octobre", "novembre", "décembre",
]

# Une version (« 1.2.3 ») ou une adresse IP : on n'y touche pas. Repérée
# d'abord, pour que ses morceaux ne soient pas lus comme des décimaux.
_VERSION = re.compile(r"(?<![\w.,])\d+(?:\.\d+){2,}(?![\w.,]*\d)")
_DATE = re.compile(r"(?<![\w/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w/])")
_HEURE = re.compile(r"(?<![\w:])(\d{1,2})\s?(?:h|H|:)\s?(\d{2})?(?![\w:])")
_EUROS = re.compile(rf"(?<![\w,.])({_ENTIER})(?:[,.](\d{{1,2}}))?\s?(€|euros?)(?!\w)")
_POURCENT = re.compile(r"\s?%")
_ORDINAL = re.compile(r"(?<![\w,.])(\d+)(er|ère|re|ème|eme|e|è)\b")
_DECIMAL = re.compile(rf"(?<![\w,.])({_ENTIER})[,.](\d+)(?![\w,.]*\d)")
_NOMBRE = re.compile(rf"(?<![\w])({_ENTIER})(?!\d)")
_SIGNES = (
    (re.compile(r"\bn°\s?", re.IGNORECASE), "numéro "),
    (re.compile(r"\s?&\s?"), " et "),
    (re.compile(r"\s?€"), " euros"),
)


def _sans_separateurs(texte: str) -> str:
    return re.sub(_SEP, "", texte)


def _date(m: re.Match[str]) -> str:
    jour, mois, annee = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= jour <= 31 and 1 <= mois <= 12):
        return m.group(0)
    jour_txt = "premier" if jour == 1 else nombre_en_lettres(jour)
    # « 24 » pour 2024 : on ne devine pas le siècle, on dit ce qui est écrit.
    return f"{jour_txt} {_MOIS[mois - 1]} {_chiffres(annee)}"


def _heure(m: re.Match[str]) -> str:
    heures, minutes = int(m.group(1)), m.group(2)
    if heures > 24 or (minutes is not None and int(minutes) > 59):
        return m.group(0)
    if ":" in m.group(0) and minutes is None:
        return m.group(0)
    heure_txt = "une heure" if heures == 1 else (
        "zéro heure" if heures == 0 else f"{nombre_en_lettres(heures)} heures"
    )
    if minutes is None or int(minutes) == 0:
        return heure_txt
    return f"{heure_txt} {nombre_en_lettres(int(minutes))}"


def _euros(m: re.Match[str]) -> str:
    entier = int(_sans_separateurs(m.group(1)))
    centimes = m.group(2)
    if entier > _MAX:
        return m.group(0)
    texte = f"{nombre_en_lettres(entier)} euro{'s' if entier > 1 else ''}"
    if centimes and int(centimes):
        # « 12,5 € » = douze euros cinquante.
        texte += f" {nombre_en_lettres(int(centimes.ljust(2, '0')))}"
    return texte


def _ordinal_match(m: re.Match[str]) -> str:
    n, suffixe = int(m.group(1)), m.group(2)
    if n == 0 or n > _MAX:
        return m.group(0)
    # « 1er », « 1re » ne valent que pour un ; « 2er » n'est pas un ordinal.
    if suffixe in ("er", "ère", "re") and n != 1:
        return m.group(0)
    return _ordinal(n, feminin=suffixe in ("ère", "re"))


def _decimal(m: re.Match[str]) -> str:
    entier = _sans_separateurs(m.group(1))
    if int(entier) > _MAX:
        return m.group(0)
    return f"{_chiffres(entier)} virgule {_chiffres(m.group(2))}"


def _entier(m: re.Match[str]) -> str:
    chiffres = _sans_separateurs(m.group(1))
    lu = _chiffres(chiffres)
    # Trop grand pour être dit : on rend le texte d'origine, espaces compris.
    return m.group(0) if lu == chiffres else lu


def nombres_en_lettres(texte: str) -> str:
    """Écrit en toutes lettres les nombres d'un texte.

    Du plus spécifique au plus général, pour que « 14h30 » ne devienne pas
    « quatorze h trente » ni « 12,50 € » « douze virgule cinquante euros ».
    """
    if not any(c.isdigit() for c in texte) and "€" not in texte and "%" not in texte:
        return texte

    # Les versions sont mises à l'abri le temps des remplacements.
    abris: list[str] = []

    def abriter(m: re.Match[str]) -> str:
        abris.append(m.group(0))
        # Un repère sans chiffre : il ne doit pas être lu comme un nombre.
        return "\x00" + "x" * len(abris) + "\x00"

    t = _VERSION.sub(abriter, texte)
    t = _DATE.sub(_date, t)
    t = _HEURE.sub(_heure, t)
    t = _EUROS.sub(_euros, t)
    t = _ORDINAL.sub(_ordinal_match, t)
    t = _DECIMAL.sub(_decimal, t)
    t = _NOMBRE.sub(_entier, t)
    t = _POURCENT.sub(" pour cent", t)
    for motif, mot in _SIGNES:
        t = motif.sub(mot, t)
    t = re.sub(r"\x00(x+)\x00", lambda m: abris[len(m.group(1)) - 1], t)
    return re.sub(r" {2,}", " ", t).strip()


# ----------------------------------------------------------------------
# Entrée unique
# ----------------------------------------------------------------------


def preparer(
    texte: str,
    prononciations: Mapping[str, str] | None = None,
    nombres: bool = True,
) -> str:
    """Le texte tel qu'il sera donné à la voix."""
    texte = lexique.corriger(texte, prononciations)
    if nombres:
        texte = nombres_en_lettres(texte)
    return texte
