"""Tests de oral.py — de l'écrit à l'oral, avant la synthèse vocale.

Pur texte. Les nombres suivent l'orthographe traditionnelle ; les cas viennent
de BENCH-VOIX-2026-11.md (la phrase que Pocket TTS rendait en charabia) et des
pièges classiques du français (quatre-vingts, cents, mille, et un).
"""

from __future__ import annotations

import pytest

from oral import nombre_en_lettres, nombres_en_lettres, preparer


@pytest.mark.parametrize(
    ("n", "attendu"),
    [
        (0, "zéro"), (1, "un"), (16, "seize"), (17, "dix-sept"),
        (21, "vingt et un"), (22, "vingt-deux"), (61, "soixante et un"),
        (70, "soixante-dix"), (71, "soixante et onze"), (72, "soixante-douze"),
        (79, "soixante-dix-neuf"), (80, "quatre-vingts"), (81, "quatre-vingt-un"),
        (90, "quatre-vingt-dix"), (91, "quatre-vingt-onze"),
        (99, "quatre-vingt-dix-neuf"), (100, "cent"), (101, "cent un"),
        (200, "deux cents"), (201, "deux cent un"), (280, "deux cent quatre-vingts"),
        (1000, "mille"), (1001, "mille un"), (1100, "mille cent"),
        (1998, "mille neuf cent quatre-vingt-dix-huit"),
        (2000, "deux mille"), (2026, "deux mille vingt-six"),
        (3500, "trois mille cinq cents"),
        # « cents » et « vingts » perdent leur s devant mille, pas devant millions.
        (80_000, "quatre-vingt mille"), (200_000, "deux cent mille"),
        (200_200, "deux cent mille deux cents"),
        (1_000_000, "un million"), (2_000_000, "deux millions"),
        (200_000_000, "deux cents millions"), (1_000_000_000, "un milliard"),
    ],
)
def test_nombre_en_lettres(n: int, attendu: str):
    assert nombre_en_lettres(n) == attendu


def test_nombre_hors_intervalle_leve():
    with pytest.raises(ValueError):
        nombre_en_lettres(10**12)
    with pytest.raises(ValueError):
        nombre_en_lettres(-1)


def test_the_bench_sentence_is_fully_spelled_out():
    """La phrase que Pocket TTS rendait en charabia quand elle était en chiffres."""
    texte = (
        "La réunion est fixée au 12 mars 2026 à 14h30, salle B, avec un "
        "budget de 3500 euros hors taxes."
    )
    assert nombres_en_lettres(texte) == (
        "La réunion est fixée au douze mars deux mille vingt-six à quatorze "
        "heures trente, salle B, avec un budget de trois mille cinq cents "
        "euros hors taxes."
    )


@pytest.mark.parametrize(
    ("texte", "attendu"),
    [
        # Heures
        ("à 9h", "à neuf heures"), ("à 1h", "à une heure"),
        ("à 14h05", "à quatorze heures cinq"), ("à 14:30", "à quatorze heures trente"),
        ("à 0h30", "à zéro heure trente"), ("à 10h00", "à dix heures"),
        # Séparateurs de milliers, y compris insécables
        ("200 000 habitants", "deux cent mille habitants"),
        ("200 000 habitants", "deux cent mille habitants"),
        ("200 000 habitants", "deux cent mille habitants"),
        # Pourcentages et décimaux
        ("18 %", "dix-huit pour cent"), ("65%", "soixante-cinq pour cent"),
        ("12,5 points", "douze virgule cinq points"),
        ("0,05 mm", "zéro virgule zéro cinq mm"),
        # Euros
        ("3500 €", "trois mille cinq cents euros"), ("1 €", "un euro"),
        ("12,50 €", "douze euros cinquante"), ("12,5 euros", "douze euros cinquante"),
        ("200 000 €", "deux cent mille euros"),
        # Ordinaux
        ("le 1er lot", "le premier lot"), ("la 1re fois", "la première fois"),
        ("la 2e phase", "la deuxième phase"), ("le 5e", "le cinquième"),
        ("le 9ème", "le neuvième"), ("la 21e édition", "la vingt et unième édition"),
        ("le 80e", "le quatre-vingtième"),
        # Dates
        ("le 03/12/2026", "le trois décembre deux mille vingt-six"),
        ("le 1/1/2027", "le premier janvier deux mille vingt-sept"),
        # Zéros de tête : on les dit un par un
        ("le 06 12 34 56 78", "le zéro six douze trente-quatre cinquante-six soixante-dix-huit"),
        # Signes
        ("n° 4 & 5", "numéro quatre et cinq"),
    ],
)
def test_nombres_en_lettres_cases(texte: str, attendu: str):
    assert nombres_en_lettres(texte) == attendu


@pytest.mark.parametrize(
    "texte",
    [
        "Bonjour, comment allez-vous ?",
        "version 1.2.3",                 # numéro de version
        "IP 192.168.1.10",               # adresse
        "le XXe siècle",                 # chiffres romains : la voix s'en sort
        "kixit.fr",                      # un point n'est pas une virgule
        "99 999 999 999 999",            # trop grand pour être dit
    ],
)
def test_what_is_left_untouched(texte: str):
    assert nombres_en_lettres(texte) == texte


def test_impossible_dates_and_hours_are_not_read_as_such():
    """« 45/13 » n'est pas un jour de treizième mois, « 31h99 » pas une heure."""
    assert "heure" not in nombres_en_lettres("à 31h99")
    lu = nombres_en_lettres("le 45/13/2026")
    assert "quarante-cinq" in lu and "janvier" not in lu and "décembre" not in lu


def test_text_without_numbers_is_returned_as_is():
    texte = "Un  texte   avec des espaces,  sans chiffre."
    assert nombres_en_lettres(texte) is texte


def test_preparer_applies_pronunciations_then_numbers():
    """Les prononciations d'abord : une entrée de l'utilisateur l'emporte sur
    les règles générales."""
    prononciations = {"enneigement": "en-neigement", "3M": "trois M"}
    assert preparer("L'enneigement chez 3M en 2026.", prononciations) == (
        "L'en-neigement chez trois M en deux mille vingt-six."
    )


def test_preparer_can_leave_numbers_alone():
    assert preparer("En 2026.", {"2026": "vingt vingt-six"}, nombres=False) == (
        "En vingt vingt-six."
    )
    assert preparer("En 2026.", None, nombres=False) == "En 2026."
