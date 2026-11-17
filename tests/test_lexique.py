"""Tests de lexique.py — corrections après transcription et amorce Whisper.

Pur texte : aucun modèle, aucun fichier. Les exemples viennent des erreurs
réellement relevées dans BENCH-2026-11.md.
"""

from __future__ import annotations

import lexique
from lexique import (
    AMORCE_MAX_CHARS,
    amorce,
    analyser_corrections,
    analyser_mots,
    corriger,
    est_echo_amorce,
    formater_corrections,
)


# ---- Le champ des Préférences ----


def test_analyser_reads_one_correction_per_line():
    texte = "KickSit = Kixit\nla TV a = la TVA\n"
    assert analyser_corrections(texte) == {"KickSit": "Kixit", "la TV a": "la TVA"}


def test_analyser_accepts_arrows_too():
    texte = "a -> b\nc → d\ne => f"
    assert analyser_corrections(texte) == {"a": "b", "c": "d", "e": "f"}


def test_analyser_ignores_lines_it_cannot_read():
    """Un champ de texte libre ne doit jamais bloquer la fermeture de la
    fenêtre : ce qui n'est pas une correction est simplement laissé de côté."""
    texte = "\n   \nsans séparateur\n = orphelin\nbon = ok"
    assert analyser_corrections(texte) == {"bon": "ok"}


def test_analyser_allows_an_empty_replacement():
    assert analyser_corrections("euh =") == {"euh": ""}


def test_formater_is_the_inverse_of_analyser():
    corrections = {"KickSit": "Kixit", "la TV a": "la TVA"}
    assert analyser_corrections(formater_corrections(corrections)) == corrections


# ---- corriger ----


def test_corriger_replaces_whole_words_ignoring_case():
    assert corriger("Kicksit et KICKSIT.", {"KickSit": "Kixit"}) == "Kixit et Kixit."


def test_corriger_does_not_bite_into_longer_words():
    """« TV » ne doit pas mordre dans « TVA », ni « art » dans « partir »."""
    assert corriger("la TVA et la TV", {"TV": "télé"}) == "la TVA et la télé"
    assert corriger("partir", {"art": "Art"}) == "partir"


def test_corriger_handles_multiword_entries_and_loose_spacing():
    assert (
        corriger("la  TV a été envoyée", {"la TV a": "la TVA a"})
        == "la TVA a été envoyée"
    )


def test_corriger_treats_both_apostrophes_alike():
    corrections = {"l'IA": "l'intelligence artificielle"}
    assert corriger("avec l’IA", corrections) == "avec l'intelligence artificielle"


def test_corriger_is_a_single_pass():
    """Le résultat d'une correction n'est pas recorrigé : pas de boucle."""
    assert corriger("A B", {"A": "B", "B": "A"}) == "B A"


def test_corriger_prefers_the_longest_entry():
    corrections = {"Jean": "J.", "Jean Jérôme": "Jean-Jérôme"}
    assert corriger("Jean Jérôme arrive", corrections) == "Jean-Jérôme arrive"


def test_corriger_survives_regex_characters():
    assert corriger("coût (HT) : 3", {"(HT)": "hors taxes"}) == "coût hors taxes : 3"


def test_corriger_without_corrections_returns_text_untouched():
    assert corriger("bonjour", {}) == "bonjour"
    assert corriger("bonjour", None) == "bonjour"
    assert corriger("", {"a": "b"}) == ""
    assert corriger("bonjour", {"  ": "x"}) == "bonjour"


def test_corriger_can_delete_a_word():
    """Et sans laisser deux espaces à la place du mot."""
    assert corriger("alors euh voilà", {"euh": ""}) == "alors voilà"
    assert corriger("Euh bonjour", {"euh": ""}) == "bonjour"


# ---- Mots et amorce ----


def test_analyser_mots_splits_dedupes_and_keeps_order():
    assert analyser_mots("Kixit, TVA\nkixit ;  Jean  Jérôme,, ") == [
        "Kixit", "TVA", "Jean Jérôme",
    ]


def test_amorce_builds_a_short_glossary_sentence():
    assert amorce(["Kixit", "TVA"]) == "Glossaire : Kixit, TVA."


def test_amorce_is_none_without_words():
    assert amorce([]) is None
    assert amorce(["", "   "]) is None


def test_amorce_drops_whole_words_beyond_the_limit():
    """Jamais de demi-nom propre : on s'arrête au dernier mot qui tient."""
    mots = [f"Client{i:03d}" for i in range(200)]
    texte = amorce(mots)
    assert texte is not None
    assert len(texte) <= AMORCE_MAX_CHARS
    assert texte.endswith(".")
    dernier = texte.rstrip(".").split(", ")[-1]
    assert dernier in mots


# ---- Écho de l'amorce ----


def test_echo_is_detected_whatever_the_punctuation():
    soufflee = amorce(["Kixit", "TVA"])
    assert est_echo_amorce("Glossaire : Kixit, TVA.", soufflee)
    assert est_echo_amorce("glossaire kixit tva", soufflee)
    assert est_echo_amorce("Glossaire : Kixit", soufflee)  # début seulement


def test_real_dictation_is_not_mistaken_for_an_echo():
    soufflee = amorce(["Kixit", "TVA"])
    assert not est_echo_amorce("Bonjour Kixit, la TVA est à envoyer.", soufflee)
    assert not est_echo_amorce("Glossaire des termes du contrat.", soufflee)
    assert not est_echo_amorce("Glossaire", soufflee)
    assert not est_echo_amorce("", soufflee)


def test_no_echo_check_without_prompt():
    assert not est_echo_amorce(lexique.AMORCE_PREFIXE + "Kixit.", None)
