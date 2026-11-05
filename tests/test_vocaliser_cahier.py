"""Tests de vocaliser_cahier.py — les voix du cahier CE1, synthèse simulée.

Trois choses à verrouiller :
  - l'adaptation à l'oral (ce que le modèle lit) ;
  - l'identifiant = celui de phrases.mjs et de l'application (SHA-1 tronqué) ;
  - la génération incrémentale : conserver, régénérer au changement de voix,
    corbeille pour les orphelins, reprise après interruption.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import vocaliser_cahier as vc
from vocaliser_cahier import (
    PHRASES_ESSAI,
    VERCEL_JSON,
    CahierError,
    Phrase,
    adapter_oral,
    charger_phrases,
    essayer_voix,
    generer,
    identifiant,
    lire_manifest,
)


# ---- Identifiants : les mêmes que phrases.mjs et l'application ----

# Couples (id, texte) copiés de la sortie réelle de phrases.mjs le 5 nov. 2026.
PAIRES_REELLES = [
    ("d60038d564827732", "!  Le point d'exclamation"),
    ("2e67d51d0502e703", "1 cm = … mm"),
    ("239e715b70e9f57c", "2 cm, c'est combien de millimètres ? 20, 2, 12"),
    ("69b14d1bdeb57e0f", "La phrase interrogative sert à interroger des gens, à poser des questions."),
    ("74ac07230fa796f0", "Le segment [CD] mesure 6 cm et 4 mm."),
]


@pytest.mark.parametrize("attendu,texte", PAIRES_REELLES)
def test_identifiant_matches_the_site(attendu: str, texte: str):
    """Le nom du MP3 doit être exactement l'id que l'application calcule :
    16 premiers hexa du SHA-1 du texte en UTF-8."""
    assert identifiant(texte) == attendu
    assert len(identifiant(texte)) == 16


def test_identifiant_is_computed_on_the_original_text_not_the_oral_one():
    texte = "1 cm = … mm"
    assert adapter_oral(texte) != texte
    assert identifiant(texte) == "2e67d51d0502e703"


def test_charger_phrases_accepts_consistent_ids():
    data = {"phrases": [{"id": i, "texte": t} for i, t in PAIRES_REELLES]}
    phrases = charger_phrases(data)
    assert [p.texte for p in phrases] == [t for _, t in PAIRES_REELLES]


def test_charger_phrases_rejects_an_inconsistent_id():
    data = {"phrases": [{"id": "0000000000000000", "texte": "Bonjour"}]}
    with pytest.raises(CahierError, match="Identifiant inattendu"):
        charger_phrases(data)


def test_charger_phrases_rejects_empty_list():
    with pytest.raises(CahierError):
        charger_phrases({"phrases": []})


# ---- Adaptation à l'oral ----


@pytest.mark.parametrize(
    "ecrit,oral",
    [
        # Règles demandées
        ("3 + 4 = 7", "3 plus 4 égale 7"),
        ("Le double de 7 = ? 14, 12, 16", "Le double de 7 égale ? 14, 12, 16"),
        ("a → b", "a, b"),
        ("Le cube a … faces.", "Le cube a. faces."),
        ("1 cm = 10 mm", "1 centimètre égale 10 millimètres"),
        ("Le segment [CD] mesure 6 cm et 4 mm.", "Le segment C D mesure 6 centimètres et 4 millimètres."),
        ("2 cm, c'est combien de millimètres ? 20, 2, 12", "2 centimètres, c'est combien de millimètres ? 20, 2, 12"),
        ("Comment note-t-on les centimètres ? cm, mm, c", "Comment note-t-on les centimètres ? centimètre, millimètre, c"),
        ("!  Le point d'exclamation", "Le point d'exclamation"),
        (",  La virgule", "La virgule"),
        ("?  Le point d'interrogation", "Le point d'interrogation"),
        ("•  Le point", "Le point"),
        # Les crochets phonétiques ne sont pas des segments
        ("on entend le son… [in], [an], [è]", "on entend le son. [in], [an], [è]"),
        ("(AB), [AB], {AB}", "(AB), A B, {AB}"),
        # Ajouts : saut de ligne, comparaisons
        ("0 : zéro\n1 : un", "0 : zéro, 1 : un"),
        ("2 < 6", "2 est plus petit que 6"),
        ("19 > 16", "19 est plus grand que 16"),
        # Nettoyage
        ("1 cm = … mm", "1 centimètre égale. millimètre"),
        ("… + 6 = 13", ". plus 6 égale 13"),
        ("3. Ce qui se passera  le futur", "3. Ce qui se passera le futur"),
        ("Rien à changer.", "Rien à changer."),
    ],
)
def test_adapter_oral(ecrit: str, oral: str):
    assert adapter_oral(ecrit) == oral


def test_adapter_oral_singular_for_zero_or_one():
    assert adapter_oral("0 cm") == "0 centimètre"
    assert adapter_oral("1 mm") == "1 millimètre"
    assert adapter_oral("12 mm") == "12 millimètres"


def test_phrases_essai_cover_the_three_cases():
    """Une règle, une question avec choix, une phrase avec nombres et cm."""
    regle, question, mesure = PHRASES_ESSAI
    assert regle.endswith(".") and "?" not in regle
    assert "?" in question and ", " in question
    assert "cm" in mesure and any(c.isdigit() for c in mesure)


# ---- Génération incrémentale (synthèse simulée) ----


PHRASES = [Phrase(identifiant(t), t) for t in ("Un.", "Deux + 2.", "Trois cm.")]


class FauxMoteur:
    """`lire` et `vers_mp3` remplacés : on compte les appels et on écrit un
    MP3 factice dont le contenu encode la voix, pour vérifier les
    régénérations."""

    def __init__(self) -> None:
        self.lus: list[tuple[str, str]] = []
        self.corbeille: list[list[Path]] = []
        self.interrompre_a: int | None = None
        self.illisibles: set[str] = set()

    def lire(self, texte: str, voix: str) -> np.ndarray:
        if self.interrompre_a is not None and len(self.lus) == self.interrompre_a:
            raise KeyboardInterrupt
        if texte in self.illisibles:
            raise vc.synthese.SynthesisError("n'arrive pas à finir")
        self.lus.append((texte, voix))
        return np.full(4, len(self.lus), dtype=np.float32)

    def vers_mp3(self, audio: np.ndarray, chemin: Path) -> None:
        chemin.write_bytes(b"mp3:" + str(self.lus[-1][1]).encode())

    def jeter(self, chemins) -> None:
        self.corbeille.append(sorted(Path(c) for c in chemins))

    def generer(self, phrases, voix, dossier):
        return generer(
            phrases, voix, dossier,
            lire=self.lire, vers_mp3=self.vers_mp3, corbeille=self.jeter,
            afficher=False,
        )


@pytest.fixture
def moteur() -> FauxMoteur:
    return FauxMoteur()


def _mp3s(dossier: Path) -> set[str]:
    return {p.stem for p in (dossier / "audio").glob("*.mp3")}


def test_first_run_generates_everything_and_writes_the_folder(
    tmp_path: Path, moteur: FauxMoteur
):
    bilan = moteur.generer(PHRASES, "fr_female", tmp_path)

    assert (bilan.generes, bilan.conserves, bilan.supprimes) == (3, 0, 0)
    assert _mp3s(tmp_path) == {p.id for p in PHRASES}
    # Le modèle a reçu le texte ADAPTÉ, avec la voix demandée.
    assert ("Deux plus 2.", "fr_female") in moteur.lus

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"voix": "fr_female", "fichiers": sorted(p.id for p in PHRASES)}
    assert "Fichiers audio du cahier de révisions CE1" in (tmp_path / "index.html").read_text()
    assert json.loads((tmp_path / "vercel.json").read_text()) == VERCEL_JSON
    assert (tmp_path / ".vercelignore").read_text().split() == ["essais/", "phrases.json"]


def test_second_run_keeps_existing_mp3s(tmp_path: Path, moteur: FauxMoteur):
    moteur.generer(PHRASES, "fr_female", tmp_path)
    moteur.lus.clear()

    bilan = moteur.generer(PHRASES, "fr_female", tmp_path)
    assert moteur.lus == []
    assert (bilan.generes, bilan.conserves) == (0, 3)
    assert moteur.corbeille == []


def test_new_phrase_is_added_without_touching_the_others(
    tmp_path: Path, moteur: FauxMoteur
):
    moteur.generer(PHRASES, "fr_female", tmp_path)
    moteur.lus.clear()
    nouvelle = Phrase(identifiant("Quatre."), "Quatre.")

    bilan = moteur.generer(PHRASES + [nouvelle], "fr_female", tmp_path)
    assert moteur.lus == [("Quatre.", "fr_female")]
    assert (bilan.generes, bilan.conserves) == (1, 3)
    assert nouvelle.id in lire_manifest(tmp_path)["fichiers"]


def test_removed_phrase_goes_to_the_trash_and_leaves_the_manifest(
    tmp_path: Path, moteur: FauxMoteur
):
    moteur.generer(PHRASES, "fr_female", tmp_path)
    retiree = PHRASES[1]
    # La corbeille est simulée : on retire le fichier nous-mêmes, comme
    # NSFileManager le ferait.
    moteur.jeter = lambda chemins: [  # type: ignore[method-assign]
        (moteur.corbeille.append(sorted(chemins)), [Path(c).unlink() for c in chemins])
    ]

    bilan = moteur.generer([PHRASES[0], PHRASES[2]], "fr_female", tmp_path)
    assert moteur.corbeille == [[tmp_path / "audio" / f"{retiree.id}.mp3"]]
    assert bilan.supprimes == 1
    assert retiree.id not in lire_manifest(tmp_path)["fichiers"]
    assert len(lire_manifest(tmp_path)["fichiers"]) == 2


def test_changing_voice_regenerates_everything(tmp_path: Path, moteur: FauxMoteur):
    moteur.generer(PHRASES, "fr_female", tmp_path)
    anciens = sorted((tmp_path / "audio").glob("*.mp3"))
    moteur.lus.clear()

    bilan = moteur.generer(PHRASES, "fr_male", tmp_path)
    assert bilan.tout_regenere is True
    assert moteur.corbeille == [anciens]
    assert [v for _, v in moteur.lus] == ["fr_male"] * 3
    assert lire_manifest(tmp_path)["voix"] == "fr_male"
    for p in (tmp_path / "audio").glob("*.mp3"):
        assert p.read_bytes() == b"mp3:fr_male"


def test_same_voice_with_empty_manifest_does_not_regenerate(
    tmp_path: Path, moteur: FauxMoteur
):
    """Un manifeste absent (ancienne génération, fichier perdu) ne doit pas
    faire jeter des MP3 valides."""
    moteur.generer(PHRASES, "fr_female", tmp_path)
    (tmp_path / "manifest.json").unlink()
    moteur.lus.clear()
    bilan = moteur.generer(PHRASES, "fr_female", tmp_path)
    assert bilan.tout_regenere is False and moteur.lus == []


def test_interruption_keeps_a_clean_state_and_resumes(
    tmp_path: Path, moteur: FauxMoteur
):
    moteur.interrompre_a = 1  # Ctrl-C pendant la 2e phrase
    with pytest.raises(KeyboardInterrupt):
        moteur.generer(PHRASES, "fr_female", tmp_path)

    # Le manifeste a quand même été écrit, avec le seul fichier présent.
    manifest = lire_manifest(tmp_path)
    assert manifest["voix"] == "fr_female"
    assert len(manifest["fichiers"]) == 1

    moteur.interrompre_a = None
    bilan = moteur.generer(PHRASES, "fr_female", tmp_path)
    assert (bilan.generes, bilan.conserves) == (2, 1)
    assert len(lire_manifest(tmp_path)["fichiers"]) == 3


def test_unreadable_phrase_is_recorded_and_does_not_stop_the_run(
    tmp_path: Path, moteur: FauxMoteur
):
    """Une génération partie en boucle (erreur du moteur) ne doit pas bloquer
    les mille autres phrases : elle est notée, le manifeste ne la liste pas,
    et la relance suivante la retente."""
    moteur.illisibles = {adapter_oral(PHRASES[1].texte)}
    bilan = moteur.generer(PHRASES, "fr_female", tmp_path)
    assert (bilan.generes, bilan.conserves) == (2, 0)
    assert [(i, t) for i, t, _ in bilan.echecs] == [(PHRASES[1].id, PHRASES[1].texte)]
    assert PHRASES[1].id not in lire_manifest(tmp_path)["fichiers"]

    moteur.illisibles = set()
    bilan = moteur.generer(PHRASES, "fr_female", tmp_path)
    assert (bilan.generes, bilan.conserves, bilan.echecs) == (1, 2, [])


def test_leftover_tmp_files_are_discarded(tmp_path: Path, moteur: FauxMoteur):
    audio = tmp_path / "audio"
    audio.mkdir(parents=True)
    (audio / f"{PHRASES[0].id}.mp3.tmp").write_bytes(b"partiel")
    moteur.generer(PHRASES, "fr_female", tmp_path)
    assert list(audio.glob("*.tmp")) == []
    assert len(moteur.lus) == 3


def test_manifest_lists_only_present_files(tmp_path: Path):
    audio = tmp_path / "audio"
    audio.mkdir(parents=True)
    (audio / "bbbb.mp3").write_bytes(b"")
    (audio / "aaaa.mp3").write_bytes(b"")
    (audio / "cccc.mp3.tmp").write_bytes(b"")
    (audio / "notes.txt").write_text("")
    assert vc.ecrire_manifest(tmp_path, "fr_male") == 2
    assert lire_manifest(tmp_path) == {"voix": "fr_male", "fichiers": ["aaaa", "bbbb"]}


def test_corrupt_manifest_is_treated_as_empty(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{pas du json", encoding="utf-8")
    assert lire_manifest(tmp_path) == {"voix": "", "fichiers": []}


# ---- Essai des voix ----


def test_essayer_voix_writes_three_files_per_voice(tmp_path: Path, moteur: FauxMoteur):
    sorties = essayer_voix(
        tmp_path, ["fr_female", "fr_male"], lire=moteur.lire, vers_mp3=moteur.vers_mp3
    )
    assert [p.name for p in sorties] == [
        "fr_female-1.mp3", "fr_female-2.mp3", "fr_female-3.mp3",
        "fr_male-1.mp3", "fr_male-2.mp3", "fr_male-3.mp3",
    ]
    assert all(p.exists() for p in sorties)
    assert all(p.parent == tmp_path / "essais" for p in sorties)
    # Les phrases d'essai sont elles aussi adaptées à l'oral.
    assert ("Le segment C D mesure 6 centimètres et 4 millimètres.", "fr_male") in moteur.lus


# ---- Commande ----


@pytest.fixture
def commande(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """main() sans ffmpeg, sans réseau, sans Finder."""
    monkeypatch.setattr(vc, "assurer_ffmpeg", lambda: None)
    monkeypatch.setattr(vc.synthese, "voix_disponibles", lambda repo=None: ["fr_female", "fr_male", "de_male"])
    monkeypatch.setattr(vc.synthese, "voix_francaises", lambda repo=None: ["fr_female", "fr_male"])
    monkeypatch.setattr(vc, "_ouvrir_dans_finder", lambda chemin: None)
    appels: dict = {}
    monkeypatch.setattr(
        vc, "essayer_voix",
        lambda dossier, voix: appels.setdefault("essai", (dossier, list(voix))),
    )
    monkeypatch.setattr(vc, "recuperer_phrases", lambda dossier: list(PHRASES))

    def fake_generer(phrases, voix, dossier):
        appels["generer"] = (list(phrases), voix)
        return vc.Bilan(3, 3, 0, 0, False, 1.0, echecs=appels.get("echecs", []))

    monkeypatch.setattr(vc, "generer", fake_generer)
    monkeypatch.setattr(vc, "publier", lambda dossier: appels.setdefault("publie", True))
    monkeypatch.setattr(vc, "verifier_publication", lambda n: True)
    appels["dossier"] = tmp_path
    return appels


def test_main_essai_tries_french_voices_then_stops(commande: dict):
    code = vc.main(["--essai", "--dossier", str(commande["dossier"])])
    assert code == 0
    assert commande["essai"] == (commande["dossier"], ["fr_female", "fr_male"])
    assert "generer" not in commande and "publie" not in commande


def test_main_requires_a_voice(commande: dict):
    with pytest.raises(SystemExit):
        vc.main(["--dossier", str(commande["dossier"])])


def test_main_rejects_unknown_voice(commande: dict, capsys):
    assert vc.main(["--voix", "robot", "--dossier", str(commande["dossier"])]) == 1
    assert "Voix inconnue" in capsys.readouterr().err
    assert "generer" not in commande


def test_main_generates_then_publishes(commande: dict):
    assert vc.main(["--voix", "fr_male", "--dossier", str(commande["dossier"])]) == 0
    assert commande["generer"] == (list(PHRASES), "fr_male")
    assert commande["publie"] is True


def test_main_reports_unread_phrases_and_fails(commande: dict, capsys):
    commande["echecs"] = [("abcd", "Phrase impossible", "n'arrive pas à finir")]
    code = vc.main(["--voix", "fr_male", "--dossier", str(commande["dossier"])])
    assert code == 1
    assert "Phrase impossible" in capsys.readouterr().out
    assert commande["publie"] is True  # ce qui existe est publié quand même


def test_main_sans_publication_skips_vercel(commande: dict):
    code = vc.main(
        ["--voix", "fr_male", "--sans-publication", "--dossier", str(commande["dossier"])]
    )
    assert code == 0
    assert "generer" in commande and "publie" not in commande
