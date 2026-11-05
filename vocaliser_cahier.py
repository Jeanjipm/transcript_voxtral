"""
Voix du cahier de révisions CE1 : toutes ses phrases en MP3, publiées sur Vercel.

    python vocaliser_cahier.py --essai                 # écouter les voix, choisir
    python vocaliser_cahier.py --voix fr_female        # tout générer puis publier
    python vocaliser_cahier.py --voix fr_female --sans-publication
    python vocaliser_cahier.py --voix fr_female --tout   # après un changement du moteur

Comment ça s'emboîte :
  1. Le site (https://revisions-ce1-ki-xi-t.vercel.app) publie un petit script
     Node, `voix/phrases.mjs`, qui charge l'application elle-même et en
     extrait toutes les phrases qu'elle lit à voix haute. C'est lui qui fait
     autorité : la liste est toujours celle que la tablette demande.
  2. Chaque phrase a un identifiant : les 16 premiers caractères du SHA-1
     (UTF-8) de son texte. L'application calcule le même de son côté et
     cherche `audio/<id>.mp3` : le nom du fichier doit donc être exactement
     cet identifiant, calculé sur le texte D'ORIGINE — l'adaptation à l'oral
     (« = » lu « égale ») se fait sur une copie.
  3. Les MP3 vont dans ~/Documents/Voxtral/voix-ce1/, avec un `manifest.json`
     qui liste les fichiers vraiment présents ; l'application le lit pour
     savoir quelles phrases ont une voix.
  4. Le dossier est déployé tel quel sur le projet Vercel `voix-ce1`.

La génération est incrémentale : un MP3 déjà là est gardé, un identifiant
disparu de la liste part à la corbeille, et changer de voix régénère tout.
Chaque MP3 est écrit de façon atomique : on peut interrompre (Ctrl-C) et
relancer, ça reprend où ça s'est arrêté.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import synthese


SITE = "https://revisions-ce1-ki-xi-t.vercel.app/"
URL_PHRASES = SITE + "voix/phrases.mjs"
URL_PUBLIC = "https://voix-ce1-ki-xi-t.vercel.app/"
DOSSIER = Path("~/Documents/Voxtral/voix-ce1").expanduser()
PROJET_VERCEL = "voix-ce1"
EQUIPE_VERCEL = "ki-xi-t"
NODE_MIN = 18
# Longueur de l'identifiant : 16 hexa, comme dans phrases.mjs.
ID_LEN = 16

# Repris tel quel de la demande : CORS ouvert (le site et les voix sont sur
# deux domaines), manifeste jamais mis en cache (il change à chaque
# génération), MP3 immuables (leur nom est le hash du texte).
VERCEL_JSON = {
    "headers": [
        {
            "source": "/(.*)",
            "headers": [{"key": "Access-Control-Allow-Origin", "value": "*"}],
        },
        {
            "source": "/manifest.json",
            "headers": [{"key": "Cache-Control", "value": "no-cache"}],
        },
        {
            "source": "/audio/(.*)",
            "headers": [
                {"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}
            ],
        },
    ]
}
INDEX_HTML = (
    "<!doctype html>\n<meta charset=\"utf-8\">\n"
    "<p>Fichiers audio du cahier de révisions CE1</p>\n"
)
# Ce qui ne doit pas partir sur Vercel : les essais de voix et la liste brute.
VERCELIGNORE = "essais/\nphrases.json\n"

# Trois phrases variées pour choisir la voix à l'oreille : une règle, une
# question avec ses choix, une phrase avec des nombres et des centimètres.
# Toutes tirées de la liste réelle du cahier.
PHRASES_ESSAI: tuple[str, ...] = (
    "La phrase interrogative sert à interroger des gens, à poser des questions.",
    "2 cm, c'est combien de millimètres ? 20, 2, 12",
    "Le segment [CD] mesure 6 cm et 4 mm.",
)


@dataclass(frozen=True)
class Phrase:
    id: str
    texte: str


class CahierError(RuntimeError):
    """Erreur destinée à l'utilisateur (message en clair, pas de trace)."""


# ----------------------------------------------------------------------
# Identifiants et phrases
# ----------------------------------------------------------------------


def identifiant(texte: str) -> str:
    """Les 16 premiers caractères hexa du SHA-1 UTF-8 du texte — exactement
    ce que calculent phrases.mjs et l'application."""
    return hashlib.sha1(texte.encode("utf-8")).hexdigest()[:ID_LEN]


def charger_phrases(data: dict) -> list[Phrase]:
    """Valide le JSON de phrases.mjs et le transforme en `Phrase`s.

    Refuse une liste dont un identifiant ne correspond pas au texte : ce
    serait un MP3 que l'application ne trouverait jamais.
    """
    brutes = data.get("phrases")
    if not isinstance(brutes, list) or not brutes:
        raise CahierError("La liste de phrases est vide ou mal formée.")
    phrases: list[Phrase] = []
    for item in brutes:
        texte = item["texte"]
        attendu = identifiant(texte)
        if item["id"] != attendu:
            raise CahierError(
                f"Identifiant inattendu pour « {texte[:40]} » : "
                f"{item['id']} au lieu de {attendu}."
            )
        phrases.append(Phrase(id=item["id"], texte=texte))
    return phrases


def verifier_node() -> None:
    node = shutil.which("node")
    if node is None:
        raise CahierError("Node.js introuvable. Installe-le : brew install node")
    out = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)
    m = re.match(r"v(\d+)", out.stdout.strip())
    if m is None or int(m.group(1)) < NODE_MIN:
        raise CahierError(
            f"Node.js {NODE_MIN} ou plus est nécessaire (trouvé : {out.stdout.strip() or '?'})."
        )


def recuperer_phrases(dossier: Path) -> list[Phrase]:
    """Télécharge phrases.mjs, l'exécute avec Node, garde `phrases.json`."""
    verifier_node()
    curl = subprocess.run(
        ["curl", "-fsSL", URL_PHRASES], capture_output=True, timeout=60, check=False
    )
    if curl.returncode != 0 or not curl.stdout:
        raise CahierError(
            f"Impossible de récupérer {URL_PHRASES} "
            f"({curl.stderr.decode('utf-8', 'replace').strip() or 'réponse vide'})."
        )
    node = subprocess.run(
        ["node", "--input-type=module", "-"],
        input=curl.stdout,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if node.returncode != 0:
        raise CahierError(
            "Le script phrases.mjs a échoué :\n"
            + node.stderr.decode("utf-8", "replace").strip()[-800:]
        )
    try:
        data = json.loads(node.stdout)
    except json.JSONDecodeError as exc:
        raise CahierError(f"phrases.mjs n'a pas rendu du JSON : {exc}") from exc
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "phrases.json").write_bytes(node.stdout)
    return charger_phrases(data)


# ----------------------------------------------------------------------
# Adaptation de l'écrit à l'oral
# ----------------------------------------------------------------------

_CROCHETS_SEGMENT = re.compile(r"\[([A-Z])([A-Z])\]")
_UNITE_AVEC_NOMBRE = re.compile(r"(\d+)\s*(cm|mm)\b")
_UNITE_SEULE = re.compile(r"\b(cm|mm)\b")
_UNITES = {"cm": "centimètre", "mm": "millimètre"}
# Signes isolés en tête, suivis d'espaces : « !  Le point d'exclamation »,
# « •  Le point ». Le point y figure aussi, pour les phrases à trous qui
# commencent par « … » (devenu « . ») : mesuré, un point en tête fait inventer
# des mots au modèle (« Vous avez de briller. Avions volent… »), alors que la
# phrase sans lui est lue proprement.
_SIGNE_EN_TETE = re.compile(r"^[!?,;:•.]+(\s+|$)")
_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("=", " égale "),
    ("+", " plus "),
    ("→", ", "),
    ("…", "."),
    ("<", " est plus petit que "),
    (">", " est plus grand que "),
    ("\n", ", "),
)


def _unite(nombre: str | None, unite: str) -> str:
    mot = _UNITES[unite]
    if nombre is not None and int(nombre) > 1:
        return f"{nombre} {mot}s"
    return f"{nombre} {mot}" if nombre is not None else mot


def adapter_oral(texte: str) -> str:
    """Rend un texte du cahier lisible par la synthèse vocale.

    Travaille sur une copie : l'identifiant du MP3 reste celui du texte
    d'origine. Règles, dans l'ordre :
      - « [AB] » (deux majuscules) → « A B » ; les crochets phonétiques
        « [an] », « [è] » restent tels quels ;
      - « 6 cm » → « 6 centimètres », « 1 mm » → « 1 millimètre » ; « cm »
        seul → « centimètre » ;
      - « = » égale, « + » plus, « → » virgule, « … » point, « < » / « > »
        est plus petit / plus grand que, saut de ligne → virgule ;
      - un signe de ponctuation isolé en tête de phrase disparaît ;
      - espaces et ponctuations doublées nettoyés.
    """
    t = _CROCHETS_SEGMENT.sub(r"\1 \2", texte)
    t = _UNITE_AVEC_NOMBRE.sub(lambda m: _unite(m.group(1), m.group(2)), t)
    t = _UNITE_SEULE.sub(lambda m: _unite(None, m.group(1)), t)
    for signe, mot in _SUBSTITUTIONS:
        t = t.replace(signe, mot)
    t = _SIGNE_EN_TETE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    # « égale . » → « égale. » ; « ,, » ou « . , » → un seul signe.
    t = re.sub(r"\s+([,.])", r"\1", t)
    t = re.sub(r"([,.])(?:\s*[,.])+", r"\1", t)
    return t


# ----------------------------------------------------------------------
# Dossier de sortie
# ----------------------------------------------------------------------


def dossier_audio(dossier: Path) -> Path:
    return dossier / "audio"


def lire_manifest(dossier: Path) -> dict:
    chemin = dossier / "manifest.json"
    if not chemin.exists():
        return {"voix": "", "fichiers": []}
    try:
        data = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"voix": "", "fichiers": []}
    return {"voix": data.get("voix", ""), "fichiers": list(data.get("fichiers", []))}


def ecrire_manifest(dossier: Path, voix: str) -> int:
    """Écrit manifest.json à partir des MP3 réellement présents. Retourne
    leur nombre."""
    fichiers = sorted(p.stem for p in dossier_audio(dossier).glob("*.mp3"))
    data = {"voix": voix, "fichiers": fichiers}
    (dossier / "manifest.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return len(fichiers)


def ecrire_fichiers_statiques(dossier: Path) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (dossier / "vercel.json").write_text(
        json.dumps(VERCEL_JSON, indent=2) + "\n", encoding="utf-8"
    )
    (dossier / ".vercelignore").write_text(VERCELIGNORE, encoding="utf-8")


def mettre_a_la_corbeille(chemins: Sequence[Path]) -> None:
    """Envoie des fichiers à la corbeille macOS (récupérables), jamais `rm`.

    Via NSFileManager (pyobjc, déjà présent pour l'app) ; repli sur le Finder
    par AppleScript si pyobjc manque.
    """
    chemins = [Path(c) for c in chemins if Path(c).exists()]
    if not chemins:
        return
    try:
        from Foundation import NSURL, NSFileManager  # type: ignore[import-not-found]

        fm = NSFileManager.defaultManager()
        for chemin in chemins:
            ok, _, err = fm.trashItemAtURL_resultingItemURL_error_(
                NSURL.fileURLWithPath_(str(chemin)), None, None
            )
            if not ok:
                raise CahierError(f"Corbeille refusée pour {chemin.name} : {err}")
        return
    except ImportError:
        pass
    liste = ", ".join(f'POSIX file "{c}"' for c in chemins)
    script = f'tell application "Finder" to delete {{{liste}}}'
    result = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise CahierError(f"Corbeille refusée : {result.stderr.strip()}")


# ----------------------------------------------------------------------
# Génération
# ----------------------------------------------------------------------

Lecteur = Callable[[str, str], np.ndarray]
Encodeur = Callable[[np.ndarray, Path], None]
Corbeille = Callable[[Sequence[Path]], None]


@dataclass
class Bilan:
    total: int
    generes: int
    conserves: int
    supprimes: int
    tout_regenere: bool
    duree_s: float
    # (id, texte, erreur) des phrases que le modèle n'a pas su lire.
    echecs: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def manquants(self) -> int:
        return self.total - self.generes - self.conserves


def _format_duree(secondes: float) -> str:
    secondes = max(0, int(round(secondes)))
    if secondes < 60:
        return f"{secondes} s"
    minutes, s = divmod(secondes, 60)
    if minutes < 60:
        return f"{minutes} min {s:02d} s"
    heures, m = divmod(minutes, 60)
    return f"{heures} h {m:02d} min"


def _afficher_progression(fait: int, total: int, debut: float) -> None:
    ecoule = time.monotonic() - debut
    restant = (ecoule / fait) * (total - fait) if fait else 0.0
    pct = int(100 * fait / total) if total else 100
    sys.stdout.write(
        f"\r  {fait}/{total}  {pct:3d} %   écoulé {_format_duree(ecoule)}"
        f"   reste ~{_format_duree(restant)}      "
    )
    sys.stdout.flush()


def generer(
    phrases: Sequence[Phrase],
    voix: str,
    dossier: Path,
    lire: Lecteur = synthese.lire,
    vers_mp3: Encodeur = synthese.vers_mp3,
    corbeille: Corbeille = mettre_a_la_corbeille,
    afficher: bool = True,
    tout: bool = False,
) -> Bilan:
    """Génère les MP3 manquants, de façon incrémentale et interruptible.

    - un MP3 déjà présent est conservé ;
    - si la voix du manifeste diffère de `voix`, ou si `tout` est demandé
      (le moteur a changé), tout est régénéré (les anciens MP3 partent à la
      corbeille) ;
    - un MP3 dont l'identifiant a disparu de la liste part à la corbeille ;
    - le manifeste est réécrit à la fin, même après une interruption, et ne
      liste que les fichiers présents.

    `lire`, `vers_mp3` et `corbeille` sont injectables pour les tests.
    """
    audio_dir = dossier_audio(dossier)
    audio_dir.mkdir(parents=True, exist_ok=True)
    debut = time.monotonic()

    # Restes d'une interruption en plein encodage : jamais pris pour finis.
    for tmp in audio_dir.glob("*.tmp"):
        tmp.unlink(missing_ok=True)

    manifest = lire_manifest(dossier)
    presents = {p.stem: p for p in audio_dir.glob("*.mp3")}
    tout_regenere = bool(presents) and (
        tout or (bool(manifest["voix"]) and manifest["voix"] != voix)
    )
    if tout_regenere:
        corbeille(list(presents.values()))
        presents = {}

    ids = {p.id for p in phrases}
    orphelins = [chemin for id_, chemin in presents.items() if id_ not in ids]
    if orphelins:
        corbeille(orphelins)
        for chemin in orphelins:
            presents.pop(chemin.stem, None)

    a_faire = [p for p in phrases if p.id not in presents]
    conserves = len(phrases) - len(a_faire)
    generes = 0
    echecs: list[tuple[str, str, str]] = []
    try:
        if afficher and a_faire:
            print(f"  {conserves} MP3 déjà présents, {len(a_faire)} à générer.")
        for i, phrase in enumerate(a_faire, start=1):
            try:
                audio = lire(adapter_oral(phrase.texte), voix)
                vers_mp3(audio, audio_dir / f"{phrase.id}.mp3")
            except synthese.SynthesisError as exc:
                # Une phrase que le modèle ne sait pas finir ne doit pas
                # bloquer les mille autres : on la note et on continue.
                echecs.append((phrase.id, phrase.texte, str(exc)))
                if afficher:
                    print(f"\n  ✗ {phrase.texte[:70]} — {exc}")
            else:
                generes += 1
            if afficher:
                _afficher_progression(i, len(a_faire), debut)
    finally:
        if afficher and a_faire:
            print()
        ecrire_manifest(dossier, voix)
        ecrire_fichiers_statiques(dossier)

    return Bilan(
        total=len(phrases),
        generes=generes,
        conserves=conserves,
        supprimes=len(orphelins),
        tout_regenere=tout_regenere,
        duree_s=time.monotonic() - debut,
        echecs=echecs,
    )


# ----------------------------------------------------------------------
# Essai des voix
# ----------------------------------------------------------------------


def essayer_voix(
    dossier: Path,
    voix: Sequence[str],
    lire: Lecteur = synthese.lire,
    vers_mp3: Encodeur = synthese.vers_mp3,
) -> list[Path]:
    """Lit les trois phrases d'essai avec chaque voix, dans `essais/`."""
    essais = dossier / "essais"
    essais.mkdir(parents=True, exist_ok=True)
    sorties: list[Path] = []
    for v in voix:
        for i, texte in enumerate(PHRASES_ESSAI, start=1):
            chemin = essais / f"{v}-{i}.mp3"
            print(f"  {v} · {i} : {adapter_oral(texte)}")
            vers_mp3(lire(adapter_oral(texte), v), chemin)
            sorties.append(chemin)
    return sorties


# ----------------------------------------------------------------------
# Publication
# ----------------------------------------------------------------------


def commande_vercel() -> list[str]:
    return ["vercel"] if shutil.which("vercel") else ["npx", "vercel"]


def publier(dossier: Path) -> None:
    """`vercel link` la première fois, puis `vercel deploy --prod`."""
    base = commande_vercel()
    if not (dossier / ".vercel" / "project.json").exists():
        print("→ Liaison au projet Vercel (première fois)…")
        link = subprocess.run(
            base + ["link", "--yes", "--project", PROJET_VERCEL, "--scope", EQUIPE_VERCEL],
            cwd=dossier,
            check=False,
        )
        if link.returncode != 0:
            raise CahierError(
                "La liaison Vercel a échoué. Si la commande demande une connexion, "
                "lance d'abord : npx vercel login"
            )
    print("→ Déploiement en production…")
    deploy = subprocess.run(
        base + ["deploy", "--prod", "--yes", "--scope", EQUIPE_VERCEL],
        cwd=dossier,
        check=False,
    )
    if deploy.returncode != 0:
        raise CahierError(
            "Le déploiement Vercel a échoué. Si la commande demande une connexion, "
            "lance d'abord : npx vercel login"
        )


def manifeste_en_ligne() -> dict:
    url = URL_PUBLIC + f"manifest.json?t={int(time.time())}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verifier_publication(attendu: int, essais: int = 6, delai_s: float = 5.0) -> bool:
    """Vérifie que le manifeste en ligne annonce `attendu` fichiers (le CDN
    peut mettre quelques secondes à servir la nouvelle version)."""
    for _ in range(essais):
        try:
            n = len(manifeste_en_ligne().get("fichiers", []))
        except Exception as exc:  # noqa: BLE001
            print(f"  (manifeste en ligne pas encore lisible : {exc})")
            n = -1
        if n == attendu:
            print(f"✓ {URL_PUBLIC}manifest.json annonce {n} fichiers.")
            return True
        time.sleep(delai_s)
    print(f"✗ Le manifeste en ligne annonce {n} fichiers, {attendu} attendus.")
    return False


# ----------------------------------------------------------------------
# Commande
# ----------------------------------------------------------------------


def assurer_ffmpeg() -> None:
    if synthese.ffmpeg_disponible():
        return
    if shutil.which("brew") is None:
        raise CahierError(
            "ffmpeg manque et Homebrew n'est pas installé. "
            "Installe Homebrew (https://brew.sh) puis : brew install ffmpeg"
        )
    print("→ ffmpeg manque, installation via Homebrew…")
    result = subprocess.run(["brew", "install", "ffmpeg"], check=False)
    if result.returncode != 0 or not synthese.ffmpeg_disponible():
        raise CahierError("L'installation de ffmpeg a échoué.")


def _ouvrir_dans_finder(chemin: Path) -> None:
    try:
        subprocess.run(["open", str(chemin)], check=False)
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Génère et publie les voix du cahier de révisions CE1."
    )
    parser.add_argument(
        "--essai", action="store_true",
        help="lit trois phrases avec chaque voix française dans essais/, puis s'arrête",
    )
    parser.add_argument("--voix", help="nom exact de la voix (ex. fr_female, fr_male)")
    parser.add_argument(
        "--sans-publication", action="store_true",
        help="génère sans déployer sur Vercel",
    )
    parser.add_argument(
        "--tout", action="store_true",
        help="régénère aussi les MP3 déjà présents (après un changement du moteur)",
    )
    parser.add_argument(
        "--dossier", type=Path, default=DOSSIER, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    dossier: Path = args.dossier

    try:
        assurer_ffmpeg()
        toutes = synthese.voix_disponibles()
        francaises = synthese.voix_francaises()
        print(f"Voix du modèle ({len(toutes)}) : {', '.join(toutes)}")
        print(f"Voix françaises : {', '.join(francaises) or 'aucune'}")

        if args.essai:
            if not francaises:
                raise CahierError("Aucune voix française trouvée dans le modèle.")
            print(f"\nEssai des voix dans {dossier / 'essais'} :")
            essayer_voix(dossier, francaises)
            print("\nÉcoute les fichiers, puis relance avec --voix <nom>.")
            _ouvrir_dans_finder(dossier / "essais")
            return 0

        if not args.voix:
            parser.error("--voix NOM est obligatoire (ou --essai pour choisir).")
        if args.voix not in toutes:
            raise CahierError(
                f"Voix inconnue « {args.voix} ». Choisis parmi : {', '.join(toutes)}"
            )

        print("\n→ Récupération des phrases du cahier…")
        phrases = recuperer_phrases(dossier)
        print(f"  {len(phrases)} phrases.")

        print(f"\n→ Génération avec la voix {args.voix} dans {dossier} :")
        bilan = generer(phrases, args.voix, dossier, tout=args.tout)
        if bilan.tout_regenere:
            print("  (tout régénéré : les anciens MP3 sont partis à la corbeille)")
        print(
            f"  {bilan.generes} générés, {bilan.conserves} conservés, "
            f"{bilan.supprimes} orphelins à la corbeille, en {_format_duree(bilan.duree_s)}."
        )
        n = len(lire_manifest(dossier)["fichiers"])
        print(f"  manifest.json : {n} fichiers.")
        if bilan.echecs:
            print(f"\n  {len(bilan.echecs)} phrase(s) non lues (relance la commande pour réessayer) :")
            for _, texte, erreur in bilan.echecs:
                print(f"    - {texte[:80]} — {erreur}")

        if args.sans_publication:
            print("\nPublication sautée (--sans-publication).")
            return 1 if bilan.echecs else 0
        print()
        publier(dossier)
        ok = verifier_publication(n)
        return 0 if ok and not bilan.echecs else 1
    except KeyboardInterrupt:
        print("\nInterrompu. Relance la même commande pour reprendre.")
        return 130
    except (CahierError, synthese.SynthesisError) as exc:
        print(f"\nErreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
