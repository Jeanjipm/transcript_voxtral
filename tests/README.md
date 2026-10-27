# Tests unitaires Voxtral

Suite de tests pytest qui couvre la logique métier des modules
principaux. Les modules natifs macOS (AppKit, sounddevice, mlx_voxtral,
pynput, rumps) sont mockés dans `conftest.py` pour que la suite tourne
sur n'importe quelle machine — y compris en CI Linux.

## Lancer les tests

Depuis la racine du projet :

```bash
# 1. Créer un venv dédié pour les tests (une fois)
python3 -m venv .venv-tests
source .venv-tests/bin/activate
pip install pytest pytest-mock pyyaml numpy soundfile

# 2. Lancer la suite
.venv-tests/bin/python -m pytest tests/ -v
```

Si tu veux juste un module :

```bash
.venv-tests/bin/python -m pytest tests/test_audio_capture.py -v
```

## Lint et intégration continue

`ruff` (configuration dans `ruff.toml`, règles « bugs » uniquement, pas de
style) :

```bash
pip install ruff
ruff check .
```

Le workflow `.github/workflows/tests.yml` lance ruff puis pytest à chaque
push et à chaque PR, sur Ubuntu en Python 3.12 et 3.13 — possible parce que
tout le natif macOS est remplacé par des stubs. Une PR avec une suite rouge
ne se fusionne pas.

## Couverture

| Module | Tests | Couvert |
|---|---:|---|
| `hotkey_manager.py` | 68 | parse, validate, state machine push-to-talk, pause/reprise, resynchronisation limitée aux touches du raccourci |
| `dictation_controller.py` | 37 | machine à états, coupe-circuit de durée, dictée pendant une transcription, arrêt infaillible du worker |
| `file_transcriber.py` | 31 | boucle long-form, garde-fous d'avance, mise en forme du .txt |
| `audio_capture.py` | 31 | cycle stream long-lived, prewarm, reprise après échec, verrous |
| `model_manager.py` | 31 | catalogue, usages dictée/fichiers, scan et purge du cache, modèles requis par la traduction |
| `hotkey_capture.py` | 27 | enregistrement d'un raccourci au clavier, machine à états pure, file d'événements |
| `diarizer.py` | 27 | tours de parole depuis les probabilités, lissage, renumérotation, fusion locuteurs × segments |
| `updater.py` | 26 | check + apply via API GitHub sur la branche installée, comportement offline |
| `transcriber.py` | 20 | factory, délégation translate, garde-fou turbo, réseau ouvert si le modèle manque |
| `hf_offline.py` | 19 | détection du cache, bascule hors-ligne, résolution locale, `network_if_missing` |
| `config.py` | 18 | I/O, fusion defaults+user, dataclasses, roundtrip |
| `file_job.py` | 18 | cycle de vie, chemin de sortie, annulation, erreurs |
| `audio_convert.py` | 17 | afconvert, formats refusés par libsndfile, lecture par blocs |
| `clipboard.py` | 15 | paste_text, préservation clipboard, règle de thread |
| `inference_worker.py` | 13 | priorité, exclusion mutuelle, robustesse |

**Total : 416 tests** (certains sont paramétrés, d'où un total supérieur à la
somme des fonctions de test).

Beaucoup de ces tests verrouillent une régression précise, avec la mesure qui
l'a révélée en docstring — par exemple « `on_hotkey_start` ne touche le
recorder zéro fois de façon synchrone » (le callback de l'event tap macOS a
environ 1 seconde de budget) ou « l'avance est plafonnée à l'audio réellement
fourni » (Whisper a annoncé une fin de segment à 51,6 s dans un bloc de 30 s).
Ne pas les alléger sans lire le pourquoi.

## Ce qui n'est PAS couvert

- `app.py` — trop lié à rumps/Cocoa pour des tests unitaires simples.
  Testé manuellement via le protocole de sprint. À noter : rumps indexe les
  items de menu par leur TITRE, donc deux items partageant un titre de repos
  s'écrasent silencieusement — vérifier le menu à l'œil après ajout.
- `audio_feedback.py` — dépend de NSSound, idem.
- `file_picker.py` — NSOpenPanel, main thread uniquement, testé à la main.
- `settings_ui.py` — UI tkinter, testée à la main. La logique testable en
  a été sortie : conversion libellé ↔ code technique et calcul de
  progression vivent dans `model_manager`, qui est couvert.
- Comportements vraiment system : capture micro réel, paste Cmd+V,
  permissions TCC. Ne se mockent pas — testés à la main sur Mac.
