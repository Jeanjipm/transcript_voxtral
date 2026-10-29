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
| `dictation_controller.py` | 49 | machine à états, coupe-circuit de durée, dictée pendant une transcription, arrêt infaillible du worker, filtre anti-silence, changement de micro (immédiat ou différé) |
| `app.py` | 47 | construction et réglages transmis, hot-reload de la config (raccourci, modèle, langue, sons, YAML invalide), transcriber des fichiers (réutilisation, cache, turbo → large-v3), fin et lancement d'un job fichier, erreurs visibles et flash, états de la dictée dans la menu bar, transcription + collage + nettoyage, suivi du micro, mises à jour, quitter |
| `file_transcriber.py` | 36 | boucle long-form, garde-fous d'avance, mise en forme du .txt |
| `audio_capture.py` | 35 | cycle stream long-lived, prewarm, reprise après échec, verrous, `reset_device` |
| `model_manager.py` | 34 | catalogue, usages dictée/fichiers, scan et purge du cache, modèles requis par la traduction |
| `hotkey_capture.py` | 33 | enregistrement d'un raccourci au clavier, machine à états pure, file d'événements |
| `diarizer.py` | 27 | tours de parole depuis les probabilités, lissage, renumérotation, fusion locuteurs × segments |
| `updater.py` | 26 | check + apply via API GitHub sur la branche installée, fichiers risqués (install.sh, requirements.txt, requirements.lock), comportement offline |
| `transcriber.py` | 26 | factory, délégation translate, garde-fou turbo, réseau ouvert si le modèle manque, budget de jetons proportionnel à la durée |
| `config.py` | 22 | I/O, fusion defaults+user, dataclasses, roundtrip |
| `audio_level.py` | 20 | RMS de la fenêtre la plus forte (dBFS), normalisation int16/float/stéréo, salve brève dans un long silence, lecture d'un WAV, décision avec filtre coupé |
| `hf_offline.py` | 19 | détection du cache, bascule hors-ligne, résolution locale, `network_if_missing` |
| `file_job.py` | 18 | cycle de vie, chemin de sortie, annulation, erreurs |
| `audio_convert.py` | 17 | afconvert, formats refusés par libsndfile, lecture par blocs |
| `clipboard.py` | 15 | paste_text, préservation clipboard, règle de thread |
| `inference_worker.py` | 13 | priorité, exclusion mutuelle, robustesse |
| `audio_device.py` | 9 | détection d'un changement de micro par défaut : premier sondage muet, lecture échouée ignorée, exceptions avalées |

**Total : 514 tests** (certains sont paramétrés, d'où un total supérieur à la
somme des fonctions de test).

Beaucoup de ces tests verrouillent une régression précise, avec la mesure qui
l'a révélée en docstring — par exemple « `on_hotkey_start` ne touche le
recorder zéro fois de façon synchrone » (le callback de l'event tap macOS a
environ 1 seconde de budget) ou « l'avance est plafonnée à l'audio réellement
fourni » (Whisper a annoncé une fin de segment à 51,6 s dans un bloc de 30 s).
Ne pas les alléger sans lire le pourquoi.

## Ce qui n'est PAS couvert

- Dans `app.py` : le rendu lui-même. `rumps` est remplacé par
  `tests/rumps_stub.py` (vraies classes `App`, `MenuItem`, `Timer`, mais
  inertes), donc les tests vérifient ce que l'app DÉCIDE (titres, callbacks,
  composants reconstruits), pas ce que macOS affiche. Restent manuels :
  l'icône `NSStatusItem` (`_set_status_icon`), les fenêtres modales, la
  fenêtre Préférences en sous-processus, le thread de vérification des
  mises à jour. À noter : le vrai rumps indexe les items de menu par leur
  TITRE, donc deux items partageant un titre de repos s'écrasent
  silencieusement — le stub ne le détecte pas, vérifier le menu à l'œil
  après ajout.
- `audio_device.py` : la lecture CoreAudio via `ctypes` elle-même (testée à
  la main : `python -c "import audio_device; print(audio_device.default_input_device())"`).
- `audio_feedback.py` — dépend de NSSound.
- `file_picker.py` — NSOpenPanel, main thread uniquement, testé à la main.
- `settings_ui.py` — UI tkinter, testée à la main. La logique testable en
  a été sortie : conversion libellé ↔ code technique et calcul de
  progression vivent dans `model_manager`, qui est couvert.
- Comportements vraiment system : capture micro réel, paste Cmd+V,
  permissions TCC. Ne se mockent pas — testés à la main sur Mac.
