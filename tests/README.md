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

## Couverture

| Module | Tests | Couvert |
|---|---:|---|
| `hotkey_manager.py` | 38 | parse, validate, state machine push-to-talk, réarmement du tap |
| `file_transcriber.py` | 33 | boucle long-form, garde-fous d'avance, mise en forme du .txt |
| `dictation_controller.py` | 27 | machine à états, coupe-circuit de durée, robustesse du worker |
| `audio_capture.py` | 28 | cycle stream long-lived, prewarm, reprise après échec, verrous |
| `updater.py` | 21 | check + apply via API GitHub, comportement offline |
| `config.py` | 18 | I/O, fusion defaults+user, dataclasses, roundtrip |
| `file_job.py` | 18 | cycle de vie, chemin de sortie, annulation, erreurs |
| `audio_convert.py` | 17 | afconvert, formats refusés par libsndfile, lecture par blocs |
| `hf_offline.py` | 17 | détection du cache, bascule hors-ligne, résolution locale |
| `clipboard.py` | 15 | paste_text, préservation clipboard, règle de thread |
| `inference_worker.py` | 13 | priorité, exclusion mutuelle, robustesse |
| `transcriber.py` | 12 | factory, delegation translate, preload |

**Total : 257 tests**

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
- `model_manager.py` — dépend de huggingface_hub, à mocker plus finement.
- `settings_ui.py` — UI tkinter, testée à la main.
- Comportements vraiment system : capture micro réel, paste Cmd+V,
  permissions TCC. Ne se mockent pas — testés à la main sur Mac.
