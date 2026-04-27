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
| `config.py` | 18 | I/O, fusion defaults+user, dataclasses, roundtrip |
| `updater.py` | 21 | check + apply via API GitHub, comportement offline |
| `audio_capture.py` | 20 | cycle stream long-lived, prewarm, callback |
| `hotkey_manager.py` | 30 | parse, validate, state machine push-to-talk |
| `transcriber.py` | 12 | factory, delegation translate, preload |
| `clipboard.py` | 10 | paste_text, préservation clipboard |

**Total : 111 tests**

## Ce qui n'est PAS couvert

- `app.py` — trop lié à rumps/Cocoa pour des tests unitaires simples.
  Testé manuellement via le protocole de sprint.
- `audio_feedback.py` — dépend de NSSound, idem.
- `model_manager.py` — dépend de huggingface_hub, à mocker plus finement.
- `settings_ui.py` — UI tkinter, testée à la main.
- Comportements vraiment system : capture micro réel, paste Cmd+V,
  permissions TCC. Ne se mockent pas — testés à la main sur Mac.
