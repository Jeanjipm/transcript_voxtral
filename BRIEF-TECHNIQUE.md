# Brief technique — Voxtral Dictée

**Version :** 0.3 | **Date :** 4 novembre 2026

Ce document décrit le logiciel **tel qu'il est** après les sprints 6a, 6b et
6c (cf. `ANALYSE-2026-10-26.md` pour le diagnostic qui les a motivés). Les
versions 0.1 et 0.2 (avril 2026) étaient un cahier des charges ; ce qui n'a
finalement pas été construit est listé en fin de document.

## Résumé

App macOS menu bar de dictée vocale locale. L'utilisateur maintient un
raccourci clavier, parle, relâche, et le texte transcrit est collé à la
position du curseur. Tout tourne en local sur Apple Silicon via MLX. Aucune
donnée ne quitte la machine.

Distribuable à des amis via un script d'installation en une commande, et mise
à jour depuis l'app (menu « Mises à jour »).

## Cible utilisateur

- Utilisateurs macOS sur Apple Silicon (M1+), macOS 13+
- Non-développeurs (installation assistée par script, réglages sans jargon)
- Francophones principalement, multilingue (langues de Voxtral et de Whisper)

## Fonctionnalités

### F1 — Dictée par raccourci clavier

**Raccourci par défaut : `alt_r`, la touche ⌥ Option droite**, en
push-to-talk : maintenir = enregistrer, relâcher = transcrire. Une seule
touche tenue, sans combinaison, est ce qui s'est avéré le plus naturel.

- Configurable dans Préférences → Dictée → « Modifier… » : on appuie sur les
  touches voulues (`hotkey_capture.py`), le code (`alt+space`, `f13`,
  `cmd+shift+h`…) est écrit dans `config.yaml`.
- Écouteur global `pynput` non suppressif (`hotkey_manager.py`) : la touche
  garde sa fonction normale, donc ⌥ droite continue de produire les
  caractères spéciaux. Seules les touches du raccourci sont suivies.
- Le relâchement peut se perdre (macOS coupe parfois la surveillance clavier,
  notamment après une veille) : coupe-circuit après `recording.max_duration_s`
  (300 s), audio conservé dans `~/.voxtral/recordings/`, et menu « Réactiver
  le raccourci ».
- On enregistre `recording.tail_padding_ms` (350 ms) après le relâchement,
  sinon le dernier mot est coupé ; `silence_padding_ms` (250 ms) de silence
  est ajouté de chaque côté avant transcription.
- **Filtre anti-silence** (`audio_level.py`) : si le moment le plus fort de
  la dictée reste sous `recording.silence_threshold_dbfs` (−45 dBFS), rien
  n'est envoyé au modèle — qui inventerait sinon « Merci. ».
- Le micro par défaut de macOS est suivi (`audio_device.py`, CoreAudio via
  ctypes) : brancher des AirPods ou un casque reconstruit le flux micro.

**Feedback visuel :** l'icône de la menu bar (SF Symbols) passe de micro à
rond rouge (écoute), sablier animé (transcription), flèche (téléchargement
d'un modèle), document (fichier en cours), triangle (erreur).

### F2 — Sons d'activation / désactivation

Sons système macOS (`Tink` au démarrage de l'écoute, `Pop` à l'arrêt) joués
via `AppKit.NSSound` (`audio_feedback.py`). Volume 0–100 %, bouton
« Écouter » dans les Préférences, désactivables. Aucun fichier son embarqué.

### F3 — Transcription locale

**Dictée : `mzbac/voxtral-mini-3b-4bit-mixed`** (Voxtral Mini 3B quantifié
4 bits, 3,2 Go, package `mlx-voxtral`). Alternatives dans Préférences →
Modèles : 8 bits (5,3 Go), pleine précision (~8 Go), ou Whisper Large V3
(MIT, qui sait aussi traduire).

- Langue `auto` ou code ISO ; tâche `transcribe` ou `translate` (vers
  l'anglais). Voxtral et les Whisper « turbo » ne traduisent pas : la
  traduction est déléguée à Whisper Large V3, téléchargé à la demande
  (`transcriber.WHISPER_TRANSLATE_REPO`).
- Budget de jetons proportionnel à la durée de l'audio (8 jetons/s, plancher
  `transcription.max_new_tokens` = 1024) : une dictée de 5 minutes n'est plus
  tronquée (`transcriber.token_budget`).
- **Hors-ligne pour de vrai** (`hf_offline.py`) : quand le modèle est en
  cache, `huggingface_hub` est mis en mode hors-ligne et on passe à
  `from_pretrained` le chemin du snapshot local, pas l'identifiant du repo
  (contournement d'un appel réseau inconditionnel de `mistral_common`). Le
  réseau n'est rouvert que le temps de charger un modèle absent.
- Les modèles vivent dans le **cache HuggingFace** (`~/.cache/huggingface/hub`),
  le seul endroit que `from_pretrained` sait lire.

### F4 — Injection du texte

`clipboard.py` : le texte (suffixé d'un espace) est écrit dans le
presse-papier, puis `Cmd+V` est simulé via `pynput`. L'ancien contenu texte
du presse-papier est restauré après `ui.clipboard_restore_delay_ms` (300 ms,
réglable : macOS ne signale pas la lecture du presse-papier par l'app cible).
`ui.auto_paste: false` copie sans coller. Pas de notification : macOS n'affiche
pas les bannières d'une app non signée, le menu est le seul canal fiable.

### F5 — Transcription de fichiers audio

Menu « Transcrire un fichier audio… » (`file_picker.py`, `file_job.py`,
`file_transcriber.py`) : tout format lisible par macOS (m4a, mp3, wav, aiff,
caf, flac, piste d'un mp4) est converti en 16 kHz mono par `afconvert`
(`audio_convert.py`), puis transcrit par blocs de 300 s avec
`mlx-community/whisper-large-v3-turbo`, qui fournit des horodatages. Un `.txt`
avec en-tête et paragraphes `[hh:mm:ss]` est écrit dans `~/Documents/Voxtral`
et révélé dans le Finder. Progression dans le menu, annulation possible (le
texte déjà produit est conservé), la dictée garde la priorité.

Option « Identifier qui parle » (`diarizer.py`) : Sortformer v2.1 via
`mlx-audio`, jusqu'à 4 locuteurs, 225 Mo téléchargés au premier usage.

### F6 — Préférences (fenêtre tkinter)

`settings_ui.py`, lancée en **sous-processus** depuis l'app (tkinter et le
run loop Cocoa de rumps ne cohabitent pas dans un même process). Elle écrit
`~/.voxtral/config.yaml` ; l'app relit le fichier toutes les 2 s
(hot-reload) et applique sans redémarrer. Le raccourci global est suspendu
tant que la fenêtre vit.

Onglets, organisés par tâche et sans code technique affiché :

- **Dictée** : raccourci (capture au clavier), langue, transcription ou
  traduction, coller ou seulement copier, sons et volume.
- **Fichiers** : dossier de sortie, horodatage, identification des locuteurs,
  durée maximale acceptée.
- **Modèles** : modèle de dictée, modèle fichiers, place occupée par chaque
  modèle en cache, suppression. Choisir un modèle absent propose son
  téléchargement avec progression ; refuser rétablit le réglage précédent.
- **Avancé** : fin de phrase conservée, durée maximale d'une dictée,
  longueur minimale du texte, seuil de silence, délai avant de rendre le
  presse-papier, vérification des mises à jour.
- **À propos**.

Fermer la fenêtre (croix, Cmd+W, bouton « Fermer », **Cmd+Q**) enregistre.

### F7 — Menu bar

Icône dans la barre de menu (`rumps`). Menu, dans l'ordre :

- État : prêt / écoute en cours… / transcription… / téléchargement…
- Raccourci : ⌥ Option droite
- Langue : Auto, 🇫🇷 Français…
- Modèle : nom court du modèle actif
- (ligne d'erreur, visible seulement après une erreur ; clic = détail)
- ---
- Transcrire un fichier audio… / Annuler la transcription
- ---
- Réactiver le raccourci
- Préférences…
- Mises à jour (« Mises à jour disponibles… » quand il y en a)
- ---
- À propos (affiche le commit installé et sa date)
- Quitter

### F8 — Mises à jour de l'app

`updater.py` : au démarrage (si `updates.auto_check`), un thread interroge
`api.github.com` pour comparer le commit installé à la tête de la **branche
installée** ; l'application se fait par `git pull --ff-only` dans
`~/.voxtral/app`. Si `install.sh`, `requirements.txt` ou `requirements.lock`
ont changé, l'app refuse la mise à jour automatique et demande de relancer
l'installation. C'est le seul appel réseau de l'app en dehors des
téléchargements de modèles, et il se désactive dans `config.yaml`.

### F9 — Entretien

`housekeeping.py`, au démarrage : `~/.voxtral/voxtral.log` est tronqué en
place au-delà de `maintenance.log_max_mb` (10 Mo, on garde le dernier Mo), et
les `dictee-*.wav` de `~/.voxtral/recordings/` plus vieux que
`maintenance.recordings_max_age_days` (30) sont supprimés.

## Architecture

```
transcript_voxtral/
├── app.py                   # Point d'entrée : menu bar rumps, orchestration, hot-reload
├── dictation_controller.py  # Machine à états de la dictée, seul pilote du micro (thread dictation-worker)
├── inference_worker.py      # Thread unique propriétaire des modèles MLX, file à priorités
├── hotkey_manager.py        # Écoute du raccourci global (pynput), push-to-talk
├── hotkey_capture.py        # Enregistrement d'un raccourci en appuyant sur les touches
├── audio_capture.py         # Capture micro (sounddevice), flux gardé chaud, reprise après échec
├── audio_device.py          # Micro par défaut de macOS (CoreAudio via ctypes) et détection de changement
├── audio_level.py           # Niveau sonore (RMS en dBFS) : filtre anti-silence
├── audio_feedback.py        # Sons système Tink / Pop (NSSound)
├── audio_convert.py         # Tout fichier audio → WAV 16 kHz mono (afconvert)
├── transcriber.py           # Voxtral et Whisper derrière une même interface, budget de jetons
├── hf_offline.py            # Mode hors-ligne HuggingFace et chemins de snapshot locaux
├── model_manager.py         # Catalogue, téléchargement, cache, modèles requis
├── file_picker.py           # NSOpenPanel et révélation dans le Finder
├── file_job.py              # Cycle de vie d'une transcription de fichier (thread file-job)
├── file_transcriber.py      # Boucle par blocs + mise en forme du .txt
├── diarizer.py              # Identification des locuteurs (Sortformer v2.1)
├── clipboard.py             # Presse-papier + Cmd+V simulé + restauration
├── config.py                # Dataclasses, fusion défauts + overrides, chemins de ~/.voxtral
├── housekeeping.py          # Journal borné, purge des vieilles dictées
├── updater.py               # Vérification et application des mises à jour (GitHub)
├── settings_ui.py           # Fenêtre Préférences (tkinter, sous-processus)
├── transcribe.py            # CLI debug : transcrit un fichier
├── download_model.py        # CLI : télécharge / met à jour un modèle
├── config.yaml              # Valeurs par défaut livrées (jamais modifiées par l'utilisateur)
├── install.sh               # Installation en une commande
├── requirements.txt         # Dépendances lisibles
├── requirements.lock        # Versions figées, installées par install.sh
├── ruff.toml                # Lint (règles « bugs » uniquement)
├── .github/workflows/tests.yml  # CI : ruff + pytest sur Python 3.12 et 3.13
├── tests/                   # 539 tests pytest, natif macOS remplacé par des stubs (cf. tests/README.md)
├── README.md                # Doc utilisateur
├── BRIEF-TECHNIQUE.md       # Ce document
└── ANALYSE-2026-10-26.md    # Revue complète du 26 octobre 2026 et plan des sprints 6a-6c
```

**Threads.** Le main thread appartient à Cocoa (rumps) et ne fait que de
l'affichage. `inference-worker` est le seul à toucher les modèles MLX
(préchargement, dictées, blocs de fichier, par priorité). `dictation-worker`
est le seul à toucher le micro ; le callback de l'event tap clavier ne fait
que déposer une commande dans sa file. `file-job` découpe le fichier et
attend chaque bloc. Les Préférences sont un process à part.

**Fichiers chez l'utilisateur.** `~/.voxtral/app` (le dépôt),
`~/.voxtral/venv`, `~/.voxtral/config.yaml` (overrides seulement),
`~/.voxtral/voxtral.log`, `~/.voxtral/recordings/`, `~/Documents/Voxtral`
(transcriptions), `~/.cache/huggingface/hub` (modèles),
`~/Desktop/Voxtral.command`, `~/Library/LaunchAgents/com.voxtral.dictee.plist`
(optionnel).

## Configuration (config.yaml)

Le `config.yaml` livré contient les défauts et leurs explications ;
`~/.voxtral/config.yaml` ne contient que les écarts choisis par l'utilisateur,
pour que les nouveaux défauts d'une mise à jour s'appliquent. Sections et clés :

```yaml
model:
  name: "mzbac/voxtral-mini-3b-4bit-mixed"   # modèle de dictée (cache HuggingFace)

hotkey:
  combo: "alt_r"                 # ⌥ Option droite ; "f13", "alt+space", "cmd+shift+h"…

transcription:
  language: "auto"               # ou fr, en, de, es, it, pt, nl, hi…
  task: "transcribe"             # ou "translate" (vers l'anglais)
  max_new_tokens: 1024           # plancher, relevé selon la durée de l'audio

sounds:
  enabled: true
  volume: 0.5

ui:
  auto_paste: true               # false = copier sans coller
  clipboard_restore_delay_ms: 300  # délai avant de rendre l'ancien presse-papier

updates:
  auto_check: true               # vérification GitHub au démarrage

recording:
  max_duration_s: 300            # coupe-circuit si le relâchement n'arrive jamais
  start_retries: 1               # reconstructions du micro après un échec
  tail_padding_ms: 350           # on continue d'enregistrer après le relâchement
  silence_padding_ms: 250        # silence ajouté de chaque côté
  silence_threshold_dbfs: -45    # dictée jetée si rien ne dépasse ce niveau

file_transcription:
  model: "mlx-community/whisper-large-v3-turbo"
  output_dir: "~/Documents/Voxtral"
  block_duration_s: 300
  max_duration_s: 14400          # fichiers plus longs refusés (4 h)
  include_timestamps: true
  diarization: false             # identifier les locuteurs

offline:
  prefer_offline: true           # aucun accès réseau quand le modèle est en cache

maintenance:
  log_max_mb: 10                 # journal tronqué au-delà (0 = jamais)
  recordings_max_age_days: 30    # dictées conservées purgées après (0 = jamais)
```

## Dépendances

Voir `requirements.txt` (commenté) ; versions exactes dans `requirements.lock`.

| Package | Rôle |
|---------|------|
| mlx | Framework ML Apple Silicon |
| mlx-voxtral | Inférence Voxtral (dictée) |
| mlx-whisper | Whisper : fichiers audio, traduction, modèle alternatif |
| mlx-audio | Identification des locuteurs (Sortformer) |
| rumps | App menu bar native macOS |
| sounddevice, numpy, soundfile | Capture micro, buffers, lecture/écriture WAV |
| pynput | Raccourci clavier global + Cmd+V simulé |
| pyobjc-framework-Cocoa | AppKit : NSSound, NSPasteboard, NSOpenPanel, NSStatusItem |
| pyyaml | config.yaml |
| huggingface_hub | Téléchargement et cache des modèles |
| tkinter (python-tk via Homebrew) | Fenêtre Préférences |

Tests et lint (hors app) : pytest, pytest-mock, ruff.

## Script d'installation

```bash
curl -fsSL https://raw.githubusercontent.com/Jeanjipm/transcript_voxtral/main/install.sh | bash
```

Le script (`install.sh`, idempotent, relançable pour réinstaller) :

1. Vérifie Apple Silicon (refuse sur Intel)
2. Installe Homebrew si absent
3. Choisit Python 3.13 (puis 3.12, 3.11, 3.14), l'installe via Homebrew si
   besoin, avec `python-tk` pour tkinter
4. Clone ou met à jour le dépôt dans `~/.voxtral/app`
5. Crée le venv et installe `requirements.lock` (repli sur `requirements.txt`)
6. Télécharge le modèle de dictée par défaut (~3,2 Go) via `download_model.py`
7. Crée la commande `voxtral` et `~/Desktop/Voxtral.command` (double-clic),
   qui redirigent la sortie vers `~/.voxtral/voxtral.log`
8. Propose le démarrage automatique (LaunchAgent)
9. Lance l'app

Temps : 3 à 5 minutes, dominé par le téléchargement du modèle.

## Mises à jour

- **De l'app** : menu « Mises à jour » (cf. F8), ou à la main :
  `pkill -f voxtral && cd ~/.voxtral/app && git pull`, puis relancer. Si
  `install.sh` ou `requirements.*` ont changé, relancer l'installation.
- **D'un modèle** : Préférences → Modèles (téléchargement avec progression),
  ou `~/.voxtral/venv/bin/python ~/.voxtral/app/download_model.py`.

## Permissions macOS requises

- **Microphone** : demandé au premier lancement
- **Accessibilité** : nécessaire pour le raccourci clavier global et le
  Cmd+V simulé (Réglages Système → Confidentialité et sécurité →
  Accessibilité, cocher l'app ou le Terminal/Python qui la lance)

## Contraintes

- Zéro réseau à l'exécution, sauf téléchargement explicite d'un modèle et
  vérification des mises à jour (désactivable)
- Compatible tout Apple Silicon ; RAM minimum 8 Go (modèle 4 bits),
  16 Go+ recommandés pour les modèles 8 bits ou pleine précision
- macOS 13+ (Ventura)
- Pas d'App Store ni de signature : distribution par script, d'où l'absence
  de notifications système (le menu les remplace)
- Python typé, un fichier = une responsabilité, aucune dépendance sans
  justification, `rumps` + `tkinter` seulement (pas d'Electron ni de Qt)

## Non construit (prévu en v0.1/0.2, abandonné ou reporté)

- Mode « toggle » du raccourci (appuyer pour démarrer, appuyer pour arrêter)
- Sons personnalisés et thèmes sonores (les sons système suffisent)
- Notification après collage (impossible sans signature)
- Température, streaming, dossier de modèles (`~/.voxtral/models/`) : jamais
  utiles, retirés de la configuration
- App iPhone, Windows/Linux, timestamps mot à mot, commandes vocales

## Licences

- Code : MIT.
- Voxtral Mini 3B : CC BY-NC 4.0, usage **non commercial**. Pour un usage
  commercial, basculer sur Whisper Large V3 (MIT) dans Préférences → Modèles.
- Whisper : MIT (OpenAI). Sortformer : voir la carte du modèle sur HuggingFace.

## Changelog

| Version | Date | Modifications |
|---------|------|--------------|
| 0.1 | 14 avril 2026 | Brief initial (cahier des charges) |
| 0.2 | 16 avril 2026 | Sons, UI de paramétrage, choix du modèle Voxtral Mini 3B et du raccourci Cmd+Shift+H |
| 0.3 | 4 novembre 2026 | Réécriture pour décrire le logiciel réel après les sprints 6a–6c : raccourci ⌥ droite capturé au clavier, transcription de fichiers avec horodatages et locuteurs, hors-ligne réel, suivi du micro, filtre anti-silence, mises à jour in-app, entretien, 539 tests et CI |
