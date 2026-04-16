# Brief technique — Voxtral Dictée

**Version :** 0.2 | **Date :** 16 avril 2026

## Résumé

App macOS menu bar de dictée vocale locale. L'utilisateur appuie sur un
raccourci clavier, parle, relâche (ou re-appuie), et le texte transcrit
est collé à la position du curseur. Tout tourne en local sur Apple Silicon
via MLX. Aucune donnée ne quitte la machine.

Distribuable gratuitement à des amis / stagiaires via un script d'installation
en une commande.

## Cible utilisateur

- Utilisateurs macOS sur Apple Silicon (M1+)
- Non-développeurs (installation assistée par script)
- Francophones principalement, mais multilingue (toutes langues supportées par Voxtral)

## Fonctionnalités v0

### F1 — Dictée par raccourci clavier

**Raccourci par défaut : `Cmd+Shift+H`** (configurable dans l'app de paramétrage)

**⚠️ Note importante sur le choix du raccourci :**
Le raccourci `Cmd+H` demandé initialement est réservé par macOS pour "Masquer
l'application active". L'intercepter globalement est techniquement possible
avec `pynput`, mais créerait des conflits dans la majorité des applications
(le comportement attendu par l'utilisateur serait rompu). Alternatives
proposées par ordre de recommandation :

1. **`Cmd+Shift+H`** — libre sur macOS, proche ergonomiquement de Cmd+H (défaut)
2. **`Option+Space`** — style Raycast/Alfred, très ergonomique
3. **`Cmd+Option+D`** (D pour Dictée)
4. **`Cmd+H`** — possible mais déconseillé (interception globale avec warning)

Le raccourci est **configurable** dans l'app de paramétrage (cf. F6).

**Modes de déclenchement :**
- **Push-to-talk** : maintenir = enregistrer, relâcher = transcrire (défaut)
- **Toggle** : appuyer une fois = démarrer, appuyer à nouveau = stop
- Configurable dans `config.yaml` et dans l'UI de paramétrage

**Feedback visuel :** icône menu bar change d'état (inactif → écoute → transcription).

### F2 — Sons d'activation / désactivation

Feedback audio pour confirmer le démarrage et l'arrêt de l'écoute, sans avoir
à regarder l'écran.

- **Son de démarrage** : "ding" court et doux (300-500 ms), joué dès que
  le micro commence à capturer
- **Son d'arrêt** : "pop" ou son descendant (200-400 ms), joué dès que
  l'enregistrement est terminé et que la transcription commence
- **Son de transcription terminée** (optionnel, désactivé par défaut) :
  notification courte quand le texte est collé

**Implémentation :**
- Fichiers WAV embarqués dans le package (`sounds/start.wav`, `sounds/stop.wav`)
- Lecture via `AppKit.NSSound` (pyobjc) pour une intégration système native,
  ou fallback `afplay` en subprocess
- Volume configurable (0-100%, défaut : 50%)
- Possibilité de désactiver complètement les sons dans l'UI de paramétrage

**Sources des sons :**
- Option A : sons système macOS (`/System/Library/Sounds/Tink.aiff` et `Pop.aiff`)
- Option B : sons custom libres de droits (Freesound.org sous CC0)
- Option C : génération via ElevenLabs Sound Effects (one-shot, puis embarqué)

### F3 — Transcription locale via Voxtral Mini 3B

**Modèle principal : `mistralai/Voxtral-Mini-3B-2507`** (Mistral AI, CC BY-NC 4.0)

**Versions quantifiées MLX (recommandées pour performance / taille) :**
- `mzbac/voxtral-mini-3b-4bit-mixed` — **3,2 Go**, précision mixte (défaut)
- `mzbac/voxtral-mini-3b-8bit` — **5,3 Go**, 8-bit, qualité supérieure
- Modèle full précision : ~8 Go, pour GPU avec plus de RAM

**Package Python : `mlx-voxtral`** (de mzbac, optimisé Apple Silicon)

**API principale :**
```python
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

model = VoxtralForConditionalGeneration.from_pretrained(
    "mzbac/voxtral-mini-3b-4bit-mixed"
)
processor = VoxtralProcessor.from_pretrained(
    "mzbac/voxtral-mini-3b-4bit-mixed"
)

# Transcription
inputs = processor.apply_transcrition_request(
    language="fr",          # ou "en", "de", "auto", etc.
    audio="recording.wav",
    task="transcribe",      # ou "translate" pour traduire vers l'anglais
)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.0,
)
text = processor.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
```

**Langues supportées :** toutes celles de Voxtral (français, anglais, allemand,
espagnol, italien, portugais, néerlandais, hindi, etc.). Détection automatique
ou forçage via paramètre `language`.

**Fallback si MLX Voxtral indisponible :** `mlx-whisper` (Whisper Large V3 Turbo),
mature et performant. L'architecture modulaire permet de swapper sans toucher
au reste du code.

**Streaming pour audio long** (dictées > 30 s) :
```python
for chunk in model.transcribe_stream(audio, processor, chunk_length_s=30):
    print(chunk, end="", flush=True)
```

### F4 — Injection du texte

- Le texte transcrit est copié dans le presse-papier
- Puis collé automatiquement (simulate `Cmd+V`) à la position du curseur
- Fonctionne dans n'importe quelle app (VS Code, Mail, Safari, Claude...)
- Notification discrète de confirmation (désactivable)

### F5 — Gestion des modèles

- Téléchargement initial depuis HuggingFace au premier lancement
- Commande CLI `voxtral update` ou bouton menu pour mettre à jour
- **Changement de modèle via l'UI de paramétrage** (cf. F6)
- Les modèles vivent dans `~/.voxtral/models/`
- Liste des modèles disponibles maintenue dans un fichier JSON (local ou distant)

### F6 — App de paramétrage (UI légère)

Petite fenêtre de configuration accessible depuis le menu bar ("Préférences...").

**Stack technique :** `rumps` + `tkinter` pour la v0 (natif Python, aucune
dépendance lourde), migration vers SwiftUI quand produit validé.

**Paramètres exposés :**

**Onglet "Modèle"**
- Choix du modèle parmi une liste :
  - Voxtral Mini 3B 4-bit (3,2 Go) — rapide, recommandé
  - Voxtral Mini 3B 8-bit (5,3 Go) — qualité supérieure
  - Voxtral Mini 3B full (8 Go) — qualité maximale
  - Whisper Large V3 Turbo (fallback)
- Bouton "Télécharger" / "Mettre à jour" pour chaque modèle
- Indicateur de taille, d'état (téléchargé / non téléchargé / à jour)

**Onglet "Langue"**
- Langue de transcription : Auto / Français / English / Deutsch / Español / ...
- Tâche : Transcription / Traduction (vers anglais)

**Onglet "Raccourci"**
- Capture de la combinaison de touches (avec détection de conflits système)
- Mode de déclenchement : Push-to-talk / Toggle
- Warning explicite si l'utilisateur choisit un raccourci système (Cmd+H, etc.)

**Onglet "Sons"**
- Activer / désactiver les sons d'activation / désactivation
- Volume (slider 0-100%)
- Choix du thème sonore : Système macOS / Doux / Subtil / Custom
- Preview des sons

**Onglet "Avancé"**
- Température de génération (0.0 à 1.0, défaut 0.0 pour transcription fidèle)
- `max_new_tokens` (128 à 2048, défaut 1024)
- Mode streaming pour audio long (on/off)
- Notification après collage (on/off)
- Dossier des modèles (par défaut `~/.voxtral/models/`)

**Onglet "À propos"**
- Version de l'app
- Version du modèle actif
- Lien GitHub
- Bouton "Mettre à jour l'app"

Les paramètres sont sauvegardés dans `~/.voxtral/config.yaml` et chargés
au démarrage de l'app menu bar.

### F7 — Menu bar

Icône discrète dans la barre de menu (rumps).

**Menu déroulant :**
- 🟢 État : "Prêt" / "Écoute en cours..." / "Transcription..."
- Raccourci actif : affichage de la combinaison (ex. "⌘⇧H")
- Langue : affichage de la langue active (ex. "🇫🇷 Français")
- Modèle : nom court du modèle actif (ex. "Voxtral 3B 4-bit")
- ---
- Préférences... (ouvre la fenêtre F6)
- Mettre à jour le modèle
- ---
- À propos
- Quitter

**Indicateur visuel pendant l'enregistrement :** icône rouge ou pulsante.

## Architecture

```
voxtral-dictee/
├── app.py                 # Point d'entrée, menu bar rumps
├── transcriber.py         # Chargement modèle MLX + inférence (Voxtral ou Whisper)
├── audio_capture.py       # Capture micro (sounddevice)
├── audio_feedback.py      # Sons d'activation / désactivation (NSSound)
├── hotkey_manager.py      # Écoute raccourci clavier global (pynput)
├── clipboard.py           # Presse-papier + paste simulé
├── model_manager.py       # Download / update depuis HuggingFace
├── config.py              # Lecture/écriture config.yaml
├── settings_ui.py         # Fenêtre de paramétrage (tkinter)
├── sounds/
│   ├── start.wav
│   └── stop.wav
├── config.yaml            # Configuration utilisateur (défaut)
├── download_model.py      # Script CLI de téléchargement
├── install.sh             # Script d'installation one-liner
├── requirements.txt       # Dépendances Python
└── README.md              # Doc utilisateur
```

## Configuration (config.yaml)

```yaml
model:
  name: "mzbac/voxtral-mini-3b-4bit-mixed"
  path: "~/.voxtral/models/"

hotkey:
  combo: "cmd+shift+h"
  mode: "push_to_talk"  # ou "toggle"

transcription:
  language: "auto"        # ou "fr", "en", "de", "es", etc.
  task: "transcribe"      # ou "translate"
  temperature: 0.0
  max_new_tokens: 1024
  streaming: false        # true pour audio > 30s

sounds:
  enabled: true
  volume: 0.5             # 0.0 à 1.0
  theme: "system"         # "system", "soft", "subtle", "custom"
  start_sound: "sounds/start.wav"
  stop_sound: "sounds/stop.wav"

ui:
  notification_on_paste: false
  auto_paste: true
```

## Dépendances

| Package | Rôle |
|---------|------|
| mlx | Framework ML Apple Silicon |
| mlx-voxtral | Inférence Voxtral (audio → texte) |
| mlx-whisper | Fallback Whisper |
| rumps | App menu bar native macOS |
| sounddevice | Capture micro |
| numpy | Traitement signal audio |
| pyyaml | Lecture/écriture config |
| huggingface_hub | Téléchargement modèles |
| pynput | Raccourci clavier global + simulate paste |
| pyobjc-framework-AppKit | NSSound pour feedback audio natif |
| tkinter | UI de paramétrage (stdlib Python, aucune install) |

## Script d'installation

```bash
curl -fsSL https://raw.githubusercontent.com/Jeanjipm/voxtral-dictee/main/install.sh | bash
```

Le script :
1. Vérifie Apple Silicon (refuse sur Intel)
2. Installe Homebrew si absent
3. Installe Python 3.11+ via Homebrew si absent
4. Clone le repo
5. Crée un venv, installe les dépendances
6. Télécharge le modèle 4-bit par défaut (~3,2 Go)
7. Crée un alias `voxtral` dans le PATH
8. Ajoute l'app au démarrage automatique (optionnel, demandé à l'utilisateur)
9. Lance l'app

Temps estimé : 3-5 min (dépend de la connexion pour le modèle).

## Mise à jour du modèle

Depuis l'UI ("Préférences" → "Modèle" → "Mettre à jour") ou via CLI :
```bash
voxtral update
```

Vérifie le hash du modèle sur HuggingFace, retélécharge si nouvelle version.

## Permissions macOS requises

- **Microphone** : demandé au premier lancement
- **Accessibilité** : nécessaire pour le raccourci clavier global et le
  paste simulé (l'utilisateur doit l'autoriser manuellement dans
  Réglages Système → Confidentialité et sécurité → Accessibilité)
- **Automation** (optionnel) : pour une meilleure intégration avec certaines apps

Le README doit documenter ces étapes avec des captures d'écran.

## Contraintes

- Zéro réseau à l'exécution (sauf update explicite du modèle)
- Compatible M1 / M2 / M3 / M4 / M5 (tout Apple Silicon)
- RAM minimum : 8 Go (modèle 4-bit tient en ~3,2 Go)
- RAM recommandée : 16 Go+ (pour le modèle 8-bit ou full)
- macOS 13+ (Ventura) minimum pour MLX
- Pas d'App Store — distribution directe

## Hors scope v0

- App iPhone (prévu plus tard, architecture Hub & Spoke)
- Diarization (identification des locuteurs)
- Support Windows/Linux
- Timestamps mot à mot
- Commandes vocales (ex. "ponctuation : virgule")

## Licence du modèle Voxtral

**Important :** Voxtral Mini 3B est distribué sous licence **CC BY-NC 4.0**
(Creative Commons Attribution Non-Commercial). Pour un usage strictement
personnel et entre amis / stagiaires, c'est OK. Pour un usage commercial
(même indirect), il faudrait négocier une licence avec Mistral AI.

**Fallback libre de droits :** Whisper Large V3 Turbo (MIT) via `mlx-whisper`
si jamais la licence de Voxtral devenait bloquante.

## Estimation

~500 lignes de code Python + ~60 lignes bash (install.sh).
Réalisable en 2 soirées pour la v0 fonctionnelle (une pour le cœur dictée,
une pour l'UI de paramétrage et les sons).

## Changelog

| Version | Date | Modifications |
|---------|------|--------------|
| 0.1 | 14 avril 2026 | Brief initial |
| 0.2 | 16 avril 2026 | Ajout sons d'activation/désactivation (F2), UI de paramétrage (F6), clarification Voxtral Mini 3B via Context7 (nom officiel `mistralai/Voxtral-Mini-3B-2507`, versions quantifiées MLX `mzbac/voxtral-mini-3b-4bit-mixed` 3.2 Go et `mzbac/voxtral-mini-3b-8bit` 5.3 Go, package `mlx-voxtral`), discussion raccourci Cmd+H et alternatives (Cmd+Shift+H par défaut). |
