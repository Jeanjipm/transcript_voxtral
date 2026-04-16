# Projet Voxtral Dictée

App macOS menu bar de dictée vocale locale. Raccourci clavier → enregistrement
micro → transcription via MLX → texte collé au curseur. Zéro cloud, tout
tourne sur le Mac de l'utilisateur. Distribuable aux amis via script install.

Utilisateur = non-développeur. Toujours expliquer les choix techniques.

## Stack
- Python 3.11+
- MLX / mlx-lm (inférence Apple Silicon)
- rumps (app menu bar native macOS)
- PyAudio ou sounddevice (capture micro)
- PyYAML (configuration)
- HuggingFace Hub (téléchargement modèles)

## Commandes
- `python app.py` — lancer l'app menu bar
- `python download_model.py` — télécharger/mettre à jour le modèle
- `python transcribe.py <fichier.wav>` — transcrire un fichier (CLI debug)
- `bash install.sh` — installation complète (Homebrew, Python, dépendances, modèle)

## Workflow
- Commencer en mode Plan → proposer → attendre validation → exécuter
- Commit sur branch feature/ avec message conventionnel (feat/fix/docs)
- Rappeler le commit git en fin de tâche
- Après implémentation, relire le code pour vérifier la cohérence

## Style
- Python typé (type hints partout)
- Un fichier = une responsabilité
- Config externalisée dans config.yaml (nom du modèle, hotkey, langue)
- Pas de dépendance inutile — chaque import doit se justifier
- Noms de fichiers : snake_case

## Interdit
- Coder sans plan validé par l'utilisateur
- Dépendance réseau à l'exécution (sauf téléchargement initial du modèle)
- Framework lourd (Electron, Qt) — rumps uniquement pour la v0
- Utiliser `--dangerously-skip-permissions`
