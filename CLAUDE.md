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

## Communication fin de sprint

À la fin de chaque sprint (PR ouverte ou changements pushés), TOUJOURS envoyer
à l'utilisateur :

1. **Commandes terminal de déploiement** (copier-coller prêt) — par défaut
   git pull suffit, voir section "Déploiement" ci-dessous.
2. **Protocole de test** structuré, en précisant pour chaque feature :
   - Les prérequis de config (ex. "ouvrir Préférences → Langue → cocher
     Traduction") — les fixes sont souvent conditionnels aux settings
   - L'action concrète à effectuer
   - Le résultat attendu
3. **Si `install.sh` a été modifié** : signaler explicitement qu'il faut
   réinstaller (et pas juste git pull).

## Déploiement chez l'utilisateur

### Cas par défaut — git pull (99% des sprints)

Quand SEULS des fichiers `.py` / `.yaml` / `.md` ont changé :
```bash
pkill -f voxtral
cd ~/.voxtral/app && git fetch --all && git checkout <branch> && git pull
# Double-clic sur ~/Desktop/Voxtral.command
```

⚠️ Ordre important : `pkill` AVANT `git checkout`, sinon le process vivant
continue de tourner avec l'ancien code en RAM.

### Réinstallation nécessaire quand

- `install.sh` a été modifié (nouveau launcher, bundle, LaunchAgent)
- `requirements.txt` a changé (nouvelle dep pip)
- État local corrompu (dernier recours)

```bash
cd ~
pkill -f voxtral 2>/dev/null
launchctl unload ~/Library/LaunchAgents/com.voxtral.dictee.plist 2>/dev/null
rm -rf ~/Applications/Voxtral.app ~/.voxtral ~/Library/LaunchAgents/com.voxtral.dictee.plist ~/Desktop/Voxtral.command
rm -f ~/.local/bin/voxtral /opt/homebrew/bin/voxtral /usr/local/bin/voxtral
curl -fsSL https://raw.githubusercontent.com/Jeanjipm/transcript_voxtral/<branch>/install.sh | bash
```

⚠️ Supprime le modèle (~3 Go) → re-téléchargement à la réinstall, compter
10+ min selon la connexion. À ne proposer que si strictement nécessaire.

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
