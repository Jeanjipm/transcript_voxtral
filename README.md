# Voxtral Dictée

> Dictée vocale 100 % locale pour macOS Apple Silicon.
> Maintenir une touche → parler → relâcher → le texte apparaît au curseur.
> Aucune donnée ne quitte le Mac.

---

## Installation (1 commande)

```bash
curl -fsSL https://raw.githubusercontent.com/Jeanjipm/transcript_voxtral/main/install.sh | bash
```

Le script installe Homebrew (si absent), Python 3.11, le code, le modèle
Voxtral 3B (~3,2 Go), et crée la commande `voxtral`. Compte 3 à 5 minutes
selon ta connexion.

> **Pré-requis :** macOS 13+ (Ventura), Mac Apple Silicon (M1/M2/M3/M4),
> 8 Go de RAM minimum.

---

## Premier lancement — autoriser deux permissions

macOS demande deux autorisations. Tu dois les **accepter manuellement**
dans les Réglages :

1. **Microphone** — popup automatique au 1ᵉʳ lancement, accepte.
2. **Accessibilité** — pour que le raccourci global et le collage
   fonctionnent. Va dans :

   > **Réglages Système → Confidentialité et sécurité → Accessibilité**

   Active la case en face de **Voxtral** (ou de **Terminal** / **Python**
   selon comment tu as lancé l'app).

Sans Accessibilité, le raccourci ne déclenche rien.

---

## Utilisation

1. Tu vois l'icône 🎤 dans la barre de menu (en haut à droite de l'écran).
2. Place ton curseur dans n'importe quelle app (Mail, Notes, VS Code,
   Safari, Slack…).
3. **Maintiens la touche ⌥ Option DROITE** → tu entends un *Tink*.
4. Parle.
5. **Relâche** → tu entends un *Pop* → le texte apparaît à la position
   du curseur.

C'est tout.

### Changer le raccourci, la langue, le modèle…

Clique l'icône 🎤 → **Préférences…**

Onglets :
- **Modèle** : choisir entre Voxtral 3B (4-bit, 8-bit, full) ou Whisper.
- **Langue** : auto-détection ou langue forcée (fr, en, de, es, …) ;
  tâche transcription ou traduction vers anglais.
- **Raccourci** : touche unique tenue (talkie-walkie) ou combinaison
  (ex. `alt+space`). Mode push-to-talk — maintenir pour enregistrer.
- **Sons** : activer/désactiver, volume.
- **Avancé** : longueur max de transcription, collage auto.

Les changements sont appliqués automatiquement en quelques secondes
(hot-reload) — pas besoin de redémarrer l'app.

---

## CLI utiles (debug)

```bash
# Transcrire un fichier WAV
voxtral-launcher.sh        # ou : python /chemin/transcript_voxtral/app.py
python ~/.voxtral/app/transcribe.py mon_enregistrement.wav --lang fr

# Re-télécharger / mettre à jour le modèle actif
python ~/.voxtral/app/download_model.py

# Lister les modèles disponibles
python ~/.voxtral/app/download_model.py --list
```

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| Le raccourci ne fait rien | Accessibilité non autorisée | Réglages Système → Confidentialité → Accessibilité → cocher Voxtral / Terminal / Python |
| Pas de son `Tink` au début | Sons désactivés ou volume à 0 | Préférences → onglet Sons |
| Texte vide après dictée | Trop court (< 0.5s) ou silence | Reparler plus longtemps ; vérifier le micro dans Réglages → Son |
| « Modèle introuvable » | Téléchargement interrompu | `python download_model.py` pour relancer |
| App lente la 1ʳᵉ fois | Modèle MLX se charge en RAM | Normal, quelques secondes à la 1ʳᵉ dictée seulement |
| Caractères spéciaux (é, ñ) cassés en dictée | Conflit Right Option | Choisir un autre raccourci dans Préférences (ex. F13 ou `cmd+shift+h`) |

### Logs

L'app écrit ses erreurs sur `stderr`. Pour voir ce qui se passe :

```bash
# Lancement en premier plan (logs directement dans le terminal)
~/.voxtral/app/voxtral-launcher.sh

# Si Voxtral est lancé via le démarrage auto (LaunchAgent)
tail -f ~/.voxtral/voxtral.log
```

---

## Désinstallation

```bash
# Si tu as activé le démarrage auto
launchctl unload ~/Library/LaunchAgents/com.voxtral.dictee.plist
rm ~/Library/LaunchAgents/com.voxtral.dictee.plist

# Supprimer la commande `voxtral` (peut être dans l'un de ces 3 dossiers,
# selon ce que install.sh avait choisi) — les chemins absents sont ignorés
rm -f ~/.local/bin/voxtral /opt/homebrew/bin/voxtral /usr/local/bin/voxtral 2>/dev/null

# Supprimer l'app + venv + modèles + config
rm -rf ~/.voxtral
```

---

## Sous le capot

- **Modèle** : [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
  de Mistral AI, version quantifiée 4-bit par
  [mzbac](https://huggingface.co/mzbac/voxtral-mini-3b-4bit-mixed) pour MLX.
- **Inférence** : [MLX](https://github.com/ml-explore/mlx) — natif Apple
  Silicon, utilise le Neural Engine + GPU intégré.
- **App menu bar** : [rumps](https://github.com/jaredks/rumps) — ultra-léger,
  pas d'Electron.
- **Capture micro** : `sounddevice` + `numpy` (16 kHz mono).
- **Raccourci global** : `pynput` (listener non-suppressif → ne casse pas
  Right Option pour les caractères spéciaux).
- **Collage** : `NSPasteboard` + `Cmd+V` simulé.

Architecture détaillée : [BRIEF-TECHNIQUE.md](BRIEF-TECHNIQUE.md).

---

## Licences

- **Code Voxtral Dictée** : MIT.
- **Modèle Voxtral Mini 3B** : CC BY-NC 4.0 (usage **non commercial**
  uniquement). Pour un usage commercial, bascule sur Whisper Large V3
  (MIT) dans Préférences → Modèle.
- **Whisper Large V3** : MIT (OpenAI). Supporte transcription et
  traduction vers l'anglais.
