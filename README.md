# Voxtral Dictée

> Dictée vocale 100 % locale pour macOS Apple Silicon.
> Maintenir une touche → parler → relâcher → le texte apparaît au curseur.
> Aucune donnée ne quitte le Mac.

---

## Installation (1 commande)

```bash
curl -fsSL https://raw.githubusercontent.com/Jeanjipm/transcript_voxtral/main/install.sh | bash
```

Le script installe Homebrew (si absent), Python 3.13, le code, le modèle
Voxtral 3B (~3,2 Go), et crée la commande `voxtral`. Compte 3 à 5 minutes
selon ta connexion.

À la fin, l'app démarre toute seule et un raccourci **Voxtral.command** est
posé sur le Bureau : double-clique dessus pour la relancer plus tard (ou tape
`voxtral` dans un terminal). Le script propose aussi le démarrage automatique
à l'ouverture de session.

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

### Transcrire un enregistrement déjà fait

Pour un fichier audio existant — un mémo vocal, une réunion enregistrée —
la dictée n'est pas le bon outil : elle est faite pour des phrases courtes.

Clique l'icône 🎤 → **Transcrire un fichier audio…**, choisis le fichier, et
un `.txt` horodaté apparaît dans `~/Documents/Voxtral` (le Finder s'ouvre
dessus à la fin). L'avancement s'affiche dans le menu, et tu peux annuler à
tout moment — le texte déjà transcrit est conservé.

```
Transcription — reunion.m4a
Modèle : mlx-community/whisper-large-v3-turbo
Durée : 00:35:12 — généré le 30/07/2026 à 14:22
Langue : fr

[00:00:00] Bonjour à tous et merci d'être présents…

[00:00:41] Concernant le budget, je voudrais insister sur un point précis…
```

Formats acceptés : tout ce que macOS sait lire — `.m4a` (Dictaphone,
iPhone), `.mp3`, `.wav`, `.aiff`, `.caf`, `.flac`, et la piste audio d'un
`.mp4` ou `.mov`. La conversion est automatique.

Tu peux continuer à dicter pendant qu'un fichier est en cours de
transcription : la dictée passe devant.

#### Identifier les locuteurs (optionnel)

Pour un enregistrement à plusieurs voix, Voxtral peut préfixer chaque
paragraphe par « Locuteur 1 », « Locuteur 2 »… Coche **Préférences →
Fichiers → Identifier qui parle**.

Le modèle (225 Mo) se télécharge au premier usage. Jusqu'à **4 personnes**.

Mesuré sur un dialogue à deux voix dont on connaît la vérité seconde par
seconde : **96 % d'attribution correcte**, dans les deux ordres de passage,
et les trois voix bien séparées sur un test à trois. Une heure
d'enregistrement coûte environ une minute de traitement.

Deux réserves honnêtes : ces mesures portent sur des voix de synthèse, pas
sur du terrain réel avec réverbération et micro lointain — vérifie sur tes
propres enregistrements avant de compter dessus. Et quand deux personnes
parlent en même temps, une seule étiquette est posée sur le passage.

Si l'identification échoue, la transcription reste écrite, et l'en-tête du
`.txt` dit pourquoi les étiquettes manquent.

### Lire le texte copié à voix haute

Le chemin inverse de la dictée. Copie un texte (Cmd+C), puis clique l'icône
🎤 → **Lire le texte copié** : il est lu au haut-parleur, sur ton Mac, avec
le modèle de synthèse vocale [Voxtral 4B TTS](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-4bit)
(2,5 Go). Il se télécharge à la première lecture, après confirmation — rien
ne part en silence. La voix se choisit dans **Préférences → Lecture** ; deux
voix françaises (femme, homme) sont en tête de liste. Un texte très long
n'est lu que sur ses 2 000 premiers caractères, et l'app le dit.

Pendant la lecture, l'icône devient un haut-parleur et l'item du menu
affiche l'avancement ; une dictée lancée au même moment passe devant.

Le même moteur sert à `vocaliser_cahier.py`, qui enregistre en MP3 toutes
les phrases du cahier de révisions CE1 et les publie (cf. CLI utiles).

### Changer le raccourci, la langue, le modèle…

Clique l'icône 🎤 → **Préférences…**

Onglets :
- **Dictée** : le raccourci, la langue, ce qui se passe avec le texte
  (collé au curseur ou seulement copié), et les sons — avec un bouton
  « Écouter » pour régler le volume sans deviner.

  Le raccourci ne se tape pas, il s'enregistre : clique **Modifier…**, puis
  appuie sur la touche (ou la combinaison) que tu veux, et relâche. Échap
  annule, « Par défaut » revient à ⌥ Option droite. La dictée est en pause
  tant que la fenêtre de préférences est ouverte — sinon appuyer sur une
  touche pour l'enregistrer déclencherait un enregistrement micro.
- **Fichiers** : dossier de destination des `.txt`, horodatage des
  paragraphes, durée maximale acceptée.
- **Modèles** : quel modèle pour la dictée, lequel pour les fichiers, et la
  place que chacun occupe sur le disque — avec un bouton pour supprimer
  ceux qui ne servent plus. Choisir un modèle absent propose son
  téléchargement immédiat, avec une barre de progression.
- **Avancé** : réglages fins (fin de phrase conservée, durée maximale d'une
  dictée, longueur du texte, seuil de silence, délai avant de rendre le
  presse-papier, vérification des mises à jour).
- **Lecture** : la voix qui lit le texte copié (les 20 voix du modèle, avec
  leur langue), et l'état du modèle de lecture avec un bouton pour le
  télécharger tout de suite.
- **À propos** : aide-mémoire et lien vers le code source.

Les changements sont appliqués automatiquement en quelques secondes
(hot-reload) — pas besoin de redémarrer l'app.

---

## CLI utiles (debug)

```bash
# Lancer l'app à la main (sinon : double-clic sur ~/Desktop/Voxtral.command)
voxtral

# Transcrire un fichier audio (wav, m4a, mp3, aiff…) avec le modèle de dictée
~/.voxtral/venv/bin/python ~/.voxtral/app/transcribe.py mon_memo.m4a --lang fr

# Re-télécharger / mettre à jour le modèle actif
~/.voxtral/venv/bin/python ~/.voxtral/app/download_model.py

# Lister les modèles disponibles
~/.voxtral/venv/bin/python ~/.voxtral/app/download_model.py --list

# Voix du cahier de révisions CE1 : écouter les voix françaises (dossier
# ~/Documents/Voxtral/voix-ce1/essais/), puis tout générer et publier sur
# Vercel (projet voix-ce1). Incrémental : relancer reprend où ça s'est arrêté.
# Nécessite ffmpeg (installé via Homebrew s'il manque) et Node 18+.
~/.voxtral/venv/bin/python ~/.voxtral/app/vocaliser_cahier.py --essai
~/.voxtral/venv/bin/python ~/.voxtral/app/vocaliser_cahier.py --voix fr_female
~/.voxtral/venv/bin/python ~/.voxtral/app/vocaliser_cahier.py --voix fr_female --sans-publication
~/.voxtral/venv/bin/python ~/.voxtral/app/vocaliser_cahier.py --voix fr_female --tout   # tout refaire après un changement du moteur
```

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| Le raccourci ne fait rien | Accessibilité non autorisée | Réglages Système → Confidentialité → Accessibilité → cocher Voxtral / Terminal / Python |
| Pas de son `Tink` au début | Sons désactivés ou volume à 0 | Préférences → Dictée → Sons |
| Texte vide après dictée | Trop court (< 0.5s) ou silence | Reparler plus longtemps ; vérifier le micro dans Réglages → Son |
| « Modèle introuvable » | Téléchargement interrompu | `python download_model.py` pour relancer |
| App lente la 1ʳᵉ fois | Modèle MLX se charge en RAM | Normal, quelques secondes à la 1ʳᵉ dictée seulement |
| Caractères spéciaux (é, ñ) cassés en dictée | Conflit Right Option | Choisir un autre raccourci dans Préférences → Dictée (ex. F13) |
| Le raccourci ne répond plus (souvent après une veille) | macOS a coupé la surveillance clavier | Menu 🎤 → **Réactiver le raccourci** |
| Ce qui est collé est l'ANCIEN contenu du presse-papier | Application lente à lire le presse-papier (bureau distant, Electron) | Préférences → Avancé → monter le « délai avant de rendre le presse-papier » à 800–1000 ms |
| Dictée coupée après 5 minutes, rien de collé | Relâchement du raccourci jamais reçu : coupe-circuit | L'audio est gardé dans `~/.voxtral/recordings/` (30 jours) ; menu 🎤 → Transcrire un fichier audio… |

### Logs

L'app écrit ses erreurs sur `stderr`. Pour voir ce qui se passe :

```bash
# Lancement en premier plan (logs directement dans le terminal)
voxtral

# Lancée par Voxtral.command ou le démarrage auto : tout va dans le journal
tail -f ~/.voxtral/voxtral.log
```

Le journal ne grossit pas sans fin : au démarrage, s'il dépasse 10 Mo, il
est tronqué et on garde le dernier mégaoctet. Les dictées coupées pour
dépassement de durée, conservées dans `~/.voxtral/recordings/`, sont
supprimées après 30 jours. Les deux seuils se règlent dans la section
`maintenance` de `config.yaml` (0 = jamais).

---

## Désinstallation

```bash
# Si tu as activé le démarrage auto
launchctl unload ~/Library/LaunchAgents/com.voxtral.dictee.plist
rm ~/Library/LaunchAgents/com.voxtral.dictee.plist

# Supprimer la commande `voxtral` (peut être dans l'un de ces 3 dossiers,
# selon ce que install.sh avait choisi) — les chemins absents sont ignorés
rm -f ~/.local/bin/voxtral /opt/homebrew/bin/voxtral /usr/local/bin/voxtral 2>/dev/null
rm -f ~/Desktop/Voxtral.command

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
- **Fichiers audio** : Whisper Large V3 Turbo via `mlx-whisper` (horodatages),
  Sortformer v2.1 via `mlx-audio` pour identifier les locuteurs.
- **Synthèse vocale** : Voxtral 4B TTS (4-bit, 24 kHz) via `mlx-audio`
  (`synthese.py`) ; lecture par `sounddevice`, MP3 du cahier par `ffmpeg`
  (mono 64 kb/s, `loudnorm`, `atempo=0.92`).

Architecture détaillée : [BRIEF-TECHNIQUE.md](BRIEF-TECHNIQUE.md).

---

## Licences

- **Code Voxtral Dictée** : MIT.
- **Modèle Voxtral Mini 3B** : CC BY-NC 4.0 (usage **non commercial**
  uniquement). Pour un usage commercial, bascule sur Whisper Large V3
  (MIT) dans Préférences → Modèles.
- **Whisper Large V3** : MIT (OpenAI). Supporte transcription et
  traduction vers l'anglais.
- **Modèle Voxtral 4B TTS** (lecture à voix haute) : CC BY-NC 4.0 (usage
  **non commercial** uniquement).
