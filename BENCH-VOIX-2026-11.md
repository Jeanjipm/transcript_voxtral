# Banc d'essai « voix » — 18 novembre 2026

Trois pistes nées de l'épisode Underscore_ sur Kyutai ont été mesurées sur le
Mac de JJ (MacBook Pro M5 Pro, 48 Go) avant toute intégration. L'outil est
`tools/bench_voix.py` (hors app, relançable par `bash tools/bench_voix.sh`).

**Verdict en quatre lignes**

| Piste | Verdict |
|---|---|
| Silero VAD à la place du filtre de volume | **À intégrer.** 21/21 contre 18/21, déjà fourni par le `mlx-audio` de l'app. |
| Pocket TTS pour « Lire le texte copié » | **À intégrer, grand modèle (`french_24l`).** 25 fois plus prompt que Voxtral TTS, préféré à l'écoute. Exige d'écrire les nombres en lettres. |
| Kyutai STT (dictée en continu) | **Pas maintenant.** Suit la voix, mais moins juste que l'existant sur les noms et les sigles. |
| Smart Turn (fin de phrase) | **Pas maintenant.** 9/12 sur la voix de JJ : il couperait la parole une fois sur trois phrases en suspens. |

---

## 1. Lecture à voix haute

| Moteur | Licence des poids | Chargement | Délai avant le 1er son | Vitesse | Pic mémoire |
|---|---|---|---|---|---|
| Voxtral TTS 4B 4-bit (actuel) | CC BY-NC | 1,6 s | 2,01 s | 3,3× | 3,0 Go + 4,1 Go GPU |
| Pocket TTS `french` (6 couches) | CC-BY | 3,4 s | 0,03 s | 13,0× | 1,3 Go |
| Pocket TTS `french_24l` (24 couches) | CC-BY | 6,7 s | 0,08 s | 4,8× | 3,2 Go |

Le délai de Voxtral est celui de l'app : le son part quand la première phrase
est entièrement synthétisée (1,3 à 3,8 s selon sa longueur). Pocket TTS rend
le son au fil de la génération.

**À l'écoute (JJ)** : `french_24l` nettement meilleur ; `french` a de petits
artefacts dans la voix ; Voxtral garde son artefact de début de phrase et des
intonations moins justes. Un mot mal prononcé par `french_24l` :
« enneigement ».

**Les nombres.** Pocket TTS ne sait pas lire un nombre écrit en chiffres.
Vérifié par aller-retour (texte → voix → Whisper) sur la phrase du banc :

| Texte donné | `french` | `french_24l` |
|---|---|---|
| « …au 12 mars 2026 à 14h30… 3500 euros… » | charabia | charabia |
| « …au douze mars deux mille vingt-six à quatorze heures trente… » | parfait | parfait |

Conséquence : la colonne « mots déformés » du rapport brut (6,4 % / 12,9 % /
11,4 %) est dominée par ce seul texte et ne départage pas les voix. Toute
intégration doit convertir les nombres en lettres avant la synthèse.

**À savoir aussi**

- Voxtral TTS en mode « flux » (que l'app n'utilise pas) : premier son après
  0,20 s au lieu de 1,4 s sur la phrase courte. Une amélioration possible sans
  changer de moteur, si Pocket TTS devait être écarté.
- Kyutai présente `french_24l` comme un aperçu non distillé : il peut changer.
- Pocket TTS tourne sur le processeur (torch), pas sur la puce graphique : il
  ne dispute pas le modèle de dictée.
- Seule voix française du catalogue : `estelle`. Le clonage d'une voix depuis
  un extrait demande les poids en accès conditionné (compte Hugging Face).

## 2. Dictée en continu : Kyutai STT 1B en/fr

| Modèle | Mots faux, phrases synthé | Mots faux, voix de JJ (6 phrases) | Vitesse |
|---|---|---|---|
| Kyutai STT 1B en/fr | 4,1 % | 9,8 % | 4,5× |
| Whisper large-v3-turbo | 2,2 % | 6,6 % | cf. BENCH-2026-11 |
| Whisper large-v3 | 1,2 % | 8,2 % | |
| Voxtral Mini 3B 8-bit | 2,9 % | 8,2 % | |

- **Il suit la voix** : retard d'affichage de 0,50 s, constant, sur tous les
  fichiers (c'est le retard de construction du modèle ; le Mac tient le temps
  réel avec une marge de 4,5).
- **Il est moins juste là où ça compte** : « URSS AF », « Danton » devenu
  « dans ton », « le lot inélivré » pour « le lot 1 est livré », fin de phrase
  tronquée (« Saint-Auvrin sur »). 20 à 27 % de mots faux sur les phrases de
  noms propres et de sigles.
- **Il n'est pas tout à fait stable** : deux passages sur les mêmes fichiers
  ont donné 4,3 % puis 4,1 %, avec des erreurs différentes.
- **La mesure sur voix réelle est mince** : six phrases courtes, une
  soixantaine de mots, choisies pour leurs pièges (« Kixit Consulting »,
  « URSSAF »). Une erreur y pèse 1,6 point : les quatre modèles sont dans la
  même zone, on ne peut pas les classer là-dessus.

Son atout, le texte qui s'écrit pendant qu'on parle, vaut peu pour une dictée
collée au curseur à la fin ; il compterait pour un mode conversation. Il
impose en outre un environnement à part (un `mlx` ancien, Python 3.12).

## 3. Mains libres

### Y a-t-il de la parole ? Silero VAD

| | Silero | Filtre de volume actuel (−45 dBFS) |
|---|---|---|
| Bonnes réponses | **21/21** | 18/21 |
| Bruit sans parole, −40 et −25 dBFS | rien | « parole » ✗ (le modèle hallucinerait) |
| Phrase chuchotée, −62 dBFS | parole | « rien » ✗ (dictée jetée) |
| Les six phrases de JJ (−24 à −27 dBFS) | parole | parole |

345 fois le temps réel, chargement 1,2 s, modèle de 2 Mo, déjà dans le
`mlx-audio` 0.4.6 de l'app : aucune dépendance à ajouter. Réserve : les bruits
testés sont du bruit blanc de synthèse, pas un clavier, une rue ou une
télévision (qui, elle, contient de la parole).

### La phrase est-elle finie ? Smart Turn v3.2

| Jeu d'essai | Phrases finies reconnues | Phrases en suspens reconnues |
|---|---|---|
| Voix de macOS (32 phrases) | 15/16 | 6/16 |
| Voix de JJ (12 phrases) | 5/6 | 4/6 |

Réponse en 2 ms. Mais sur la voix de JJ, deux phrases en suspens sur six
(« …à Claire et ensuite… », « …des œufs, et aussi… ») sont jugées finies, à
0,95 et 0,77 de probabilité : même en n'écoutant le modèle que lorsqu'il est
très sûr de lui, il couperait la parole. Et une phrase finie sur six est jugée
en suspens. Douze phrases ne font pas une statistique, mais elles suffisent à
ne pas lui confier seul la décision d'arrêter l'enregistrement.

Un mode mains libres reste faisable sans lui : Silero détecte la parole, et un
silence d'une durée réglable met fin à la dictée.

## 4. Ce qui ferait changer d'avis

- Un modèle Kyutai STT plus juste sur les noms propres, ou un moyen de lui
  souffler un lexique.
- Une version de Smart Turn qui tient sur une centaine de phrases réelles.
- Un Pocket TTS qui lit les nombres, ou une seconde voix française.

## 5. Relancer

```bash
bash tools/bench_voix.sh               # tout
bash tools/bench_voix.sh tts           # une partie : tts, stt ou vad
bash tools/bench_voix.sh enregistrer   # douze phrases dites par soi, puis stt et vad
```

Les deux environnements du banc vivent dans `~/.voxtral/bench/` (rien n'est
installé dans l'app) ; `rm -rf ~/.voxtral/bench` les efface. Les sons sont
dans `~/Documents/Voxtral/banc-voix/`.
