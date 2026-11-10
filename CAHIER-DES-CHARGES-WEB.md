# Cahier des charges — bibliothèque web de transcriptions

Version 0.5 — 10 novembre 2026 — **à valider avant tout développement**

*0.2 : ajout du fonctionnement réel de Plaud Web (§ 3).*
*0.3 : l'enregistrement des visios entre dans le projet (§ 7, lot 6).*
*0.4 : résultat du prototype de capture du son du Mac (§ 7).*
*0.5 : retrait de tout ce qui décrivait un usage personnel. Restent le
fonctionnement de Plaud et des ordres de grandeur.*

## 1. L'idée en une phrase

Un site privé, hébergé sur Vercel avec une base Neon, où tu retrouves toutes
tes transcriptions depuis n'importe quel appareil. La transcription elle-même
reste faite par Voxtral sur ton Mac : le site ne fait que **ranger, afficher,
chercher et exporter**.

```
Enregistreur  →  fichier audio sur le Mac  →  Voxtral (local)  →  texte
                                                                    │
                                              envoi du TEXTE seul   ▼
                          navigateur / téléphone  ←  site (Vercel) + base (Neon)
```

Ce qui quitte le Mac : le texte, les horodatages, les locuteurs. **Jamais
l'audio** (en version 1).

## 2. Ce que j'ai pu étudier, et ce qui reste à voir

| Sujet | État |
|---|---|
| Ce que Voxtral produit aujourd'hui | Vérifié dans le code |
| Fonctions de Plaud Web | Documentation publique de Plaud |
| Plaud Web (écrans réels) | **Observé le 10 novembre** : liste, transcription ouverte, menus, recherche. Non ouverts : la fenêtre de renommage d'un locuteur et celle d'export (décrites d'après la documentation) |
| Identification des locuteurs | État des lieux fait, **test terrain à faire** |
| Plaud comme simple enregistreur | Documentation lue, **test de 5 minutes à faire par toi** |

## 3. Comment Plaud Web fonctionne, et ce qu'on en reprend

### Le fonctionnement observé

- **La liste.** Un tableau trié par date : titre, durée, date de création.
  Sur chaque ligne, un menu : renommer, déplacer dans un dossier, mettre à
  la corbeille. Chaque fichier porte son origine : appareil en mode note,
  appareil en mode appel, application de bureau (visios), import.
- **Les titres sont rédigés par l'IA de Plaud**, sur le modèle « date, type
  d'échange, sujet ». Les fichiers sans titre portent simplement leur date
  et heure.
- **La recherche** porte sur les titres et les mots-clés, et garde les
  recherches récentes.
- **L'écran de transcription** tient en trois colonnes : la liste des
  fichiers en version compacte à gauche, le texte au centre, « Ask Plaud » à
  droite. Le texte est découpé en tours de parole courts, chacun avec son
  horodatage et son locuteur. Un lecteur audio reste fixé en haut.

### Les ordres de grandeur à prévoir

- **Quelques centaines de fichiers, presque aucun dossier.** À ce volume on
  ne range pas, on cherche : la recherche est la fonction n° 1 du site, les
  dossiers passent au second plan.
- **Environ un tiers des enregistrements vient de l'application de bureau**,
  donc de visios enregistrées sur le Mac, et non de l'appareil. Voir § 7 :
  c'est un point que la demande initiale ne couvrait pas.
- **Durées d'une minute à deux heures environ.** Tout passe sous la limite
  de 4 h.

### Ce qu'on reprend, ce qu'on laisse

| Fonction de Plaud Web | Version 1 | Remarque |
|---|---|---|
| Liste des fichiers (titre, durée, date) | Oui | + nombre de locuteurs |
| Recherche par titre ou mot-clé, avec recherches récentes | **Oui, en priorité** | Dans les titres **et** dans le texte |
| Liste compacte à côté de la transcription ouverte | Oui | Pour passer d'un fichier à l'autre sans revenir en arrière |
| Transcription avec horodatage et locuteurs | Oui | Même découpage en paragraphes que le .txt actuel |
| Renommer un locuteur (un paragraphe ou tous) | Oui | « Speaker 1 » → un prénom |
| Copier la transcription | Oui | |
| Export de la transcription (TXT, SRT, DOCX, PDF) | Oui | Avec ou sans horodatage, avec ou sans locuteurs |
| Renommer, déplacer, mettre à la corbeille | Oui | Suppression définitive après 30 jours |
| Filtre par origine (« Comes from ») | Oui | Appareil, visio, import |
| Corriger le texte, rechercher-remplacer | Oui (lot 4) | Un nom propre mal reconnu se corrige partout d'un coup |
| Dossiers | Oui (lot 3) | Un seul niveau |
| Titres rédigés automatiquement | **Non** | Titre = nom du fichier, modifiable à la main — voir § 11 |
| Texte « nettoyé » par IA (« Transcript cleaned up ») | **Non** | Le site affiche la transcription brute |
| Lecteur audio, lecture à partir d'un passage | **Non** | Demande de stocker l'audio en ligne — voir § 10 |
| Import d'un audio, re-transcription depuis le site | **Non** | La transcription se fait sur le Mac, pas sur le serveur |
| Lien de partage, invitations | **Non** | Tout reste derrière ta connexion |
| Résumés, modèles, cartes mentales, « Ask Plaud » | **Non** | Exclus à ta demande |

## 4. Les écrans

**Liste.** Colonne de gauche : Tous les fichiers, Non classés, tes dossiers,
Corbeille. Au centre : un tableau trié par date, avec titre, date
d'enregistrement, durée, nombre de locuteurs. Barre de recherche en haut.
Sélection multiple pour déplacer ou supprimer.

**Détail d'une transcription.** En tête : titre modifiable, date, durée,
modèle utilisé. En dessous, le texte par paragraphes — `[00:12:34]`, nom du
locuteur dans sa couleur, puis le texte. Boutons : Copier tout, Exporter,
Déplacer, Supprimer. Recherche dans la page. Un clic sur un nom de locuteur
ouvre le renommage.

**Réglages.** Les jetons d'accès du Mac (créer, révoquer), et « Tout
exporter » (une archive de toutes tes transcriptions, pour ne jamais être
prisonnier du site).

**Téléphone.** Le site doit être confortable sur un écran de téléphone :
c'est le principal intérêt par rapport aux .txt du Mac.

## 5. Côté Mac : ce qui change dans Voxtral

Aujourd'hui, « Transcrire un fichier audio… » écrit un .txt dans
`~/Documents/Voxtral`. Ça ne change pas. On ajoute, **après** l'écriture du
.txt, l'envoi du résultat au site.

- **Désactivé par défaut.** Tes amis qui installent Voxtral ne voient rien de
  nouveau et rien ne part de chez eux.
- **Le .txt local d'abord.** Si le réseau est coupé, la transcription est
  quand même terminée et rangée ; l'envoi attend dans une file
  (`~/.voxtral/outbox/`) et repart au prochain lancement.
- **Pas de doublon.** Chaque transcription reçoit un identifiant unique ; un
  renvoi de la même transcription est ignoré par le site.
- **Préférences → nouvel onglet « Web »** : adresse du site, jeton d'accès,
  case « Envoyer chaque transcription », bouton « Tester la connexion ».
- **Jeton rangé dans le Trousseau macOS**, pas dans un fichier lisible.
- **Aucune dépendance ajoutée** : l'envoi utilise ce que Python fournit déjà.
- **Les dictées au raccourci ne sont pas envoyées** : elles sont collées au
  curseur et ne sont pas conservées. Seules les transcriptions de fichiers
  vont sur le site.

⚠️ **Ce point contredit une règle du projet.** `CLAUDE.md` interdit toute
dépendance réseau à l'exécution et annonce « zéro cloud ». L'envoi vers le
site est une exception voulue : il faudra l'écrire noir sur blanc dans
`CLAUDE.md` et le README (« optionnel, désactivé par défaut, texte seul »).

Rappel des durées : un fichier est accepté jusqu'à 4 h (réglable jusqu'à
12 h). Une réunion longue passe donc par « Transcrire un fichier », pas par
la dictée au raccourci (5 min, réglable jusqu'à 30).

## 6. Les locuteurs (diarisation)

### Ce qui est établi

- **Voxtral ne sait pas le faire en local.** L'identification des locuteurs
  annoncée par Mistral existe, mais seulement dans leur service en ligne
  (Voxtral Mini Transcribe V2, payant à la minute). Mistral ne l'annonce
  nulle part pour le modèle Voxtral téléchargeable. S'en servir reviendrait
  donc à envoyer ton audio chez Mistral.
- **L'app le fait déjà autrement**, avec un second modèle local :
  Sortformer v2.1 de NVIDIA. C'est l'option « Identifier les locuteurs ».
- **Mesuré dans le projet :** 96 % de paroles bien attribuées sur un dialogue
  à deux voix, trois voix séparées correctement, une heure traitée en une
  minute environ.

### Ce qui n'est pas établi

- Ces mesures ont été faites sur des **voix de synthèse**, pas sur une vraie
  réunion (pièce qui résonne, micro posé loin, gens qui se coupent la parole).
- Aucune vraie réunion n'a encore été transcrite avec l'app : il n'existe
  **aucune preuve terrain** pour l'instant.
- **4 locuteurs maximum.** À 5 personnes ou plus, les voix en trop sont
  rabattues sur les 4 premières.
- Les comparatifs publics ne désignent pas de gagnant net. L'alternative
  sérieuse (pyannote) avait été écartée parce qu'elle exige un compte et une
  licence à accepter — gênant pour tes amis, acceptable pour toi seul.

### Conséquences pour le site

1. Le locuteur est **facultatif** : une transcription sans locuteurs s'affiche
   normalement.
2. Le site donne les moyens de **corriger** : renommer, réattribuer un
   paragraphe à un autre locuteur, fusionner deux locuteurs qui sont en
   réalité la même personne.
3. Un avertissement s'affiche quand 4 locuteurs sont détectés (« il y en
   avait peut-être davantage »).

### Le test à faire avant de s'engager

Tu exportes de Plaud **une vraie réunion** : l'audio, et la transcription de
Plaud avec ses locuteurs. Je passe le même audio dans Voxtral et je compare
paragraphe par paragraphe. On obtient un pourcentage sur *ton* type
d'enregistrement, pas sur des voix de synthèse. Si c'est mauvais, on essaie
pyannote avant de construire quoi que ce soit.

Cet audio et sa transcription restent sur le Mac. Seul le pourcentage est
noté ici : rien de leur contenu n'entre dans le dépôt.

## 7. Plaud comme simple enregistreur

Objectif : enregistrer avec le Plaud, sans que l'audio parte chez Plaud.

La réponse dépend du modèle de Plaud et de la version de son firmware : à
confirmer (§ 11).

Deux sources à remplacer : l'appareil, et Plaud Desktop, qui enregistre les
visios sur le Mac. L'appareil ne remplace pas le second.

### L'appareil

| Piste | Ce que dit la documentation Plaud |
|---|---|
| Brancher le Plaud en USB et lire les fichiers | **Non** pour NotePin et NotePin S. Sur le Plaud Note, le réglage « Access via USB » existait mais il est désactivé depuis le firmware 2.1. Rien trouvé d'officiel pour le Note Pro |
| Couper l'envoi vers le cloud | **Oui** : app Plaud → Explore → Private Cloud Sync → décocher. Les enregistrements restent sur le téléphone |
| Sortir l'audio de l'app | **Oui**, en MP3 ou WAV, un fichier à la fois |
| Ce qui est déjà chez Plaud | Y reste tant que tu ne le supprimes pas (« Delete recordings from the cloud ») |

Deux réserves honnêtes :

- Une page de Plaud laisse entendre que l'export exige la synchronisation
  activée. Si c'est vrai, cette piste tombe. **Seul un essai tranche.**
- L'app Plaud reste nécessaire pour vider l'appareil, et elle exige ton
  compte. On peut viser « aucun audio chez Plaud », pas « aucun contact avec
  Plaud ».

**L'essai (5 minutes).** Coupe Private Cloud Sync. Enregistre une minute.
Transfère vers le téléphone. **N'appuie pas sur « Générer »** (c'est ce bouton
qui envoie l'audio pour transcription). Exporte l'audio en WAV vers le Mac
(AirDrop). Vérifie sur web.plaud.ai que le fichier n'apparaît pas.

Si l'essai échoue : un dictaphone classique qui se branche en USB comme une
clé fait le même travail sans aucun compte.

### Les visios : un enregistreur de réunion dans Voxtral

*Décidé le 10 novembre : traité dans ce projet.*

**Le besoin.** Enregistrer une réunion en visio, quelle que soit
l'application (Teams, Meet, Zoom, navigateur) : la voix des autres, qui sort
du Mac, et la tienne, qui entre par le micro. Voxtral ne capte aujourd'hui
que le micro.

**Ce que tu verras.**

- Dans le menu Voxtral : « Enregistrer une réunion ». L'icône de la barre de
  menu change et affiche le temps écoulé, pour ne pas l'oublier.
- « Arrêter et transcrire » : l'audio est rangé dans
  `~/Documents/Voxtral/Réunions/`, la transcription démarre seule (même
  circuit que « Transcrire un fichier audio… »), puis le texte part sur le
  site.
- La dictée au raccourci reste utilisable pendant la réunion.
- Démarrage à la main. Plaud Desktop détecte les réunions tout seul ; pas en
  version 1.

**Garde-fous.**

- **Écriture sur disque au fil de l'eau.** La dictée garde tout en mémoire
  jusqu'à la fin, ce qui va pour 5 minutes, pas pour 2 heures : un plantage
  ou une mise en veille ne doit pas coûter la réunion entière.
- **Alerte si rien n'est capté.** Sans autorisation, macOS ne signale aucune
  erreur : il livre du silence. Si les premières secondes sont entièrement
  muettes, Voxtral doit le dire tout de suite, pas après deux heures.
- **Guidage vers le bon réglage après un refus.** macOS ne pose la question
  qu'une fois. Si la réponse est non, même par mégarde, il ne redemande
  jamais et livre du silence à chaque essai suivant. Voxtral doit alors
  ouvrir le bon écran des Réglages Système et dire quel interrupteur
  activer. Avant le tout premier enregistrement, il doit aussi prévenir
  qu'une fenêtre macOS va apparaître.
- **Reprise après un changement de sortie son ou une mise en veille.** La
  brique `catap` ne s'en remet pas toujours seule : brancher un casque en
  cours de réunion peut arrêter la capture. Voxtral devra la relancer et
  recoller les morceaux. Le passage d'un projecteur aux haut-parleurs, lui,
  a été encaissé pendant l'essai du 10 novembre.
- **Arrêt automatique à 4 h**, la même limite que pour les fichiers.
- **Audio gardé 30 jours sur le Mac**, puis supprimé (réglable). Une heure
  pèse environ 115 Mo.

**La voie technique.** macOS offre deux façons de capter le son du Mac.

| Voie | Autorisation que macOS te demandera | Avis |
|---|---|---|
| Prises audio « Core Audio » (macOS 14.2 et suivants) | « Enregistrement de l'audio système » seulement | **Recommandée.** Une brique Python existe déjà (`catap`), à évaluer |
| ScreenCaptureKit | « Enregistrement de l'écran et de l'audio système » | Plus intrusive : macOS considère que l'app peut filmer l'écran |
| Carte son virtuelle (BlackHole) | Installation d'un pilote, changement de la sortie son du Mac | **Écartée** : trop fragile pour un non-développeur |

Les deux premières voies sont disponibles sur le Mac de développement.

**Ce qu'il fallait prouver.** Que la capture fonctionne depuis un programme
lancé comme Voxtral. C'est fait pour le démarrage automatique : voir les
essais plus bas.

**À qui macOS donne l'autorisation.** Pas à « Voxtral » : à ce qui l'a
lancé. Or Voxtral démarre de deux façons, vérifiées le 10 novembre.

| Démarrage de Voxtral | macOS rattache l'autorisation à |
|---|---|
| Automatique, à l'ouverture de session (le cas pendant les essais) | « Python » |
| Double-clic sur `Voxtral.command` | « Terminal » |

L'autorisation devra donc être accordée pour les deux, ou Voxtral devra
toujours démarrer de la même façon. À trancher au lot 6.

**Essais du 10 novembre — la capture fonctionne.** Le paquet `catap` a été
examiné puis installé dans un environnement de test séparé. Trois essais.

| Essai | Lancé comme | Résultat |
|---|---|---|
| 1 | Double-clic (« Terminal ») | Fichier muet |
| 2 | Double-clic (« Terminal ») | Fichier muet |
| 3 | Démarrage automatique (« Python ») | 40 secondes captées, transcrites mot pour mot par Voxtral |

- **Par la voie « Python », ça marche.** Le Mac a prononcé quatre fois une
  phrase numérotée ; Voxtral a retranscrit les quatre sans une faute. macOS
  n'a rien demandé pendant cet essai : l'autorisation était déjà accordée
  sur le Mac d'essai. Le premier affichage reste donc à observer. L'essai
  portait la même identité que le Voxtral lancé à l'ouverture de session,
  vérifié dans le journal de macOS.
- **Par la voie « Terminal », le silence vient d'un refus.** Au premier
  essai, macOS a affiché une demande d'autorisation au nom du Terminal ;
  elle s'est refermée sur un refus dix secondes plus tard. Au second, il n'a
  rien redemandé.
- **Le son est capté même quand on ne l'entend pas.** Pendant les trois
  premières phrases, la sortie du Mac était un projecteur qui ne diffusait
  rien. Elles sont dans le fichier quand même.
- **Un changement de sortie en cours de route a été encaissé.** Le son est
  repassé sur les haut-parleurs du Mac entre la troisième et la quatrième
  phrase ; la capture a continué. Un seul cas, à ne pas généraliser : la
  documentation de `catap` prévient qu'un changement de format, comme peut
  en provoquer un casque Bluetooth, n'est pas rattrapé.
- C'est bien l'autorisation légère, « audio système uniquement », que macOS
  vérifie. Pas celle de l'enregistrement d'écran.

Pas encore vérifié : une vraie visio (Teams, Meet, Zoom) plutôt qu'une voix
de synthèse ; un casque, filaire ou Bluetooth ; les deux pistes séparées,
micro et son du Mac ; deux heures d'affilée ; une mise en veille ; la voie
« Terminal » une fois l'autorisation accordée.

**Ce que ça entraîne.**

- Une dépendance de plus et une autorisation macOS à accepter : il faudra
  **réinstaller** Voxtral chez toi, pas seulement le mettre à jour.
- Pour tes amis : fonction optionnelle, masquée sur les Mac antérieurs à
  macOS 14.2.

**Une piste prometteuse pour les locuteurs.** Le micro et le son du Mac
arrivent séparément. En gardant les deux pistes, ta voix est identifiée à
coup sûr, et l'identification automatique ne travaille plus que sur les
autres. C'est net avec un casque ; avec les haut-parleurs du Mac, le micro
reprend la voix des autres et l'avantage se réduit. Version 1 : une seule
piste mélangée. Les deux pistes sont à évaluer pendant le prototype.

**À ne pas oublier.** Enregistrer une réunion suppose d'en prévenir les
participants. L'outil ne le fait pas à ta place.

## 8. Choix techniques, expliqués

| Choix | Pourquoi |
|---|---|
| **Next.js sur Vercel** | Le couple standard de Vercel : le site et son « guichet » de réception (l'API) vivent dans le même projet, mis en ligne à chaque modification |
| **Neon (Postgres), région Europe** | Une vraie base de données, avec une recherche plein texte en français intégrée (trouve « réunion » en tapant « reunion ») |
| **Dépôt séparé** pour le site | Autre langage, autre cycle de mise en ligne. Le dépôt actuel ne reçoit que la partie « envoi » |
| **Un seul compte, inscription fermée** | C'est ta bibliothèque personnelle. Connexion e-mail + mot de passe ; la brique exacte (Neon Auth ou équivalent) se choisit au lot 1 |
| **Jeton d'accès pour le Mac** | Le Mac ne se « connecte » pas comme un humain : il présente un jeton long, affiché une seule fois, révocable depuis le site |
| **Texte seul** | Une heure de réunion pèse environ 0,1 Mo en texte. Des milliers d'heures tiennent dans les offres d'entrée de gamme (tarifs à vérifier au lot 1) |

### Ce que la base contient

- **transcriptions** : titre, fichier d'origine, date d'enregistrement, durée,
  langue, modèle, dossier, état (complète / interrompue), date de mise à la
  corbeille.
- **segments** : pour chaque passage — début, fin, locuteur, texte, numéro de
  paragraphe (calculé par Voxtral, pour que le site affiche exactement le
  même découpage que le .txt).
- **locuteurs** : le nom que tu as donné à chaque locuteur d'une transcription.
- **dossiers**, **jetons d'accès** (jamais stockés en clair).

### Confidentialité

Une transcription de réunion contient les propos d'autres personnes : ce sont
des données personnelles. D'où : hébergement en Europe, aucune page publique,
site non référencé par les moteurs de recherche, suppression réelle après la
corbeille, export complet possible à tout moment.

**Règle pour les dépôts de code.** Le dépôt de Voxtral est public sur
GitHub. Aucun contenu réel n'y entre, ni dans celui du site : pas de
transcription, pas de titre d'enregistrement, pas de nom de personne ou
d'organisation, rien sur le contenu des échanges. Cela vaut pour le code,
les tests, les exemples et la documentation. Les essais utilisent des textes
inventés ; seuls des chiffres agrégés peuvent être publiés.

## 9. Découpage en lots

| Lot | Contenu | C'est fini quand… |
|---|---|---|
| **0 — Vérifications** | Test terrain des locuteurs ; essai « Plaud simple enregistreur » ; prototype de capture du son du Mac (**fait le 10 novembre : la capture fonctionne**, voir § 7) | On sait si les trois hypothèses tiennent, et ce cahier est corrigé en conséquence |
| **1 — Socle du site** | Projet Vercel + base Neon, connexion, réception d'une transcription, liste, détail en lecture, **recherche** | Une transcription envoyée à la main apparaît sur ton téléphone, et tu la retrouves en tapant un mot |
| **2 — Envoi depuis Voxtral** | File d'attente, onglet « Web » des Préférences, tests automatiques | Tu transcris un fichier sur le Mac, il apparaît seul sur le site ; sans réseau, il part au lancement suivant |
| **3 — Rangement** | Renommage des locuteurs, exports, corbeille, dossiers, filtre par origine | Tu exportes en PDF une réunion dont les locuteurs portent leur vrai nom |
| **4 — Corrections** | Modifier le texte, rechercher-remplacer, réattribuer et fusionner des locuteurs | Tu corriges une transcription sans repasser par le Mac |
| **5 — Dossier surveillé** (optionnel) | Tout audio déposé dans un dossier du Mac est transcrit puis envoyé, sans rien cliquer | Tu déposes un fichier, tu le lis sur le site quelques minutes plus tard |
| **6 — Enregistreur de réunion** | Menu, capture micro + son du Mac, écriture au fil de l'eau, arrêt automatique, transcription et envoi enchaînés | Tu enregistres une vraie visio d'une heure et tu la lis sur le site sans rien faire d'autre |

Le lot 6 ne dépend que du prototype du lot 0 et du lot 2. Il peut passer
avant les lots 3 à 5 si les visios sont plus urgentes pour toi que le
rangement.

## 10. Ce qui est volontairement laissé de côté

- **L'audio en ligne** et la lecture synchronisée. Faisable plus tard, mais
  c'est l'audio qui est sensible et lourd.
- **La reprise de l'existant Plaud.** Possible, mais l'export Plaud se fait
  fichier par fichier : à plusieurs centaines, c'est un chantier en soi.
- **La détection automatique des réunions** (Plaud Desktop le fait ; ici le
  démarrage est manuel).
- **Plusieurs utilisateurs**, partage, liens publics.
- **Tout ce qui est rédigé par une IA** : résumés, comptes rendus, questions,
  titres automatiques, texte « nettoyé ».

## 11. Décisions que j'attends de toi

1. **En visio, casque ou haut-parleurs du Mac ?** La séparation de ta voix
   en dépend (§ 7).
2. **Les titres.** Plaud les rédige par IA. Sans IA, le titre est le nom du
   fichier et tu le corriges à la main. Ça te va ?
3. **Audio en ligne : non en version 1 ?** (ma recommandation : non.)
4. **Reprendre les fichiers déjà chez Plaud, ou repartir de zéro ?**
5. **Quel modèle de Plaud, et quelle version de firmware ?**
6. **Un seul utilisateur, toi ?**
7. **Nom et adresse du site.**

## Sources

- Plaud — [Export](https://support.plaud.ai/hc/en-us/articles/50835453223705-Export-files),
  [Nommer les locuteurs](https://support.plaud.ai/hc/en-us/articles/51039619094553-Name-Speakers),
  [Dossiers](https://support.plaud.ai/hc/en-us/articles/51073116390425-Create-rename-and-delete-folders),
  [Import audio](https://support.plaud.ai/hc/en-us/articles/51038386639897-Add-Audio),
  [Lien de partage](https://support.plaud.ai/hc/en-us/articles/51023340108185-Share-link),
  [Guide transcription](https://support.plaud.ai/hc/en-us/articles/60037380408601-How-to-transcribe-edit-and-export-your-recordings-Complete-guide)
- Plaud, accès aux fichiers — [Plaud Note en USB](https://support.plaud.ai/hc/en-us/articles/8127779432079-Why-can-t-I-see-my-recordings-after-I-plug-the-device-into-my-PC),
  [NotePin](https://support.plaud.ai/hc/en-us/articles/50841476214553-Can-I-plug-my-NotePin-into-my-PC-to-access-files),
  [NotePin S](https://support.plaud.ai/hc/en-us/articles/53788775968409-Can-I-access-Plaud-NotePin-S-recordings-by-connecting-to-a-PC-via-USB),
  [Private Cloud Sync](https://support.plaud.ai/hc/en-us/articles/53793945763737-Private-Cloud-Sync),
  [Permissions](https://support.plaud.ai/hc/en-us/articles/53792891824537-Permission-settings)
- Capture du son du Mac — [catap](https://pypi.org/project/catap/),
  [AudioCap](https://github.com/insidegui/AudioCap),
  [prises audio Core Audio](https://www.recall.ai/blog/core-audio-taps),
  [ScreenCaptureKit](https://www.recall.ai/blog/how-to-use-screencapturekit-to-record-a-meeting)
- Mistral — [Voxtral Transcribe 2](https://mistral.ai/news/voxtral-transcribe-2)
- Diarisation — [comparatif Soniqo](https://soniqo.audio/benchmarks/diarization),
  [mesures FluidAudio](https://docs.fluidinference.com/reference/benchmarks.md),
  [article arXiv 2509.26177](https://arxiv.org/html/2509.26177v1)
- Projet — en-tête de `diarizer.py`, `file_transcriber.py`, `config.yaml`
