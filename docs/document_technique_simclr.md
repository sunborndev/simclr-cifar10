# Guide technique et pédagogique du projet SimCLR sur CIFAR-10

## Comprendre la logique, l'implémentation, les expériences et l'intérêt réel

**Version de travail : 15 août 2026**  
**Niveau visé : lecteur débutant qui souhaite comprendre en profondeur**

> **Avertissement scientifique.** Ce document distingue ce qui est implémenté,
> ce qui est mesuré et ce qui reste à faire. Le préentraînement et l'évaluation
> linéaire sont terminés. L'expérience supervisée depuis zéro possède, au moment
> de cette rédaction, une seule graine aléatoire complète. Le fine-tuning,
> l'étude d'ablation des augmentations, la projection en deux dimensions et
> l'application de démonstration restent nécessaires pour satisfaire toutes les
> consignes du projet. Aucun résultat manquant n'est inventé.

---

## 1. Le problème traité

### 1.1 Pourquoi les labels sont-ils précieux ?

Une image numérique n'arrive pas naturellement avec une phrase disant ce
qu'elle contient. Un humain doit souvent l'examiner et lui associer une
étiquette, aussi appelée **label** : « chat », « camion », « avion », etc. Cette
annotation peut coûter du temps et de l'argent. Dans certains domaines, elle
demande aussi une expertise rare. Par exemple, annoter une radiographie peut
nécessiter un professionnel de santé et annoter une image satellite peut
nécessiter un spécialiste de la télédétection.

Pourtant, il est souvent facile de collecter beaucoup d'images non étiquetées.
Le projet explore donc une question très concrète : peut-on apprendre quelque
chose d'utile en observant des images avant de connaître leurs noms ?

### 1.2 Question de recherche

La question centrale est :

> Dans quelle mesure un préentraînement visuel sans étiquettes permet-il de
> réduire le nombre d'images étiquetées nécessaire pour classifier correctement
> les images du jeu CIFAR-10 ?

Le terme **CIFAR-10** désigne un jeu de données produit dans le cadre du
Canadian Institute for Advanced Research. Il contient dix classes d'images. Le
projet ne cherche pas seulement à entraîner un modèle. Il cherche à mesurer la
valeur d'une stratégie d'apprentissage lorsque les labels sont rares.

### 1.3 Hypothèse, expérience et résultat ne sont pas la même chose

Une démarche scientifique sépare cinq niveaux :

1. **Hypothèse** : avant l'expérience, nous pensons qu'un préentraînement sans
   labels aidera surtout lorsque peu de labels sont disponibles.
2. **Expérience** : nous préentraînons un encodeur sans labels, puis nous
   évaluons ses représentations avec 1 %, 10 % et 100 % des labels.
3. **Observation** : nous relevons les valeurs réellement présentes dans les
   fichiers de résultats.
4. **Interprétation** : nous proposons une explication compatible avec ces
   observations.
5. **Conclusion** : nous formulons uniquement ce que le protocole autorise à
   affirmer.

Cette séparation empêche de transformer une intuition en « preuve » simplement
parce qu'elle semble plausible.

---

## 2. Les familles d'apprentissage

### 2.1 Intelligence artificielle

L'**intelligence artificielle** est un domaine large qui cherche à construire
des systèmes capables d'effectuer des tâches associées à des formes
d'intelligence : reconnaître une image, traduire un texte, planifier une action
ou recommander un contenu.

Un programme à règles fixes peut appartenir à ce domaine, mais le projet
s'intéresse à une sous-famille : l'apprentissage automatique.

### 2.2 Apprentissage automatique

L'**apprentissage automatique** consiste à ajuster automatiquement un système à
partir de données. Au lieu d'écrire toutes les règles permettant de reconnaître
un chat, on montre des exemples au système et on ajuste ses paramètres pour
réduire ses erreurs.

### 2.3 Apprentissage profond

L'**apprentissage profond** utilise des réseaux de neurones constitués de
nombreuses transformations successives. Le mot « profond » indique qu'il existe
plusieurs niveaux de traitement entre l'entrée et la sortie.

Dans une image, les premiers niveaux peuvent apprendre des motifs simples comme
des contours. Des niveaux plus avancés peuvent combiner ces motifs pour
représenter des textures, des formes et des parties d'objets.

### 2.4 Apprentissage supervisé

Dans l'**apprentissage supervisé**, chaque exemple d'entraînement possède une
réponse connue. Pour une image, cette réponse peut être la classe « chat ». Le
modèle prédit une classe, compare sa prédiction au label réel et modifie ses
poids pour réduire l'erreur.

Avantage : l'objectif correspond directement à la tâche finale.

Limite : il faut des labels, parfois coûteux.

### 2.5 Apprentissage non supervisé

Dans l'**apprentissage non supervisé**, le modèle analyse des données sans
réponse cible fournie par un humain. Il peut chercher des regroupements, des
structures ou des facteurs communs.

Ce terme est très large. Il ne décrit pas précisément la manière dont le signal
d'apprentissage est construit.

### 2.6 Apprentissage auto-supervisé

Dans l'**apprentissage auto-supervisé**, les données créent elles-mêmes une
tâche d'entraînement. Aucun humain n'écrit le label « chat », mais le programme
peut fabriquer une question dont il connaît automatiquement la réponse.

Dans notre projet, deux versions transformées de la même image sont considérées
comme liées. Le modèle doit apprendre à les reconnaître comme deux vues d'une
même source. La supervision vient donc de la relation entre les transformations,
pas d'une classe écrite par un humain.

### 2.7 Apprentissage contrastif

L'**apprentissage contrastif** apprend par comparaison. Il rapproche certaines
représentations et en éloigne d'autres.

- Une **paire positive** contient deux vues dérivées de la même image.
- Une **paire négative** contient des vues dérivées de deux images sources
  différentes.

Le mot « positif » ne signifie pas que l'image est bonne. Il signifie que la
relation doit être reconnue comme correspondante.

---

## 3. Le jeu d'images CIFAR-10

### 3.1 Contenu

CIFAR-10 contient 60 000 images couleur de 32 pixels par 32 pixels. Chaque
pixel possède trois composantes : rouge, vert et bleu. Une image brute peut
donc être représentée par un tableau de forme :

```text
[3, 32, 32]
```

Les dix classes sont : avion, automobile, oiseau, chat, cerf, chien, grenouille,
cheval, bateau et camion.

Le jeu est équilibré : chaque classe contient 6 000 images.

- 50 000 images servent au train, c'est-à-dire à l'apprentissage ;
- 10 000 images servent au test final.

Le mot **train** désigne ici l'ensemble d'entraînement. Le mot **test** désigne
un ensemble conservé à part pour mesurer la généralisation.

### 3.2 Pourquoi ce jeu est utile

CIFAR-10 est assez petit pour permettre des expériences sur une carte graphique
grand public. Ses classes variées permettent de mesurer une classification
visuelle. Son équilibre évite qu'une classe majoritaire suffise à obtenir une
bonne accuracy artificielle.

### 3.3 Limites

Les images sont très petites. Un objet de 32 pixels par 32 pixels contient peu
de détails. Une méthode efficace sur CIFAR-10 n'est pas automatiquement efficace
sur des radiographies, des vidéos, des images satellites ou des photographies
haute résolution.

CIFAR-10 constitue donc un banc d'essai, pas une preuve universelle.

### 3.4 Normalisation

Les valeurs des pixels sont converties en nombres utilisables par le réseau,
puis normalisées canal par canal. Le code soustrait une moyenne et divise par
un écart-type calculés pour CIFAR-10.

Intuition : si une composante varie autour d'une valeur moyenne, la
normalisation replace son centre près de zéro et rend les échelles plus
comparables. Cela aide généralement l'optimisation numérique.

Fichier concerné : `simclr/augmentations.py`.

---

## 4. Comment un réseau de neurones apprend

### 4.1 Entrée

Un lot de `N` images est un tenseur, c'est-à-dire un tableau numérique à
plusieurs dimensions :

```text
[N, 3, 32, 32]
```

- `N` : nombre d'images du lot ;
- `3` : canaux rouge, vert et bleu ;
- `32, 32` : hauteur et largeur.

### 4.2 Poids

Les **poids** sont les paramètres ajustables du réseau. Ils déterminent la
réponse des différentes transformations. Au début, ils sont généralement
initialisés avec de petites valeurs aléatoires. Ils ne représentent encore
rien d'utile.

Pendant l'entraînement, les poids changent pour réduire une fonction d'erreur.

### 4.3 Passage avant

Le **passage avant** est le calcul qui transforme les images en sorties à partir
des poids actuels. En anglais, il est souvent appelé *forward pass*.

Dans une classification supervisée :

```text
images -> réseau -> scores de classes
```

Dans le préentraînement contrastif :

```text
images -> encodeur -> représentations -> projections contrastives
```

### 4.4 Score, probabilité et prédiction

Un **score**, parfois appelé logit, est une valeur brute produite pour une
classe. Ces scores ne sont pas encore des probabilités.

La fonction softmax transforme les scores en valeurs positives dont la somme
vaut un. Pour une classe `k` :

```text
probabilité(k) = exp(score(k)) / somme des exp(score(j))
```

Le symbole `exp` représente la fonction exponentielle. La classe ayant la plus
grande probabilité devient la prédiction.

### 4.5 Fonction de perte

La **fonction de perte**, souvent appelée *loss*, mesure l'inadéquation entre le
comportement du modèle et l'objectif.

Dans une classification, l'entropie croisée peut s'écrire simplement :

```text
perte = -log(probabilité donnée à la bonne réponse)
```

Si le modèle attribue une forte probabilité à la bonne réponse, la perte est
petite. S'il lui attribue une faible probabilité, la perte est grande.

Une perte plus basse est généralement préférable pour le même objectif et le
même protocole. Elle ne peut pas être comparée naïvement entre deux fonctions
de perte différentes.

### 4.6 Gradient

Le **gradient** indique comment une petite variation de chaque poids ferait
varier la perte. Il ne dit pas seulement que le modèle s'est trompé ; il indique
dans quelle direction chaque paramètre doit être ajusté pour réduire l'erreur
localement.

### 4.7 Rétropropagation

La **rétropropagation** calcule les gradients en parcourant les opérations du
réseau dans le sens inverse. Le terme anglais est *backpropagation*.

Elle applique la règle de dérivation en chaîne : lorsqu'une sortie dépend d'une
suite de transformations, l'influence d'un paramètre est calculée en combinant
les influences le long de cette suite.

### 4.8 Optimiseur

L'**optimiseur** applique la mise à jour des poids. Une forme simplifiée est :

```text
nouveau poids = ancien poids - taux d'apprentissage × gradient
```

Le projet utilise deux optimiseurs selon l'expérience :

- AdamW pour le préentraînement contrastif ;
- descente de gradient stochastique avec momentum pour la baseline supervisée
  et les classifieurs linéaires.

Le **momentum** accumule une direction de déplacement au fil des mises à jour.
Il réduit certaines oscillations et aide l'optimisation à conserver une
direction cohérente.

### 4.9 Taux d'apprentissage

Le **taux d'apprentissage** contrôle l'amplitude des mises à jour.

- Trop faible : l'apprentissage peut être très lent.
- Trop élevé : les poids peuvent dépasser les bonnes régions et l'entraînement
  devenir instable.

Le projet ne garde pas toujours un taux constant. Une montée progressive est
utilisée au début du préentraînement, puis une décroissance suivant une forme de
cosinus réduit les pas vers la fin.

### 4.10 Lot, itération et époque

Un **lot** est un petit groupe d'images traité ensemble. Le terme anglais est
*batch*.

Une **itération** correspond généralement au traitement d'un lot et à une mise
à jour des poids.

Une **époque** correspond à un passage sur l'ensemble des données
d'entraînement. Avec 50 000 images et des lots de 256 images, le script conserve
195 lots complets par époque :

```text
50 000 / 256 = 195 lots complets, avec un reste non utilisé
```

Le dernier lot incomplet est ignoré pendant le préentraînement pour garder une
taille constante, ce qui simplifie le calcul contrastif.

### 4.11 Boucle complète

```text
lot d'images
    -> passage avant
    -> sorties
    -> fonction de perte
    -> rétropropagation
    -> gradients
    -> optimiseur
    -> nouveaux poids
```

Cette boucle est répétée pendant plusieurs époques.

---

## 5. Les réseaux convolutifs et ResNet18

### 5.1 Convolution

Une **convolution** applique un petit filtre à différentes positions d'une
image. Un filtre peut devenir sensible à une orientation de contour, une
texture ou une combinaison de couleurs.

Le même filtre est partagé sur toute l'image. Cette propriété réduit le nombre
de paramètres et permet de détecter un motif à plusieurs positions.

### 5.2 Canaux de caractéristiques

Après une convolution, le réseau produit des cartes de caractéristiques. Un
canal peut répondre fortement à certains motifs. À mesure que le réseau avance,
les cartes deviennent plus abstraites.

### 5.3 Réseau résiduel

ResNet signifie **réseau résiduel**. Le nombre 18 indique une variante de 18
couches principales apprenables.

Un bloc résiduel apprend une transformation `F(x)` et ajoute l'entrée `x` :

```text
sortie = F(x) + x
```

Cette connexion directe offre un chemin plus simple pour l'information et les
gradients. Elle facilite l'entraînement de réseaux profonds.

### 5.4 Adaptation à CIFAR-10

La version habituelle de ResNet18 a été pensée pour des images plus grandes.
Dans `simclr/encodeur.py`, deux adaptations sont réalisées :

1. la première convolution utilise un noyau de 3 par 3 avec un pas de 1 ;
2. le premier max-pooling est supprimé.

Le **pas** indique de combien de pixels le filtre se déplace. Un pas de 1 évite
de réduire trop tôt la petite image. Le **max-pooling** garde normalement la
valeur maximale dans de petites zones ; le supprimer préserve davantage de
détails à la résolution 32 par 32.

La couche de classification ImageNet est aussi retirée. Le réseau produit alors
une représentation de 512 valeurs.

### 5.5 Encodeur

Un **encodeur** transforme une entrée détaillée en une représentation numérique
plus compacte :

```text
image [N, 3, 32, 32]
    -> ResNet18
    -> h [N, 512]
```

La lettre `h` est simplement le nom choisi pour cette représentation. Elle ne
correspond pas encore à une classe. Elle résume des propriétés visuelles que le
réseau a apprises.

---

## 6. La méthode SimCLR

### 6.1 Signification du nom

SimCLR est le nom abrégé de *A Simple Framework for Contrastive Learning of
Visual Representations*, que l'on peut traduire par « cadre simple pour
l'apprentissage contrastif de représentations visuelles ».

Le projet implémente une version simplifiée et adaptée aux ressources
disponibles. Il ne prétend pas reproduire exactement le protocole massif du
papier original.

### 6.2 Deux vues augmentées

Pour chaque image source, le programme applique deux fois une chaîne de
transformations aléatoires :

```text
image A
  -> transformation aléatoire -> vue A1
  -> transformation aléatoire -> vue A2
```

Les deux vues n'ont pas exactement les mêmes pixels. Elles partagent toutefois
leur origine. Elles forment une paire positive.

Le but n'est pas que le modèle mémorise les pixels. Il doit reconnaître ce qui
reste stable malgré des changements raisonnables.

### 6.3 Transformations utilisées

#### Recadrage redimensionné aléatoire

Une zone de l'image est choisie puis redimensionnée à 32 par 32. Cela enseigne
une certaine robustesse à la position et au cadrage.

Risque : un recadrage trop agressif peut supprimer l'objet. La paire positive
devient alors incohérente.

#### Retournement horizontal

L'image peut être inversée de gauche à droite avec une probabilité de 50 %.

Cette transformation est raisonnable pour de nombreuses classes de CIFAR-10 :
un chat reste un chat lorsqu'il regarde dans l'autre direction.

#### Perturbation des couleurs

La luminosité, le contraste, la saturation et la teinte peuvent changer.
L'objectif est d'éviter que le modèle s'appuie uniquement sur une couleur
précise.

Risque : une transformation excessive peut détruire une information utile.

#### Passage en niveaux de gris

Certaines vues perdent leurs couleurs. Le modèle doit alors exploiter aussi la
forme et la texture.

#### Flou gaussien

Le flou atténue les détails fins. Il est disponible mais désactivé par défaut
dans la configuration CIFAR-10 du projet, car les images sont déjà très petites.

### 6.4 Pourquoi les augmentations sont au cœur de la méthode

Le choix des augmentations définit implicitement ce que le modèle doit
considérer comme invariant.

Si deux vues très différentes sont déclarées positives, le modèle reçoit un
signal contradictoire. Si les vues sont presque identiques, la tâche peut être
trop facile et le modèle peut apprendre des détails peu utiles.

L'étude d'ablation exigée par le projet doit mesurer cet effet au lieu de le
supposer.

---

## 7. De l'image à h puis à z

### 7.1 Représentation h

L'encodeur ResNet18 produit `h`, un vecteur de 512 nombres par image.

Pour un lot de 8 images :

```text
h : [8, 512]
```

Le nombre 8 décrit les exemples. Le nombre 512 décrit les caractéristiques de
chaque exemple.

### 7.2 Tête de projection

La **tête de projection** est un petit réseau placé après l'encodeur. Dans le
projet :

```text
512 -> couche linéaire -> 512 -> activation -> couche linéaire -> 128
```

Elle transforme `h` en un vecteur `z` de 128 valeurs :

```text
z : [8, 128]
```

Elle n'est pas un modèle préentraîné séparé. Ses poids sont appris en même temps
que l'encodeur pendant le préentraînement contrastif.

### 7.3 Pourquoi ne pas appliquer directement la perte à h ?

La perte contrastive impose des contraintes particulières : rapprocher les
paires positives et les séparer des négatifs. La tête de projection crée un
espace spécialisé pour cet objectif.

L'encodeur peut conserver dans `h` des informations utiles à d'autres tâches,
tandis que `z` absorbe davantage les contraintes de la comparaison
contrastive. Le papier SimCLR rapporte que cette séparation améliore la qualité
des représentations utilisées ensuite.

Dans notre projet :

- `z` sert à calculer la perte contrastive ;
- `h` sert ensuite à la classification.

### 7.4 Modèle global

```text
image
  -> encodeur ResNet18
  -> h de dimension 512
  -> tête de projection
  -> z de dimension 128
```

Le fichier `simclr/modele.py` assemble ces composants.

---

## 8. Similarité cosinus

### 8.1 Pourquoi comparer des vecteurs ?

Après le passage dans le réseau, les images ne sont plus comparées pixel par
pixel. Le modèle compare leurs vecteurs `z`.

Deux vecteurs orientés dans une direction similaire sont considérés comme
proches.

### 8.2 Formule

Pour deux vecteurs `u` et `v` :

```text
similarité(u, v) = produit scalaire(u, v)
                   / (norme(u) × norme(v))
```

Le **produit scalaire** multiplie les composantes correspondantes puis les
additionne. La **norme** mesure la longueur du vecteur.

La division par les normes retire l'effet de l'amplitude. La comparaison porte
surtout sur l'orientation.

### 8.3 Normalisation

Le code normalise chaque `z` pour que sa norme soit égale à un. La similarité
cosinus devient alors un simple produit matriciel.

### 8.4 Matrice des similarités

Avec `N` images, le projet produit `2N` vues. Une matrice carrée compare toutes
les vues :

```text
[2N, 128] × [128, 2N] -> [2N, 2N]
```

Avec quatre images :

```text
4 images -> 8 vues -> matrice [8, 8]
```

La diagonale vaut un, car chaque vecteur est comparé à lui-même. Cette
auto-comparaison est exclue de la perte.

Chaque ancre possède :

- un positif : l'autre vue de la même image ;
- `2N - 2` négatifs : toutes les vues des autres images.

Avec le lot final de 256 images :

```text
256 images -> 512 vues
511 candidats après exclusion de l'ancre
1 positif
510 négatifs
```

---

## 9. Perte contrastive avec température

### 9.1 Nom complet

La perte utilisée est appelée **entropie croisée normalisée et mise à l'échelle
par une température**. Son nom anglais est *Normalized Temperature-scaled
Cross-Entropy Loss*.

Le code l'appelle `NTXentLoss`, mais le document emploie le nom complet avant
d'utiliser cette forme abrégée.

### 9.2 Objectif

Pour une vue ancre, la perte demande :

> Parmi toutes les autres vues du lot, quelle vue est la seconde transformation
> de la même image source ?

Il s'agit donc d'un problème de classification interne au lot. Le bon candidat
est la vue positive.

### 9.3 Formule

Pour une ancre `i` et son positif `j` :

```text
perte(i, j) = -log(
    exp(similarité(z_i, z_j) / température)
    --------------------------------------------------
    somme sur k différent de i de
    exp(similarité(z_i, z_k) / température)
)
```

Interprétation :

1. le numérateur contient le score du positif ;
2. le dénominateur contient les scores de tous les candidats ;
3. le rapport se comporte comme une probabilité attribuée au positif ;
4. le logarithme négatif pénalise une faible probabilité ;
5. la moyenne est calculée dans les deux directions pour toutes les ancres.

### 9.4 Température

La **température** contrôle la concentration des probabilités.

- Température faible : les différences de similarité sont amplifiées. La
  distribution devient plus tranchée, mais l'entraînement peut devenir
  sensible ou instable.
- Température élevée : les probabilités deviennent plus uniformes. Les
  différences sont moins fortement exploitées.

Le projet utilise 0,5, un compromis courant dans les expériences SimCLR de
petite échelle. Cette valeur doit être présentée comme un choix de protocole,
pas comme une constante universelle.

### 9.5 Taux de positif au premier rang

Le **taux de positif au premier rang** mesure la proportion des ancres dont le
positif obtient le plus grand score parmi tous les candidats.

Avec 512 vues, une sélection purement uniforme parmi 511 candidats aurait une
probabilité d'environ :

```text
1 / 511 ≈ 0,196 %
```

Le taux observé passe de 7,07 % à la première époque à 80,47 % à la centième.
Cela montre que la tâche contrastive est apprise. Cela ne suffit pas à prouver
la qualité de `h` pour reconnaître les classes ; l'évaluation aval remplit ce
rôle.

---

## 10. Préentraînement complet

### 10.1 Flux d'une itération

Pour un lot de 256 images :

```text
256 images sources
  -> deux transformations indépendantes
  -> 256 vues 1 + 256 vues 2
  -> concaténation : [512, 3, 32, 32]
  -> encodeur partagé
  -> h : [512, 512]
  -> tête de projection
  -> z : [512, 128]
  -> séparation en z_1 [256, 128] et z_2 [256, 128]
  -> perte contrastive
  -> rétropropagation
  -> mise à jour de l'encodeur et de la tête de projection
```

Le label CIFAR-10 est chargé par le jeu de données mais stocké dans une variable
explicitement ignorée. Il n'entre jamais dans le modèle pendant cette phase.

### 10.2 Configuration finale

| Élément | Valeur | Rôle et justification |
|---|---:|---|
| Images | 50 000 | Ensemble d'entraînement complet de CIFAR-10 |
| Époques | 100 | Compromis entre apprentissage et budget matériel |
| Taille de lot | 256 images | Fournit 510 négatifs par ancre tout en tenant dans la mémoire graphique |
| Température | 0,5 | Contrôle la concentration des scores contrastifs |
| Encodeur | ResNet18 adapté | Architecture connue, assez expressive et raisonnable sur une carte de 8 gigaoctets |
| Dimension de h | 512 | Sortie naturelle de ResNet18 après retrait de la classification |
| Dimension de z | 128 | Espace compact utilisé par la perte contrastive |
| Optimiseur | AdamW | Optimiseur adaptatif avec décroissance des poids découplée |
| Taux maximal | 0,0003 | Pas d'apprentissage principal du préentraînement |
| Montée progressive | 10 époques | Réduit le risque d'instabilité au démarrage |
| Décroissance | Cosinus jusqu'à 0,000001 | Affine les poids en fin d'entraînement |
| Précision mixte | Activée | Réduit le coût mémoire et accélère les calculs compatibles |
| Graine | 42 | Rend l'expérience principale plus reproductible |

### 10.3 Précision mixte

La **précision mixte** utilise des nombres sur 16 bits pour certains calculs et
conserve des opérations sensibles dans une précision plus élevée. Elle diminue
la consommation mémoire et peut accélérer la carte graphique.

Un mécanisme de mise à l'échelle de la perte évite que de très petits gradients
disparaissent dans la représentation numérique sur 16 bits.

### 10.4 AdamW et décroissance des poids

AdamW ajuste le pas de chaque paramètre à partir de statistiques des gradients.
La **décroissance des poids** pénalise les poids trop grands. Le `W` du nom
indique une application découplée de cette pénalisation.

### 10.5 Montée progressive et décroissance cosinus

Pendant les dix premières époques, le taux d'apprentissage augmente
progressivement. Le modèle évite ainsi des mises à jour trop brutales alors que
les représentations sont encore aléatoires.

Ensuite, le taux diminue selon une courbe de cosinus. Les grands pas du début
explorent l'espace des solutions ; les petits pas de la fin affinent les poids.

### 10.6 Checkpoints

Un **checkpoint** est une sauvegarde de l'état de l'entraînement. Le projet
sauvegarde :

- les poids du modèle ;
- l'état de l'optimiseur ;
- l'état du planificateur de taux d'apprentissage ;
- l'état de la précision mixte ;
- l'historique ;
- les états des générateurs aléatoires.

Cette dernière information permet une reprise plus fidèle après interruption.

Fichier principal :
`outputs/simclr_cifar10_100ep/checkpoint_latest.pt`.

### 10.7 Matériel

L'expérience a utilisé une carte graphique NVIDIA GeForce RTX 4060 pour
ordinateur portable avec 8 gigaoctets de mémoire vidéo. Le pic de mémoire
enregistré est d'environ 1 376 mégaoctets pour les allocations suivies par
PyTorch.

La somme des durées d'époque est d'environ 5 526 secondes, soit 92 minutes.

---

## 11. Résultats du préentraînement

### 11.1 Perte

La perte contrastive moyenne passe de :

```text
5,5419 à l'époque 1
à
4,5502 à l'époque 100
```

La meilleure valeur enregistrée est 4,5488 à l'époque 93.

Cette baisse montre une amélioration selon l'objectif contrastif. La variation
devient faible vers la fin parce que le taux d'apprentissage est lui-même très
faible et que le modèle approche un plateau avec cette configuration.

Une baisse de 0,0001 est mathématiquement une amélioration sur la perte mesurée,
mais elle peut être trop petite pour être significative. Elle peut provenir du
bruit entre les lots. Il faut observer la tendance et surtout l'évaluation aval.

### 11.2 Positif au premier rang

Le taux passe de :

```text
7,07 % à l'époque 1
à
80,47 % à l'époque 100
```

Le maximum enregistré est 80,87 % à l'époque 93.

Le modèle retrouve donc la seconde vue de la même image pour environ quatre
ancres sur cinq à la fin du préentraînement.

### 11.3 Ce que ces résultats prouvent

Ils prouvent que :

- les poids ont appris à résoudre la tâche contrastive ;
- les augmentations ne sont pas complètement incohérentes ;
- l'implémentation fournit un signal d'apprentissage exploitable.

Ils ne prouvent pas encore que :

- le modèle reconnaît les dix classes ;
- la représentation est meilleure qu'une baseline supervisée ;
- les résultats se généralisent à un autre jeu d'images.

---

## 12. Comment évaluer une représentation

### 12.1 Tâche aval

Une **tâche aval** est une tâche réalisée après le préentraînement. Ici, il
s'agit de classifier les images dans les dix classes CIFAR-10.

### 12.2 Évaluation linéaire

Dans l'**évaluation linéaire** :

1. l'encodeur préentraîné produit `h` ;
2. tous ses poids sont gelés ;
3. une seule couche linéaire apprend à associer les 512 valeurs de `h` aux dix
   classes.

```text
h [N, 512] -> couche linéaire -> scores [N, 10]
```

Le mot **gelé** signifie que la rétropropagation ne modifie pas les poids de
l'encodeur.

Cette évaluation répond à la question : les classes sont-elles déjà facilement
séparables dans `h` ?

### 12.3 Fine-tuning

Le **fine-tuning**, ou ajustement fin, part d'un encodeur préentraîné mais
autorise la modification de tout ou partie de ses poids avec les labels.

Cette méthode est plus flexible que l'évaluation linéaire. Elle peut adapter les
caractéristiques à la classification. Elle peut aussi surapprendre lorsque très
peu de labels sont disponibles.

Le fine-tuning est explicitement demandé par le sujet et reste à implémenter
dans les artefacts actuels.

### 12.4 Baseline

Une **baseline** est une méthode de référence utilisée pour donner du sens à un
résultat.

Le projet possède deux comparaisons distinctes :

#### Encodeur aléatoire gelé

ResNet18 reste aléatoire et gelé. Seule la couche linéaire apprend.

Ce contrôle demande : un classifieur linéaire peut-il exploiter des
caractéristiques aléatoires ?

Ce n'est pas une vraie méthode supervisée complète.

#### ResNet18 supervisé depuis zéro

Tous les poids sont initialisés sans préentraînement puis appris avec les
labels. Cette méthode constitue la vraie baseline supervisée.

### 12.5 Accuracy

L'**accuracy**, ou taux de bonnes classifications, vaut :

```text
nombre de prédictions correctes / nombre total d'images
```

Sur 10 000 images de test, une accuracy de 0,80 signifie 8 000 prédictions
correctes.

Comme CIFAR-10 est équilibré, cette métrique résume raisonnablement la
classification globale. Une analyse par classe reste utile pour comprendre les
erreurs.

---

## 13. Protocole 1 %, 10 % et 100 % des labels

### 13.1 Pourquoi ces fractions ?

Les fractions simulent trois régimes :

- 1 %, soit 500 images : labels très rares ;
- 10 %, soit 5 000 images : labels limités ;
- 100 %, soit 50 000 images : tous les labels disponibles.

La question n'est pas seulement « quelle méthode est la meilleure ? », mais «
comment l'avantage change-t-il avec la quantité de supervision humaine ? ».

### 13.2 Stratification

Un sous-ensemble **stratifié** contient un nombre presque égal d'images de
chaque classe.

Avec 500 labels et dix classes, le projet sélectionne environ 50 images par
classe. Cela évite qu'une variation aléatoire crée un sous-ensemble contenant
beaucoup de chats mais très peu de camions.

### 13.3 Sous-ensembles emboîtés

Pour une même graine, le sous-ensemble de 1 % est inclus dans celui de 10 %, lui-
même inclus dans celui de 100 %.

Ainsi, l'expérience augmente la quantité de labels sans remplacer complètement
les exemples précédents.

### 13.4 Graine aléatoire

Une **graine aléatoire** initialise un générateur pseudo-aléatoire. Un ordinateur
produit une suite déterministe de valeurs qui semble aléatoire. Avec la même
graine et le même environnement, on peut retrouver la même suite.

Les graines 42, 123 et 2026 modifient notamment :

- la sélection des images étiquetées ;
- l'ordre des lots ;
- certaines initialisations.

Plusieurs graines permettent de vérifier qu'un résultat n'est pas uniquement
dû à un tirage favorable.

### 13.5 Moyenne et écart-type

La **moyenne** additionne les résultats et divise par le nombre d'exécutions.

L'**écart-type** mesure leur dispersion autour de la moyenne. Un faible
écart-type indique que les exécutions ont donné des résultats proches dans ce
protocole.

`78,84 % ± 0,27 point` signifie ici :

- moyenne de 78,84 % sur trois graines ;
- écart-type de 0,27 point de pourcentage entre ces trois exécutions.

Ce n'est pas une mesure de l'incertitude pour chaque image. Avec seulement trois
graines, il s'agit d'une estimation descriptive encore limitée.

---

## 14. Stabilisation de l'évaluation linéaire

### 14.1 Problème observé

Un premier protocole donnait un résultat SimCLR anormalement instable avec 100
% des labels. Une graine chutait fortement alors que la représentation n'avait
pas changé.

Le problème venait de l'optimisation du classifieur linéaire, pas d'une
dégradation de `h`.

### 14.2 Standardisation de h

Pour chaque dimension de `h`, le code calcule sur le train :

```text
h_standardisé = (h - moyenne_train) / écart_type_train
```

Les statistiques du test ne sont jamais utilisées pour construire cette
transformation. Cela évite une fuite d'information.

La standardisation met les dimensions sur des échelles comparables et rend la
descente de gradient plus stable.

### 14.3 Initialisation à zéro de la couche linéaire

Toutes les expériences démarrent avec les mêmes poids et biais nuls. Les dix
classes reçoivent initialement les mêmes scores.

Dans une couche linéaire directement entraînée par entropie croisée, cette
initialisation n'empêche pas l'apprentissage : les labels produisent des
gradients différents pour les classes.

### 14.4 Décroissance cosinus du taux d'apprentissage

Le taux est assez grand au début pour apprendre rapidement, puis diminue pour
éviter des oscillations autour d'une bonne solution.

### 14.5 Ordre reproductible des lots

Le générateur du chargeur de données reçoit la graine. Cela rend l'ordre des
lots plus contrôlé.

### 14.6 Conséquence

Après correction, l'accuracy augmente régulièrement avec le nombre de labels et
la variance devient faible.

Cette correction ne modifie pas les poids de l'encodeur préentraîné. Elle
améliore uniquement la fiabilité de la mesure.

---

## 15. Résultats de l'évaluation linéaire

### 15.1 Contrôle aléatoire gelé

| Labels | Images étiquetées | Accuracy de test moyenne | Écart-type entre graines |
|---:|---:|---:|---:|
| 1 % | 500 | 26,05 % | 0,60 point |
| 10 % | 5 000 | 33,88 % | 0,60 point |
| 100 % | 50 000 | 42,50 % | 0,10 point |

Même aléatoire, un réseau convolutif produit des transformations structurées.
Une couche linéaire peut exploiter une partie de cette structure. Cependant,
les performances restent bien inférieures à celles de l'encodeur préentraîné.

### 15.2 Encodeur SimCLR gelé

| Labels | Images étiquetées | Accuracy de test moyenne | Écart-type entre graines |
|---:|---:|---:|---:|
| 1 % | 500 | 72,48 % | 0,95 point |
| 10 % | 5 000 | 78,84 % | 0,27 point |
| 100 % | 50 000 | 81,93 % | 0,01 point |

### 15.3 Gains sur le contrôle aléatoire gelé

| Labels | Gain de SimCLR |
|---:|---:|
| 1 % | +46,43 points |
| 10 % | +44,96 points |
| 100 % | +39,44 points |

Le gain reste élevé dans les trois régimes. Le résultat le plus marquant est
qu'avec seulement 500 labels, un simple classifieur linéaire atteint déjà 72,48
% grâce à `h`.

### 15.4 Interprétation bornée

L'expérience permet d'affirmer que le préentraînement a produit une
représentation beaucoup plus facilement séparable par une couche linéaire que
les caractéristiques d'un encodeur aléatoire gelé.

Elle ne permet pas encore d'affirmer que SimCLR est meilleur qu'un réseau
supervisé entraîné de bout en bout, car ce dernier modifie tous ses poids.

---

## 16. Baseline supervisée depuis zéro

### 16.1 Protocole

Pour chaque fraction de labels, un nouveau ResNet18 est initialisé. Tous ses
11 173 962 paramètres apprenables sont modifiés.

Le réseau utilise :

- un recadrage CIFAR-10 avec marge ;
- un retournement horizontal ;
- la descente de gradient stochastique ;
- un momentum de 0,9 ;
- une décroissance des poids de 0,0005 ;
- 100 époques ;
- un taux initial de 0,1 qui suit une décroissance cosinus.

### 16.2 Résultats disponibles pour la graine 42

| Labels | Accuracy train | Accuracy test | Perte train finale | Durée |
|---:|---:|---:|---:|---:|
| 1 % | 77,4 % | 33,37 % | 0,7808 | 56,96 s |
| 10 % | 99,94 % | 67,80 % | 0,0088 | 185,23 s |
| 100 % | 100 % | 94,70 % | 0,0016 | 1 512,77 s |

### 16.3 Surapprentissage

Le **surapprentissage** apparaît lorsqu'un modèle s'adapte très fortement aux
données d'entraînement mais généralise moins bien à de nouvelles images.

Avec 10 % des labels, le modèle atteint presque 100 % sur le train mais 67,80 %
sur le test. La perte train très basse signifie qu'il prédit avec confiance les
images connues. Elle ne garantit pas qu'il a appris une règle générale.

### 16.4 Comparaison appariée sur la graine 42

| Labels | SimCLR gelé | Supervisé depuis zéro | Méthode en tête |
|---:|---:|---:|---|
| 1 % | 73,43 % | 33,37 % | SimCLR, +40,06 points |
| 10 % | 79,15 % | 67,80 % | SimCLR, +11,35 points |
| 100 % | 81,94 % | 94,70 % | Supervisé, +12,76 points |

Avec peu de labels, le préentraînement profite des 50 000 images observées sans
labels. Le réseau supervisé depuis zéro ne voit que 500 ou 5 000 images.

Avec tous les labels, le réseau supervisé modifie tout l'encodeur directement
pour la classification. Le modèle SimCLR de cette comparaison garde son
encodeur gelé. Il est donc raisonnable que le supervisé complet obtienne une
meilleure accuracy.

### 16.5 Limite statistique

La baseline supervisée ne possède actuellement qu'une graine complète. Sa
variabilité n'est pas estimée. Il est incorrect d'écrire `± 0,00` comme si cela
prouvait une stabilité. Les graines 123 et 2026 doivent être terminées.

---

## 17. Évaluation linéaire et fine-tuning : différence essentielle

| Protocole | Poids de l'encodeur | Poids du classifieur | Question mesurée |
|---|---|---|---|
| Évaluation linéaire | Gelés | Entraînés | `h` contient-il déjà des classes séparables ? |
| Fine-tuning | Entraînés, parfois avec un faible taux | Entraînés | Le préentraînement offre-t-il un meilleur point de départ ? |
| Supervisé depuis zéro | Entraînés depuis une initialisation aléatoire | Entraînés | Que peut apprendre le même réseau uniquement avec les labels ? |

Le sujet officiel demande une comparaison de fine-tuning avec et sans
préentraînement. L'évaluation linéaire est très informative mais ne remplace pas
cette exigence.

Un protocole futur correct doit garder constants :

- l'architecture ;
- les sous-ensembles de labels ;
- les graines ;
- le nombre d'époques de fine-tuning ;
- les augmentations supervisées ;
- la métrique et le test.

La seule différence principale doit être l'initialisation : poids SimCLR ou
poids aléatoires.

---

## 18. Étude d'ablation des augmentations

### 18.1 Définition

Une **étude d'ablation** retire ou remplace une composante pour mesurer sa
contribution.

Elle ne consiste pas à essayer plusieurs systèmes très différents puis à
choisir le meilleur. Pour soutenir une conclusion causale limitée, il faut
contrôler les autres variables.

### 18.2 Configurations recommandées

| Configuration | Recadrage | Retournement | Couleurs | Niveaux de gris | Flou |
|---|---|---|---|---|---|
| A | Oui | Non | Non | Non | Non |
| B | Oui | Oui | Non | Non | Non |
| C | Oui | Oui | Oui | Non | Non |
| D | Oui | Oui | Oui | Oui | Non |
| E | Oui | Oui | Oui | Oui | Oui |

L'analyse peut comparer au minimum trois configurations, mais le sujet exige au
moins deux variantes scientifiquement comparées.

### 18.3 Variables à maintenir constantes

- mêmes images ;
- mêmes graines ;
- même ResNet18 ;
- même tête de projection ;
- même taille de lot ;
- même température ;
- même optimiseur ;
- même nombre d'époques ;
- même protocole d'évaluation.

### 18.4 Mesures

Pour chaque configuration :

- perte contrastive finale ;
- taux de positif au premier rang ;
- accuracy de test après évaluation linéaire ;
- moyenne et écart-type ;
- durée ;
- mémoire ;
- exemples visuels des vues.

### 18.5 Conclusion autorisée

Si retirer la perturbation des couleurs réduit l'accuracy dans un protocole
contrôlé, on peut dire qu'elle contribue à la qualité de la représentation dans
ce contexte.

On ne peut pas dire qu'elle est toujours indispensable pour tous les jeux de
données.

---

## 19. Fuite de données

### 19.1 Définition

Une **fuite de données** se produit lorsqu'une information qui devrait être
inconnue pendant l'entraînement influence le modèle ou les décisions
expérimentales.

### 19.2 Exemple simple

Supposons que l'on choisisse le taux d'apprentissage qui donne la meilleure
accuracy sur le test. Le test a alors servi au réglage. Il n'est plus une
évaluation indépendante.

### 19.3 Risques dans ce projet

- transmettre les labels au préentraînement auto-supervisé ;
- standardiser avec les statistiques du test ;
- choisir le meilleur checkpoint selon l'accuracy test ;
- sélectionner les hyperparamètres après avoir observé le test ;
- construire les sous-ensembles à partir d'informations du test.

### 19.4 Protections actuelles

- les labels sont explicitement ignorés pendant le préentraînement ;
- la standardisation de `h` utilise uniquement le train ;
- les sous-ensembles proviennent uniquement des cibles du train ;
- le test officiel reste complet et séparé ;
- les checkpoints sont choisis selon un protocole d'entraînement, pas selon le
  test.

### 19.5 Amélioration souhaitable

Pour une rigueur supérieure, une partie du train devrait devenir un ensemble de
validation. Les hyperparamètres seraient choisis sur la validation, et le test
ne serait consulté qu'après verrouillage du protocole.

---

## 20. Reproductibilité

### 20.1 Ce qui est enregistré

- graines ;
- hyperparamètres ;
- historique au format Comma-Separated Values, c'est-à-dire valeurs séparées
  par des virgules ;
- rapports au format JavaScript Object Notation, un format textuel structuré ;
- checkpoints ;
- état des générateurs pour la reprise du préentraînement ;
- scripts séparés pour les principales expériences.

### 20.2 Pourquoi une reproduction exacte peut rester difficile

Des différences peuvent venir :

- de la version de PyTorch ;
- de la carte graphique ;
- des bibliothèques de calcul ;
- de certaines opérations parallèles ;
- de l'ordre des travailleurs de chargement ;
- de la précision mixte.

La reproductibilité ne signifie pas toujours obtenir le dernier chiffre
décimal identique. Elle signifie aussi retrouver la même conclusion sous un
protocole documenté.

### 20.3 Commandes principales

Visualiser les augmentations :

```powershell
C:\venvs\simclr\Scripts\python.exe -m scripts.visualiser_augmentations
```

Préentraîner SimCLR :

```powershell
C:\venvs\simclr\Scripts\python.exe -m scripts.preentrainer_experimental
```

Reprendre un préentraînement interrompu :

```powershell
C:\venvs\simclr\Scripts\python.exe -m scripts.preentrainer_experimental --resume auto
```

Évaluer 1 %, 10 % et 100 % des labels :

```powershell
C:\venvs\simclr\Scripts\python.exe -m scripts.experimenter_fractions_labels --seeds 42 123 2026
```

Entraîner la baseline supervisée :

```powershell
C:\venvs\simclr\Scripts\python.exe -m scripts.experimenter_supervise_fractions --seeds 42 123 2026
```

Ce dernier script reprend automatiquement les couples fraction et graine déjà
présents dans le rapport.

---

## 21. Organisation de l'implémentation

### 21.1 Modules centraux

| Fichier | Responsabilité |
|---|---|
| `simclr/augmentations.py` | Produit deux vues et définit les transformations |
| `simclr/encodeur.py` | Adapte ResNet18 et produit `h` |
| `simclr/modele.py` | Ajoute la tête de projection et produit `z` |
| `simclr/contrastif.py` | Construit les similarités, les cibles et la perte contrastive |
| `simclr/evaluation.py` | Sélection stratifiée, extraction de `h`, standardisation et évaluation linéaire |

### 21.2 Scripts pédagogiques

| Script | Question vérifiée |
|---|---|
| `inspecter_encodeur.py` | Quelles sont les dimensions avant et après ResNet18 ? |
| `inspecter_projection.py` | Comment passe-t-on de 512 à 128 valeurs ? |
| `inspecter_similarites.py` | Comment se construit la matrice de comparaison ? |
| `inspecter_nt_xent.py` | Où se trouve le positif et comment la température agit-elle ? |
| `preentrainer_simclr.py` | Le pipeline apprend-il sur une petite expérience ? |

### 21.3 Scripts expérimentaux

| Script | Rôle scientifique |
|---|---|
| `preentrainer_experimental.py` | Préentraînement final avec reprise, historique et précision mixte |
| `evaluer_lineaire.py` | Contrôle rapide sur 2 000 labels |
| `experimenter_fractions_labels.py` | Évaluation linéaire sur 1 %, 10 %, 100 % et plusieurs graines |
| `experimenter_supervise_fractions.py` | Baseline supervisée depuis zéro et reprise des résultats |

### 21.4 Tests courts et expériences finales

Une **expérience de débogage** vérifie que le code s'exécute. Elle peut utiliser
une époque ou quelques lots. Ses métriques ne doivent pas figurer dans la
conclusion scientifique.

Une **expérience finale** utilise le protocole annoncé, le jeu complet et le
nombre de graines prévu.

Les dossiers contenant `smoke` sont des sorties de test court. Ils doivent être
exclus des tableaux finaux.

---

## 22. Intérêt réel de la méthode

### 22.1 Réduire la dépendance aux annotations

Le résultat central suggère que la représentation peut être apprise à partir de
nombreuses images non étiquetées, puis adaptée avec un petit ensemble annoté.

Cette stratégie est intéressante lorsque :

- les données brutes sont abondantes ;
- les labels sont coûteux ;
- plusieurs tâches futures peuvent réutiliser le même encodeur.

### 22.2 Transfert d'apprentissage

Le **transfert d'apprentissage** consiste à réutiliser ce qui a été appris sur
une première tâche ou phase pour une seconde.

Ici, la première phase apprend à reconnaître la cohérence entre deux vues. La
seconde utilise `h` pour classifier des objets.

### 22.3 Exemples de domaines potentiels

- imagerie médicale avec beaucoup d'images mais peu de diagnostics validés ;
- inspection industrielle avec peu de défauts annotés ;
- agriculture avec images de plantes et expertise limitée ;
- télédétection avec volumes massifs d'images ;
- catalogues de produits qui évoluent rapidement.

Ces exemples illustrent l'intérêt potentiel. Le projet n'a pas testé ces
domaines et ne doit pas revendiquer une performance réelle dans ces contextes.

### 22.4 Réutilisation

Un encodeur préentraîné peut servir de point de départ à :

- une classification ;
- une détection d'objets ;
- une segmentation ;
- une recherche d'images similaires ;
- un regroupement exploratoire.

La capacité de transfert dépend du domaine et doit être mesurée.

### 22.5 Coût caché

L'apprentissage auto-supervisé économise des labels mais consomme du calcul. Il
ne supprime pas le coût ; il déplace une partie du coût de l'annotation vers le
préentraînement informatique.

L'intérêt réel dépend donc du rapport entre :

- coût des labels ;
- quantité d'images non étiquetées ;
- coût du calcul ;
- nombre de tâches qui réutiliseront l'encodeur ;
- performance nécessaire.

---

## 23. Projection en deux dimensions et démonstration

### 23.1 Pourquoi projeter h ?

`h` possède 512 dimensions. Un humain ne peut pas visualiser directement un
espace à 512 axes.

Une méthode de réduction de dimension construit une carte en deux dimensions.

### 23.2 t-SNE

Le nom complet de t-SNE est **incorporation stochastique de voisins distribuée
en t**. La méthode cherche à préserver surtout les voisinages locaux.

Elle peut produire des groupes visuellement séparés même lorsque les distances
globales sont déformées. Il faut donc éviter d'interpréter trop précisément la
distance entre deux groupes éloignés.

### 23.3 UMAP

UMAP signifie **approximation et projection uniformes de variétés**. La méthode
cherche également une représentation basse dimension en préservant certaines
relations de voisinage.

Les deux méthodes dépendent de paramètres et d'une graine. Une belle figure
n'est pas une preuve quantitative.

### 23.4 Usage correct des labels

Les labels réels peuvent colorer les points après calcul de `h`. Ils ne doivent
pas guider le préentraînement.

### 23.5 Application proposée

Une application simple pourrait permettre :

1. de sélectionner une image CIFAR-10 ;
2. d'afficher deux vues augmentées ;
3. de calculer sa représentation ;
4. d'afficher sa prédiction ;
5. de la situer dans une projection en deux dimensions ;
6. de comparer un encodeur aléatoire et un encodeur SimCLR.

Cette application illustre le comportement. Elle ne remplace pas les métriques
sur l'ensemble du test.

---

## 24. Limites actuelles et conséquences

| Limite | Conséquence | Expérience corrective |
|---|---|---|
| CIFAR-10 uniquement | Généralisation externe non démontrée | Tester STL-10 ou un second domaine |
| Images 32 par 32 | Peu de détails visuels | Tester une résolution ou un jeu plus riche |
| ResNet18 uniquement | Dépendance à une architecture | Comparer une autre taille d'encodeur |
| 100 époques | Budget inférieur aux grands protocoles | Courbe selon 50, 100 et 200 époques |
| Lot de 256 | Moins de négatifs que les grands entraînements | Tester la taille de lot ou une mémoire de négatifs |
| Pas de validation séparée | Réglage méthodologique fragile | Extraire un ensemble de validation du train |
| Linear probe à la place du fine-tuning final | Réponse incomplète au sujet | Implémenter le fine-tuning contrôlé |
| Ablation non terminée | Rôle des augmentations non mesuré | Exécuter les variantes contrôlées |
| Baseline supervisée incomplète en graines | Variabilité non estimée | Terminer 123 et 2026 |
| Pas d'application actuelle | Livrable obligatoire absent | Construire la démonstration |
| Pas de projection actuelle | Analyse qualitative absente | Produire t-SNE ou UMAP |
| Faux négatifs possibles | Deux images de même classe peuvent être éloignées | Discuter ou comparer une méthode adaptée |

### 24.1 Faux négatif contrastif

Deux images sources différentes sont traitées comme négatives, même si elles
représentent toutes deux des chats. Le modèle ne connaît pas les classes et ne
peut pas détecter ce cas.

Cette limite est inhérente au protocole contrastif par instance. Elle ne rend
pas la méthode inutile, mais elle nuance l'interprétation du mot « négatif ».

---

## 25. Feuille de route pour terminer le projet

### Priorité 1 — Terminer la baseline supervisée

- graines 123 et 2026 ;
- rapporter moyenne et écart-type ;
- conserver les résultats par graine.

### Priorité 2 — Implémenter le fine-tuning

- initialisation SimCLR contre initialisation aléatoire ;
- 1 %, 10 %, 100 % ;
- mêmes sous-ensembles et graines ;
- taux d'apprentissage plus faible pour l'encodeur préentraîné ;
- protocole fixé avant le test.

### Priorité 3 — Étude d'ablation

- au moins trois configurations d'augmentations ;
- budget identique ;
- évaluation linéaire identique ;
- plusieurs graines si possible.

### Priorité 4 — Visualisations

- perte et positif au premier rang ;
- accuracy selon les labels ;
- train contre test ;
- projection en deux dimensions ;
- matrice de confusion recommandée.

### Priorité 5 — Application

- interface simple ;
- chargement du checkpoint ;
- visualisation d'une image et de ses vues ;
- prédiction et projection ;
- notice indiquant qu'il s'agit d'une démonstration.

### Priorité 6 — Reproductibilité et dépôt

- initialiser Git ;
- commits thématiques ;
- publier sur GitHub ;
- compléter les dépendances ;
- documenter les commandes ;
- décider comment distribuer les checkpoints volumineux.

### Priorité 7 — Rapport et soutenance

- mettre à jour uniquement avec les résultats finaux ;
- numéroter figures et tableaux ;
- ajouter la répartition du travail ;
- préparer les réponses du jury ;
- vérifier la durée de 20 minutes.

---

## 26. Questions fréquentes

### Le modèle sait-il qu'une image est un chat pendant le préentraînement ?

Non. Il sait seulement que deux vues viennent de la même image source. La classe
« chat » n'est utilisée qu'après le préentraînement.

### Apprend-il déjà des classes cachées ?

Il n'est pas entraîné à produire dix classes. Il apprend des caractéristiques
qui peuvent rendre les classes plus séparables. L'évaluation linéaire mesure
cette propriété.

### Pourquoi z a-t-il 128 valeurs et h 512 ?

`h` est la sortie de ResNet18 et conserve 512 caractéristiques. `z` est une
projection compacte spécialisée pour la perte contrastive. Le nombre 128 est un
choix d'architecture courant et réduit le coût des comparaisons.

### Une loss plus basse signifie-t-elle toujours un meilleur modèle ?

Pour une même loss et un même protocole, une baisse indique une amélioration de
l'objectif d'entraînement. Elle ne garantit pas une meilleure accuracy test.
Une perte contrastive et une perte supervisée ne sont pas directement
comparables.

### Pourquoi le supervisé gagne-t-il avec 100 % des labels ?

Il modifie tous les poids avec l'objectif exact de classification. Dans le
linear probe, l'encodeur SimCLR reste gelé. Le fine-tuning dira si le point de
départ SimCLR peut combiner ses représentations avec une adaptation complète.

### Pourquoi ne pas utiliser les labels pendant SimCLR ?

L'objectif est précisément d'évaluer ce que l'on peut apprendre sans annotation
humaine. Utiliser les labels changerait la méthode et créerait une fuite par
rapport à la question étudiée.

### Le taux de positif au premier rang est-il l'accuracy CIFAR-10 ?

Non. Il mesure si une vue retrouve l'autre vue de la même image dans un lot. Il
ne mesure pas la classe réelle.

### Pourquoi plusieurs graines ?

La sélection des données, l'initialisation et l'ordre des lots peuvent changer
le résultat. Plusieurs graines estiment cette variabilité.

---

## 27. Glossaire pédagogique

**Ablation** : expérience qui retire ou remplace une composante en gardant le
reste aussi constant que possible.

**Accuracy** : proportion de prédictions correctes.

**Activation** : transformation non linéaire appliquée entre des couches.

**AdamW** : optimiseur adaptatif qui applique séparément une décroissance des
poids.

**Apprentissage auto-supervisé** : apprentissage où le signal cible est construit
automatiquement à partir des données.

**Apprentissage contrastif** : apprentissage qui rapproche certaines
représentations et en éloigne d'autres.

**Apprentissage profond** : apprentissage par un réseau possédant plusieurs
niveaux de transformation.

**Apprentissage supervisé** : apprentissage à partir d'exemples accompagnés de
réponses connues.

**Baseline** : méthode de référence utilisée pour interpréter un résultat.

**Biais** : paramètre ajouté à la sortie d'une transformation linéaire ; le mot
peut aussi désigner une distorsion systématique dans les données ou le
protocole, selon le contexte.

**Checkpoint** : sauvegarde des poids et de l'état d'entraînement.

**Classifieur** : système qui associe une entrée à une classe.

**Convolution** : filtre partagé appliqué localement à une image.

**Décroissance des poids** : pénalisation visant à limiter l'amplitude des
paramètres.

**Encodeur** : réseau qui transforme une image en représentation numérique.

**Écart-type** : mesure de dispersion de plusieurs valeurs autour de leur
moyenne.

**Époque** : passage sur l'ensemble d'entraînement.

**Fine-tuning** : ajustement des poids d'un modèle préentraîné pour une tâche
aval.

**Fonction de perte** : quantité que l'entraînement cherche à minimiser.

**Fuite de données** : utilisation indue d'une information réservée à
l'évaluation.

**Graine aléatoire** : valeur initialisant un générateur pseudo-aléatoire.

**Généralisation** : capacité à réussir sur des exemples non utilisés pour
l'apprentissage.

**Gradient** : dérivée indiquant l'influence locale de chaque poids sur la
perte.

**Hyperparamètre** : valeur choisie avant ou autour de l'entraînement, comme la
taille de lot ou le taux d'apprentissage.

**Label** : réponse ou catégorie associée à un exemple.

**Lot** : groupe d'exemples traité ensemble.

**Momentum** : accumulation d'une direction de mise à jour dans un optimiseur.

**Normalisation** : transformation mettant des valeurs sur une échelle mieux
contrôlée.

**Optimiseur** : algorithme qui met à jour les poids à partir des gradients.

**Paire négative** : vues provenant de sources différentes dans le protocole
contrastif.

**Paire positive** : deux vues provenant de la même image source.

**Paramètre** : valeur apprise par le réseau ; les poids et biais sont des
paramètres.

**Passage avant** : calcul des sorties à partir des entrées et des poids
actuels.

**Précision mixte** : utilisation contrôlée de plusieurs précisions numériques
pendant les calculs.

**Préentraînement** : phase d'apprentissage précédant la tâche finale.

**Projection** : transformation de `h` vers l'espace contrastif `z` ; le mot
peut aussi désigner une réduction en deux dimensions pour visualisation.

**Rétropropagation** : calcul des gradients depuis la perte vers les poids.

**Représentation** : vecteur numérique résumant une entrée.

**Réseau résiduel** : réseau utilisant des connexions qui ajoutent une entrée à
une transformation apprise.

**Softmax** : fonction transformant des scores en probabilités dont la somme
vaut un.

**Sous-ensemble stratifié** : sélection préservant approximativement la
répartition des classes.

**Surapprentissage** : adaptation excessive aux données d'entraînement au
détriment de nouvelles données.

**Tâche aval** : tâche réalisée après le préentraînement.

**Taux d'apprentissage** : amplitude principale des mises à jour de poids.

**Température** : hyperparamètre qui contrôle la concentration des scores
contrastifs.

**Tenseur** : tableau numérique à plusieurs dimensions.

**Test** : ensemble réservé à l'évaluation finale.

**Train** : ensemble utilisé pour apprendre ; le terme français recommandé est
ensemble d'entraînement.

**Vue augmentée** : transformation aléatoire d'une image source.

---

## 28. Références principales

1. Ting Chen, Simon Kornblith, Mohammad Norouzi et Geoffrey Hinton. « A Simple
   Framework for Contrastive Learning of Visual Representations ». 2020.
   https://arxiv.org/abs/2002.05709
2. Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie et Ross Girshick. « Momentum
   Contrast for Unsupervised Visual Representation Learning ». 2020.
   https://arxiv.org/abs/1911.05722
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren et Jian Sun. « Deep Residual Learning
   for Image Recognition ». 2015. https://arxiv.org/abs/1512.03385
4. Alex Krizhevsky. « Learning Multiple Layers of Features from Tiny Images ».
   2009. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
5. Page officielle des jeux CIFAR-10 et CIFAR-100.
   https://www.cs.toronto.edu/~kriz/cifar.html

---

## 29. Conclusion pédagogique

Le projet démontre déjà trois idées importantes.

Premièrement, un réseau peut apprendre sans recevoir le nom des objets. En
comparant deux transformations d'une même image, il apprend à produire une
représentation stable.

Deuxièmement, la qualité d'une représentation ne se juge pas uniquement avec la
perte de préentraînement. Elle doit être testée sur une tâche aval. Ici, `h`
permet à une simple couche linéaire d'atteindre 72,48 % d'accuracy moyenne avec
seulement 500 labels.

Troisièmement, l'avantage dépend du régime. Lorsque les labels sont rares, le
préentraînement est très avantageux. Lorsque tous les labels sont disponibles,
un réseau supervisé entraîné de bout en bout atteint 94,70 % sur l'exécution
actuelle, contre 81,94 % pour l'encodeur SimCLR gelé de la même graine.

La prochaine question scientifique n'est donc pas « SimCLR est-il toujours
meilleur ? ». Elle est plus précise :

> À quantité de labels, architecture et budget contrôlés, dans quelles
> conditions le préentraînement offre-t-il un meilleur point de départ, et
> quelles augmentations produisent ce bénéfice ?
