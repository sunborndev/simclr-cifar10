# Apprentissage contrastif auto-supervisé sur CIFAR-10

**Projet 10 — Cours 14, Projets de recherche appliquée (Master)**
KAMBIA Rafiatou · ALLAGLO Kossiko · AMEGNRAN Athanase — Août 2026

De combien d'annotations un pré-entraînement contrastif de type SimCLR dispense-t-il un
classifieur CIFAR-10, à budget de calcul strictement égal ?

| | 450 labels (1 %) | 4 500 (10 %) | 45 000 (100 %) |
|---|---|---|---|
| **Fine-tuning SimCLR** | **72,10 %** | **82,71 %** | **93,56 %** |
| Supervisé *from scratch* | 39,52 % | 70,25 % | 93,13 % |
| *Écart* | *+32,58* | *+12,46* | *+0,43* |

Accuracy top-1 sur les 10 000 images du test officiel, moyenne sur 3 graines.
Le rapport complet est dans [`docs/rapport/rapport_projet10.pdf`](docs/rapport/rapport_projet10.pdf).

---

## Installation

```bash
git clone https://github.com/sunborndev/simclr-cifar10.git
cd simclr-cifar10

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Carte NVIDIA :** installer PyTorch avec le support CUDA *avant* le reste, en suivant
<https://pytorch.org/get-started/locally/>. Le projet fonctionne sur CPU, mais un
pré-entraînement complet y prendrait plusieurs jours au lieu de ~51 minutes sur GPU.

**Données :** CIFAR-10 (170 Mo) se télécharge automatiquement dans `data/` au premier
lancement. Aucune action manuelle.

**Poids pré-entraînés :** les checkpoints font 132 Mo et ne sont pas versionnés (limite
GitHub de 100 Mo par fichier). Le checkpoint principal est publié dans la
[release `v1.0-checkpoint`](https://github.com/sunborndev/simclr-cifar10/releases) ;
le placer dans `outputs/simclr_cifar10_100ep/checkpoint_latest.pt`.

---

## Vérifier une installation en deux minutes

```bash
python -m scripts.inspecter_nt_xent          # la perte contrastive, sur un mini-lot
python -m scripts.inspecter_split_validation # doit afficher « VERDICT : protocole propre »
```

---

## Reproduire les résultats

Chaque chiffre du rapport correspond à une commande unique. La table complète
résultat → commande → durée → fichier de sortie est dans
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

Les trois commandes principales :

```bash
# 1. Pré-entraînement contrastif — 50 000 images, AUCUN label   (~51 min sur RTX 4060)
python -m scripts.preentrainer_experimental --augmentations complet --epochs 100

# 2. La courbe accuracy / % de labels — les quatre bras comparés  (~3 h)
python -m scripts.experimenter_fractions_labels --seeds 42 123 2026 --evaluer-sur test
python -m scripts.experimenter_supervise_fractions --checkpoint outputs/simclr_cifar10_100ep/checkpoint_latest.pt --learning-rate 0.03 --seeds 42 123 2026
python -m scripts.experimenter_supervise_fractions --learning-rate 0.03 --seeds 42 123 2026

# 3. L'étude d'ablation des augmentations — 5 configurations      (~4,5 h)
python -m scripts.reproduire_ablation                        # évaluation à 100 % des labels
python -m scripts.reproduire_ablation --sauter-preentrainement --fraction-probe 0.10
```

Les résultats déjà calculés sont versionnés dans `outputs/` (CSV et JSON) : il est donc
possible de vérifier chaque tableau du rapport **sans relancer le moindre calcul**.

---

## Application de démonstration

```bash
python -m scripts.preparer_demo      # régénère demo/index.html à partir des checkpoints
```

Puis ouvrir `demo/index.html` dans un navigateur. Page autonome : aucun serveur, aucune
dépendance réseau. Elle projette en 2D (t-SNE et PCA) les représentations apprises par six
encodeurs, colorées par la classe réelle — classe qui n'a jamais servi au pré-entraînement.

---

## Architecture du dépôt

```
simclr/                     le cœur de la méthode
├── augmentations.py        chaîne d'augmentations, pilotable brique par brique
├── encodeur.py             ResNet-18 adapté au 32×32 (conv 3×3 pas 1, sans max pooling)
├── modele.py               encodeur + tête de projection 512 → 512 → 128
├── contrastif.py           la perte NT-Xent, écrite à partir de l'équation
└── evaluation.py           découpage stratifié anti-fuite, linear probe, standardisation

scripts/                    tout ce qui s'exécute
├── preentrainer_experimental.py    pré-entraînement contrastif, sans labels
├── experimenter_fractions_labels.py    évaluation linéaire à 1 % / 10 % / 100 %
├── experimenter_supervise_fractions.py fine-tuning et baseline supervisée
├── reproduire_ablation.py          l'ablation complète, en une commande
├── preparer_demo.py                génère l'application de démonstration
├── inspecter_*.py                  vérifications unitaires (perte, encodeur, split…)
└── visualiser_augmentations.py     planche des deux vues SimCLR

outputs/                    résultats versionnés (CSV, JSON) — poids exclus
docs/                       rapport, présentation, documentation technique
tools/                      génération des figures du rapport
01_fondations/              scripts pédagogiques du démarrage du projet
```

### Le pipeline en une phrase

Deux vues augmentées de la même image passent dans un **encodeur partagé** `f`, puis dans
une **tête de projection** `g` ; la perte **NT-Xent** rapproche les deux vues d'une même
image et éloigne toutes les autres du lot. Après le pré-entraînement, `g` est **jetée** et
seul `f` est réutilisé en aval.

---

## Répartition du travail

| Membre | Rôle | Périmètre |
|---|---|---|
| **KAMBIA Rafiatou** | Data / Experiment Engineer | données, augmentations, découpage anti-fuite, balayage du taux d'apprentissage, reproductibilité |
| **ALLAGLO Kossiko** | Model / Research Engineer | encodeur, tête de projection, NT-Xent, pré-entraînement, protocoles d'évaluation, ablation |
| **AMEGNRAN Athanase** | Reporting / Backend Developer | application de démonstration, figures, documentation, rapport et présentation |

Le détail des contributions figure en section 9 du rapport.

---

## Références

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020 — <https://arxiv.org/abs/2002.05709>
- He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*, CVPR 2020 — <https://arxiv.org/abs/1911.05722>
- Grill et al., *Bootstrap Your Own Latent*, NeurIPS 2020 — <https://arxiv.org/abs/2006.07733>
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021 — <https://arxiv.org/abs/2104.14294>
