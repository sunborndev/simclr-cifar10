# Reproduire chaque résultat du rapport

Ce document répond à l'exigence du sujet : *« script unique pour reproduire chaque résultat
du tableau d'ablation »*. Chaque ligne ci-dessous donne le résultat publié, la commande
exacte qui le régénère, sa durée observée et le fichier produit.

Toutes les commandes se lancent **depuis la racine du dépôt**, avec l'environnement activé.
Sous Windows, remplacer `python` par le chemin complet de l'interpréteur du venv
(`.venv\Scripts\python.exe`).

---

## 0. Vérifications préalables (2 minutes, sans GPU)

| Vérification | Commande | Sortie attendue |
|---|---|---|
| La perte NT-Xent est correcte | `python -m scripts.inspecter_nt_xent` | dimensions et cibles cohérentes, perte initiale ≈ ln(511) ≈ 6,24 |
| Le protocole ne fuit pas | `python -m scripts.inspecter_split_validation` | `VERDICT : protocole propre` |
| L'encodeur est bien adapté au 32×32 | `python -m scripts.inspecter_encodeur` | `h` de dimension 512, pas de max pooling |

Si l'une de ces trois commandes échoue, inutile de lancer un entraînement.

---

## 1. Pré-entraînement contrastif

Aucun label n'est lu. Il produit l'encodeur réutilisé par toutes les évaluations.

| Résultat | Commande | Durée | Sortie |
|---|---|---|---|
| Encodeur principal (configuration `complet`), figure 4 du rapport | `python -m scripts.preentrainer_experimental --augmentations complet --epochs 100` | ≈ 51 min (RTX 4060 Laptop) | `outputs/simclr_cifar10_100ep/` |

Le fichier `historique.csv` produit contient la perte, le taux de positifs au premier rang,
le taux d'apprentissage et le pic mémoire, époque par époque.

> **Reprise après interruption :** ajouter `--resume auto`. Le script repart du dernier
> checkpoint sauvegardé.

---

## 2. Tableau 4 — la courbe accuracy / % de labels

Les quatre bras sont appariés : même découpage, mêmes graines, mêmes sous-ensembles.

| Ligne du tableau | Commande | Durée | Sortie |
|---|---|---|---|
| *Linear probe SimCLR* **et** *Encodeur aléatoire gelé* (les deux en une passe) | `python -m scripts.experimenter_fractions_labels --seeds 42 123 2026 --evaluer-sur test` | ≈ 25 min | `outputs/experience_labels_45k/` |
| *Fine-tuning SimCLR* | `python -m scripts.experimenter_supervise_fractions --checkpoint outputs/simclr_cifar10_100ep/checkpoint_latest.pt --learning-rate 0.03 --seeds 42 123 2026` | ≈ 1 h 15 | `outputs/finetuning_simclr_45k/` |
| *Supervisé from scratch* | `python -m scripts.experimenter_supervise_fractions --learning-rate 0.03 --seeds 42 123 2026` | ≈ 1 h 15 | `outputs/supervise_from_scratch_45k/` |

Les deux dernières commandes ne diffèrent que par la présence de `--checkpoint`. **C'est
volontaire** : c'est ce qui garantit que les deux bras partagent tous leurs
hyperparamètres. Le fichier `resultats.json` de chaque run sérialise la configuration
complète et permet de le vérifier.

---

## 3. Tableau 2 — l'étude d'ablation des augmentations

| Résultat | Commande | Durée | Sortie |
|---|---|---|---|
| Ablation, évaluation à **100 %** des labels | `python -m scripts.reproduire_ablation` | ≈ 4 h 30 | `outputs/ablation/tableau_ablation.csv` |
| Ablation, évaluation à **10 %** des labels | `python -m scripts.reproduire_ablation --sauter-preentrainement --fraction-probe 0.10` | ≈ 15 min | `outputs/ablation/tableau_ablation_10pct.csv` |

La première commande enchaîne les cinq pré-entraînements et les cinq évaluations. Elle
**réutilise** le run principal pour la configuration `complet` au lieu de le recalculer, ce
qui économise ~51 minutes ; utiliser `--forcer-complet` pour le recalculer quand même.

La seconde suppose que les cinq checkpoints existent déjà (`--sauter-preentrainement`) :
elle ne relance que les évaluations linéaires.

> **Rappel de lecture.** Les colonnes *perte NT-Xent* et *positif au 1er rang* sont des
> diagnostics d'optimisation, **pas** des critères de qualité. Le script les affiche en le
> précisant explicitement. Le classement se lit uniquement sur l'accuracy de l'évaluation
> linéaire.

---

## 4. Figure 5 — le balayage du taux d'apprentissage

Balayage **symétrique** : la même grille pour les deux méthodes, sur la validation, au
régime 10 % des labels. Le jeu de test n'est jamais consulté ici.

```bash
for LR in 0.003 0.01 0.03 0.1 0.2; do
  python -m scripts.experimenter_supervise_fractions \
      --checkpoint outputs/simclr_cifar10_100ep/checkpoint_latest.pt \
      --learning-rate $LR --fractions 0.10 --seeds 42 \
      --evaluer-sur validation --output-dir outputs/finetuning_recherche_lr/lr_$LR
  python -m scripts.experimenter_supervise_fractions \
      --learning-rate $LR --fractions 0.10 --seeds 42 \
      --evaluer-sur validation --output-dir outputs/supervise_recherche_lr/lr_$LR
done
```

Durée : ≈ 50 min pour les dix exécutions. Sorties : `outputs/*_recherche_lr/lr_*/`.

---

## 5. Les figures du rapport

Elles se régénèrent à partir des CSV versionnés, **sans GPU et sans les checkpoints** :

```bash
python tools/fig_resultats.py     # courbe, ablation, inversion, préentraînement, LR
python tools/fig_schemas.py       # pipeline SimCLR, protocole, t-SNE, raccourci
python tools/fig_eda.py           # analyse exploratoire, diagnostic du raccourci
```

Sorties : `docs/rapport/fig/*.pdf` (pour LaTeX) et `*.png` 300 dpi (pour la présentation).

Le rapport se compile ensuite avec **xelatex** (deux passes, pour les renvois) :

```bash
cd docs/rapport && xelatex rapport_projet10.tex && xelatex rapport_projet10.tex
```

> `xelatex` est obligatoire : le document utilise `fontspec` et `polyglossia`. `pdflatex`
> échouera.

---

## 6. L'application de démonstration

```bash
python -m scripts.preparer_demo
```

Durée : ≈ 8 min. Nécessite les six checkpoints (l'aléatoire est généré à la volée).
Sortie : `demo/index.html`, page autonome de 7 Mo.

---

## Les graines, et pourquoi elles ne bougent jamais

| Graine | Valeur | Ce qu'elle contrôle |
|---|---|---|
| Découpage train/validation | **7** | quelles 5 000 images forment la validation. **Constante `SEED_SPLIT_VALIDATION` dans `simclr/evaluation.py` — ne jamais la modifier.** |
| Pré-entraînement | **42** | initialisation des poids, ordre des lots, tirages d'augmentations |
| Évaluations | **42, 123, 2026** | trois répétitions dont nous rapportons moyenne et écart-type |

La graine de découpage est **volontairement distincte** des graines d'expérience. Si la
validation changeait d'une graine à l'autre, l'écart-type que nous publions mélangerait la
variabilité de l'entraînement et celle du découpage, et ne mesurerait plus rien
d'interprétable.

---

## Environnement de référence

Les chiffres publiés ont été obtenus dans cette configuration :

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 Go), CUDA, précision mixte activée |
| Pic mémoire GPU | 1 376 Mo, identique pour les cinq pré-entraînements |
| Système | Windows 11, PowerShell |
| Python | 3.11, environnement virtuel dédié |

**Sur les durées.** Celles indiquées plus haut sont des ordres de grandeur calculés à partir
de la **médiane par époque** (29,7 à 33,2 s selon la configuration). Les durées totales
réellement observées vont de 53 à 914 minutes pour des configurations dont le coût est
identique à 12 % près : la machine était partagée avec d'autres charges. C'est la raison
pour laquelle le rapport ne publie **aucune** comparaison de temps entre méthodes.
