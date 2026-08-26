"""Figures d'analyse exploratoire et diagnostic du raccourci de couleur."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, "/tmp/p10")
from style_p10 import (AQUA, BLEU, ENCRE, ENCRE_2, ENCRE_3, GRIS, GRIS_CLAIR,
                       ORANGE, RAMPE_BLEUE, appliquer_style, depouiller,
                       enregistrer)

appliquer_style()
A = Path("/mnt/user-data/uploads/projet_annuel_simclr_cifar/docs/assets")

FR = {"airplane": "avion", "automobile": "automobile", "bird": "oiseau", "cat": "chat",
      "deer": "cerf", "dog": "chien", "frog": "grenouille", "horse": "cheval",
      "ship": "bateau", "truck": "camion"}


# =====================================================================
# FIGURE — la couleur identifie l'image, pas la classe
# =====================================================================
def figure_eda_couleur():
    D = json.loads((A / "eda_cifar10.json").read_text(encoding="utf-8"))
    cls = [FR[c] for c in D["classes"]]
    moy = np.array([D["couleur_moyenne_par_classe"][c] for c in D["classes"]])
    intra = np.array([D["dispersion_couleur_intra_classe"][c] for c in D["classes"]])
    inter = D["dispersion_couleur_inter_classe"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- panneau A : la couleur moyenne de chaque classe
    ordre = np.argsort(moy[:, 2] - moy[:, 0])   # du plus chaud au plus froid
    for rang, i in enumerate(ordre[::-1]):
        y = rang
        ax1.add_patch(Rectangle((0, y - 0.34), 1.0, 0.68,
                                facecolor=tuple(moy[i] / 255.0), edgecolor="#d8d8d4", lw=0.7))
        ax1.text(1.15, y, cls[i], va="center", fontsize=8.5, color=ENCRE)
        ax1.text(4.20, y, f"({moy[i,0]:.0f}, {moy[i,1]:.0f}, {moy[i,2]:.0f})",
                 va="center", fontsize=7.5, color=ENCRE_3, ha="right")
    ax1.set_xlim(-0.05, 4.30)
    ax1.set_ylim(-0.75, 9.75)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.grid(False)
    depouiller(ax1, garder=())
    ax1.set_title("Couleur moyenne de chaque classe", loc="left", pad=8)

    # --- panneau B : dispersion intra vs inter
    y = np.arange(len(cls))[::-1]
    ordre2 = np.argsort(-intra)
    ax2.barh(y, intra[ordre2], height=0.6, color=RAMPE_BLEUE[2], zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([cls[i] for i in ordre2], fontsize=8)
    ax2.axvline(inter, color=ORANGE, lw=2, zorder=4)
    ax2.text(inter + 1.2, 4.0, f"écart\nentre classes\n{inter}".replace(".", ","),
             fontsize=8.5, color=ORANGE, fontweight="bold", va="center",
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92))
    ax2.set_xlim(0, 46)
    ax2.set_xlabel("Écart-type de la couleur moyenne (niveaux, 0–255)")
    ax2.set_title("Dispersion à l'intérieur de chaque classe", loc="left", pad=8)
    ax2.grid(axis="y", visible=False)
    depouiller(ax2, garder=("bottom",))

    fig.subplots_adjust(wspace=0.42)
    fig.text(0.5, -0.06,
             "La couleur varie 3 fois plus d'une image à l'autre DANS une classe "
             "qu'entre les classes : elle identifie l'image, pas la catégorie.",
             ha="center", fontsize=8.5, color=ENCRE, fontweight="bold")
    enregistrer(fig, "fig_eda_couleur")


# =====================================================================
# FIGURE — mesure directe du raccourci, sans aucun réseau de neurones
# =====================================================================
def figure_diagnostic_couleur():
    D = json.loads((A / "diagnostic_raccourci_couleur.json").read_text(encoding="utf-8"))
    modes = ["crop", "jitter", "complet"]
    lib = ["crop seul", "crop\n+ color jitter", "crop + jitter\n+ niveaux de gris"]
    top1 = [D[m]["top1_couleur_seule"] for m in modes]
    ratio = [D[m]["ratio"] for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    couleurs = [ORANGE, RAMPE_BLEUE[3], BLEU]
    x = np.arange(3)
    ax1.bar(x, top1, width=0.55, color=couleurs, zorder=3)
    for xi, v in zip(x, top1):
        ax1.text(xi, v + 1.6, f"{v:.1f} %".replace(".", ","), ha="center",
                 fontsize=11, fontweight="bold", color=ENCRE)
    ax1.axhline(0.05, color=ENCRE_3, lw=1, ls=":", zorder=4)
    ax1.text(1.5, 30, "hasard : 0,05 %\n(1 chance sur 2 000)", fontsize=8.5,
             color=ENCRE_3, ha="center", va="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lib, fontsize=8.5)
    ax1.set_ylim(0, 62)
    ax1.set_yticks([0, 20, 40, 60])
    ax1.set_yticklabels(["0", "20", "40", "60 %"])
    ax1.set_ylabel("Bonne image retrouvée au 1er rang\nparmi 2 000 candidates")
    ax1.set_title("Appariement par la COULEUR SEULE", loc="left", pad=8)
    ax1.grid(axis="x", visible=False)
    depouiller(ax1)

    ax2.bar(x, ratio, width=0.55, color=couleurs, zorder=3)
    for xi, v in zip(x, ratio):
        ax2.text(xi, v + 0.08, f"×{v:.2f}".replace(".", ","), ha="center",
                 fontsize=11, fontweight="bold", color=ENCRE)
    ax2.axhline(1.0, color=ENCRE_3, lw=1, ls=":", zorder=4)
    ax2.set_xlim(-0.55, 3.05)
    ax2.text(2.62, 1.0, "×1\naucun\nsignal", fontsize=8, color=ENCRE_3,
             ha="left", va="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["crop", "+ jitter", "+ gris"], fontsize=8.5)
    ax2.set_title("Séparation négatifs / positifs", loc="left", pad=8)
    ax2.set_ylim(0, 3.7)
    ax2.set_ylabel("Distance négative ÷ distance positive")
    ax2.grid(axis="x", visible=False)
    depouiller(ax2)

    fig.subplots_adjust(wspace=0.4)
    fig.text(0.5, -0.10,
             "Diagnostic mené sur les données seules : histogramme 3 × 8 bins, distance L1, "
             "aucun réseau de neurones n'intervient.",
             ha="center", fontsize=8.5, color=ENCRE_2)
    enregistrer(fig, "fig_diagnostic_couleur")


if __name__ == "__main__":
    figure_eda_couleur()
    figure_diagnostic_couleur()
