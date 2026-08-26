"""Schemas conceptuels et projection t-SNE du Projet 10."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, "/tmp/p10")
from style_p10 import (AQUA, BLEU, CLASSES_CIFAR, COULEURS_CLASSES, ENCRE,
                       ENCRE_2, ENCRE_3, GRILLE, GRIS, GRIS_CLAIR, JAUNE,
                       ORANGE, RAMPE_BLEUE, VIOLET, appliquer_style, depouiller,
                       enregistrer)

appliquer_style()
PROJ = Path("/mnt/user-data/uploads/projet_annuel_simclr_cifar/docs/assets/projections_tsne.json")


def boite(ax, x, y, w, h, texte, fc="#ffffff", ec=ENCRE_3, tc=ENCRE,
          taille=8.5, gras=False, rayon=0.02, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rayon}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, texte, ha="center", va="center",
            fontsize=taille, color=tc, zorder=3,
            fontweight="bold" if gras else "normal", linespacing=1.35)


def fleche(ax, p1, p2, couleur=ENCRE_3, style="-|>", lw=1.2, courbe=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, color=couleur, linewidth=lw, linestyle=ls,
        mutation_scale=11, zorder=4,
        connectionstyle=f"arc3,rad={courbe}"))


# =====================================================================
# SCHEMA 1 — le cadre SimCLR
# =====================================================================
def schema_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 50)
    ax.axis("off")

    # image source
    boite(ax, 2, 18, 13, 10, "image x\nsans label", fc="#f4f4f2", gras=True)

    # deux vues
    boite(ax, 23, 31, 16, 11, "vue x̃ᵢ\ncrop + couleur", fc="#e8f2fd", ec=BLEU, tc=BLEU)
    boite(ax, 23, 4, 16, 11, "vue x̃ⱼ\ncrop + gris", fc="#e8f2fd", ec=BLEU, tc=BLEU)
    fleche(ax, (15, 25), (23, 36), courbe=-0.18)
    fleche(ax, (15, 21), (23, 10), courbe=0.18)
    ax.text(15.8, 36.2, r"$t \sim \mathcal{T}$", fontsize=9, color=ENCRE_2)
    ax.text(15.8, 10.4, r"$t' \sim \mathcal{T}$", fontsize=9, color=ENCRE_2)

    # encodeur partage
    boite(ax, 45, 31, 15, 11, "encodeur f(·)\nResNet-18", fc="#ffffff", ec=ENCRE, gras=True)
    boite(ax, 45, 4, 15, 11, "encodeur f(·)\nResNet-18", fc="#ffffff", ec=ENCRE, gras=True)
    fleche(ax, (39, 36.5), (45, 36.5))
    fleche(ax, (39, 9.5), (45, 9.5))
    ax.annotate("poids partagés", xy=(52.5, 30.6), xytext=(52.5, 23),
                ha="center", fontsize=7.5, color=ENCRE_2, va="center")
    ax.plot([52.5, 52.5], [15.4, 20.6], color=ENCRE_3, lw=1, ls=":", zorder=1)

    # h
    ax.text(62.6, 37.6, r"$h_i$", fontsize=11, ha="center", va="center")
    ax.text(62.6, 10.6, r"$h_j$", fontsize=11, ha="center", va="center")
    fleche(ax, (60, 36.5), (65.5, 36.5))
    fleche(ax, (60, 9.5), (65.5, 9.5))

    # tete de projection
    boite(ax, 66, 31, 14, 11, "tête g(·)\nMLP", fc="#fdf1e9", ec=ORANGE, tc=ORANGE)
    boite(ax, 66, 4, 14, 11, "tête g(·)\nMLP", fc="#fdf1e9", ec=ORANGE, tc=ORANGE)

    # z
    ax.text(84, 37.6, r"$z_i$", fontsize=11, ha="center", va="center")
    ax.text(84, 10.6, r"$z_j$", fontsize=11, ha="center", va="center")
    fleche(ax, (80, 36.5), (87, 36.5))
    fleche(ax, (80, 9.5), (87, 9.5))

    # loss
    ax.add_patch(FancyArrowPatch((90, 34), (90, 12), arrowstyle="<|-|>",
                                 color=BLEU, lw=1.8, mutation_scale=12, zorder=4))
    ax.text(92.5, 23, "NT-Xent\nrapprocher", fontsize=8.5, color=BLEU,
            fontweight="bold", va="center", linespacing=1.35)

    # ce qu'on garde
    ax.add_patch(FancyBboxPatch((64.6, 2.4), 17, 41.2,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="none", edgecolor=ORANGE, lw=1.1,
                                linestyle=(0, (4, 3)), zorder=1))
    ax.text(73, -2.6, "tête jetée après le préentraînement", fontsize=8,
            color=ORANGE, ha="center", fontweight="bold")
    ax.add_patch(FancyBboxPatch((44.2, 2.4), 16.6, 41.2,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="none", edgecolor=AQUA, lw=1.4,
                                linestyle=(0, (4, 3)), zorder=1))
    ax.text(52.5, 46.5, "seul l'encodeur f(·) est réutilisé en aval", fontsize=8,
            color=AQUA, ha="center", fontweight="bold")

    enregistrer(fig, "fig_pipeline")


# =====================================================================
# SCHEMA 2 — protocole experimental : donnees et trois evaluations
# =====================================================================
def schema_protocole():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    ax.text(0, 49.5, "CIFAR-10 — 60 000 images 32×32, 10 classes", fontsize=9.5,
            fontweight="bold", color=ENCRE)

    # bandeau train officiel
    boite(ax, 0, 38, 62, 8, "train officiel — 50 000 images", fc="#f4f4f2", taille=8.5)
    boite(ax, 64, 38, 36, 8, "test officiel — 10 000", fc="#f4f4f2", taille=8.5)
    ax.text(82, 34.5, "ouvert une seule fois, à la fin", fontsize=7.5,
            color=ENCRE_2, ha="center", style="italic")

    # split
    fleche(ax, (31, 37.6), (31, 32.5))
    boite(ax, 0, 23, 44, 8, "réserve étiquetable — 45 000", fc="#e8f2fd", ec=BLEU, tc=BLEU, gras=True)
    boite(ax, 46, 23, 16, 8, "validation\n5 000", fc="#fdf1e9", ec=ORANGE, tc=ORANGE, taille=7.5)
    ax.text(54, 19.5, "choix du LR", fontsize=7.5, color=ORANGE, ha="center")

    # fractions
    for x, w, txt in [(0, 5, "1 %\n450"), (6.5, 9, "10 %\n4 500"), (17, 27, "100 %\n45 000")]:
        boite(ax, x, 9, w, 8, txt, fc=RAMPE_BLEUE[1], ec=BLEU, tc="#104281", taille=7.2)
    ax.text(22, 5.5, "trois fractions stratifiées, emboîtées, 3 graines (42 / 123 / 2026)",
            fontsize=7.8, color=ENCRE_2, ha="center")
    fleche(ax, (22, 22.6), (22, 17.5))

    # preentrainement
    ax.add_patch(FancyBboxPatch((66, 8), 34, 22,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="#f0faf6", edgecolor=AQUA, lw=1.2, zorder=1))
    ax.text(83, 26.5, "PRÉENTRAÎNEMENT SimCLR", fontsize=8.2, fontweight="bold",
            color=AQUA, ha="center")
    ax.text(83, 18.5,
            "les 50 000 images du train,\nSANS aucun label\n100 époques, lot 256",
            fontsize=8, color=ENCRE, ha="center", va="center", linespacing=1.5)
    ax.text(83, 10.5, "→ encodeur réutilisé partout", fontsize=7.8,
            color=AQUA, ha="center", fontweight="bold")
    fleche(ax, (50, 37.4), (72, 30.6), couleur=AQUA, lw=1.5, courbe=0.22)
    ax.text(60.5, 36.2, "aucun label", fontsize=7.5, color=AQUA,
            ha="center", fontweight="bold")

    enregistrer(fig, "fig_protocole")


# =====================================================================
# SCHEMA 3 — projection t-SNE des representations
# =====================================================================
def figure_tsne():
    D = json.loads(PROJ.read_text(encoding="utf-8"))
    labels = np.array(D["labels"])
    panneaux = [
        ("aleatoire_tsne", "Encodeur aléatoire\n(jamais entraîné) — 25,7 %"),
        ("crop_tsne", "Préentraîné avec le crop seul\n59,99 %"),
        ("complet_tsne", "Préentraîné, augmentations complètes\n81,87 %"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, (cle, titre) in zip(axes, panneaux):
        P = np.array(D["projections"][cle])
        for c in range(10):
            m = labels == c
            ax.scatter(P[m, 0], P[m, 1], s=2.2, c=COULEURS_CLASSES[c],
                       linewidths=0, alpha=0.75, zorder=3)
        # etiquettes directes aux centroides sur le panneau de droite :
        # l'identite des classes n'est jamais portee par la couleur seule
        if cle == "complet_tsne":
            for c in range(10):
                m = labels == c
                cx, cy = np.median(P[m, 0]), np.median(P[m, 1])
                ax.text(cx, cy, CLASSES_CIFAR[c], fontsize=6.2, ha="center",
                        va="center", color="#111111", fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.78))
        ax.set_title(titre, loc="left", fontsize=8, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRILLE)

    fig.subplots_adjust(wspace=0.08)
    fig.text(0.5, -0.055,
             "2 000 images de test projetées en 2D (t-SNE) à partir de h ∈ ℝ⁵¹²  ·  "
             "couleur = classe réelle, jamais vue au préentraînement",
             ha="center", fontsize=7.5, color=ENCRE_2)
    enregistrer(fig, "fig_tsne")


# =====================================================================
# SCHEMA 4 — le raccourci de couleur (pourquoi crop seul echoue)
# =====================================================================
def schema_raccourci():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(-6, 34); ax.axis("off")

    ax.text(0, 31, "Pourquoi « crop seul » résout la tâche sans rien apprendre",
            fontsize=9.5, fontweight="bold", color=ENCRE)

    boite(ax, 0, 6, 24, 17,
          "crop SEUL\n\nles deux vues gardent\nles mêmes teintes",
          fc="#fdf1e9", ec=ORANGE, tc=ENCRE, taille=7.6)
    fleche(ax, (24.5, 14.5), (28.5, 14.5), couleur=ORANGE)
    boite(ax, 29, 6, 22, 17,
          "le réseau apparie\npar l'histogramme\nde couleur",
          fc="#ffffff", ec=ORANGE, tc=ENCRE, taille=7.6)
    fleche(ax, (51.5, 14.5), (55.5, 14.5), couleur=ORANGE)
    boite(ax, 56, 6, 22, 17, "98,0 % de positifs\nau 1er rang\nperte 4,3739",
          fc="#ffffff", ec=ORANGE, tc=ORANGE, taille=7.6, gras=True)
    fleche(ax, (78.5, 14.5), (82.5, 14.5), couleur=ORANGE)
    boite(ax, 83, 6, 17, 17, "mais 59,99 %\nen aval\n(la pire)",
          fc="#fdf1e9", ec=ORANGE, tc=ORANGE, taille=7.6, gras=True)

    ax.text(0, 1.0,
            "Perturber les couleurs supprime le raccourci : la tâche devient plus dure\n"
            "(perte 4,5502, 80,5 % au 1er rang) et la représentation passe à 81,87 %.",
            fontsize=8, color=ENCRE_2, va="top", linespacing=1.4)
    enregistrer(fig, "fig_raccourci")


if __name__ == "__main__":
    schema_pipeline()
    schema_protocole()
    figure_tsne()
    schema_raccourci()
