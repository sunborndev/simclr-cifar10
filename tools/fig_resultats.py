"""Figures de resultats du Projet 10, lues directement dans les CSV de outputs/."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/tmp/p10")
from style_p10 import (AQUA, BLEU, CLASSES_CIFAR, COULEURS_CLASSES, ENCRE,
                       ENCRE_2, ENCRE_3, GRIS, GRIS_CLAIR, GRILLE, JAUNE,
                       ORANGE, RAMPE_BLEUE, VIOLET, appliquer_style, depouiller,
                       enregistrer)

DATA = Path("/tmp/p10/data/outputs")
appliquer_style()


def lire_csv(chemin):
    with open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =====================================================================
# FIGURE 1 — la courbe centrale : accuracy vs nombre de labels
# =====================================================================
def figure_accuracy_labels():
    lignes = lire_csv(DATA / "tableau2_resultats_consolides.csv")
    series = {}
    for l in lignes:
        series.setdefault(l["methode"], []).append(
            (int(l["nombre_labels"]),
             float(l["accuracy_test_moyenne_pct"]),
             float(l["ecart_type_pct"]))
        )
    for k in series:
        series[k].sort()

    ordre = [
        ("Fine-tuning SimCLR", BLEU, "-", "o"),
        ("Linear probe SimCLR", AQUA, "-", "s"),
        ("Supervise from scratch", ORANGE, "-", "^"),
        ("Encodeur aleatoire gele", GRIS, "--", "D"),
    ]
    noms_affiches = {
        "Fine-tuning SimCLR": "Fine-tuning SimCLR",
        "Linear probe SimCLR": "Linear probe SimCLR",
        "Supervise from scratch": "Supervisé from scratch",
        "Encodeur aleatoire gele": "Encodeur aléatoire (témoin)",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x_ticks = [450, 4500, 45000]

    for nom, couleur, style, marqueur in ordre:
        pts = series[nom]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        e = [p[2] for p in pts]
        ax.errorbar(x, y, yerr=e, color=couleur, linestyle=style, marker=marqueur,
                    markersize=6, capsize=3, elinewidth=1.2, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.2)
        # etiquette directe (regle de relief : jamais l'identite par la couleur seule)
        dy = {"Fine-tuning SimCLR": 7, "Linear probe SimCLR": -7,
              "Supervise from scratch": -7, "Encodeur aleatoire gele": 0}[nom]
        ax.annotate(noms_affiches[nom], xy=(x[-1], y[-1]), xytext=(9, dy),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color=couleur)

    # l'ecart qui porte la demonstration
    ax.annotate("", xy=(450, 72.10), xytext=(450, 39.52),
                arrowprops=dict(arrowstyle="<->", color=ENCRE_3, lw=1.1))
    ax.text(500, 83.5, "+32,6 points\navec 1 % des labels", fontsize=8.5,
            color=ENCRE, fontweight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax.annotate("parité (+0,4)", xy=(45000, 93.3), xytext=(-10, 26),
                textcoords="offset points", ha="right", fontsize=8,
                color=ENCRE_2,
                arrowprops=dict(arrowstyle="-", color=ENCRE_3, lw=0.8))

    ax.set_xscale("log")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["450\n(1 %)", "4 500\n(10 %)", "45 000\n(100 %)"])
    ax.set_xlim(330, 330000)
    ax.set_ylim(18, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20 %", "40 %", "60 %", "80 %", "100 %"])
    ax.set_xlabel("Nombre d'images étiquetées disponibles (échelle logarithmique)")
    ax.set_ylabel("Accuracy sur le test officiel (10 000 images)")
    ax.grid(axis="x", visible=False)
    depouiller(ax)
    enregistrer(fig, "fig_accuracy_labels")


# =====================================================================
# FIGURE 2 — ablation des augmentations : niveau et gain par brique
# =====================================================================
def figure_ablation():
    """Deux regimes d'annotation, meme classement. Une seule unite : des points d'accuracy."""
    l100 = lire_csv(DATA / "ablation" / "tableau_ablation.csv")
    l10 = lire_csv(DATA / "ablation" / "tableau_ablation_10pct.csv")
    noms = {"crop": "crop", "crop_flip": "+ flip",
            "crop_flip_jitter": "+ color jitter", "complet": "+ niveaux de gris",
            "complet_flou": "+ flou gaussien"}
    etiquettes = [noms[l["configuration"]] for l in l100]
    a100 = [float(l["accuracy_probe_moyenne"]) * 100 for l in l100]
    e100 = [float(l["accuracy_probe_ecart_type"]) * 100 for l in l100]
    a10 = [float(l["accuracy_probe_moyenne"]) * 100 for l in l10]
    e10 = [float(l["accuracy_probe_ecart_type"]) * 100 for l in l10]
    g100 = [a100[i] - a100[i - 1] for i in range(1, len(a100))]
    g10 = [a10[i] - a10[i - 1] for i in range(1, len(a10))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- panneau A : niveau atteint, aux deux regimes
    y = np.arange(len(a100))[::-1].astype(float)
    ax1.barh(y + 0.19, a100, height=0.34, color=BLEU, zorder=3)
    ax1.barh(y - 0.19, a10, height=0.34, color=RAMPE_BLEUE[2], zorder=3)
    ax1.errorbar(a100, y + 0.19, xerr=e100, fmt="none", ecolor=ENCRE_3, elinewidth=0.9, capsize=2, zorder=4)
    ax1.errorbar(a10, y - 0.19, xerr=e10, fmt="none", ecolor=ENCRE_3, elinewidth=0.9, capsize=2, zorder=4)
    for yi, v in zip(y + 0.19, a100):
        ax1.text(v + 1.3, yi, f"{v:.2f}".replace(".", ","), va="center", fontsize=7.6,
                 color=ENCRE, fontweight="bold")
    for yi, v in zip(y - 0.19, a10):
        ax1.text(v + 1.3, yi, f"{v:.2f}".replace(".", ","), va="center", fontsize=7.6,
                 color=ENCRE_2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(etiquettes)
    ax1.set_xlim(0, 126)
    ax1.set_xticks([0, 20, 40, 60, 80])
    ax1.set_xticklabels(["0", "20", "40", "60", "80 %"])
    ax1.set_title("Qualité de la représentation, aux deux régimes", loc="left", pad=8)
    ax1.set_xlabel("Accuracy de l'évaluation linéaire")
    ax1.grid(axis="y", visible=False)
    depouiller(ax1, garder=("bottom",))
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor=BLEU, label="100 % des labels"),
                        Patch(facecolor=RAMPE_BLEUE[2], label="10 % des labels")],
               loc="lower right", fontsize=8, handlelength=1.1, handleheight=0.9,
               borderpad=0.3, labelspacing=0.3)

    # --- panneau B : contribution de chaque brique ajoutee
    y2 = np.arange(1, len(a100))[::-1].astype(float)
    ax2.barh(y2 + 0.19, g100, height=0.34,
             color=[ORANGE if v < 1 else BLEU for v in g100], zorder=3)
    ax2.barh(y2 - 0.19, g10, height=0.34,
             color=[ORANGE if v < 1 else RAMPE_BLEUE[2] for v in g10], zorder=3)
    for yi, v in zip(y2 + 0.19, g100):
        ax2.text(v + (0.5 if v >= 0 else -0.5), yi, f"{v:+.2f}".replace(".", ","),
                 va="center", ha="left" if v >= 0 else "right", fontsize=7.6,
                 color=ENCRE, fontweight="bold")
    for yi, v in zip(y2 - 0.19, g10):
        ax2.text(v + (0.5 if v >= 0 else -0.5), yi, f"{v:+.2f}".replace(".", ","),
                 va="center", ha="left" if v >= 0 else "right", fontsize=7.6, color=ENCRE_2)
    ax2.axvline(0, color=ENCRE_3, lw=0.9, zorder=4)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(etiquettes[1:])
    ax2.set_xlim(-6, 22)
    ax2.set_title("Contribution de la brique ajoutée", loc="left", pad=8)
    ax2.set_xlabel("Écart d'accuracy, en points")
    ax2.grid(axis="y", visible=False)
    depouiller(ax2, garder=("bottom",))
    ax2.text(1.0, -0.34,
             "les deux briques de couleur : +22,54 points à 100 %, +26,16 à 10 %",
             transform=ax2.transAxes, ha="right", fontsize=8.5,
             color=BLEU, fontweight="bold")

    fig.subplots_adjust(wspace=0.58)
    enregistrer(fig, "fig_ablation")


# =====================================================================
# FIGURE 3 — l'inversion : bien resoudre la tache pretexte != bien apprendre
# =====================================================================
def figure_inversion():
    lignes = lire_csv(DATA / "ablation" / "tableau_ablation.csv")
    noms = {"crop": "crop seul", "crop_flip": "crop + flip",
            "crop_flip_jitter": "+ color jitter", "complet": "+ niveaux de gris",
            "complet_flou": "+ flou"}
    conf = [noms[l["configuration"]] for l in lignes]
    loss = [float(l["loss_ntxent_finale"]) for l in lignes]
    top1 = [float(l["top1_positif_final"]) * 100 for l in lignes]
    acc = [float(l["accuracy_probe_moyenne"]) * 100 for l in lignes]

    # rang 1 = "le meilleur" selon chaque critere
    rang_pretexte = np.argsort(np.argsort(loss)) + 1            # perte la + basse = rang 1
    rang_aval = np.argsort(np.argsort([-a for a in acc])) + 1   # accuracy la + haute = rang 1

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, nom in enumerate(conf):
        couleur = BLEU if conf[i] == "+ niveaux de gris" else (
            ORANGE if conf[i] == "crop seul" else GRIS)
        largeur = 2.6 if couleur != GRIS else 1.5
        ax.plot([0, 1], [rang_pretexte[i], rang_aval[i]], color=couleur,
                lw=largeur, marker="o", markersize=7, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(-0.045, rang_pretexte[i], nom, ha="right", va="center",
                fontsize=8.5, color=couleur,
                fontweight="bold" if couleur != GRIS else "normal")
        ax.text(1.045, rang_aval[i], nom, ha="left", va="center",
                fontsize=8.5, color=couleur,
                fontweight="bold" if couleur != GRIS else "normal")

    ax.set_xlim(-0.80, 1.80)
    ax.set_ylim(5.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["classé par la PERTE NT-Xent\n(la tâche prétexte)",
                        "classé par l'ACCURACY\n(ce qui nous intéresse)"],
                       fontsize=8.5, fontweight="bold")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([f"{r}er" if r == 1 else f"{r}e" for r in [1, 2, 3, 4, 5]])
    ax.set_ylabel("Rang")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", pad=8)
    depouiller(ax, garder=("left",))
    ax.set_title("Le classement s'inverse presque exactement", loc="left", pad=12)
    enregistrer(fig, "fig_inversion")


# =====================================================================
# FIGURE 4 — courbes de preentrainement
# =====================================================================
def figure_preentrainement():
    hist = lire_csv(DATA / "simclr_cifar10_100ep" / "historique.csv")
    ep = [float(h["epoch"]) for h in hist]
    loss = [float(h["loss"]) for h in hist]
    top1 = [float(h["top1_positif"]) * 100 for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.plot(ep, loss, color=BLEU, zorder=3)
    ax1.set_title("Perte NT-Xent", loc="left", pad=6)
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte")
    ax1.annotate(f"{loss[-1]:.2f}".replace(".", ","), xy=(ep[-1], loss[-1]),
                 xytext=(-6, 10), textcoords="offset points", ha="right",
                 fontsize=8.5, fontweight="bold", color=BLEU)
    depouiller(ax1)

    ax2.plot(ep, top1, color=AQUA, zorder=3)
    ax2.set_title("Positif classé premier", loc="left", pad=6)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Part des ancres")
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0", "25", "50", "75", "100 %"])
    ax2.annotate(f"{top1[-1]:.1f} %".replace(".", ","), xy=(ep[-1], top1[-1]),
                 xytext=(-6, -14), textcoords="offset points", ha="right",
                 fontsize=8.5, fontweight="bold", color=AQUA)
    depouiller(ax2)

    fig.subplots_adjust(wspace=0.32)
    enregistrer(fig, "fig_preentrainement")


# =====================================================================
# FIGURE 5 — balayage symetrique du taux d'apprentissage
# =====================================================================
def figure_lr():
    lr = [0.003, 0.01, 0.03, 0.1, 0.2]
    ft = [81.60, 81.44, 83.52, 82.14, 63.46]
    sup = [60.60, 65.40, 70.46, 68.84, 66.64]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(lr, ft, color=BLEU, marker="o", markersize=6, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(lr, sup, color=ORANGE, marker="^", markersize=6, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate("Fine-tuning SimCLR", xy=(0.003, 81.60), xytext=(4, 11),
                textcoords="offset points", ha="left", fontsize=8.5,
                fontweight="bold", color=BLEU)
    ax.annotate("Supervisé from scratch", xy=(0.003, 60.60), xytext=(4, -16),
                textcoords="offset points", ha="left", fontsize=8.5,
                fontweight="bold", color=ORANGE)

    ax.axvline(0.03, color=ENCRE_3, lw=0.9, ls=":", zorder=2)
    ax.text(0.034, 47.0, "0,03 : optimum\ndes DEUX méthodes", fontsize=8.5,
            color=ENCRE, fontweight="bold", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(lr)
    ax.set_xticklabels(["0,003", "0,01", "0,03", "0,1", "0,2"])
    ax.set_ylim(44, 94)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_yticklabels(["50", "60", "70", "80", "90 %"])
    ax.set_xlabel("Taux d'apprentissage (échelle logarithmique)")
    ax.set_ylabel("Accuracy sur la validation (5 000 images)")
    ax.grid(axis="x", visible=False)
    depouiller(ax)
    enregistrer(fig, "fig_lr")


if __name__ == "__main__":
    figure_accuracy_labels()
    figure_ablation()
    figure_inversion()
    figure_preentrainement()
    figure_lr()
