"""Style commun a toutes les figures du Projet 10.

Palette validee (validate_palette.js, mode clair, --pairs all) :
  bleu #2a78d6, aqua #1baf7a, orange #eb6834 -> ALL CHECKS PASS.
Le WARN de contraste sur l'aqua est leve par la regle de relief :
toutes les series portent une etiquette directe visible.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- roles de couleur -------------------------------------------------------
BLEU = "#2a78d6"      # methode principale (fine-tuning SimCLR)
AQUA = "#1baf7a"      # second protocole SSL (linear probe)
ORANGE = "#eb6834"    # baseline concurrente (supervise from scratch)
JAUNE = "#eda100"     # 4e slot, usage ponctuel
VIOLET = "#4a3aa7"
GRIS = "#8f8f89"      # temoin diagnostic, jamais une "serie"
GRIS_CLAIR = "#c9c8c3"

ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
ENCRE_3 = "#7a7975"
GRILLE = "#e6e5e1"
SURFACE = "#ffffff"

# ramp sequentiel bleu (une seule teinte, clair -> fonce)
RAMPE_BLEUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

# 10 classes CIFAR-10 : l'identite n'est jamais portee par la couleur seule,
# chaque amas recoit une etiquette directe a son centroide.
CLASSES_CIFAR = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
]
COULEURS_CLASSES = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948", "#00a3c4", "#8a6d3b",
]


def appliquer_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": ENCRE,
        "axes.labelsize": 9,
        "axes.labelcolor": ENCRE_2,
        "axes.edgecolor": GRILLE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRILLE,
        "grid.linewidth": 0.7,
        "xtick.color": ENCRE_3,
        "ytick.color": ENCRE_3,
        "xtick.labelcolor": ENCRE_2,
        "ytick.labelcolor": ENCRE_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def depouiller(ax, garder=("left", "bottom")):
    for cote in ("top", "right", "left", "bottom"):
        ax.spines[cote].set_visible(cote in garder)


def enregistrer(fig, nom, dossier="/tmp/p10/fig"):
    """Ecrit la figure en PDF vectoriel (LaTeX) et en PNG 300 dpi (PowerPoint)."""
    from pathlib import Path
    d = Path(dossier)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{nom}.pdf")
    fig.savefig(d / f"{nom}.png", dpi=300)
    plt.close(fig)
    print(f"  -> {nom}.pdf / {nom}.png")
