"""Verifie le decoupage train / validation avant de lancer la moindre experience.

Ce script n'entraine rien. Il repond a quatre questions, dans l'ordre :

1. combien d'images restent pour l'entrainement, combien partent en validation ?
2. les dix classes sont-elles equilibrees des deux cotes ?
3. le decoupage est-il bien IDENTIQUE pour les seeds 42, 123 et 2026 ?
4. les sous-ensembles 1 %, 10 % et 100 % touchent-ils la validation ?

La question 4 est la plus importante : une seule image de validation etiquetee
pendant l'entrainement suffit a fausser le choix des hyperparametres.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from torchvision.datasets import CIFAR10

from simclr.evaluation import (
    SEED_SPLIT_VALIDATION,
    creer_indices_stratifies,
    separer_train_validation,
)


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--nombre-validation", type=int, default=5000)
    parser.add_argument("--seed-split", type=int, default=SEED_SPLIT_VALIDATION)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.10, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    return parser.parse_args()


def afficher_equilibre(nom: str, cibles: list[int], classes: list[str]) -> None:
    compteur = Counter(cibles)
    total = len(cibles)
    detail = "  ".join(
        f"{classes[classe][:5]}:{compteur[classe]}" for classe in sorted(compteur)
    )
    minimum = min(compteur.values())
    maximum = max(compteur.values())
    print(f"{nom:<26} {total:6d} images | {detail}")
    print(f"{'':26} ecart max entre classes : {maximum - minimum}")


def main() -> None:
    args = lire_arguments()
    train = CIFAR10(root=args.data_dir, train=True, download=False)
    classes = train.classes
    cibles = list(train.targets)

    indices_train, indices_validation = separer_train_validation(
        cibles,
        args.nombre_validation,
        args.seed_split,
    )
    ensemble_validation = set(indices_validation)

    print("=" * 78)
    print("1. Tailles")
    print("=" * 78)
    print(f"Train officiel CIFAR-10    : {len(cibles)} images")
    print(f"-> entrainement disponible : {len(indices_train)} images")
    print(f"-> validation reservee     : {len(indices_validation)} images")
    print(f"Recouvrement               : {len(ensemble_validation & set(indices_train))}")

    print()
    print("=" * 78)
    print("2. Equilibre des classes")
    print("=" * 78)
    afficher_equilibre(
        "Entrainement",
        [cibles[indice] for indice in indices_train],
        classes,
    )
    afficher_equilibre(
        "Validation",
        [cibles[indice] for indice in indices_validation],
        classes,
    )

    print()
    print("=" * 78)
    print("3. Stabilite du decoupage")
    print("=" * 78)
    def empreinte(indices: list[int]) -> str:
        return hashlib.sha256(
            ",".join(str(indice) for indice in indices).encode()
        ).hexdigest()[:16]

    reference = empreinte(indices_validation)
    print(f"  seed-split {args.seed_split:3d} | empreinte {reference}  <- reference")
    for essai in range(2):
        _, validation_bis = separer_train_validation(
            cibles,
            args.nombre_validation,
            args.seed_split,
        )
        egal = empreinte(validation_bis) == reference
        print(
            f"  seed-split {args.seed_split:3d} | empreinte {empreinte(validation_bis)}"
            f"  | rappel {essai + 1} identique : {egal}"
        )
    _, validation_autre = separer_train_validation(
        cibles,
        args.nombre_validation,
        args.seed_split + 1,
    )
    print(
        f"  seed-split {args.seed_split + 1:3d} | empreinte "
        f"{empreinte(validation_autre)}  | doit differer : "
        f"{empreinte(validation_autre) != reference}"
    )
    print()
    print("  Le decoupage ne depend que de --seed-split. Les seeds d'experience")
    print("  (42, 123, 2026) ne le touchent pas : les trois voient la meme")
    print("  validation, donc l'ecart-type ne mesure qu'une seule source de bruit.")

    print()
    print("=" * 78)
    print("4. Absence de fuite dans les sous-ensembles etiquetes")
    print("=" * 78)
    cibles_pool = [cibles[indice] for indice in indices_train]
    tout_est_propre = True
    for seed in args.seeds:
        ensembles: dict[float, set[int]] = {}
        for fraction in sorted(args.fractions):
            nombre_labels = max(10, round(len(indices_train) * fraction))
            positions = creer_indices_stratifies(cibles_pool, nombre_labels, seed)
            absolus = {indices_train[position] for position in positions}
            ensembles[fraction] = absolus
            fuite = len(absolus & ensemble_validation)
            tout_est_propre = tout_est_propre and fuite == 0
            print(
                f"  seed {seed:4d} | {fraction * 100:5.1f} % | "
                f"{nombre_labels:5d} labels | fuite validation : {fuite}"
            )
        fractions_triees = sorted(ensembles)
        for petite, grande in zip(fractions_triees, fractions_triees[1:]):
            emboite = ensembles[petite].issubset(ensembles[grande])
            tout_est_propre = tout_est_propre and emboite
            print(
                f"  seed {seed:4d} | {petite * 100:.0f} % inclus dans "
                f"{grande * 100:.0f} % : {emboite}"
            )

    print()
    print("=" * 78)
    print("VERDICT :", "protocole propre" if tout_est_propre else "PROBLEME DETECTE")
    print("=" * 78)


if __name__ == "__main__":
    main()
