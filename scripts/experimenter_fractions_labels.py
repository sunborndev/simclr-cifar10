"""Mesure une linear probe avec 1 %, 10 % et 100 % des labels CIFAR-10."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from simclr.augmentations import ECART_TYPE_CIFAR10, MOYENNE_CIFAR10
from simclr.encodeur import EncodeurResNet18
from simclr.evaluation import (
    SEED_SPLIT_VALIDATION,
    charger_encodeur_simclr,
    creer_indices_stratifies,
    entrainer_linear_probe,
    extraire_caracteristiques,
    separer_train_validation,
    standardiser_caracteristiques,
)
from torch.utils.data import Subset


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/simclr_cifar10_100ep/checkpoint_latest.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/simclr_cifar10_100ep/experience_labels_v2"),
    )
    parser.add_argument(
        "--nombre-validation",
        type=int,
        default=5000,
        help="Images retirees du train pour la validation. 0 desactive le split.",
    )
    parser.add_argument(
        "--seed-split",
        type=int,
        default=SEED_SPLIT_VALIDATION,
        help="Seed du decoupage train/validation. Ne jamais la faire varier.",
    )
    parser.add_argument(
        "--evaluer-sur",
        choices=["validation", "test"],
        default="test",
        help="Ensemble sur lequel l'accuracy publiee est mesuree.",
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.10, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-size-extraction", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limite-train", type=int, default=None)
    parser.add_argument("--limite-test", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--standardiser-h",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--scheduler-cosinus",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--initialisation-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def valider_arguments(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {args.checkpoint}")
    if not args.fractions or any(not 0 < fraction <= 1 for fraction in args.fractions):
        raise ValueError("Chaque fraction doit etre comprise entre 0 et 1")
    if len(set(args.fractions)) != len(args.fractions):
        raise ValueError("Les fractions ne doivent pas contenir de doublons")
    if not args.seeds:
        raise ValueError("Il faut fournir au moins une seed")
    if args.nombre_validation < 0:
        raise ValueError("nombre-validation ne peut pas etre negatif")
    if args.evaluer_sur == "validation" and args.nombre_validation == 0:
        raise ValueError(
            "Impossible d'evaluer sur la validation sans en reserver des images"
        )


def restreindre_indices(
    indices: list[int],
    cibles_completes: list[int],
    limite: int | None,
    seed: int,
) -> list[int]:
    """Reduit une liste d'indices en gardant l'equilibre entre classes."""
    if limite is None or limite >= len(indices):
        return indices
    if limite < 10:
        raise ValueError("Une limite doit contenir au moins dix exemples")
    cibles_locales = [cibles_completes[indice] for indice in indices]
    positions = creer_indices_stratifies(cibles_locales, limite, seed)
    return [indices[position] for position in positions]


def creer_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    appareil: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size_extraction,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=appareil.type == "cuda",
    )


def evaluer_fractions(
    nom_encodeur: str,
    encodeur: EncodeurResNet18,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
    appareil: torch.device,
) -> list[dict[str, float | int | str]]:
    encodeur = encodeur.to(appareil)
    for parametre in encodeur.parameters():
        parametre.requires_grad = False

    print(f"\nExtraction des representations : {nom_encodeur}")
    debut_extraction = time.perf_counter()
    h_train, y_train = extraire_caracteristiques(encodeur, train_loader, appareil)
    h_test, y_test = extraire_caracteristiques(encodeur, test_loader, appareil)
    if args.standardiser_h:
        h_train, h_test = standardiser_caracteristiques(h_train, h_test)
    print(
        f"h_train {tuple(h_train.shape)} | h_test {tuple(h_test.shape)} | "
        f"{time.perf_counter() - debut_extraction:.1f} s"
    )

    lignes = []
    for seed in args.seeds:
        ensembles_indices: dict[float, set[int]] = {}
        for fraction in sorted(args.fractions):
            nombre_labels = max(10, round(len(h_train) * fraction))
            indices = creer_indices_stratifies(y_train, nombre_labels, seed)
            ensembles_indices[fraction] = set(indices)
            indices_tenseur = torch.tensor(indices, dtype=torch.long)

            debut_probe = time.perf_counter()
            resultats = entrainer_linear_probe(
                h_train[indices_tenseur],
                y_train[indices_tenseur],
                h_test,
                y_test,
                appareil,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=seed,
                scheduler_cosinus=args.scheduler_cosinus,
                initialisation_zero=args.initialisation_zero,
            )
            duree = time.perf_counter() - debut_probe
            ligne: dict[str, float | int | str] = {
                "encodeur": nom_encodeur,
                "fraction_labels": fraction,
                "nombre_labels": nombre_labels,
                "seed": seed,
                "loss_train": resultats["loss_train"],
                "accuracy_train": resultats["accuracy_train"],
                "accuracy_evaluation": resultats["accuracy_test"],
                "ensemble_evaluation": args.evaluer_sur,
                "duree_probe_secondes": duree,
            }
            lignes.append(ligne)
            print(
                f"{fraction * 100:5.1f} % | {nombre_labels:5d} labels | "
                f"seed {seed:4d} | train {resultats['accuracy_train'] * 100:5.1f} % | "
                f"{args.evaluer_sur} {resultats['accuracy_test'] * 100:5.1f} %"
            )

        fractions_triees = sorted(ensembles_indices)
        for petite, grande in zip(fractions_triees, fractions_triees[1:]):
            if not ensembles_indices[petite].issubset(ensembles_indices[grande]):
                raise RuntimeError("Les sous-ensembles de labels ne sont pas emboites")

    return lignes


def resumer(lignes: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    resume = []
    groupes = sorted(
        {(str(ligne["encodeur"]), float(ligne["fraction_labels"])) for ligne in lignes}
    )
    for encodeur, fraction in groupes:
        groupe = [
            ligne
            for ligne in lignes
            if ligne["encodeur"] == encodeur
            and float(ligne["fraction_labels"]) == fraction
        ]
        accuracies = [float(ligne["accuracy_evaluation"]) for ligne in groupe]
        resume.append(
            {
                "encodeur": encodeur,
                "fraction_labels": fraction,
                "nombre_labels": int(groupe[0]["nombre_labels"]),
                "nombre_seeds": len(accuracies),
                "accuracy_moyenne": statistics.fmean(accuracies),
                "accuracy_ecart_type": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
            }
        )
    return resume


def ecrire_csv(chemin: Path, lignes: list[dict[str, float | int | str]]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        writer.writeheader()
        writer.writerows(lignes)


def main() -> None:
    args = lire_arguments()
    valider_arguments(args)
    appareil = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10),
        ]
    )
    train_complet = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transformation,
    )
    test_complet = CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transformation,
    )
    # Etape 1 : retirer la validation du train officiel, avec le meme decoupage
    # que les experiences de fine-tuning et de baseline supervisee.
    indices_pool, indices_validation = separer_train_validation(
        train_complet.targets,
        args.nombre_validation,
        args.seed_split,
    )
    indices_pool = restreindre_indices(
        indices_pool,
        train_complet.targets,
        args.limite_train,
        args.seeds[0],
    )
    indices_test = restreindre_indices(
        list(range(len(test_complet))),
        test_complet.targets,
        args.limite_test,
        args.seeds[0],
    )

    train_set = Subset(train_complet, indices_pool)
    if args.evaluer_sur == "validation":
        evaluation_set = Subset(train_complet, indices_validation)
    else:
        evaluation_set = Subset(test_complet, indices_test)

    # Garde-fou de fuite : la validation ne doit jamais servir a entrainer.
    if set(indices_pool) & set(indices_validation):
        raise RuntimeError("Fuite detectee entre la reserve etiquetable et la validation")

    train_loader = creer_loader(train_set, args, appareil)
    test_loader = creer_loader(evaluation_set, args, appareil)

    print(f"Appareil                 : {appareil}")
    print(f"Train disponible         : {len(train_set)} images")
    print(
        f"Validation reservee      : {len(indices_validation)} images "
        f"(seed de split {args.seed_split})"
    )
    print(
        f"Evaluation publiee sur   : {args.evaluer_sur} "
        f"({len(evaluation_set)} images)"
    )
    print(f"Fractions                : {args.fractions}")
    print(f"Seeds                    : {args.seeds}")
    print(f"Epochs par linear probe  : {args.epochs}")
    print(f"Standardisation de h     : {args.standardiser_h} (statistiques train)")
    print(f"Scheduler cosinus        : {args.scheduler_cosinus}")
    print(f"Initialisation zero      : {args.initialisation_zero}")
    print("Poids ResNet modifies    : 0")

    torch.manual_seed(args.seeds[0])
    lignes = evaluer_fractions(
        "aleatoire",
        EncodeurResNet18(),
        train_loader,
        test_loader,
        args,
        appareil,
    )
    lignes.extend(
        evaluer_fractions(
            "simclr",
            charger_encodeur_simclr(args.checkpoint),
            train_loader,
            test_loader,
            args,
            appareil,
        )
    )

    resume = resumer(lignes)
    print(f"\nResume accuracy {args.evaluer_sur}")
    for ligne in resume:
        print(
            f"{str(ligne['encodeur']):<10} | "
            f"{float(ligne['fraction_labels']) * 100:5.1f} % | "
            f"{float(ligne['accuracy_moyenne']) * 100:5.1f} % "
            f"+/- {float(ligne['accuracy_ecart_type']) * 100:.2f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chemin_csv = args.output_dir / "resultats.csv"
    chemin_json = args.output_dir / "resultats.json"
    ecrire_csv(chemin_csv, lignes)
    rapport = {
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "fractions": args.fractions,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "standardiser_h": args.standardiser_h,
            "scheduler_cosinus": args.scheduler_cosinus,
            "initialisation_zero": args.initialisation_zero,
            "nombre_validation": args.nombre_validation,
            "seed_split": args.seed_split,
            "evaluer_sur": args.evaluer_sur,
            "nombre_train_total": len(train_set),
            "nombre_evaluation": len(evaluation_set),
        },
        "resultats": lignes,
        "resume": resume,
    }
    chemin_json.write_text(json.dumps(rapport, indent=2), encoding="utf-8")
    print(f"\nCSV  : {chemin_csv.resolve()}")
    print(f"JSON : {chemin_json.resolve()}")


if __name__ == "__main__":
    main()
