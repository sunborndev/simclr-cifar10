"""Entraine un ResNet18 sur plusieurs fractions de labels CIFAR-10.

Un seul script couvre les deux branches de la comparaison exigee par le sujet :

- `--checkpoint-simclr` absent  -> encodeur initialise au hasard (supervise
  "from scratch"), c'est la baseline concurrente ;
- `--checkpoint-simclr CHEMIN`  -> encodeur initialise avec les poids appris
  sans labels par SimCLR, puis entierement reentrainable (fine-tuning).

Tout le reste du protocole est partage ligne pour ligne : memes augmentations,
meme optimiseur, meme scheduler, memes epochs, memes seeds, memes
sous-ensembles stratifies emboites. C'est ce partage qui rend la comparaison
valide : entre les deux branches, SEULE l'initialisation des poids change.
"""

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from simclr.augmentations import ECART_TYPE_CIFAR10, MOYENNE_CIFAR10
from simclr.encodeur import DIMENSION_H, EncodeurResNet18
from simclr.evaluation import (
    SEED_SPLIT_VALIDATION,
    charger_encodeur_simclr,
    creer_indices_stratifies,
    separer_train_validation,
)


class ClassificateurResNet18(nn.Module):
    """ResNet18 CIFAR-10 suivi d'une couche de classification a dix classes."""

    def __init__(self, checkpoint_simclr: Path | None = None) -> None:
        super().__init__()
        if checkpoint_simclr is None:
            self.encodeur = EncodeurResNet18()
        else:
            self.encodeur = charger_encodeur_simclr(checkpoint_simclr)
        self.classificateur = nn.Linear(DIMENSION_H, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classificateur(self.encodeur(images))


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/supervise_from_scratch"),
    )
    parser.add_argument(
        "--checkpoint-simclr",
        type=Path,
        default=None,
        help="Checkpoint SimCLR pour le fine-tuning. Absent = from scratch.",
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--limite-train", type=int, default=None)
    parser.add_argument("--limite-test", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--resume-results",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def valider_arguments(args: argparse.Namespace) -> None:
    if not args.fractions or any(not 0 < fraction <= 1 for fraction in args.fractions):
        raise ValueError("Chaque fraction doit etre comprise entre 0 et 1")
    if len(set(args.fractions)) != len(args.fractions):
        raise ValueError("Les fractions ne doivent pas contenir de doublons")
    if not args.seeds:
        raise ValueError("Il faut fournir au moins une seed")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs et batch-size doivent etre positifs")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Les learning rates doivent etre positifs")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate ne peut pas depasser learning-rate")
    if args.num_workers < 0 or args.print_every < 0:
        raise ValueError("num-workers et print-every ne peuvent pas etre negatifs")
    if args.nombre_validation < 0:
        raise ValueError("nombre-validation ne peut pas etre negatif")
    if args.evaluer_sur == "validation" and args.nombre_validation == 0:
        raise ValueError(
            "Impossible d'evaluer sur la validation sans en reserver des images"
        )
    if args.checkpoint_simclr is not None and not args.checkpoint_simclr.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {args.checkpoint_simclr}")


def nom_methode(args: argparse.Namespace) -> str:
    return (
        "supervise_from_scratch"
        if args.checkpoint_simclr is None
        else "finetuning_simclr"
    )


def creer_transformations() -> tuple[transforms.Compose, transforms.Compose]:
    transformation_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10),
        ]
    )
    transformation_evaluation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10),
        ]
    )
    return transformation_train, transformation_evaluation


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


def fixer_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def calculer_accuracy(
    modele: nn.Module,
    loader: DataLoader,
    appareil: torch.device,
) -> float:
    modele.eval()
    nombre_correct = 0
    nombre_total = 0
    for images, etiquettes in loader:
        images = images.to(appareil, non_blocking=True)
        etiquettes = etiquettes.to(appareil, non_blocking=True)
        predictions = modele(images).argmax(dim=1)
        nombre_correct += (predictions == etiquettes).sum().item()
        nombre_total += etiquettes.size(0)
    return nombre_correct / nombre_total


def creer_loader(
    dataset: Subset,
    batch_size: int,
    appareil: torch.device,
    num_workers: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generateur = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=appareil.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generateur,
    )


def entrainer_configuration(
    train_dataset: Subset,
    train_evaluation_dataset: Subset,
    evaluation_dataset: Subset,
    args: argparse.Namespace,
    appareil: torch.device,
    seed: int,
    description: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    fixer_seed(seed)
    amp_active = appareil.type == "cuda" and not args.no_amp
    train_loader = creer_loader(
        train_dataset,
        args.batch_size,
        appareil,
        args.num_workers,
        shuffle=True,
        seed=seed,
    )
    train_evaluation_loader = creer_loader(
        train_evaluation_dataset,
        args.batch_size,
        appareil,
        args.num_workers,
        shuffle=False,
    )
    evaluation_loader = creer_loader(
        evaluation_dataset,
        args.batch_size,
        appareil,
        args.num_workers,
        shuffle=False,
    )

    modele = ClassificateurResNet18(args.checkpoint_simclr).to(appareil)
    optimiseur = SGD(
        modele.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(
        optimiseur,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler(device=appareil.type, enabled=amp_active)
    fonction_loss = nn.CrossEntropyLoss()
    debut = time.perf_counter()
    derniere_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        modele.train()
        somme_loss = 0.0
        nombre_total = 0
        nombre_batches = 0
        learning_rate = optimiseur.param_groups[0]["lr"]

        for images, etiquettes in train_loader:
            images = images.to(appareil, non_blocking=True)
            etiquettes = etiquettes.to(appareil, non_blocking=True)
            optimiseur.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=appareil.type,
                dtype=torch.float16,
                enabled=amp_active,
            ):
                scores = modele(images)
                loss = fonction_loss(scores, etiquettes)

            scaler.scale(loss).backward()
            scaler.step(optimiseur)
            scaler.update()

            somme_loss += loss.item() * etiquettes.size(0)
            nombre_total += etiquettes.size(0)
            nombre_batches += 1
            if args.max_batches is not None and nombre_batches >= args.max_batches:
                break

        derniere_loss = somme_loss / nombre_total
        scheduler.step()
        if args.print_every and (
            epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs
        ):
            print(
                f"  {description} | epoch {epoch:03d}/{args.epochs} | "
                f"loss {derniere_loss:.4f} | lr {learning_rate:.2e}"
            )

    resultats = {
        "loss_train": derniere_loss,
        "accuracy_train": calculer_accuracy(
            modele,
            train_evaluation_loader,
            appareil,
        ),
        "accuracy_evaluation": calculer_accuracy(modele, evaluation_loader, appareil),
        "duree_secondes": time.perf_counter() - debut,
    }
    etat_modele = None
    if args.save_checkpoints:
        etat_modele = {
            nom: valeur.detach().cpu() for nom, valeur in modele.state_dict().items()
        }
    return resultats, etat_modele


def resumer(
    lignes: list[dict[str, float | int | str]],
    methode: str,
    ensemble_evaluation: str,
) -> list[dict[str, float | int | str]]:
    resume = []
    for fraction in sorted({float(ligne["fraction_labels"]) for ligne in lignes}):
        groupe = [
            ligne for ligne in lignes if float(ligne["fraction_labels"]) == fraction
        ]
        accuracies = [float(ligne["accuracy_evaluation"]) for ligne in groupe]
        resume.append(
            {
                "methode": methode,
                "ensemble_evaluation": ensemble_evaluation,
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


def construire_configuration(
    args: argparse.Namespace,
    amp_active: bool,
    nombre_train_total: int,
    nombre_evaluation: int,
) -> dict[str, object]:
    return {
        "methode": nom_methode(args),
        "checkpoint_simclr": (
            None if args.checkpoint_simclr is None else str(args.checkpoint_simclr)
        ),
        "fractions": args.fractions,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "mixed_precision": amp_active,
        "nombre_validation": args.nombre_validation,
        "seed_split": args.seed_split,
        "evaluer_sur": args.evaluer_sur,
        "nombre_train_total": nombre_train_total,
        "nombre_evaluation": nombre_evaluation,
        "max_batches": args.max_batches,
        "augmentation_train": "RandomCrop(32, padding=4) + RandomHorizontalFlip",
    }


CLES_PROTOCOLE = [
    "methode",
    "checkpoint_simclr",
    "epochs",
    "batch_size",
    "learning_rate",
    "min_learning_rate",
    "momentum",
    "weight_decay",
    "mixed_precision",
    "nombre_validation",
    "seed_split",
    "evaluer_sur",
    "nombre_train_total",
    "nombre_evaluation",
    "max_batches",
    "augmentation_train",
]


def charger_resultats_existants(
    chemin_json: Path,
    configuration: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    if not args.resume_results or not chemin_json.exists():
        return []

    rapport = json.loads(chemin_json.read_text(encoding="utf-8"))
    ancienne_configuration = rapport.get("configuration", {})
    for cle in CLES_PROTOCOLE:
        if ancienne_configuration.get(cle) != configuration[cle]:
            raise ValueError(
                f"Reprise impossible : la configuration '{cle}' a change "
                f"({ancienne_configuration.get(cle)!r} -> {configuration[cle]!r}). "
                "Utiliser un autre --output-dir ou --no-resume-results."
            )

    fractions_demandees = {round(float(fraction), 12) for fraction in args.fractions}
    seeds_demandees = set(args.seeds)
    lignes: list[dict[str, float | int | str]] = []
    cles_vues: set[tuple[int, float]] = set()
    for ligne_brute in rapport.get("resultats", []):
        ligne = dict(ligne_brute)
        seed = int(ligne["seed"])
        fraction = round(float(ligne["fraction_labels"]), 12)
        if seed not in seeds_demandees or fraction not in fractions_demandees:
            raise ValueError(
                "Le rapport existant contient une seed ou une fraction non demandee. "
                "Utiliser un autre --output-dir ou --no-resume-results."
            )
        cle = (seed, fraction)
        if cle in cles_vues:
            raise ValueError(f"Resultat duplique dans le rapport : {cle}")
        cles_vues.add(cle)
        lignes.append(ligne)

    if lignes:
        print(f"Resultats repris          : {len(lignes)} configurations")
    return lignes


def sauvegarder_rapport(
    output_dir: Path,
    configuration: dict[str, object],
    lignes: list[dict[str, float | int | str]],
    methode: str,
    ensemble_evaluation: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chemin_csv = output_dir / "resultats.csv"
    chemin_json = output_dir / "resultats.json"
    ecrire_csv(chemin_csv, lignes)
    rapport = {
        "configuration": configuration,
        "resultats": lignes,
        "resume": resumer(lignes, methode, ensemble_evaluation),
    }
    chemin_temporaire = chemin_json.with_suffix(".json.tmp")
    chemin_temporaire.write_text(json.dumps(rapport, indent=2), encoding="utf-8")
    chemin_temporaire.replace(chemin_json)
    return chemin_csv, chemin_json


def main() -> None:
    args = lire_arguments()
    valider_arguments(args)
    methode = nom_methode(args)
    appareil = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    amp_active = appareil.type == "cuda" and not args.no_amp
    transformation_train, transformation_evaluation = creer_transformations()

    train_augmentation = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transformation_train,
    )
    train_evaluation = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transformation_evaluation,
    )
    test_complet = CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transformation_evaluation,
    )

    # Etape 1 : retirer la validation du train officiel, une fois pour toutes.
    indices_pool, indices_validation = separer_train_validation(
        train_augmentation.targets,
        args.nombre_validation,
        args.seed_split,
    )

    # Etape 2 : restreindre eventuellement pour un test rapide de plomberie.
    indices_pool = restreindre_indices(
        indices_pool,
        train_augmentation.targets,
        args.limite_train,
        args.seeds[0],
    )
    indices_test = restreindre_indices(
        list(range(len(test_complet))),
        test_complet.targets,
        args.limite_test,
        args.seeds[0],
    )

    if args.evaluer_sur == "validation":
        evaluation_dataset = Subset(train_evaluation, indices_validation)
    else:
        evaluation_dataset = Subset(test_complet, indices_test)

    cibles_pool = [train_augmentation.targets[indice] for indice in indices_pool]
    configuration = construire_configuration(
        args,
        amp_active,
        len(indices_pool),
        len(evaluation_dataset),
    )
    chemin_json = args.output_dir / "resultats.json"
    lignes = charger_resultats_existants(chemin_json, configuration, args)
    cles_terminees = {
        (int(ligne["seed"]), round(float(ligne["fraction_labels"]), 12))
        for ligne in lignes
    }

    print(f"Methode                  : {methode}")
    if args.checkpoint_simclr is not None:
        print(f"Poids initiaux           : {args.checkpoint_simclr}")
    print(f"Appareil                 : {appareil}")
    if appareil.type == "cuda":
        print(f"GPU                      : {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision          : {amp_active}")
    print(f"Train disponible         : {len(indices_pool)} images")
    print(
        f"Validation reservee      : {len(indices_validation)} images "
        f"(seed de split {args.seed_split}, jamais etiquetee pour l'entrainement)"
    )
    print(f"Evaluation publiee sur   : {args.evaluer_sur} ({len(evaluation_dataset)} images)")
    print(f"Fractions                : {args.fractions}")
    print(f"Seeds                    : {args.seeds}")
    print(f"Epochs par modele        : {args.epochs}")
    print(f"Learning rate            : {args.learning_rate}")
    print("Poids ResNet modifies    : tous")

    for seed in args.seeds:
        ensembles_indices: dict[float, set[int]] = {}
        for fraction in sorted(args.fractions):
            nombre_labels = max(10, round(len(indices_pool) * fraction))
            positions = creer_indices_stratifies(cibles_pool, nombre_labels, seed)
            indices_absolus = [indices_pool[position] for position in positions]
            ensembles_indices[fraction] = set(indices_absolus)

            # Garde-fou de fuite : aucun indice de validation ne doit entrer ici.
            if set(indices_validation) & ensembles_indices[fraction]:
                raise RuntimeError(
                    "Fuite detectee : des images de validation sont etiquetees"
                )

            train_dataset = Subset(train_augmentation, indices_absolus)
            train_evaluation_dataset = Subset(train_evaluation, indices_absolus)
            description = (
                f"{fraction * 100:5.1f} % | {nombre_labels:5d} labels | seed {seed:4d}"
            )
            cle_configuration = (seed, round(float(fraction), 12))
            if cle_configuration in cles_terminees:
                print(f"  {description} | deja calcule -> ignore")
                continue

            resultats, etat_modele = entrainer_configuration(
                train_dataset,
                train_evaluation_dataset,
                evaluation_dataset,
                args,
                appareil,
                seed,
                description,
            )
            ligne: dict[str, float | int | str] = {
                "methode": methode,
                "fraction_labels": fraction,
                "nombre_labels": nombre_labels,
                "seed": seed,
                "loss_train": resultats["loss_train"],
                "accuracy_train": resultats["accuracy_train"],
                "accuracy_evaluation": resultats["accuracy_evaluation"],
                "ensemble_evaluation": args.evaluer_sur,
                "duree_secondes": resultats["duree_secondes"],
            }
            lignes.append(ligne)
            cles_terminees.add(cle_configuration)
            print(
                f"  resultat | train {resultats['accuracy_train'] * 100:5.1f} % | "
                f"{args.evaluer_sur} {resultats['accuracy_evaluation'] * 100:5.1f} % | "
                f"{resultats['duree_secondes']:.1f} s"
            )

            if args.save_checkpoints:
                if etat_modele is None:
                    raise RuntimeError("L'etat du modele n'a pas ete conserve")
                checkpoint_dir = args.output_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                nom_fraction = str(fraction).replace(".", "p")
                torch.save(
                    {
                        "modele": etat_modele,
                        "methode": methode,
                        "fraction_labels": fraction,
                        "nombre_labels": nombre_labels,
                        "seed": seed,
                        "configuration": vars(args),
                    },
                    checkpoint_dir / f"{methode}_fraction_{nom_fraction}_seed_{seed}.pt",
                )

            del etat_modele
            if appareil.type == "cuda":
                torch.cuda.empty_cache()
            sauvegarder_rapport(
                args.output_dir,
                configuration,
                lignes,
                methode,
                args.evaluer_sur,
            )

        fractions_triees = sorted(ensembles_indices)
        for petite, grande in zip(fractions_triees, fractions_triees[1:]):
            if not ensembles_indices[petite].issubset(ensembles_indices[grande]):
                raise RuntimeError("Les sous-ensembles de labels ne sont pas emboites")

    resume = resumer(lignes, methode, args.evaluer_sur)
    print(f"\nResume accuracy {args.evaluer_sur}")
    for ligne in resume:
        print(
            f"{methode} | {float(ligne['fraction_labels']) * 100:5.1f} % | "
            f"{float(ligne['accuracy_moyenne']) * 100:5.1f} % "
            f"+/- {float(ligne['accuracy_ecart_type']) * 100:.2f} "
            f"({int(ligne['nombre_seeds'])} seeds)"
        )

    chemin_csv, chemin_json = sauvegarder_rapport(
        args.output_dir,
        configuration,
        lignes,
        methode,
        args.evaluer_sur,
    )
    print(f"\nCSV  : {chemin_csv.resolve()}")
    print(f"JSON : {chemin_json.resolve()}")


if __name__ == "__main__":
    main()
