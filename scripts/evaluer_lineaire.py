"""Compare un encodeur aleatoire et l'encodeur SimCLR avec une linear probe."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from simclr.augmentations import ECART_TYPE_CIFAR10, MOYENNE_CIFAR10
from simclr.encodeur import EncodeurResNet18
from simclr.evaluation import (
    charger_encodeur_simclr,
    creer_subset_stratifie,
    entrainer_linear_probe,
    extraire_caracteristiques,
)


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/simclr_demo.pt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation_lineaire_demo.json"),
    )
    parser.add_argument("--nombre-train", type=int, default=2_000)
    parser.add_argument("--nombre-test", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def evaluer_encodeur(
    nom: str,
    encodeur: EncodeurResNet18,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
    appareil: torch.device,
) -> dict[str, float]:
    encodeur = encodeur.to(appareil)
    for parametre in encodeur.parameters():
        parametre.requires_grad = False

    debut = time.perf_counter()
    h_train, y_train = extraire_caracteristiques(encodeur, train_loader, appareil)
    h_test, y_test = extraire_caracteristiques(encodeur, test_loader, appareil)
    resultats = entrainer_linear_probe(
        h_train,
        y_train,
        h_test,
        y_test,
        appareil,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    resultats["duree_secondes"] = time.perf_counter() - debut

    print(
        f"{nom:<18} | "
        f"train {resultats['accuracy_train'] * 100:5.1f} % | "
        f"test {resultats['accuracy_test'] * 100:5.1f} % | "
        f"{resultats['duree_secondes']:.1f} s"
    )
    return resultats


def main() -> None:
    args = lire_arguments()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {args.checkpoint}")

    torch.manual_seed(args.seed)
    appareil = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10),
        ]
    )
    dataset_train_complet = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transformation,
    )
    dataset_test_complet = CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transformation,
    )
    train_set = creer_subset_stratifie(
        dataset_train_complet,
        min(args.nombre_train, len(dataset_train_complet)),
        args.seed,
    )
    test_set = creer_subset_stratifie(
        dataset_test_complet,
        min(args.nombre_test, len(dataset_test_complet)),
        args.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=appareil.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=appareil.type == "cuda",
    )

    print(f"Appareil                 : {appareil}")
    print(f"Images etiquetees train  : {len(train_set)}")
    print(f"Images officielles test  : {len(test_set)}")
    print(f"Epochs du linear probe   : {args.epochs}")
    print("Poids entraines          : 5 130 (Linear 512 -> 10)")
    print("Poids ResNet entraines   : 0 (encodeur gele)\n")

    torch.manual_seed(args.seed)
    resultats_aleatoire = evaluer_encodeur(
        "ResNet aleatoire",
        EncodeurResNet18(),
        train_loader,
        test_loader,
        args,
        appareil,
    )
    resultats_simclr = evaluer_encodeur(
        "ResNet SimCLR",
        charger_encodeur_simclr(args.checkpoint),
        train_loader,
        test_loader,
        args,
        appareil,
    )

    gain = resultats_simclr["accuracy_test"] - resultats_aleatoire["accuracy_test"]
    print(f"\nGain SimCLR sur le test : {gain * 100:+.2f} points")

    rapport = {
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "nombre_train": len(train_set),
            "nombre_test": len(test_set),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "resnet_aleatoire": resultats_aleatoire,
        "resnet_simclr": resultats_simclr,
        "gain_test_points": gain * 100,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rapport, indent=2), encoding="utf-8")
    print(f"Rapport : {args.output.resolve()}")


if __name__ == "__main__":
    main()
