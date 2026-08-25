"""Fait passer deux vues CIFAR-10 dans le meme encodeur ResNet18."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from simclr.augmentations import DeuxVuesTransform, creer_transform_simclr
from simclr.encodeur import DIMENSION_H, EncodeurResNet18


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = lire_arguments()
    torch.manual_seed(args.seed)

    appareil = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=DeuxVuesTransform(creer_transform_simclr()),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    (vues_1, vues_2), etiquettes = next(iter(loader))
    vues_1 = vues_1.to(appareil)
    vues_2 = vues_2.to(appareil)

    encodeur = EncodeurResNet18().to(appareil)
    encodeur.eval()

    with torch.no_grad():
        h_1 = encodeur(vues_1)
        h_2 = encodeur(vues_2)

    nombre_parametres = sum(parametre.numel() for parametre in encodeur.parameters())

    print(f"Appareil                   : {appareil}")
    if appareil.type == "cuda":
        print(f"GPU                        : {torch.cuda.get_device_name(0)}")
    print(f"Nombre d'images du batch   : {args.batch_size}")
    print(f"Forme des vues 1           : {tuple(vues_1.shape)}")
    print(f"Forme des vues 2           : {tuple(vues_2.shape)}")
    print(f"Forme des etiquettes       : {tuple(etiquettes.shape)}")
    print(f"Forme de h_1               : {tuple(h_1.shape)}")
    print(f"Forme de h_2               : {tuple(h_2.shape)}")
    print(f"Dimension attendue de h    : {DIMENSION_H}")
    print(f"Parametres de l'encodeur   : {nombre_parametres:,}")
    print(f"8 premieres valeurs de h_1 : {h_1[0, :8].tolist()}")
    print("Partage des poids          : un seul encodeur appele pour les deux vues")
    print(f"Etiquettes chargees mais non donnees a ResNet : {etiquettes.tolist()}")


if __name__ == "__main__":
    main()
