"""Observe les representations h et z produites sur un batch CIFAR-10."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from simclr.augmentations import DeuxVuesTransform, creer_transform_simclr
from simclr.modele import DIMENSION_Z, ModeleSimCLR


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

    modele = ModeleSimCLR().to(appareil)
    modele.eval()

    with torch.no_grad():
        h_1, z_1 = modele(vues_1)
        h_2, z_2 = modele(vues_2)

        # La normalisation place chaque z sur une sphere de rayon 1. Elle rend
        # ensuite le produit scalaire equivalent a la similarite cosinus.
        z_1_normalise = F.normalize(z_1, dim=1)
        z_2_normalise = F.normalize(z_2, dim=1)

    normes_avant = z_1.norm(dim=1)
    normes_apres = z_1_normalise.norm(dim=1)

    print(f"Appareil                         : {appareil}")
    print(f"Forme des images                 : {tuple(vues_1.shape)}")
    print(f"Forme de h                       : {tuple(h_1.shape)}")
    print(f"Forme de z                       : {tuple(z_1.shape)}")
    print(f"Dimension attendue de z          : {DIMENSION_Z}")
    print(f"Norme moyenne de z avant         : {normes_avant.mean().item():.4f}")
    print(f"Norme moyenne de z apres         : {normes_apres.mean().item():.4f}")
    print(f"8 premieres valeurs de h_1[0]   : {h_1[0, :8].tolist()}")
    print(f"8 premieres valeurs de z_1[0]   : {z_1[0, :8].tolist()}")
    print("Partage des poids                : un seul modele pour les deux vues")
    print(f"Etiquettes ignorees par SimCLR   : {etiquettes.tolist()}")


if __name__ == "__main__":
    main()

