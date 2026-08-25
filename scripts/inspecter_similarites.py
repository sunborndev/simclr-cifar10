"""Affiche les positifs, negatifs et similarites d'un petit batch SimCLR."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from simclr.augmentations import DeuxVuesTransform, creer_transform_simclr
from simclr.contrastif import (
    calculer_indices_positifs,
    calculer_matrice_similarites,
    reunir_et_normaliser,
)
from simclr.modele import ModeleSimCLR


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def nom_vue(indice: int, taille_batch: int) -> str:
    """Transforme un indice concatene en un nom lisible, par exemple A2."""
    numero_image = indice % taille_batch
    numero_vue = 1 if indice < taille_batch else 2
    lettre_image = chr(ord("A") + numero_image)
    return f"{lettre_image}{numero_vue}"


def afficher_matrice(matrice: torch.Tensor, taille_batch: int) -> None:
    noms = [nom_vue(i, taille_batch) for i in range(2 * taille_batch)]
    print("\nMatrice des similarites cosinus")
    print("      " + " ".join(f"{nom:>6}" for nom in noms))

    for nom, ligne in zip(noms, matrice.cpu()):
        valeurs = " ".join(f"{valeur.item():6.2f}" for valeur in ligne)
        print(f"{nom:>4}  {valeurs}")


def main() -> None:
    args = lire_arguments()
    if args.batch_size < 2:
        raise ValueError("Le batch doit contenir au moins deux images")
    if args.batch_size > 26:
        raise ValueError("Ce script pedagogique accepte au maximum 26 images")

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
        _, z_1 = modele(vues_1)
        _, z_2 = modele(vues_2)
        z = reunir_et_normaliser(z_1, z_2)
        matrice = calculer_matrice_similarites(z)

    indices_positifs = calculer_indices_positifs(args.batch_size, appareil)
    nombre_vues = 2 * args.batch_size

    print(f"Appareil                   : {appareil}")
    print(f"Images originales          : {args.batch_size}")
    print(f"Vues apres concatenation   : {nombre_vues}")
    print(f"Forme de z                 : {tuple(z.shape)}")
    print(f"Forme de la matrice        : {tuple(matrice.shape)}")
    print(f"Candidats par ancre        : {nombre_vues - 1}")
    print("Positifs par ancre         : 1")
    print(f"Negatifs par ancre         : {nombre_vues - 2}")

    afficher_matrice(matrice, args.batch_size)

    print("\nCorrespondances positives")
    for indice_ancre, indice_positif in enumerate(indices_positifs.tolist()):
        ancre = nom_vue(indice_ancre, args.batch_size)
        positif = nom_vue(indice_positif, args.batch_size)
        similarite = matrice[indice_ancre, indice_positif].item()
        print(f"{ancre} -> {positif} | similarite = {similarite:.3f}")

    print("\nLa diagonale vaut 1 car chaque vecteur est compare a lui-meme.")
    print("Elle sera exclue de NT-Xent.")
    print(f"Etiquettes non utilisees par SimCLR : {etiquettes.tolist()}")
    print("Les similarites ne sont pas encore apprises : les poids sont aleatoires.")


if __name__ == "__main__":
    main()

