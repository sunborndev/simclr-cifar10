"""Telecharge CIFAR-10 et visualise deux vues SimCLR par image."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_pil_image

from simclr.augmentations import DeuxVuesTransform, creer_transform_simclr


NOMS_CLASSES_FR = [
    "avion",
    "automobile",
    "oiseau",
    "chat",
    "cerf",
    "chien",
    "grenouille",
    "cheval",
    "bateau",
    "camion",
]


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vues_simclr_cifar10.png"),
    )
    parser.add_argument("--nombre-images", type=int, default=6, choices=range(1, 11))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def trouver_une_image_par_classe(dataset: CIFAR10, nombre: int) -> list[int]:
    indices = []
    classes_trouvees = set()

    for indice, classe in enumerate(dataset.targets):
        if classe not in classes_trouvees:
            indices.append(indice)
            classes_trouvees.add(classe)
        if len(indices) == nombre:
            break

    return indices


def charger_police(taille: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", taille)
    except OSError:
        return ImageFont.load_default()


def creer_planche(
    dataset: CIFAR10,
    indices: list[int],
    deux_vues: DeuxVuesTransform,
) -> Image.Image:
    taille_tuile = 192
    hauteur_titre = 32
    marge = 12
    largeur = 3 * taille_tuile + 4 * marge
    hauteur_ligne = hauteur_titre + taille_tuile + marge
    hauteur = len(indices) * hauteur_ligne + marge

    planche = Image.new("RGB", (largeur, hauteur), color="white")
    dessin = ImageDraw.Draw(planche)
    police = charger_police(18)

    for numero_ligne, indice in enumerate(indices):
        image_originale, classe = dataset[indice]
        vue_1, vue_2 = deux_vues(image_originale)
        images = [image_originale, to_pil_image(vue_1), to_pil_image(vue_2)]
        titres = [f"originale : {NOMS_CLASSES_FR[classe]}", "vue 1", "vue 2"]

        y_titre = marge + numero_ligne * hauteur_ligne
        y_image = y_titre + hauteur_titre

        for colonne, (image, titre) in enumerate(zip(images, titres)):
            x = marge + colonne * (taille_tuile + marge)
            dessin.text((x, y_titre), titre, fill="black", font=police)
            image_agrandie = image.resize(
                (taille_tuile, taille_tuile),
                resample=Image.Resampling.NEAREST,
            )
            planche.paste(image_agrandie, (x, y_image))

    return planche


def main() -> None:
    args = lire_arguments()
    torch.manual_seed(args.seed)

    dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
    )
    transformation = creer_transform_simclr(normaliser=False)
    deux_vues = DeuxVuesTransform(transformation)
    indices = trouver_une_image_par_classe(dataset, args.nombre_images)

    planche = creer_planche(dataset, indices, deux_vues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    planche.save(args.output)

    print(f"Dataset CIFAR-10 : {len(dataset)} images d'entrainement")
    print(f"Planche enregistree : {args.output.resolve()}")
    print(f"Seed des augmentations : {args.seed}")


if __name__ == "__main__":
    main()

