"""Transformations utilisees pour creer les deux vues SimCLR.

Chaque composante d'augmentation est desormais pilotable separement. C'est ce
qui rend l'etude d'ablation possible : on retire UNE brique a la fois et on
mesure ce que la representation y perd.

Attention : les valeurs par defaut reproduisent EXACTEMENT le comportement
d'origine (crop + flip + jitter + niveaux de gris, sans flou). Le
preentrainement de 100 epoques deja realise correspond donc a la configuration
`complet`, et n'a pas besoin d'etre relance.
"""

from typing import Any

import torch
from torchvision import transforms


MOYENNE_CIFAR10 = (0.4914, 0.4822, 0.4465)
ECART_TYPE_CIFAR10 = (0.2470, 0.2435, 0.2616)


# Design additif : on part du recadrage seul et on ajoute une brique a la fois.
# Une ablation n'est valide que si UNE SEULE chose change entre deux lignes.
CONFIGURATIONS_ABLATION: dict[str, dict[str, bool]] = {
    "crop": {
        "utiliser_flip": False,
        "utiliser_jitter": False,
        "utiliser_grayscale": False,
        "utiliser_flou": False,
    },
    "crop_flip": {
        "utiliser_flip": True,
        "utiliser_jitter": False,
        "utiliser_grayscale": False,
        "utiliser_flou": False,
    },
    "crop_flip_jitter": {
        "utiliser_flip": True,
        "utiliser_jitter": True,
        "utiliser_grayscale": False,
        "utiliser_flou": False,
    },
    # `complet` = la configuration du preentrainement principal deja realise.
    "complet": {
        "utiliser_flip": True,
        "utiliser_jitter": True,
        "utiliser_grayscale": True,
        "utiliser_flou": False,
    },
    "complet_flou": {
        "utiliser_flip": True,
        "utiliser_jitter": True,
        "utiliser_grayscale": True,
        "utiliser_flou": True,
    },
}


def decrire_configuration(nom: str) -> str:
    """Retourne une description lisible d'une configuration d'ablation."""
    if nom not in CONFIGURATIONS_ABLATION:
        raise KeyError(f"Configuration inconnue : {nom}")
    options = CONFIGURATIONS_ABLATION[nom]
    briques = ["recadrage aleatoire"]
    if options["utiliser_flip"]:
        briques.append("retournement horizontal")
    if options["utiliser_jitter"]:
        briques.append("perturbation des couleurs")
    if options["utiliser_grayscale"]:
        briques.append("niveaux de gris")
    if options["utiliser_flou"]:
        briques.append("flou gaussien")
    return " + ".join(briques)


class DeuxVuesTransform:
    """Applique deux fois une transformation aleatoire a la meme image."""

    def __init__(self, transformation: transforms.Compose) -> None:
        self.transformation = transformation

    def __call__(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        vue_1 = self.transformation(image)
        vue_2 = self.transformation(image)
        return vue_1, vue_2


def creer_transform_simclr(
    taille_image: int = 32,
    normaliser: bool = True,
    utiliser_flip: bool = True,
    utiliser_jitter: bool = True,
    utiliser_grayscale: bool = True,
    utiliser_flou: bool = False,
) -> transforms.Compose:
    """Construit la chaine d'augmentations SimCLR adaptee a CIFAR-10.

    Les valeurs par defaut reproduisent la configuration `complet`, c'est-a-dire
    celle du preentrainement principal.
    """
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )

    # Le recadrage aleatoire n'est jamais retire : sans lui, les deux vues
    # seraient identiques en geometrie et la tache contrastive perdrait son sens.
    transformations: list[Any] = [
        transforms.RandomResizedCrop(
            size=taille_image,
            scale=(0.2, 1.0),
        ),
    ]

    if utiliser_flip:
        transformations.append(transforms.RandomHorizontalFlip(p=0.5))

    if utiliser_jitter:
        transformations.append(transforms.RandomApply([color_jitter], p=0.8))

    if utiliser_grayscale:
        transformations.append(transforms.RandomGrayscale(p=0.2))

    if utiliser_flou:
        transformations.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.5,
            )
        )

    transformations.append(transforms.ToTensor())

    if normaliser:
        transformations.append(
            transforms.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10)
        )

    return transforms.Compose(transformations)


def creer_transform_ablation(
    nom: str,
    taille_image: int = 32,
    normaliser: bool = True,
) -> transforms.Compose:
    """Construit la chaine d'augmentations d'une configuration d'ablation."""
    if nom not in CONFIGURATIONS_ABLATION:
        disponibles = ", ".join(CONFIGURATIONS_ABLATION)
        raise KeyError(f"Configuration inconnue : {nom}. Disponibles : {disponibles}")
    return creer_transform_simclr(
        taille_image=taille_image,
        normaliser=normaliser,
        **CONFIGURATIONS_ABLATION[nom],
    )
