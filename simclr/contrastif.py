"""Outils pour comparer les vecteurs contrastifs d'un batch SimCLR."""

import torch
import torch.nn.functional as F
from torch import nn


def reunir_et_normaliser(z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
    """Reunit les deux groupes de vues et normalise chaque vecteur z."""
    if z_1.shape != z_2.shape:
        raise ValueError("z_1 et z_2 doivent avoir la meme forme")

    z = torch.cat([z_1, z_2], dim=0)
    return F.normalize(z, dim=1)


def calculer_matrice_similarites(z: torch.Tensor) -> torch.Tensor:
    """Calcule toutes les similarites cosinus entre les vecteurs normalises."""
    return z @ z.T


def calculer_indices_positifs(taille_batch: int, appareil: torch.device) -> torch.Tensor:
    """Retourne, pour chaque ancre, l'indice de l'autre vue de son image."""
    indices = torch.arange(2 * taille_batch, device=appareil)
    return (indices + taille_batch) % (2 * taille_batch)


def construire_logits_et_cibles(
    z_1: torch.Tensor,
    z_2: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construit les candidats NT-Xent et la position du positif par ancre."""
    if temperature <= 0:
        raise ValueError("La temperature doit etre strictement positive")

    z = reunir_et_normaliser(z_1, z_2)
    nombre_vues = z.size(0)
    taille_batch = nombre_vues // 2
    similarites = calculer_matrice_similarites(z) / temperature

    # Chaque ligne perd sa diagonale : une ancre ne doit pas se choisir
    # elle-meme. Il reste donc 2N - 1 candidats par ancre.
    masque_diagonale = torch.eye(
        nombre_vues,
        dtype=torch.bool,
        device=z.device,
    )
    logits = similarites[~masque_diagonale].view(nombre_vues, nombre_vues - 1)

    indices_ancres = torch.arange(nombre_vues, device=z.device)
    indices_positifs = calculer_indices_positifs(taille_batch, z.device)

    # Apres suppression de la diagonale, les colonnes situees a droite de
    # l'ancre sont decalees d'une position vers la gauche.
    cibles = indices_positifs - (indices_positifs > indices_ancres).long()
    return logits, cibles


class NTXentLoss(nn.Module):
    """Loss contrastive normalisee et ajustee par la temperature."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("La temperature doit etre strictement positive")
        self.temperature = temperature

    def forward(self, z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
        logits, cibles = construire_logits_et_cibles(
            z_1,
            z_2,
            self.temperature,
        )
        return F.cross_entropy(logits, cibles)
