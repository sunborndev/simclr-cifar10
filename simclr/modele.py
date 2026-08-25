"""Modele SimCLR compose d'un encodeur et d'un projection head."""

import torch
from torch import nn

from simclr.encodeur import DIMENSION_H, EncodeurResNet18


DIMENSION_Z = 128


class ProjectionHead(nn.Module):
    """Projette la representation generale h dans l'espace contrastif z."""

    def __init__(
        self,
        dimension_entree: int = DIMENSION_H,
        dimension_cachee: int = DIMENSION_H,
        dimension_sortie: int = DIMENSION_Z,
    ) -> None:
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(dimension_entree, dimension_cachee),
            nn.ReLU(),
            nn.Linear(dimension_cachee, dimension_sortie),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.couches(h)


class ModeleSimCLR(nn.Module):
    """Retourne h pour les usages futurs et z pour la loss contrastive."""

    def __init__(self) -> None:
        super().__init__()
        self.encodeur = EncodeurResNet18()
        self.projection_head = ProjectionHead()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encodeur(images)
        z = self.projection_head(h)
        return h, z

