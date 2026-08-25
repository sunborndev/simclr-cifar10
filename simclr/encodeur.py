"""Encodeur ResNet18 adapte aux images CIFAR-10 de 32 x 32 pixels."""

import torch
from torch import nn
from torchvision.models import resnet18


DIMENSION_H = 512


class EncodeurResNet18(nn.Module):
    """Transforme une image ou un batch d'images en representations h."""

    def __init__(self) -> None:
        super().__init__()
        reseau = resnet18(weights=None)

        # ResNet18 est concu pour des images ImageNet de 224 x 224. CIFAR-10
        # utilise 32 x 32 : une convolution 3 x 3 preserve mieux les details.
        reseau.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        reseau.maxpool = nn.Identity()

        # La couche fc originale produit 1000 scores ImageNet. SimCLR veut la
        # representation h de 512 nombres qui la precede.
        reseau.fc = nn.Identity()
        self.reseau = reseau

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.reseau(images)

