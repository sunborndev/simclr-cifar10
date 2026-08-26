"""Premier entrainement supervise avec PyTorch.

Les quatre nombres d'entree remplacent temporairement les pixels d'une image :
[pelage, quatre_pattes, aboie, roues].
"""

import torch
from torch import nn


torch.manual_seed(42)

NOMS_CLASSES = ["chat", "chien", "voiture"]

# Chaque ligne est un exemple. Les valeurs varient legerement pour eviter
# d'apprendre une seule ligne exacte par classe.
caracteristiques = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0],
        [1.0, 0.9, 0.1, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.9, 1.0, 0.9, 0.0],
        [1.0, 0.9, 0.8, 0.0],
        [0.8, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.1, 0.0, 0.9],
        [0.1, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.1, 0.8],
    ],
    dtype=torch.float32,
)

# 0 = chat, 1 = chien, 2 = voiture.
etiquettes = torch.tensor(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    dtype=torch.long,
)


class PetitClassificateur(nn.Module):
    """Reseau qui transforme 4 caracteristiques en 3 scores de classe."""

    def __init__(self) -> None:
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.couches(x)


modele = PetitClassificateur()
fonction_loss = nn.CrossEntropyLoss()
optimiseur = torch.optim.SGD(modele.parameters(), lr=0.1)

poids_avant = modele.couches[0].weight.detach().clone()

print("Prediction du premier exemple AVANT l'entrainement :")
with torch.no_grad():
    scores_avant = modele(caracteristiques[0:1])
    probabilites_avant = torch.softmax(scores_avant, dim=1)
    print(dict(zip(NOMS_CLASSES, probabilites_avant[0].tolist())))

for epoch in range(1, 301):
    # 1. Forward pass : le reseau produit trois scores par exemple.
    scores = modele(caracteristiques)

    # 2. La loss compare les scores avec les bonnes etiquettes.
    loss = fonction_loss(scores, etiquettes)

    # 3. On efface les gradients calcules au tour precedent.
    optimiseur.zero_grad()

    # 4. La backpropagation calcule le gradient de chaque poids.
    loss.backward()

    # 5. L'optimiseur modifie les poids dans le sens qui reduit la loss.
    optimiseur.step()

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f}")

with torch.no_grad():
    scores_finaux = modele(caracteristiques)
    predictions = scores_finaux.argmax(dim=1)
    precision = (predictions == etiquettes).float().mean()

    probabilites_apres = torch.softmax(modele(caracteristiques[0:1]), dim=1)
    print("\nPrediction du premier exemple APRES l'entrainement :")
    print(dict(zip(NOMS_CLASSES, probabilites_apres[0].tolist())))
    print(f"Precision sur les 12 exemples : {precision.item() * 100:.1f} %")

poids_apres = modele.couches[0].weight.detach()
changement_moyen = (poids_apres - poids_avant).abs().mean()
print(f"Modification moyenne des poids : {changement_moyen.item():.4f}")

