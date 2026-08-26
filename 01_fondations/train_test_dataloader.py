"""Apprendre avec des batches et evaluer sur des exemples jamais entraines."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


torch.manual_seed(42)

NOMS_CLASSES = ["chat", "chien", "voiture"]
TAILLE_BATCH = 8
NOMBRE_EPOCHS = 100


def creer_dataset() -> TensorDataset:
    """Cree 20 variantes de chaque classe autour d'un profil simple."""
    profils = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],  # chat
            [1.0, 1.0, 1.0, 0.0],  # chien
            [0.0, 0.0, 0.0, 1.0],  # voiture
        ],
        dtype=torch.float32,
    )

    exemples = []
    etiquettes = []

    for numero_classe, profil in enumerate(profils):
        bruit = torch.randn(20, 4) * 0.08
        variantes = (profil + bruit).clamp(0.0, 1.0)
        exemples.append(variantes)
        etiquettes.append(torch.full((20,), numero_classe, dtype=torch.long))

    caracteristiques = torch.cat(exemples)
    classes = torch.cat(etiquettes)
    return TensorDataset(caracteristiques, classes)


class PetitClassificateur(nn.Module):
    """Transforme quatre caracteristiques en trois scores de classe."""

    def __init__(self) -> None:
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.couches(x)


def entrainer_une_epoch(
    modele: nn.Module,
    loader: DataLoader,
    fonction_loss: nn.Module,
    optimiseur: torch.optim.Optimizer,
) -> tuple[float, float]:
    modele.train()
    somme_loss = 0.0
    nombre_correct = 0
    nombre_exemples = 0

    for caracteristiques, etiquettes in loader:
        scores = modele(caracteristiques)
        loss = fonction_loss(scores, etiquettes)

        optimiseur.zero_grad()
        loss.backward()
        optimiseur.step()

        taille_batch = etiquettes.size(0)
        somme_loss += loss.item() * taille_batch
        nombre_correct += (scores.argmax(dim=1) == etiquettes).sum().item()
        nombre_exemples += taille_batch

    return somme_loss / nombre_exemples, nombre_correct / nombre_exemples


def evaluer(
    modele: nn.Module,
    loader: DataLoader,
    fonction_loss: nn.Module,
) -> tuple[float, float]:
    modele.eval()
    somme_loss = 0.0
    nombre_correct = 0
    nombre_exemples = 0

    with torch.no_grad():
        for caracteristiques, etiquettes in loader:
            scores = modele(caracteristiques)
            loss = fonction_loss(scores, etiquettes)

            taille_batch = etiquettes.size(0)
            somme_loss += loss.item() * taille_batch
            nombre_correct += (scores.argmax(dim=1) == etiquettes).sum().item()
            nombre_exemples += taille_batch

    return somme_loss / nombre_exemples, nombre_correct / nombre_exemples


dataset = creer_dataset()

# 48 exemples pour apprendre, 12 exemples gardes pour le test.
generateur_split = torch.Generator().manual_seed(42)
train_set, test_set = random_split(dataset, [48, 12], generator=generateur_split)

train_loader = DataLoader(
    train_set,
    batch_size=TAILLE_BATCH,
    shuffle=True,
)
test_loader = DataLoader(
    test_set,
    batch_size=TAILLE_BATCH,
    shuffle=False,
)

modele = PetitClassificateur()
fonction_loss = nn.CrossEntropyLoss()
optimiseur = torch.optim.SGD(modele.parameters(), lr=0.1)

print(f"Dataset complet : {len(dataset)} exemples")
print(f"Train           : {len(train_set)} exemples")
print(f"Test            : {len(test_set)} exemples")
print(f"Taille batch    : {TAILLE_BATCH} exemples")
print(f"Batches/epoch   : {len(train_loader)}")

for epoch in range(1, NOMBRE_EPOCHS + 1):
    train_loss, train_accuracy = entrainer_une_epoch(
        modele,
        train_loader,
        fonction_loss,
        optimiseur,
    )
    test_loss, test_accuracy = evaluer(modele, test_loader, fonction_loss)

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} | "
            f"train accuracy {train_accuracy * 100:5.1f} % | "
            f"test loss {test_loss:.4f} | "
            f"test accuracy {test_accuracy * 100:5.1f} %"
        )

# Une evaluation ne doit pas modifier les poids.
poids_avant_evaluation = [parametre.detach().clone() for parametre in modele.parameters()]
test_loss, test_accuracy = evaluer(modele, test_loader, fonction_loss)
poids_apres_evaluation = list(modele.parameters())

poids_inchanges = all(
    torch.equal(avant, apres)
    for avant, apres in zip(poids_avant_evaluation, poids_apres_evaluation)
)

print("\nBilan final")
print(f"Loss test      : {test_loss:.4f}")
print(f"Accuracy test  : {test_accuracy * 100:.1f} %")
print(f"Poids inchanges pendant l'evaluation : {poids_inchanges}")

