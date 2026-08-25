"""Outils pour mesurer la qualite des representations h."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from simclr.encodeur import EncodeurResNet18
from simclr.modele import ModeleSimCLR


# Le decoupage train / validation doit etre IDENTIQUE pour toutes les
# experiences et pour toutes les seeds. On lui donne donc une seed dediee,
# volontairement differente des seeds d'experience (42, 123, 2026), et on ne la
# change jamais. Si chaque seed d'experience voyait une validation differente,
# l'ecart-type melangerait deux sources de variation et ne voudrait plus rien
# dire.
SEED_SPLIT_VALIDATION = 7


def creer_indices_stratifies(
    cibles: torch.Tensor | list[int],
    nombre_exemples: int,
    seed: int,
) -> list[int]:
    """Selectionne un nombre presque egal d'indices pour chaque classe."""
    cibles_tenseur = torch.as_tensor(cibles)
    if not 1 <= nombre_exemples <= len(cibles_tenseur):
        raise ValueError("nombre_exemples doit etre compris dans le nombre de cibles")

    classes = torch.unique(cibles_tenseur).sort().values
    generateur = torch.Generator().manual_seed(seed)
    base, reste = divmod(nombre_exemples, len(classes))
    indices_selectionnes = []

    for position, classe in enumerate(classes):
        indices_classe = torch.where(cibles_tenseur == classe)[0]
        permutation = torch.randperm(len(indices_classe), generator=generateur)
        quota = base + int(position < reste)
        if quota > len(indices_classe):
            raise ValueError("Une classe ne contient pas assez d'exemples")
        indices_selectionnes.extend(indices_classe[permutation[:quota]].tolist())

    permutation_finale = torch.randperm(
        len(indices_selectionnes),
        generator=generateur,
    )
    return torch.tensor(indices_selectionnes)[permutation_finale].tolist()


def creer_subset_stratifie(
    dataset: Dataset,
    nombre_exemples: int,
    seed: int,
) -> Subset:
    """Selectionne un nombre presque egal d'exemples pour chaque classe."""
    if not hasattr(dataset, "targets"):
        raise ValueError("Le dataset doit exposer un attribut targets")
    if not 1 <= nombre_exemples <= len(dataset):
        raise ValueError("nombre_exemples doit etre compris dans la taille du dataset")

    indices = creer_indices_stratifies(dataset.targets, nombre_exemples, seed)
    return Subset(dataset, indices)


def separer_train_validation(
    cibles: torch.Tensor | list[int],
    nombre_validation: int,
    seed_split: int = SEED_SPLIT_VALIDATION,
) -> tuple[list[int], list[int]]:
    """Coupe le train officiel en une part d'entrainement et une validation.

    La validation sert uniquement a choisir des hyperparametres. Le jeu de test
    officiel ne doit etre consulte qu'une seule fois, a la toute fin, pour les
    chiffres publies.

    Retourne (indices_train, indices_validation), tries, sans recouvrement.
    Un nombre_validation nul desactive le decoupage : la validation est vide et
    l'entrainement recupere tout, ce qui reproduit l'ancien protocole.
    """
    cibles_tenseur = torch.as_tensor(cibles)
    if nombre_validation == 0:
        return list(range(len(cibles_tenseur))), []
    if not 0 < nombre_validation < len(cibles_tenseur):
        raise ValueError(
            "nombre_validation doit etre nul ou compris entre 1 et n - 1"
        )

    indices_validation = creer_indices_stratifies(
        cibles_tenseur,
        nombre_validation,
        seed_split,
    )
    ensemble_validation = set(indices_validation)
    indices_train = [
        indice
        for indice in range(len(cibles_tenseur))
        if indice not in ensemble_validation
    ]

    # Garde-fou : les deux parts doivent etre disjointes et couvrir tout le train.
    if len(indices_train) + len(indices_validation) != len(cibles_tenseur):
        raise RuntimeError("Le decoupage train / validation a perdu des exemples")
    if ensemble_validation & set(indices_train):
        raise RuntimeError("Recouvrement entre train et validation")

    return indices_train, sorted(indices_validation)


class ClassificateurLineaire(nn.Module):
    """Associe directement une representation h aux dix classes CIFAR-10."""

    def __init__(self, dimension_h: int = 512, nombre_classes: int = 10) -> None:
        super().__init__()
        self.couche = nn.Linear(dimension_h, nombre_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.couche(h)


def standardiser_caracteristiques(
    h_train: torch.Tensor,
    h_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardise h avec des statistiques calculees uniquement sur le train."""
    moyenne = h_train.mean(dim=0, keepdim=True)
    ecart_type = h_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (h_train - moyenne) / ecart_type, (h_test - moyenne) / ecart_type


@torch.no_grad()
def extraire_caracteristiques(
    encodeur: nn.Module,
    loader: DataLoader,
    appareil: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcule h sans gradient et conserve les caracteristiques sur le CPU."""
    encodeur.eval()
    caracteristiques = []
    etiquettes = []

    for images, classes in loader:
        images = images.to(appareil, non_blocking=True)
        h = encodeur(images)
        caracteristiques.append(h.cpu())
        etiquettes.append(classes)

    return torch.cat(caracteristiques), torch.cat(etiquettes)


@torch.no_grad()
def calculer_accuracy(
    classificateur: nn.Module,
    loader: DataLoader,
    appareil: torch.device,
) -> float:
    classificateur.eval()
    nombre_correct = 0
    nombre_total = 0

    for h, etiquettes in loader:
        h = h.to(appareil, non_blocking=True)
        etiquettes = etiquettes.to(appareil, non_blocking=True)
        predictions = classificateur(h).argmax(dim=1)
        nombre_correct += (predictions == etiquettes).sum().item()
        nombre_total += etiquettes.size(0)

    return nombre_correct / nombre_total


def entrainer_linear_probe(
    h_train: torch.Tensor,
    y_train: torch.Tensor,
    h_test: torch.Tensor,
    y_test: torch.Tensor,
    appareil: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    scheduler_cosinus: bool = False,
    initialisation_zero: bool = False,
) -> dict[str, float]:
    """Entraine uniquement Linear(512, 10), jamais l'encodeur."""
    torch.manual_seed(seed)
    classificateur = ClassificateurLineaire().to(appareil)
    if initialisation_zero:
        nn.init.zeros_(classificateur.couche.weight)
        nn.init.zeros_(classificateur.couche.bias)
    optimiseur = SGD(
        classificateur.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )
    scheduler = (
        CosineAnnealingLR(
            optimiseur,
            T_max=epochs,
            eta_min=learning_rate * 0.001,
        )
        if scheduler_cosinus
        else None
    )
    fonction_loss = nn.CrossEntropyLoss()
    generateur_loader = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(h_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=appareil.type == "cuda",
        generator=generateur_loader,
    )
    test_loader = DataLoader(
        TensorDataset(h_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=appareil.type == "cuda",
    )

    derniere_loss = 0.0
    for _epoch in range(epochs):
        classificateur.train()
        somme_loss = 0.0
        nombre_total = 0

        for h, etiquettes in train_loader:
            h = h.to(appareil, non_blocking=True)
            etiquettes = etiquettes.to(appareil, non_blocking=True)

            optimiseur.zero_grad()
            scores = classificateur(h)
            loss = fonction_loss(scores, etiquettes)
            loss.backward()
            optimiseur.step()

            somme_loss += loss.item() * etiquettes.size(0)
            nombre_total += etiquettes.size(0)

        derniere_loss = somme_loss / nombre_total
        if scheduler is not None:
            scheduler.step()

    return {
        "loss_train": derniere_loss,
        "accuracy_train": calculer_accuracy(classificateur, train_loader, appareil),
        "accuracy_test": calculer_accuracy(classificateur, test_loader, appareil),
    }


def charger_encodeur_simclr(checkpoint_path: Path) -> EncodeurResNet18:
    """Charge le modele local puis ne conserve que son encodeur."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modele = ModeleSimCLR()
    modele.load_state_dict(checkpoint["modele"])
    return modele.encodeur
