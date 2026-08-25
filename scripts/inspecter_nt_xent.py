"""Explique la temperature et calcule NT-Xent sur un batch CIFAR-10."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from simclr.augmentations import DeuxVuesTransform, creer_transform_simclr
from simclr.contrastif import NTXentLoss, construire_logits_et_cibles
from simclr.modele import ModeleSimCLR


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = lire_arguments()
    if args.batch_size < 2:
        raise ValueError("Le batch doit contenir au moins deux images")

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

    logits, cibles = construire_logits_et_cibles(
        z_1,
        z_2,
        args.temperature,
    )
    probabilites = logits.softmax(dim=1)
    indices_ancres = torch.arange(2 * args.batch_size, device=appareil)
    probabilites_positives = probabilites[indices_ancres, cibles]
    rangs_positifs = (
        (logits > logits[indices_ancres, cibles].unsqueeze(1)).sum(dim=1) + 1
    )

    fonction_loss = NTXentLoss(temperature=args.temperature)
    loss = fonction_loss(z_1, z_2)
    loss_reference = torch.log(
        torch.tensor(2 * args.batch_size - 1, dtype=torch.float32)
    )

    print(f"Appareil                         : {appareil}")
    print(f"Temperature                      : {args.temperature}")
    print(f"Forme de z_1 et z_2              : {tuple(z_1.shape)}")
    print(f"Forme des logits                 : {tuple(logits.shape)}")
    print(f"Candidats par ancre              : {logits.size(1)}")
    print(f"Positions des positifs           : {cibles.tolist()}")
    print(
        "Probabilites des positifs      : "
        f"{[round(valeur, 4) for valeur in probabilites_positives.tolist()]}"
    )
    print(f"Rangs des positifs               : {rangs_positifs.tolist()}")
    print(f"NT-Xent moyenne                  : {loss.item():.4f}")
    print(f"Reference au hasard log(2N - 1)  : {loss_reference.item():.4f}")
    print(f"Etiquettes ignorees              : {etiquettes.tolist()}")
    print("\nInterpretation : rang 1 signifie que le positif est le candidat")
    print("le plus similaire. Avant entrainement, cela n'est pas garanti.")


if __name__ == "__main__":
    main()

