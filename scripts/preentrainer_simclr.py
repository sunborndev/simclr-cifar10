"""Premier pre-entrainement SimCLR sur un sous-ensemble de CIFAR-10."""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from simclr.augmentations import DeuxVuesTransform, creer_transform_simclr
from simclr.contrastif import NTXentLoss, construire_logits_et_cibles
from simclr.modele import ModeleSimCLR


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/simclr_demo.pt"))
    parser.add_argument("--nombre-images", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def calculer_top1_positif(z_1: torch.Tensor, z_2: torch.Tensor, temperature: float) -> int:
    """Compte les ancres dont le positif possede le plus grand logit."""
    logits, cibles = construire_logits_et_cibles(z_1, z_2, temperature)
    return (logits.argmax(dim=1) == cibles).sum().item()


def main() -> None:
    args = lire_arguments()
    if args.nombre_images < args.batch_size:
        raise ValueError("nombre-images doit etre superieur ou egal au batch-size")
    if args.epochs < 1:
        raise ValueError("epochs doit etre au moins egal a 1")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    appareil = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    dataset_complet = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=DeuxVuesTransform(creer_transform_simclr()),
    )
    nombre_images = min(args.nombre_images, len(dataset_complet))
    generateur = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset_complet), generator=generateur)[:nombre_images]
    dataset = Subset(dataset_complet, indices.tolist())

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=appareil.type == "cuda",
    )

    modele = ModeleSimCLR().to(appareil)
    fonction_loss = NTXentLoss(temperature=args.temperature)
    optimiseur = Adam(modele.parameters(), lr=args.learning_rate)

    print(f"Appareil             : {appareil}")
    if appareil.type == "cuda":
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"Images utilisees     : {len(dataset)}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Vues par batch       : {2 * args.batch_size}")
    print(f"Negatifs par ancre   : {2 * args.batch_size - 2}")
    print(f"Epochs               : {args.epochs}")
    print(f"Temperature          : {args.temperature}")
    print(f"Learning rate        : {args.learning_rate}")

    debut_total = time.perf_counter()
    historique = []

    for epoch in range(1, args.epochs + 1):
        modele.train()
        somme_loss = 0.0
        nombre_ancres = 0
        nombre_top1 = 0
        nombre_batches = 0
        debut_epoch = time.perf_counter()

        for numero_batch, ((vues_1, vues_2), _etiquettes) in enumerate(loader, start=1):
            vues_1 = vues_1.to(appareil, non_blocking=True)
            vues_2 = vues_2.to(appareil, non_blocking=True)

            optimiseur.zero_grad()
            _, z_1 = modele(vues_1)
            _, z_2 = modele(vues_2)
            loss = fonction_loss(z_1, z_2)
            loss.backward()
            optimiseur.step()

            ancres_batch = 2 * vues_1.size(0)
            somme_loss += loss.item() * ancres_batch
            nombre_ancres += ancres_batch

            with torch.no_grad():
                nombre_top1 += calculer_top1_positif(
                    z_1.detach(),
                    z_2.detach(),
                    args.temperature,
                )

            nombre_batches += 1
            if args.max_batches is not None and numero_batch >= args.max_batches:
                break

        loss_moyenne = somme_loss / nombre_ancres
        top1 = nombre_top1 / nombre_ancres
        duree_epoch = time.perf_counter() - debut_epoch
        historique.append(
            {
                "epoch": epoch,
                "loss": loss_moyenne,
                "top1_positif": top1,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"batches {nombre_batches:02d} | "
            f"loss {loss_moyenne:.4f} | "
            f"top-1 positif {top1 * 100:5.1f} % | "
            f"{duree_epoch:.1f} s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modele": modele.state_dict(),
            "optimiseur": optimiseur.state_dict(),
            "historique": historique,
            "configuration": vars(args),
        },
        args.output,
    )

    duree_totale = time.perf_counter() - debut_total
    print(f"\nCheckpoint : {args.output.resolve()}")
    print(f"Duree totale : {duree_totale:.1f} s")
    print("Les etiquettes CIFAR-10 ont ete ignorees pendant tout le pre-entrainement.")


if __name__ == "__main__":
    main()

