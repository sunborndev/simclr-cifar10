"""Pre-entrainement SimCLR reproductible sur l'ensemble de CIFAR-10."""

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from simclr.augmentations import (
    CONFIGURATIONS_ABLATION,
    DeuxVuesTransform,
    creer_transform_ablation,
    decrire_configuration,
)
from simclr.contrastif import NTXentLoss, construire_logits_et_cibles
from simclr.modele import ModeleSimCLR


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/simclr_cifar10_100ep"),
    )
    parser.add_argument(
        "--augmentations",
        choices=sorted(CONFIGURATIONS_ABLATION),
        default="complet",
        help=(
            "Configuration d'augmentations. 'complet' reproduit le "
            "preentrainement principal ; les autres servent a l'ablation."
        ),
    )
    parser.add_argument("--nombre-images", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin d'un checkpoint ou 'auto' pour checkpoint_latest.pt",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def valider_arguments(args: argparse.Namespace) -> None:
    if args.nombre_images < args.batch_size:
        raise ValueError("nombre-images doit etre superieur ou egal au batch-size")
    if args.epochs < 1:
        raise ValueError("epochs doit etre au moins egal a 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs doit etre compris entre 0 et epochs - 1")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Les learning rates doivent etre positifs")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate ne peut pas depasser learning-rate")
    if args.save_every < 0 or args.print_every < 0:
        raise ValueError("save-every et print-every ne peuvent pas etre negatifs")


def creer_dataset(args: argparse.Namespace) -> Dataset:
    dataset_complet = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=DeuxVuesTransform(creer_transform_ablation(args.augmentations)),
    )
    nombre_images = min(args.nombre_images, len(dataset_complet))
    if nombre_images == len(dataset_complet):
        return dataset_complet

    generateur = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset_complet), generator=generateur)[:nombre_images]
    return Subset(dataset_complet, indices.tolist())


def creer_scheduler(
    optimiseur: AdamW,
    args: argparse.Namespace,
) -> CosineAnnealingLR | SequentialLR:
    epochs_cosinus = args.epochs - args.warmup_epochs
    cosinus = CosineAnnealingLR(
        optimiseur,
        T_max=epochs_cosinus,
        eta_min=args.min_learning_rate,
    )
    if args.warmup_epochs == 0:
        return cosinus

    warmup = LinearLR(
        optimiseur,
        start_factor=1.0 / args.warmup_epochs,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    return SequentialLR(
        optimiseur,
        schedulers=[warmup, cosinus],
        milestones=[args.warmup_epochs],
    )


def calculer_top1_positif(
    z_1: torch.Tensor,
    z_2: torch.Tensor,
    temperature: float,
) -> int:
    logits, cibles = construire_logits_et_cibles(z_1, z_2, temperature)
    return (logits.argmax(dim=1) == cibles).sum().item()


def resoudre_checkpoint_reprise(args: argparse.Namespace) -> Path | None:
    if args.resume is None:
        return None
    if args.resume.lower() == "auto":
        return args.run_dir / "checkpoint_latest.pt"
    return Path(args.resume)


def etat_rng(generateur_loader: torch.Generator) -> dict[str, Any]:
    etat: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "loader": generateur_loader.get_state(),
    }
    if torch.cuda.is_available():
        etat["cuda"] = torch.cuda.get_rng_state_all()
    return etat


def restaurer_rng(etat: dict[str, Any], generateur_loader: torch.Generator) -> None:
    random.setstate(etat["python"])
    # map_location peut deplacer ces ByteTensor sur le GPU lors du chargement,
    # alors que les generateurs CPU exigent explicitement un etat sur le CPU.
    torch.set_rng_state(etat["torch"].cpu())
    generateur_loader.set_state(etat["loader"].cpu())
    if torch.cuda.is_available() and "cuda" in etat:
        torch.cuda.set_rng_state_all([etat_gpu.cpu() for etat_gpu in etat["cuda"]])


def sauvegarder_checkpoint(
    chemin: Path,
    epoch: int,
    modele: ModeleSimCLR,
    optimiseur: AdamW,
    scheduler: CosineAnnealingLR | SequentialLR,
    scaler: torch.amp.GradScaler,
    historique: list[dict[str, float]],
    args: argparse.Namespace,
    generateur_loader: torch.Generator,
    complet: bool,
) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    contenu: dict[str, Any] = {
        "epoch": epoch,
        "modele": modele.state_dict(),
        "historique": historique,
        "configuration": vars(args),
    }
    if complet:
        contenu.update(
            {
                "optimiseur": optimiseur.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": etat_rng(generateur_loader),
            }
        )

    chemin_temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    torch.save(contenu, chemin_temporaire)
    chemin_temporaire.replace(chemin)


def ecrire_historique_csv(chemin: Path, historique: list[dict[str, float]]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    champs = [
        "epoch",
        "loss",
        "top1_positif",
        "learning_rate",
        "duree_secondes",
        "vram_pic_mo",
    ]
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=champs)
        writer.writeheader()
        writer.writerows(historique)


def main() -> None:
    args = lire_arguments()
    valider_arguments(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    appareil = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    amp_active = appareil.type == "cuda" and not args.no_amp
    generateur_loader = torch.Generator().manual_seed(args.seed)
    dataset = creer_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=appareil.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generateur_loader,
    )

    modele = ModeleSimCLR().to(appareil)
    fonction_loss = NTXentLoss(temperature=args.temperature)
    optimiseur = AdamW(
        modele.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = creer_scheduler(optimiseur, args)
    scaler = torch.amp.GradScaler(device=appareil.type, enabled=amp_active)

    debut_epoch = 1
    historique: list[dict[str, float]] = []
    checkpoint_reprise = resoudre_checkpoint_reprise(args)
    if checkpoint_reprise is not None:
        if not checkpoint_reprise.exists():
            raise FileNotFoundError(f"Checkpoint de reprise introuvable : {checkpoint_reprise}")
        checkpoint = torch.load(
            checkpoint_reprise,
            map_location=appareil,
            weights_only=False,
        )
        modele.load_state_dict(checkpoint["modele"])
        optimiseur.load_state_dict(checkpoint["optimiseur"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        historique = checkpoint.get("historique", [])
        debut_epoch = checkpoint["epoch"] + 1
        restaurer_rng(checkpoint["rng"], generateur_loader)
        print(f"Reprise du checkpoint : {checkpoint_reprise.resolve()}")

    print(f"Appareil             : {appareil}")
    if appareil.type == "cuda":
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision      : {amp_active}")
    print(f"Augmentations        : {args.augmentations}")
    print(f"  -> {decrire_configuration(args.augmentations)}")
    print(f"Images utilisees     : {len(dataset)}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Batches par epoch    : {len(loader)}")
    print(f"Negatifs par ancre   : {2 * args.batch_size - 2}")
    print(f"Epochs               : {args.epochs}")
    print(f"Temperature          : {args.temperature}")
    print(f"Learning rate max    : {args.learning_rate}")
    print(f"Warmup               : {args.warmup_epochs} epochs")
    print(f"Loss hasard approx.  : {math.log(2 * args.batch_size - 1):.4f}")

    if debut_epoch > args.epochs:
        print("Le checkpoint a deja atteint le nombre d'epochs demande.")
        return

    debut_total = time.perf_counter()
    chemin_csv = args.run_dir / "historique.csv"

    try:
        for epoch in range(debut_epoch, args.epochs + 1):
            modele.train()
            somme_loss = 0.0
            nombre_ancres = 0
            nombre_top1 = 0
            nombre_batches = 0
            debut = time.perf_counter()
            learning_rate = optimiseur.param_groups[0]["lr"]

            if appareil.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            for numero_batch, ((vues_1, vues_2), _etiquettes) in enumerate(
                loader,
                start=1,
            ):
                vues = torch.cat([vues_1, vues_2], dim=0).to(
                    appareil,
                    non_blocking=True,
                )
                optimiseur.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=appareil.type,
                    dtype=torch.float16,
                    enabled=amp_active,
                ):
                    _, z = modele(vues)

                z_1, z_2 = z.float().chunk(2, dim=0)
                loss = fonction_loss(z_1, z_2)
                scaler.scale(loss).backward()
                scaler.step(optimiseur)
                scaler.update()

                ancres_batch = z.size(0)
                somme_loss += loss.item() * ancres_batch
                nombre_ancres += ancres_batch
                nombre_batches += 1

                with torch.no_grad():
                    nombre_top1 += calculer_top1_positif(
                        z_1.detach(),
                        z_2.detach(),
                        args.temperature,
                    )

                if args.print_every and numero_batch % args.print_every == 0:
                    print(
                        f"  epoch {epoch:03d} | batch {numero_batch:03d}/{len(loader)} | "
                        f"loss moyenne {somme_loss / nombre_ancres:.4f}"
                    )
                if args.max_batches is not None and numero_batch >= args.max_batches:
                    break

            scheduler.step()
            duree = time.perf_counter() - debut
            vram_pic_mo = (
                torch.cuda.max_memory_allocated() / 1024**2
                if appareil.type == "cuda"
                else 0.0
            )
            ligne = {
                "epoch": float(epoch),
                "loss": somme_loss / nombre_ancres,
                "top1_positif": nombre_top1 / nombre_ancres,
                "learning_rate": learning_rate,
                "duree_secondes": duree,
                "vram_pic_mo": vram_pic_mo,
            }
            historique.append(ligne)
            ecrire_historique_csv(chemin_csv, historique)

            sauvegarder_checkpoint(
                args.run_dir / "checkpoint_latest.pt",
                epoch,
                modele,
                optimiseur,
                scheduler,
                scaler,
                historique,
                args,
                generateur_loader,
                complet=True,
            )
            if args.save_every and epoch % args.save_every == 0:
                sauvegarder_checkpoint(
                    args.run_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                    epoch,
                    modele,
                    optimiseur,
                    scheduler,
                    scaler,
                    historique,
                    args,
                    generateur_loader,
                    complet=False,
                )

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"batches {nombre_batches:03d} | "
                f"loss {ligne['loss']:.4f} | "
                f"top-1 positif {ligne['top1_positif'] * 100:5.1f} % | "
                f"lr {learning_rate:.2e} | "
                f"VRAM {vram_pic_mo:.0f} Mo | "
                f"{duree:.1f} s"
            )
    except KeyboardInterrupt:
        print("\nEntrainement interrompu. Le dernier checkpoint d'epoch reste disponible.")
        print("Reprise : ajouter --resume auto a la meme commande.")
        return

    print(f"\nCheckpoint final : {(args.run_dir / 'checkpoint_latest.pt').resolve()}")
    print(f"Historique CSV   : {chemin_csv.resolve()}")
    print(f"Duree totale     : {time.perf_counter() - debut_total:.1f} s")
    print("Les etiquettes CIFAR-10 n'ont jamais ete transmises au modele.")


if __name__ == "__main__":
    main()
