"""Reproduit l'integralite du tableau d'ablation des augmentations.

Le sujet exige « un script unique pour reproduire chaque resultat du tableau
d'ablation ». C'est celui-ci.

Il enchaine, pour chaque configuration d'augmentations :

1. le preentrainement SimCLR sans labels, 100 epoques, tous les autres
   hyperparametres identiques (c'est la regle d'or de l'ablation : une seule
   chose change a la fois) ;
2. l'evaluation lineaire de la representation obtenue, sur 3 graines.

La configuration `complet` correspond au preentrainement principal deja
realise : le script la reutilise au lieu de la relancer, ce qui economise
environ 50 minutes de calcul. Utiliser --forcer-complet pour la recalculer.

RAPPEL METHODOLOGIQUE IMPORTANT
On ne classe PAS les configurations par la perte NT-Xent. Retirer une
augmentation rend la tache contrastive plus facile, donc la perte descend et le
taux de positifs au premier rang monte, alors meme que la representation
devient moins utile. Le seul critere de classement valide est l'accuracy de
l'evaluation lineaire, derniere colonne du tableau produit.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from simclr.augmentations import CONFIGURATIONS_ABLATION, decrire_configuration


# Ordre additif : on part du recadrage seul et on ajoute une brique a la fois.
ORDRE_CONFIGURATIONS = [
    "crop",
    "crop_flip",
    "crop_flip_jitter",
    "complet",
    "complet_flou",
]

# Le preentrainement principal EST la configuration `complet`.
RUN_PRINCIPAL = Path("outputs/simclr_cifar10_100ep")


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--racine", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Graine du preentrainement")
    parser.add_argument(
        "--seeds-probe",
        type=int,
        nargs="+",
        default=[42, 123, 2026],
        help="Graines de l'evaluation lineaire",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        default=ORDRE_CONFIGURATIONS,
        choices=sorted(CONFIGURATIONS_ABLATION),
    )
    parser.add_argument(
        "--forcer-complet",
        action="store_true",
        help="Relancer `complet` au lieu de reutiliser le run principal",
    )
    parser.add_argument(
        "--sauter-preentrainement",
        action="store_true",
        help="N'executer que les evaluations et le tableau final",
    )
    parser.add_argument(
        "--fraction-probe",
        type=float,
        default=1.0,
        help=(
            "Fraction de labels utilisee par l'evaluation lineaire. "
            "1.0 = labels complets, mesure la moins bruitee. "
            "0.10 = protocole demande en seance 8, diapositive 12, point 4."
        ),
    )
    parser.add_argument(
        "--suffixe-sortie",
        default=None,
        help=(
            "Nom du sous-dossier de probe et suffixe du tableau. Par defaut : "
            "'probe' pour la fraction 1.0, 'probe_XXpct' sinon."
        ),
    )
    return parser.parse_args()


def nom_sortie(args: argparse.Namespace) -> str:
    """Nomme le dossier de probe et le tableau selon la fraction de labels."""
    if args.suffixe_sortie:
        return args.suffixe_sortie
    if abs(args.fraction_probe - 1.0) < 1e-9:
        return "probe"
    return f"probe_{round(args.fraction_probe * 100):02d}pct"


def executer(commande: list[str], titre: str) -> None:
    print("\n" + "=" * 78)
    print(titre)
    print("=" * 78)
    print(" ".join(commande))
    print("-" * 78, flush=True)
    debut = time.perf_counter()
    resultat = subprocess.run(commande)
    if resultat.returncode != 0:
        raise RuntimeError(f"Echec (code {resultat.returncode}) : {titre}")
    print(f"-> termine en {time.perf_counter() - debut:.0f} s", flush=True)


def dossier_run(args: argparse.Namespace, nom: str) -> Path:
    if nom == "complet" and not args.forcer_complet:
        return RUN_PRINCIPAL
    return args.racine / nom


def preentrainer(args: argparse.Namespace, nom: str) -> Path:
    run_dir = dossier_run(args, nom)
    checkpoint = run_dir / "checkpoint_latest.pt"

    if nom == "complet" and not args.forcer_complet:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Le run principal est introuvable : {checkpoint}. "
                "Utiliser --forcer-complet pour le recalculer."
            )
        print(f"\n[{nom}] run principal reutilise : {checkpoint}")
        return checkpoint

    commande = [
        sys.executable,
        "-m",
        "scripts.preentrainer_experimental",
        "--data-dir", str(args.data_dir),
        "--run-dir", str(run_dir),
        "--augmentations", nom,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--save-every", "0",
        "--print-every", "0",
    ]
    if checkpoint.exists():
        commande += ["--resume", "auto"]
        print(f"\n[{nom}] checkpoint existant detecte -> reprise")

    executer(commande, f"PREENTRAINEMENT [{nom}] — {decrire_configuration(nom)}")
    return checkpoint


def evaluer(args: argparse.Namespace, nom: str, checkpoint: Path) -> Path:
    sortie = args.racine / nom / nom_sortie(args)
    commande = [
        sys.executable,
        "-m",
        "scripts.experimenter_fractions_labels",
        "--data-dir", str(args.data_dir),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(sortie),
        "--fractions", str(args.fraction_probe),
        "--seeds", *[str(s) for s in args.seeds_probe],
        "--evaluer-sur", "test",
    ]
    executer(
        commande,
        f"EVALUATION LINEAIRE [{nom}] — {args.fraction_probe:.0%} des labels",
    )
    return sortie / "resultats.json"


def lire_historique(run_dir: Path) -> dict[str, float]:
    """Recupere la perte finale et le taux de positifs au premier rang."""
    chemin = run_dir / "historique.csv"
    if not chemin.exists():
        return {"loss_finale": float("nan"), "top1_positif": float("nan")}
    with chemin.open(encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))
    if not lignes:
        return {"loss_finale": float("nan"), "top1_positif": float("nan")}
    derniere = lignes[-1]
    return {
        "loss_finale": float(derniere["loss"]),
        "top1_positif": float(derniere["top1_positif"]),
    }


def lire_probe(chemin_json: Path) -> dict[str, float]:
    rapport = json.loads(chemin_json.read_text(encoding="utf-8"))
    for ligne in rapport["resume"]:
        if ligne["encodeur"] == "simclr":
            return {
                "accuracy_moyenne": float(ligne["accuracy_moyenne"]),
                "accuracy_ecart_type": float(ligne["accuracy_ecart_type"]),
                "nombre_seeds": int(ligne["nombre_seeds"]),
            }
    raise RuntimeError(f"Aucun resume 'simclr' dans {chemin_json}")


def main() -> None:
    args = lire_arguments()
    args.racine.mkdir(parents=True, exist_ok=True)
    lignes: list[dict[str, object]] = []

    for nom in args.configurations:
        checkpoint = (
            dossier_run(args, nom) / "checkpoint_latest.pt"
            if args.sauter_preentrainement
            else preentrainer(args, nom)
        )
        chemin_probe = evaluer(args, nom, checkpoint)

        mesures = lire_historique(dossier_run(args, nom))
        probe = lire_probe(chemin_probe)
        lignes.append(
            {
                "configuration": nom,
                "composantes": decrire_configuration(nom),
                "epochs": args.epochs,
                "seed_preentrainement": args.seed,
                "fraction_labels_probe": args.fraction_probe,
                "loss_ntxent_finale": mesures["loss_finale"],
                "top1_positif_final": mesures["top1_positif"],
                "accuracy_probe_moyenne": probe["accuracy_moyenne"],
                "accuracy_probe_ecart_type": probe["accuracy_ecart_type"],
                "nombre_seeds_probe": probe["nombre_seeds"],
            }
        )

    suffixe = "" if nom_sortie(args) == "probe" else f"_{nom_sortie(args)[6:]}"
    chemin_csv = args.racine / f"tableau_ablation{suffixe}.csv"
    with chemin_csv.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        writer.writeheader()
        writer.writerows(lignes)
    (args.racine / f"tableau_ablation{suffixe}.json").write_text(
        json.dumps(lignes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(
        "TABLEAU D'ABLATION DES AUGMENTATIONS "
        f"— evaluation lineaire a {args.fraction_probe:.0%} des labels"
    )
    print("=" * 100)
    print(
        f"{'configuration':<18}{'loss NT-Xent':>14}{'top-1 positif':>15}"
        f"{'ACCURACY LINEAIRE':>22}"
    )
    print(f"{'':<18}{'(diagnostic)':>14}{'(diagnostic)':>15}{'(critere de classement)':>22}")
    print("-" * 100)
    for ligne in lignes:
        print(
            f"{str(ligne['configuration']):<18}"
            f"{float(ligne['loss_ntxent_finale']):>14.4f}"
            f"{float(ligne['top1_positif_final']) * 100:>14.1f} %"
            f"{float(ligne['accuracy_probe_moyenne']) * 100:>17.2f} % "
            f"+/- {float(ligne['accuracy_probe_ecart_type']) * 100:.2f}"
        )
    print("-" * 100)
    print(
        "Rappel : les deux colonnes de gauche sont des diagnostics. Une perte plus\n"
        "basse peut signaler une tache trop facile, donc une representation plus\n"
        "pauvre. Le classement se lit uniquement sur la colonne de droite."
    )
    print(f"\nCSV  : {chemin_csv.resolve()}")


if __name__ == "__main__":
    main()
