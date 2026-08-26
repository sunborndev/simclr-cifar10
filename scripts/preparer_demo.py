"""Construit l'application web de demonstration, en un seul fichier HTML autonome.

Le sujet demande : « Application web : visualisation des representations apprises
(projection t-SNE/UMAP coloree par classe reelle, non utilisee a l'entrainement) ».

CHOIX D'ARCHITECTURE, a defendre en section 7 du rapport
Tout le calcul lourd est fait ici, hors ligne, et le resultat est embarque dans un
unique fichier HTML : projections, vignettes, predictions, matrices. Aucune dependance
a l'execution, aucun serveur, aucun demarrage a froid, rien qui puisse tomber le jour
de la soutenance. Ce n'est pas un renoncement : une projection t-SNE ne peut de toute
facon pas etre recalculee a la demande (l'algorithme optimise la position de tous les
points ensemble et n'a pas d'extension hors echantillon). Le compromis assume est
l'absence de televersement d'image.

L'application montre :
  1. la projection des representations pour les SIX encodeurs du projet — l'encodeur
     aleatoire temoin et les cinq configurations de l'etude d'ablation. On voit la
     representation se degrader quand on retire les augmentations de couleur : c'est
     le tableau d'ablation rendu visible ;
  2. les plus proches voisins DE CHAQUE ENCODEUR, donc la geometrie apprise ;
  3. un visualiseur des deux vues augmentees, configuration par configuration : la
     cause du resultat d'ablation, la ou la projection en montre l'effet ;
  4. deux classifieurs commutables, leurs probabilites et leurs matrices de confusion.

Les classes reelles ne servent qu'a colorer et a mesurer, apres coup. Elles n'ont
jamais ete transmises au preentrainement.
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from simclr.augmentations import (
    ECART_TYPE_CIFAR10,
    MOYENNE_CIFAR10,
    creer_transform_ablation,
    decrire_configuration,
)
from simclr.encodeur import DIMENSION_H, EncodeurResNet18
from simclr.evaluation import (
    SEED_SPLIT_VALIDATION,
    charger_encodeur_simclr,
    creer_indices_stratifies,
    extraire_caracteristiques,
    separer_train_validation,
    standardiser_caracteristiques,
)


ORDRE_ABLATION = ["crop", "crop_flip", "crop_flip_jitter", "complet", "complet_flou"]

LIBELLES = {
    "aleatoire": "Aleatoire",
    "crop": "crop",
    "crop_flip": "crop + flip",
    "crop_flip_jitter": "+ color jitter",
    "complet": "complet",
    "complet_flou": "+ flou",
}

COMMENTAIRES = {
    "crop": (
        "Les deux vues gardent exactement les memes couleurs. Le reseau peut les "
        "apparier en comparant leurs teintes, sans jamais regarder la forme."
    ),
    "crop_flip": (
        "Le retournement ne change ni les couleurs ni le contenu : le raccourci reste "
        "disponible. Aucun gain mesurable."
    ),
    "crop_flip_jitter": (
        "Les couleurs sont perturbees independamment dans chaque vue. Le raccourci est "
        "casse, le reseau doit s'appuyer sur la forme."
    ),
    "complet": (
        "Le passage aleatoire en niveaux de gris supprime parfois toute l'information "
        "de couleur. Configuration retenue pour le projet."
    ),
    "complet_flou": (
        "Un noyau 3x3 sur une image de 32x32 detruit beaucoup pour ce qu'il apporte. "
        "Aucun gain ici, contrairement a ImageNet en 224x224."
    ),
}

EXPLICATIONS = {
    "aleatoire": (
        "Encodeur aleatoire, jamais entraine : le nuage est informe et les couleurs se "
        "melangent. Meme reseau, meme protocole, meme code — seule la representation "
        "change. C'est le temoin qui prouve que le gain vient du preentrainement et non "
        "de l'architecture."
    ),
    "crop": (
        "Preentraine avec le recadrage seul. Le reseau a resolu la tache contrastive a "
        "98 % en comparant les couleurs, sans apprendre les formes : les amas sont flous "
        "et les classes se chevauchent."
    ),
    "crop_flip": (
        "Le retournement horizontal ne casse pas le raccourci des couleurs. Le nuage est "
        "pratiquement identique au precedent."
    ),
    "crop_flip_jitter": (
        "La perturbation des couleurs force le reseau a regarder la forme. Les amas "
        "commencent nettement a se separer."
    ),
    "complet": (
        "Configuration retenue. Les dix classes forment des amas distincts alors "
        "qu'aucune etiquette n'a servi a les construire : la representation encode le "
        "contenu semantique de l'image."
    ),
    "complet_flou": (
        "Le flou gaussien ne change pratiquement rien a cette resolution : le nuage est "
        "equivalent a la configuration complete."
    ),
}


def lire_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/simclr_cifar10_100ep/checkpoint_latest.pt"),
        help="Run principal, qui est aussi la configuration 'complet' de l'ablation.",
    )
    parser.add_argument("--racine-ablation", type=Path, default=Path("outputs/ablation"))
    parser.add_argument(
        "--checkpoint-finetuning",
        type=Path,
        default=None,
        help="Poids du reseau fine-tune (--save-checkpoints). Absent : linear probe seul.",
    )
    parser.add_argument("--sortie", type=Path, default=Path("demo/index.html"))
    parser.add_argument("--nombre-points", type=int, default=2000)
    parser.add_argument("--nombre-voisins", type=int, default=8)
    parser.add_argument("--images-augmentations", type=int, default=12)
    parser.add_argument("--tirages-augmentations", type=int, default=2)
    parser.add_argument("--perplexite", type=float, default=30.0)
    parser.add_argument("--epochs-probe", type=int, default=50)
    parser.add_argument("--nombre-validation", type=int, default=5000)
    parser.add_argument("--seed-split", type=int, default=SEED_SPLIT_VALIDATION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sans-ablation", action="store_true",
                        help="N'utiliser que l'encodeur principal et le temoin aleatoire.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def normaliser(projection) -> list[list[float]]:
    minimum = projection.min(axis=0)
    etendue = projection.max(axis=0) - minimum
    etendue[etendue == 0] = 1.0
    return [
        [round(float(x), 4), round(float(y), 4)]
        for x, y in (projection - minimum) / etendue
    ]


def projeter_2d(nom: str, h: torch.Tensor, args: argparse.Namespace) -> dict[str, list]:
    from sklearn.manifold import TSNE

    donnees = h.numpy()
    sorties: dict[str, list] = {}

    debut = time.perf_counter()
    sorties["tsne"] = normaliser(
        TSNE(
            n_components=2,
            perplexity=args.perplexite,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(donnees)
    )
    print(f"    t-SNE {nom:<18} {time.perf_counter() - debut:5.1f} s", flush=True)

    try:
        import umap  # type: ignore

        debut = time.perf_counter()
        sorties["umap"] = normaliser(
            umap.UMAP(n_components=2, random_state=args.seed).fit_transform(donnees)
        )
        print(f"    UMAP  {nom:<18} {time.perf_counter() - debut:5.1f} s", flush=True)
    except ImportError:
        pass

    return sorties


def calculer_voisins(h: torch.Tensor, k: int) -> list[list[int]]:
    """Les k plus proches voisins de chaque point, par similarite cosinus."""
    normes = torch.nn.functional.normalize(h, dim=1)
    similarites = normes @ normes.T
    similarites.fill_diagonal_(-2.0)
    return similarites.topk(k, dim=1).indices.tolist()


def encoder_planche(images: list, taille: int, colonnes: int) -> str:
    from PIL import Image

    lignes = (len(images) + colonnes - 1) // colonnes
    planche = Image.new("RGB", (colonnes * taille, lignes * taille), (255, 255, 255))
    for position, image in enumerate(images):
        planche.paste(image, ((position % colonnes) * taille, (position // colonnes) * taille))
    tampon = io.BytesIO()
    planche.save(tampon, format="PNG", optimize=True)
    return base64.b64encode(tampon.getvalue()).decode("ascii")


def matrice_confusion(vraies: torch.Tensor, predites: torch.Tensor) -> list[list[int]]:
    matrice = [[0] * 10 for _ in range(10)]
    for vraie, predite in zip(vraies.tolist(), predites.tolist()):
        matrice[vraie][predite] += 1
    return matrice


def entrainer_probe(h_train, y_train, h_test, args, appareil) -> torch.Tensor:
    """Reentraine le meme linear probe que l'experience principale."""
    torch.manual_seed(args.seed)
    tete = nn.Linear(DIMENSION_H, 10).to(appareil)
    nn.init.zeros_(tete.weight)
    nn.init.zeros_(tete.bias)
    optimiseur = torch.optim.SGD(tete.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiseur, T_max=args.epochs_probe, eta_min=1e-4
    )
    perte = nn.CrossEntropyLoss()
    h_gpu, y_gpu = h_train.to(appareil), y_train.to(appareil)
    for _ in range(args.epochs_probe):
        permutation = torch.randperm(len(h_gpu), device=appareil)
        for debut in range(0, len(h_gpu), 256):
            lot = permutation[debut:debut + 256]
            optimiseur.zero_grad()
            perte(tete(h_gpu[lot]), y_gpu[lot]).backward()
            optimiseur.step()
        scheduler.step()
    with torch.no_grad():
        return torch.softmax(tete(h_test.to(appareil)), dim=1).cpu()


def charger_finetuning(chemin: Path, appareil: torch.device) -> nn.Module:
    etat = torch.load(chemin, map_location="cpu", weights_only=False)
    poids = etat["modele"]
    encodeur, tete = EncodeurResNet18(), nn.Linear(DIMENSION_H, 10)
    pe, pc = "encodeur.", "classificateur."
    encodeur.load_state_dict({k[len(pe):]: v for k, v in poids.items() if k.startswith(pe)})
    tete.load_state_dict({k[len(pc):]: v for k, v in poids.items() if k.startswith(pc)})
    return nn.Sequential(encodeur, tete).to(appareil).eval()


def lire_tableau_ablation(racine: Path) -> dict[str, dict]:
    chemin = racine / "tableau_ablation.json"
    if not chemin.exists():
        return {}
    return {
        ligne["configuration"]: ligne
        for ligne in json.loads(chemin.read_text(encoding="utf-8"))
    }


def construire_augmentations(
    dataset_brut: CIFAR10,
    indices: list[int],
    configurations: list[str],
    args: argparse.Namespace,
) -> tuple[dict, list]:
    """Genere les deux vues de chaque image, pour chaque configuration."""
    vers_pil = transforms.ToPILImage()
    tuiles: list = []
    index_original: list[int] = []
    index_vues: list[dict[str, list[list[int]]]] = []

    torch.manual_seed(args.seed)
    for indice in indices:
        image, _ = dataset_brut[indice]
        index_original.append(len(tuiles))
        tuiles.append(image)
        par_config: dict[str, list[list[int]]] = {}
        for nom in configurations:
            transformation = creer_transform_ablation(nom, normaliser=False)
            tirages = []
            for _ in range(args.tirages_augmentations):
                paire = []
                for _ in range(2):
                    paire.append(len(tuiles))
                    tuiles.append(vers_pil(transformation(image)))
                tirages.append(paire)
            par_config[nom] = tirages
        index_vues.append(par_config)

    return {"index_original": index_original, "index_vues": index_vues}, tuiles


def main() -> None:
    args = lire_arguments()
    gabarit = Path(__file__).with_name("gabarit_demo.html")
    if not gabarit.exists():
        raise FileNotFoundError(f"Gabarit HTML introuvable : {gabarit}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint SimCLR introuvable : {args.checkpoint}")

    appareil = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10)]
    )
    train_norm = CIFAR10(args.data_dir, train=True, download=False, transform=transformation)
    test_norm = CIFAR10(args.data_dir, train=False, download=False, transform=transformation)
    test_brut = CIFAR10(args.data_dir, train=False, download=False)
    classes = list(test_norm.classes)

    indices_pool, _ = separer_train_validation(
        train_norm.targets, args.nombre_validation, args.seed_split
    )
    indices_demo = sorted(
        creer_indices_stratifies(test_norm.targets, args.nombre_points, args.seed)
    )
    indices_tenseur = torch.tensor(indices_demo, dtype=torch.long)

    # ---------- inventaire des encodeurs disponibles ----------
    tableau = lire_tableau_ablation(args.racine_ablation)
    sources: list[tuple[str, Path | None]] = [("aleatoire", None)]
    if args.sans_ablation:
        sources.append(("complet", args.checkpoint))
    else:
        for nom in ORDRE_ABLATION:
            chemin = (
                args.checkpoint
                if nom == "complet"
                else args.racine_ablation / nom / "checkpoint_latest.pt"
            )
            if chemin.exists():
                sources.append((nom, chemin))
            else:
                print(f"  ! checkpoint absent, configuration ignoree : {nom} ({chemin})")

    print(f"Appareil                 : {appareil}")
    print(f"Encodeurs                : {', '.join(nom for nom, _ in sources)}")
    print(f"Images projetees         : {len(indices_demo)} (test, equilibrees)")

    def loader(dataset, indices):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=appareil.type == "cuda",
        )

    loader_train = loader(train_norm, indices_pool)
    loader_test = loader(test_norm, list(range(len(test_norm))))

    # ---------- extraction des representations ----------
    representations: dict[str, torch.Tensor] = {}
    h_train_reference = y_train_reference = y_test = None

    torch.manual_seed(args.seed)
    for nom, chemin in sources:
        encodeur = (EncodeurResNet18() if chemin is None else charger_encodeur_simclr(chemin))
        encodeur = encodeur.to(appareil).eval()
        for parametre in encodeur.parameters():
            parametre.requires_grad = False
        debut = time.perf_counter()
        h_train, y_train = extraire_caracteristiques(encodeur, loader_train, appareil)
        h_test, y_test_courant = extraire_caracteristiques(encodeur, loader_test, appareil)
        h_train, h_test = standardiser_caracteristiques(h_train, h_test)
        representations[nom] = h_test
        if nom == "complet":
            h_train_reference, y_train_reference, y_test = h_train, y_train, y_test_courant
        elif y_test is None:
            y_test = y_test_courant
        print(f"  representations {nom:<18} {time.perf_counter() - debut:5.1f} s")
        del encodeur
        if appareil.type == "cuda":
            torch.cuda.empty_cache()

    if h_train_reference is None:
        raise RuntimeError("La configuration 'complet' est indispensable au classifieur")

    # ---------- classifieurs ----------
    classifieurs: list[dict] = []
    predictions: dict[str, list[int]] = {}
    probabilites: dict[str, list[list[float]]] = {}
    matrices: dict[str, list[list[int]]] = {}

    def enregistrer(cle: str, court: str, description: str, probas: torch.Tensor) -> None:
        predites = probas.argmax(dim=1)
        accuracy = (predites == y_test).float().mean().item()
        classifieurs.append(
            {"cle": cle, "court": court, "accuracy": round(accuracy, 6), "description": description}
        )
        predictions[cle] = [int(predites[i]) for i in indices_demo]
        probabilites[cle] = [
            [round(float(v), 4) for v in probas[i].tolist()] for i in indices_demo
        ]
        matrices[cle] = matrice_confusion(y_test, predites)
        print(f"  {court:<14} accuracy test {accuracy * 100:.2f} %")

    print("\nClassifieurs")
    debut = time.perf_counter()
    enregistrer(
        "probe",
        "Linear probe",
        "un classifieur lineaire (512 -> 10) pose sur l'encodeur SimCLR entierement gele. "
        "Seuls ~5 130 poids sont entraines ; l'encodeur n'a jamais vu de label.",
        entrainer_probe(
            h_train_reference, y_train_reference, representations["complet"], args, appareil
        ),
    )
    print(f"  (probe entraine en {time.perf_counter() - debut:.1f} s)")

    if args.checkpoint_finetuning is not None and args.checkpoint_finetuning.exists():
        modele = charger_finetuning(args.checkpoint_finetuning, appareil)
        lots = []
        with torch.no_grad():
            for images, _ in loader_test:
                lots.append(torch.softmax(modele(images.to(appareil)), dim=1).cpu())
        enregistrer(
            "finetuning",
            "Fine-tuning",
            "le reseau entier (~11,2 M poids), initialise avec les poids appris sans labels "
            "par SimCLR puis reentraine sur 100 % des labels.",
            torch.cat(lots),
        )
    else:
        print("  Fine-tuning    ignore (--checkpoint-finetuning non fourni)")

    # ---------- projections et voisins ----------
    print("\nProjections en deux dimensions")
    projections: dict[str, list] = {}
    voisins: dict[str, list[list[int]]] = {}
    methodes: list[str] = []
    for nom in representations:
        h_demo = representations[nom][indices_tenseur]
        for methode, valeurs in projeter_2d(nom, h_demo, args).items():
            projections[f"{nom}_{methode}"] = valeurs
            if methode not in methodes:
                methodes.append(methode)
        voisins[nom] = calculer_voisins(h_demo, args.nombre_voisins)

    # ---------- vignettes ----------
    print("\nPlanches de vignettes")
    colonnes = int(len(indices_demo) ** 0.5) + 1
    sprite = encoder_planche(
        [test_brut[i][0] for i in indices_demo], 32, colonnes
    )
    print(f"  nuage : {len(indices_demo)} vignettes, {len(sprite) / 1024:.0f} Ko")

    configurations_aug = [nom for nom, _ in sources if nom != "aleatoire"]
    indices_aug = creer_indices_stratifies(
        test_norm.targets, args.images_augmentations, args.seed + 1
    )
    aug, tuiles_aug = construire_augmentations(test_brut, indices_aug, configurations_aug, args)
    colonnes_aug = 1 + len(configurations_aug) * args.tirages_augmentations * 2
    sprite_aug = encoder_planche(tuiles_aug, 32, colonnes_aug)
    print(f"  augmentations : {len(tuiles_aug)} vignettes, {len(sprite_aug) / 1024:.0f} Ko")

    aug.update(
        {
            "images": indices_aug,
            "nombre_tirages": args.tirages_augmentations,
            "configurations": [
                {
                    "cle": nom,
                    "libelle": LIBELLES.get(nom, nom),
                    "composantes": decrire_configuration(nom),
                    "accuracy": round(float(tableau[nom]["accuracy_probe_moyenne"]), 6)
                    if nom in tableau else None,
                    "loss": round(float(tableau[nom]["loss_ntxent_finale"]), 4)
                    if nom in tableau else None,
                    "top1": round(float(tableau[nom]["top1_positif_final"]), 4)
                    if nom in tableau else None,
                    "commentaire": COMMENTAIRES.get(nom, ""),
                }
                for nom in configurations_aug
            ],
        }
    )

    donnees = {
        "classes": classes,
        "n": len(indices_demo),
        "methodes": methodes,
        "encodeurs": [
            {
                "cle": nom,
                "libelle": LIBELLES.get(nom, nom),
                "composantes": "poids jamais entraines"
                if nom == "aleatoire" else decrire_configuration(nom),
                "accuracy": round(float(tableau[nom]["accuracy_probe_moyenne"]), 6)
                if nom in tableau else None,
                "explication": EXPLICATIONS.get(nom, ""),
            }
            for nom, _ in sources
        ],
        "classifieurs": classifieurs,
        "labels": [int(y_test[i]) for i in indices_demo],
        "predictions": predictions,
        "probabilites": probabilites,
        "matrices": matrices,
        "voisins": voisins,
        "projections": projections,
        "sprite": {"cols": colonnes, "taille": 32},
        "sprite_aug": {"cols": colonnes_aug, "taille": 32},
        "aug": aug,
        "stats": [
            ["Images de test", "10 000"],
            ["Images projetees", str(len(indices_demo))],
            ["Encodeurs comparables", str(len(sources))],
            ["Dimension de h", str(DIMENSION_H)],
            ["Epoques de preentrainement", "100"],
            ["Labels vus au preentrainement", "0"],
        ],
    }

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    html = (
        gabarit.read_text(encoding="utf-8")
        .replace("__DONNEES__", json.dumps(donnees, separators=(",", ":")))
        .replace("__SPRITE_AUG__", sprite_aug)
        .replace("__SPRITE__", sprite)
        .replace(
            "__SOUS_TITRE__",
            "Encodeur ResNet18 preentraine 100 epoques par apprentissage contrastif sur "
            "50 000 images sans etiquettes. Projection "
            + " et ".join(m.upper() for m in methodes)
            + f" de {len(indices_demo)} images de test, pour {len(sources)} encodeurs.",
        )
    )
    args.sortie.write_text(html, encoding="utf-8")
    print(
        f"\nApplication ecrite : {args.sortie.resolve()}  "
        f"({args.sortie.stat().st_size / 1024**2:.1f} Mo)"
    )
    print("Fichier autonome : aucune installation, aucun serveur. Double-clic pour ouvrir.")


if __name__ == "__main__":
    main()
