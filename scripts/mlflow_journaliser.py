"""Rejoue tout l'historique experimental du projet dans MLflow.

Le suivi d'experiences a ete tenu jusqu'ici en CSV et JSON dans `outputs/`.
Ce script relit ces fichiers et les rejoue dans MLflow, sans rien recalculer :
aucune carte graphique n'est necessaire, et l'execution prend quelques minutes.

Cinq experiences sont creees, dans l'ordre du travail :

  01_preentrainement       les 5 preentrainements SimCLR, avec leurs courbes
                           epoque par epoque
  02_recherche_lr          le balayage du taux d'apprentissage sur validation,
                           applique identiquement aux deux methodes
  03_budget_labels         fine-tuning contre supervise from scratch,
                           3 fractions x 3 graines
  04_evaluation_lineaire   les sondes lineaires sur encodeur gele
  05_ablation              les 5 configurations d'augmentations

Le script est idempotent : relance-le autant que tu veux, il saute les runs
deja journalises (reperes par leur nom).

Le stockage se fait dans une base SQLite locale (`mlflow.db`). Le stockage sur
fichiers simples est passe en mode maintenance dans les versions recentes de
MLflow, et surtout le REGISTRE DE MODELES exige une base de donnees : c'est
donc le seul choix compatible avec la suite du travail.

Usage :
    python -m scripts.mlflow_journaliser
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import csv
import json
from pathlib import Path

import mlflow


RACINE_ARTEFACTS = Path("mlartifacts")


# ---------------------------------------------------------------- utilitaires

def lire_json(chemin: Path) -> dict | list | None:
    if not chemin.exists():
        return None
    return json.loads(chemin.read_text(encoding="utf-8"))


def lire_csv(chemin: Path) -> list[dict]:
    if not chemin.exists():
        return []
    with chemin.open(encoding="utf-8") as fichier:
        return list(csv.DictReader(fichier))


def assurer_experience(nom: str, racine_artefacts: Path) -> None:
    """Cree l'experience si besoin, avec un emplacement d'artefacts explicite.

    Sans emplacement explicite, l'interface et le client peuvent chercher les
    artefacts a deux endroits differents et les figures n'apparaissent pas.
    """
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(nom) is None:
        dossier = (racine_artefacts / nom).resolve()
        dossier.mkdir(parents=True, exist_ok=True)
        client.create_experiment(nom, artifact_location=dossier.as_uri())
    mlflow.set_experiment(nom)


def noms_deja_journalises(experience: str) -> set[str]:
    """Retourne les noms de runs deja presents dans une experience."""
    client = mlflow.MlflowClient()
    info = client.get_experiment_by_name(experience)
    if info is None:
        return set()
    noms = set()
    for run in client.search_runs([info.experiment_id], max_results=50_000):
        nom = run.data.tags.get("mlflow.runName")
        if nom:
            noms.add(nom)
    return noms


def journaliser(
    experience: str,
    nom: str,
    parametres: dict,
    metriques: dict,
    etiquettes: dict | None = None,
    courbes: list[dict] | None = None,
    artefacts: list[Path] | None = None,
    deja: set[str] | None = None,
) -> bool:
    """Cree un run MLflow. Retourne False s'il existait deja."""
    if deja is not None and nom in deja:
        return False

    with mlflow.start_run(run_name=nom):
        mlflow.log_params({k: v for k, v in parametres.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metriques.items() if v is not None})
        if etiquettes:
            mlflow.set_tags(etiquettes)
        for ligne in courbes or []:
            etape = int(ligne.pop("_etape"))
            for cle, valeur in ligne.items():
                mlflow.log_metric(cle, float(valeur), step=etape)
        for chemin in artefacts or []:
            if chemin.exists():
                mlflow.log_artifact(str(chemin))
    return True


# ------------------------------------------------------- 01 preentrainement

CONFIGS_PREENTRAINEMENT = {
    "simclr_cifar10_100ep": ("complet", "run principal du projet"),
    "ablation/crop": ("crop", "ablation : recadrage seul"),
    "ablation/crop_flip": ("crop_flip", "ablation : + retournement"),
    "ablation/crop_flip_jitter": ("crop_flip_jitter", "ablation : + color jitter"),
    "ablation/complet_flou": ("complet_flou", "ablation : + flou gaussien"),
}


def journaliser_preentrainements(sorties: Path) -> int:
    experience = "01_preentrainement"
    assurer_experience(experience, RACINE_ARTEFACTS)
    deja = noms_deja_journalises(experience)
    compte = 0

    for dossier, (configuration, description) in CONFIGS_PREENTRAINEMENT.items():
        historique = lire_csv(sorties / dossier / "historique.csv")
        if not historique:
            print(f"  ! historique absent, ignore : {dossier}")
            continue

        derniere = historique[-1]
        meilleure_perte = min(float(l["loss"]) for l in historique)

        # La duree totale est du temps d'horloge : elle inclut les periodes ou
        # la machine faisait autre chose. La MEDIANE par epoque est insensible
        # a ces pics et donne le cout de calcul reel. Le rapport entre les deux
        # mesure la contamination.
        durees = sorted(float(l["duree_secondes"]) for l in historique)
        mediane = durees[len(durees) // 2]
        duree_horloge = sum(float(l["duree_secondes"]) for l in historique)
        duree_propre = mediane * len(historique)
        courbes = [
            {
                "_etape": int(float(ligne["epoch"])),
                "perte_ntxent": ligne["loss"],
                "top1_positif": ligne["top1_positif"],
                "learning_rate": ligne["learning_rate"],
                "duree_epoque_s": ligne["duree_secondes"],
                "vram_pic_mo": ligne["vram_pic_mo"],
            }
            for ligne in historique
        ]

        cree = journaliser(
            experience,
            nom=f"preentrainement_{configuration}",
            parametres={
                "configuration_augmentations": configuration,
                "epochs": len(historique),
                "batch_size": 256,
                "vues_par_batch": 512,
                "negatifs_par_ancre": 510,
                "optimiseur": "AdamW",
                "learning_rate_max": 3e-4,
                "warmup_epochs": 10,
                "scheduler": "cosinus",
                "temperature": 0.5,
                "architecture": "ResNet18 adapte 32x32",
                "dimension_h": 512,
                "dimension_z": 128,
                "images": 50_000,
                "labels_utilises": 0,
                "seed": 42,
                "precision_mixte": True,
            },
            metriques={
                "perte_ntxent_finale": derniere["loss"],
                "perte_ntxent_meilleure": meilleure_perte,
                "top1_positif_final": derniere["top1_positif"],
                "duree_horloge_s": duree_horloge,
                "duree_epoque_mediane_s": mediane,
                "duree_calcul_estimee_s": duree_propre,
                "facteur_contamination": duree_horloge / max(duree_propre, 1e-9),
                "vram_pic_mo": max(float(l["vram_pic_mo"]) for l in historique),
            },
            etiquettes={
                "etape_projet": "preentrainement",
                "supervise": "non",
                "description": description,
                "avertissement_duree": (
                    "duree_horloge_s inclut les periodes ou la machine etait "
                    "partagee. Utiliser duree_calcul_estimee_s pour toute "
                    "comparaison de cout."
                ),
            },
            courbes=courbes,
            artefacts=[sorties / dossier / "historique.csv"],
            deja=deja,
        )
        compte += int(cree)
    return compte


# --------------------------------------------------------- 02 recherche lr

def journaliser_balayage_lr(sorties: Path) -> int:
    experience = "02_recherche_lr"
    assurer_experience(experience, RACINE_ARTEFACTS)
    deja = noms_deja_journalises(experience)
    compte = 0

    for dossier, methode in (
        ("finetuning_recherche_lr", "finetuning_simclr"),
        ("supervise_recherche_lr", "supervise_from_scratch"),
    ):
        for chemin in sorted((sorties / dossier).glob("*/resultats.json")):
            rapport = lire_json(chemin)
            if not rapport:
                continue
            config = rapport["configuration"]
            for ligne in rapport["resultats"]:
                lr = config["learning_rate"]
                cree = journaliser(
                    experience,
                    nom=f"{methode}_lr_{lr}",
                    parametres={
                        "methode": methode,
                        "learning_rate": lr,
                        "fraction_labels": ligne["fraction_labels"],
                        "nombre_labels": ligne["nombre_labels"],
                        "seed": ligne["seed"],
                        "epochs": config["epochs"],
                        "batch_size": config["batch_size"],
                        "optimiseur": "SGD nesterov",
                        "momentum": config["momentum"],
                        "weight_decay": config["weight_decay"],
                        "evaluer_sur": config["evaluer_sur"],
                    },
                    metriques={
                        "accuracy_validation": ligne["accuracy_evaluation"],
                        "accuracy_train": ligne["accuracy_train"],
                        "perte_train": ligne["loss_train"],
                        "duree_s": ligne["duree_secondes"],
                    },
                    etiquettes={
                        "etape_projet": "reglage_hyperparametres",
                        "note": "meme grille appliquee aux deux methodes, "
                                "aucun avantage de reglage",
                    },
                    deja=deja,
                )
                compte += int(cree)
    return compte


# ------------------------------------------------------- 03 budget labels

def journaliser_budget_labels(sorties: Path) -> int:
    experience = "03_budget_labels"
    assurer_experience(experience, RACINE_ARTEFACTS)
    deja = noms_deja_journalises(experience)
    compte = 0

    series = (
        ("finetuning_simclr_45k", "finetuning_simclr", "protocole final"),
        ("supervise_from_scratch_45k", "supervise_from_scratch", "protocole final"),
        ("supervise_from_scratch", "supervise_from_scratch", "serie 50k, perimee"),
        ("finetuning_demo", "finetuning_simclr", "modele conserve pour la demo"),
    )

    for dossier, methode, statut in series:
        rapport = lire_json(sorties / dossier / "resultats.json")
        if not rapport:
            continue
        config = rapport["configuration"]
        for ligne in rapport["resultats"]:
            pourcent = int(round(float(ligne["fraction_labels"]) * 100))
            cree = journaliser(
                experience,
                nom=f"{dossier}_{pourcent}pct_seed{ligne['seed']}",
                parametres={
                    "methode": methode,
                    "fraction_labels": ligne["fraction_labels"],
                    "nombre_labels": ligne["nombre_labels"],
                    "seed": ligne["seed"],
                    "learning_rate": config.get("learning_rate"),
                    "epochs": config.get("epochs"),
                    "batch_size": config.get("batch_size"),
                    "nombre_validation": config.get("nombre_validation"),
                    "seed_split": config.get("seed_split"),
                    "train_disponible": config.get("nombre_train_total"),
                    "augmentation_train": config.get("augmentation_train"),
                    "poids_entrainables": "~11,2 M",
                },
                metriques={
                    "accuracy_test": ligne.get("accuracy_evaluation",
                                               ligne.get("accuracy_test")),
                    "accuracy_train": ligne["accuracy_train"],
                    "perte_train": ligne["loss_train"],
                    "duree_s": ligne["duree_secondes"],
                },
                etiquettes={
                    "etape_projet": "campagne_principale",
                    "statut": statut,
                    "encodeur_gele": "non",
                },
                artefacts=[sorties / dossier / "resultats.csv"],
                deja=deja,
            )
            compte += int(cree)
    return compte


# -------------------------------------------------- 04 evaluation lineaire

def journaliser_evaluation_lineaire(sorties: Path) -> int:
    experience = "04_evaluation_lineaire"
    assurer_experience(experience, RACINE_ARTEFACTS)
    deja = noms_deja_journalises(experience)
    compte = 0

    series = (
        ("experience_labels_45k", "protocole final 45k"),
        ("simclr_cifar10_100ep/experience_labels_v2", "serie 50k, perimee"),
        ("simclr_cifar10_100ep/experience_labels", "v1 sans standardisation, diagnostic"),
    )

    for dossier, statut in series:
        rapport = lire_json(sorties / dossier / "resultats.json")
        if not rapport:
            continue
        config = rapport["configuration"]
        for ligne in rapport["resultats"]:
            pourcent = int(round(float(ligne["fraction_labels"]) * 100))
            nom = (f"{Path(dossier).name}_{ligne['encodeur']}"
                   f"_{pourcent}pct_seed{ligne['seed']}")
            cree = journaliser(
                experience,
                nom=nom,
                parametres={
                    "encodeur": ligne["encodeur"],
                    "fraction_labels": ligne["fraction_labels"],
                    "nombre_labels": ligne["nombre_labels"],
                    "seed": ligne["seed"],
                    "epochs": config.get("epochs"),
                    "learning_rate": config.get("learning_rate"),
                    "standardiser_h": config.get("standardiser_h"),
                    "initialisation_zero": config.get("initialisation_zero"),
                    "poids_entrainables": "~5 130",
                },
                metriques={
                    "accuracy_test": ligne.get("accuracy_evaluation",
                                               ligne.get("accuracy_test")),
                    "accuracy_train": ligne["accuracy_train"],
                    "perte_train": ligne["loss_train"],
                    "duree_s": ligne["duree_probe_secondes"],
                },
                etiquettes={
                    "etape_projet": "evaluation_lineaire",
                    "statut": statut,
                    "encodeur_gele": "oui",
                },
                deja=deja,
            )
            compte += int(cree)
    return compte


# --------------------------------------------------------------- 05 ablation

def journaliser_ablation(sorties: Path) -> int:
    experience = "05_ablation"
    assurer_experience(experience, RACINE_ARTEFACTS)
    deja = noms_deja_journalises(experience)
    tableau = lire_json(sorties / "ablation" / "tableau_ablation.json")
    if not tableau:
        print("  ! tableau_ablation.json absent")
        return 0

    compte = 0
    for ligne in tableau:
        cree = journaliser(
            experience,
            nom=f"ablation_{ligne['configuration']}",
            parametres={
                "configuration": ligne["configuration"],
                "composantes": ligne["composantes"],
                "epochs": ligne["epochs"],
                "seed_preentrainement": ligne["seed_preentrainement"],
                "graines_probe": ligne["nombre_seeds_probe"],
            },
            metriques={
                # Les deux premieres sont des DIAGNOSTICS : une perte plus basse
                # signale une tache prétexte plus facile, pas une meilleure
                # representation. Le classement se lit sur accuracy_probe.
                "perte_ntxent_finale": ligne["loss_ntxent_finale"],
                "top1_positif_final": ligne["top1_positif_final"],
                "accuracy_probe": ligne["accuracy_probe_moyenne"],
                "accuracy_probe_ecart_type": ligne["accuracy_probe_ecart_type"],
            },
            etiquettes={
                "etape_projet": "ablation",
                "critere_de_classement": "accuracy_probe",
                "avertissement": "ne PAS classer par perte_ntxent : correlation "
                                 "de rang +0,80 avec l'accuracy, donc classement inverse",
            },
            artefacts=[sorties / "ablation" / "tableau_ablation.csv"],
            deja=deja,
        )
        compte += int(cree)
    return compte


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sorties", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--racine-suivi",
        default="sqlite:///mlflow.db",
        help=(
            "URI du serveur de suivi. Base SQLite locale par defaut : le "
            "stockage sur fichiers est deprecie et ne supporte pas le registre."
        ),
    )
    parser.add_argument(
        "--racine-artefacts",
        type=Path,
        default=Path("mlartifacts"),
        help="Dossier ou MLflow depose les artefacts (CSV, figures).",
    )
    args = parser.parse_args()

    global RACINE_ARTEFACTS
    RACINE_ARTEFACTS = args.racine_artefacts

    if not args.sorties.exists():
        raise FileNotFoundError(f"Dossier de resultats introuvable : {args.sorties}")

    mlflow.set_tracking_uri(args.racine_suivi)
    print(f"Serveur de suivi : {args.racine_suivi}\n")

    etapes = (
        ("01_preentrainement", journaliser_preentrainements),
        ("02_recherche_lr", journaliser_balayage_lr),
        ("03_budget_labels", journaliser_budget_labels),
        ("04_evaluation_lineaire", journaliser_evaluation_lineaire),
        ("05_ablation", journaliser_ablation),
    )

    total = 0
    for nom, fonction in etapes:
        print(f"{nom}")
        cree = fonction(args.sorties)
        total += cree
        print(f"  -> {cree} nouveaux runs\n")

    print(f"{total} runs journalises au total.")
    print("\nPour consulter l'interface :")
    print(f"    mlflow ui --backend-store-uri {args.racine_suivi}")
    print("puis ouvrir http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
