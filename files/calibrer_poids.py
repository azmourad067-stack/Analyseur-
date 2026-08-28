"""Calibre des statistiques empiriques à partir de l'historique de courses
collecté par collecte_historique.py (fichier historique_courses.csv).

⚠️ CE QUE CE SCRIPT NE FAIT PAS : il ne recalibre pas automatiquement les
5 critères de scoring.py à l'identique. Deux d'entre eux (aptitude distance,
aptitude terrain) sont des jugements qualitatifs qu'aucune donnée historique
de ce type ne permet de dériver, et le poids porté n'est pas fourni par
cette API historique (contrairement à l'API "programme du jour" utilisée
par pmu_connector.py pour les courses à venir).

CE QUE CE SCRIPT CALCULE RÉELLEMENT, à partir des seules données
disponibles dans l'historique (musique, corde, gains, âge, jockey,
entraîneur) :
  1. Le taux de podium (top 3) par jockey et par entraîneur sur la période
     collectée — un signal directement exploitable pour remplacer
     l'évaluation manuelle "Fort/Moyen/Faible" par un vrai chiffre.
  2. Une régression logistique estimant l'importance relative de chaque
     critère disponible pour prédire une entrée dans le top 3.

Usage :
    pip install pandas scikit-learn
    python calibrer_poids.py [chemin_vers_historique_courses.csv]

Sortie : resultats_calibration.json
"""

import json
import sys

import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("scikit-learn est requis : pip install scikit-learn")

from pmu_connector import estimer_forme_depuis_musique  # réutilise le parseur déjà testé

SEUIL_MIN_COURSES_ACTEUR = 5  # nb minimal de courses pour qu'un taux jockey/entraîneur soit retenu
TAUX_PAR_DEFAUT_ACTEUR_INCONNU = 0.25  # valeur neutre si jockey/entraîneur sous le seuil ou absent


def charger(chemin_csv: str) -> pd.DataFrame:
    df = pd.read_csv(chemin_csv)
    for col in ("cote", "gains", "corde", "annee_naissance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["annee_course"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df["age"] = df["annee_course"] - df["annee_naissance"]
    df["forme_estimee"] = df["musique"].apply(
        lambda m: estimer_forme_depuis_musique(m) if isinstance(m, str) else None
    )
    return df


def calculer_taux_par_acteur(df: pd.DataFrame, colonne_acteur: str) -> dict:
    """Taux de podium par jockey ou entraîneur, avec seuil minimal de courses.

    ⚠️ Calcul global sur tout l'historique collecté (pas de séparation
    entraînement/validation) : statistique descriptive utile, mais à
    interpréter avec prudence en dessous de quelques dizaines de courses.
    """
    stats = (
        df.groupby(colonne_acteur)["top3"]
        .agg(nb_courses="count", taux_podium="mean")
        .query("nb_courses >= @SEUIL_MIN_COURSES_ACTEUR")
        .sort_values("taux_podium", ascending=False)
    )
    return stats.round(3).to_dict(orient="index")


def normaliser_par_course(df: pd.DataFrame, colonne: str, inverser: bool = False) -> pd.Series:
    """Normalise une colonne entre 0 et 1, relativement aux autres partants
    de la même course (même logique que scoring.py, appliquée ici pour que
    le modèle appris reflète des écarts relatifs et non des échelles brutes).
    """
    groupe = df.groupby("r_c")[colonne]
    minimum, maximum = groupe.transform("min"), groupe.transform("max")
    ecart = (maximum - minimum).replace(0, pd.NA)
    normalise = (df[colonne] - minimum) / ecart
    normalise = normalise.fillna(0.5)
    return (1 - normalise) if inverser else normalise


def ajuster_modele(df: pd.DataFrame, taux_jockey: dict, taux_entraineur: dict) -> dict:
    travail = df.dropna(subset=["forme_estimee", "corde", "gains", "age", "top3", "r_c"]).copy()

    travail["taux_jockey"] = travail["nom_jockey"].map(
        lambda j: taux_jockey.get(j, {}).get("taux_podium", TAUX_PAR_DEFAUT_ACTEUR_INCONNU)
    )
    travail["taux_entraineur"] = travail["nom_entraineur"].map(
        lambda e: taux_entraineur.get(e, {}).get("taux_podium", TAUX_PAR_DEFAUT_ACTEUR_INCONNU)
    )

    travail["norme_forme"] = normaliser_par_course(travail, "forme_estimee")
    travail["norme_corde"] = normaliser_par_course(travail, "corde", inverser=True)
    travail["norme_gains"] = normaliser_par_course(travail, "gains")
    travail["norme_age"] = normaliser_par_course(travail, "age", inverser=True)

    caracteristiques = [
        "norme_forme", "norme_corde", "norme_gains", "norme_age",
        "taux_jockey", "taux_entraineur",
    ]
    X = travail[caracteristiques].fillna(0.5)
    y = travail["top3"]

    if len(travail) < 200 or y.nunique() < 2:
        return {
            "avertissement": (
                f"Échantillon insuffisant ({len(travail)} ligne(s) exploitable(s)) "
                "pour une calibration fiable. Élargis la période de collecte, ou "
                "vérifie le format de date utilisé dans collecte_historique.py "
                "si ce nombre te semble anormalement bas."
            )
        }

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    modele = LogisticRegression(max_iter=1000)
    modele.fit(X_norm, y)

    importance_brute = dict(zip(caracteristiques, abs(modele.coef_[0])))
    total = sum(importance_brute.values()) or 1
    importance_relative = {k: round(v / total, 3) for k, v in importance_brute.items()}

    return {
        "nb_lignes_utilisees": len(travail),
        "taux_podium_moyen_general": round(float(y.mean()), 3),
        "importance_relative_des_criteres": importance_relative,
        "coefficients_bruts_standardises": {
            k: round(float(v), 4) for k, v in zip(caracteristiques, modele.coef_[0])
        },
    }


def main():
    chemin_csv = sys.argv[1] if len(sys.argv) > 1 else "historique_courses.csv"
    print(f"Chargement de {chemin_csv}...")
    df = charger(chemin_csv)
    print(f"{len(df)} ligne(s) cheval/course chargée(s).")

    print("Calcul des taux de podium par jockey et par entraîneur...")
    taux_jockey = calculer_taux_par_acteur(df, "nom_jockey")
    taux_entraineur = calculer_taux_par_acteur(df, "nom_entraineur")
    print(f"  {len(taux_jockey)} jockey(s) et {len(taux_entraineur)} entraîneur(s) "
          f"au-dessus du seuil de {SEUIL_MIN_COURSES_ACTEUR} courses.")

    print("Ajustement de la régression logistique...")
    resultat_modele = ajuster_modele(df, taux_jockey, taux_entraineur)

    sortie = {
        "periode_analysee": {
            "debut": str(df["date"].min()) if len(df) else None,
            "fin": str(df["date"].max()) if len(df) else None,
        },
        "nb_courses_distinctes": int(df["r_c"].nunique()) if "r_c" in df else None,
        "taux_podium_par_jockey": taux_jockey,
        "taux_podium_par_entraineur": taux_entraineur,
        "modele": resultat_modele,
    }

    with open("resultats_calibration.json", "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)

    print()
    print("Résultats écrits dans resultats_calibration.json")
    if "importance_relative_des_criteres" in resultat_modele:
        print("Importance relative des critères disponibles (trouvée par le modèle) :")
        for critere, valeur in resultat_modele["importance_relative_des_criteres"].items():
            print(f"  {critere} : {valeur}")
    elif "avertissement" in resultat_modele:
        print(f"⚠️ {resultat_modele['avertissement']}")


if __name__ == "__main__":
    main()
