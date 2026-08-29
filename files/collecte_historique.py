"""Collecte l'historique des arrivées de courses PMU via le projet open-pmu-api
(https://github.com/.../open-pmu-api), pour constituer un jeu de données local
exploitable par calibrer_poids.py.

⚠️ À EXÉCUTER EN LOCAL, pas dans l'environnement ayant servi à générer ce
projet (celui-ci n'a pas accès au domaine vercel.app). Ce script fait un
appel HTTP par jour sur la période demandée — pour janvier 2025 à juillet
2026 (~570 jours), prévoir plusieurs dizaines de minutes d'exécution, un
délai volontaire étant appliqué entre chaque appel pour ne pas surcharger
ce service gratuit et non officiel (usage personnel/ponctuel recommandé).

Usage :
    pip install requests
    python collecte_historique.py

Sortie : historique_courses.csv (une ligne par cheval par course).
"""

import csv
import sys
import time
from datetime import date, timedelta

import requests

# Force l'affichage immédiat des print(), même quand la sortie standard n'est
# pas un terminal interactif (certains IDE/lanceurs bufferisent sinon tout
# jusqu'à la fin du script, donnant l'impression que "rien ne se passe").
sys.stdout.reconfigure(line_buffering=True)

API_BASE = "https://open-pmu-api.vercel.app/api/arrivees"
DELAI_ENTRE_APPELS_SECONDES = 1.5
TIMEOUT_SECONDES = 15

DATE_DEBUT = date(2025, 1, 1)
DATE_FIN = date(2026, 7, 31)
FICHIER_SORTIE = "historique_courses.csv"

COLONNES_CSV = [
    "date", "r_c", "lieu", "type", "distance", "prix", "partants",
    "numero", "nom_cheval", "sexe", "annee_naissance", "nom_jockey",
    "nom_entraineur", "musique", "cote", "gains", "corde",
    "position_arrivee", "top3", "vainqueur",
]

# Date connue comme ayant des données selon le README du projet (couverture
# annoncée jusqu'au 18/08/2026), utilisée uniquement pour détecter le bon
# format de date avant de lancer la collecte complète.
DATE_TEST_DETECTION = date(2026, 8, 18)


def _appeler_api(date_str: str) -> dict | None:
    try:
        reponse = requests.get(API_BASE, params={"date": date_str}, timeout=TIMEOUT_SECONDES)
        reponse.raise_for_status()
        return reponse.json()
    except (requests.RequestException, ValueError):
        return None


def detecter_format_date() -> str:
    """Le README documente DD/MM/YYYY, mais son propre exemple ('08/18/2026')
    ne respecte pas ce format (18 n'est pas un mois valide). On teste les
    deux avant de lancer la collecte, plutôt que de le découvrir 570 appels
    plus tard.
    """
    print("Test de connexion à l'API (détection du format de date)...")
    candidats = {
        "%d/%m/%Y": DATE_TEST_DETECTION.strftime("%d/%m/%Y"),
        "%m/%d/%Y": DATE_TEST_DETECTION.strftime("%m/%d/%Y"),
    }
    for format_str, date_str in candidats.items():
        data = _appeler_api(date_str)
        if data and not data.get("error") and data.get("message"):
            print(f"Format de date détecté : {format_str} (test réussi sur {date_str})")
            return format_str
    sys.exit(
        "Impossible de confirmer le format de date attendu par l'API — "
        "le service est peut-être indisponible, ou sa structure a changé. "
        "Teste manuellement : "
        f"{API_BASE}?date={DATE_TEST_DETECTION.strftime('%d/%m/%Y')}"
    )


def recuperer_jour(jour: date, format_date: str) -> list[dict]:
    data = _appeler_api(jour.strftime(format_date))
    if not data or data.get("error"):
        return []
    return data.get("message", [])


def extraire_lignes(course: dict, jour: date) -> list[dict]:
    """Convertit une course brute de l'API en lignes plates (une par cheval)."""
    arrivee = course.get("arrivee") or []
    podium = set(arrivee[:3])
    vainqueur = arrivee[0] if arrivee else None

    lignes = []
    for numero_str, cheval in (course.get("arrivee_details") or {}).items():
        try:
            numero = int(numero_str)
        except (TypeError, ValueError):
            continue
        cotes = cheval.get("cotes") or []
        lignes.append(
            {
                "date": jour.isoformat(),
                "r_c": course.get("r/c"),
                "lieu": course.get("lieu"),
                "type": course.get("type"),
                "distance": course.get("distance"),
                "prix": course.get("prix"),
                "partants": course.get("partants"),
                "numero": numero,
                "nom_cheval": cheval.get("nom_cheval"),
                "sexe": cheval.get("sexe"),
                "annee_naissance": cheval.get("annee_de_naissance"),
                "nom_jockey": cheval.get("nom_jockey"),
                "nom_entraineur": cheval.get("nom_entraineur"),
                "musique": cheval.get("musique"),
                # Dernière cote de la liste = la plus proche du départ, selon
                # l'exemple du README (liste croissante dans le temps).
                "cote": cotes[-1] if cotes else None,
                "gains": cheval.get("gains"),
                "corde": cheval.get("corde"),
                "position_arrivee": (arrivee.index(numero) + 1) if numero in arrivee else None,
                "top3": int(numero in podium),
                "vainqueur": int(numero == vainqueur),
            }
        )
    return lignes


def collecter(date_debut: date, date_fin: date, chemin_sortie: str, format_date: str) -> None:
    jour = date_debut
    total_lignes = 0
    total_jours_avec_courses = 0
    nb_jours_total = (date_fin - date_debut).days + 1

    with open(chemin_sortie, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_CSV)
        writer.writeheader()

        for i in range(nb_jours_total):
            courses = recuperer_jour(jour, format_date)
            if courses:
                total_jours_avec_courses += 1
                lignes_du_jour = []
                for course in courses:
                    lignes_du_jour.extend(extraire_lignes(course, jour))
                writer.writerows(lignes_du_jour)
                total_lignes += len(lignes_du_jour)
                print(f"  {jour.isoformat()} : {len(courses)} course(s), {len(lignes_du_jour)} partant(s)")

            if (i + 1) % 25 == 0:
                print(f"--- {i + 1}/{nb_jours_total} jours traités ---")

            jour += timedelta(days=1)
            time.sleep(DELAI_ENTRE_APPELS_SECONDES)

    print()
    print(
        f"Terminé. {total_jours_avec_courses} jour(s) avec courses trouvées, "
        f"{total_lignes} ligne(s) cheval/course écrites dans {chemin_sortie}."
    )
    if total_lignes == 0:
        print(
            "Aucune donnée récupérée : vérifie que le service est en ligne "
            f"({API_BASE}) et que la période demandée est bien couverte."
        )


if __name__ == "__main__":
    print(
        f"Collecte de {DATE_DEBUT.isoformat()} à {DATE_FIN.isoformat()} "
        f"({(DATE_FIN - DATE_DEBUT).days + 1} jours) depuis {API_BASE}"
    )
    format_date = detecter_format_date()
    print("Démarrage de la collecte complète (peut prendre du temps)...")
    print()
    collecter(DATE_DEBUT, DATE_FIN, FICHIER_SORTIE, format_date)
