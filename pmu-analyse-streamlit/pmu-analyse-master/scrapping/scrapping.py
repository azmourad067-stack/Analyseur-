import logging
import requests
from datetime import timedelta
from data_traitement.traitement import save_race_data
from database.database import save_participants

HEADERS = {
    'accept': 'application/json',
    'user-agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
}

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/61/programme/{date}/R{reunion}?specialisation=INTERNET"
PARTICIPANTS_URL = ("https://online.turfinfo.api.pmu.fr/rest/client/61/programme/"
                    "{date}/R{reunion}/C{course}/participants?specialisation=INTERNET")


def get_race_dates(start_date, end_date):
    """Calcule les dates intermédiaires entre deux dates données."""
    current_date = start_date
    race_dates = []
    while current_date <= end_date:
        race_dates.append(current_date.strftime("%d%m%Y"))
        current_date += timedelta(days=1)
    return race_dates


def iter_reunions(start_date, end_date):
    """Itère sur toutes les réunions trouvées entre deux dates.

    Génère des tuples (date, numéro_réunion, données_json).
    """
    current_date = start_date
    while current_date <= end_date:
        reunion_number = 1
        while True:
            url = BASE_URL.format(date=current_date.strftime("%d%m%Y"), reunion=reunion_number)
            try:
                response = requests.get(url, headers=HEADERS, timeout=20)
            except requests.RequestException as exc:
                logging.error(f"Erreur réseau pour {url}: {exc}")
                break

            if response.status_code == 204:
                # La réunion n'existe pas : on passe au jour suivant
                logging.info(f"No more reunion [{reunion_number}] on {current_date} : 204")
                break
            elif response.status_code == 200:
                data = response.json()
                logging.debug(f"Response 200 for reunions [{reunion_number}] on {current_date}")
                yield current_date, reunion_number, data
            else:
                logging.error(f"API request failed. Status code: {response.status_code}, "
                              f"Date: {current_date}, Reunion: {reunion_number}")
            reunion_number += 1
        current_date += timedelta(days=1)


def scrap_participants(current_date, reunion_number, data):
    """Récupère et enregistre les participants de chaque course d'une réunion.

    Retourne (nb_courses, nb_participants).
    """
    courses_data = data.get('courses', [])
    nb_participants = 0
    for course in courses_data:
        num_ordre = course.get('numOrdre')
        url = PARTICIPANTS_URL.format(date=current_date.strftime("%d%m%Y"),
                                      reunion=reunion_number,
                                      course=num_ordre)
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
        except requests.RequestException as exc:
            logging.error(f"Erreur réseau pour {url}: {exc}")
            continue
        if response.status_code == 200:
            payload = response.json()
            participants = payload.get('participants', [])
            save_participants(participants, course, data)
            nb_participants += len(participants)
            logging.debug(f"{len(participants)} participants pour R{reunion_number}C{num_ordre}")
        else:
            logging.error(f"Participants request failed. Status: {response.status_code}, URL: {url}")
    return len(courses_data), nb_participants


def call_api_between_dates(start_date, end_date, progress_callback=None):
    """Appelle l'API PMU pour récupérer réunions, courses et participants entre deux dates."""
    total_days = max((end_date - start_date).days + 1, 1)
    counts = {"reunions": 0, "courses": 0, "participants": 0}
    for i, (current_date, reunion_number, data) in enumerate(iter_reunions(start_date, end_date)):
        if progress_callback:
            progress_callback(i, total_days, current_date, reunion_number)
        save_race_data(data)
        counts["reunions"] += 1
        n_courses, n_participants = scrap_participants(current_date, reunion_number, data)
        counts["courses"] += n_courses
        counts["participants"] += n_participants
    return counts
