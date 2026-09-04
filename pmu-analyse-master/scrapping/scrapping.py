import logging
import requests
from datetime import timedelta
from data_traitement.traitement import save_race_data

# Calcul les dates intermédiaires entre deux dates données
def get_race_dates(start_date, end_date):
    current_date = start_date
    race_dates = []

    while current_date <= end_date:
        race_dates.append(current_date.strftime("%d%m%Y"))
        current_date += timedelta(days=1)

    return race_dates

# Apelle l'api pmu afin de récupérer l'ensemble des données des courses, réunions, hippodrome, participants
def call_api_between_dates(start_date, end_date, progress_callback=None, should_stop=None):
    """
    progress_callback(current_date, reunion_number, status, data) est appelé,
    si fourni, après chaque appel API : status vaut "ok", "end_of_day" ou "error".
    Sert à brancher une UI (barre de progression Streamlit, etc.) sans changer
    le comportement de l'appel en mode CLI (main.py) où progress_callback vaut None.

    should_stop() est un callable optionnel, réévalué entre chaque appel API ;
    s'il renvoie True, la boucle s'arrête proprement (utile pour un bouton
    "Stop" dans l'UI plutôt que de laisser tourner un scraping trop long).
    """
    current_date = start_date
    while current_date <= end_date:
        if should_stop and should_stop():
            logging.info("Scraping interrompu par l'utilisateur.")
            return

        reunion_number = 1
        while True:
            if should_stop and should_stop():
                logging.info("Scraping interrompu par l'utilisateur.")
                return

            base_url = "https://online.turfinfo.api.pmu.fr/rest/client/61/programme/{}/{}?specialisation=INTERNET"
            url = base_url.format(current_date.strftime("%d%m%Y"), f"R{reunion_number}")

            headers = {
                'accept': 'application/json',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            logging.debug(f"Attempting to call API for {current_date}, {reunion_number}""")
            try:
                response = requests.get(url, headers=headers, timeout=15)
            except requests.RequestException as exc:
                # Une erreur réseau (timeout, DNS, connexion refusée...) ne doit pas
                # faire planter tout le scraping en cours : on logge, on remonte
                # l'info à l'UI si besoin, et on passe à la réunion suivante.
                logging.error(f"Network error calling API for {current_date}, {reunion_number}: {exc}")
                if progress_callback:
                    progress_callback(current_date, reunion_number, "error", None)
                reunion_number += 1
                continue

            if response.status_code == 204:
                # The reunion does not exist
                logging.info(f"No more reunion [{reunion_number}] on {current_date} : 204")
                if progress_callback:
                    progress_callback(current_date, reunion_number, "end_of_day", None)
                break
            elif response.status_code == 200:
                # Courses are available for this reunion
                data = response.json()
                logging.debug(f"Response 200 for reunions [{reunion_number}] on {current_date}")
                save_race_data(data)
                scrap_participants(current_date, reunion_number, data)
                if progress_callback:
                    progress_callback(current_date, reunion_number, "ok", data)
            else:
                logging.error(f"API request failed. Status code: {response.status_code}, Date: {current_date}, Reunion: {reunion_number}")
                if progress_callback:
                    progress_callback(current_date, reunion_number, "error", None)

            reunion_number += 1  # Move to the next reunion

        current_date += timedelta(days=1)  # Move to the next date


def scrap_participants(current_date, reunion_number, data):
    pass
