from datetime import datetime
from database.database import save_pays, save_hippodrome, save_reunions, save_courses


def save_race_data(reunion_data):
    """Enregistre une réunion (et ses courses) dans la base. Retourne la liste des courses."""
    # Copie pour ne pas muter le dictionnaire d'origine
    data = dict(reunion_data)

    # Nettoyage des infos inutiles
    data.pop('parisEvenement', None)
    data.pop('meteo', None)
    data.pop('offresInternet', None)
    data.pop('regionHippique', None)
    data.pop('cagnottes', None)
    data['dateReunion'] = datetime.utcfromtimestamp(data['dateReunion'] / 1000.0)

    save_pays(data.get('pays', {}))
    save_hippodrome(data.get('hippodrome', {}))

    courses_data = data.get('courses', {})
    save_reunions(data)
    save_courses(courses_data)

    return courses_data
