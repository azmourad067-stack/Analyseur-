from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from .config import PMU_BASE_URL, REQUEST_TIMEOUT


HEADERS = {
    "User-Agent": "HorsePronoStreamlit/1.0 (+https://streamlit.io)",
    "Accept": "application/json",
}


# ============================================================
# HTTP / API PMU
# ============================================================

def _get_json(url: str) -> Any:
    """
    Effectue une requête HTTP GET et retourne le JSON.
    """
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


def get_programme(race_date: date) -> Any:
    """
    Récupère le programme PMU d'une journée.
    Exemple :
        /programme/02092026
    """

    d = race_date.strftime("%d%m%Y")

    url = f"{PMU_BASE_URL}/programme/{d}"

    return _get_json(url)


def get_participants(
    race_date: date,
    reunion: int,
    course: int,
) -> Any:
    """
    Récupère les participants d'une course PMU.
    Exemple :
        /programme/02092026/R1/C4/participants
    """

    d = race_date.strftime("%d%m%Y")

    url = (
        f"{PMU_BASE_URL}/programme/"
        f"{d}/R{reunion}/C{course}/participants"
    )

    return _get_json(url)


# ============================================================
# OUTILS
# ============================================================

def _as_int(value):
    def _weight_kg(value):
    """
    Convertit les poids PMU vers des kilogrammes.

    Exemples observés :
        580 -> 58.0 kg
        585 -> 58.5 kg
        550 -> 55.0 kg
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value <= 0:
        return None

    # Les poids PMU observés sont exprimés
    # en dixièmes de kilogramme.
    if value >= 200:
        value = value / 10.0

    if not 30 <= value <= 100:
        return None

    return round(value, 1)
    """
    Convertit une valeur en entier de façon sûre.
    """

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


def _pick(obj: dict, *paths, default=None):
    """
    Cherche une valeur dans plusieurs chemins possibles.

    Exemple :
        _pick(obj, "jockey.nom", "driver.nom", "jockey")
    """

    for path in paths:

        current = obj

        if isinstance(path, str):
            path = path.split(".")

        try:

            for part in path:
                current = current[part]

            if current is not None:
                return current

        except (KeyError, TypeError, IndexError):
            continue

    return default


def _find_list_of_dicts(
    obj: Any,
    keys_hint: tuple[str, ...],
) -> list[dict]:
    """
    Recherche récursivement des listes contenant
    des dictionnaires ressemblant à des participants.
    """

    found: list[dict] = []

    if isinstance(obj, dict):

        for _, value in obj.items():

            if isinstance(value, list):

                if all(isinstance(x, dict) for x in value):

                    if any(
                        any(key in item for key in keys_hint)
                        for item in value
                    ):
                        found.extend(value)

                # On continue également la recherche à l'intérieur
                # de la liste pour être compatible avec plusieurs
                # structures JSON PMU.
                for item in value:
                    found.extend(
                        _find_list_of_dicts(
                            item,
                            keys_hint,
                        )
                    )

            elif isinstance(value, dict):

                found.extend(
                    _find_list_of_dicts(
                        value,
                        keys_hint,
                    )
                )

    elif isinstance(obj, list):

        for item in obj:

            found.extend(
                _find_list_of_dicts(
                    item,
                    keys_hint,
                )
            )

    return found


# ============================================================
# PROGRAMME PMU
# ============================================================

def programme_choices(programme: Any) -> list[dict]:
    """
    Extrait toutes les courses disponibles dans le programme PMU.

    Structures supportées :

        {
            "reunions": [...]
        }

    ou :

        {
            "programme": {
                "reunions": [...]
            }
        }

    Retour :

        [
            {"reunion": 1, "course": 1},
            {"reunion": 1, "course": 2},
            ...
        ]
    """

    choices: list[dict] = []

    if not isinstance(programme, dict):
        return choices

    # --------------------------------------------------------
    # Le programme peut être directement à la racine
    # ou enveloppé dans {"programme": {...}}
    # --------------------------------------------------------

    root = programme.get("programme")

    if not isinstance(root, dict):
        root = programme

    reunions = root.get("reunions")

    # --------------------------------------------------------
    # Fallback : recherche récursive d'une clé "reunions"
    # --------------------------------------------------------

    if not isinstance(reunions, list):

        def find_reunions(obj):

            if isinstance(obj, dict):

                if isinstance(obj.get("reunions"), list):
                    return obj["reunions"]

                for value in obj.values():

                    result = find_reunions(value)

                    if result is not None:
                        return result

            elif isinstance(obj, list):

                for value in obj:

                    result = find_reunions(value)

                    if result is not None:
                        return result

            return None

        reunions = find_reunions(programme)

    if not isinstance(reunions, list):
        return choices

    # --------------------------------------------------------
    # Parcours réunions -> courses
    # --------------------------------------------------------

    for reunion_obj in reunions:

        if not isinstance(reunion_obj, dict):
            continue

        reunion = _as_int(
            reunion_obj.get("numOfficiel")
            or reunion_obj.get("numReunion")
            or reunion_obj.get("numReunionProgramme")
            or reunion_obj.get("numero")
        )

        courses = reunion_obj.get("courses", [])

        if not isinstance(courses, list):
            continue

        for course_obj in courses:

            if not isinstance(course_obj, dict):
                continue

            # Certains JSON répètent numReunion
            # dans chaque objet course.
            current_reunion = reunion

            if current_reunion is None:

                current_reunion = _as_int(
                    course_obj.get("numReunion")
                    or course_obj.get("numReunionProgramme")
                )

            course = _as_int(
                course_obj.get("numOrdre")
                or course_obj.get("numCourse")
                or course_obj.get("numOfficiel")
                or course_obj.get("numero")
            )

            if current_reunion is None:
                continue

            if course is None:
                continue

            choices.append(
                {
                    "reunion": current_reunion,
                    "course": course,
                }
            )

    # --------------------------------------------------------
    # Suppression des doublons
    # --------------------------------------------------------

    unique = {}

    for choice in choices:

        key = (
            choice["reunion"],
            choice["course"],
        )

        unique[key] = choice

    return sorted(
        unique.values(),
        key=lambda x: (
            x["reunion"],
            x["course"],
        ),
    )


# ============================================================
# PARTICIPANTS PMU -> DATAFRAME
# ============================================================

def participants_to_df(
    payload: Any,
    race_date: date,
    reunion: int,
    course: int,
) -> pd.DataFrame:
    """
    Transforme le JSON participants PMU
    en DataFrame standard HorseProno.
    """

    candidates = _find_list_of_dicts(
        payload,
        (
            "numPmu",
            "numero",
            "numParticipant",
            "nom",
            "nomCheval",
            "cheval",
            "jockey",
            "driver",
        ),
    )

    # --------------------------------------------------------
    # Suppression des doublons
    # --------------------------------------------------------

    unique_participants = []

    seen = set()

    for participant in candidates:

        if not isinstance(participant, dict):
            continue

        number = _pick(
            participant,
            "numPmu",
            "numero",
            "numParticipant",
            default=None,
        )

        name = _pick(
            participant,
            "nom",
            "nomCheval",
            "cheval.nom",
            default="",
        )

        key = (
            str(number),
            str(name),
        )

        if key in seen:
            continue

        if number is None and not name:
            continue

        seen.add(key)

        unique_participants.append(participant)

    # --------------------------------------------------------
    # Transformation des participants
    # --------------------------------------------------------

    rows = []

    for participant in unique_participants:

        number = _pick(
            participant,
            "numPmu",
            "numero",
            "numParticipant",
            default=None,
        )

        name = _pick(
            participant,
            "nom",
            "nomCheval",
            "cheval.nom",
            default="",
        )

        # ----------------------------------------------------
        # Cote
        # ----------------------------------------------------

        odds = _pick(
            participant,
            "dernierRapportDirect",
            "dernierRapport",
            "cote",
            "coteMatin",
            "coteReference",
            default=None,
        )

        if isinstance(odds, dict):

            odds = _pick(
                odds,
                "rapport",
                "valeur",
                "value",
                default=None,
            )

        # ----------------------------------------------------
        # Jockey / driver
        # ----------------------------------------------------

        jockey = _pick(
            participant,
            "jockey.nom",
            "driver.nom",
            "jockey",
            "driver",
            default="",
        )

        if isinstance(jockey, dict):

            jockey = _pick(
                jockey,
                "nom",
                "libelle",
                default="",
            )

        # ----------------------------------------------------
        # Entraîneur
        # ----------------------------------------------------

        trainer = _pick(
            participant,
            "entraineur.nom",
            "trainer.nom",
            "entraineur",
            "trainer",
            default="",
        )

        if isinstance(trainer, dict):

            trainer = _pick(
                trainer,
                "nom",
                "libelle",
                default="",
            )

        # ----------------------------------------------------
        # Numéro de corde
        # ----------------------------------------------------

        draw = _pick(
            participant,
            "corde",
            "numCorde",
            "placeCorde",
            default=None,
        )

        # ----------------------------------------------------
        # Poids
        # ----------------------------------------------------

       age = _as_int(
    _pick(
        participant,
        "age",
        default=None,
    )
)

sex = _pick(
    participant,
    "sexe",
    "sex",
    default="",
)

condition_weight = _weight_kg(
    _pick(
        participant,
        "poidsConditionMonte",
        default=None,
    )
)

handicap_weight = _weight_kg(
    _pick(
        participant,
        "handicapPoids",
        default=None,
    )
)

generic_weight = _weight_kg(
    _pick(
        participant,
        "poids",
        "poidsCheval",
        default=None,
    )
)

weight = (
    condition_weight
    or handicap_weight
    or generic_weight
)

        # ----------------------------------------------------
        # Musique
        # ----------------------------------------------------

        recent_form = _pick(
            participant,
            "musique",
            "performance",
            "musiqueCheval",
            default="",
        )

        # ----------------------------------------------------
        # Position arrivée
        # ----------------------------------------------------

        finish_position = _pick(
            participant,
            "placeFinal",
            "ordreArrivee",
            "arrivee",
            "positionArrivee",
            "classement",
            default=None,
        )

        # ----------------------------------------------------
        # Informations course
        # ----------------------------------------------------

        discipline = _pick(
            payload,
            "discipline",
            "course.discipline",
            default=None,
        )

        if discipline is None:

            discipline = _pick(
                participant,
                "discipline",
                default="INCONNU",
            )

        hippodrome = _pick(
            payload,
            "hippodrome.libelleCourt",
            "hippodrome.nom",
            "reunion.hippodrome.libelleCourt",
            "reunion.hippodrome.nom",
            "course.hippodrome.nom",
            default="INCONNU",
        )

        distance = _pick(
            payload,
            "distance",
            "course.distance",
            default=None,
        )

        terrain = _pick(
            payload,
            "terrain.libelle",
            "terrain",
            "course.terrain.libelle",
            default="INCONNU",
        )

        if isinstance(terrain, dict):

            terrain = _pick(
                terrain,
                "libelle",
                "nom",
                default="INCONNU",
            )

        # ----------------------------------------------------
        # Ligne finale
        # ----------------------------------------------------

        rows.append(
            {
                "race_id": (
                    f"R{reunion}C{course}_"
                    f"{race_date.isoformat()}"
                ),

                "race_date": pd.Timestamp(race_date),

                "reunion": reunion,

                "course_number": course,

                "discipline": discipline,

                "hippodrome": hippodrome,

                "distance": distance,

                "terrain": terrain,

                "field_size": None,

                "horse_number": number,

                "horse_name": name,

                "jockey": jockey,

                "trainer": trainer,

                "odds": odds,

                "draw": draw,

                "weight": weight, "age": age,
"sex": sex,

                "recent_form": recent_form,

                "career_runs": None,

                "career_wins": None,

                "career_places": None,

                "finish_position": finish_position,
            }
        )

    df = pd.DataFrame(rows)

    # Nombre de partants de la course
    if not df.empty:

        df["field_size"] = len(df)

    return df
