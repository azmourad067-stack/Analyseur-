from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from .config import PMU_BASE_URL, REQUEST_TIMEOUT

HEADERS = {
    "User-Agent": "HorsePronoStreamlit/1.0 (+https://streamlit.io)",
    "Accept": "application/json",
}


def _get_json(url: str) -> Any:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_programme(race_date: date) -> Any:
    # Community-documented PMU endpoint. PMU does not expose a stable public API
    # contract for this endpoint, so this adapter is intentionally defensive.
    d = race_date.strftime("%d%m%Y")
    return _get_json(f"{PMU_BASE_URL}/programme/{d}")


def get_participants(race_date: date, reunion: int, course: int) -> Any:
    d = race_date.strftime("%d%m%Y")
    url = f"{PMU_BASE_URL}/programme/{d}/R{reunion}/C{course}/participants"
    return _get_json(url)


def _find_list_of_dicts(obj: Any, keys_hint: tuple[str, ...]) -> list[dict]:
    found: list[dict] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                if any(any(h in x for h in keys_hint) for x in v):
                    found.extend(v)
            else:
                found.extend(_find_list_of_dicts(v, keys_hint))
    elif isinstance(obj, list):
        for x in obj:
            found.extend(_find_list_of_dicts(x, keys_hint))
    return found


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def programme_choices(programme):
    """
    Extrait les couples (réunion, course) depuis le programme PMU.

    Structure PMU attendue :
        programme
          └── reunions
                └── courses
    """

    choices = []

    if not isinstance(programme, dict):
        return choices

    # ---------------------------------------------------------
    # 1. Structure PMU standard : reunions -> courses
    # ---------------------------------------------------------
    reunions = programme.get("reunions")

    if isinstance(reunions, list):
        for reunion_obj in reunions:

            if not isinstance(reunion_obj, dict):
                continue

            reunion = (
                reunion_obj.get("numOfficiel")
                or reunion_obj.get("numReunion")
                or reunion_obj.get("numReunionProgramme")
                or reunion_obj.get("numOrdre")
            )

            reunion = _as_int(reunion)

            if reunion is None:
                continue

            courses = reunion_obj.get("courses", [])

            if not isinstance(courses, list):
                continue

            for course_obj in courses:

                if not isinstance(course_obj, dict):
                    continue

                course = (
                    course_obj.get("numOrdre")
                    or course_obj.get("numCourse")
                    or course_obj.get("numOfficiel")
                )

                course = _as_int(course)

                if course is None:
                    continue

                choices.append(
                    {
                        "reunion": reunion,
                        "course": course,
                    }
                )

    # ---------------------------------------------------------
    # 2. Suppression des doublons
    # ---------------------------------------------------------
    unique = {}

    for item in choices:
        key = (item["reunion"], item["course"])
        unique[key] = item

    return list(unique.values())


def _pick(obj: dict, *paths, default=None):
    for p in paths:
        cur = obj
        if isinstance(p, str):
            p = p.split(".")
        try:
            for part in p:
                cur = cur[part]
            if cur is not None:
                return cur
        except Exception:
            continue
    return default


def participants_to_df(payload: Any, race_date: date, reunion: int, course: int) -> pd.DataFrame:
    # PMU payload schemas vary; locate participant-like dictionaries defensively.
    candidates = _find_list_of_dicts(payload, ("numPmu", "numero", "nom", "cheval", "jockey", "driver"))
    # De-duplicate by participant number or object identity.
    unique = []
    seen = set()
    for p in candidates:
        number = _pick(p, "numPmu", "numero", "numParticipant", default=None)
        name = _pick(p, "nom", "nomCheval", "cheval.nom", default="")
        key = (str(number), str(name))
        if key in seen or (number is None and not name):
            continue
        seen.add(key)
        unique.append(p)

    rows = []
    for p in unique:
        odds = _pick(p, "dernierRapportDirect", "dernierRapport", "cote", "coteMatin", "coteReference", default=None)
        if isinstance(odds, dict):
            odds = _pick(odds, "rapport", "valeur", "value", default=None)
        rows.append({
            "race_id": f"R{reunion}C{course}_{race_date.isoformat()}",
            "race_date": pd.Timestamp(race_date),
            "discipline": _pick(p, "discipline", default="INCONNU"),
            "hippodrome": _pick(payload, "hippodrome.libelleCourt", "hippodrome.nom", "reunion.hippodrome.nom", default="INCONNU"),
            "distance": _pick(payload, "distance", "course.distance", default=None),
            "terrain": _pick(payload, "terrain.libelle", "terrain", default="INCONNU"),
            "field_size": None,
            "horse_number": number,
            "horse_name": name,
            "jockey": _pick(p, "jockey.nom", "jockey", "driver.nom", "driver", default=""),
            "trainer": _pick(p, "entraineur.nom", "entraineur", "trainer.nom", "trainer", default=""),
            "odds": odds,
            "draw": _pick(p, "corde", "numCorde", default=None),
            "weight": _pick(p, "poids", "poidsCheval", default=None),
            "recent_form": _pick(p, "musique", "performance", "musiqueCheval", default=""),
            "career_runs": None,
            "career_wins": None,
            "career_places": None,
            "finish_position": _pick(p, "placeFinal", "ordreArrivee", "arrivee", "positionArrivee", "classement", default=None),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["field_size"] = len(df)
    return df
