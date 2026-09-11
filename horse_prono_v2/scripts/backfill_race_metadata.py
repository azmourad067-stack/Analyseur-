from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from typing import Any

import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    ),
)

from app_core.db import upsert_races
from app_core.pmu import get_programme


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def text(value, default=""):
    if value is None:
        return default

    if isinstance(value, dict):
        value = (
            value.get("libelle")
            or value.get("nom")
            or value.get("value")
        )

    if value is None:
        return default

    value = str(value).strip()

    return value or default


def normalize_discipline(
    course: dict[str, Any],
) -> str:

    discipline = text(
        course.get("discipline"),
        "INCONNU",
    ).upper()

    specialite = text(
        course.get("specialite"),
        "",
    ).upper()

    category = text(
        course.get(
            "categorieParticularite"
        ),
        "",
    ).upper()

    combined = (
        f"{discipline} "
        f"{specialite} "
        f"{category}"
    )

    if "ATTELE" in combined:

        if "AUTOSTART" in combined:
            return "ATTELE_AUTOSTART"

        return "ATTELE_VOLTE"

    if "MONTE" in combined:
        return "TROT_MONTE"

    if "STEEPLE" in combined:
        return "STEEPLECHASE"

    if "CROSS" in combined:
        return "CROSS_COUNTRY"

    if "HAIE" in combined:
        return "HAIES"

    if "PLAT" in combined:
        return "PLAT"

    return discipline


def extract_terrain(
    reunion: dict,
    course: dict,
) -> str:

    candidates = [
        course.get("terrain"),
        course.get("etatTerrain"),
        course.get("etatPiste"),
        course.get("natureTerrain"),
        reunion.get("terrain"),
        reunion.get("etatTerrain"),
        reunion.get("etatPiste"),
    ]

    meteo = reunion.get("meteo")

    if isinstance(meteo, dict):

        candidates.extend(
            [
                meteo.get("terrain"),
                meteo.get("etatTerrain"),
                meteo.get("etatPiste"),
            ]
        )

    for candidate in candidates:

        value = text(candidate)

        if value:
            return value

    return "INCONNU"


def programme_metadata(
    programme: Any,
    target_date: date,
) -> pd.DataFrame:

    if not isinstance(programme, dict):
        return pd.DataFrame()

    root = programme.get("programme")

    if not isinstance(root, dict):
        root = programme

    reunions = root.get("reunions")

    if not isinstance(reunions, list):
        return pd.DataFrame()

    rows = []

    for reunion in reunions:

        if not isinstance(reunion, dict):
            continue

        reunion_number = as_int(
            reunion.get("numOfficiel")
            or reunion.get("numReunion")
            or reunion.get(
                "numReunionProgramme"
            )
        )

        if reunion_number is None:
            continue

        hippodrome_obj = reunion.get(
            "hippodrome"
        )

        if isinstance(
            hippodrome_obj,
            dict,
        ):

            hippodrome = text(
                hippodrome_obj.get(
                    "libelleCourt"
                )
                or hippodrome_obj.get(
                    "libelleLong"
                )
                or hippodrome_obj.get(
                    "nom"
                ),
                "INCONNU",
            )

        else:

            hippodrome = text(
                hippodrome_obj,
                "INCONNU",
            )

        courses = reunion.get(
            "courses",
            [],
        )

        if not isinstance(
            courses,
            list,
        ):
            continue

        for course in courses:

            if not isinstance(
                course,
                dict,
            ):
                continue

            course_number = as_int(
                course.get("numOrdre")
                or course.get("numCourse")
                or course.get(
                    "numOfficiel"
                )
            )

            if course_number is None:
                continue

            rows.append(
                {
                    "race_id":
                        (
                            f"R{reunion_number}"
                            f"C{course_number}_"
                            f"{target_date.isoformat()}"
                        ),

                    "race_date":
                        pd.Timestamp(
                            target_date
                        ),

                    "reunion":
                        reunion_number,

                    "course_number":
                        course_number,

                    "discipline":
                        normalize_discipline(
                            course
                        ),

                    "hippodrome":
                        hippodrome,

                    "distance":
                        as_int(
                            course.get(
                                "distance"
                            )
                        ),

                    "terrain":
                        extract_terrain(
                            reunion,
                            course,
                        ),

                    "field_size":
                        as_int(
                            course.get(
                                "nombreDeclaresPartants"
                            )
                            or course.get(
                                "nombrePartants"
                            )
                        ),

                    "status":
                        "finished",
                }
            )

    return pd.DataFrame(rows)


def backfill_day(
    target_date: date,
) -> int:

    programme = get_programme(
        target_date
    )

    races = programme_metadata(
        programme,
        target_date,
    )

    if races.empty:

        print(
            f"{target_date}: "
            "aucune course interprétable",
            flush=True,
        )

        return 0

    written = upsert_races(
        races
    )

    discipline_ok = (
        races["discipline"]
        .astype(str)
        .str.upper()
        .ne("INCONNU")
        .sum()
    )

    distance_ok = (
        pd.to_numeric(
            races["distance"],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    hippo_ok = (
        races["hippodrome"]
        .astype(str)
        .str.upper()
        .ne("INCONNU")
        .sum()
    )

    terrain_ok = (
        races["terrain"]
        .astype(str)
        .str.upper()
        .ne("INCONNU")
        .sum()
    )

    print(
        f"{target_date}: "
        f"{written} courses | "
        f"discipline {discipline_ok}/{len(races)} | "
        f"distance {distance_ok}/{len(races)} | "
        f"hippodrome {hippo_ok}/{len(races)} | "
        f"terrain {terrain_ok}/{len(races)}",
        flush=True,
    )

    return written


def backfill_range(
    start: date,
    end: date,
) -> None:

    if end < start:

        raise SystemExit(
            "La date de fin doit être "
            ">= à la date de début"
        )

    total = 0

    current = start

    while current <= end:

        try:

            total += backfill_day(
                current
            )

        except Exception as exc:

            print(
                f"{current}: "
                f"ERREUR: {exc}",
                flush=True,
            )

        current += timedelta(
            days=1
        )

    print()
    print(
        f"TOTAL {start} -> {end}: "
        f"{total} courses mises à jour",
        flush=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Backfill métadonnées "
            "PMU HorseProno"
        )
    )

    parser.add_argument(
        "--start",
        required=True,
    )

    parser.add_argument(
        "--end",
        required=True,
    )

    args = parser.parse_args()

    backfill_range(
        date.fromisoformat(
            args.start
        ),
        date.fromisoformat(
            args.end
        ),
    )
