from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    ),
)

from app_core.db import get_supabase_client
from app_core.pmu import (
    get_participants,
    participants_to_df,
    programme_choices,
    get_programme,
)


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def backfill_day(target_date: date) -> tuple[int, int]:

    client = get_supabase_client()

    if client is None:
        raise RuntimeError(
            "Supabase non configuré"
        )

    programme = get_programme(
        target_date
    )

    choices = programme_choices(
        programme
    )

    races_seen = 0
    updated = 0

    for choice in choices:

        reunion = choice["reunion"]
        course = choice["course"]

        external_race_id = (
            f"R{reunion}C{course}_"
            f"{target_date.isoformat()}"
        )

        # ----------------------------------------------------
        # Récupère races.id interne
        # ----------------------------------------------------

        race_response = (
            client
            .table("races")
            .select("id")
            .eq(
                "external_id",
                external_race_id,
            )
            .limit(1)
            .execute()
        )

        race_rows = (
            race_response.data
            or []
        )

        if not race_rows:
            continue

        race_id = int(
            race_rows[0]["id"]
        )

        # ----------------------------------------------------
        # Participants déjà présents en base
        # ----------------------------------------------------

        existing_response = (
            client
            .table("participants")
            .select(
                "horse_number"
            )
            .eq(
                "race_id",
                race_id,
            )
            .execute()
        )

        existing_numbers = {
            int(row["horse_number"])
            for row in (
                existing_response.data
                or []
            )
            if row.get(
                "horse_number"
            ) is not None
        }

        if not existing_numbers:
            continue

        # ----------------------------------------------------
        # Payload PMU
        # ----------------------------------------------------

        try:

            payload = get_participants(
                target_date,
                reunion,
                course,
            )

            df = participants_to_df(
                payload,
                target_date,
                reunion,
                course,
            )

        except Exception as exc:

            print(
                f"{target_date} "
                f"R{reunion}C{course}: "
                f"{exc}",
                flush=True,
            )

            continue

        if df.empty:
            continue

        races_seen += 1

        rows = []

        for _, participant in df.iterrows():

            number = clean_value(
                participant.get(
                    "horse_number"
                )
            )

            try:
                number = int(number)
            except (TypeError, ValueError):
                continue

            # Ne jamais créer de participant
            # nouveau pendant le backfill.
            if number not in existing_numbers:
                continue

            age = clean_value(
                participant.get(
                    "age"
                )
            )

            if age is not None:
                try:
                    age = int(age)
                except (TypeError, ValueError):
                    age = None

            weight = clean_value(
                participant.get(
                    "weight"
                )
            )

            if weight is not None:
                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    weight = None

            sex = clean_value(
                participant.get(
                    "sex"
                )
            )

            horse_name = clean_value(
                participant.get(
                    "horse_name"
                )
            )

            rows.append(
                {
                    "race_id":
                        race_id,

                    "horse_number":
                        number,

                    "horse_name":
                        str(
                            horse_name
                            or "Inconnu"
                        ),

                    "age":
                        age,

                    "sex":
                        (
                            str(sex)
                            if sex
                            else None
                        ),

                    "weight_kg":
                        weight,
                }
            )

        if not rows:
            continue

        # Upsert partiel :
        # seules les colonnes présentes sont mises à jour.
        response = (
            client
            .table("participants")
            .upsert(
                rows,
                on_conflict=(
                    "race_id,"
                    "horse_number"
                ),
            )
            .execute()
        )

        updated += len(rows)

    print(
        f"{target_date}: "
        f"{races_seen} courses / "
        f"{updated} participants mis à jour",
        flush=True,
    )

    return (
        races_seen,
        updated,
    )


def backfill_range(
    start: date,
    end: date,
) -> None:

    total_races = 0
    total_participants = 0

    current = start

    while current <= end:

        try:

            races, participants = (
                backfill_day(
                    current
                )
            )

            total_races += races
            total_participants += participants

        except Exception as exc:

            print(
                f"{current}: "
                f"ERREUR JOURNÉE: {exc}",
                flush=True,
            )

        current += timedelta(
            days=1
        )

    print()
    print("=" * 60)

    print(
        f"TOTAL {start} -> {end}: "
        f"{total_races} courses / "
        f"{total_participants} participants"
    )

    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
