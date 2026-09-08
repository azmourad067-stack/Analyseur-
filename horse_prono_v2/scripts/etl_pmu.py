from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from app_core.db import save_ingestion_run, upsert_participants, upsert_races
from app_core.pmu import get_participants, get_programme, participants_to_df, programme_choices


def race_row(participants: pd.DataFrame) -> dict:
    first = participants.iloc[0]
    return {
        "race_id": str(first["race_id"]),
        "race_date": pd.Timestamp(first["race_date"]).date().isoformat(),
        "reunion": _int(first.get("reunion")),
        "course_number": _int(first.get("course_number")),
        "discipline": str(first.get("discipline") or "INCONNU"),
        "hippodrome": str(first.get("hippodrome") or "INCONNU"),
        "distance": _int(first.get("distance")),
        "terrain": str(first.get("terrain") or "INCONNU"),
        "field_size": _int(first.get("field_size")),
        "source": "pmu_adapter",
        "source_url": "https://online.turfinfo.api.pmu.fr/rest/client/1",
    }


def _int(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


def ingest_day(target: date) -> tuple[int, int]:
    programme = get_programme(target)
    choices = programme_choices(programme)
    read_rows = 0
    written_rows = 0
    for choice in choices:
        participants = participants_to_df(get_participants(target, choice["reunion"], choice["course"]), target, choice["reunion"], choice["course"])
        if participants.empty:
            continue
        participants["reunion"] = choice["reunion"]
        participants["course_number"] = choice["course"]
        read_rows += len(participants)
        upsert_races(pd.DataFrame([race_row(participants)]))
        written_rows += upsert_participants(participants.drop(columns=["reunion", "course_number"], errors="ignore"))
    save_ingestion_run("pmu", "success", read_rows, written_rows, f"Date {target.isoformat()} : {len(choices)} courses détectées.")
    return read_rows, written_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion PMU -> Supabase")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--days", type=int, default=1, help="Nombre de jours consécutifs à ingérer")
    args = parser.parse_args()
    start = date.fromisoformat(args.date)
    for i in range(args.days):
        target = start + timedelta(days=i)
        try:
            r, w = ingest_day(target)
            print(f"{target}: {r} lignes lues, {w} écrites")
        except Exception as exc:
            save_ingestion_run("pmu", "error", 0, 0, f"{target}: {exc}")
            print(f"ERREUR {target}: {exc}")
            raise


if __name__ == "__main__":
    main()
