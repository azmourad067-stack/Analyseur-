from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_core.data import normalize_history
from app_core.db import save_ingestion_run, upsert_participants, upsert_races
from app_core.pmu import get_programme, get_participants, participants_to_df, programme_choices


def import_day(target_date: date) -> tuple[int, int]:
    programme = get_programme(target_date)
    choices = programme_choices(programme)
    if not choices:
        save_ingestion_run("pmu_history", "empty", 0, 0, "Programme PMU non interprétable", target_date, target_date)
        print(f"{target_date}: aucune course")
        return 0, 0

    frames = []
    errors = 0
    for choice in choices:
        try:
            payload = get_participants(target_date, choice["reunion"], choice["course"])
            rows = participants_to_df(payload, target_date, choice["reunion"], choice["course"])
            if not rows.empty:
                frames.append(rows)
        except Exception as exc:
            errors += 1
            print(f"{target_date} R{choice['reunion']}C{choice['course']} ignorée: {exc}")

    if not frames:
        save_ingestion_run("pmu_history", "error", 0, 0, "Aucun partant récupéré", target_date, target_date)
        return 0, 0

    df = pd.concat(frames, ignore_index=True)
    # Normalize before writing.
    df = normalize_history(df)
    races = df[["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain",
                "field_size", "reunion", "course_number"]].drop_duplicates("race_id").copy()
    races["source"] = "pmu"
    races["status"] = "finished" if df["finish_position"].notna().any() else "scheduled"
   pcols = [
    "race_id",
    "horse_number",
    "horse_name",
    "jockey",
    "trainer",
    "odds",
    "draw",
    "weight",
    "age",
    "sex",
    "recent_form",
    "finish_position",
]
    participants = df[pcols].copy()
    n_races = upsert_races(races)
    n_participants = upsert_participants(participants)
    status = "success" if errors == 0 else "partial"
    save_ingestion_run("pmu_history", status, len(df), n_races + n_participants,
                       f"{len(choices)} courses; {n_races} courses; {n_participants} partants; {errors} erreurs",
                       target_date, target_date)
    print(f"{target_date}: {n_races} courses / {n_participants} partants / {errors} erreurs")
    return n_races, n_participants


def import_range(start: date, end: date) -> None:
    if end < start:
        raise SystemExit("La date de fin doit être >= à la date de début")
    total_races = total_participants = 0
    d = start
    while d <= end:
        try:
            r, p = import_day(d)
            total_races += r
            total_participants += p
        except Exception as exc:
            print(f"{d}: ERREUR JOURNÉE: {exc}")
            save_ingestion_run("pmu_history", "error", 0, 0, str(exc), d, d)
        d += timedelta(days=1)
    print(f"TOTAL {start} -> {end}: {total_races} courses / {total_participants} partants")


def import_csv(path: str) -> None:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"Fichier introuvable: {source}")
    df = normalize_history(pd.read_csv(source))
    if df.empty:
        raise SystemExit("CSV vide")
    races = df[["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain", "field_size"]].drop_duplicates("race_id").copy()
    races["status"] = "finished"
    participants = df[["race_id", "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight", "recent_form", "finish_position"]].copy()
    n_races = upsert_races(races)
    n_participants = upsert_participants(participants)
    save_ingestion_run("csv", "success", len(df), n_races + n_participants, f"{n_races} courses, {n_participants} partants")
    print(f"CSV: {n_races} courses / {n_participants} partants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import historique HorseProno")
    sub = parser.add_subparsers(dest="mode", required=True)
    api = sub.add_parser("api")
    api.add_argument("--start", required=True)
    api.add_argument("--end", required=True)
    csv = sub.add_parser("csv")
    csv.add_argument("path")
    args = parser.parse_args()
    if args.mode == "api":
        import_range(date.fromisoformat(args.start), date.fromisoformat(args.end))
    else:
        import_csv(args.path)
