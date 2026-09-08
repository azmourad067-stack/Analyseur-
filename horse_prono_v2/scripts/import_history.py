from __future__ import annotations

import argparse
import pandas as pd

from app_core.data import normalize_history, read_uploaded_csv
from app_core.db import save_ingestion_run, upsert_participants, upsert_races


def main() -> None:
    parser = argparse.ArgumentParser(description="Import historique CSV -> Supabase")
    parser.add_argument("csv_path")
    args = parser.parse_args()

    df = normalize_history(pd.read_csv(args.csv_path, sep=None, engine="python"))
    if df.empty:
        raise SystemExit("CSV vide")

    race_cols = ["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain", "field_size"]
    races = df[race_cols].drop_duplicates("race_id").copy()
    races["source"] = "csv_import"
    upsert_races(races)

    participant_cols = [
        "race_id", "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight",
        "recent_form", "career_runs", "career_wins", "career_places", "finish_position"
    ]
    rows = df[participant_cols].copy()
    written = upsert_participants(rows)
    save_ingestion_run("csv", "success", len(df), written, f"Import {args.csv_path}")
    print(f"{len(df)} lignes lues, {written} partants écrits")


if __name__ == "__main__":
    main()
