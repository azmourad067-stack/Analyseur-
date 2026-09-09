from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_core.data import normalize_history
from app_core.db import save_ingestion_run, upsert_participants, upsert_races


def main(path: str) -> None:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"Fichier introuvable: {source}")
    df = normalize_history(pd.read_csv(source))
    if df.empty:
        raise SystemExit("CSV vide")

    races = df[["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain", "field_size"]].drop_duplicates("race_id").copy()
    races["source"] = f"csv:{source.name}"
    races["status"] = races["race_id"].map(lambda _: "finished" if False else "unknown")
    # Mark a race finished when at least one finish position is present.
    finished = set(df.loc[df["finish_position"].notna(), "race_id"].astype(str))
    races["status"] = races["race_id"].astype(str).map(lambda x: "finished" if x in finished else "unknown")

    participant_cols = ["race_id", "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight", "recent_form", "finish_position"]
    participants = df[participant_cols].copy()
    n_races = upsert_races(races)
    n_participants = upsert_participants(participants)
    save_ingestion_run("csv", "success", len(df), n_races + n_participants, message=f"{n_races} courses, {n_participants} partants")
    print(f"Import terminé: {n_races} courses / {n_participants} partants")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/import_history.py historique.csv")
    main(sys.argv[1])
