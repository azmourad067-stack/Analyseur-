from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from app_core.db import save_ingestion_run, upsert_participants, upsert_races
from app_core.pmu import get_programme, participants_to_df, programme_choices, get_participants


def main(target_date: date) -> None:
    programme = get_programme(target_date)
    choices = programme_choices(programme)
    if not choices:
        save_ingestion_run("pmu", "empty", 0, 0, "Programme PMU non interprétable", target_date, target_date)
        print("Aucune course trouvée")
        return
    all_rows = []
    for choice in choices:
        try:
            rows = participants_to_df(get_participants(target_date, choice["reunion"], choice["course"]), target_date, choice["reunion"], choice["course"])
            if not rows.empty:
                all_rows.append(rows)
        except Exception as exc:
            print(f"R{choice['reunion']}C{choice['course']} ignorée: {exc}")
    if not all_rows:
        save_ingestion_run("pmu", "error", 0, 0, "Aucun partant récupéré", target_date, target_date)
        return
    df = pd.concat(all_rows, ignore_index=True)
    races = df[["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain", "field_size", "reunion", "course_number"]].drop_duplicates("race_id").copy()
    races["source"] = "pmu"
    races["status"] = "unknown"
    pcols = ["race_id", "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight", "recent_form", "finish_position"]
    n1 = upsert_races(races)
    n2 = upsert_participants(df[pcols])
    save_ingestion_run("pmu", "success", len(df), n1 + n2, f"{len(choices)} courses; {n1} courses écrites; {n2} partants", target_date, target_date)
    print(f"PMU: {n1} courses, {n2} partants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    main(date.fromisoformat(args.date))
