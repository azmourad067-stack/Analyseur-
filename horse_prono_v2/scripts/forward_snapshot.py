from __future__ import annotations

import os
import re
import sys

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

HORSE_PRONO_ROOT = os.path.dirname(
    os.path.dirname(__file__)
)

PROJECT_ROOT = os.path.dirname(
    HORSE_PRONO_ROOT
)

sys.path.insert(
    0,
    HORSE_PRONO_ROOT,
)

sys.path.insert(
    0,
    PROJECT_ROOT,
)

from app_core.data import normalize_race_input
from app_core.db import (
    get_history_as_of,
    get_supabase_client,
    upsert_participants,
    upsert_races,
)
from app_core.features import entity_snapshot_from_history
from app_core.forward_utils import course_objects, course_start_time
from app_core.model import HorseRacingModel
from app_core.pmu import (
    get_participants,
    get_programme,
    participants_to_df,
    programme_choices,
)


PARIS_TZ = ZoneInfo("Europe/Paris")


def _rows(response) -> list[dict]:
    return list(getattr(response, "data", None) or [])


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _active_experiment(client) -> dict | None:
    rows = _rows(
        client.table("forward_experiments")
        .select("*")
        .eq("status", "active")
        .order("started_at")
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _model_record(client, experiment: dict) -> dict:
    model_id = int(experiment["model_version_id"])
    rows = _rows(
        client.table("model_versions")
        .select("*")
        .eq("id", model_id)
        .limit(1)
        .execute()
    )
    if not rows:
        raise RuntimeError(f"Modèle #{model_id} introuvable.")

    record = rows[0]
    expected_hash = str(experiment["artifact_hash"])
    actual_hash = str(record.get("artifact_hash") or "")
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Artifact hash différent du modèle gelé : "
            f"{actual_hash} != {expected_hash}"
        )
    return record


def _effective_count(client, experiment_id: int) -> int:
    rows = _rows(
        client.table("forward_bets")
        .select("id,status")
        .eq("experiment_id", experiment_id)
        .neq("status", "void")
        .execute()
    )
    return len(rows)


def _existing_participant_ids(client, experiment_id: int) -> set[int]:
    rows = _rows(
        client.table("forward_bets")
        .select("participant_id")
        .eq("experiment_id", experiment_id)
        .execute()
    )
    return {
        int(row["participant_id"])
        for row in rows
        if row.get("participant_id") is not None
    }


def _apply_choice_metadata(race: pd.DataFrame, choice: dict) -> pd.DataFrame:
    out = race.copy()

    metadata = {
        "discipline": choice.get("discipline"),
        "hippodrome": choice.get("hippodrome"),
        "distance": choice.get("distance"),
        "terrain": choice.get("terrain"),
        "field_size": choice.get("field_size"),
    }

    for column, value in metadata.items():
        if value is not None:
            out[column] = value

    if (
        "field_size" not in out.columns
        or pd.to_numeric(out["field_size"], errors="coerce").isna().all()
    ):
        out["field_size"] = len(out)

    return out


def _enrich_from_snapshot(
    race: pd.DataFrame,
    snapshots: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = normalize_race_input(race)

    for entity, column, prefix in [
        ("horse", "horse_name", "horse"),
        ("jockey", "jockey", "jockey"),
        ("trainer", "trainer", "trainer"),
    ]:
        snap = snapshots.get(entity)
        if snap is None or snap.empty:
            continue

        keys = out[column].map(_norm_text)
        tmp = snap.set_index(f"{prefix}_key")

        for metric in ["win_rate", "place_rate", "starts_prior"]:
            target = f"{prefix}_{metric}"
            mapped = keys.map(tmp[target])
            if target in out.columns:
                out[target] = mapped.fillna(out[target])
            else:
                out[target] = mapped

    return out


def _persist_live_race(
    client,
    race: pd.DataFrame,
) -> tuple[int, dict[int, dict]]:
    if race.empty:
        raise ValueError("Course vide.")

    races_df = (
        race[
            [
                "race_id",
                "race_date",
                "discipline",
                "hippodrome",
                "distance",
                "terrain",
                "field_size",
                "reunion",
                "course_number",
            ]
        ]
        .drop_duplicates("race_id")
        .copy()
    )
    races_df["status"] = "scheduled"

    upsert_races(races_df)
    upsert_participants(race)

    external_id = str(race["race_id"].iloc[0])

    race_rows = _rows(
        client.table("races")
        .select("id,external_id")
        .eq("external_id", external_id)
        .limit(1)
        .execute()
    )
    if not race_rows:
        raise RuntimeError(f"Course Supabase introuvable : {external_id}")

    internal_race_id = int(race_rows[0]["id"])

    participant_rows = _rows(
        client.table("participants")
        .select("id,horse_number,horse_name")
        .eq("race_id", internal_race_id)
        .execute()
    )

    mapping: dict[int, dict] = {}
    for row in participant_rows:
        number = row.get("horse_number")
        if number is not None:
            mapping[int(number)] = row

    return internal_race_id, mapping


def _candidate_courses(
    programme,
    target_date: date,
    now_utc: datetime,
    min_minutes: int,
    max_minutes: int,
) -> list[tuple[datetime, dict]]:
    choices = programme_choices(programme)
    raw_courses = course_objects(programme)
    candidates = []

    for choice in choices:
        key = (int(choice["reunion"]), int(choice["course"]))
        start = course_start_time(raw_courses.get(key), target_date)
        if start is None:
            continue

        minutes_before = (start - now_utc).total_seconds() / 60.0
        if min_minutes <= minutes_before <= max_minutes:
            candidates.append((start, choice))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]["reunion"],
            item[1]["course"],
        )
    )
    return candidates


def main() -> None:
    client = get_supabase_client()
    if client is None:
        raise SystemExit("Supabase non configuré.")

    experiment = _active_experiment(client)
    if not experiment:
        print("Aucune expérience forward active.")
        return

    experiment_id = int(experiment["id"])
    target_bets = int(experiment["target_bets"])
    current_effective = _effective_count(client, experiment_id)
    remaining = max(0, target_bets - current_effective)

    print("=" * 72)
    print("HORSEPRONO - FORWARD SNAPSHOT")
    print("=" * 72)
    print(f"Expérience : #{experiment_id} {experiment['name']}")
    print(f"Modèle gelé : #{experiment['model_version_id']}")
    print(f"Compteur effectif : {current_effective}/{target_bets}")

    if remaining <= 0:
        print("Objectif déjà rempli en paris non void. Aucun nouveau snapshot.")
        return

    record = _model_record(client, experiment)
    model = HorseRacingModel.from_stored_record(record)

    threshold = float(experiment["edge_threshold"])
    min_minutes = int(experiment["capture_min_minutes"])
    max_minutes = int(experiment["capture_max_minutes"])

    now_utc = datetime.now(timezone.utc)
    target_date = now_utc.astimezone(PARIS_TZ).date()

    print(f"Date PMU : {target_date}")
    print(
        "Fenêtre de capture : "
        f"{min_minutes} à {max_minutes} min avant départ."
    )

    programme = get_programme(target_date)

    all_choices = programme_choices(programme)
    raw_courses = course_objects(programme)
    parsed_starts = 0

    for choice in all_choices:
        key = (
            int(choice["reunion"]),
            int(choice["course"]),
        )
        if course_start_time(
            raw_courses.get(key),
            target_date,
        ) is not None:
            parsed_starts += 1

    print(
        f"Programme PMU : {len(all_choices)} courses, "
        f"{parsed_starts} horaires interprétés."
    )

    if all_choices and parsed_starts == 0:
        raise RuntimeError(
            "Aucun horaire PMU n'a pu être interprété. "
            "Le forward test reste protégé : aucun pari n'est enregistré."
        )

    candidates = _candidate_courses(
        programme,
        target_date,
        now_utc,
        min_minutes,
        max_minutes,
    )

    if not candidates:
        print("Aucune course dans la fenêtre de capture.")
        return

    print(f"Courses candidates : {len(candidates)}")
    print("Chargement historique strictement antérieur à la date...")

    history = get_history_as_of(target_date)
    print(f"Historique chargé : {len(history)} lignes")

    snapshots = entity_snapshot_from_history(history, target_date)
    existing_ids = _existing_participant_ids(client, experiment_id)
    inserts: list[dict] = []

    for start_time, choice in candidates:
        if remaining <= 0:
            break

        reunion = int(choice["reunion"])
        course = int(choice["course"])

        print()
        print(f"R{reunion}C{course} @ {start_time.isoformat()}")

        payload = get_participants(target_date, reunion, course)
        race = participants_to_df(
            payload,
            target_date,
            reunion,
            course,
        )

        if race.empty:
            print("  Aucun participant.")
            continue

        race = _apply_choice_metadata(race, choice)
        internal_race_id, participant_map = _persist_live_race(client, race)

        enriched = _enrich_from_snapshot(race, snapshots)
        predictions = model.predict(enriched)

        predictions["edge"] = (
            pd.to_numeric(predictions["win_probability"], errors="coerce")
            - pd.to_numeric(predictions["market_probability"], errors="coerce")
        )
        predictions["odds_num"] = pd.to_numeric(
            predictions["odds"],
            errors="coerce",
        )

        eligible = predictions[
            predictions["edge"].ge(threshold)
            & predictions["odds_num"].gt(1.0)
        ].copy()

        eligible = eligible.sort_values(
            ["rank", "horse_number"],
            ascending=True,
        )

        print(f"  Edge >= {threshold:.2%} : {len(eligible)} chevaux")

        for _, row in eligible.iterrows():
            if remaining <= 0:
                break

            horse_number_raw = pd.to_numeric(
                pd.Series([row.get("horse_number")]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(horse_number_raw):
                continue

            horse_number = int(horse_number_raw)
            participant = participant_map.get(horse_number)
            if not participant:
                continue

            participant_id = int(participant["id"])
            if participant_id in existing_ids:
                continue

            odds_snapshot = float(row["odds_num"])
            model_probability = float(row["win_probability"])
            market_probability = float(row["market_probability"])
            edge = float(row["edge"])

            insert_row = {
                "experiment_id": experiment_id,
                "race_id": internal_race_id,
                "participant_id": participant_id,
                "model_version_id": int(experiment["model_version_id"]),
                "artifact_hash": str(experiment["artifact_hash"]),
                "scheduled_start": start_time.isoformat(),
                "discipline": str(row.get("discipline") or "INCONNU"),
                "horse_name": str(
                    row.get("horse_name")
                    or participant.get("horse_name")
                    or "INCONNU"
                ),
                "horse_number": horse_number,
                "odds_snapshot": odds_snapshot,
                "market_probability": market_probability,
                "model_probability": model_probability,
                "place_probability": float(row["place_probability"]),
                "edge": edge,
                "model_rank": int(row["rank"]),
                "stake": 1.0,
                "status": "pending",
            }

            inserts.append(insert_row)
            existing_ids.add(participant_id)
            remaining -= 1

            print(
                "  + "
                f"N°{horse_number} {insert_row['horse_name']} | "
                f"cote {odds_snapshot:.2f} | "
                f"P {model_probability:.3f} | "
                f"marché {market_probability:.3f} | "
                f"edge {edge:+.3f}"
            )

    if inserts:
        response = (
            client.table("forward_bets")
            .insert(inserts)
            .execute()
        )
        written = len(_rows(response)) or len(inserts)
    else:
        written = 0

    final_effective = _effective_count(client, experiment_id)

    print()
    print("=" * 72)
    print(f"Nouveaux paris capturés : {written}")
    print(f"Compteur forward : {final_effective}/{target_bets}")
    print("=" * 72)


if __name__ == "__main__":
    main()
