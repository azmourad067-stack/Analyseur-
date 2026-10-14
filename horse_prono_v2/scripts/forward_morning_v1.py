from __future__ import annotations

"""HorseProno Morning Forward V1.

Run from the repository root:
    python horse_prono_v2/scripts/forward_morning_v1.py

The script uses the frozen M8 (#8) and Neural (#9) artifacts, selects at most
one horse per eligible race, and records the selection in forward_bets for the
currently active forward_experiments row. It does not train or modify models.
"""

import os
import sys
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HORSE_PRONO_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(HORSE_PRONO_ROOT)
for path in (REPO_ROOT, HORSE_PRONO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from app_core.data import normalize_race_input
from app_core.db import (
    get_history_as_of,
    get_supabase_client,
    upsert_participants,
    upsert_races,
)
from app_core.features import entity_snapshot_from_history
from app_core.forward_utils import (
    course_objects,
    course_start_time,
    non_runner_numbers,
)
from app_core.model import HorseRacingModel
from app_core.pmu import (
    get_participants,
    get_programme,
    participants_to_df,
    programme_choices,
)
from app_core.morning_forward_rules import select_morning_candidate
from neural.forward_capture import load_forward_neural
from neural.predict import predict_dataframe

PARIS_TZ = ZoneInfo("Europe/Paris")
EXPERIMENT_NAME = os.getenv(
    "HORSEPRONO_EXPERIMENT_NAME",
    "horseprono_morning_forward_v1",
)
MODEL8_ID = 8


def _rows(response) -> list[dict]:
    return list(getattr(response, "data", None) or [])


def _norm(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _active_experiment(client) -> dict:
    rows = _rows(
        client.table("forward_experiments")
        .select("*")
        .eq("name", EXPERIMENT_NAME)
        .eq("status", "active")
        .order("started_at")
        .limit(1)
        .execute()
    )
    if not rows:
        raise RuntimeError(
            f"Aucune expérience active nommée '{EXPERIMENT_NAME}' dans "
            "forward_experiments. Crée/active l'expérience dans Supabase avant le run."
        )
    experiment = rows[0]
    if int(experiment.get("model_version_id") or 0) != MODEL8_ID:
        raise RuntimeError(
            f"L'expérience doit utiliser le modèle M8 #{MODEL8_ID}; "
            f"model_version_id trouvé: {experiment.get('model_version_id')}"
        )
    return experiment


def _load_model8(client, experiment: dict) -> HorseRacingModel:
    rows = _rows(
        client.table("model_versions")
        .select("*")
        .eq("id", MODEL8_ID)
        .limit(1)
        .execute()
    )
    if not rows:
        raise RuntimeError("Modèle M8 #8 introuvable dans model_versions.")
    record = rows[0]
    expected = str(experiment.get("artifact_hash") or "")
    actual = str(record.get("artifact_hash") or "")
    if not expected or actual != expected:
        raise RuntimeError(
            "Hash M8 différent de celui figé dans l'expérience. "
            f"Expérience={expected!r}; modèle={actual!r}"
        )
    return HorseRacingModel.from_stored_record(record)


def _apply_choice_metadata(race: pd.DataFrame, choice: dict) -> pd.DataFrame:
    out = race.copy()
    metadata = {
        "discipline": choice.get("discipline"),
        "hippodrome": choice.get("hippodrome"),
        "distance": choice.get("distance"),
        "terrain": choice.get("terrain"),
        "field_size": choice.get("field_size"),
        "reunion": choice.get("reunion"),
        "course_number": choice.get("course"),
    }
    for column, value in metadata.items():
        if value is not None:
            out[column] = value
    out["field_size"] = len(out)
    return out


def _enrich_from_history(race: pd.DataFrame, snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = normalize_race_input(race)
    for entity, column, prefix in (
        ("horse", "horse_name", "horse"),
        ("jockey", "jockey", "jockey"),
        ("trainer", "trainer", "trainer"),
    ):
        snap = snapshots.get(entity)
        if snap is None or snap.empty or column not in out.columns:
            continue
        key_column = f"{prefix}_key"
        if key_column not in snap.columns:
            continue
        tmp = snap.set_index(key_column)
        keys = out[column].map(_norm)
        for metric in ("win_rate", "place_rate", "starts_prior"):
            target = f"{prefix}_{metric}"
            if target not in tmp.columns:
                continue
            mapped = keys.map(tmp[target])
            if target in out.columns:
                out[target] = mapped.fillna(out[target])
            else:
                out[target] = mapped
    return out


def _persist_race(client, race: pd.DataFrame) -> tuple[int, dict[int, dict]]:
    """Upsert race and participants, then return internal IDs keyed by horse number."""
    if race.empty:
        raise ValueError("Course vide; rien à enregistrer.")

    race_columns = [
        "race_id", "race_date", "discipline", "hippodrome", "distance",
        "terrain", "field_size", "reunion", "course_number",
    ]
    races_df = race[race_columns].drop_duplicates("race_id").copy()
    races_df["status"] = "scheduled"
    upsert_races(races_df)
    upsert_participants(race)

    external_id = str(race["race_id"].iloc[0])
    race_rows = _rows(
        client.table("races").select("id,external_id")
        .eq("external_id", external_id).limit(1).execute()
    )
    if not race_rows:
        raise RuntimeError(f"Course non retrouvée dans Supabase: {external_id}")
    internal_race_id = int(race_rows[0]["id"])

    participants = _rows(
        client.table("participants")
        .select("id,horse_number,horse_name")
        .eq("race_id", internal_race_id)
        .execute()
    )
    mapping: dict[int, dict] = {}
    for participant in participants:
        number = pd.to_numeric(
            pd.Series([participant.get("horse_number")]), errors="coerce"
        ).iloc[0]
        if pd.notna(number):
            mapping[int(number)] = participant
    return internal_race_id, mapping


def _existing_participant_ids(client, experiment_id: int) -> set[int]:
    rows = _rows(
        client.table("forward_bets").select("participant_id")
        .eq("experiment_id", experiment_id).execute()
    )
    return {
        int(row["participant_id"])
        for row in rows
        if row.get("participant_id") is not None
    }


def _normalise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    if "horse_number" not in out.columns:
        raise RuntimeError("Prédictions M8 sans colonne horse_number.")
    for col in ("horse_number", "odds", "win_probability", "place_probability"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "win_probability" not in out.columns:
        raise RuntimeError("Prédictions M8 sans win_probability.")
    out["rank_win"] = out["win_probability"].rank(
        method="first", ascending=False
    ).astype("Int64")
    return out


def main() -> None:
    client = get_supabase_client()
    if client is None:
        raise SystemExit("Supabase non configuré: vérifie SUPABASE_URL et SUPABASE_KEY.")

    experiment = _active_experiment(client)
    experiment_id = int(experiment["id"])
    model8 = _load_model8(client, experiment)
    neural_bundle = load_forward_neural(client)

    now_utc = datetime.now(timezone.utc)
    target_date: date = now_utc.astimezone(PARIS_TZ).date()
    print("=" * 72)
    print("HORSEPRONO — MORNING FORWARD V1")
    print(f"Date Paris : {target_date}; capture : {now_utc.astimezone(PARIS_TZ).isoformat()}")
    print(f"Expérience : #{experiment_id} {experiment.get('name')}")
    print("Règles : trot attelé autostart/volté, 8–14 partants, M8 rang gagnant #2,")
    print("         cote snapshot [5, 8[, Neural rang placé <= 3; max 1 sélection/course.")

    history = get_history_as_of(target_date)
    snapshots = entity_snapshot_from_history(history, target_date)
    programme = get_programme(target_date)
    choices = programme_choices(programme)
    raw_courses = course_objects(programme)
    print(f"Programme PMU : {len(choices)} courses détectées.")

    already_recorded = _existing_participant_ids(client, experiment_id)
    inserted = 0
    eligible_races = 0
    skipped = 0

    for choice in choices:
        reunion = int(choice["reunion"])
        course_number = int(choice["course"])
        key = (reunion, course_number)
        start = course_start_time(raw_courses.get(key), target_date)
        if start is None:
            print(f"R{reunion}C{course_number}: ignorée, horaire PMU non interprété.")
            skipped += 1
            continue
        if start <= now_utc:
            continue

        discipline = str(choice.get("discipline") or "")
        field_size = pd.to_numeric(
            pd.Series([choice.get("field_size")]), errors="coerce"
        ).iloc[0]
        if pd.isna(field_size):
            field_size = 0
        # The pure rules module performs the canonical discipline/field-size checks.
        if not (8 <= int(field_size) <= 14):
            continue
        discipline_norm = discipline.lower()
        if "attel" not in discipline_norm or not any(
            token in discipline_norm for token in ("autostart", "volte", "volté")
        ):
            continue

        try:
            payload = get_participants(target_date, reunion, course_number)
            race = participants_to_df(payload, target_date, reunion, course_number)
            if race.empty:
                print(f"R{reunion}C{course_number}: aucun participant, ignorée.")
                skipped += 1
                continue
            race = _apply_choice_metadata(race, choice)
            non_runners = non_runner_numbers(payload)
            numbers = pd.to_numeric(race["horse_number"], errors="coerce")
            race = race[~numbers.isin(list(non_runners))].copy()
            race = race.reset_index(drop=True)
            if not 8 <= len(race) <= 14:
                print(f"R{reunion}C{course_number}: {len(race)} partants actifs, hors filtre.")
                continue

            enriched = _enrich_from_history(race, snapshots)
            m8_predictions = _normalise_predictions(model8.predict(enriched))
            neural_predictions = predict_dataframe(enriched, neural_bundle).copy()
            if "horse_number" not in neural_predictions.columns or "neural_rank" not in neural_predictions.columns:
                raise RuntimeError("Sortie Neural inattendue: horse_number/neural_rank manquants.")
            neural_predictions["horse_number"] = pd.to_numeric(
                neural_predictions["horse_number"], errors="coerce"
            )
            neural_predictions["rank_place"] = pd.to_numeric(
                neural_predictions["neural_rank"], errors="coerce"
            )

            candidate = select_morning_candidate(
                m8_predictions.rename(columns={"rank_win": "rank_win"}),
                neural_predictions,
                discipline=discipline,
                field_size=len(race),
            )
            if candidate is None:
                print(f"R{reunion}C{course_number}: aucun cheval conforme.")
                continue

            eligible_races += 1
            # Snapshot live race/participant records before writing the bet.
            internal_race_id, participant_map = _persist_race(client, race)
            horse_number = int(candidate["horse_number"])
            participant = participant_map.get(horse_number)
            if not participant:
                print(f"R{reunion}C{course_number}: participant N°{horse_number} introuvable en base.")
                skipped += 1
                continue
            participant_id = int(participant["id"])
            if participant_id in already_recorded:
                print(f"R{reunion}C{course_number}: N°{horse_number} déjà enregistré, pas de doublon.")
                continue

            m8_row = m8_predictions[
                pd.to_numeric(m8_predictions["horse_number"], errors="coerce").eq(horse_number)
            ]
            if m8_row.empty:
                raise RuntimeError(f"N°{horse_number}: ligne M8 absente après sélection.")
            m8_row = m8_row.iloc[0]
            odds = float(candidate["odds_snapshot"])
            model_probability = float(m8_row["win_probability"])
            market_probability = 1.0 / odds if odds > 1 else None
            place_probability = pd.to_numeric(
                pd.Series([m8_row.get("place_probability")]), errors="coerce"
            ).iloc[0]
            place_probability = float(place_probability) if pd.notna(place_probability) else None
            edge = model_probability - market_probability if market_probability is not None else None

            bet = {
                "experiment_id": experiment_id,
                "race_id": internal_race_id,
                "participant_id": participant_id,
                "model_version_id": MODEL8_ID,
                "artifact_hash": str(experiment["artifact_hash"]),
                "scheduled_start": start.isoformat(),
                "discipline": discipline,
                "horse_name": str(m8_row.get("horse_name") or participant.get("horse_name") or "INCONNU"),
                "horse_number": horse_number,
                "odds_snapshot": odds,
                "market_probability": market_probability,
                "model_probability": model_probability,
                "place_probability": place_probability,
                "edge": edge,
                "model_rank": int(candidate["model8_rank_win"]),
                "stake": float(candidate["stake"]),
                "status": "pending",
            }
            response = client.table("forward_bets").insert(bet).execute()
            if not _rows(response):
                # Supabase clients may return no rows when representation is disabled.
                inserted += 1
            else:
                inserted += len(_rows(response))
            already_recorded.add(participant_id)
            print(
                f"✅ R{reunion}C{course_number} — N°{horse_number} "
                f"{bet['horse_name']} | cote {odds:.2f} | "
                f"rang M8 gagnant {candidate['model8_rank_win']} | "
                f"rang Neural placé {candidate['neural_rank_place']} | "
                f"départ {start.astimezone(PARIS_TZ).strftime('%H:%M')}"
            )
        except Exception as exc:
            print(f"⚠️ R{reunion}C{course_number}: erreur, course ignorée : {exc}")
            skipped += 1

    print("=" * 72)
    print(f"Sélections conformes : {eligible_races}")
    print(f"Enregistrements ajoutés : {inserted}")
    print(f"Courses ignorées / erreurs : {skipped}")
    print("Fin du Morning Forward V1.")


if __name__ == "__main__":
    main()
