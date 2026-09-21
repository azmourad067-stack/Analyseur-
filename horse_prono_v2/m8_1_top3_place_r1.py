from __future__ import annotations

"""
HorseProno M8.1 - TOP 3 PLACE R1
================================

Objectif : construire exactement 3 chevaux susceptibles de finir dans les
3 premiers, uniquement sur la réunion R1, à partir du modèle M8 figé (#8).

Logique M8.1 :
- PLAT : 70 % rang M8 place_probability + 30 % rang marché
- AUTRES DISCIPLINES : 10 % rang M8 place_probability + 90 % rang marché

Repère historique R1 : médiane de la cote pré-course des chevaux réellement
placés = 7.45. Cette médiane est descriptive uniquement : elle ne filtre pas.

Validation historique sur 60 courses R1 dédupliquées :
- 15.00 % de 3/3 exacts
- 68.33 % avec au moins 2/3
- overlap moyen : 1.783 / 3

Le script ne place aucun pari et n'écrit rien en base.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HORSE_PRONO_ROOT = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(HORSE_PRONO_ROOT)
sys.path.insert(0, HORSE_PRONO_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from app_core.data import normalize_race_input
from app_core.db import get_history_as_of, get_supabase_client
from app_core.features import entity_snapshot_from_history
from app_core.forward_utils import non_runner_numbers
from app_core.model import HorseRacingModel
from app_core.pmu import (
    get_participants,
    get_programme,
    participants_to_df,
    programme_choices,
)

PARIS_TZ = ZoneInfo("Europe/Paris")
MODEL_VERSION_ID = 8
EXPECTED_ARTIFACT_HASH = (
    "78d65eeab7f9264521aabf158e1addeae0566549f3b72b2e4ea10083d7114539"
)
MEETING_NUMBER = 1
R1_PLACED_ODDS_MEDIAN = 7.45

PLAT_PLACE_WEIGHT = 0.70
PLAT_MARKET_WEIGHT = 0.30
OTHER_PLACE_WEIGHT = 0.10
OTHER_MARKET_WEIGHT = 0.90


def _rows(response) -> list[dict]:
    return list(getattr(response, "data", None) or [])


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_discipline(value) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "FLAT": "PLAT",
        "TROT ATTELE": "ATTELE_VOLTE",
        "ATTELE": "ATTELE_VOLTE",
        "AUTOSTART": "ATTELE_AUTOSTART",
        "MONTE": "TROT_MONTE",
    }
    return aliases.get(text, text)


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
        if snap is None or snap.empty or column not in out.columns:
            continue

        key_column = f"{prefix}_key"
        if key_column not in snap.columns:
            continue

        keys = out[column].map(_norm_text)
        tmp = snap.set_index(key_column)

        for metric in ["win_rate", "place_rate", "starts_prior"]:
            target = f"{prefix}_{metric}"
            if target not in tmp.columns:
                continue
            mapped = keys.map(tmp[target])
            if target in out.columns:
                out[target] = mapped.fillna(out[target])
            else:
                out[target] = mapped

    return out


def _load_frozen_m8(client) -> HorseRacingModel:
    rows = _rows(
        client.table("model_versions")
        .select("*")
        .eq("id", MODEL_VERSION_ID)
        .limit(1)
        .execute()
    )
    if not rows:
        raise RuntimeError(f"Modèle #{MODEL_VERSION_ID} introuvable dans Supabase.")

    record = rows[0]
    artifact_hash = str(record.get("artifact_hash") or "")
    if artifact_hash != EXPECTED_ARTIFACT_HASH:
        raise RuntimeError(
            "Le hash du modèle #8 a changé. M8.1 refuse de continuer pour "
            "éviter un mélange de versions.\n"
            f"Attendu : {EXPECTED_ARTIFACT_HASH}\n"
            f"Trouvé : {artifact_hash}"
        )
    return HorseRacingModel.from_stored_record(record)


def _market_probability_from_odds(odds: pd.Series) -> pd.Series:
    odds_num = pd.to_numeric(odds, errors="coerce").clip(lower=1.01)
    inv = 1.0 / odds_num
    total = inv.sum(skipna=True)
    if not np.isfinite(total) or total <= 0:
        n = max(len(odds_num), 1)
        return pd.Series(np.repeat(1 / n, len(odds_num)), index=odds_num.index)
    return inv.fillna(0.0) / total


def build_top3_place(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit le TOP 3 placé M8.1. Plus petit score = meilleur."""
    if predictions.empty:
        return predictions.copy(), predictions.copy()

    df = predictions.copy()
    df["odds_num"] = pd.to_numeric(df.get("odds"), errors="coerce")
    df["place_probability_num"] = pd.to_numeric(
        df.get("place_probability"), errors="coerce"
    ).fillna(0.0)

    if "market_probability" in df.columns:
        market = pd.to_numeric(df["market_probability"], errors="coerce")
    else:
        market = pd.Series(np.nan, index=df.index)

    if market.isna().all():
        market = _market_probability_from_odds(df["odds_num"])
    df["market_probability_num"] = market.fillna(0.0)

    df["m8_place_rank"] = (
        df["place_probability_num"].rank(method="first", ascending=False).astype(int)
    )
    df["market_rank"] = (
        df["market_probability_num"].rank(method="first", ascending=False).astype(int)
    )

    discipline_series = df.get("discipline")
    if discipline_series is not None and not discipline_series.empty:
        discipline = _normalize_discipline(discipline_series.iloc[0])
    else:
        discipline = ""

    if discipline == "PLAT":
        w_place = PLAT_PLACE_WEIGHT
        w_market = PLAT_MARKET_WEIGHT
        selector_mode = "R1_PLAT_70_M8PLACE_30_MARKET"
    else:
        w_place = OTHER_PLACE_WEIGHT
        w_market = OTHER_MARKET_WEIGHT
        selector_mode = "R1_OTHER_10_M8PLACE_90_MARKET"

    df["m81_place_score"] = (
        w_place * df["m8_place_rank"] + w_market * df["market_rank"]
    )
    df["consensus_top3"] = (
        df["m8_place_rank"].le(3) & df["market_rank"].le(3)
    )

    # La médiane est un diagnostic uniquement, jamais un filtre.
    df["odds_vs_r1_placed_median"] = np.where(
        df["odds_num"].isna(),
        "COTE_INCONNUE",
        np.where(
            df["odds_num"] <= R1_PLACED_ODDS_MEDIAN,
            "SOUS_MEDIANE_7_45",
            "AU_DESSUS_MEDIANE_7_45",
        ),
    )
    df["selector_mode"] = selector_mode

    df = df.sort_values(
        [
            "m81_place_score",
            "consensus_top3",
            "place_probability_num",
            "market_probability_num",
            "horse_number",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    df["m81_rank"] = np.arange(1, len(df) + 1)

    def confidence(row: pd.Series) -> str:
        if row["m8_place_rank"] <= 3 and row["market_rank"] <= 3:
            return "FORTE"
        if row["m8_place_rank"] <= 4 or row["market_rank"] <= 4:
            return "MOYENNE"
        return "FAIBLE"

    df["m81_confidence"] = df.apply(confidence, axis=1)
    return df.head(3).copy(), df


def predict_r1_course(
    *,
    model: HorseRacingModel,
    snapshots: dict[str, pd.DataFrame],
    target_date,
    choice: dict,
) -> dict:
    reunion = int(choice["reunion"])
    course = int(choice["course"])
    if reunion != MEETING_NUMBER:
        raise ValueError(f"M8.1 TOP3 PLACE accepte uniquement R{MEETING_NUMBER}.")

    payload = get_participants(target_date, reunion, course)
    race = participants_to_df(payload, target_date, reunion, course)
    if race.empty:
        raise RuntimeError(f"R{reunion}C{course} : aucun participant.")

    race = _apply_choice_metadata(race, choice)
    non_runners = non_runner_numbers(payload)

    if non_runners and "horse_number" in race.columns:
        horse_numbers = pd.to_numeric(race["horse_number"], errors="coerce")
        race = race[~horse_numbers.isin(list(non_runners))].copy()

    if race.empty:
        raise RuntimeError(f"R{reunion}C{course} : aucun partant après retrait des NP.")

    enriched = _enrich_from_snapshot(race, snapshots)
    predictions = model.predict(enriched)
    top3, full = build_top3_place(predictions)

    discipline = _normalize_discipline(
        choice.get("discipline")
        or (
            top3["discipline"].iloc[0]
            if "discipline" in top3.columns and not top3.empty
            else ""
        )
    )

    return {
        "date": str(target_date),
        "reunion": reunion,
        "course": course,
        "hippodrome": choice.get("hippodrome"),
        "discipline": discipline,
        "median_odds_reference": R1_PLACED_ODDS_MEDIAN,
        "consensus_count": int(top3["consensus_top3"].sum()),
        "top3": top3,
        "full": full,
    }


def run_day(target_date, course_filter: int | None = None) -> list[dict]:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase non configuré.")

    model = _load_frozen_m8(client)
    print(f"Chargement historique strictement antérieur au {target_date}...")
    history = get_history_as_of(target_date)
    snapshots = entity_snapshot_from_history(history, target_date)

    programme = get_programme(target_date)
    choices = [
        choice
        for choice in programme_choices(programme)
        if int(choice["reunion"]) == MEETING_NUMBER
        and (course_filter is None or int(choice["course"]) == course_filter)
    ]
    if not choices:
        raise RuntimeError(f"Aucune course R1 trouvée pour {target_date}.")

    results = []
    for choice in choices:
        try:
            results.append(
                predict_r1_course(
                    model=model,
                    snapshots=snapshots,
                    target_date=target_date,
                    choice=choice,
                )
            )
        except Exception as exc:
            print(f"⚠️ R1C{choice.get('course')} ignorée : {exc}")
    return results


def _display_result(result: dict) -> None:
    top3 = result["top3"].copy()
    print()
    print("=" * 78)
    print(
        f"{result['date']} | R{result['reunion']}C{result['course']} | "
        f"{result.get('hippodrome') or 'Hippodrome inconnu'} | "
        f"{result.get('discipline') or 'Discipline inconnue'}"
    )
    print("=" * 78)

    if top3.empty:
        print("Aucun cheval sélectionné.")
        return

    print(
        "TOP 3 PLACE M8.1 : "
        + " - ".join(str(int(x)) for x in top3["horse_number"])
    )
    print(f"Consensus M8/Marché : {result['consensus_count']}/3")
    print(f"Médiane cote placés R1 : {R1_PLACED_ODDS_MEDIAN:.2f}")

    columns = [
        "m81_rank",
        "horse_number",
        "horse_name",
        "odds_num",
        "place_probability_num",
        "m8_place_rank",
        "market_rank",
        "m81_place_score",
        "consensus_top3",
        "m81_confidence",
        "odds_vs_r1_placed_median",
    ]
    display = top3[[c for c in columns if c in top3.columns]].copy()
    display = display.rename(
        columns={
            "m81_rank": "Rang M8.1",
            "horse_number": "N°",
            "horse_name": "Cheval",
            "odds_num": "Cote",
            "place_probability_num": "P(placé) M8",
            "m8_place_rank": "Rang M8 placé",
            "market_rank": "Rang marché",
            "m81_place_score": "Score M8.1",
            "consensus_top3": "Consensus",
            "m81_confidence": "Confiance",
            "odds_vs_r1_placed_median": "Profil cote",
        }
    )
    for col in ["Cote", "P(placé) M8", "Score M8.1"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    print(display.to_string(index=False))


def _flat_export(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        for _, row in result["top3"].iterrows():
            rows.append(
                {
                    "date": result["date"],
                    "reunion": result["reunion"],
                    "course": result["course"],
                    "hippodrome": result.get("hippodrome"),
                    "discipline": result.get("discipline"),
                    "m81_rank": int(row["m81_rank"]),
                    "horse_number": int(row["horse_number"]),
                    "horse_name": row.get("horse_name"),
                    "odds": row.get("odds_num"),
                    "place_probability": row.get("place_probability_num"),
                    "m8_place_rank": int(row["m8_place_rank"]),
                    "market_rank": int(row["market_rank"]),
                    "m81_place_score": float(row["m81_place_score"]),
                    "consensus_top3": bool(row["consensus_top3"]),
                    "confidence": row["m81_confidence"],
                    "odds_profile": row["odds_vs_r1_placed_median"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HorseProno M8.1 - TOP 3 PLACE des courses R1."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date YYYY-MM-DD. Défaut : aujourd'hui Europe/Paris.",
    )
    parser.add_argument(
        "--course",
        type=int,
        default=None,
        help="Numéro de course R1. Sans option : toutes les courses R1.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Chemin facultatif d'export CSV.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Affiche aussi un JSON compact.",
    )
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(PARIS_TZ).date()

    results = run_day(target_date=target_date, course_filter=args.course)
    if not results:
        raise SystemExit("Aucune prédiction M8.1 produite.")

    for result in results:
        _display_result(result)

    export = _flat_export(results)

    if args.csv:
        output_path = Path(args.csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(output_path, index=False)
        print(f"\nCSV enregistré : {output_path}")

    if args.json:
        print(
            "\n"
            + json.dumps(
                export.to_dict(orient="records"),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
