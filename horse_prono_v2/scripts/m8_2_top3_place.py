from __future__ import annotations

"""
HorseProno M8.2 - TOP 3 PLACE multi-réunions
=============================================

Objectif :
- construire exactement 3 chevaux susceptibles de finir dans les 3 premiers ;
- fonctionner sur toutes les réunions disponibles au programme ;
- conserver une logique R1 spécifique lorsque l'historique le justifie ;
- afficher séparément un outsider placé à surveiller, sans le forcer dans le trio.

Base ML : modèle M8 figé (#8), sans réentraînement.

Backtest exact sur les prédictions M8 stockées et dédupliquées (12-20/09/2026) :
- 340 courses toutes réunions
- M8.2 : 16.18 % de 3/3 exacts
- M8.2 : 62.65 % avec au moins 2/3
- Marché : 13.82 % de 3/3 exacts
- M8 Place pur : 14.41 % de 3/3 exacts

Outsider :
- hors Top 3 M8.2
- cote 8 <= cote < 25
- rang M8 Place <= 8 ET rang marché <= 8
- meilleur rang marché, puis rang M8 Place
- historique : environ 32.4 % de ces candidats ont fini placés sur 287 courses

Ce module ne place aucun pari et n'écrit rien en base.
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

# Repères descriptifs uniquement.
R1_PLACED_ODDS_MEDIAN = 7.45
ALL_PLACED_ODDS_MEDIAN = 7.70

# Pondérations = part du rang M8 Place. Le complément vient du rang marché.
# R1 conserve le réglage spécialisé déjà rétro-testé.
R1_WEIGHTS = {
    "PLAT": 0.70,
    "DEFAULT": 0.10,
}

# Pour les autres réunions, pondérations issues du backtest multi-réunions.
OTHER_MEETING_WEIGHTS = {
    "PLAT": 0.30,
    "ATTELE_AUTOSTART": 0.80,
    "ATTELE_VOLTE": 0.30,
    "TROT_MONTE": 0.30,
    "HAIES": 0.00,
    "STEEPLECHASE": 0.50,
    "CROSS_COUNTRY": 0.00,
    "DEFAULT": 0.20,
}

OUTSIDER_MIN_ODDS = 8.0
OUTSIDER_MAX_ODDS_EXCLUSIVE = 25.0
OUTSIDER_MAX_RANK = 8


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
        "TROT MONTE": "TROT_MONTE",
        "STEEPLE": "STEEPLECHASE",
        "CROSS": "CROSS_COUNTRY",
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
            "Le hash du modèle #8 a changé. M8.2 refuse de continuer pour "
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


def _place_weight(reunion: int, discipline: str) -> tuple[float, str]:
    discipline = _normalize_discipline(discipline)

    if reunion == 1:
        weight = R1_WEIGHTS.get(discipline, R1_WEIGHTS["DEFAULT"])
        mode = f"R1_{discipline or 'DEFAULT'}"
    else:
        weight = OTHER_MEETING_WEIGHTS.get(
            discipline,
            OTHER_MEETING_WEIGHTS["DEFAULT"],
        )
        mode = f"MULTI_{discipline or 'DEFAULT'}"

    return float(weight), mode


def _race_confidence(field_size: int, consensus_count: int) -> str:
    if field_size <= 10 and consensus_count == 3:
        return "TRÈS FORTE"
    if consensus_count == 3:
        return "FORTE"
    if field_size <= 10 and consensus_count >= 2:
        return "FORTE"
    if consensus_count >= 2:
        return "MOYENNE"
    return "PRUDENTE"


def build_top3_place(
    predictions: pd.DataFrame,
    *,
    reunion: int,
    discipline: str,
    field_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retourne (top3, classement complet, outsider placé à surveiller)."""
    if predictions.empty:
        empty = predictions.copy()
        return empty, empty, empty

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

    discipline = _normalize_discipline(discipline)
    w_place, selector_mode = _place_weight(reunion, discipline)
    w_market = 1.0 - w_place

    df["m82_place_weight"] = w_place
    df["m82_market_weight"] = w_market
    df["m82_place_score"] = (
        w_place * df["m8_place_rank"] + w_market * df["market_rank"]
    )
    df["consensus_top3"] = (
        df["m8_place_rank"].le(3) & df["market_rank"].le(3)
    )
    df["selector_mode"] = selector_mode

    median_ref = R1_PLACED_ODDS_MEDIAN if reunion == 1 else ALL_PLACED_ODDS_MEDIAN
    df["odds_median_reference"] = median_ref
    df["odds_profile"] = np.where(
        df["odds_num"].isna(),
        "COTE_INCONNUE",
        np.where(
            df["odds_num"] <= median_ref,
            f"SOUS_MEDIANE_{median_ref:.2f}",
            f"AU_DESSUS_MEDIANE_{median_ref:.2f}",
        ),
    )

    df = df.sort_values(
        [
            "m82_place_score",
            "consensus_top3",
            "place_probability_num",
            "market_probability_num",
            "horse_number",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    df["m82_rank"] = np.arange(1, len(df) + 1)

    def horse_confidence(row: pd.Series) -> str:
        if row["m8_place_rank"] <= 3 and row["market_rank"] <= 3:
            return "FORTE"
        if row["m8_place_rank"] <= 5 and row["market_rank"] <= 5:
            return "MOYENNE"
        return "PRUDENTE"

    df["m82_confidence"] = df.apply(horse_confidence, axis=1)
    top3 = df.head(3).copy()

    # Outsider M8.2 : signal séparé, jamais forcé dans le trio principal.
    outsider_pool = df[
        (df["m82_rank"] > 3)
        & df["odds_num"].ge(OUTSIDER_MIN_ODDS)
        & df["odds_num"].lt(OUTSIDER_MAX_ODDS_EXCLUSIVE)
        & df["m8_place_rank"].le(OUTSIDER_MAX_RANK)
        & df["market_rank"].le(OUTSIDER_MAX_RANK)
    ].copy()

    if outsider_pool.empty:
        outsider = outsider_pool
    else:
        outsider = (
            outsider_pool.sort_values(
                [
                    "market_rank",
                    "m8_place_rank",
                    "place_probability_num",
                    "horse_number",
                ],
                ascending=[True, True, False, True],
            )
            .head(1)
            .copy()
        )
        outsider["outsider_signal"] = np.where(
            (outsider["market_rank"] <= 5) & (outsider["m8_place_rank"] <= 5),
            "FORT",
            "MODÉRÉ",
        )

    actual_field_size = int(field_size or len(df))
    consensus_count = int(top3["consensus_top3"].sum())
    race_confidence = _race_confidence(actual_field_size, consensus_count)

    df["race_confidence"] = race_confidence
    top3["race_confidence"] = race_confidence
    if not outsider.empty:
        outsider["race_confidence"] = race_confidence

    return top3, df, outsider


def predict_course(
    *,
    model: HorseRacingModel,
    snapshots: dict[str, pd.DataFrame],
    target_date,
    choice: dict,
) -> dict:
    reunion = int(choice["reunion"])
    course = int(choice["course"])

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

    actual_field_size = len(race)
    enriched = _enrich_from_snapshot(race, snapshots)
    predictions = model.predict(enriched)

    discipline = _normalize_discipline(
        choice.get("discipline")
        or (
            predictions["discipline"].iloc[0]
            if "discipline" in predictions.columns and not predictions.empty
            else ""
        )
    )

    top3, full, outsider = build_top3_place(
        predictions,
        reunion=reunion,
        discipline=discipline,
        field_size=actual_field_size,
    )

    w_place, selector_mode = _place_weight(reunion, discipline)
    median_ref = R1_PLACED_ODDS_MEDIAN if reunion == 1 else ALL_PLACED_ODDS_MEDIAN

    return {
        "date": str(target_date),
        "reunion": reunion,
        "course": course,
        "hippodrome": choice.get("hippodrome"),
        "discipline": discipline,
        "field_size": actual_field_size,
        "median_odds_reference": median_ref,
        "consensus_count": int(top3["consensus_top3"].sum()),
        "race_confidence": str(top3["race_confidence"].iloc[0]) if not top3.empty else "—",
        "place_weight": w_place,
        "market_weight": 1.0 - w_place,
        "selector_mode": selector_mode,
        "top3": top3,
        "full": full,
        "outsider": outsider,
    }


def run_day(
    target_date,
    meeting_filter: int | None = None,
    course_filter: int | None = None,
) -> list[dict]:
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
        if (meeting_filter is None or int(choice["reunion"]) == meeting_filter)
        and (course_filter is None or int(choice["course"]) == course_filter)
    ]

    if not choices:
        raise RuntimeError(f"Aucune course trouvée pour {target_date}.")

    results = []
    for choice in choices:
        try:
            results.append(
                predict_course(
                    model=model,
                    snapshots=snapshots,
                    target_date=target_date,
                    choice=choice,
                )
            )
        except Exception as exc:
            print(
                f"⚠️ R{choice.get('reunion')}C{choice.get('course')} ignorée : {exc}"
            )
    return results


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
                    "field_size": result.get("field_size"),
                    "m82_rank": int(row["m82_rank"]),
                    "horse_number": int(row["horse_number"]),
                    "horse_name": row.get("horse_name"),
                    "odds": row.get("odds_num"),
                    "place_probability": row.get("place_probability_num"),
                    "m8_place_rank": int(row["m8_place_rank"]),
                    "market_rank": int(row["market_rank"]),
                    "m82_place_score": float(row["m82_place_score"]),
                    "consensus_top3": bool(row["consensus_top3"]),
                    "confidence": row["m82_confidence"],
                    "race_confidence": result.get("race_confidence"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HorseProno M8.2 - TOP 3 PLACE multi-réunions."
    )
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD")
    parser.add_argument("--meeting", type=int, default=None, help="Réunion, ex: 1, 2, 3")
    parser.add_argument("--course", type=int, default=None, help="Numéro de course")
    parser.add_argument("--csv", type=str, default=None, help="Export CSV facultatif")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(PARIS_TZ).date()

    results = run_day(
        target_date=target_date,
        meeting_filter=args.meeting,
        course_filter=args.course,
    )
    if not results:
        raise SystemExit("Aucune prédiction M8.2 produite.")

    export = _flat_export(results)
    print(export.to_string(index=False))

    for result in results:
        outsider = result.get("outsider")
        if outsider is not None and not outsider.empty:
            row = outsider.iloc[0]
            print(
                f"R{result['reunion']}C{result['course']} outsider: "
                f"N°{int(row['horse_number'])} {row.get('horse_name','')} "
                f"cote {float(row['odds_num']):.2f}"
            )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(path, index=False)
        print(f"CSV enregistré : {path}")

    if args.json:
        print(json.dumps(export.to_dict(orient="records"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
