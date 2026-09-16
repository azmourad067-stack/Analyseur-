from __future__ import annotations

import hashlib
import itertools
import json
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from supabase import create_client


# =============================================================================
# HORSEPRONO MULTI M9
# Méta-modèle spécialisé Multi, calibré sur les 84 courses Multi du 12-15/09/2026
#
# IMPORTANT
# - Ne modifie PAS le Model #8 gelé.
# - Ne modifie PAS le Neural #9 gelé.
# - "Multi M9" est un méta-modèle Streamlit, pas l'ID Supabase #9.
# - Les 84 courses sont un jeu de CALIBRATION, pas un test indépendant.
# - Les courses futures constituent la vraie validation.
# =============================================================================

APP_NAME = "HorseProno Multi M9"
APP_VERSION = "M9-MULTI-V1"

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": "HorsePronoMultiM9/1.0 (+https://streamlit.io)",
    "Accept": "application/json",
}

MODEL8_ID = 8
NEURAL_ID = 9

MODEL8_HASH = (
    "78d65eeab7f9264521aabf158e1addeae0566549f3b72b2e4ea10083d7114539"
)
NEURAL_HASH = (
    "a2d77a6767e2bfa3258d3fad98173f0bec217c9562a5a6b983651257b8d320b8"
)

CALIBRATION_START = date(2026, 9, 12)
CALIBRATION_END = date(2026, 9, 15)

EXPECTED_MULTI_SUPPORTS = 84

# Sanity checks issus du backtest Multi M8 déjà mesuré.
EXPECTED_M8_HITS = {
    4: 5,
    5: 15,
    6: 23,
}

# 5 signaux => 10 626 combinaisons avec un pas de 0.05.
GRID_STEP = 0.05
GRID_UNITS = int(round(1.0 / GRID_STEP))

SIGNAL_NAMES = [
    "m8_win",
    "m8_place",
    "neural_top3",
    "market",
    "consensus",
]


# =============================================================================
# STREAMLIT
# =============================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏇",
    layout="wide",
)

st.title("🏇 HorseProno Multi M9")
st.caption(
    "Méta-modèle Multi : Model #8 + Neural #9 + marché + consensus. "
    "Calibration figée sur les 84 courses Multi du 12 au 15 septembre 2026."
)


# =============================================================================
# OUTILS GÉNÉRAUX
# =============================================================================

def _secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:
        pass

    value = os.getenv(name)
    return str(value) if value else None


def _rows(response) -> list[dict]:
    return list(getattr(response, "data", None) or [])


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    url = _secret("SUPABASE_URL")
    key = (
        _secret("SUPABASE_SERVICE_KEY")
        or _secret("SUPABASE_KEY")
    )

    if not url or not key:
        raise RuntimeError(
            "Secrets absents : ajoute SUPABASE_URL et SUPABASE_SERVICE_KEY "
            "dans Streamlit Secrets."
        )

    return create_client(url, key)


# =============================================================================
# PMU : PROGRAMME ET DÉTECTION DES COURSES MULTI
# =============================================================================

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_programme(day_iso: str) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")

    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_reunions(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        reunions = obj.get("reunions")
        if isinstance(reunions, list):
            return [x for x in reunions if isinstance(x, dict)]

        for value in obj.values():
            found = _find_reunions(value)
            if found:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = _find_reunions(value)
            if found:
                return found

    return []


def course_objects(programme: Any) -> list[dict]:
    if not isinstance(programme, dict):
        return []

    root = programme.get("programme")
    if not isinstance(root, dict):
        root = programme

    output: list[dict] = []

    for reunion_obj in _find_reunions(root):
        reunion = _as_int(
            reunion_obj.get("numOfficiel")
            or reunion_obj.get("numReunion")
            or reunion_obj.get("numReunionProgramme")
            or reunion_obj.get("numero")
        )
        if reunion is None:
            continue

        hippodrome_obj = reunion_obj.get("hippodrome")
        if isinstance(hippodrome_obj, dict):
            hippodrome = (
                hippodrome_obj.get("libelleCourt")
                or hippodrome_obj.get("libelleLong")
                or hippodrome_obj.get("nom")
                or "INCONNU"
            )
        else:
            hippodrome = str(hippodrome_obj or "INCONNU")

        courses = reunion_obj.get("courses")
        if not isinstance(courses, list):
            continue

        for course_obj in courses:
            if not isinstance(course_obj, dict):
                continue

            course = _as_int(
                course_obj.get("numOrdre")
                or course_obj.get("numCourse")
                or course_obj.get("numOfficiel")
                or course_obj.get("numero")
            )
            if course is None:
                continue

            label = (
                course_obj.get("libelle")
                or course_obj.get("nom")
                or course_obj.get("libelleCourt")
                or f"Course {course}"
            )

            field_size = _as_int(
                course_obj.get("nombreDeclaresPartants")
                or course_obj.get("nombrePartants")
            )

            output.append(
                {
                    "reunion": reunion,
                    "course": course,
                    "hippodrome_pmu": str(hippodrome),
                    "label_pmu": str(label),
                    "field_size_pmu": field_size,
                    "raw": course_obj,
                }
            )

    return output


def _collect_strings(obj: Any) -> list[str]:
    values: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "codePari",
                "typePari",
                "libelle",
                "libelleCourt",
                "libelleLong",
                "code",
                "type",
            } and value is not None:
                values.append(str(value))
            values.extend(_collect_strings(value))

    elif isinstance(obj, list):
        for value in obj:
            values.extend(_collect_strings(value))

    return values


def multi_bet_codes(course_obj: dict) -> list[str]:
    paris = (
        course_obj.get("paris")
        or course_obj.get("typesParis")
        or course_obj.get("bets")
        or []
    )

    # Premier garde-fou : le pari doit réellement être proposé sur la course.
    raw_text = json.dumps(paris, ensure_ascii=False, default=str).upper()
    if "MULTI" not in raw_text:
        return []

    codes = []
    for text in _collect_strings(paris):
        upper = text.upper()
        if "MULTI" in upper:
            codes.append(upper)

    if not codes:
        codes = ["MULTI"]

    return sorted(set(codes))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def discover_multi_supports(start_iso: str, end_iso: str) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)

    rows: list[dict] = []

    for day in _daterange(start, end):
        programme = get_programme(day.isoformat())

        for item in course_objects(programme):
            codes = multi_bet_codes(item["raw"])
            if not codes:
                continue

            rows.append(
                {
                    "race_date": day.isoformat(),
                    "meeting_number": item["reunion"],
                    "race_number": item["course"],
                    "hippodrome_pmu": item["hippodrome_pmu"],
                    "label_pmu": item["label_pmu"],
                    "field_size_pmu": item["field_size_pmu"],
                    "multi_codes": " | ".join(codes),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).drop_duplicates(
        ["race_date", "meeting_number", "race_number"]
    )

    return result.sort_values(
        ["race_date", "meeting_number", "race_number"]
    ).reset_index(drop=True)


# =============================================================================
# SUPABASE : LECTURE PAGINÉE
# =============================================================================

def load_races(start_day: date, end_day: date) -> pd.DataFrame:
    client = get_supabase_client()
    output: list[dict] = []

    page_size = 1000
    start = 0

    while True:
        response = (
            client.table("races")
            .select(
                "id,external_id,race_date,meeting_number,race_number,"
                "hippodrome,discipline,distance_m,terrain,field_size,status"
            )
            .gte("race_date", start_day.isoformat())
            .lte("race_date", end_day.isoformat())
            .range(start, start + page_size - 1)
            .execute()
        )

        batch = _rows(response)
        output.extend(batch)

        if len(batch) < page_size:
            break

        start += page_size

    if not output:
        return pd.DataFrame()

    df = pd.DataFrame(output)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)

    return df


def _chunks(values: list[int], size: int = 30):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def load_participants(race_ids: list[int]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()

    client = get_supabase_client()
    output: list[dict] = []

    for chunk in _chunks(race_ids, 30):
        response = (
            client.table("participants")
            .select(
                "id,race_id,horse_name,horse_number,odds,"
                "finish_position,is_non_runner"
            )
            .in_("race_id", chunk)
            .execute()
        )
        output.extend(_rows(response))

    if not output:
        return pd.DataFrame()

    return pd.DataFrame(output)


def load_predictions(race_ids: list[int]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()

    client = get_supabase_client()
    output: list[dict] = []

    # 20 courses / requête pour rester largement sous les limites.
    for chunk in _chunks(race_ids, 20):
        response = (
            client.table("forward_model_predictions")
            .select(
                "id,experiment_id,race_id,participant_id,model_version_id,"
                "artifact_hash,scheduled_start,predicted_at,discipline,"
                "horse_name,horse_number,odds_snapshot,market_probability,"
                "win_probability,place_probability,rank_win,rank_place"
            )
            .in_("race_id", chunk)
            .in_("model_version_id", [MODEL8_ID, NEURAL_ID])
            .execute()
        )
        output.extend(_rows(response))

    if not output:
        return pd.DataFrame()

    df = pd.DataFrame(output)

    # On ne garde que les deux artefacts gelés attendus.
    good8 = (
        (pd.to_numeric(df["model_version_id"], errors="coerce") == MODEL8_ID)
        & (df["artifact_hash"].astype(str) == MODEL8_HASH)
    )
    good9 = (
        (pd.to_numeric(df["model_version_id"], errors="coerce") == NEURAL_ID)
        & (df["artifact_hash"].astype(str) == NEURAL_HASH)
    )

    df = df[good8 | good9].copy()

    # Si plusieurs snapshots existent, on garde le plus récent par cheval/modèle.
    df["predicted_at_dt"] = pd.to_datetime(
        df["predicted_at"],
        errors="coerce",
        utc=True,
    )

    df = (
        df.sort_values(["predicted_at_dt", "id"])
        .drop_duplicates(
            ["race_id", "horse_number", "model_version_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# CONSTRUCTION DU JEU DE CALIBRATION
# =============================================================================

def map_supports_to_races(
    supports: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    if supports.empty or races.empty:
        return pd.DataFrame()

    left = supports.copy()
    right = races.copy()

    left["race_date"] = left["race_date"].astype(str)
    right["race_date"] = right["race_date"].astype(str)

    mapped = left.merge(
        right,
        on=["race_date", "meeting_number", "race_number"],
        how="left",
        suffixes=("_pmu", "_db"),
    )

    mapped = mapped.rename(columns={"id": "race_id"})

    mapped["race_id"] = pd.to_numeric(
        mapped["race_id"],
        errors="coerce",
    )

    return mapped


def actual_top4_by_race(
    participants: pd.DataFrame,
) -> dict[int, frozenset[int]]:
    targets: dict[int, frozenset[int]] = {}

    if participants.empty:
        return targets

    data = participants.copy()
    data["race_id"] = pd.to_numeric(data["race_id"], errors="coerce")
    data["horse_number"] = pd.to_numeric(
        data["horse_number"],
        errors="coerce",
    )
    data["finish_position"] = pd.to_numeric(
        data["finish_position"],
        errors="coerce",
    )

    if "is_non_runner" in data.columns:
        data = data[data["is_non_runner"].fillna(False) != True].copy()

    data = data[
        data["race_id"].notna()
        & data["horse_number"].notna()
        & data["finish_position"].notna()
        & data["finish_position"].gt(0)
    ].copy()

    for race_id, group in data.groupby("race_id"):
        group = group.sort_values(
            ["finish_position", "horse_number"],
            ascending=[True, True],
        )

        top4 = group.head(4)
        if len(top4) != 4:
            continue

        targets[int(race_id)] = frozenset(
            int(x) for x in top4["horse_number"].tolist()
        )

    return targets


def paired_prediction_frames(
    predictions: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    result: dict[int, pd.DataFrame] = {}

    if predictions.empty:
        return result

    p8 = predictions[
        pd.to_numeric(
            predictions["model_version_id"],
            errors="coerce",
        )
        == MODEL8_ID
    ].copy()

    p9 = predictions[
        pd.to_numeric(
            predictions["model_version_id"],
            errors="coerce",
        )
        == NEURAL_ID
    ].copy()

    if p8.empty or p9.empty:
        return result

    p8 = p8.rename(
        columns={
            "horse_name": "horse_name_m8",
            "odds_snapshot": "odds",
            "market_probability": "market_probability",
            "win_probability": "m8_win_probability",
            "place_probability": "m8_place_probability",
            "rank_win": "m8_rank_win",
            "rank_place": "m8_rank_place",
            "scheduled_start": "scheduled_start_m8",
        }
    )

    p9 = p9.rename(
        columns={
            "horse_name": "horse_name_neural",
            "place_probability": "neural_top3_probability",
            "rank_place": "neural_rank",
            "scheduled_start": "scheduled_start_neural",
        }
    )

    keep8 = [
        "race_id",
        "horse_number",
        "horse_name_m8",
        "discipline",
        "odds",
        "market_probability",
        "m8_win_probability",
        "m8_place_probability",
        "m8_rank_win",
        "m8_rank_place",
        "scheduled_start_m8",
    ]

    keep9 = [
        "race_id",
        "horse_number",
        "horse_name_neural",
        "neural_top3_probability",
        "neural_rank",
        "scheduled_start_neural",
    ]

    merged = p8[keep8].merge(
        p9[keep9],
        on=["race_id", "horse_number"],
        how="inner",
    )

    merged["race_id"] = pd.to_numeric(
        merged["race_id"],
        errors="coerce",
    )
    merged["horse_number"] = pd.to_numeric(
        merged["horse_number"],
        errors="coerce",
    )

    merged = merged[
        merged["race_id"].notna()
        & merged["horse_number"].notna()
    ].copy()

    merged["race_id"] = merged["race_id"].astype(int)
    merged["horse_number"] = merged["horse_number"].astype(int)

    for race_id, group in merged.groupby("race_id"):
        result[int(race_id)] = group.copy().reset_index(drop=True)

    return result


# =============================================================================
# FEATURES M9
# =============================================================================

def _minmax_high_is_good(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)

    low = values.min()
    high = values.max()

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(0.5, index=series.index, dtype=float)

    out = (values - low) / (high - low)
    return out.fillna(0.5).clip(0.0, 1.0)


def _rank_score(rank_series: pd.Series, n: int) -> pd.Series:
    ranks = pd.to_numeric(rank_series, errors="coerce")

    if n <= 1:
        return pd.Series(1.0, index=rank_series.index, dtype=float)

    score = 1.0 - ((ranks - 1.0) / float(n - 1))
    return score.fillna(0.5).clip(0.0, 1.0)


def engineer_race_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy().reset_index(drop=True)
    n = len(df)

    numeric_columns = [
        "odds",
        "market_probability",
        "m8_win_probability",
        "m8_place_probability",
        "m8_rank_win",
        "m8_rank_place",
        "neural_top3_probability",
        "neural_rank",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Fallback marché depuis la cote.
    fallback_market = 1.0 / df["odds"].where(df["odds"] > 1.0)
    df["market_probability"] = df["market_probability"].fillna(fallback_market)

    # Fallback rangs depuis les probabilités si nécessaire.
    if df["m8_rank_win"].isna().any():
        inferred = df["m8_win_probability"].rank(
            method="first",
            ascending=False,
        )
        df["m8_rank_win"] = df["m8_rank_win"].fillna(inferred)

    if df["m8_rank_place"].isna().any():
        inferred = df["m8_place_probability"].rank(
            method="first",
            ascending=False,
        )
        df["m8_rank_place"] = df["m8_rank_place"].fillna(inferred)

    if df["neural_rank"].isna().any():
        inferred = df["neural_top3_probability"].rank(
            method="first",
            ascending=False,
        )
        df["neural_rank"] = df["neural_rank"].fillna(inferred)

    df["market_rank"] = df["market_probability"].rank(
        method="first",
        ascending=False,
    )

    # Chaque signal mélange position relative + intensité de probabilité.
    df["sig_m8_win"] = (
        0.70 * _rank_score(df["m8_rank_win"], n)
        + 0.30 * _minmax_high_is_good(df["m8_win_probability"])
    )

    df["sig_m8_place"] = (
        0.70 * _rank_score(df["m8_rank_place"], n)
        + 0.30 * _minmax_high_is_good(df["m8_place_probability"])
    )

    df["sig_neural_top3"] = (
        0.70 * _rank_score(df["neural_rank"], n)
        + 0.30 * _minmax_high_is_good(df["neural_top3_probability"])
    )

    df["sig_market"] = (
        0.70 * _rank_score(df["market_rank"], n)
        + 0.30 * _minmax_high_is_good(df["market_probability"])
    )

    top4_votes = (
        (df["m8_rank_win"] <= 4).astype(float)
        + (df["m8_rank_place"] <= 4).astype(float)
        + (df["neural_rank"] <= 4).astype(float)
        + (df["market_rank"] <= 4).astype(float)
    ) / 4.0

    top7_votes = (
        (df["m8_rank_win"] <= 7).astype(float)
        + (df["m8_rank_place"] <= 7).astype(float)
        + (df["neural_rank"] <= 7).astype(float)
        + (df["market_rank"] <= 7).astype(float)
    ) / 4.0

    # Le max M8/Neural préserve un "divergent fort" soutenu par l'un des deux.
    union_strength = np.maximum(
        df["sig_m8_win"].to_numpy(float),
        df["sig_neural_top3"].to_numpy(float),
    )

    df["sig_consensus"] = (
        0.50 * top4_votes
        + 0.20 * top7_votes
        + 0.30 * union_strength
    ).clip(0.0, 1.0)

    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[
        [
            "sig_m8_win",
            "sig_m8_place",
            "sig_neural_top3",
            "sig_market",
            "sig_consensus",
        ]
    ].to_numpy(dtype=float)


# =============================================================================
# ÉVALUATION MULTI
# =============================================================================

def _ordered_indices(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    horse_numbers = pd.to_numeric(
        frame["horse_number"],
        errors="coerce",
    ).fillna(999).to_numpy(float)

    m8_ranks = pd.to_numeric(
        frame["m8_rank_win"],
        errors="coerce",
    ).fillna(999).to_numpy(float)

    # np.lexsort : dernière clé = clé primaire.
    return np.lexsort(
        (
            horse_numbers,
            m8_ranks,
            -scores,
        )
    )


def _hit(target: frozenset[int], selected: list[int]) -> bool:
    return target.issubset(set(selected))


def evaluate_weights(
    race_cache: list[dict],
    weights: np.ndarray,
) -> dict:
    hits = {4: 0, 5: 0, 6: 0, 7: 0}
    eligible = {4: 0, 5: 0, 6: 0, 7: 0}
    overlap4 = 0
    overlap5 = 0
    overlap7 = 0

    for item in race_cache:
        frame = item["frame"]
        matrix = item["matrix"]
        target = item["target"]

        scores = matrix @ weights
        order = _ordered_indices(frame, scores)

        ranked_numbers = [
            int(x)
            for x in frame.iloc[order]["horse_number"].tolist()
        ]

        for size in (4, 5, 6, 7):
            if len(ranked_numbers) < size:
                continue

            eligible[size] += 1
            selection = ranked_numbers[:size]

            if _hit(target, selection):
                hits[size] += 1

        top4 = ranked_numbers[: min(4, len(ranked_numbers))]
        top5 = ranked_numbers[: min(5, len(ranked_numbers))]
        top7 = ranked_numbers[: min(7, len(ranked_numbers))]

        overlap4 += len(target.intersection(top4))
        overlap5 += len(target.intersection(top5))
        overlap7 += len(target.intersection(top7))

    return {
        "hits4": hits[4],
        "hits5": hits[5],
        "hits6": hits[6],
        "hits7": hits[7],
        "eligible4": eligible[4],
        "eligible5": eligible[5],
        "eligible6": eligible[6],
        "eligible7": eligible[7],
        "overlap4": overlap4,
        "overlap5": overlap5,
        "overlap7": overlap7,
    }


def evaluate_baseline(
    race_cache: list[dict],
    rank_column: str,
) -> dict:
    hits = {4: 0, 5: 0, 6: 0, 7: 0}
    eligible = {4: 0, 5: 0, 6: 0, 7: 0}
    overlap4 = 0

    for item in race_cache:
        frame = item["frame"].copy()
        target = item["target"]

        frame["_rank"] = pd.to_numeric(
            frame[rank_column],
            errors="coerce",
        )

        frame = frame.sort_values(
            ["_rank", "horse_number"],
            ascending=[True, True],
        )

        ranked = [
            int(x) for x in frame["horse_number"].tolist()
        ]

        for size in (4, 5, 6, 7):
            if len(ranked) < size:
                continue

            eligible[size] += 1
            if _hit(target, ranked[:size]):
                hits[size] += 1

        overlap4 += len(target.intersection(ranked[:4]))

    return {
        "hits4": hits[4],
        "hits5": hits[5],
        "hits6": hits[6],
        "hits7": hits[7],
        "eligible4": eligible[4],
        "eligible5": eligible[5],
        "eligible6": eligible[6],
        "eligible7": eligible[7],
        "overlap4": overlap4,
    }


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return

    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def optimize_weights(
    race_cache: list[dict],
    progress=None,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    candidates = list(
        _compositions(
            GRID_UNITS,
            len(SIGNAL_NAMES),
        )
    )

    best_weights: np.ndarray | None = None
    best_metrics: dict | None = None
    best_key = None
    leaderboard: list[dict] = []

    total = len(candidates)

    for index, units in enumerate(candidates, start=1):
        weights = np.array(
            units,
            dtype=float,
        ) / float(GRID_UNITS)

        metrics = evaluate_weights(
            race_cache,
            weights,
        )

        # Objectif principal : MULTI 4 exact.
        # Puis couverture 5/6/7 et overlaps.
        # A égalité parfaite, on préfère des poids moins extrêmes.
        key = (
            metrics["hits4"],
            metrics["hits5"],
            metrics["hits6"],
            metrics["hits7"],
            metrics["overlap4"],
            metrics["overlap5"],
            metrics["overlap7"],
            -float(weights.max()),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights.copy()
            best_metrics = dict(metrics)

        leaderboard.append(
            {
                **{
                    name: float(weight)
                    for name, weight in zip(SIGNAL_NAMES, weights)
                },
                **metrics,
                "max_weight": float(weights.max()),
            }
        )

        if progress is not None and (
            index == 1
            or index == total
            or index % 150 == 0
        ):
            progress.progress(
                index / total,
                text=(
                    f"Optimisation M9 : {index:,}/{total:,} "
                    "combinaisons de poids"
                ),
            )

    if best_weights is None or best_metrics is None:
        raise RuntimeError("Optimisation impossible.")

    board = pd.DataFrame(leaderboard)

    board = board.sort_values(
        [
            "hits4",
            "hits5",
            "hits6",
            "hits7",
            "overlap4",
            "overlap5",
            "overlap7",
            "max_weight",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    ).head(30)

    return best_weights, best_metrics, board.reset_index(drop=True)


# =============================================================================
# CHARGEMENT / CALIBRATION
# =============================================================================

def build_calibration_data() -> dict:
    supports = discover_multi_supports(
        CALIBRATION_START.isoformat(),
        CALIBRATION_END.isoformat(),
    )

    races = load_races(
        CALIBRATION_START,
        CALIBRATION_END,
    )

    mapped = map_supports_to_races(
        supports,
        races,
    )

    mapped_ok = mapped[
        mapped["race_id"].notna()
    ].copy()

    mapped_ok["race_id"] = mapped_ok["race_id"].astype(int)

    race_ids = sorted(
        mapped_ok["race_id"].unique().tolist()
    )

    participants = load_participants(
        race_ids
    )

    predictions = load_predictions(
        race_ids
    )

    targets = actual_top4_by_race(
        participants
    )

    frames = paired_prediction_frames(
        predictions
    )

    race_cache: list[dict] = []
    audit_rows: list[dict] = []

    for _, support in mapped_ok.iterrows():
        race_id = int(support["race_id"])

        target = targets.get(race_id)
        raw_frame = frames.get(race_id)

        reason = None

        if target is None or len(target) != 4:
            reason = "arrivée Top4 indisponible"

        elif raw_frame is None or raw_frame.empty:
            reason = "prédictions M8/Neural absentes"

        elif len(raw_frame) < 4:
            reason = "moins de 4 prédictions appariées"

        if reason is not None:
            audit_rows.append(
                {
                    "race_date": support["race_date"],
                    "meeting_number": support["meeting_number"],
                    "race_number": support["race_number"],
                    "race_id": race_id,
                    "usable": False,
                    "reason": reason,
                }
            )
            continue

        frame = engineer_race_features(
            raw_frame
        )

        race_cache.append(
            {
                "race_id": race_id,
                "race_date": str(support["race_date"]),
                "meeting_number": int(support["meeting_number"]),
                "race_number": int(support["race_number"]),
                "hippodrome": support.get("hippodrome", support.get("hippodrome_pmu")),
                "label_pmu": support.get("label_pmu"),
                "target": target,
                "frame": frame,
                "matrix": feature_matrix(frame),
            }
        )

        audit_rows.append(
            {
                "race_date": support["race_date"],
                "meeting_number": support["meeting_number"],
                "race_number": support["race_number"],
                "race_id": race_id,
                "usable": True,
                "reason": "",
                "target_top4": "-".join(map(str, sorted(target))),
                "paired_horses": len(frame),
            }
        )

    return {
        "supports": supports,
        "mapped": mapped,
        "race_cache": race_cache,
        "audit": pd.DataFrame(audit_rows),
    }


def calibration_fingerprint(
    weights: np.ndarray,
) -> str:
    payload = {
        "app_version": APP_VERSION,
        "period": [
            CALIBRATION_START.isoformat(),
            CALIBRATION_END.isoformat(),
        ],
        "model8_hash": MODEL8_HASH,
        "neural_hash": NEURAL_HASH,
        "grid_step": GRID_STEP,
        "signals": SIGNAL_NAMES,
        "weights": [
            round(float(x), 6)
            for x in weights
        ],
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def run_calibration() -> dict:
    with st.spinner(
        "Chargement des 84 courses Multi, résultats et prédictions M8/Neural…"
    ):
        package = build_calibration_data()

    supports = package["supports"]
    mapped = package["mapped"]
    race_cache = package["race_cache"]

    baseline_m8 = evaluate_baseline(
        race_cache,
        "m8_rank_win",
    )

    baseline_m8_place = evaluate_baseline(
        race_cache,
        "m8_rank_place",
    )

    baseline_neural = evaluate_baseline(
        race_cache,
        "neural_rank",
    )

    progress = st.progress(
        0.0,
        text="Préparation de l'optimiseur M9…",
    )

    best_weights, best_metrics, leaderboard = optimize_weights(
        race_cache,
        progress=progress,
    )

    progress.empty()

    fingerprint = calibration_fingerprint(
        best_weights
    )

    return {
        **package,
        "baseline_m8": baseline_m8,
        "baseline_m8_place": baseline_m8_place,
        "baseline_neural": baseline_neural,
        "weights": best_weights,
        "metrics": best_metrics,
        "leaderboard": leaderboard,
        "fingerprint": fingerprint,
        "support_count": len(supports),
        "mapped_count": int(mapped["race_id"].notna().sum()) if not mapped.empty else 0,
        "usable_count": len(race_cache),
    }


# =============================================================================
# APPLICATION DES POIDS À UNE COURSE FUTURE
# =============================================================================

def rank_with_m9(
    raw_frame: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    frame = engineer_race_features(
        raw_frame
    )

    scores = feature_matrix(frame) @ weights

    order = _ordered_indices(
        frame,
        scores,
    )

    ranked = frame.iloc[order].copy().reset_index(drop=True)
    ranked["m9_score"] = scores[order]
    ranked["m9_rank"] = np.arange(1, len(ranked) + 1)

    ranked["agreement_top4_votes"] = (
        (ranked["m8_rank_win"] <= 4).astype(int)
        + (ranked["m8_rank_place"] <= 4).astype(int)
        + (ranked["neural_rank"] <= 4).astype(int)
        + (ranked["market_rank"] <= 4).astype(int)
    )

    ranked["horse_name"] = (
        ranked["horse_name_m8"]
        .fillna(ranked["horse_name_neural"])
    )

    return ranked


def load_future_multi(day: date) -> dict:
    supports = discover_multi_supports(
        day.isoformat(),
        day.isoformat(),
    )

    races = load_races(
        day,
        day,
    )

    mapped = map_supports_to_races(
        supports,
        races,
    )

    if mapped.empty or "race_id" not in mapped.columns:
        return {
            "supports": supports,
            "mapped": mapped,
            "frames": {},
        }

    mapped_ok = mapped[
        mapped["race_id"].notna()
    ].copy()

    if mapped_ok.empty:
        return {
            "supports": supports,
            "mapped": mapped,
            "frames": {},
        }

    mapped_ok["race_id"] = mapped_ok["race_id"].astype(int)

    predictions = load_predictions(
        sorted(mapped_ok["race_id"].unique().tolist())
    )

    frames = paired_prediction_frames(
        predictions
    )

    return {
        "supports": supports,
        "mapped": mapped,
        "frames": frames,
    }


# =============================================================================
# UI : SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Multi M9")

    st.write(f"**Version :** `{APP_VERSION}`")
    st.write("**Model #8 :** gelé")
    st.write("**Neural #9 :** gelé")
    st.write(
        f"**Calibration :** "
        f"{CALIBRATION_START.strftime('%d/%m/%Y')} → "
        f"{CALIBRATION_END.strftime('%d/%m/%Y')}"
    )
    st.write(f"**Objectif :** maximiser le Multi 4 / {EXPECTED_MULTI_SUPPORTS}")

    st.divider()

    if "multi_m9_calibration" in st.session_state:
        cal = st.session_state["multi_m9_calibration"]
        st.success(
            f"M9 calibré : {cal['metrics']['hits4']}/"
            f"{cal['metrics']['eligible4']} Multi 4"
        )
        st.caption(
            "Empreinte : "
            + cal["fingerprint"][:12]
            + "…"
        )
    else:
        st.info("Calibration non chargée dans cette session.")


# =============================================================================
# UI : ONGLETS
# =============================================================================

tab_cal, tab_live, tab_method = st.tabs(
    [
        "🧪 Calibration 84",
        "🎯 Pronostics Multi",
        "📐 Méthode",
    ]
)


# =============================================================================
# ONGLET CALIBRATION
# =============================================================================

with tab_cal:
    st.subheader("🧪 Calibration spécialisée Multi")

    st.warning(
        "Les 84 courses du 12 au 15 septembre servent désormais à CALIBRER M9. "
        "Le score obtenu sur ces 84 courses n'est donc pas une validation indépendante. "
        "Les prochaines courses seront le vrai test forward."
    )

    if st.button(
        "🚀 Calibrer / recalibrer Multi M9 sur les 84 courses",
        type="primary",
        use_container_width=True,
    ):
        try:
            st.session_state["multi_m9_calibration"] = run_calibration()
        except Exception as exc:
            st.exception(exc)

    cal = st.session_state.get("multi_m9_calibration")

    if cal is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Supports PMU Multi détectés",
            cal["support_count"],
            delta=(
                "OK"
                if cal["support_count"] == EXPECTED_MULTI_SUPPORTS
                else f"attendu {EXPECTED_MULTI_SUPPORTS}"
            ),
        )

        c2.metric(
            "Supports raccordés Supabase",
            cal["mapped_count"],
        )

        c3.metric(
            "Courses exploitables M8 + Neural",
            cal["usable_count"],
        )

        delta_hits = (
            cal["metrics"]["hits4"]
            - cal["baseline_m8"]["hits4"]
        )

        c4.metric(
            "Multi 4 touchés par M9",
            f"{cal['metrics']['hits4']}/{cal['metrics']['eligible4']}",
            delta=f"{delta_hits:+d} vs M8",
        )

        st.divider()

        sanity_ok = (
            cal["support_count"] == EXPECTED_MULTI_SUPPORTS
            and cal["usable_count"] == EXPECTED_MULTI_SUPPORTS
            and all(
                cal["baseline_m8"][f"hits{size}"] == expected
                for size, expected in EXPECTED_M8_HITS.items()
            )
        )

        if sanity_ok:
            st.success(
                "✅ Sanity check validé : on retrouve bien la référence M8 "
                "5/84 en Multi4, 15/84 en Top5 et 23/84 en Top6."
            )
        else:
            observed = ", ".join(
                f"Top{size}={cal['baseline_m8'][f'hits{size}']}"
                for size in (4, 5, 6)
            )
            st.error(
                "⚠️ La référence historique n'est pas reproduite exactement. "
                f"Observé : {observed}. "
                "Ne considère pas la calibration comme définitive tant que cet écart "
                "n'est pas audité."
            )

        comparison = pd.DataFrame(
            [
                {
                    "Stratégie": "M8 win — référence",
                    "Multi4": cal["baseline_m8"]["hits4"],
                    "Top5 contient les 4": cal["baseline_m8"]["hits5"],
                    "Top6 contient les 4": cal["baseline_m8"]["hits6"],
                    "Top7 contient les 4": cal["baseline_m8"]["hits7"],
                },
                {
                    "Stratégie": "M8 place",
                    "Multi4": cal["baseline_m8_place"]["hits4"],
                    "Top5 contient les 4": cal["baseline_m8_place"]["hits5"],
                    "Top6 contient les 4": cal["baseline_m8_place"]["hits6"],
                    "Top7 contient les 4": cal["baseline_m8_place"]["hits7"],
                },
                {
                    "Stratégie": "Neural #9",
                    "Multi4": cal["baseline_neural"]["hits4"],
                    "Top5 contient les 4": cal["baseline_neural"]["hits5"],
                    "Top6 contient les 4": cal["baseline_neural"]["hits6"],
                    "Top7 contient les 4": cal["baseline_neural"]["hits7"],
                },
                {
                    "Stratégie": "🏇 Multi M9 optimisé",
                    "Multi4": cal["metrics"]["hits4"],
                    "Top5 contient les 4": cal["metrics"]["hits5"],
                    "Top6 contient les 4": cal["metrics"]["hits6"],
                    "Top7 contient les 4": cal["metrics"]["hits7"],
                },
            ]
        )

        st.markdown("#### Comparaison sur les 84 courses")
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Poids optimaux M9")

        weights_df = pd.DataFrame(
            {
                "Signal": SIGNAL_NAMES,
                "Poids": cal["weights"],
                "Poids %": [
                    f"{100*x:.0f}%"
                    for x in cal["weights"]
                ],
            }
        )

        st.dataframe(
            weights_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Recherche exhaustive : pas de {GRID_STEP:.2f}, "
            f"{len(cal['leaderboard'])} meilleures configurations conservées. "
            "L'optimiseur teste 10 626 combinaisons de poids."
        )

        with st.expander("🏆 Top configurations proches de l'optimum"):
            st.dataframe(
                cal["leaderboard"],
                use_container_width=True,
                hide_index=True,
            )

        config_payload = {
            "name": APP_NAME,
            "version": APP_VERSION,
            "calibration_start": CALIBRATION_START.isoformat(),
            "calibration_end": CALIBRATION_END.isoformat(),
            "expected_supports": EXPECTED_MULTI_SUPPORTS,
            "model8_id": MODEL8_ID,
            "model8_hash": MODEL8_HASH,
            "neural_id": NEURAL_ID,
            "neural_hash": NEURAL_HASH,
            "grid_step": GRID_STEP,
            "signals": SIGNAL_NAMES,
            "weights": {
                name: float(value)
                for name, value in zip(
                    SIGNAL_NAMES,
                    cal["weights"],
                )
            },
            "metrics": cal["metrics"],
            "baseline_m8": cal["baseline_m8"],
            "fingerprint": cal["fingerprint"],
        }

        st.download_button(
            "⬇️ Télécharger la configuration M9 figée (.json)",
            data=json.dumps(
                config_payload,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="horseprono_multi_m9_config.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("🔎 Audit des 84 supports"):
            st.dataframe(
                cal["audit"],
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# ONGLET PRONOSTICS
# =============================================================================

with tab_live:
    st.subheader("🎯 Pronostics des courses Multi")

    cal = st.session_state.get("multi_m9_calibration")

    if cal is None:
        st.info(
            "Commence par l'onglet « Calibration 84 » puis clique sur "
            "« Calibrer Multi M9 »."
        )
    else:
        selected_day = st.date_input(
            "Date PMU",
            value=date.today(),
        )

        if st.button(
            "🔎 Analyser les Multi de cette date",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Recherche des courses Multi et des prédictions M8 + Neural…"
                ):
                    live = load_future_multi(
                        selected_day
                    )

                supports = live["supports"]
                mapped = live["mapped"]
                frames = live["frames"]

                if supports.empty:
                    st.warning(
                        "Aucune course E_MULTI / E_MINI_MULTI détectée pour cette date."
                    )
                else:
                    st.success(
                        f"{len(supports)} course(s) Multi détectée(s)."
                    )

                    for _, row in mapped.iterrows():
                        reunion = int(row["meeting_number"])
                        course = int(row["race_number"])
                        label = row.get("label_pmu") or ""
                        hippodrome = (
                            row.get("hippodrome")
                            or row.get("hippodrome_pmu")
                            or "INCONNU"
                        )

                        race_id_value = row.get("race_id")

                        title = (
                            f"R{reunion}C{course} — {hippodrome}"
                            + (f" — {label}" if label else "")
                        )

                        st.markdown(f"### {title}")

                        st.caption(
                            f"Pari PMU : {row.get('multi_codes', 'MULTI')}"
                        )

                        if pd.isna(race_id_value):
                            st.warning(
                                "Course pas encore raccordée dans Supabase. "
                                "Le snapshot forward M8/Neural n'a probablement "
                                "pas encore été capturé."
                            )
                            continue

                        race_id = int(race_id_value)
                        raw_frame = frames.get(race_id)

                        if raw_frame is None or raw_frame.empty:
                            st.warning(
                                "Pas encore de prédictions appariées M8 + Neural "
                                "pour cette course. Le scheduler les capture "
                                "habituellement peu avant le départ."
                            )
                            continue

                        ranked = rank_with_m9(
                            raw_frame,
                            cal["weights"],
                        )

                        if len(ranked) < 4:
                            st.warning(
                                "Moins de quatre partants prédits : course non exploitable."
                            )
                            continue

                        top4 = ranked.head(4)["horse_number"].astype(int).tolist()
                        top5 = ranked.head(min(5, len(ranked)))["horse_number"].astype(int).tolist()
                        top6 = ranked.head(min(6, len(ranked)))["horse_number"].astype(int).tolist()
                        top7 = ranked.head(min(7, len(ranked)))["horse_number"].astype(int).tolist()

                        a, b, c, d = st.columns(4)
                        a.metric("🎯 Multi 4 M9", " - ".join(map(str, top4)))
                        b.metric("Top 5", " - ".join(map(str, top5)))
                        c.metric("Top 6", " - ".join(map(str, top6)))
                        d.metric("Top 7", " - ".join(map(str, top7)))

                        display = ranked.head(min(10, len(ranked))).copy()

                        display = display[
                            [
                                "m9_rank",
                                "horse_number",
                                "horse_name",
                                "m9_score",
                                "agreement_top4_votes",
                                "m8_rank_win",
                                "m8_rank_place",
                                "neural_rank",
                                "market_rank",
                                "odds",
                            ]
                        ].rename(
                            columns={
                                "m9_rank": "Rang M9",
                                "horse_number": "N°",
                                "horse_name": "Cheval",
                                "m9_score": "Score M9",
                                "agreement_top4_votes": "Votes Top4 / 4",
                                "m8_rank_win": "Rang M8 win",
                                "m8_rank_place": "Rang M8 placé",
                                "neural_rank": "Rang Neural",
                                "market_rank": "Rang marché",
                                "odds": "Cote",
                            }
                        )

                        display["Score M9"] = display["Score M9"].round(4)

                        st.dataframe(
                            display,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.caption(
                            "Multi M9 = ranking figé issu de la calibration 84. "
                            "Aucun résultat futur n'est réinjecté automatiquement."
                        )

                        st.divider()

            except Exception as exc:
                st.exception(exc)


# =============================================================================
# ONGLET MÉTHODE
# =============================================================================

with tab_method:
    st.subheader("📐 Comment fonctionne Multi M9 ?")

    st.markdown(
        """
**But du modèle :** augmenter le nombre de courses où les **4 premiers à
l'arrivée** sont contenus dans les **4 premiers chevaux sélectionnés par M9**.

M9 ne remplace pas les modèles existants. Il combine cinq signaux :

1. classement gagnant du **Model #8** ;
2. classement placé du **Model #8** ;
3. probabilité **Top 3 du Neural #9** ;
4. signal du **marché** ;
5. **consensus** entre les modèles et le marché.

L'optimiseur teste **10 626 combinaisons de poids** avec un pas de 5 %.  
Le critère n°1 est le nombre de **Multi 4 touchés sur les 84 courses**.
En cas d'égalité, il privilégie successivement la couverture Top5, Top6,
Top7, puis les overlaps avec l'arrivée.

Le résultat des 84 courses est un **score de calibration**. Une fois les
poids choisis, ils restent figés pour les courses suivantes : les nouvelles
courses servent alors de validation forward.
        """
    )

    st.info(
        "Conseil méthodologique : ne change pas les poids après chaque course future. "
        "On accumule d'abord un vrai échantillon forward, puis on réévalue."
    )
