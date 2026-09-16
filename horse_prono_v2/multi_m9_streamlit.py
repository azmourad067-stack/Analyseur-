from __future__ import annotations

import hashlib
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
# HORSEPRONO MULTI M9 — V2
#
# Correction principale :
#   - la cohorte de calibration n'est PLUS définie par "tout texte contenant MULTI"
#   - elle est définie par les rapports définitifs PMU réellement exploitables
#   - la cible Top4 provient du rapport Multi PMU, pas de finish_position Supabase
#   - on audite M8 rank_win ET M8 rank_place pour retrouver la référence 5/84,15/84,23/84
#
# Les 84 courses 12→15/09/2026 sont une cohorte de calibration.
# Les courses suivantes sont la vraie validation forward.
# =============================================================================

APP_NAME = "HorseProno Multi M9"
APP_VERSION = "M9-MULTI-V2"

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": "HorsePronoMultiM9/2.0 (+https://streamlit.io)",
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

EXPECTED_COHORT = 84
EXPECTED_M8 = {
    4: 5,
    5: 15,
    6: 23,
}

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

st.title("🏇 HorseProno Multi M9 — V2")
st.caption(
    "Méta-modèle spécialisé Multi. V2 reconstruit d'abord exactement la cohorte "
    "PMU à partir des rapports définitifs, puis calibre le sélecteur."
)


# =============================================================================
# UTILS
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


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@st.cache_resource(show_spinner=False)
def get_client():
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_SERVICE_KEY") or _secret("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Secrets absents : SUPABASE_URL et SUPABASE_SERVICE_KEY."
        )

    return create_client(url, key)


# =============================================================================
# PMU — PROGRAMME
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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_final_reports(
    day_iso: str,
    reunion: int,
    course: int,
) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")

    url = (
        f"{PMU_BASE_URL}/programme/{d}"
        f"/R{reunion}/C{course}/rapports-definitifs"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        params={
            "combinaisonEnTableau": "true",
            "specialisation": "INTERNET",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


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


def _course_number(course_obj: dict) -> int | None:
    return _safe_int(
        course_obj.get("numOrdre")
        or course_obj.get("numCourse")
        or course_obj.get("numOfficiel")
        or course_obj.get("numero")
    )


def _meeting_number(reunion_obj: dict) -> int | None:
    return _safe_int(
        reunion_obj.get("numOfficiel")
        or reunion_obj.get("numReunion")
        or reunion_obj.get("numReunionProgramme")
        or reunion_obj.get("numero")
    )


def _hippodrome(reunion_obj: dict) -> str:
    value = reunion_obj.get("hippodrome")

    if isinstance(value, dict):
        return str(
            value.get("libelleCourt")
            or value.get("libelleLong")
            or value.get("nom")
            or "INCONNU"
        )

    return str(value or "INCONNU")


def _label(course_obj: dict, course: int) -> str:
    return str(
        course_obj.get("libelle")
        or course_obj.get("nom")
        or course_obj.get("libelleCourt")
        or f"Course {course}"
    )


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_dicts(value)


def exact_multi_codes(course_obj: dict) -> list[str]:
    """
    IMPORTANT : on ne cherche plus simplement le mot MULTI.
    On cherche les vrais codes PMU.
    """
    paris = (
        course_obj.get("paris")
        or course_obj.get("typesParis")
        or course_obj.get("bets")
        or []
    )

    found: set[str] = set()

    for obj in _iter_dicts(paris):
        for key in ("codePari", "typePari", "code"):
            value = obj.get(key)
            if value is None:
                continue

            code = str(value).strip().upper()

            if code in {"E_MULTI", "E_MINI_MULTI"}:
                found.add(code)

    return sorted(found)


def programme_multi_candidates(
    start: date,
    end: date,
) -> pd.DataFrame:
    rows: list[dict] = []

    for day in _daterange(start, end):
        programme = get_programme(day.isoformat())

        root = programme.get("programme")
        if not isinstance(root, dict):
            root = programme

        for reunion_obj in _find_reunions(root):
            reunion = _meeting_number(reunion_obj)
            if reunion is None:
                continue

            courses = reunion_obj.get("courses")
            if not isinstance(courses, list):
                continue

            for course_obj in courses:
                if not isinstance(course_obj, dict):
                    continue

                course = _course_number(course_obj)
                if course is None:
                    continue

                codes = exact_multi_codes(course_obj)

                if not codes:
                    continue

                rows.append(
                    {
                        "race_date": day.isoformat(),
                        "meeting_number": reunion,
                        "race_number": course,
                        "hippodrome_pmu": _hippodrome(reunion_obj),
                        "label_pmu": _label(course_obj, course),
                        "multi_codes": " | ".join(codes),
                    }
                )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            ["race_date", "meeting_number", "race_number"]
        )
        .sort_values(
            ["race_date", "meeting_number", "race_number"]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# PMU — RAPPORTS MULTI
# =============================================================================

def iter_bet_blocks(obj: Any):
    if isinstance(obj, dict):
        if "typePari" in obj and "rapports" in obj:
            yield obj

        for value in obj.values():
            yield from iter_bet_blocks(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from iter_bet_blocks(value)


def combination_numbers(value: Any) -> list[int]:
    result: list[int] = []

    if value is None:
        return result

    if isinstance(value, (int, np.integer)):
        return [int(value)]

    if isinstance(value, float):
        if value.is_integer():
            return [int(value)]
        return result

    if isinstance(value, str):
        import re

        return [
            int(x)
            for x in re.findall(r"\d+", value)
        ]

    if isinstance(value, list):
        for item in value:
            result.extend(combination_numbers(item))
        return result

    if isinstance(value, dict):
        preferred = [
            "numPmu",
            "numero",
            "numParticipant",
            "cheval",
        ]

        for key in preferred:
            if key in value:
                numbers = combination_numbers(value[key])
                if numbers:
                    return numbers

        for item in value.values():
            result.extend(combination_numbers(item))

    return result


def extract_multi_target(payload: Any) -> tuple[frozenset[int] | None, str]:
    """
    Extrait les 4 chevaux de la combinaison gagnante à partir
    des rapports définitifs du vrai pari Multi.
    """
    candidates: list[tuple[int, int, int, int]] = []
    matched_types: list[str] = []

    for block in iter_bet_blocks(payload):
        bet_type = str(
            block.get("typePari") or ""
        ).strip().upper()

        if bet_type not in {"E_MULTI", "E_MINI_MULTI"}:
            # Certaines versions ajoutent un libellé autour du code.
            if not (
                "MULTI" in bet_type
                and "COUPLE" not in bet_type
                and "TRIO" not in bet_type
                and "QUINTE" not in bet_type
            ):
                continue

        matched_types.append(bet_type)

        reports = block.get("rapports") or []

        if not isinstance(reports, list):
            continue

        for report in reports:
            if not isinstance(report, dict):
                continue

            numbers = combination_numbers(
                report.get("combinaison")
            )

            unique = list(dict.fromkeys(numbers))

            # Un rapport Multi gagnant doit nous permettre
            # d'identifier les 4 chevaux de l'arrivée utile.
            if len(unique) == 4:
                candidates.append(
                    tuple(sorted(int(x) for x in unique))
                )

    if not candidates:
        return None, "aucune combinaison Multi Top4 exploitable"

    # En cas de répétition du même résultat dans plusieurs sous-rapports,
    # on prend la combinaison la plus fréquente.
    counts: dict[tuple[int, int, int, int], int] = {}

    for combo in candidates:
        counts[combo] = counts.get(combo, 0) + 1

    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    best_combo, best_count = ordered[0]

    # Si plusieurs combinaisons différentes ont exactement la même fréquence,
    # on refuse de fabriquer une cible.
    tied = [
        combo
        for combo, count in ordered
        if count == best_count
    ]

    if len(tied) > 1:
        return None, "rapports Multi ambigus"

    return frozenset(best_combo), ""


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def build_exact_pmu_cohort(
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)

    candidates = programme_multi_candidates(
        start,
        end,
    )

    rows: list[dict] = []

    if candidates.empty:
        return pd.DataFrame()

    for _, row in candidates.iterrows():
        day_iso = str(row["race_date"])
        reunion = int(row["meeting_number"])
        course = int(row["race_number"])

        target = None
        reason = ""
        report_ok = False

        try:
            payload = get_final_reports(
                day_iso,
                reunion,
                course,
            )

            target, reason = extract_multi_target(
                payload
            )

            report_ok = target is not None

        except Exception as exc:
            reason = f"rapport indisponible: {type(exc).__name__}"

        rows.append(
            {
                **row.to_dict(),
                "report_ok": report_ok,
                "target_top4": (
                    "-".join(map(str, sorted(target)))
                    if target is not None
                    else ""
                ),
                "target_set": target,
                "report_reason": reason,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SUPABASE
# =============================================================================

def load_races(start: date, end: date) -> pd.DataFrame:
    client = get_client()
    output: list[dict] = []

    page_size = 1000
    offset = 0

    while True:
        response = (
            client.table("races")
            .select(
                "id,external_id,race_date,meeting_number,race_number,"
                "hippodrome,discipline,distance_m,terrain,field_size,status"
            )
            .gte("race_date", start.isoformat())
            .lte("race_date", end.isoformat())
            .range(offset, offset + page_size - 1)
            .execute()
        )

        batch = _rows(response)
        output.extend(batch)

        if len(batch) < page_size:
            break

        offset += page_size

    if not output:
        return pd.DataFrame()

    df = pd.DataFrame(output)

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)

    return df


def _chunks(values: list[int], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def load_predictions(race_ids: list[int]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()

    client = get_client()
    output: list[dict] = []

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

    model_ids = pd.to_numeric(
        df["model_version_id"],
        errors="coerce",
    )

    good8 = (
        (model_ids == MODEL8_ID)
        & (df["artifact_hash"].astype(str) == MODEL8_HASH)
    )

    good9 = (
        (model_ids == NEURAL_ID)
        & (df["artifact_hash"].astype(str) == NEURAL_HASH)
    )

    df = df[good8 | good9].copy()

    df["_predicted_at"] = pd.to_datetime(
        df["predicted_at"],
        errors="coerce",
        utc=True,
    )

    df = (
        df.sort_values(["_predicted_at", "id"])
        .drop_duplicates(
            ["race_id", "horse_number", "model_version_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def map_pmu_to_db(
    pmu: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    if pmu.empty or races.empty:
        return pd.DataFrame()

    left = pmu.copy()
    right = races.copy()

    left["race_date"] = left["race_date"].astype(str)
    right["race_date"] = right["race_date"].astype(str)

    merged = left.merge(
        right,
        on=["race_date", "meeting_number", "race_number"],
        how="left",
        suffixes=("_pmu", "_db"),
    )

    return merged.rename(
        columns={"id": "race_id"}
    )


def paired_frames(
    predictions: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    result: dict[int, pd.DataFrame] = {}

    if predictions.empty:
        return result

    model_id = pd.to_numeric(
        predictions["model_version_id"],
        errors="coerce",
    )

    p8 = predictions[model_id == MODEL8_ID].copy()
    p9 = predictions[model_id == NEURAL_ID].copy()

    if p8.empty or p9.empty:
        return result

    p8 = p8.rename(
        columns={
            "horse_name": "horse_name_m8",
            "odds_snapshot": "odds",
            "win_probability": "m8_win_probability",
            "place_probability": "m8_place_probability",
            "rank_win": "m8_rank_win",
            "rank_place": "m8_rank_place",
        }
    )

    p9 = p9.rename(
        columns={
            "horse_name": "horse_name_neural",
            "place_probability": "neural_top3_probability",
            "rank_place": "neural_rank",
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
    ]

    keep9 = [
        "race_id",
        "horse_number",
        "horse_name_neural",
        "neural_top3_probability",
        "neural_rank",
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
# FEATURES
# =============================================================================

def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    low = values.min()
    high = values.max()

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    return (
        (values - low) / (high - low)
    ).fillna(0.5).clip(0.0, 1.0)


def _rank_score(
    ranks: pd.Series,
    n: int,
) -> pd.Series:
    values = pd.to_numeric(
        ranks,
        errors="coerce",
    )

    if n <= 1:
        return pd.Series(
            1.0,
            index=ranks.index,
            dtype=float,
        )

    score = (
        1.0
        - (values - 1.0) / float(n - 1)
    )

    return score.fillna(0.5).clip(0.0, 1.0)


def engineer_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    df = raw.copy().reset_index(drop=True)
    n = len(df)

    numeric = [
        "odds",
        "market_probability",
        "m8_win_probability",
        "m8_place_probability",
        "m8_rank_win",
        "m8_rank_place",
        "neural_top3_probability",
        "neural_rank",
    ]

    for column in numeric:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    fallback_market = (
        1.0
        / df["odds"].where(df["odds"] > 1.0)
    )

    df["market_probability"] = (
        df["market_probability"]
        .fillna(fallback_market)
    )

    if df["m8_rank_win"].isna().any():
        fallback = (
            df["m8_win_probability"]
            .rank(method="first", ascending=False)
        )
        df["m8_rank_win"] = (
            df["m8_rank_win"].fillna(fallback)
        )

    if df["m8_rank_place"].isna().any():
        fallback = (
            df["m8_place_probability"]
            .rank(method="first", ascending=False)
        )
        df["m8_rank_place"] = (
            df["m8_rank_place"].fillna(fallback)
        )

    if df["neural_rank"].isna().any():
        fallback = (
            df["neural_top3_probability"]
            .rank(method="first", ascending=False)
        )
        df["neural_rank"] = (
            df["neural_rank"].fillna(fallback)
        )

    df["market_rank"] = (
        df["market_probability"]
        .rank(method="first", ascending=False)
    )

    df["sig_m8_win"] = (
        0.70 * _rank_score(df["m8_rank_win"], n)
        + 0.30 * _minmax(df["m8_win_probability"])
    )

    df["sig_m8_place"] = (
        0.70 * _rank_score(df["m8_rank_place"], n)
        + 0.30 * _minmax(df["m8_place_probability"])
    )

    df["sig_neural_top3"] = (
        0.70 * _rank_score(df["neural_rank"], n)
        + 0.30 * _minmax(df["neural_top3_probability"])
    )

    df["sig_market"] = (
        0.70 * _rank_score(df["market_rank"], n)
        + 0.30 * _minmax(df["market_probability"])
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

    union_strength = np.maximum(
        df["sig_m8_place"].to_numpy(float),
        df["sig_neural_top3"].to_numpy(float),
    )

    df["sig_consensus"] = (
        0.50 * top4_votes
        + 0.20 * top7_votes
        + 0.30 * union_strength
    ).clip(0.0, 1.0)

    return df


def feature_matrix(
    df: pd.DataFrame,
) -> np.ndarray:
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
# BACKTEST
# =============================================================================

def _ordered_indices(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    horse = pd.to_numeric(
        frame["horse_number"],
        errors="coerce",
    ).fillna(999).to_numpy(float)

    m8_place = pd.to_numeric(
        frame["m8_rank_place"],
        errors="coerce",
    ).fillna(999).to_numpy(float)

    return np.lexsort(
        (
            horse,
            m8_place,
            -scores,
        )
    )


def _is_hit(
    target: frozenset[int],
    selection: list[int],
) -> bool:
    return target.issubset(set(selection))


def evaluate_rank_column(
    cache: list[dict],
    column: str,
) -> dict:
    hits = {
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }

    eligible = {
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }

    for item in cache:
        frame = item["frame"].copy()

        frame["_rank"] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        frame = frame.sort_values(
            ["_rank", "horse_number"],
            ascending=[True, True],
        )

        ranked = [
            int(x)
            for x in frame["horse_number"].tolist()
        ]

        for size in (4, 5, 6, 7):
            if len(ranked) < size:
                continue

            eligible[size] += 1

            if _is_hit(
                item["target"],
                ranked[:size],
            ):
                hits[size] += 1

    return {
        **{
            f"hits{size}": hits[size]
            for size in (4, 5, 6, 7)
        },
        **{
            f"eligible{size}": eligible[size]
            for size in (4, 5, 6, 7)
        },
    }


def evaluate_weights(
    cache: list[dict],
    weights: np.ndarray,
) -> dict:
    hits = {
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }

    eligible = {
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }

    overlaps = {
        4: 0,
        5: 0,
        7: 0,
    }

    for item in cache:
        frame = item["frame"]
        scores = item["matrix"] @ weights
        order = _ordered_indices(frame, scores)

        ranked = [
            int(x)
            for x in frame.iloc[order]["horse_number"].tolist()
        ]

        for size in (4, 5, 6, 7):
            if len(ranked) < size:
                continue

            eligible[size] += 1

            if _is_hit(
                item["target"],
                ranked[:size],
            ):
                hits[size] += 1

        for size in (4, 5, 7):
            overlaps[size] += len(
                item["target"].intersection(
                    ranked[:size]
                )
            )

    return {
        **{
            f"hits{size}": hits[size]
            for size in (4, 5, 6, 7)
        },
        **{
            f"eligible{size}": eligible[size]
            for size in (4, 5, 6, 7)
        },
        "overlap4": overlaps[4],
        "overlap5": overlaps[5],
        "overlap7": overlaps[7],
    }


def _compositions(
    total: int,
    parts: int,
):
    if parts == 1:
        yield (total,)
        return

    for first in range(total + 1):
        for rest in _compositions(
            total - first,
            parts - 1,
        ):
            yield (first,) + rest


def optimize(
    cache: list[dict],
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    candidates = list(
        _compositions(
            GRID_UNITS,
            len(SIGNAL_NAMES),
        )
    )

    progress = st.progress(
        0.0,
        text="Optimisation M9 V2…",
    )

    best_weights = None
    best_metrics = None
    best_key = None
    board: list[dict] = []

    total = len(candidates)

    for i, units in enumerate(
        candidates,
        start=1,
    ):
        weights = (
            np.array(units, dtype=float)
            / float(GRID_UNITS)
        )

        metrics = evaluate_weights(
            cache,
            weights,
        )

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

        board.append(
            {
                **{
                    signal: float(weight)
                    for signal, weight
                    in zip(SIGNAL_NAMES, weights)
                },
                **metrics,
                "max_weight": float(weights.max()),
            }
        )

        if (
            i == 1
            or i == total
            or i % 150 == 0
        ):
            progress.progress(
                i / total,
                text=(
                    f"Optimisation M9 V2 : "
                    f"{i:,}/{total:,}"
                ),
            )

    progress.empty()

    if best_weights is None or best_metrics is None:
        raise RuntimeError("Optimisation impossible.")

    leaderboard = (
        pd.DataFrame(board)
        .sort_values(
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
        )
        .head(30)
        .reset_index(drop=True)
    )

    return (
        best_weights,
        best_metrics,
        leaderboard,
    )


# =============================================================================
# CALIBRATION COHORT
# =============================================================================

def build_calibration_package() -> dict:
    pmu_all = build_exact_pmu_cohort(
        CALIBRATION_START.isoformat(),
        CALIBRATION_END.isoformat(),
    )

    if pmu_all.empty:
        raise RuntimeError(
            "Aucun support Multi PMU détecté."
        )

    # La cohorte historique est définie par un VRAI rapport Multi
    # dont la combinaison Top4 est exploitable.
    pmu_cohort = pmu_all[
        pmu_all["report_ok"] == True
    ].copy()

    races = load_races(
        CALIBRATION_START,
        CALIBRATION_END,
    )

    mapped = map_pmu_to_db(
        pmu_cohort,
        races,
    )

    if mapped.empty or "race_id" not in mapped.columns:
        return {
            "pmu_all": pmu_all,
            "pmu_cohort": pmu_cohort,
            "mapped": mapped,
            "cache": [],
            "audit": pd.DataFrame(),
        }

    mapped_ok = mapped[
        mapped["race_id"].notna()
    ].copy()

    mapped_ok["race_id"] = pd.to_numeric(
        mapped_ok["race_id"],
        errors="coerce",
    )

    mapped_ok = mapped_ok[
        mapped_ok["race_id"].notna()
    ].copy()

    mapped_ok["race_id"] = (
        mapped_ok["race_id"].astype(int)
    )

    race_ids = sorted(
        mapped_ok["race_id"].unique().tolist()
    )

    predictions = load_predictions(
        race_ids
    )

    frames = paired_frames(
        predictions
    )

    cache: list[dict] = []
    audit_rows: list[dict] = []

    for _, row in mapped.iterrows():
        race_id = _safe_int(
            row.get("race_id")
        )

        target = row.get("target_set")
        raw = (
            frames.get(race_id)
            if race_id is not None
            else None
        )

        reason = ""

        if race_id is None:
            reason = "course absente Supabase"

        elif not isinstance(target, frozenset) or len(target) != 4:
            reason = "cible PMU Top4 invalide"

        elif raw is None or raw.empty:
            reason = "prédictions M8/Neural absentes"

        elif len(raw) < 4:
            reason = "moins de 4 chevaux appariés"

        else:
            predicted_numbers = set(
                int(x)
                for x in raw["horse_number"].tolist()
            )

            if not target.issubset(predicted_numbers):
                reason = "un cheval du Top4 PMU absent des prédictions"

        usable = reason == ""

        audit_rows.append(
            {
                "race_date": row.get("race_date"),
                "meeting_number": row.get("meeting_number"),
                "race_number": row.get("race_number"),
                "race_id": race_id,
                "multi_codes": row.get("multi_codes"),
                "target_top4": row.get("target_top4"),
                "usable": usable,
                "reason": reason,
                "paired_horses": (
                    len(raw)
                    if raw is not None
                    else 0
                ),
            }
        )

        if not usable:
            continue

        frame = engineer_features(
            raw
        )

        cache.append(
            {
                "race_id": race_id,
                "race_date": str(row.get("race_date")),
                "meeting_number": int(row.get("meeting_number")),
                "race_number": int(row.get("race_number")),
                "hippodrome": (
                    row.get("hippodrome")
                    or row.get("hippodrome_pmu")
                    or "INCONNU"
                ),
                "label_pmu": row.get("label_pmu"),
                "target": target,
                "frame": frame,
                "matrix": feature_matrix(frame),
            }
        )

    return {
        "pmu_all": pmu_all,
        "pmu_cohort": pmu_cohort,
        "mapped": mapped,
        "cache": cache,
        "audit": pd.DataFrame(audit_rows),
    }


def _matches_expected(
    metrics: dict,
) -> bool:
    return all(
        metrics.get(f"hits{size}")
        == EXPECTED_M8[size]
        for size in (4, 5, 6)
    )


def run_calibration() -> dict:
    with st.spinner(
        "Reconstruction de la cohorte exacte via les rapports définitifs PMU…"
    ):
        package = build_calibration_package()

    cache = package["cache"]

    if not cache:
        raise RuntimeError(
            "Aucune course exploitable."
        )

    m8_win = evaluate_rank_column(
        cache,
        "m8_rank_win",
    )

    m8_place = evaluate_rank_column(
        cache,
        "m8_rank_place",
    )

    neural = evaluate_rank_column(
        cache,
        "neural_rank",
    )

    # Référence historique : on ne devine pas le tri.
    # On choisit uniquement celui qui reproduit EXACTEMENT 5/15/23.
    if _matches_expected(m8_win):
        reference_column = "m8_rank_win"
        reference_name = "M8 win"
        reference_metrics = m8_win

    elif _matches_expected(m8_place):
        reference_column = "m8_rank_place"
        reference_name = "M8 place"
        reference_metrics = m8_place

    else:
        reference_column = None
        reference_name = "NON REPRODUITE"
        reference_metrics = None

    weights, optimized, leaderboard = optimize(
        cache
    )

    lock_ok = (
        len(package["pmu_cohort"]) == EXPECTED_COHORT
        and len(cache) == EXPECTED_COHORT
        and reference_column is not None
    )

    fingerprint_payload = {
        "version": APP_VERSION,
        "cohort": len(cache),
        "reference": reference_name,
        "weights": [
            round(float(x), 6)
            for x in weights
        ],
        "model8_hash": MODEL8_HASH,
        "neural_hash": NEURAL_HASH,
    }

    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        **package,
        "m8_win": m8_win,
        "m8_place": m8_place,
        "neural": neural,
        "reference_column": reference_column,
        "reference_name": reference_name,
        "reference_metrics": reference_metrics,
        "weights": weights,
        "optimized": optimized,
        "leaderboard": leaderboard,
        "lock_ok": lock_ok,
        "fingerprint": fingerprint,
    }


# =============================================================================
# FUTURE
# =============================================================================

def rank_m9(
    raw: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    frame = engineer_features(
        raw
    )

    scores = (
        feature_matrix(frame)
        @ weights
    )

    order = _ordered_indices(
        frame,
        scores,
    )

    ranked = (
        frame.iloc[order]
        .copy()
        .reset_index(drop=True)
    )

    ranked["m9_score"] = scores[order]
    ranked["m9_rank"] = np.arange(
        1,
        len(ranked) + 1,
    )

    ranked["horse_name"] = (
        ranked["horse_name_m8"]
        .fillna(ranked["horse_name_neural"])
    )

    return ranked


def future_supports(
    day: date,
) -> pd.DataFrame:
    # Pour le futur, pas de rapport définitif évidemment :
    # on utilise les vrais codes E_MULTI / E_MINI_MULTI du programme.
    return programme_multi_candidates(
        day,
        day,
    )


def future_data(
    day: date,
) -> dict:
    supports = future_supports(day)

    races = load_races(
        day,
        day,
    )

    mapped = map_pmu_to_db(
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

    mapped_ok["race_id"] = pd.to_numeric(
        mapped_ok["race_id"],
        errors="coerce",
    )

    mapped_ok = mapped_ok[
        mapped_ok["race_id"].notna()
    ].copy()

    mapped_ok["race_id"] = (
        mapped_ok["race_id"].astype(int)
    )

    predictions = load_predictions(
        sorted(
            mapped_ok["race_id"]
            .unique()
            .tolist()
        )
    )

    return {
        "supports": supports,
        "mapped": mapped,
        "frames": paired_frames(predictions),
    }


# =============================================================================
# UI
# =============================================================================

with st.sidebar:
    st.header("⚙️ Multi M9 V2")
    st.write(f"**Version :** `{APP_VERSION}`")
    st.write("**Model #8 :** gelé")
    st.write("**Neural #9 :** gelé")
    st.write(
        f"**Calibration :** "
        f"{CALIBRATION_START.strftime('%d/%m/%Y')} → "
        f"{CALIBRATION_END.strftime('%d/%m/%Y')}"
    )
    st.write("**Cohorte attendue :** 84")

    if "m9v2" in st.session_state:
        cal = st.session_state["m9v2"]

        if cal["lock_ok"]:
            st.success(
                "✅ Calibration cohérente et verrouillable"
            )
        else:
            st.error(
                "⚠️ Calibration non verrouillable"
            )


tab_cal, tab_live, tab_method = st.tabs(
    [
        "🧪 Calibration exacte 84",
        "🎯 Pronostics Multi",
        "📐 Méthode",
    ]
)


with tab_cal:
    st.subheader(
        "🧪 Reconstruction et calibration exacte"
    )

    st.info(
        "V2 n'utilise plus finish_position Supabase pour définir la cible. "
        "Elle relit les rapports définitifs PMU et extrait la combinaison Multi gagnante."
    )

    if st.button(
        "🚀 Reconstruire les 84 et calibrer M9 V2",
        type="primary",
        use_container_width=True,
    ):
        try:
            st.session_state["m9v2"] = (
                run_calibration()
            )
        except Exception as exc:
            st.exception(exc)

    cal = st.session_state.get("m9v2")

    if cal is not None:
        pmu_all = cal["pmu_all"]
        pmu_cohort = cal["pmu_cohort"]
        cache = cal["cache"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Candidats exacts E_MULTI / E_MINI_MULTI",
            len(pmu_all),
        )

        c2.metric(
            "Rapports Multi Top4 exploitables",
            len(pmu_cohort),
            delta=(
                "OK = 84"
                if len(pmu_cohort) == EXPECTED_COHORT
                else "attendu 84"
            ),
        )

        c3.metric(
            "Courses exploitables M8 + Neural",
            len(cache),
            delta=(
                "OK = 84"
                if len(cache) == EXPECTED_COHORT
                else "attendu 84"
            ),
        )

        c4.metric(
            "Multi4 M9 optimisé",
            f"{cal['optimized']['hits4']}/"
            f"{cal['optimized']['eligible4']}",
        )

        st.divider()

        comparison = pd.DataFrame(
            [
                {
                    "Stratégie": "M8 win",
                    "Multi4": cal["m8_win"]["hits4"],
                    "Top5 contient les 4": cal["m8_win"]["hits5"],
                    "Top6 contient les 4": cal["m8_win"]["hits6"],
                    "Top7 contient les 4": cal["m8_win"]["hits7"],
                },
                {
                    "Stratégie": "M8 place",
                    "Multi4": cal["m8_place"]["hits4"],
                    "Top5 contient les 4": cal["m8_place"]["hits5"],
                    "Top6 contient les 4": cal["m8_place"]["hits6"],
                    "Top7 contient les 4": cal["m8_place"]["hits7"],
                },
                {
                    "Stratégie": "Neural #9",
                    "Multi4": cal["neural"]["hits4"],
                    "Top5 contient les 4": cal["neural"]["hits5"],
                    "Top6 contient les 4": cal["neural"]["hits6"],
                    "Top7 contient les 4": cal["neural"]["hits7"],
                },
                {
                    "Stratégie": "🏇 Multi M9 V2 optimisé",
                    "Multi4": cal["optimized"]["hits4"],
                    "Top5 contient les 4": cal["optimized"]["hits5"],
                    "Top6 contient les 4": cal["optimized"]["hits6"],
                    "Top7 contient les 4": cal["optimized"]["hits7"],
                },
            ]
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        if cal["reference_column"] is not None:
            st.success(
                "✅ Référence historique reproduite : "
                f"{cal['reference_name']} = "
                f"{EXPECTED_M8[4]}/84 en Top4, "
                f"{EXPECTED_M8[5]}/84 en Top5, "
                f"{EXPECTED_M8[6]}/84 en Top6."
            )
        else:
            st.error(
                "❌ La référence M8 5/84 — 15/84 — 23/84 "
                "n'est toujours pas reproduite. "
                "Ne fige pas les poids : l'audit doit continuer."
            )

        if cal["lock_ok"]:
            gain = (
                cal["optimized"]["hits4"]
                - EXPECTED_M8[4]
            )

            st.success(
                f"🔒 Calibration cohérente : M9 V2 gagne {gain:+d} "
                "Multi4 sur la cohorte de calibration par rapport au 5/84 de référence."
            )
        else:
            st.warning(
                "La configuration reste DIAGNOSTIQUE : "
                "pas de gel tant que cohorte=84 et référence M8 ne sont pas toutes deux validées."
            )

        st.markdown("#### Poids optimaux")

        st.dataframe(
            pd.DataFrame(
                {
                    "Signal": SIGNAL_NAMES,
                    "Poids": cal["weights"],
                    "Poids %": [
                        f"{100*x:.0f}%"
                        for x in cal["weights"]
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "🏆 Top 30 configurations"
        ):
            st.dataframe(
                cal["leaderboard"],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "🔎 Audit course par course"
        ):
            st.dataframe(
                cal["audit"],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "🔎 Candidats PMU + état des rapports"
        ):
            display = pmu_all.drop(
                columns=["target_set"],
                errors="ignore",
            )
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

        config = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "lock_ok": bool(cal["lock_ok"]),
            "cohort_size": len(cache),
            "reference": cal["reference_name"],
            "weights": {
                signal: float(weight)
                for signal, weight
                in zip(
                    SIGNAL_NAMES,
                    cal["weights"],
                )
            },
            "optimized": cal["optimized"],
            "model8_hash": MODEL8_HASH,
            "neural_hash": NEURAL_HASH,
            "fingerprint": cal["fingerprint"],
        }

        st.download_button(
            "⬇️ Télécharger la configuration M9 V2",
            data=json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="horseprono_multi_m9_v2_config.json",
            mime="application/json",
            disabled=not cal["lock_ok"],
            use_container_width=True,
        )


with tab_live:
    st.subheader("🎯 Pronostics Multi")

    cal = st.session_state.get("m9v2")

    if cal is None:
        st.info(
            "Lance d'abord la calibration V2."
        )
    elif not cal["lock_ok"]:
        st.warning(
            "Pronostics futurs désactivés tant que la cohorte historique "
            "et la référence M8 ne sont pas reproduites."
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
                live = future_data(
                    selected_day
                )

                if live["supports"].empty:
                    st.warning(
                        "Aucune course E_MULTI / E_MINI_MULTI détectée."
                    )
                else:
                    for _, row in live["mapped"].iterrows():
                        reunion = int(row["meeting_number"])
                        course = int(row["race_number"])

                        hippodrome = (
                            row.get("hippodrome")
                            or row.get("hippodrome_pmu")
                            or "INCONNU"
                        )

                        st.markdown(
                            f"### R{reunion}C{course} — {hippodrome}"
                        )

                        race_id = _safe_int(
                            row.get("race_id")
                        )

                        if race_id is None:
                            st.warning(
                                "Course pas encore raccordée dans Supabase."
                            )
                            continue

                        raw = live["frames"].get(
                            race_id
                        )

                        if raw is None or raw.empty:
                            st.warning(
                                "Snapshots M8/Neural pas encore disponibles."
                            )
                            continue

                        ranked = rank_m9(
                            raw,
                            cal["weights"],
                        )

                        top4 = (
                            ranked.head(4)["horse_number"]
                            .astype(int)
                            .tolist()
                        )

                        top5 = (
                            ranked.head(min(5, len(ranked)))["horse_number"]
                            .astype(int)
                            .tolist()
                        )

                        top6 = (
                            ranked.head(min(6, len(ranked)))["horse_number"]
                            .astype(int)
                            .tolist()
                        )

                        top7 = (
                            ranked.head(min(7, len(ranked)))["horse_number"]
                            .astype(int)
                            .tolist()
                        )

                        a, b, c, d = st.columns(4)

                        a.metric(
                            "🎯 Multi 4 M9",
                            " - ".join(map(str, top4)),
                        )

                        b.metric(
                            "Top 5",
                            " - ".join(map(str, top5)),
                        )

                        c.metric(
                            "Top 6",
                            " - ".join(map(str, top6)),
                        )

                        d.metric(
                            "Top 7",
                            " - ".join(map(str, top7)),
                        )

                        show = ranked.head(
                            min(10, len(ranked))
                        ).copy()

                        show = show[
                            [
                                "m9_rank",
                                "horse_number",
                                "horse_name",
                                "m9_score",
                                "m8_rank_win",
                                "m8_rank_place",
                                "neural_rank",
                                "market_rank",
                                "odds",
                            ]
                        ]

                        show["m9_score"] = (
                            show["m9_score"].round(4)
                        )

                        st.dataframe(
                            show,
                            use_container_width=True,
                            hide_index=True,
                        )

            except Exception as exc:
                st.exception(exc)


with tab_method:
    st.subheader("📐 Pourquoi cette V2 ?")

    st.markdown(
        """
La première version avait deux défauts visibles dans ton export :

- elle détectait **115** supports au lieu des **84** de notre analyse ;
- elle construisait la cible avec `finish_position` Supabase, alors que
  notre étude historique reposait sur les **rapports définitifs PMU**.

V2 corrige ça en trois étages :

1. détection stricte des codes **`E_MULTI` / `E_MINI_MULTI`** ;
2. validation par **rapport définitif PMU** et extraction de la combinaison
   gagnante de quatre chevaux ;
3. audit des deux classements M8 (`rank_win` et `rank_place`) jusqu'à
   reproduction exacte de **5/84, 15/84, 23/84**.

Les poids M9 ne deviennent téléchargeables et utilisables en forward
que si ces garde-fous sont tous validés.
        """
    )

    st.warning(
        "Même après validation, le score obtenu sur les 84 courses est un score "
        "de calibration. Il faut figer les poids et juger ensuite M9 sur de nouvelles courses."
    )
