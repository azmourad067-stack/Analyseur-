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

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# HORSEPRONO MULTI M9 — V5 SELECTOR
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
APP_VERSION = "M9-MULTI-V6.1-COMBORANKER-AUDIT"

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


# V4 figée uniquement comme baseline historique.
V4_WEIGHTS = np.array([0.00, 0.60, 0.05, 0.35, 0.00], dtype=float)

# V5 : union des Top6 M8 placé + Neural + marché.
SELECTOR_POOL_K = 6
SELECTOR_FOLDS = 6
SELECTOR_RANDOM_STATE = 42


# =============================================================================
# STREAMLIT
# =============================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏇",
    layout="wide",
)

st.title("🏇 HorseProno Multi M9 — V6.1 ComboRanker Audit")
st.caption(
    "V6.1 Audit : même ComboRanker, avec tie-break OOF strictement indépendant de la cible."
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



def load_participants(race_ids: list[int]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()

    client = get_client()
    output: list[dict] = []

    for chunk in _chunks(race_ids, 30):
        response = (
            client.table("participants")
            .select(
                "id,race_id,horse_name,horse_number,"
                "finish_position,is_non_runner"
            )
            .in_("race_id", chunk)
            .execute()
        )

        output.extend(_rows(response))

    if not output:
        return pd.DataFrame()

    return pd.DataFrame(output)


def historical_finish_targets(
    participants: pd.DataFrame,
) -> dict[int, frozenset[int]]:
    """
    Reproduit exactement la logique de l'analyse historique initiale :
    on trie les arrivées positives et on prend les 4 premiers chevaux,
    y compris en cas d'ex-aequo.

    Exemple réel du 12/09/2026 R4C4 :
        3 -> 1er
        6 -> 1er
        5 -> 3e
        9 -> 4e

    L'ancienne V3 exigeait à tort les positions distinctes {1,2,3,4},
    ce qui rejetait cette course et donnait 83 au lieu de 84.
    """
    result: dict[int, frozenset[int]] = {}

    if participants.empty:
        return result

    data = participants.copy()

    data["race_id"] = pd.to_numeric(
        data["race_id"],
        errors="coerce",
    )
    data["horse_number"] = pd.to_numeric(
        data["horse_number"],
        errors="coerce",
    )
    data["finish_position"] = pd.to_numeric(
        data["finish_position"],
        errors="coerce",
    )

    if "is_non_runner" in data.columns:
        data = data[
            data["is_non_runner"].fillna(False) != True
        ].copy()

    data = data[
        data["race_id"].notna()
        & data["horse_number"].notna()
        & data["finish_position"].notna()
        & data["finish_position"].gt(0)
    ].copy()

    for race_id, group in data.groupby("race_id"):
        ordered = group.sort_values(
            ["finish_position", "horse_number"],
            ascending=[True, True],
        )

        top4 = ordered.head(4)

        if len(top4) != 4:
            continue

        result[int(race_id)] = frozenset(
            int(x)
            for x in top4["horse_number"].tolist()
        )

    return result



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
        text="Optimisation M9 V4…",
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
                    f"Optimisation M9 V4 : "
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
# V5 SELECTOR — POOL TOP6 -> 4 CHEVAUX
# =============================================================================

SELECTOR_NUMERIC_FEATURES = [
    "m8_rank_place",
    "neural_rank",
    "market_rank",
    "m8_rank_win",
    "sig_m8_place",
    "sig_neural_top3",
    "sig_market",
    "sig_m8_win",
    "m8_place_probability",
    "neural_top3_probability",
    "market_probability",
    "m8_win_probability",
    "best_rank",
    "mean_rank",
    "worst_rank",
    "rank_std",
    "rank_range",
    "gap_m8_neural",
    "gap_m8_market",
    "gap_neural_market",
    "votes_top3",
    "votes_top4",
    "votes_top5",
    "votes_top6",
    "only_m8_top6",
    "only_neural_top6",
    "only_market_top6",
    "m8_and_neural_top6",
    "m8_and_market_top6",
    "neural_and_market_top6",
    "all3_top6",
    "odds",
    "log_odds",
    "field_size",
    "pool_size",
    "is_e_multi",
    "is_mini_multi",
]


def candidate_pool(
    item: dict,
    include_target: bool,
    top_k: int = SELECTOR_POOL_K,
) -> pd.DataFrame:
    """
    Construit le pool des chevaux apparaissant dans au moins un TopK :
    M8 placé, Neural, marché.
    """
    frame = engineer_features(
        item["frame"]
    ).copy()

    mask = (
        (pd.to_numeric(frame["m8_rank_place"], errors="coerce") <= top_k)
        | (pd.to_numeric(frame["neural_rank"], errors="coerce") <= top_k)
        | (pd.to_numeric(frame["market_rank"], errors="coerce") <= top_k)
    )

    pool = frame[mask].copy().reset_index(drop=True)

    if pool.empty:
        return pool

    ranks = pool[
        ["m8_rank_place", "neural_rank", "market_rank"]
    ].apply(pd.to_numeric, errors="coerce")

    pool["best_rank"] = ranks.min(axis=1)
    pool["mean_rank"] = ranks.mean(axis=1)
    pool["worst_rank"] = ranks.max(axis=1)
    pool["rank_std"] = ranks.std(axis=1).fillna(0.0)
    pool["rank_range"] = pool["worst_rank"] - pool["best_rank"]

    pool["gap_m8_neural"] = (
        ranks["m8_rank_place"] - ranks["neural_rank"]
    ).abs()
    pool["gap_m8_market"] = (
        ranks["m8_rank_place"] - ranks["market_rank"]
    ).abs()
    pool["gap_neural_market"] = (
        ranks["neural_rank"] - ranks["market_rank"]
    ).abs()

    for k in (3, 4, 5, 6):
        pool[f"votes_top{k}"] = (
            (ranks["m8_rank_place"] <= k).astype(int)
            + (ranks["neural_rank"] <= k).astype(int)
            + (ranks["market_rank"] <= k).astype(int)
        )

    m8_top6 = ranks["m8_rank_place"] <= 6
    n_top6 = ranks["neural_rank"] <= 6
    market_top6 = ranks["market_rank"] <= 6

    pool["only_m8_top6"] = (
        m8_top6 & ~n_top6 & ~market_top6
    ).astype(int)
    pool["only_neural_top6"] = (
        ~m8_top6 & n_top6 & ~market_top6
    ).astype(int)
    pool["only_market_top6"] = (
        ~m8_top6 & ~n_top6 & market_top6
    ).astype(int)

    pool["m8_and_neural_top6"] = (
        m8_top6 & n_top6
    ).astype(int)
    pool["m8_and_market_top6"] = (
        m8_top6 & market_top6
    ).astype(int)
    pool["neural_and_market_top6"] = (
        n_top6 & market_top6
    ).astype(int)
    pool["all3_top6"] = (
        m8_top6 & n_top6 & market_top6
    ).astype(int)

    pool["odds"] = pd.to_numeric(
        pool["odds"],
        errors="coerce",
    )
    pool["log_odds"] = np.log1p(
        pool["odds"].clip(lower=0)
    )

    pool["field_size"] = float(
        _safe_int(item.get("field_size"))
        or len(frame)
    )
    pool["pool_size"] = float(len(pool))

    multi_codes = str(
        item.get("multi_codes") or ""
    ).upper()

    pool["is_e_multi"] = int(
        "E_MULTI" in multi_codes
        and "E_MINI_MULTI" not in multi_codes
    )
    pool["is_mini_multi"] = int(
        "E_MINI_MULTI" in multi_codes
    )

    pool["discipline_selector"] = str(
        item.get("discipline") or "INCONNU"
    ).upper()

    pool["race_id_selector"] = int(
        item.get("race_id") or -1
    )

    pool["horse_name"] = (
        pool["horse_name_m8"]
        .fillna(pool["horse_name_neural"])
    )

    if include_target:
        target = item.get("target") or frozenset()
        pool["is_target"] = (
            pool["horse_number"]
            .astype(int)
            .isin(set(target))
            .astype(int)
        )

    return pool


def selector_dataset(
    cache: list[dict],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for item in cache:
        pool = candidate_pool(
            item,
            include_target=True,
        )

        if pool.empty:
            continue

        parts.append(pool)

    if not parts:
        return pd.DataFrame()

    return pd.concat(
        parts,
        ignore_index=True,
    )


def selector_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    numeric = df.copy()

    for column in SELECTOR_NUMERIC_FEATURES:
        if column not in numeric.columns:
            numeric[column] = np.nan

    x_num = numeric[
        SELECTOR_NUMERIC_FEATURES
    ].apply(pd.to_numeric, errors="coerce")

    discipline = pd.get_dummies(
        numeric["discipline_selector"].fillna("INCONNU"),
        prefix="discipline",
        dtype=float,
    )

    x = pd.concat(
        [
            x_num.reset_index(drop=True),
            discipline.reset_index(drop=True),
        ],
        axis=1,
    )

    if feature_columns is None:
        feature_columns = list(x.columns)
    else:
        for column in feature_columns:
            if column not in x.columns:
                x[column] = 0.0

        x = x.reindex(
            columns=feature_columns,
            fill_value=0.0,
        )

    return x, feature_columns


def selector_specs() -> list[dict]:
    specs: list[dict] = []

    for c in (0.03, 0.10, 0.30, 1.0, 3.0, 10.0):
        specs.append(
            {
                "name": f"Logistic C={c}",
                "kind": "logistic",
                "C": c,
            }
        )

    specs.extend(
        [
            {
                "name": "RF depth=2 leaf=6",
                "kind": "rf",
                "max_depth": 2,
                "min_samples_leaf": 6,
            },
            {
                "name": "RF depth=3 leaf=5",
                "kind": "rf",
                "max_depth": 3,
                "min_samples_leaf": 5,
            },
            {
                "name": "RF depth=4 leaf=3",
                "kind": "rf",
                "max_depth": 4,
                "min_samples_leaf": 3,
            },
            {
                "name": "RF depth=6 leaf=2",
                "kind": "rf",
                "max_depth": 6,
                "min_samples_leaf": 2,
            },
            {
                "name": "RF full leaf=1",
                "kind": "rf",
                "max_depth": None,
                "min_samples_leaf": 1,
            },
            {
                "name": "ExtraTrees depth=3 leaf=5",
                "kind": "extra",
                "max_depth": 3,
                "min_samples_leaf": 5,
            },
            {
                "name": "ExtraTrees depth=5 leaf=3",
                "kind": "extra",
                "max_depth": 5,
                "min_samples_leaf": 3,
            },
            {
                "name": "ExtraTrees depth=7 leaf=2",
                "kind": "extra",
                "max_depth": 7,
                "min_samples_leaf": 2,
            },
            {
                "name": "ExtraTrees full leaf=2",
                "kind": "extra",
                "max_depth": None,
                "min_samples_leaf": 2,
            },
            {
                "name": "ExtraTrees full leaf=1",
                "kind": "extra",
                "max_depth": None,
                "min_samples_leaf": 1,
            },
            {
                "name": "GB 80x0.03 depth1",
                "kind": "gb",
                "n_estimators": 80,
                "learning_rate": 0.03,
                "max_depth": 1,
            },
            {
                "name": "GB 120x0.05 depth1",
                "kind": "gb",
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 1,
            },
            {
                "name": "GB 120x0.05 depth2",
                "kind": "gb",
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 2,
            },
            {
                "name": "GB 160x0.03 depth2",
                "kind": "gb",
                "n_estimators": 160,
                "learning_rate": 0.03,
                "max_depth": 2,
            },
        ]
    )

    return specs


def build_selector_model(
    spec: dict,
):
    kind = spec["kind"]

    if kind == "logistic":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=float(spec["C"]),
                        max_iter=2500,
                        random_state=SELECTOR_RANDOM_STATE,
                    ),
                ),
            ]
        )

    if kind == "rf":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=spec["max_depth"],
                        min_samples_leaf=int(spec["min_samples_leaf"]),
                        max_features="sqrt",
                        random_state=SELECTOR_RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if kind == "extra":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=350,
                        max_depth=spec["max_depth"],
                        min_samples_leaf=int(spec["min_samples_leaf"]),
                        max_features="sqrt",
                        random_state=SELECTOR_RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if kind == "gb":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=int(spec["n_estimators"]),
                        learning_rate=float(spec["learning_rate"]),
                        max_depth=int(spec["max_depth"]),
                        random_state=SELECTOR_RANDOM_STATE,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Type de modèle inconnu : {kind}"
    )


def positive_probability(
    model,
    x: pd.DataFrame,
) -> np.ndarray:
    proba = model.predict_proba(x)

    if proba.shape[1] == 1:
        # Cas théorique d'une classe unique.
        return np.zeros(len(x), dtype=float)

    classes = list(
        getattr(model, "classes_", [0, 1])
    )

    if hasattr(model, "named_steps"):
        classes = list(
            model.named_steps["model"].classes_
        )

    positive_index = (
        classes.index(1)
        if 1 in classes
        else -1
    )

    return proba[:, positive_index]


def selector_metrics(
    meta: pd.DataFrame,
    scores: np.ndarray,
) -> dict:
    data = meta[
        [
            "race_id_selector",
            "horse_number",
            "m8_rank_place",
            "is_target",
        ]
    ].copy()

    data["score"] = scores

    exact = 0
    three_plus = 0
    total_target_hits = 0
    oracle_pool = 0
    races = 0

    detail_rows: list[dict] = []

    for race_id, group in data.groupby(
        "race_id_selector",
        sort=False,
    ):
        races += 1

        group = group.sort_values(
            ["score", "m8_rank_place", "horse_number"],
            ascending=[False, True, True],
        )

        selected = group.head(4)
        selected_hits = int(
            selected["is_target"].sum()
        )
        target_in_pool = int(
            group["is_target"].sum()
        )

        total_target_hits += selected_hits
        three_plus += int(
            selected_hits >= 3
        )
        exact += int(
            selected_hits == 4
        )
        oracle_pool += int(
            target_in_pool == 4
        )

        detail_rows.append(
            {
                "race_id": int(race_id),
                "selected_hits": selected_hits,
                "target_in_pool": target_in_pool,
                "exact_multi4": selected_hits == 4,
                "selection": "-".join(
                    str(int(x))
                    for x in selected["horse_number"].tolist()
                ),
            }
        )

    return {
        "races": races,
        "exact_hits": exact,
        "three_plus": three_plus,
        "total_target_hits": total_target_hits,
        "avg_target_hits": (
            total_target_hits / races
            if races
            else 0.0
        ),
        "oracle_pool_hits": oracle_pool,
        "detail": pd.DataFrame(detail_rows),
    }


def oracle_pool_table(
    cache: list[dict],
) -> pd.DataFrame:
    rows = []

    for k in (4, 5, 6, 7):
        hits = 0
        sizes = []

        for item in cache:
            frame = engineer_features(
                item["frame"]
            )

            mask = (
                (frame["m8_rank_place"] <= k)
                | (frame["neural_rank"] <= k)
                | (frame["market_rank"] <= k)
            )

            numbers = set(
                frame.loc[
                    mask,
                    "horse_number",
                ].astype(int)
            )

            sizes.append(
                len(numbers)
            )

            if set(item["target"]).issubset(
                numbers
            ):
                hits += 1

        rows.append(
            {
                "Pool": f"Union Top{k}",
                "Oracle Multi4": hits,
                "Sur 84": f"{hits}/{len(cache)}",
                "Taille moyenne pool": round(
                    float(np.mean(sizes)),
                    2,
                ),
                "Taille max": int(max(sizes)),
            }
        )

    return pd.DataFrame(rows)


def train_selector_search(
    cache: list[dict],
) -> dict:
    data = selector_dataset(
        cache
    )

    if data.empty:
        raise RuntimeError(
            "Dataset Selector vide."
        )

    x, feature_columns = selector_matrix(
        data
    )

    y = data["is_target"].astype(int).to_numpy()
    groups = data[
        "race_id_selector"
    ].astype(int).to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(
        SELECTOR_FOLDS,
        len(unique_groups),
    )

    if n_splits < 2:
        raise RuntimeError(
            "Pas assez de courses pour la validation groupée."
        )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    specs = selector_specs()
    progress = st.progress(
        0.0,
        text="V5 Selector : recherche des modèles…",
    )

    board: list[dict] = []
    best_oof_key = None
    best_oof_bundle = None
    best_train_key = None
    best_train_bundle = None

    for i, spec in enumerate(
        specs,
        start=1,
    ):
        oof_scores = np.full(
            len(data),
            np.nan,
            dtype=float,
        )

        for train_idx, valid_idx in splitter.split(
            x,
            y,
            groups,
        ):
            model = build_selector_model(
                spec
            )
            model.fit(
                x.iloc[train_idx],
                y[train_idx],
            )
            oof_scores[valid_idx] = (
                positive_probability(
                    model,
                    x.iloc[valid_idx],
                )
            )

        if np.isnan(oof_scores).any():
            raise RuntimeError(
                "Scores OOF incomplets."
            )

        oof_metrics = selector_metrics(
            data,
            oof_scores,
        )

        full_model = build_selector_model(
            spec
        )
        full_model.fit(
            x,
            y,
        )

        train_scores = positive_probability(
            full_model,
            x,
        )

        train_metrics = selector_metrics(
            data,
            train_scores,
        )

        row = {
            "Modèle": spec["name"],
            "Type": spec["kind"],
            "OOF Multi4": oof_metrics["exact_hits"],
            "OOF ≥3/4": oof_metrics["three_plus"],
            "OOF chevaux trouvés": oof_metrics["total_target_hits"],
            "OOF moy./4": round(
                oof_metrics["avg_target_hits"],
                4,
            ),
            "Train Multi4": train_metrics["exact_hits"],
            "Train ≥3/4": train_metrics["three_plus"],
            "Train chevaux trouvés": train_metrics["total_target_hits"],
            "Oracle pool": oof_metrics["oracle_pool_hits"],
        }
        board.append(row)

        # Pour le futur : priorité à l'OOF groupé.
        oof_key = (
            oof_metrics["exact_hits"],
            oof_metrics["three_plus"],
            oof_metrics["total_target_hits"],
            train_metrics["exact_hits"],
        )

        if best_oof_key is None or oof_key > best_oof_key:
            best_oof_key = oof_key
            best_oof_bundle = {
                "spec": dict(spec),
                "model": full_model,
                "oof_scores": oof_scores.copy(),
                "oof_metrics": oof_metrics,
                "train_metrics": train_metrics,
            }

        # Pure maximisation historique : affichée mais jamais présentée
        # comme validation indépendante.
        train_key = (
            train_metrics["exact_hits"],
            train_metrics["three_plus"],
            train_metrics["total_target_hits"],
            oof_metrics["exact_hits"],
        )

        if best_train_key is None or train_key > best_train_key:
            best_train_key = train_key
            best_train_bundle = {
                "spec": dict(spec),
                "model": full_model,
                "oof_metrics": oof_metrics,
                "train_metrics": train_metrics,
            }

        progress.progress(
            i / len(specs),
            text=(
                f"V5 Selector : {i}/{len(specs)} modèles testés"
            ),
        )

    progress.empty()

    leaderboard = (
        pd.DataFrame(board)
        .sort_values(
            [
                "OOF Multi4",
                "OOF ≥3/4",
                "OOF chevaux trouvés",
                "Train Multi4",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return {
        "dataset": data,
        "feature_columns": feature_columns,
        "leaderboard": leaderboard,
        "best_oof": best_oof_bundle,
        "best_train": best_train_bundle,
    }


def selector_live_rank(
    item: dict,
    model,
    feature_columns: list[str],
) -> pd.DataFrame:
    pool = candidate_pool(
        item,
        include_target=False,
    )

    if len(pool) < 4:
        raise RuntimeError(
            "Pool Selector inférieur à 4 chevaux."
        )

    x, _ = selector_matrix(
        pool,
        feature_columns=feature_columns,
    )

    pool["selector_probability"] = (
        positive_probability(
            model,
            x,
        )
    )

    pool = pool.sort_values(
        [
            "selector_probability",
            "m8_rank_place",
            "horse_number",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    pool["selector_rank"] = (
        np.arange(1, len(pool) + 1)
    )

    return pool




# =============================================================================
# V6 COMBORANKER — EVALUATION DIRECTE DES QUARTETS
# =============================================================================

def _safe_mean(values) -> float:
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(float)

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return 0.0

    return float(np.mean(arr))


def _safe_std(values) -> float:
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(float)

    arr = arr[np.isfinite(arr)]

    if len(arr) <= 1:
        return 0.0

    return float(np.std(arr))


def _safe_min(values) -> float:
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(float)

    arr = arr[np.isfinite(arr)]

    return float(np.min(arr)) if len(arr) else 0.0


def _safe_max(values) -> float:
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(float)

    arr = arr[np.isfinite(arr)]

    return float(np.max(arr)) if len(arr) else 0.0


def quartet_feature_row(
    item: dict,
    pool: pd.DataFrame,
    chosen_idx: tuple[int, int, int, int],
    include_target: bool,
) -> dict:
    chosen = pool.iloc[list(chosen_idx)].copy()
    excluded = pool.drop(
        pool.index[list(chosen_idx)]
    ).copy()

    row: dict[str, Any] = {
        "race_id_combo": int(item["race_id"]),
        "selection_tuple": tuple(
            sorted(
                int(x)
                for x in chosen["horse_number"].tolist()
            )
        ),
        "discipline_combo": str(
            item.get("discipline") or "INCONNU"
        ).upper(),
        "pool_size": float(len(pool)),
        "field_size": float(
            _safe_int(item.get("field_size"))
            or len(item["frame"])
        ),
        "is_e_multi": int(
            "E_MULTI" in str(item.get("multi_codes") or "").upper()
            and "E_MINI_MULTI" not in str(item.get("multi_codes") or "").upper()
        ),
        "is_mini_multi": int(
            "E_MINI_MULTI" in str(item.get("multi_codes") or "").upper()
        ),
    }

    # Variables continues et de rang : résumé du quartet sélectionné.
    numeric_cols = [
        "m8_rank_place",
        "neural_rank",
        "market_rank",
        "m8_rank_win",
        "best_rank",
        "mean_rank",
        "worst_rank",
        "rank_std",
        "rank_range",
        "m8_place_probability",
        "neural_top3_probability",
        "market_probability",
        "m8_win_probability",
        "sig_m8_place",
        "sig_neural_top3",
        "sig_market",
        "sig_m8_win",
        "odds",
        "log_odds",
        "votes_top3",
        "votes_top4",
        "votes_top5",
        "votes_top6",
    ]

    for col in numeric_cols:
        values = chosen[col] if col in chosen.columns else pd.Series(dtype=float)

        row[f"{col}_sum"] = float(
            pd.to_numeric(
                values,
                errors="coerce",
            ).fillna(0.0).sum()
        )
        row[f"{col}_mean"] = _safe_mean(values)
        row[f"{col}_min"] = _safe_min(values)
        row[f"{col}_max"] = _safe_max(values)
        row[f"{col}_std"] = _safe_std(values)

        # Comparaison quartet vs chevaux laissés de côté.
        if len(excluded):
            excl_values = (
                excluded[col]
                if col in excluded.columns
                else pd.Series(dtype=float)
            )
            row[f"{col}_delta_excluded_mean"] = (
                _safe_mean(values)
                - _safe_mean(excl_values)
            )
        else:
            row[f"{col}_delta_excluded_mean"] = 0.0

    # Compteurs de consensus / divergence.
    binary_cols = [
        "only_m8_top6",
        "only_neural_top6",
        "only_market_top6",
        "m8_and_neural_top6",
        "m8_and_market_top6",
        "neural_and_market_top6",
        "all3_top6",
    ]

    for col in binary_cols:
        row[f"{col}_count"] = int(
            pd.to_numeric(
                chosen[col],
                errors="coerce",
            ).fillna(0).sum()
        )

    # Combien des Top4 de chaque source sont retenus.
    row["count_m8_top4"] = int(
        (pd.to_numeric(chosen["m8_rank_place"], errors="coerce") <= 4).sum()
    )
    row["count_neural_top4"] = int(
        (pd.to_numeric(chosen["neural_rank"], errors="coerce") <= 4).sum()
    )
    row["count_market_top4"] = int(
        (pd.to_numeric(chosen["market_rank"], errors="coerce") <= 4).sum()
    )

    row["sum_three_source_top4"] = (
        row["count_m8_top4"]
        + row["count_neural_top4"]
        + row["count_market_top4"]
    )

    row["count_votes3_top4"] = int(
        (pd.to_numeric(chosen["votes_top4"], errors="coerce") == 3).sum()
    )
    row["count_votes2plus_top4"] = int(
        (pd.to_numeric(chosen["votes_top4"], errors="coerce") >= 2).sum()
    )
    row["count_votes3_top6"] = int(
        (pd.to_numeric(chosen["votes_top6"], errors="coerce") == 3).sum()
    )

    if include_target:
        target = set(item.get("target") or [])
        overlap = len(
            target.intersection(
                set(
                    int(x)
                    for x in chosen["horse_number"].tolist()
                )
            )
        )
        row["overlap_target"] = int(overlap)
        row["exact_target"] = int(overlap == 4)

    return row


def combo_dataset(
    cache: list[dict],
    include_target: bool = True,
) -> pd.DataFrame:
    import itertools

    rows: list[dict] = []

    for item in cache:
        pool = candidate_pool(
            item,
            include_target=include_target,
        )

        if len(pool) < 4:
            continue

        for combo in itertools.combinations(
            range(len(pool)),
            4,
        ):
            rows.append(
                quartet_feature_row(
                    item,
                    pool,
                    combo,
                    include_target=include_target,
                )
            )

    return pd.DataFrame(rows)


def combo_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    excluded = {
        "race_id_combo",
        "selection_tuple",
        "discipline_combo",
        "overlap_target",
        "exact_target",
    }

    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded
        and col != "discipline_combo"
    ]

    x_num = df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    discipline = pd.get_dummies(
        df["discipline_combo"].fillna("INCONNU"),
        prefix="discipline",
        dtype=float,
    )

    x = pd.concat(
        [
            x_num.reset_index(drop=True),
            discipline.reset_index(drop=True),
        ],
        axis=1,
    )

    if feature_columns is None:
        feature_columns = list(x.columns)
    else:
        for col in feature_columns:
            if col not in x.columns:
                x[col] = 0.0

        x = x.reindex(
            columns=feature_columns,
            fill_value=0.0,
        )

    return x, feature_columns


def combo_specs() -> list[dict]:
    specs = []

    for alpha in (0.1, 1.0, 10.0, 100.0):
        specs.append(
            {
                "name": f"Ridge alpha={alpha}",
                "kind": "ridge",
                "alpha": alpha,
            }
        )

    specs.extend(
        [
            {
                "name": "RFreg depth=2 leaf=8",
                "kind": "rfreg",
                "max_depth": 2,
                "min_samples_leaf": 8,
            },
            {
                "name": "RFreg depth=3 leaf=6",
                "kind": "rfreg",
                "max_depth": 3,
                "min_samples_leaf": 6,
            },
            {
                "name": "RFreg depth=4 leaf=4",
                "kind": "rfreg",
                "max_depth": 4,
                "min_samples_leaf": 4,
            },
            {
                "name": "RFreg depth=6 leaf=2",
                "kind": "rfreg",
                "max_depth": 6,
                "min_samples_leaf": 2,
            },
            {
                "name": "ExtraReg depth=3 leaf=6",
                "kind": "extrareg",
                "max_depth": 3,
                "min_samples_leaf": 6,
            },
            {
                "name": "ExtraReg depth=5 leaf=4",
                "kind": "extrareg",
                "max_depth": 5,
                "min_samples_leaf": 4,
            },
            {
                "name": "ExtraReg depth=7 leaf=2",
                "kind": "extrareg",
                "max_depth": 7,
                "min_samples_leaf": 2,
            },
            {
                "name": "GBreg 100x0.03 depth1",
                "kind": "gbreg",
                "n_estimators": 100,
                "learning_rate": 0.03,
                "max_depth": 1,
            },
            {
                "name": "GBreg 140x0.04 depth1",
                "kind": "gbreg",
                "n_estimators": 140,
                "learning_rate": 0.04,
                "max_depth": 1,
            },
            {
                "name": "GBreg 140x0.03 depth2",
                "kind": "gbreg",
                "n_estimators": 140,
                "learning_rate": 0.03,
                "max_depth": 2,
            },
        ]
    )

    return specs


def build_combo_model(
    spec: dict,
):
    kind = spec["kind"]

    if kind == "ridge":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=float(spec["alpha"]),
                    ),
                ),
            ]
        )

    if kind == "rfreg":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=spec["max_depth"],
                        min_samples_leaf=int(spec["min_samples_leaf"]),
                        max_features="sqrt",
                        random_state=SELECTOR_RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if kind == "extrareg":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=350,
                        max_depth=spec["max_depth"],
                        min_samples_leaf=int(spec["min_samples_leaf"]),
                        max_features="sqrt",
                        random_state=SELECTOR_RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if kind == "gbreg":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=int(spec["n_estimators"]),
                        learning_rate=float(spec["learning_rate"]),
                        max_depth=int(spec["max_depth"]),
                        random_state=SELECTOR_RANDOM_STATE,
                        loss="huber",
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Combo model inconnu : {kind}"
    )


def combo_metrics(
    data: pd.DataFrame,
    scores: np.ndarray,
) -> dict:
    work = data[
        [
            "race_id_combo",
            "selection_tuple",
            "overlap_target",
            "exact_target",
        ]
    ].copy()

    work["combo_score"] = scores

    exact = 0
    three_plus = 0
    total_hits = 0
    detail_rows = []

    for race_id, group in work.groupby(
        "race_id_combo",
        sort=False,
    ):
        # IMPORTANT :
        # le tri doit reproduire le forward SANS utiliser la cible réelle.
        # L'ancienne V6 utilisait overlap_target comme second critère en cas
        # d'égalité de score, ce qui introduisait une fuite de cible.
        #
        # Ici, on conserve l'ordre déterministe de génération des quartets
        # et on trie UNIQUEMENT sur combo_score avec un tri stable.
        group = group.sort_values(
            "combo_score",
            ascending=False,
            kind="mergesort",
        )

        chosen = group.iloc[0]

        hits = int(
            chosen["overlap_target"]
        )

        exact += int(hits == 4)
        three_plus += int(hits >= 3)
        total_hits += hits

        detail_rows.append(
            {
                "race_id": int(race_id),
                "selected_hits": hits,
                "exact_multi4": hits == 4,
                "selection": "-".join(
                    str(x)
                    for x in chosen["selection_tuple"]
                ),
                "score": float(
                    chosen["combo_score"]
                ),
            }
        )

    races = int(
        work["race_id_combo"].nunique()
    )

    return {
        "races": races,
        "exact_hits": exact,
        "three_plus": three_plus,
        "total_target_hits": total_hits,
        "avg_target_hits": (
            total_hits / races
            if races
            else 0.0
        ),
        "detail": pd.DataFrame(
            detail_rows
        ),
    }


def train_combo_search(
    cache: list[dict],
) -> dict:
    data = combo_dataset(
        cache,
        include_target=True,
    )

    if data.empty:
        raise RuntimeError(
            "Dataset ComboRanker vide."
        )

    x, feature_columns = combo_matrix(
        data
    )

    y = data[
        "overlap_target"
    ].astype(float).to_numpy()

    groups = data[
        "race_id_combo"
    ].astype(int).to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(
        SELECTOR_FOLDS,
        len(unique_groups),
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    specs = combo_specs()
    board = []
    best_key = None
    best_bundle = None

    progress = st.progress(
        0.0,
        text="V6 ComboRanker : recherche des modèles…",
    )

    for i, spec in enumerate(
        specs,
        start=1,
    ):
        oof_scores = np.full(
            len(data),
            np.nan,
            dtype=float,
        )

        for train_idx, valid_idx in splitter.split(
            x,
            y,
            groups,
        ):
            model = build_combo_model(
                spec
            )

            model.fit(
                x.iloc[train_idx],
                y[train_idx],
            )

            oof_scores[valid_idx] = (
                model.predict(
                    x.iloc[valid_idx]
                )
            )

        if np.isnan(oof_scores).any():
            raise RuntimeError(
                "Scores ComboRanker OOF incomplets."
            )

        oof = combo_metrics(
            data,
            oof_scores,
        )

        full_model = build_combo_model(
            spec
        )
        full_model.fit(
            x,
            y,
        )

        train_scores = full_model.predict(
            x
        )
        train = combo_metrics(
            data,
            train_scores,
        )

        row = {
            "Modèle": spec["name"],
            "Type": spec["kind"],
            "OOF Multi4": oof["exact_hits"],
            "OOF ≥3/4": oof["three_plus"],
            "OOF chevaux trouvés": oof["total_target_hits"],
            "OOF moy./4": round(
                oof["avg_target_hits"],
                4,
            ),
            "Train Multi4": train["exact_hits"],
            "Train ≥3/4": train["three_plus"],
            "Train chevaux trouvés": train["total_target_hits"],
        }

        board.append(row)

        key = (
            oof["exact_hits"],
            oof["three_plus"],
            oof["total_target_hits"],
            train["exact_hits"],
        )

        if best_key is None or key > best_key:
            best_key = key
            best_bundle = {
                "spec": dict(spec),
                "model": full_model,
                "oof_metrics": oof,
                "train_metrics": train,
            }

        progress.progress(
            i / len(specs),
            text=(
                f"V6 ComboRanker : "
                f"{i}/{len(specs)} modèles"
            ),
        )

    progress.empty()

    leaderboard = (
        pd.DataFrame(board)
        .sort_values(
            [
                "OOF Multi4",
                "OOF ≥3/4",
                "OOF chevaux trouvés",
                "Train Multi4",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return {
        "dataset": data,
        "feature_columns": feature_columns,
        "leaderboard": leaderboard,
        "best": best_bundle,
    }


def combo_live_rank(
    item: dict,
    model,
    feature_columns: list[str],
) -> tuple[list[int], pd.DataFrame]:
    pool = candidate_pool(
        item,
        include_target=False,
    )

    if len(pool) < 4:
        raise RuntimeError(
            "Pool V6 inférieur à 4 chevaux."
        )

    import itertools

    rows = []

    for combo in itertools.combinations(
        range(len(pool)),
        4,
    ):
        rows.append(
            quartet_feature_row(
                item,
                pool,
                combo,
                include_target=False,
            )
        )

    combos = pd.DataFrame(rows)

    x, _ = combo_matrix(
        combos,
        feature_columns=feature_columns,
    )

    combos["combo_score"] = model.predict(
        x
    )

    combos = combos.sort_values(
        "combo_score",
        ascending=False,
    ).reset_index(drop=True)

    best = list(
        combos.iloc[0]["selection_tuple"]
    )

    return best, combos



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

    # Etage 1 : vrais rapports Multi avec cible Top4 PMU exploitable.
    pmu_reports_ok = pmu_all[
        pmu_all["report_ok"] == True
    ].copy()

    races = load_races(
        CALIBRATION_START,
        CALIBRATION_END,
    )

    mapped = map_pmu_to_db(
        pmu_reports_ok,
        races,
    )

    if mapped.empty or "race_id" not in mapped.columns:
        return {
            "pmu_all": pmu_all,
            "pmu_reports_ok": pmu_reports_ok,
            "historical_complete": pd.DataFrame(),
            "mapped": mapped,
            "cache": [],
            "audit": pd.DataFrame(),
        }

    mapped["race_id"] = pd.to_numeric(
        mapped["race_id"],
        errors="coerce",
    )

    mapped_ok = mapped[
        mapped["race_id"].notna()
    ].copy()

    mapped_ok["race_id"] = (
        mapped_ok["race_id"].astype(int)
    )

    race_ids = sorted(
        mapped_ok["race_id"].unique().tolist()
    )

    # Etage 2 : reproduire exactement la condition "résultats complets"
    # qui définissait le bloc historique de 84 courses.
    participants = load_participants(
        race_ids
    )

    finish_targets = historical_finish_targets(
        participants
    )

    # Etage 3 : prédictions gelées M8 + Neural.
    predictions = load_predictions(
        race_ids
    )

    frames = paired_frames(
        predictions
    )

    cache: list[dict] = []
    audit_rows: list[dict] = []
    complete_rows: list[dict] = []

    for _, row in mapped.iterrows():
        race_id = _safe_int(
            row.get("race_id")
        )

        pmu_target = row.get("target_set")
        finish_target = (
            finish_targets.get(race_id)
            if race_id is not None
            else None
        )
        raw = (
            frames.get(race_id)
            if race_id is not None
            else None
        )

        reason = ""

        if race_id is None:
            reason = "course absente Supabase"

        elif not isinstance(pmu_target, frozenset) or len(pmu_target) != 4:
            reason = "cible PMU Top4 invalide"

        elif not isinstance(finish_target, frozenset) or len(finish_target) != 4:
            reason = "arrivée Top4 Supabase indisponible"

        elif pmu_target != finish_target:
            reason = "désaccord PMU / arrivée Supabase"

        elif raw is None or raw.empty:
            reason = "prédictions M8/Neural absentes"

        elif len(raw) < 4:
            reason = "moins de 4 chevaux appariés"

        else:
            predicted_numbers = set(
                int(x)
                for x in raw["horse_number"].tolist()
            )

            if not pmu_target.issubset(predicted_numbers):
                reason = "un cheval du Top4 absent des prédictions"

        usable = reason == ""

        audit_rows.append(
            {
                "race_date": row.get("race_date"),
                "meeting_number": row.get("meeting_number"),
                "race_number": row.get("race_number"),
                "race_id": race_id,
                "multi_codes": row.get("multi_codes"),
                "target_top4_pmu": row.get("target_top4"),
                "target_top4_supabase": (
                    "-".join(map(str, sorted(finish_target)))
                    if isinstance(finish_target, frozenset)
                    else ""
                ),
                "usable_84": usable,
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

        complete_rows.append(
            row.to_dict()
        )

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
                "multi_codes": row.get("multi_codes"),
                "discipline": (
                    row.get("discipline")
                    or (
                        str(frame["discipline"].iloc[0])
                        if "discipline" in frame.columns and len(frame)
                        else "INCONNU"
                    )
                ),
                "field_size": (
                    _safe_int(row.get("field_size"))
                    or len(frame)
                ),
                "target": pmu_target,
                "frame": frame,
                "matrix": feature_matrix(frame),
            }
        )

    historical_complete = pd.DataFrame(
        complete_rows
    )

    return {
        "pmu_all": pmu_all,
        "pmu_reports_ok": pmu_reports_ok,
        "historical_complete": historical_complete,
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
        "Reconstruction des 84 courses exactes…"
    ):
        package = build_calibration_package()

    cache = package["cache"]

    if not cache:
        raise RuntimeError(
            "Aucune course exploitable."
        )

    m8_place = evaluate_rank_column(
        cache,
        "m8_rank_place",
    )

    if not (
        len(cache) == EXPECTED_COHORT
        and _matches_expected(m8_place)
    ):
        raise RuntimeError(
            "Sanity check refusé : la cohorte 84 ou "
            "M8 place 5/15/23 n'est pas reproduite."
        )

    v4 = evaluate_weights(
        cache,
        V4_WEIGHTS,
    )

    oracle = oracle_pool_table(
        cache
    )

    # Baseline V5 pointwise.
    selector_v5 = train_selector_search(
        cache
    )

    # V6 : modèle combinatoire directement aligné sur le quartet.
    combo_v6 = train_combo_search(
        cache
    )

    fingerprint_payload = {
        "version": APP_VERSION,
        "cohort": len(cache),
        "pool_k": SELECTOR_POOL_K,
        "v5_spec": selector_v5["best_oof"]["spec"],
        "v6_spec": combo_v6["best"]["spec"],
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
        "m8_place": m8_place,
        "v4_metrics": v4,
        "oracle": oracle,
        "selector_v5": selector_v5,
        "combo_v6": combo_v6,
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
    st.header("⚙️ Multi M9 V6.1 Audit")
    st.write(f"**Version :** `{APP_VERSION}`")
    st.write("**Cohorte :** 84 courses")
    st.write("**Pool :** union Top6 M8 place + Neural + marché")
    st.write("**Quartets :** toutes les combinaisons de 4")
    st.write("**Choix forward :** meilleur modèle OOF groupé")

    if "m9v6" in st.session_state:
        cal = st.session_state["m9v6"]
        st.success("✅ V6 calibré")
        st.caption(
            "Empreinte : "
            + cal["fingerprint"][:14]
            + "…"
        )


tab_cal, tab_live, tab_method = st.tabs(
    [
        "🧩 Calibration V6 ComboRanker",
        "🎯 Pronostics Multi",
        "📐 Méthode",
    ]
)


with tab_cal:
    st.subheader(
        "🧩 V6 : noter directement chaque combinaison de 4 chevaux"
    )

    st.warning(
        "Le score OOF groupé par course est le critère principal. "
        "Le score Train reste descriptif : il ne constitue pas une validation indépendante."
    )

    st.error(
        "Audit méthodologique : V6 utilisait involontairement l'overlap réel comme "
        "second critère lorsqu'au moins deux quartets avaient exactement le même score. "
        "V6.1 supprime ce tie-break. Le score V6 précédent (9/84) ne doit donc pas "
        "être figé tant que ce recalcul n'est pas terminé."
    )

    if st.button(
        "🧪 Recalculer V6.1 sans fuite de tie-break",
        type="primary",
        use_container_width=True,
    ):
        try:
            st.session_state["m9v6"] = (
                run_calibration()
            )
        except Exception as exc:
            st.exception(exc)

    cal = st.session_state.get("m9v6")

    if cal is not None:
        v5 = cal["selector_v5"]["best_oof"]
        v6 = cal["combo_v6"]["best"]

        a, b, c, d, e = st.columns(5)

        a.metric(
            "M8 place",
            f"{cal['m8_place']['hits4']}/84",
        )
        b.metric(
            "M9 V4",
            f"{cal['v4_metrics']['hits4']}/84",
        )
        c.metric(
            "V5 OOF",
            f"{v5['oof_metrics']['exact_hits']}/84",
        )
        d.metric(
            "V6 OOF",
            f"{v6['oof_metrics']['exact_hits']}/84",
            delta=f"{v6['oof_metrics']['exact_hits'] - v5['oof_metrics']['exact_hits']:+d} vs V5",
        )
        e.metric(
            "Oracle Top6",
            "39/84",
        )

        st.success(
            "✅ Sanity check : cohorte 84 et M8 place = "
            f"{cal['m8_place']['hits4']}/84, "
            f"{cal['m8_place']['hits5']}/84, "
            f"{cal['m8_place']['hits6']}/84."
        )

        st.markdown("### 🏆 Modèle V6 retenu")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "OOF Multi4",
            f"{v6['oof_metrics']['exact_hits']}/84",
        )
        c2.metric(
            "OOF ≥3/4",
            f"{v6['oof_metrics']['three_plus']}/84",
        )
        c3.metric(
            "OOF chevaux trouvés",
            f"{v6['oof_metrics']['total_target_hits']}/336",
        )
        c4.metric(
            "Train Multi4",
            f"{v6['train_metrics']['exact_hits']}/84",
        )

        st.write(
            f"**ComboRanker choisi :** `{v6['spec']['name']}`"
        )

        st.markdown("### 🔭 Oracle du pool")
        st.dataframe(
            cal["oracle"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🧩 Tous les modèles V6")
        st.dataframe(
            cal["combo_v6"]["leaderboard"],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("📊 Baseline V5 pointwise"):
            st.dataframe(
                cal["selector_v5"]["leaderboard"],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("🔎 Détail course par course — V6 OOF"):
            detail = v6["oof_metrics"]["detail"].copy()

            audit = cal["audit"][
                cal["audit"]["usable_84"] == True
            ][
                [
                    "race_date",
                    "meeting_number",
                    "race_number",
                    "race_id",
                    "multi_codes",
                    "target_top4_pmu",
                ]
            ].copy()

            detail = audit.merge(
                detail,
                on="race_id",
                how="left",
            )

            st.dataframe(
                detail,
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "⬇️ Télécharger les résultats V6 OOF",
            data=v6["oof_metrics"]["detail"].to_csv(
                index=False
            ),
            file_name="m9_v6_comboranker_oof.csv",
            mime="text/csv",
            use_container_width=True,
        )


with tab_live:
    st.subheader(
        "🎯 Pronostics Multi — V6 ComboRanker"
    )

    cal = st.session_state.get("m9v6")

    if cal is None:
        st.info(
            "Calibre d'abord V6 dans le premier onglet."
        )
    else:
        selected_day = st.date_input(
            "Date PMU",
            value=date.today(),
        )

        if st.button(
            "🔎 Analyser les Multi avec V6",
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
                    bundle = cal["combo_v6"]["best"]
                    model = bundle["model"]
                    feature_columns = (
                        cal["combo_v6"]["feature_columns"]
                    )

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

                        item = {
                            "race_id": race_id,
                            "frame": raw,
                            "field_size": (
                                _safe_int(row.get("field_size"))
                                or len(raw)
                            ),
                            "discipline": (
                                row.get("discipline")
                                or (
                                    str(raw["discipline"].iloc[0])
                                    if "discipline" in raw.columns and len(raw)
                                    else "INCONNU"
                                )
                            ),
                            "multi_codes": row.get("multi_codes"),
                        }

                        best, combos = combo_live_rank(
                            item,
                            model,
                            feature_columns,
                        )

                        st.metric(
                            "🎯 Multi4 V6",
                            " - ".join(
                                map(str, best)
                            ),
                        )

                        st.caption(
                            f"{len(combos)} quartets évalués · "
                            f"modèle : {bundle['spec']['name']}"
                        )

                        show = combos.head(10).copy()
                        show["selection"] = show[
                            "selection_tuple"
                        ].apply(
                            lambda x: "-".join(
                                str(n)
                                for n in x
                            )
                        )

                        st.dataframe(
                            show[
                                [
                                    "selection",
                                    "combo_score",
                                ]
                            ].rename(
                                columns={
                                    "selection": "Quartet",
                                    "combo_score": "Score V6",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.divider()

            except Exception as exc:
                st.exception(exc)


with tab_method:
    st.subheader("📐 Pourquoi V6 ?")

    st.markdown(
        """
V5 notait chaque cheval séparément puis prenait les quatre meilleurs.
C'est utile, mais ce n'est pas exactement notre objectif.

V6 travaille directement au niveau du **quartet** :

1. pool = union des **Top6 M8 placé + Neural + marché** ;
2. toutes les combinaisons possibles de **4 chevaux** sont générées ;
3. chaque quartet reçoit des caractéristiques collectives :
   consensus, sommes/moyennes de rangs, probabilités, écarts entre modèles,
   nombre de chevaux soutenus par 2 ou 3 sources, cotes et contexte de course ;
4. un modèle prédit le **nombre attendu de chevaux de l'arrivée Top4**
   présents dans ce quartet ;
5. on sélectionne le quartet au score maximal.

L'entraînement et l'évaluation OOF sont séparés par **course entière**. V6.1 garantit aussi qu'aucune information de l'arrivée n'intervient dans le départage de deux quartets ayant le même score.
Ainsi aucune combinaison d'une course de validation n'apparaît dans
l'entraînement du fold qui la prédit.

Le but est d'améliorer surtout les nombreuses courses où V5 trouvait déjà
**3 chevaux sur 4**, en choisissant mieux le quatrième divergent.
        """
    )

    st.info(
        "Les 84 courses restent un jeu de calibration. "
        "Le modèle choisi devra ensuite être figé pour le forward."
    )
