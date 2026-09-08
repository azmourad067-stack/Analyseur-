from __future__ import annotations

import re
from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd

from .config import REQUIRED_HISTORY_COLUMNS


def _num(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        return float(text) if text else np.nan
    except ValueError:
        return np.nan


def parse_form(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    tokens = re.findall(r"(?:\d+|[aA])", str(value))
    out: list[int] = []
    for token in tokens:
        if token.lower() == "a":
            out.append(10)
        else:
            try:
                out.append(int(token))
            except ValueError:
                pass
    return out


def form_score(value: object, decay: float = 0.72) -> float:
    """Higher is better; recent races receive more weight."""
    values = parse_form(value)[:8]
    if not values:
        return np.nan
    weights = np.array([decay ** i for i in range(len(values))], dtype=float)
    points = []
    for pos in values:
        if pos <= 1:
            points.append(1.00)
        elif pos <= 3:
            points.append(0.82)
        elif pos <= 5:
            points.append(0.65)
        elif pos <= 7:
            points.append(0.45)
        else:
            points.append(0.18)
    return float(np.average(points, weights=weights))


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    numeric_cols = [
        "distance", "field_size", "horse_number", "odds", "draw", "weight",
        "career_runs", "career_wins", "career_places", "finish_position"
    ]
    for col in numeric_cols:
        out[col] = out[col].map(_num)

    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce", dayfirst=False)
    out["horse_name"] = out["horse_name"].fillna("").astype(str).str.strip()
    out["jockey"] = out["jockey"].fillna("").astype(str).str.strip()
    out["trainer"] = out["trainer"].fillna("").astype(str).str.strip()
    out["discipline"] = out["discipline"].fillna("INCONNU").astype(str).str.upper().str.strip()
    out["terrain"] = out["terrain"].fillna("INCONNU").astype(str).str.upper().str.strip()
    out["recent_form_score"] = out["recent_form"].map(form_score)
    out["career_win_rate"] = np.where(
        out["career_runs"] > 0, out["career_wins"] / out["career_runs"], np.nan
    )
    out["career_place_rate"] = np.where(
        out["career_runs"] > 0, out["career_places"] / out["career_runs"], np.nan
    )
    return out


def normalize_race_input(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_history(df)
    if out["race_id"].isna().all() or (out["race_id"].astype(str).str.len() == 0).all():
        out["race_id"] = "manual_race"
    return out


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    return pd.read_csv(BytesIO(raw), sep=None, engine="python")


def example_race() -> pd.DataFrame:
    """Small illustrative field; deliberately synthetic and not historical data."""
    rows = [
        [1, "ALPHA TURF", "JOCKEY A", "TRAINER A", 5.8, 2, 57, "2 1 4 3 5", 30, 7, 16],
        [2, "BETA RUN", "JOCKEY B", "TRAINER B", 9.5, 8, 58, "3 4 2 6 1", 25, 4, 12],
        [3, "GAMMA STAR", "JOCKEY C", "TRAINER C", 4.1, 1, 57, "1 1 2 2 4", 22, 8, 15],
        [4, "DELTA WIND", "JOCKEY D", "TRAINER D", 15.0, 6, 59, "5 7 3 4 6", 28, 3, 10],
        [5, "EPSILON", "JOCKEY E", "TRAINER E", 8.1, 11, 56, "4 2 5 3 8", 19, 4, 9],
        [6, "ZETA QUEEN", "JOCKEY F", "TRAINER F", 12.5, 4, 56, "6 5 6 1 3", 31, 5, 13],
        [7, "ETA SPEED", "JOCKEY G", "TRAINER G", 22.0, 10, 58, "8 6 7 9 5", 34, 3, 11],
        [8, "THETA", "JOCKEY H", "TRAINER H", 18.0, 7, 55, "7 3 8 5 4", 15, 2, 7],
    ]
    return normalize_race_input(pd.DataFrame(rows, columns=[
        "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight",
        "recent_form", "career_runs", "career_wins", "career_places"
    ]).assign(
        race_id="DEMO",
        race_date=pd.Timestamp.today().normalize(),
        discipline="PLAT",
        hippodrome="DEMO",
        distance=2100,
        terrain="BON",
        field_size=8,
    ))
