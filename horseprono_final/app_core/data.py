from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from .config import HISTORY_COLUMNS


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "date": "race_date", "date_course": "race_date", "cheval": "horse_name", "n°": "horse_number",
        "numero": "horse_number", "cote": "odds", "jockey_name": "jockey", "entraineur": "trainer",
        "position": "finish_position", "classement": "finish_position", "corde": "draw",
    }
    out.columns = [str(c).strip() for c in out.columns]
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    for c in HISTORY_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    for c in ["distance", "field_size", "horse_number", "draw", "finish_position"]:
        out[c] = _num(out[c])
    for c in ["odds", "weight", "career_runs", "career_wins", "career_places"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = _num(out[c])
    if "career_win_rate" not in out.columns:
        out["career_win_rate"] = np.where(out["career_runs"] > 0, out["career_wins"] / out["career_runs"], np.nan)
    if "career_place_rate" not in out.columns:
        out["career_place_rate"] = np.where(out["career_runs"] > 0, out["career_places"] / out["career_runs"], np.nan)
    out["race_id"] = out["race_id"].astype(str)
    out["horse_name"] = out["horse_name"].fillna("Inconnu").astype(str)
    out["jockey"] = out["jockey"].fillna("").astype(str)
    out["trainer"] = out["trainer"].fillna("").astype(str)
    # Derive field size when absent.
    if out["field_size"].isna().all():
        out["field_size"] = out.groupby("race_id")["horse_number"].transform("count")
    else:
        out["field_size"] = out["field_size"].fillna(out.groupby("race_id")["horse_number"].transform("count"))

    def form_score(value):
        if pd.isna(value):
            return np.nan
        nums = [int(x) for x in __import__("re").findall(r"\d+", str(value))[:5]]
        if not nums:
            return np.nan
        # 1st = 1.0, then progressively lower scores.
        return float(np.mean([max(0.0, 1.0 - (n - 1) / 9.0) for n in nums]))
    out["recent_form_score"] = out["recent_form"].map(form_score)
    return out


def normalize_race_input(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_history(df)
    if out["race_id"].eq("nan").all() or out["race_id"].eq("").all():
        out["race_id"] = "manual-race"
    out["race_id"] = out["race_id"].replace({"nan": "manual-race", "None": "manual-race"})
    out["race_date"] = out["race_date"].fillna(pd.Timestamp.today().normalize())
    return out


def read_uploaded_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    return normalize_history(pd.read_csv(BytesIO(raw)))


def example_race() -> pd.DataFrame:
    rows = [
        ["demo-1", pd.Timestamp.today(), "PLAT", "ParisLongchamp", 2100, "Bon", 12, 1, "Cheval Alpha", "Jockey A", "Entraîneur A", 4.5, 2, 57, "1a 2a 4a", None],
        ["demo-1", pd.Timestamp.today(), "PLAT", "ParisLongchamp", 2100, "Bon", 12, 2, "Cheval Bravo", "Jockey B", "Entraîneur B", 6.2, 7, 58, "2a 5a 1a", None],
        ["demo-1", pd.Timestamp.today(), "PLAT", "ParisLongchamp", 2100, "Bon", 12, 3, "Cheval Charlie", "Jockey C", "Entraîneur C", 9.8, 4, 56, "3a 3a 6a", None],
        ["demo-1", pd.Timestamp.today(), "PLAT", "ParisLongchamp", 2100, "Bon", 12, 4, "Cheval Delta", "Jockey D", "Entraîneur D", 12.5, 9, 59, "5a 1a 2a", None],
        ["demo-1", pd.Timestamp.today(), "PLAT", "ParisLongchamp", 2100, "Bon", 12, 5, "Cheval Echo", "Jockey E", "Entraîneur E", 18.0, 1, 55, "7a 4a 3a", None],
        ["demo-1", pd.Timestamp.today(), "PLAT", "ParisLongchamp", 2100, "Bon", 12, 6, "Cheval Foxtrot", "Jockey F", "Entraîneur F", 25.0, 6, 57, "6a 8a 4a", None],
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)
