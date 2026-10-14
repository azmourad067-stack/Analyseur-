"""Règles pures et gelées de HorseProno Morning Forward V1."""
from __future__ import annotations
import re
from typing import Any
import pandas as pd

EXPERIMENT_NAME = "horseprono_morning_forward_v1"
MODEL8_ID = 8
NEURAL_ID = 9
MIN_RUNNERS, MAX_RUNNERS = 8, 14
MIN_ODDS_INCLUSIVE, MAX_ODDS_EXCLUSIVE = 5.0, 8.0
REQUIRED_MODEL8_WIN_RANK = 2
MAX_NEURAL_PLACE_RANK = 3
STAKE_UNITS = 1.0

def _norm(value: Any) -> str:
    return re.sub(r"\\s+", " ", str(value or "").strip().lower())

def is_eligible_discipline(discipline: Any) -> bool:
    d = _norm(discipline)
    trot_attele = ("trot attelé" in d or "trot attele" in d or "attel" in d)
    variant = any(x in d for x in ("autostart", "volté", "volte"))
    return trot_attele and variant

def select_morning_candidate(model8_predictions: pd.DataFrame,
                             neural_predictions: pd.DataFrame, *,
                             discipline: Any, field_size: int) -> dict | None:
    """Retourne au maximum un cheval admissible par course; cote du snapshot 09:00."""
    if not is_eligible_discipline(discipline):
        return None
    if not MIN_RUNNERS <= int(field_size) <= MAX_RUNNERS:
        return None
    req8 = {"horse_number", "odds", "rank_win"}
    req9 = {"horse_number", "rank_place"}
    if not req8.issubset(model8_predictions.columns):
        raise ValueError(f"Colonnes M8 manquantes: {sorted(req8 - set(model8_predictions.columns))}")
    if not req9.issubset(neural_predictions.columns):
        raise ValueError(f"Colonnes Neural manquantes: {sorted(req9 - set(neural_predictions.columns))}")
    m8 = model8_predictions.copy()
    n9 = neural_predictions.copy()
    for df, cols in ((m8, ("horse_number", "odds", "rank_win")),
                     (n9, ("horse_number", "rank_place"))):
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    m8 = m8[m8.rank_win.eq(REQUIRED_MODEL8_WIN_RANK)
            & m8.odds.ge(MIN_ODDS_INCLUSIVE)
            & m8.odds.lt(MAX_ODDS_EXCLUSIVE)]
    n9 = n9[n9.rank_place.le(MAX_NEURAL_PLACE_RANK)]
    joined = m8.merge(n9[["horse_number", "rank_place"]],
                      on="horse_number", how="inner", validate="one_to_one")
    if joined.empty:
        return None
    row = joined.sort_values(["rank_win", "horse_number"]).iloc[0]
    return {"horse_number": int(row.horse_number),
            "odds_snapshot": float(row.odds),
            "model8_rank_win": int(row.rank_win),
            "neural_rank_place": int(row.rank_place),
            "stake": STAKE_UNITS, "selection_rule": EXPERIMENT_NAME}
