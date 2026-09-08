from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .data import normalize_race_input


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = normalize_race_input(df)
    field = x["field_size"].fillna(x["field_size"].median() if x["field_size"].notna().any() else len(x))
    odds = x["odds"].clip(lower=1.01)
    draw = x["draw"]
    if draw.notna().any():
        max_draw = max(float(draw.max()), 1.0)
        draw_norm = (draw - 1) / max(max_draw - 1, 1.0)
    else:
        draw_norm = pd.Series(0.5, index=x.index)

    weight_median = x["weight"].median() if x["weight"].notna().any() else 57.0
    distance_median = x["distance"].median() if x["distance"].notna().any() else 2100.0

    features = pd.DataFrame(index=x.index)
    features["log_odds"] = np.log(odds)
    features["odds_implied"] = 1.0 / odds
    features["draw_norm"] = draw_norm.fillna(0.5)
    features["weight_norm"] = ((x["weight"].fillna(weight_median) - weight_median) / 5.0).clip(-3, 3)
    features["recent_form_score"] = x["recent_form_score"].fillna(0.5)
    features["career_win_rate"] = x["career_win_rate"].fillna(0.10).clip(0, 1)
    features["career_place_rate"] = x["career_place_rate"].fillna(0.30).clip(0, 1)
    features["field_size_norm"] = ((field - 10.0) / 10.0).clip(-2, 2)
    features["distance_norm"] = ((x["distance"].fillna(distance_median) - distance_median) / 1000.0).clip(-3, 3)
    return features[FEATURE_COLUMNS]


def explain_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable fallback score based on known directional effects.

    This is intentionally transparent and is only used when no trained model is available.
    """
    x = normalize_race_input(df).copy()
    odds_score = (1 / x["odds"].clip(lower=1.01)).rank(pct=True).fillna(0.5)
    form_score = x["recent_form_score"].fillna(0.5)
    win_rate = x["career_win_rate"].fillna(0.10).clip(0, 1)
    place_rate = x["career_place_rate"].fillna(0.30).clip(0, 1)
    draw_score = 1 - x["draw"].fillna(x["draw"].median() if x["draw"].notna().any() else 5).rank(pct=True) + 0.5
    draw_score = draw_score.clip(0, 1)
    return pd.DataFrame({
        "Cote": odds_score,
        "Forme récente": form_score,
        "Taux victoire": win_rate,
        "Taux placé": place_rate,
        "Corde": draw_score,
    }, index=x.index)
