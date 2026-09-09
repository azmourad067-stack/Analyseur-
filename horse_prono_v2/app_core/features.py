from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .data import normalize_history, normalize_race_input


def _norm_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _safe_rate(num: pd.Series, den: pd.Series, prior: float) -> pd.Series:
    return ((num + 1.0 * prior) / (den + 1.0)).clip(0, 1)


def build_temporal_training_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Create strictly pre-race entity statistics.

    The critical rule is shift-before-aggregation: a horse/jockey/trainer's result
    from the current race is never included in its features for that race.
    """
    df = normalize_history(history).copy()
    df = df.dropna(subset=["finish_position", "race_date"]).copy()
    if df.empty:
        return df
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date"])
    df["horse_key"] = df["horse_name"].map(_norm_text)
    df["jockey_key"] = df["jockey"].map(_norm_text)
    df["trainer_key"] = df["trainer"].map(_norm_text)
    df["race_key"] = df["race_id"].astype(str)
    df = df.sort_values(["race_date", "race_key", "horse_number"]).reset_index(drop=True)

    # Outcome signals of prior races only.
    df["is_win"] = (df["finish_position"] == 1).astype(float)
    df["is_place"] = (df["finish_position"] <= 3).astype(float)
    for entity in ["horse_key", "jockey_key", "trainer_key"]:
        grouped = df.groupby(entity, dropna=False)
        prior_wins = grouped["is_win"].cumsum() - df["is_win"]
        prior_places = grouped["is_place"].cumsum() - df["is_place"]
        prior_starts = grouped.cumcount()
        df[f"{entity.split('_')[0]}_win_rate"] = (prior_wins + 1.0) / (prior_starts + 10.0)
        df[f"{entity.split('_')[0]}_place_rate"] = (prior_places + 3.0) / (prior_starts + 10.0)
        df[f"{entity.split('_')[0]}_starts_prior"] = prior_starts.astype(float)

    # Recent form: exponentially weighted score of previous five finishes.
    form_values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
    consistency_values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
    recent_form = []
    recent_consistency = []
    for _, row in df.iterrows():
        key = row["horse_key"]
        hist = list(form_values[key])
        if hist:
            weights = np.exp(-0.45 * np.arange(len(hist))[::-1])
            weights = weights / weights.sum()
            recent_form.append(float(np.dot(hist, weights)))
            recent_consistency.append(float(1 - np.std(hist)))
        else:
            recent_form.append(0.45)
            recent_consistency.append(0.50)
        field = max(float(row.get("field_size") or 10), 2.0)
        finish = float(row["finish_position"])
        score = max(0.0, min(1.0, 1.0 - (finish - 1.0) / (field - 1.0)))
        form_values[key].append(score)
        consistency_values[key].append(score)
    df["recent_form_score"] = recent_form
    df["recent_consistency"] = recent_consistency
    return df


def make_features(df: pd.DataFrame, *, already_temporal: bool = False) -> pd.DataFrame:
    x = df.copy() if already_temporal else normalize_race_input(df)
    x = x.copy()
    field = pd.to_numeric(x["field_size"], errors="coerce")
    field = field.fillna(field.median() if field.notna().any() else len(x)).clip(lower=2)
    odds = pd.to_numeric(x["odds"], errors="coerce").clip(lower=1.01)
    market = 1.0 / odds

    draw = pd.to_numeric(x["draw"], errors="coerce")
    draw_norm = (draw - 1.0) / (field - 1.0)
    draw_norm = draw_norm.replace([np.inf, -np.inf], np.nan).fillna(0.50).clip(0, 1)
    draw_low = (1 - draw_norm).clip(0, 1)

    weight = pd.to_numeric(x["weight"], errors="coerce")
    weight_ref = weight.groupby(x["race_id"] if "race_id" in x.columns else pd.Series("one", index=x.index)).transform("median")
    weight_ref = weight_ref.fillna(weight.median() if weight.notna().any() else 57.0)
    weight_rel = ((weight - weight_ref) / 5.0).fillna(0.0).clip(-3, 3)

    distance = pd.to_numeric(x["distance"], errors="coerce")
    distance_norm = ((distance.fillna(2100.0) - 2100.0) / 1000.0).clip(-3, 3)
    field_norm = ((field - 10.0) / 10.0).clip(-2, 2)

    out = pd.DataFrame(index=x.index)
    out["log_odds"] = np.log(odds.fillna(20.0))
    out["market_prob"] = market.fillna(1.0 / 20.0)
    out["draw_norm"] = draw_norm
    out["draw_low_advantage"] = draw_low
    out["weight_rel"] = weight_rel
    out["distance_norm"] = distance_norm
    out["field_size_norm"] = field_norm

    # Use temporal columns when present; otherwise use any supplied snapshot fields.
    def col_or_default(name: str, default: float) -> pd.Series:
        if name in x.columns:
            s = pd.to_numeric(x[name], errors="coerce")
        else:
            s = pd.Series(default, index=x.index, dtype=float)
        return s.fillna(default)

    out["horse_win_rate"] = col_or_default("horse_win_rate", 0.10).clip(0, 1)
    if "horse_win_rate" not in x.columns and "career_win_rate" in x.columns:
        out["horse_win_rate"] = col_or_default("career_win_rate", 0.10).clip(0, 1)
    out["horse_place_rate"] = col_or_default("horse_place_rate", 0.30).clip(0, 1)
    if "horse_place_rate" not in x.columns and "career_place_rate" in x.columns:
        out["horse_place_rate"] = col_or_default("career_place_rate", 0.30).clip(0, 1)
    out["jockey_win_rate"] = col_or_default("jockey_win_rate", 0.10).clip(0, 1)
    out["jockey_place_rate"] = col_or_default("jockey_place_rate", 0.30).clip(0, 1)
    out["trainer_win_rate"] = col_or_default("trainer_win_rate", 0.10).clip(0, 1)
    out["trainer_place_rate"] = col_or_default("trainer_place_rate", 0.30).clip(0, 1)
    out["recent_form_score"] = col_or_default("recent_form_score", 0.45).clip(0, 1)
    out["recent_consistency"] = col_or_default("recent_consistency", 0.50).clip(0, 1)
    out["horse_starts_prior"] = col_or_default("horse_starts_prior", 0).clip(0, 1000)
    out["jockey_starts_prior"] = col_or_default("jockey_starts_prior", 0).clip(0, 10000)
    out["trainer_starts_prior"] = col_or_default("trainer_starts_prior", 0).clip(0, 10000)
    return out[FEATURE_COLUMNS]


def entity_snapshot_from_history(history: pd.DataFrame, race_date: Any) -> dict[str, pd.DataFrame]:
    """Return latest known horse/jockey/trainer stats strictly before race_date."""
    df = build_temporal_training_frame(history)
    if df.empty:
        return {"horse": pd.DataFrame(), "jockey": pd.DataFrame(), "trainer": pd.DataFrame()}
    cutoff = pd.Timestamp(race_date)
    df = df[df["race_date"] < cutoff].copy()
    result: dict[str, pd.DataFrame] = {}
    for entity, key_col, prefix in [
        ("horse", "horse_key", "horse"),
        ("jockey", "jockey_key", "jockey"),
        ("trainer", "trainer_key", "trainer"),
    ]:
        if df.empty:
            result[entity] = pd.DataFrame()
            continue
        cols = [key_col, f"{prefix}_win_rate", f"{prefix}_place_rate", f"{prefix}_starts_prior"]
        snap = df.sort_values("race_date").groupby(key_col, as_index=False).tail(1)[cols].copy()
        snap = snap.rename(columns={key_col: f"{prefix}_key"})
        result[entity] = snap
    return result


def enrich_live_race(race: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the latest historical entity stats before the target race."""
    out = normalize_race_input(race)
    if history is None or history.empty or out.empty:
        return out
    race_date = pd.to_datetime(out["race_date"].iloc[0], errors="coerce")
    if pd.isna(race_date):
        return out
    snap = entity_snapshot_from_history(history, race_date)
    for entity, col, prefix in [("horse", "horse_name", "horse"), ("jockey", "jockey", "jockey"), ("trainer", "trainer", "trainer")]:
        if snap[entity].empty:
            continue
        keys = out[col].map(_norm_text)
        tmp = snap[entity].set_index(f"{prefix}_key")
        for metric in ["win_rate", "place_rate", "starts_prior"]:
            out[f"{prefix}_{metric}"] = keys.map(tmp[f"{prefix}_{metric}"]).fillna(out.get(f"{prefix}_{metric}", np.nan))
    return out
