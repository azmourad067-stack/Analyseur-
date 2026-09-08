from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def market_probabilities(df: pd.DataFrame) -> np.ndarray:
    odds = pd.to_numeric(df["odds"], errors="coerce").clip(lower=1.01).fillna(100)
    raw = 1 / odds
    out = np.zeros(len(df))
    for _, idx in pd.Series(df["race_id"].astype(str).to_numpy()).groupby(pd.Series(df["race_id"].astype(str).to_numpy())).groups.items():
        ix = np.asarray(idx)
        s = raw.iloc[ix].sum()
        out[ix] = raw.iloc[ix] / s if s > 0 else 1 / len(ix)
    return out


def evaluate_predictions(df: pd.DataFrame, p_win: np.ndarray) -> dict[str, Any]:
    rows = []
    x = df.copy()
    x["p_win"] = p_win
    x["market_prob"] = market_probabilities(x)
    for race_id, g in x.groupby("race_id", sort=False):
        g = g.sort_values("p_win", ascending=False)
        top = g.iloc[0]
        odds = float(top["odds"]) if pd.notna(top["odds"]) else np.nan
        rows.append({
            "race_id": str(race_id),
            "top1_hit": int(top["finish_position"] == 1),
            "top3_hit": int((g.head(3)["finish_position"] <= 3).any()),
            "model_top1_prob": float(top["p_win"]),
            "market_top1_prob": float(top["market_prob"]),
            "odds": odds,
            "roi_return": odds if top["finish_position"] == 1 and np.isfinite(odds) else 0.0,
        })
    r = pd.DataFrame(rows)
    if r.empty:
        return {}
    stake = len(r)
    gross = r["roi_return"].sum()
    return {
        "races": int(len(r)),
        "top1_accuracy": float(r["top1_hit"].mean()),
        "top3_coverage": float(r["top3_hit"].mean()),
        "avg_model_top1_prob": float(r["model_top1_prob"].mean()),
        "avg_market_top1_prob": float(r["market_top1_prob"].mean()),
        "flat_stake_roi": float((gross - stake) / stake),
        "mean_top1_edge": float((r["model_top1_prob"] - r["market_top1_prob"]).mean()),
    }
