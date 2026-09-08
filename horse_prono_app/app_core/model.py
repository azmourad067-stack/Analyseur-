from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .features import make_features


class HorseRacingModel:
    """Two binary logistic models: win and place.

    Why logistic regression?
    * Horse racing outcomes are naturally binary for "win" and "placed" labels.
    * Logistic regression maps features to conditional probabilities via a logit link.
    * Regularization reduces overfitting when the historical sample is modest.
    * The output is probabilistic and interpretable through feature contributions.

    Important: per-horse probabilities are conditional estimates; the app normalizes
    win probabilities across a race so they form a race-level distribution. This is
    not a full joint ranking model such as Bradley-Terry/Plackett-Luce.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.win_model: Pipeline | None = None
        self.place_model: Pipeline | None = None
        self.metrics: dict[str, Any] = {}
        self.trained = False

    def _make_pipeline(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(
                C=self.config.c,
                penalty="l2",
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.config.random_state,
            )),
        ])

    def fit(self, history: pd.DataFrame) -> dict[str, Any]:
        from .data import normalize_history
        df = normalize_history(history)
        df = df.dropna(subset=["finish_position"])
        if len(df) < self.config.min_train_rows:
            raise ValueError(
                f"Historique insuffisant: {len(df)} lignes; il faut au moins {self.config.min_train_rows}."
            )
        features = make_features(df)
        y_win = (df["finish_position"] == 1).astype(int)
        y_place = (df["finish_position"] <= self.config.place_cutoff).astype(int)
        if y_win.nunique() < 2 or y_place.nunique() < 2:
            raise ValueError("Les cibles historiques ne contiennent pas assez de classes distinctes.")

        # Time-ordered validation is preferred to a random split to reduce look-ahead bias.
        order = df["race_date"].fillna(pd.Timestamp("2000-01-01")).sort_values().index
        split = max(int(len(order) * 0.8), 1)
        train_idx, valid_idx = order[:split], order[split:]
        self.win_model = self._make_pipeline().fit(features.loc[train_idx], y_win.loc[train_idx])
        self.place_model = self._make_pipeline().fit(features.loc[train_idx], y_place.loc[train_idx])

        metrics = {"train_rows": int(len(train_idx)), "validation_rows": int(len(valid_idx))}
        if len(valid_idx) >= 10:
            for name, mdl, y in [("win", self.win_model, y_win), ("place", self.place_model, y_place)]:
                proba = mdl.predict_proba(features.loc[valid_idx])[:, 1]
                if y.loc[valid_idx].nunique() >= 2:
                    metrics[f"{name}_logloss"] = float(log_loss(y.loc[valid_idx], proba, labels=[0, 1]))
                    metrics[f"{name}_auc"] = float(roc_auc_score(y.loc[valid_idx], proba))
        else:
            metrics["warning"] = "Échantillon de validation trop petit pour des métriques fiables."
        self.metrics = metrics
        self.trained = True
        return metrics

    @staticmethod
    def _normalize_race_probabilities(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        total = p.sum()
        if total <= 0:
            return np.repeat(1.0 / len(p), len(p))
        return p / total

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        from .data import normalize_race_input
        df = normalize_race_input(race)
        features = make_features(df)
        result = df.copy()
        result["win_probability_raw"] = np.nan
        result["place_probability"] = np.nan
        if self.trained and self.win_model is not None and self.place_model is not None:
            result["win_probability_raw"] = self.win_model.predict_proba(features)[:, 1]
            result["place_probability"] = self.place_model.predict_proba(features)[:, 1]
            result["win_probability"] = self._normalize_race_probabilities(result["win_probability_raw"].to_numpy())
            result["method"] = "Régression logistique entraînée"
        else:
            from .features import explain_contributions
            c = explain_contributions(df)
            raw = (
                0.36 * c["Cote"] +
                0.29 * c["Forme récente"] +
                0.14 * c["Taux victoire"] +
                0.13 * c["Taux placé"] +
                0.08 * c["Corde"]
            ).to_numpy()
            win = self._normalize_race_probabilities(raw)
            place = np.clip(0.55 * c["Taux placé"].to_numpy() + 0.30 * c["Forme récente"].to_numpy() + 0.15 * c["Cote"].to_numpy(), 0.02, 0.98)
            result["win_probability"] = win
            result["place_probability"] = place
            result["win_probability_raw"] = raw
            result["method"] = "Score probabiliste transparent (sans entraînement)"

        result["score_100"] = (result["win_probability"] * 100).round(2)
        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        return result

    def state(self) -> dict[str, Any]:
        return {"trained": self.trained, "metrics": self.metrics, "config": asdict(self.config)}
