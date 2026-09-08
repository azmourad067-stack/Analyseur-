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
from .model_store import load_stored_models


class HorseRacingModel:
    """Dual logistic model used by the offline training job.

    Why logistic regression?
    - Win/place are binary targets with a natural probabilistic interpretation.
    - L2 regularization limits overfitting when historical samples are noisy.
    - Coefficients are inspectable and the model can be serialized as plain JSON.

    This is a conditional probability model, not a complete joint ranking model.
    Race-level win probabilities are normalized after prediction so the field sums to 100%.
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
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.config.random_state,
            )),
        ])

    def fit(self, history: pd.DataFrame) -> dict[str, Any]:
        from .data import normalize_history
        df = normalize_history(history).dropna(subset=["finish_position"]).copy()
        if len(df) < self.config.min_train_rows:
            raise ValueError(f"Historique insuffisant: {len(df)} lignes; minimum {self.config.min_train_rows}.")

        features = make_features(df)
        y_win = (df["finish_position"] == 1).astype(int)
        y_place = (df["finish_position"] <= self.config.place_cutoff).astype(int)
        if y_win.nunique() < 2 or y_place.nunique() < 2:
            raise ValueError("Les cibles historiques ne contiennent pas assez de classes distinctes.")

        order = df["race_date"].fillna(pd.Timestamp("2000-01-01")).sort_values().index
        split = int(len(order) * 0.8)
        if split <= 0 or split >= len(order):
            raise ValueError("Impossible de construire une validation temporelle.")
        train_idx, valid_idx = order[:split], order[split:]

        # No random shuffle: future races must never leak into the training period.
        self.win_model = self._make_pipeline().fit(features.loc[train_idx], y_win.loc[train_idx])
        self.place_model = self._make_pipeline().fit(features.loc[train_idx], y_place.loc[train_idx])

        metrics: dict[str, Any] = {
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "validation_start": str(df.loc[valid_idx, "race_date"].min().date()) if len(valid_idx) else None,
            "validation_end": str(df.loc[valid_idx, "race_date"].max().date()) if len(valid_idx) else None,
        }
        if len(valid_idx) >= 10:
            for name, mdl, y in [("win", self.win_model, y_win), ("place", self.place_model, y_place)]:
                proba = mdl.predict_proba(features.loc[valid_idx])[:, 1]
                yv = y.loc[valid_idx]
                if yv.nunique() >= 2:
                    metrics[f"{name}_logloss"] = float(log_loss(yv, proba, labels=[0, 1]))
                    metrics[f"{name}_auc"] = float(roc_auc_score(yv, proba))
        else:
            metrics["warning"] = "Validation trop petite pour des métriques fiables."
        self.metrics = metrics
        self.trained = True
        return metrics

    @staticmethod
    def normalize_race_probabilities(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        total = float(p.sum())
        return p / total if total > 0 else np.repeat(1.0 / len(p), len(p))

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        from .data import normalize_race_input
        df = normalize_race_input(race)
        features = make_features(df)
        result = df.copy()

        if not self.trained or self.win_model is None or self.place_model is None:
            from .features import explain_contributions
            c = explain_contributions(df)
            raw = (0.36*c["Cote"] + 0.29*c["Forme récente"] + 0.14*c["Taux victoire"] +
                   0.13*c["Taux placé"] + 0.08*c["Corde"]).to_numpy()
            result["win_probability"] = self.normalize_race_probabilities(raw)
            result["place_probability"] = np.clip(
                0.55*c["Taux placé"].to_numpy() + 0.30*c["Forme récente"].to_numpy() + 0.15*c["Cote"].to_numpy(),
                0.02, 0.98,
            )
            result["method"] = "Score probabiliste transparent (sans entraînement)"
        else:
            result["win_probability"] = self.normalize_race_probabilities(self.win_model.predict_proba(features)[:, 1])
            result["place_probability"] = self.place_model.predict_proba(features)[:, 1]
            result["method"] = "Régression logistique entraînée"

        return self._finalize(result)

    def _finalize(self, result: pd.DataFrame) -> pd.DataFrame:
        result = result.copy()
        result["score_100"] = (result["win_probability"] * 100).round(2)
        result = result.sort_values(["win_probability", "place_probability"], ascending=False).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        return result

    @classmethod
    def from_stored_record(cls, record: dict[str, Any]) -> "HorseRacingModel":
        model = cls()
        model.win_model, model.place_model = load_stored_models(record)
        model.metrics = record.get("metrics", {}) or {}
        model.trained = True
        return model

    def state(self) -> dict[str, Any]:
        return {"trained": self.trained, "metrics": self.metrics, "config": asdict(self.config)}
