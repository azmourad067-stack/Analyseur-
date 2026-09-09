from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, ModelConfig
from .features import build_temporal_training_frame, make_features
from .model_store import load_stored_models


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    total = e.sum()
    return e / total if total else np.repeat(1 / len(x), len(x))


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        out[i] = _softmax(scores[i])
    return out


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


class BradleyTerryModel:
    """Pairwise Bradley-Terry model.

    For each race, training constructs pairwise observations. The model estimates
    a latent strength s_i such that P(i beats j)=sigmoid(s_i-s_j).
    This directly addresses the *within-race ranking* structure of horse racing,
    unlike an independent binary classifier.
    """

    def __init__(self, c: float = 0.5, random_state: int = 42, max_pairs_per_race: int = 180):
        self.c = c
        self.random_state = random_state
        self.max_pairs_per_race = max_pairs_per_race
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=c, max_iter=2000, random_state=random_state)
        self.fitted = False

    def fit(self, X: pd.DataFrame, ranks: pd.Series, race_ids: pd.Series) -> "BradleyTerryModel":
        pair_X, pair_y = [], []
        rng = np.random.default_rng(self.random_state)
        all_rows = pd.DataFrame({"race": race_ids.astype(str), "rank": ranks.to_numpy()}, index=X.index)
        for race_id, idx in all_rows.groupby("race").groups.items():
            idx = list(idx)
            if len(idx) < 2:
                continue
            pairs = [(a, b) for pos, a in enumerate(idx) for b in idx[pos + 1:]]
            if len(pairs) > self.max_pairs_per_race:
                take = rng.choice(len(pairs), self.max_pairs_per_race, replace=False)
                pairs = [pairs[int(i)] for i in take]
            for a, b in pairs:
                ya, yb = int(ranks.loc[a]), int(ranks.loc[b])
                # Only finished ranking information is used: higher rank beats lower rank.
                if ya == yb:
                    continue
                xa = X.loc[a].to_numpy(dtype=float)
                xb = X.loc[b].to_numpy(dtype=float)
                diff = xa - xb if ya < yb else xb - xa
                pair_X.extend([diff, -diff])
                pair_y.extend([1, 0])
        if not pair_X:
            raise ValueError("Impossible de construire les paires Bradley-Terry.")
        mat = np.asarray(pair_X)
        self.scaler.fit(mat)
        self.model.fit(self.scaler.transform(mat), np.asarray(pair_y))
        self.fitted = True
        return self

    def strength(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Bradley-Terry non entraîné")
        z = self.scaler.transform(X.to_numpy(dtype=float))
        return (z @ self.model.coef_[0]) + self.model.intercept_[0]

    def race_probabilities(self, X: pd.DataFrame, race_ids: pd.Series) -> np.ndarray:
        scores = self.strength(X)
        result = np.zeros(len(X), dtype=float)
        tmp = pd.DataFrame({"race": race_ids.astype(str).to_numpy(), "score": scores, "pos": np.arange(len(X))})
        for _, grp in tmp.groupby("race"):
            result[grp["pos"].to_numpy()] = _softmax(grp["score"].to_numpy())
        return result


class HorseRacingModel:
    """V3 ensemble: Bradley-Terry + calibrated logistic + gradient boosting.

    The architecture combines:
      1) Bradley-Terry for *relative race strength*;
      2) regularized logistic regression for interpretable conditional probabilities;
      3) gradient boosting for modest nonlinear interactions.

    Probabilities are calibrated on a chronological holdout and evaluated on a later
    untouched test period. This is deliberately time-aware because future race outcomes
    must never influence earlier features or validation.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.logit_win = LogisticRegression(C=self.config.logit_c, max_iter=2500, random_state=self.config.random_state, class_weight="balanced")
        self.logit_place = LogisticRegression(C=self.config.logit_c, max_iter=2500, random_state=self.config.random_state, class_weight="balanced")
        self.gb_win = HistGradientBoostingClassifier(max_depth=self.config.gb_max_depth, learning_rate=self.config.gb_learning_rate, max_iter=self.config.gb_n_estimators, random_state=self.config.random_state)
        self.gb_place = HistGradientBoostingClassifier(max_depth=self.config.gb_max_depth, learning_rate=self.config.gb_learning_rate, max_iter=self.config.gb_n_estimators, random_state=self.config.random_state)
        self.bt = BradleyTerryModel(self.config.bt_c, self.config.random_state, self.config.bt_max_pairs_per_race)
        self.win_calibrator: tuple[float, float] = (1.0, 0.0)
        self.place_calibrator: tuple[float, float] = (1.0, 0.0)
        self.metrics: dict[str, Any] = {}
        self.trained = False
        self.portable_artifact = False

    @staticmethod
    def _split_by_race_time(df: pd.DataFrame, train_fraction: float, cal_fraction: float):
        races = df[["race_id", "race_date"]].drop_duplicates().sort_values(["race_date", "race_id"]).reset_index(drop=True)
        n = len(races)
        a = max(1, int(n * train_fraction))
        b = max(a + 1, int(n * (train_fraction + cal_fraction)))
        b = min(b, n - 1)
        train_races = set(races.iloc[:a]["race_id"].astype(str))
        cal_races = set(races.iloc[a:b]["race_id"].astype(str))
        test_races = set(races.iloc[b:]["race_id"].astype(str))
        return train_races, cal_races, test_races

    def _fit_base(self, train: pd.DataFrame) -> None:
        X = make_features(train, already_temporal=True)
        y_win = (train["finish_position"] == 1).astype(int)
        y_place = (train["finish_position"] <= self.config.place_cutoff).astype(int)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.logit_win.fit(Xs, y_win)
        self.logit_place.fit(Xs, y_place)
        self.gb_win.fit(X, y_win)
        self.gb_place.fit(X, y_place)
        self.bt.fit(X, train["finish_position"], train["race_id"])

    def _component_predictions(self, df: pd.DataFrame):
        X = make_features(df, already_temporal=True)
        Xs = self.scaler.transform(X)
        p_logit_win = self._normalize_by_race(self.logit_win.predict_proba(Xs)[:, 1], df["race_id"])
        p_bt = self.bt.race_probabilities(X, df["race_id"])
        p_logit_place = self.logit_place.predict_proba(Xs)[:, 1]
        out = {"bt_win": p_bt, "logit_win": p_logit_win, "logit_place": p_logit_place}
        if not self.portable_artifact:
            out["gb_win"] = self._normalize_by_race(self.gb_win.predict_proba(X)[:, 1], df["race_id"])
            out["gb_place"] = self.gb_place.predict_proba(X)[:, 1]
        return out

    def _raw_predictions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        c = self._component_predictions(df)
        if self.portable_artifact:
            raw_win = 0.60 * c["bt_win"] + 0.40 * c["logit_win"]
            raw_place = c["logit_place"]
        else:
            raw_ind = self._normalize_by_race(0.70 * c["logit_win"] + 0.30 * c["gb_win"], df["race_id"])
            raw_win = 0.55 * c["bt_win"] + 0.45 * raw_ind
            raw_place = 0.65 * c["logit_place"] + 0.35 * c["gb_place"]
        return raw_win, raw_place

    @staticmethod
    def _normalize_by_race(values: np.ndarray, race_ids: pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = np.zeros_like(arr)
        for _, idx in pd.Series(race_ids.astype(str).to_numpy()).groupby(pd.Series(race_ids.astype(str).to_numpy())).groups.items():
            vals = arr[np.asarray(idx)]
            total = vals.sum()
            out[np.asarray(idx)] = vals / total if total > 0 else 1 / len(vals)
        return out

    @staticmethod
    def _fit_calibrator(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        # Platt scaling on logit(raw); only two numbers are stored in production.
        z = np.log(_clip_prob(raw) / (1 - _clip_prob(raw))).reshape(-1, 1)
        lr = LogisticRegression(C=10.0, max_iter=1000)
        lr.fit(z, y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])

    @staticmethod
    def _apply_calibrator(raw: np.ndarray, params: tuple[float, float]) -> np.ndarray:
        z = np.log(_clip_prob(raw) / (1 - _clip_prob(raw)))
        a, b = params
        q = 1 / (1 + np.exp(-np.clip(a * z + b, -40, 40)))
        return q

    def fit(self, history: pd.DataFrame) -> dict[str, Any]:
        self.training_rows = len(history)
        temporal = build_temporal_training_frame(history)
        temporal = temporal.dropna(subset=["finish_position", "race_id", "race_date"]).copy()
        if len(temporal) < self.config.min_train_rows:
            raise ValueError(f"Historique insuffisant: {len(temporal)} lignes; minimum {self.config.min_train_rows}.")
        tr, cal, te = self._split_by_race_time(temporal, self.config.train_fraction, self.config.calibration_fraction)
        train = temporal[temporal["race_id"].astype(str).isin(tr)].copy()
        calibration = temporal[temporal["race_id"].astype(str).isin(cal)].copy()
        test = temporal[temporal["race_id"].astype(str).isin(te)].copy()
        if train.empty or calibration.empty or test.empty:
            raise ValueError("Découpage temporel insuffisant : il faut des courses train/calibration/test.")

        self._fit_base(train)
        raw_cal_win, raw_cal_place = self._raw_predictions(calibration)
        self.win_calibrator = self._fit_calibrator(raw_cal_win, (calibration["finish_position"] == 1).astype(int).to_numpy())
        self.place_calibrator = self._fit_calibrator(raw_cal_place, (calibration["finish_position"] <= self.config.place_cutoff).astype(int).to_numpy())

        raw_test_win, raw_test_place = self._raw_predictions(test)
        p_win = self._normalize_by_race(self._apply_calibrator(raw_test_win, self.win_calibrator), test["race_id"])
        p_place = self._apply_calibrator(raw_test_place, self.place_calibrator)
        y_win = (test["finish_position"] == 1).astype(int).to_numpy()
        y_place = (test["finish_position"] <= self.config.place_cutoff).astype(int).to_numpy()
        components = self._component_predictions(test)
        market = self._normalize_by_race((1 / pd.to_numeric(test["odds"], errors="coerce").clip(lower=1.01).fillna(100)).to_numpy(), test["race_id"])
        metrics = {
            "train_rows": int(len(train)), "calibration_rows": int(len(calibration)), "test_rows": int(len(test)),
            "train_races": int(train["race_id"].nunique()), "test_races": int(test["race_id"].nunique()),
            "test_start": str(test["race_date"].min().date()), "test_end": str(test["race_date"].max().date()),
            "win_logloss": float(log_loss(y_win, p_win, labels=[0, 1])),
            "win_brier": float(brier_score_loss(y_win, p_win)),
            "place_logloss": float(log_loss(y_place, p_place, labels=[0, 1])),
            "place_brier": float(brier_score_loss(y_place, p_place)),
            "market_win_logloss": float(log_loss(y_win, market, labels=[0, 1])),
            "market_win_brier": float(brier_score_loss(y_win, market)),
            "bt_win_logloss": float(log_loss(y_win, components["bt_win"], labels=[0, 1])),
            "logit_win_logloss": float(log_loss(y_win, components["logit_win"], labels=[0, 1])),
        }
        if "gb_win" in components:
            metrics["gb_win_logloss"] = float(log_loss(y_win, components["gb_win"], labels=[0, 1]))
        if len(np.unique(y_win)) == 2:
            metrics["win_auc"] = float(roc_auc_score(y_win, p_win))
        if len(np.unique(y_place)) == 2:
            metrics["place_auc"] = float(roc_auc_score(y_place, p_place))
        backtest = self.backtest_metrics(test, p_win)
        metrics.update(backtest)
        self.metrics = metrics
        self.trained = True
        return metrics

    def backtest_metrics(self, test: pd.DataFrame, win_probs: np.ndarray) -> dict[str, Any]:
        x = test.copy()
        x["p"] = win_probs
        rows = []
        for race_id, g in x.groupby("race_id", sort=False):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            market = 1.0 / pd.to_numeric(g["odds"], errors="coerce")
            market = market.fillna(0)
            market = market / market.sum() if market.sum() > 0 else market
            rows.append({
                "race_id": race_id,
                "top1_hit": int(top["finish_position"] == 1),
                "top3_hit": int((g.head(3)["finish_position"] <= 3).any()),
                "top1_model_prob": float(top["p"]),
                "top1_market_prob": float(market.loc[top.name]) if top.name in market.index else np.nan,
                "top1_odds": float(top["odds"]) if pd.notna(top["odds"]) else np.nan,
                "top1_return": (float(top["odds"]) if pd.notna(top["odds"]) else 0.0) if top["finish_position"] == 1 else 0.0,
            })
        r = pd.DataFrame(rows)
        if r.empty:
            return {}
        stake = len(r)
        gross = r["top1_return"].sum()
        return {
            "backtest_races": int(len(r)),
            "top1_accuracy": float(r["top1_hit"].mean()),
            "top3_coverage": float(r["top3_hit"].mean()),
            "top1_market_prob_mean": float(r["top1_market_prob"].mean()),
            "flat_stake_roi_top1": float((gross - stake) / stake),
        }

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        from .data import normalize_race_input
        df = normalize_race_input(race).copy()
        if df.empty:
            return df
        if not self.trained:
            # Transparent cold start: market/form score, not disguised as a trained model.
            odds = 1 / pd.to_numeric(df["odds"], errors="coerce").clip(lower=1.01)
            if "recent_form_score" in df.columns:
                form = pd.to_numeric(df["recent_form_score"], errors="coerce").fillna(0.45)
            else:
                form = pd.Series(0.45, index=df.index, dtype=float)
            raw = 0.70 * odds + 0.30 * form
            win = self._normalize_by_race(raw.to_numpy(), pd.Series(df["race_id"].astype(str)))
            place = np.clip(0.55 * form.to_numpy() + 0.45 * odds.to_numpy(), 0.02, 0.98)
            method = "Score de repli transparent"
        else:
            temporal_like = df.copy()
            raw_win, raw_place = self._raw_predictions(temporal_like)
            win = self._normalize_by_race(self._apply_calibrator(raw_win, self.win_calibrator), df["race_id"])
            place = self._apply_calibrator(raw_place, self.place_calibrator)
            method = "BT + Logistique calibrés (GB challengers offline)"

        result = df.copy()
        result["win_probability"] = win
        result["place_probability"] = place
        result["market_probability"] = self._normalize_by_race((1 / pd.to_numeric(df["odds"], errors="coerce").clip(lower=1.01).fillna(20.0)).to_numpy(), pd.Series(df["race_id"].astype(str)))
        result["expected_value"] = result["win_probability"] * pd.to_numeric(df["odds"], errors="coerce").fillna(0) - 1.0
        result["method"] = method
        result["score_100"] = result["win_probability"] * 100
        result = result.sort_values(["win_probability", "place_probability"], ascending=False).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        result["recommendation"] = np.where(result["expected_value"] > 0.10, "Valeur théorique", "Neutre")
        result["explanation"] = result.apply(lambda r: self._explain_row(r), axis=1)
        return result

    @staticmethod
    def _explain_row(r: pd.Series) -> str:
        drivers = []
        if float(r.get("win_probability", 0)) > float(r.get("market_probability", 0)) * 1.15:
            drivers.append("modèle > marché")
        if float(r.get("recent_form_score", 0)) >= 0.65:
            drivers.append("forme récente")
        if float(r.get("horse_win_rate", 0)) >= 0.12:
            drivers.append("taux de victoire cheval")
        if float(r.get("jockey_place_rate", 0)) >= 0.35:
            drivers.append("jockey fiable")
        if float(r.get("trainer_place_rate", 0)) >= 0.35:
            drivers.append("entraîneur fiable")
        return ", ".join(drivers[:3]) or "signal équilibré"

    @classmethod
    def from_stored_record(cls, record: dict[str, Any]) -> "HorseRacingModel":
        model = cls()
        load_stored_models(record, model)
        model.metrics = record.get("metrics", {}) or {}
        model.trained = True
        model.portable_artifact = True
        return model

    def state(self) -> dict[str, Any]:
        return {"trained": self.trained, "metrics": self.metrics, "config": asdict(self.config)}
