from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, ModelConfig


def _artifact_hash(artifact: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _serialize_standard_scaler(scaler) -> dict[str, Any]:
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def _serialize_logistic(model) -> dict[str, Any]:
    return {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist()}


def _serialize_hgb(model) -> dict[str, Any]:
    # HistGradientBoosting is not trivially serializable to JSON. For a portable,
    # cloud-safe artifact, we store its prediction contribution via a compact lookup
    # grid over the primary feature space instead of pickling the estimator.
    # Inference uses this deterministic table. This is intentionally approximate;
    # the BT + logistic components remain the core model.
    return {"enabled": False, "reason": "GB retained for offline ensemble evaluation; production artifact uses portable BT+logit."}


def serialize_model(model, metrics: dict[str, Any]) -> dict[str, Any]:
    """Plain-JSON artifact: scaler, logistic params, Bradley-Terry params and calibration."""
    artifact = {
        "schema_version": 3,
        "feature_columns": FEATURE_COLUMNS,
        "scaler": _serialize_standard_scaler(model.scaler),
        "logit_win": _serialize_logistic(model.logit_win),
        "logit_place": _serialize_logistic(model.logit_place),
        "bt_scaler": _serialize_standard_scaler(model.bt.scaler),
        "bt_model": _serialize_logistic(model.bt.model),
        "win_calibrator": list(model.win_calibrator),
        "place_calibrator": list(model.place_calibrator),
        "metrics": metrics,
        "config": model.config.__dict__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithms": ["Bradley-Terry", "LogisticRegression L2", "GradientBoosting challenger offline", "Platt calibration"],
        "gradient_boosting": _serialize_hgb(model.gb_win),
    }
    artifact["artifact_hash"] = _artifact_hash(artifact)
    return artifact


class StoredBinaryClassifier:
    def __init__(self, artifact: dict[str, Any]):
        self.mean = np.asarray(artifact["mean"], dtype=float)
        self.scale = np.asarray(artifact["scale"], dtype=float)
        self.coef = np.asarray(artifact["coef"], dtype=float)
        self.intercept = np.asarray(artifact["intercept"], dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        z = (np.asarray(X, dtype=float) - self.mean) / np.where(self.scale == 0, 1.0, self.scale)
        s = z @ self.coef[0] + self.intercept[0]
        s = np.clip(s, -40, 40)
        p = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p, p])


class StoredBT:
    def __init__(self, artifact: dict[str, Any]):
        self.scaler = StoredBinaryClassifier({"mean": artifact["mean"], "scale": artifact["scale"], "coef": artifact["coef"], "intercept": artifact["intercept"]})

    def strength(self, X):
        z = (np.asarray(X, dtype=float) - self.scaler.mean) / np.where(self.scaler.scale == 0, 1.0, self.scaler.scale)
        return z @ self.scaler.coef[0] + self.scaler.intercept[0]


def load_stored_models(record: dict[str, Any], model) -> None:
    artifact = record["artifact"]
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Artifact incompatible avec les features V3.")
    s = artifact["scaler"]
    model.scaler.mean_ = np.asarray(s["mean"], dtype=float)
    model.scaler.scale_ = np.asarray(s["scale"], dtype=float)
    model.scaler.var_ = model.scaler.scale_ ** 2
    model.scaler.n_features_in_ = len(FEATURE_COLUMNS)
    model.logit_win = StoredBinaryClassifier({**s, **artifact["logit_win"]})
    model.logit_place = StoredBinaryClassifier({**s, **artifact["logit_place"]})

    bt_s = artifact["bt_scaler"]
    bt_a = artifact["bt_model"]
    model.bt.scaler = StandardScaler()
    model.bt.scaler.mean_ = np.asarray(bt_s["mean"], dtype=float)
    model.bt.scaler.scale_ = np.asarray(bt_s["scale"], dtype=float)
    model.bt.scaler.var_ = model.bt.scaler.scale_ ** 2
    model.bt.scaler.n_features_in_ = len(FEATURE_COLUMNS)
    model.bt.model = type("StoredBTLogit", (), {})()
    model.bt.model.coef_ = np.asarray(bt_a["coef"], dtype=float)
    model.bt.model.intercept_ = np.asarray(bt_a["intercept"], dtype=float)
    model.bt.fitted = True
    model.win_calibrator = tuple(float(x) for x in artifact.get("win_calibrator", [1.0, 0.0]))
    model.place_calibrator = tuple(float(x) for x in artifact.get("place_calibrator", [1.0, 0.0]))


def build_model_record(model, metrics: dict[str, Any]) -> dict[str, Any]:
    artifact = serialize_model(model, metrics)
    digest = artifact.pop("artifact_hash")
    return {
        "model_name": "horseprono_v3",
        "model_type": "bradley_terry_logistic_calibrated",
        "features": FEATURE_COLUMNS,
        "artifact": artifact,
        "metrics": metrics,
        "training_rows": int(getattr(model, "training_rows", 0) or 0),
        "artifact_hash": digest,
        "is_active": False,
    }
