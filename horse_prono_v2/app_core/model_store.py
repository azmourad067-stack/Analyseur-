from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import FEATURE_COLUMNS, ModelConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def serialize_pipeline(pipeline) -> dict[str, Any]:
    """Serialize only numerical model parameters, never a pickle/joblib object."""
    scaler = pipeline.named_steps["scale"]
    logit = pipeline.named_steps["logit"]
    return {
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": logit.coef_.tolist(),
        "intercept": logit.intercept_.tolist(),
        "classes": logit.classes_.tolist(),
        "algorithm": "standard_scaler + logistic_regression_l2",
    }


def artifact_hash(win_artifact: dict[str, Any], place_artifact: dict[str, Any]) -> str:
    canonical = {
        "feature_columns": FEATURE_COLUMNS,
        "win_artifact": win_artifact,
        "place_artifact": place_artifact,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StoredBinaryClassifier:
    """Small safe inference wrapper reconstructed from numerical JSON only."""

    def __init__(self, artifact: dict[str, Any]):
        if artifact.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Artifact incompatible avec les features actuelles.")
        self.mean = np.asarray(artifact["mean"], dtype=float)
        self.scale = np.asarray(artifact["scale"], dtype=float)
        self.coef = np.asarray(artifact["coef"], dtype=float)
        self.intercept = np.asarray(artifact["intercept"], dtype=float)
        self.classes = np.asarray(artifact["classes"])

    def predict_proba(self, X: Any) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mean) / np.where(self.scale == 0, 1.0, self.scale)
        if self.coef.shape[0] != 1:
            raise ValueError("Seuls les modèles binaires sont supportés.")
        p1 = _sigmoid(Xs @ self.coef[0] + self.intercept[0])
        return np.column_stack([1 - p1, p1])


def build_model_record(win_pipeline, place_pipeline, metrics: dict[str, Any], config: ModelConfig) -> dict[str, Any]:
    win_artifact = serialize_pipeline(win_pipeline)
    place_artifact = serialize_pipeline(place_pipeline)
    digest = artifact_hash(win_artifact, place_artifact)
    artifact = {
        "feature_columns": FEATURE_COLUMNS,
        "win_artifact": win_artifact,
        "place_artifact": place_artifact,
        "config": {
            "min_train_rows": config.min_train_rows,
            "c": config.c,
            "place_cutoff": config.place_cutoff,
            "random_state": config.random_state,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "model_name": "horseprono_dual_logit",
        "model_type": "dual_logistic_regression",
        "feature_columns": FEATURE_COLUMNS,
        "artifact": artifact,
        "metrics": metrics,
        "artifact_hash": digest,
        "is_active": True,
    }


def load_stored_models(record: dict[str, Any]):
    artifact = record["artifact"]
    return StoredBinaryClassifier(artifact["win_artifact"]), StoredBinaryClassifier(artifact["place_artifact"])
