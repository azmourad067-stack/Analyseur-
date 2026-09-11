from __future__ import annotations

import hashlib
import json

from datetime import (
    datetime,
    timezone,
)

from typing import Any

import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
)

from .config import (
    FEATURE_COLUMNS,
)


# ============================================================
# HASH
# ============================================================

def _artifact_hash(
    artifact: dict[str, Any],
) -> str:

    payload = json.dumps(
        artifact,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        payload.encode()
    ).hexdigest()


# ============================================================
# SERIALISATION
# ============================================================

def _serialize_standard_scaler(
    scaler,
) -> dict[str, Any]:

    return {
        "mean":
            scaler.mean_.tolist(),

        "scale":
            scaler.scale_.tolist(),
    }


def _serialize_logistic(
    model,
) -> dict[str, Any]:

    return {
        "coef":
            model.coef_.tolist(),

        "intercept":
            model.intercept_.tolist(),
    }


def _serialize_hgb(
    model,
) -> dict[str, Any]:
    """
    HistGradientBoosting reste utilisé pour l'évaluation
    offline mais n'est pas sérialisé dans l'artefact portable.
    """

    return {
        "enabled":
            False,

        "reason": (
            "Gradient Boosting retained for offline "
            "evaluation; production artifact uses "
            "portable Bradley-Terry + logistic models."
        ),
    }


def serialize_model(
    model,
    metrics: dict[str, Any],
) -> dict[str, Any]:

    artifact = {

        "schema_version":
    5,

"model_family":
    "HorseProno V4.1 age-sex discipline-aware",

        "feature_columns":
            FEATURE_COLUMNS,

        "scaler":
            _serialize_standard_scaler(
                model.scaler
            ),

        "logit_win":
            _serialize_logistic(
                model.logit_win
            ),

        "logit_place":
            _serialize_logistic(
                model.logit_place
            ),

        "bt_scaler":
            _serialize_standard_scaler(
                model.bt.scaler
            ),

        "bt_model":
            _serialize_logistic(
                model.bt.model
            ),

        "win_calibrator":
            list(
                model.win_calibrator
            ),

        "place_calibrator":
            list(
                model.place_calibrator
            ),

        "metrics":
            metrics,

        "config":
            model.config.__dict__,

        "algorithms": [
            "Bradley-Terry",
            "LogisticRegression L2",
            "GradientBoosting challenger offline",
            "Platt calibration",
            "Discipline-aware interactions",
        ],

        "gradient_boosting":
            _serialize_hgb(
                model.gb_win
            ),
    }

    # Hash stable : created_at n'entre pas
    # dans la signature du modèle.
    digest = _artifact_hash(
        artifact
    )

    artifact[
        "artifact_hash"
    ] = digest

    artifact[
        "created_at"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    return artifact


# ============================================================
# CLASSIFICATEUR LOGISTIQUE PORTABLE
# ============================================================

class StoredBinaryClassifier:

    def __init__(
        self,
        artifact: dict[str, Any],
    ):

        self.mean = np.asarray(
            artifact[
                "mean"
            ],
            dtype=float,
        )

        self.scale = np.asarray(
            artifact[
                "scale"
            ],
            dtype=float,
        )

        self.coef = np.asarray(
            artifact[
                "coef"
            ],
            dtype=float,
        )

        self.intercept = np.asarray(
            artifact[
                "intercept"
            ],
            dtype=float,
        )

    def predict_proba(
        self,
        X: Any,
    ) -> np.ndarray:

        values = np.asarray(
            X,
            dtype=float,
        )

        z = (
            values
            - self.mean
        ) / np.where(
            self.scale == 0,
            1.0,
            self.scale,
        )

        score = (
            z
            @ self.coef[0]
            + self.intercept[0]
        )

        score = np.clip(
            score,
            -40,
            40,
        )

        p = (
            1.0
            / (
                1.0
                + np.exp(
                    -score
                )
            )
        )

        return np.column_stack(
            [
                1 - p,
                p,
            ]
        )


# ============================================================
# BT PORTABLE
# ============================================================

class StoredBT:

    def __init__(
        self,
        artifact: dict[str, Any],
    ):

        self.scaler = (
            StoredBinaryClassifier(
                {
                    "mean":
                        artifact[
                            "mean"
                        ],

                    "scale":
                        artifact[
                            "scale"
                        ],

                    "coef":
                        artifact[
                            "coef"
                        ],

                    "intercept":
                        artifact[
                            "intercept"
                        ],
                }
            )
        )

    def strength(
        self,
        X,
    ):

        values = np.asarray(
            X,
            dtype=float,
        )

        z = (
            values
            - self.scaler.mean
        ) / np.where(
            self.scaler.scale == 0,
            1.0,
            self.scaler.scale,
        )

        return (
            z
            @ self.scaler.coef[0]
            + self.scaler.intercept[0]
        )


# ============================================================
# CHARGEMENT MODELE
# ============================================================

def load_stored_models(
    record: dict[str, Any],
    model,
) -> None:

    artifact = record[
        "artifact"
    ]

    artifact_features = (
        artifact.get(
            "feature_columns"
        )
    )

    if (
        artifact_features
        != FEATURE_COLUMNS
    ):

        raise ValueError(
            "Artefact incompatible avec "
            "les features HorseProno V4.1."
        )

    schema_version = (
        artifact.get(
            "schema_version"
        )
    )

    if schema_version != 5:

        raise ValueError(
            "Artefact incompatible avec "
            "le schéma HorseProno V4."
        )

    # ========================================================
    # SCALER LOGISTIQUE
    # ========================================================

    scaler_artifact = (
        artifact[
            "scaler"
        ]
    )

    model.scaler.mean_ = (
        np.asarray(
            scaler_artifact[
                "mean"
            ],
            dtype=float,
        )
    )

    model.scaler.scale_ = (
        np.asarray(
            scaler_artifact[
                "scale"
            ],
            dtype=float,
        )
    )

    model.scaler.var_ = (
        model.scaler.scale_
        ** 2
    )

    model.scaler.n_features_in_ = (
        len(
            FEATURE_COLUMNS
        )
    )

    # ========================================================
    # LOGISTIQUE WIN / PLACE
    # ========================================================

    model.logit_win = (
        StoredBinaryClassifier(
            {
                **scaler_artifact,
                **artifact[
                    "logit_win"
                ],
            }
        )
    )

    model.logit_place = (
        StoredBinaryClassifier(
            {
                **scaler_artifact,
                **artifact[
                    "logit_place"
                ],
            }
        )
    )

    # ========================================================
    # BRADLEY-TERRY
    # ========================================================

    bt_scaler = (
        artifact[
            "bt_scaler"
        ]
    )

    bt_model = (
        artifact[
            "bt_model"
        ]
    )

    model.bt.scaler = (
        StandardScaler()
    )

    model.bt.scaler.mean_ = (
        np.asarray(
            bt_scaler[
                "mean"
            ],
            dtype=float,
        )
    )

    model.bt.scaler.scale_ = (
        np.asarray(
            bt_scaler[
                "scale"
            ],
            dtype=float,
        )
    )

    model.bt.scaler.var_ = (
        model.bt.scaler.scale_
        ** 2
    )

    model.bt.scaler.n_features_in_ = (
        len(
            FEATURE_COLUMNS
        )
    )

    model.bt.model = type(
        "StoredBTLogit",
        (),
        {},
    )()

    model.bt.model.coef_ = (
        np.asarray(
            bt_model[
                "coef"
            ],
            dtype=float,
        )
    )

    model.bt.model.intercept_ = (
        np.asarray(
            bt_model[
                "intercept"
            ],
            dtype=float,
        )
    )

    model.bt.fitted = True

    # ========================================================
    # CALIBRATION
    # ========================================================

    model.win_calibrator = tuple(
        float(x)
        for x in artifact.get(
            "win_calibrator",
            [
                1.0,
                0.0,
            ],
        )
    )

    model.place_calibrator = tuple(
        float(x)
        for x in artifact.get(
            "place_calibrator",
            [
                1.0,
                0.0,
            ],
        )
    )


# ============================================================
# ENREGISTREMENT SUPABASE
# ============================================================

def build_model_record(
    model,
    metrics: dict[str, Any],
) -> dict[str, Any]:

    artifact = serialize_model(
        model,
        metrics,
    )

    digest = artifact.pop(
        "artifact_hash"
    )

    return {

        "model_name":
            "horseprono_v4_1",

        "model_type":
            (
                "age_sex_discipline_aware_"
                "bradley_terry_"
                "logistic_calibrated"
            ),

        "features":
            FEATURE_COLUMNS,

        "artifact":
            artifact,

        "metrics":
            metrics,

        "training_rows":
            int(
                getattr(
                    model,
                    "training_rows",
                    0,
                )
                or 0
            ),

        "artifact_hash":
            digest,

        "is_active":
            False,
    }
