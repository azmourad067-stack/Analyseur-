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


SCHEMA_VERSION = 6

MODEL_FAMILY = (
    "HorseProno V4.2 "
    "production-aligned "
    "age-sex discipline-aware"
)


def _artifact_hash(
    artifact: dict[
        str,
        Any,
    ],
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


def _serialize_standard_scaler(
    scaler,
) -> dict[
    str,
    Any,
]:

    return {
        "mean":
            scaler.mean_.tolist(),

        "scale":
            scaler.scale_.tolist(),
    }


def _serialize_logistic(
    model,
) -> dict[
    str,
    Any,
]:

    return {
        "coef":
            model.coef_.tolist(),

        "intercept":
            model.intercept_.tolist(),
    }


def serialize_model(
    model,
    metrics: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    artifact = {

        "schema_version":
            SCHEMA_VERSION,

        "model_family":
            MODEL_FAMILY,

        "feature_columns":
            FEATURE_COLUMNS,

        "prediction_recipe": {

            "win": {

                "bt_weight":
                    float(
                        model
                        .win_bt_weight
                    ),

                "logit_weight":
                    float(
                        model
                        .win_logit_weight
                    ),

                "calibration":
                    "platt",
            },

            "place": {

                "model":
                    "logistic",

                "calibration":
                    "platt",
            },
        },

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
            "Bradley-Terry production",
            "LogisticRegression L2 production",
            "Platt calibration on production blend",
            "HistGradientBoosting challenger offline only",
            "Age-sex-discipline interactions",
        ],

        "gradient_boosting": {

            "enabled_in_production":
                False,

            "role":
                "offline_challenger_only",
        },
    }

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


class StoredLogisticClassifier:
    """
    X est DEJA standardisé par model.scaler.

    Il ne faut surtout pas appliquer le scaler
    une deuxième fois ici.
    """

    def __init__(
        self,
        artifact: dict[
            str,
            Any,
        ],
    ):

        self.coef_ = np.asarray(
            artifact[
                "coef"
            ],
            dtype=float,
        )

        self.intercept_ = np.asarray(
            artifact[
                "intercept"
            ],
            dtype=float,
        )

        self.classes_ = np.asarray(
            [
                0,
                1,
            ],
            dtype=int,
        )

    def predict_proba(
        self,
        X: Any,
    ) -> np.ndarray:

        values = np.asarray(
            X,
            dtype=float,
        )

        score = (
            values
            @ self.coef_[0]
            + self.intercept_[0]
        )

        score = np.clip(
            score,
            -40,
            40,
        )

        p = (
            1.0
            /
            (
                1.0
                + np.exp(
                    -score
                )
            )
        )

        return np.column_stack(
            [
                1.0 - p,
                p,
            ]
        )


def _load_standard_scaler(
    scaler: StandardScaler,
    artifact: dict[
        str,
        Any,
    ],
) -> None:

    scaler.mean_ = np.asarray(
        artifact[
            "mean"
        ],
        dtype=float,
    )

    scaler.scale_ = np.asarray(
        artifact[
            "scale"
        ],
        dtype=float,
    )

    scaler.var_ = (
        scaler.scale_
        ** 2
    )

    scaler.n_features_in_ = (
        len(
            FEATURE_COLUMNS
        )
    )


def load_stored_models(
    record: dict[
        str,
        Any,
    ],
    model,
) -> None:

    artifact = record[
        "artifact"
    ]

    if (
        artifact.get(
            "feature_columns"
        )
        != FEATURE_COLUMNS
    ):

        raise ValueError(
            "Artefact incompatible "
            "avec les features "
            "HorseProno V4.2."
        )

    if (
        artifact.get(
            "schema_version"
        )
        != SCHEMA_VERSION
    ):

        raise ValueError(
            "Artefact incompatible "
            "avec HorseProno V4.2 "
            "production-aligned."
        )

    recipe = artifact.get(
        "prediction_recipe",
        {},
    )

    win_recipe = recipe.get(
        "win",
        {},
    )

    model.win_bt_weight = float(
        win_recipe.get(
            "bt_weight",
            0.60,
        )
    )

    model.win_logit_weight = float(
        win_recipe.get(
            "logit_weight",
            0.40,
        )
    )

    if not np.isclose(
        (
            model.win_bt_weight
            + model.win_logit_weight
        ),
        1.0,
        atol=1e-12,
    ):

        raise ValueError(
            "Poids de blend invalides "
            "dans l'artefact."
        )

    # ========================================================
    # LOGISTIC SCALER
    # ========================================================

    _load_standard_scaler(
        model.scaler,
        artifact[
            "scaler"
        ],
    )

    # Pas de deuxième scaling ici.
    model.logit_win = (
        StoredLogisticClassifier(
            artifact[
                "logit_win"
            ]
        )
    )

    model.logit_place = (
        StoredLogisticClassifier(
            artifact[
                "logit_place"
            ]
        )
    )

    # ========================================================
    # BRADLEY TERRY
    # ========================================================

    model.bt.scaler = (
        StandardScaler()
    )

    _load_standard_scaler(
        model.bt.scaler,
        artifact[
            "bt_scaler"
        ],
    )

    model.bt.model = type(
        "StoredBTLogit",
        (),
        {},
    )()

    model.bt.model.coef_ = (
        np.asarray(
            artifact[
                "bt_model"
            ][
                "coef"
            ],
            dtype=float,
        )
    )

    model.bt.model.intercept_ = (
        np.asarray(
            artifact[
                "bt_model"
            ][
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


def build_model_record(
    model,
    metrics: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    artifact = serialize_model(
        model,
        metrics,
    )

    digest = artifact.pop(
        "artifact_hash"
    )

    return {

        "model_name":
            "horseprono_v4_2",

        "model_type":
            (
                "production_aligned_"
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
