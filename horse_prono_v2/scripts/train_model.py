from __future__ import annotations

import json
import os
import sys

from datetime import (
    datetime,
    timezone,
)

import numpy as np

from sklearn.metrics import (
    log_loss,
)


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            __file__
        )
    ),
)


from app_core.backtest_v2 import (
    build_advanced_backtest,
)

from app_core.db import (
    activate_model,
    get_latest_active_model,
    get_training_history,
    save_backtest_run,
    save_model_version,
)

from app_core.features import (
    build_temporal_training_frame,
)

from app_core.model import (
    HorseRacingModel,
)

from app_core.model_store import (
    build_model_record,
)


PARITY_TOLERANCE = 1e-10


def _get_test_frame(
    model: HorseRacingModel,
    history,
):

    temporal = (
        build_temporal_training_frame(
            history
        )
    )

    temporal = (
        temporal
        .dropna(
            subset=[
                "finish_position",
                "race_id",
                "race_date",
            ]
        )
        .copy()
    )

    _, _, test_races = (
        model._split_by_race_time(
            temporal,
            model.config.train_fraction,
            model.config.calibration_fraction,
        )
    )

    return temporal[
        temporal[
            "race_id"
        ]
        .astype(str)
        .isin(
            test_races
        )
    ].copy()


def _final_probabilities(
    model: HorseRacingModel,
    test,
):

    raw_win, raw_place = (
        model._raw_predictions(
            test
        )
    )

    p_win = (
        model._normalize_by_race(
            model._apply_calibrator(
                raw_win,
                model.win_calibrator,
            ),
            test[
                "race_id"
            ],
        )
    )

    p_place = (
        model._apply_calibrator(
            raw_place,
            model.place_calibrator,
        )
    )

    return (
        p_win,
        p_place,
    )


def verify_artifact_parity(
    trained_model: HorseRacingModel,
    provisional_record: dict,
    history,
) -> dict:

    test = _get_test_frame(
        trained_model,
        history,
    )

    if test.empty:

        raise RuntimeError(
            "Parity guard impossible : "
            "holdout vide."
        )

    offline_win, offline_place = (
        _final_probabilities(
            trained_model,
            test,
        )
    )

    # Recharge EXACTEMENT ce qui serait
    # utilisé par Streamlit.
    stored_model = (
        HorseRacingModel
        .from_stored_record(
            provisional_record
        )
    )

    stored_win, stored_place = (
        _final_probabilities(
            stored_model,
            test,
        )
    )

    max_abs_win = float(
        np.max(
            np.abs(
                offline_win
                - stored_win
            )
        )
    )

    max_abs_place = float(
        np.max(
            np.abs(
                offline_place
                - stored_place
            )
        )
    )

    y_win = (
        test[
            "finish_position"
        ]
        .eq(1)
        .astype(int)
        .to_numpy()
    )

    offline_logloss = float(
        log_loss(
            y_win,
            offline_win,
            labels=[
                0,
                1,
            ],
        )
    )

    stored_logloss = float(
        log_loss(
            y_win,
            stored_win,
            labels=[
                0,
                1,
            ],
        )
    )

    logloss_gap = abs(
        offline_logloss
        - stored_logloss
    )

    # Vérifie aussi le favori modèle
    # de chaque course.
    top1_mismatches = 0

    tmp = test[
        [
            "race_id"
        ]
    ].copy()

    tmp[
        "offline"
    ] = offline_win

    tmp[
        "stored"
    ] = stored_win

    for _, group in tmp.groupby(
        "race_id",
        sort=False,
    ):

        offline_top = (
            group[
                "offline"
            ]
            .idxmax()
        )

        stored_top = (
            group[
                "stored"
            ]
            .idxmax()
        )

        if (
            offline_top
            != stored_top
        ):

            top1_mismatches += 1

    parity = {

        "tolerance":
            PARITY_TOLERANCE,

        "test_rows":
            int(
                len(test)
            ),

        "test_races":
            int(
                test[
                    "race_id"
                ]
                .nunique()
            ),

        "max_abs_win_probability_gap":
            max_abs_win,

        "max_abs_place_probability_gap":
            max_abs_place,

        "offline_win_logloss":
            offline_logloss,

        "stored_win_logloss":
            stored_logloss,

        "win_logloss_gap":
            float(
                logloss_gap
            ),

        "top1_mismatches":
            int(
                top1_mismatches
            ),
    }

    parity[
        "passed"
    ] = bool(
        (
            max_abs_win
            <= PARITY_TOLERANCE
        )
        and
        (
            max_abs_place
            <= PARITY_TOLERANCE
        )
        and
        (
            logloss_gap
            <= PARITY_TOLERANCE
        )
        and
        (
            top1_mismatches
            == 0
        )
    )

    if not parity[
        "passed"
    ]:

        raise RuntimeError(
            "PRODUCTION PARITY FAILED: "
            +
            json.dumps(
                parity,
                ensure_ascii=False,
            )
        )

    return parity


def main() -> None:

    started_at = datetime.now(
        timezone.utc
    )

    print(
        "=" * 70
    )

    print(
        "HORSEPRONO V4.2 "
        "- PRODUCTION ALIGNED TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        "Chargement de l'historique "
        "depuis Supabase..."
    )

    history = (
        get_training_history()
    )

    if history.empty:

        raise SystemExit(
            "Aucun historique terminé "
            "dans Supabase."
        )

    print(
        f"Historique chargé : "
        f"{len(history)} participants"
    )

    if (
        "race_id"
        in history.columns
    ):

        print(
            "Courses disponibles : "
            f"{history['race_id'].nunique()}"
        )

    if (
        "race_date"
        in history.columns
    ):

        print(
            "Période : "
            f"{history['race_date'].min()} "
            "-> "
            f"{history['race_date'].max()}"
        )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print(
        "Entraînement du modèle "
        "production-aligned..."
    )

    model = HorseRacingModel()

    metrics = model.fit(
        history
    )

    print()
    print(
        "Entraînement terminé."
    )

    print(
        "Win log-loss production :",
        metrics.get(
            "win_logloss"
        ),
    )

    print(
        "Market win log-loss      :",
        metrics.get(
            "market_win_logloss"
        ),
    )

    print(
        "GB challenger log-loss   :",
        metrics.get(
            "gb_win_logloss_challenger"
        ),
    )

    # ========================================================
    # BACKTEST V2
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "BACKTEST V2 "
        "- PRODUCTION ALIGNED"
    )

    print(
        "=" * 70
    )

    advanced = (
        build_advanced_backtest(
            model,
            history,
        )
    )

    metrics[
        "backtest_v2"
    ] = advanced

    model.metrics = metrics

    if advanced:

        overall = advanced.get(
            "overall",
            {},
        )

        print(
            "Courses test :",
            overall.get(
                "races"
            ),
        )

        print(
            "Top 1 :",
            overall.get(
                "top1_accuracy"
            ),
        )

        print(
            "Gagnant Top 3 :",
            overall.get(
                "winner_in_top3"
            ),
        )

        print(
            "Gagnant Top 5 :",
            overall.get(
                "winner_in_top5"
            ),
        )

        print(
            "Trio complet Top 3 :",
            overall.get(
                "trio_complete_top3"
            ),
        )

        edge_2pp = (
            advanced
            .get(
                "edge_strategies",
                {},
            )
            .get(
                "all_runners",
                {},
            )
            .get(
                "edge_2pp",
                {},
            )
        )

        print(
            "All-runners edge >= 2pp :",
            edge_2pp,
        )

    # ========================================================
    # ARTEFACT PARITY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PRODUCTION ARTIFACT "
        "PARITY GUARD"
    )

    print(
        "=" * 70
    )

    provisional_record = (
        build_model_record(
            model,
            metrics,
        )
    )

    parity = (
        verify_artifact_parity(
            model,
            provisional_record,
            history,
        )
    )

    print(
        json.dumps(
            parity,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "✅ Artefact sauvegardable : "
        "offline = production."
    )

    # Enregistre le résultat du test
    # dans les métriques définitives.
    metrics[
        "artifact_parity"
    ] = parity

    model.metrics = metrics

    record = build_model_record(
        model,
        metrics,
    )

    previous = (
        get_latest_active_model()
    )

    if (
        previous
        and
        previous.get(
            "artifact_hash"
        )
        == record[
            "artifact_hash"
        ]
    ):

        print()
        print(
            "Modèle inchangé : "
            "pas de nouvelle version."
        )

        return

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print(
        "Sauvegarde du modèle "
        "dans Supabase..."
    )

    created = save_model_version(
        record
    )

    model_id = int(
        created[
            "id"
        ]
    )

    print(
        "Nouvelle version créée :",
        model_id,
    )

    # Activation seulement APRÈS
    # réussite du garde de parité.
    activate_model(
        model_id
    )

    print(
        "Modèle actif :",
        model_id,
    )

    # ========================================================
    # BACKTEST DB
    # ========================================================

    finished_at = datetime.now(
        timezone.utc
    )

    notes = (
        "HorseProno V4.2 production-aligned. "
        "BT 60% + Logistic 40%. "
        "Calibration apprise sur la recette "
        "production exacte. "
        "Gradient Boosting challenger offline. "
        "Artifact parity guard validé avant activation. "
        f"Test du {metrics.get('test_start')} "
        f"au {metrics.get('test_end')}."
    )

    save_backtest_run(
        {
            "model_version_id":
                model_id,

            "started_at":
                started_at.isoformat(),

            "finished_at":
                finished_at.isoformat(),

            "metrics":
                metrics,

            "notes":
                notes,
        }
    )

    print(
        "Backtest enregistré."
    )

    print()
    print(
        "=" * 70
    )

    print(
        "METRICS COMPLETES"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"MODÈLE {model_id} "
        "ENTRAÎNÉ, VÉRIFIÉ "
        "ET ACTIVÉ"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
