from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
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

from app_core.model import HorseRacingModel
from app_core.model_store import build_model_record


def main() -> None:

    started_at = datetime.now(
        timezone.utc
    )

    print("=" * 60)
    print("HORSEPRONO - TRAINING")
    print("=" * 60)

    print(
        "Chargement de l'historique "
        "depuis Supabase..."
    )

    history = get_training_history()

    if history.empty:
        raise SystemExit(
            "Aucun historique terminé dans Supabase."
        )

    print(
        f"Historique chargé : "
        f"{len(history)} participants"
    )

    if "race_id" in history.columns:

        print(
            f"Courses disponibles : "
            f"{history['race_id'].nunique()}"
        )

    if "race_date" in history.columns:

        print(
            f"Période : "
            f"{history['race_date'].min()} "
            f"-> "
            f"{history['race_date'].max()}"
        )

    # ========================================================
    # ENTRAINEMENT
    # ========================================================

    print()
    print(
        "Entraînement du modèle..."
    )

    model = HorseRacingModel()

    metrics = model.fit(
        history
    )

    print()
    print(
        "Entraînement terminé."
    )

    # ========================================================
    # BACKTEST V2
    # ========================================================

    print()
    print("=" * 60)
    print("BACKTEST V2")
    print("=" * 60)

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
            "Gagnant dans Top 2 :",
            overall.get(
                "winner_in_top2"
            ),
        )

        print(
            "Gagnant dans Top 3 :",
            overall.get(
                "winner_in_top3"
            ),
        )

        print(
            "Gagnant dans Top 5 :",
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

        print(
            "Trio complet couvert Top 5 :",
            overall.get(
                "trio_complete_top5"
            ),
        )

        print(
            "Podium moyen capturé Top 3 :",
            overall.get(
                "avg_podium_captured_top3"
            ),
        )

        print(
            "ECE calibration :",
            advanced
            .get(
                "calibration",
                {},
            )
            .get(
                "expected_calibration_error"
            ),
        )

    # ========================================================
    # ARTEFACT
    # ========================================================

    record = build_model_record(
        model,
        metrics,
    )

    previous = (
        get_latest_active_model()
    )

    if (
        previous
        and previous.get(
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

        print(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    print()
    print(
        "Sauvegarde du modèle "
        "dans Supabase..."
    )

    created = (
        save_model_version(
            record
        )
    )

    model_id = int(
        created["id"]
    )

    print(
        f"Nouvelle version créée : "
        f"{model_id}"
    )

    activate_model(
        model_id
    )

    print(
        f"Modèle actif : "
        f"{model_id}"
    )

    # ========================================================
    # BACKTEST DATABASE
    # ========================================================

    finished_at = datetime.now(
        timezone.utc
    )

    notes = (
        "Backtest chronologique HorseProno V3 "
        "+ Backtest V2. "
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

    # ========================================================
    # RESULTATS
    # ========================================================

    print()
    print("=" * 60)
    print("METRICS COMPLETES")
    print("=" * 60)

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 60)

    print(
        f"MODÈLE {model_id} "
        "ENTRAÎNÉ ET ACTIVÉ"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
