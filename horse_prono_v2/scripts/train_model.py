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
    """
    Entraîne le modèle HorseProno à partir
    de l'historique stocké dans Supabase.
    """

    started_at = datetime.now(
        timezone.utc
    )

    print("=" * 60)
    print("HORSEPRONO - TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Chargement historique
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Entraînement
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3. Création artefact
    # --------------------------------------------------------

    record = build_model_record(
        model,
        metrics,
    )

    previous = get_latest_active_model()

    # --------------------------------------------------------
    # 4. Vérification modèle identique
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. Sauvegarde version
    # --------------------------------------------------------

    print()
    print(
        "Sauvegarde du modèle "
        "dans Supabase..."
    )

    created = save_model_version(
        record
    )

    model_id = int(
        created["id"]
    )

    print(
        f"Nouvelle version créée : "
        f"{model_id}"
    )

    # --------------------------------------------------------
    # 6. Activation
    # --------------------------------------------------------

    activate_model(
        model_id
    )

    print(
        f"Modèle actif : "
        f"{model_id}"
    )

    # --------------------------------------------------------
    # 7. Backtest
    # --------------------------------------------------------

    finished_at = datetime.now(
        timezone.utc
    )

    notes = (
        "Backtest chronologique HorseProno V3. "
        f"Test du {metrics.get('test_start')} "
        f"au {metrics.get('test_end')}. "
        f"{metrics.get('backtest_races', 0)} "
        "courses de test."
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

    # --------------------------------------------------------
    # 8. Résultats
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("METRICS")
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
